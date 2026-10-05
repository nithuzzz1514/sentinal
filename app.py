"""Sentinel API + dashboard.  Run:  python app.py  ->  http://localhost:5000"""
import os, time, secrets, hashlib, threading
import numpy as np
from functools import wraps
from flask import Flask, jsonify, request, send_from_directory
from core import Engine, USERS, CAN, FLAG

app = Flask(__name__, static_folder='static'); E = Engine(); TOK = {}; rng = np.random.default_rng(99)

def auth(perm=None, roles=None):
    def deco(fn):
        @wraps(fn)
        def w(*a, **k):
            u = TOK.get(request.headers.get('Authorization', '').replace('Bearer ', ''))
            if not u: return jsonify(error='Sign in required'), 401
            if (roles and u['role'] not in roles) or (perm and perm not in CAN[u['role']]): return jsonify(error='Not authorised for this action'), 403
            return fn(u, *a, **k)
        return w
    return deco

def visible(u): return [c for c in E.cases if u['role'] != 'analyst' or c['assignee'] == u['name']]
def brief(c): return dict(id=c['id'], name=c['name'], acc=c['acc'], status=c['status'], assignee=c['assignee'], priority=c['priority'],
                          typology=c['typology'], alerts=len(c['tx']), exposure=sum(x['amount'] for x in c['tx']), sla_hours=round((c['sla_due'] - time.time()) / 3600))

@app.get('/')
def index(): return send_from_directory('static', 'index.html')

@app.post('/api/login')
def login():
    d = request.get_json(force=True); u = USERS.get(d.get('user', '').lower())
    if not u or u['pw'] != hashlib.sha256(d.get('password', '').encode()).hexdigest(): return jsonify(error='Wrong username or password'), 401
    t = secrets.token_hex(16); TOK[t] = u; E.log(None, u['name'], u['role'], 'Signed in')
    return jsonify(token=t, name=u['name'], role=u['role'], can=sorted(CAN[u['role']]))

@app.get('/api/summary')
@auth()
def summary(u):
    cs = visible(u); h = [0] * 10; drv = {}
    for t in E.txns:
        h[min(9, t['risk'] // 10)] += 1
        if t['risk'] >= FLAG:
            for d in t['drivers']: drv[d['label']] = drv.get(d['label'], 0) + d['pts']
    fl = [t for t in E.txns if t['risk'] >= FLAG]
    top = sorted(drv.items(), key=lambda z: -z[1])[:7]
    act = [c for c in cs if c['status'] in ('Open', 'Escalated')]
    return jsonify(monitored=len(E.txns), flagged=len(fl), active=len(act), exposure=sum(x['amount'] for c in act for x in c['tx']), hist=h,
                   drivers=[dict(label=k, pts=round(v / max(1, len(fl)), 1)) for k, v in top], live=E.live, metrics=E.metrics,
                   latest=[dict(case=next((c['id'] for c in cs if t in c['tx']), None), name=E.accs[t['acc']].name, risk=t['risk'], amount=t['amount'],
                                why=[d['label'] for d in t['drivers'][:2]]) for t in fl[-6:][::-1]])

@app.get('/api/cases')
@auth()
def cases(u): return jsonify(sorted(map(brief, visible(u)), key=lambda c: -c['priority']))

@app.get('/api/cases/<cid>')
@auth()
def case(u, cid):
    c = next((c for c in visible(u) if c['id'] == cid), None)
    if not c: return jsonify(error='Case not found or not assigned to you'), 404
    a = E.accs[c['acc']]; act = [dict(ts=ts, amount=am) for ts, am in a.hist[-40:]]
    return jsonify(**brief(c), txns=c['tx'], notes=c['notes'], log=c['log'], activity=act, flagged_ts=[x['ts'] for x in c['tx']])

@app.post('/api/cases/<cid>/action')
@auth()
def action(u, cid):
    d = request.get_json(force=True)
    try: c = E.act(cid, u, d.get('action'), d.get('value', ''))
    except PermissionError: return jsonify(error='Not authorised for this action'), 403
    except StopIteration: return jsonify(error='Case not found'), 404
    return jsonify(brief(c))

@app.get('/api/audit')
@auth(roles={'supervisor', 'auditor'})
def audit(u): return jsonify(E.audit[:100])

@app.post('/api/sim')
@auth(perm='close')
def sim(u):
    E.live = bool(request.get_json(force=True).get('on')); E.log(None, u['name'], u['role'], f"Live feed {'started' if E.live else 'stopped'}")
    return jsonify(live=E.live)

def loop():
    while True:
        time.sleep(1.8)
        if E.live: E.simulate_once(rng)

# start the live-feed thread on import so it also works under gunicorn (hosting)
threading.Thread(target=loop, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), threaded=True)
