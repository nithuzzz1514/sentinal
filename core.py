"""Sentinel core: synthetic data, Isolation Forest + rules hybrid scorer with
explainable drivers, and an automated case manager."""
import time, threading, secrets, hashlib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

FEATS = ['amount_z', 'velocity_1h', 'night', 'new_bene', 'geo_risk', 'near_threshold', 'round_amount', 'amt_ratio']
FLABEL = {'amount_z': 'Amount far above account norm', 'velocity_1h': 'Burst of rapid transactions',
          'night': 'Unusual hour (late night)', 'new_bene': 'First-time beneficiary',
          'geo_risk': 'High-risk jurisdiction', 'near_threshold': 'Just under reporting threshold',
          'round_amount': 'Large round-figure amount', 'amt_ratio': 'Large vs. declared profile'}
TYPO = {'near_threshold': 'Structuring (smurfing)', 'velocity_1h': 'Rapid movement of funds', 'geo_risk': 'High-risk geography',
        'new_bene': 'Possible mule / new payee', 'amount_z': 'Abnormal value spike', 'night': 'Off-hours activity',
        'round_amount': 'Round-figure layering', 'amt_ratio': 'Abnormal value spike'}
GEO = {'NG': .8, 'KY': .9, 'AE': .4}
FLAG = 60
ANALYSTS = ['Priya', 'Arun']

# ---------------- data + feature extraction ----------------
FN = ['Aarav', 'Diya', 'Karthik', 'Meera', 'Ravi', 'Sneha', 'Vikram', 'Anjali', 'Suresh', 'Divya']
LN = ['Iyer', 'Reddy', 'Nair', 'Sharma', 'Kumar', 'Pillai', 'Menon', 'Das', 'Rao', 'Gupta']

class Account:
    def __init__(self, i, rng):
        self.id = f'AC{1001+i}'; self.name = f'{rng.choice(FN)} {rng.choice(LN)}'
        self.avg = float(rng.integers(20, 80) * 1000)
        self.benes = {f'B{i:02d}{k}' for k in range(1, 5)}; self.hist = []

def make_txn(acc, ts, kind, rng):
    t = dict(acc=acc.id, ts=ts, hour=int(rng.integers(9, 20)), cc='IN', kind=kind,
             amount=round(acc.avg * float(np.exp((rng.random() - .5) * .9)), -2), bene=rng.choice(sorted(acc.benes)))
    nb = f'NB{int(rng.integers(0, 99)):02d}'
    if kind == 'struct': t.update(amount=float(9e5 + rng.integers(0, 99) * 1e3), bene='NB' + acc.id[-2:])
    elif kind == 'burst': t.update(amount=round(acc.avg * (2 + rng.random() * 2), -2), bene=nb)
    elif kind == 'geo': t.update(hour=int(rng.integers(1, 4)), cc=str(rng.choice(['NG', 'KY', 'AE'])), amount=round(acc.avg * (5 + rng.random() * 5), -3), bene=nb)
    elif kind == 'new': t.update(amount=round(acc.avg * (8 + rng.random() * 4), -3), bene=nb)
    return t

def extract(t, acc):
    L = np.log([a for _, a in acc.hist]) if acc.hist else np.array([np.log(acc.avg)])
    mu = L.mean(); sd = max(.35, L.std()) if len(L) > 4 else .5
    vel = sum(1 for ts, _ in acc.hist if t['ts'] - 3600 < ts <= t['ts'] + 1)
    a = t['amount']
    return [(np.log(a) - mu) / sd, vel, float(t['hour'] < 5), float(t['bene'] not in acc.benes), GEO.get(t['cc'], 0.),
            float(9e5 <= a < 1e6), float(a >= 5e5 and a % 1e5 == 0), a / acc.avg]

def commit(t, acc):
    acc.hist.append((t['ts'], t['amount'])); acc.benes.add(t['bene'])

# ---------------- hybrid explainable scorer ----------------
class Scorer:
    def fit(self, X):
        self.iso = IsolationForest(n_estimators=100, random_state=7).fit(X)
        raw = -self.iso.score_samples(X); self.lo, self.hi = np.percentile(raw, [50, 99.3]); self.med = np.median(X, axis=0)
        return self
    def _pts(self, raw): return np.clip((raw - self.lo) / (self.hi - self.lo), 0, 1) * 100
    def rules(self, x):
        r = []
        if x[5]: r.append(('Rule: structuring just under Rs 10L threshold', 40))
        if x[1] >= 3: r.append(('Rule: 3+ transfers within an hour', 30))
        if x[4] >= .8: r.append(('Rule: high-risk jurisdiction', 30))
        if x[3] and x[0] > 3: r.append(('Rule: large payment to new beneficiary', 30))
        if x[6]: r.append(('Rule: round-figure large amount', 15))
        return r
    def score(self, X):
        X = np.atleast_2d(X); base = self._pts(-self.iso.score_samples(X)); out = []
        occ = []
        for j in range(X.shape[1]):               # occlusion attribution: neutralise one feature at a time
            Z = X.copy(); Z[:, j] = self.med[j]
            occ.append(np.clip(base - self._pts(-self.iso.score_samples(Z)), 0, None) * .7)
        for i, x in enumerate(X):
            rl = self.rules(x); rp = min(100, sum(p for _, p in rl))
            d = [(FLABEL[f], round(float(occ[j][i]), 1), f) for j, f in enumerate(FEATS) if occ[j][i] > .5]
            d += [(n, p * .5, 'rule') for n, p in rl]
            out.append((int(min(100, round(.7 * base[i] + .5 * rp))), sorted(d, key=lambda z: -z[1])))
        return out

def train(seed=7):
    rng = np.random.default_rng(seed); now = time.time()
    accs = {a.id: a for a in (Account(i, rng) for i in range(30))}; ev = []
    for a in accs.values():
        for _ in range(26): ev.append(make_txn(a, now - rng.random() * 14 * 86400, 'normal', rng))
    for _ in range(18):
        a = accs[rng.choice(list(accs))]; k = str(rng.choice(['struct', 'burst', 'geo', 'new'])); t0 = now - rng.random() * 3 * 86400
        n = {'struct': 4, 'burst': 6}.get(k, 1); gap = {'struct': 10800, 'burst': 240}.get(k, 0)
        for j in range(n): ev.append(make_txn(a, t0 + j * gap, k, rng))
    ev.sort(key=lambda t: t['ts']); X, y = [], []
    shadow = {k: Account(0, rng) for k in accs}                  # replay on shadow state for training features
    for k, a in accs.items(): shadow[k].id, shadow[k].avg, shadow[k].benes = a.id, a.avg, {f'B{k[-2:]}{i}' for i in range(1, 5)}
    for t in ev: X.append(extract(t, shadow[t['acc']])); y.append(t['kind'] != 'normal'); commit(t, shadow[t['acc']])
    X = np.array(X); y = np.array(y); sc = Scorer().fit(X)
    res = sc.score(X); risk = np.array([r for r, _ in res]); fl = risk >= FLAG
    metrics = dict(auc=round(float(roc_auc_score(y, risk)), 3), precision=round(float((y & fl).sum() / max(1, fl.sum())), 3),
                   recall=round(float((y & fl).sum() / y.sum()), 3), n=int(len(y)), anomalies=int(y.sum()))
    return sc, accs, ev, metrics

# ---------------- case management + automation ----------------
USERS = {u: dict(name=n, role=r, pw=hashlib.sha256(b'demo123').hexdigest()) for u, n, r in
         [('priya', 'Priya', 'analyst'), ('arun', 'Arun', 'analyst'), ('rajesh', 'Rajesh', 'supervisor'), ('kavya', 'Kavya', 'auditor')]}
CAN = {'analyst': {'note', 'escalate'}, 'supervisor': {'note', 'escalate', 'close', 'sar', 'assign'}, 'auditor': set()}

class Engine:
    def __init__(self):
        self.lock = threading.RLock(); self.scorer, self.accs, ev, self.metrics = train()
        self.txns, self.cases, self.audit, self.live, self.n = [], [], [], False, 0
        for a in self.accs.values(): a.hist, a.benes = [], {f'B{a.id[-2:]}{i}' for i in range(1, 5)}
        X = []; tmp = {k: Account(0, np.random.default_rng(0)) for k in self.accs}
        for k, a in self.accs.items(): tmp[k].id, tmp[k].avg, tmp[k].benes = a.id, a.avg, set(a.benes)
        for t in ev: X.append(extract(t, tmp[t['acc']])); commit(t, tmp[t['acc']])
        res = self.scorer.score(np.array(X))                    # batch-score history once, then replay into cases
        for t, p in zip(ev, res): self.ingest(t, p)
        self.audit.clear()
    def log(self, c, who, role, msg):
        e = dict(ts=time.time(), who=who, role=role, msg=msg, case=c['id'] if c else None); self.audit.insert(0, e)
        if c: c['log'].insert(0, e)
    def ingest(self, t, pre=None):
        with self.lock:
            a = self.accs[t['acc']]; x = extract(t, a)
            risk, drv = pre if pre else self.scorer.score(x)[0]; commit(t, a); self.n += 1
            r = dict(id=self.n, ts=t['ts'], acc=a.id, amount=t['amount'], bene=t['bene'], cc=t['cc'], hour=t['hour'], risk=risk,
                     drivers=[dict(label=l, pts=round(p, 1), f=f) for l, p, f in drv]); self.txns.append(r)
            if risk < FLAG: return r
            c = next((c for c in self.cases if c['acc'] == a.id and c['status'] in ('Open', 'Escalated')), None)
            if not c:
                c = dict(id=f'CS-{1001+len(self.cases)}', acc=a.id, name=a.name, status='Open', assignee=None, tx=[], notes=[], log=[], created=t['ts'])
                self.cases.append(c); self.log(c, 'Model', 'system', 'Case auto-created from alert')
            c['tx'].append(r); self.rank(c)
            if not c['assignee']:
                load = lambda n: sum(1 for k in self.cases if k['assignee'] == n and k['status'] in ('Open', 'Escalated'))
                c['assignee'] = min(ANALYSTS, key=load); self.log(c, 'Router', 'system', f"Auto-assigned to {c['assignee']} (lowest workload)")
            if risk >= 85 and c['status'] == 'Open': c['status'] = 'Escalated'; self.log(c, 'Model', 'system', f'Auto-escalated: risk {risk}')
            return r
    def rank(self, c):
        ex = sum(x['amount'] for x in c['tx']); c['priority'] = round(.5 * max(x['risk'] for x in c['tx']) + .3 * min(100, len(c['tx']) * 25) + .2 * min(100, ex / 5e4))
        s = {}
        for x in c['tx']:
            for d in x['drivers']:
                if d['f'] != 'rule': s[d['f']] = s.get(d['f'], 0) + d['pts']
            if any(d['f'] == 'rule' and 'structuring' in d['label'] for d in x['drivers']): s['near_threshold'] = s.get('near_threshold', 0) + 40
        c['typology'] = TYPO[max(s, key=s.get)] if s else 'Anomalous activity'
        c['sla_due'] = c['created'] + (4 if c['priority'] >= 80 else 24 if c['priority'] >= 60 else 72) * 3600
    def act(self, cid, user, action, value=''):
        with self.lock:
            c = next(k for k in self.cases if k['id'] == cid)
            if action not in CAN[user['role']]: raise PermissionError
            if user['role'] == 'analyst' and c['assignee'] != user['name']: raise PermissionError
            if action == 'note' and value.strip(): c['notes'].insert(0, dict(who=user['name'], ts=time.time(), text=value)); self.log(c, user['name'], user['role'], 'Note: ' + value)
            elif action == 'escalate': c['status'] = 'Escalated'; self.log(c, user['name'], user['role'], 'Escalated for review')
            elif action == 'close': c['status'] = 'Closed'; self.log(c, user['name'], user['role'], 'Closed as false positive')
            elif action == 'sar': c['status'] = 'SAR filed'; self.log(c, user['name'], user['role'], 'Suspicious Activity Report filed')
            elif action == 'assign' and value in ANALYSTS: c['assignee'] = value; self.log(c, user['name'], user['role'], 'Reassigned to ' + value)
            return c
    def simulate_once(self, rng):
        a = self.accs[rng.choice(list(self.accs))]; r = rng.random(); now = time.time()
        if r < .82: self.ingest(make_txn(a, now, 'normal', rng))
        else:
            k = str(rng.choice(['struct', 'burst', 'geo', 'new'])); n = 4 if k == 'burst' else 1
            for j in range(n): self.ingest(make_txn(a, now - (n - 1 - j) * 90, k, rng))
