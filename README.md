# Sentinel – Transaction Monitoring Prototype

Detects anomalous financial activity, gives every transaction an explainable 0-100 risk score,
prioritises cases automatically and offers a role-based investigation dashboard.

## Run
    pip install -r requirements.txt
    python app.py          # open http://localhost:5000

Demo users (password `demo123`): `rajesh` supervisor, `priya` / `arun` analysts, `kavya` auditor.
Click **Start live feed** (supervisor) to watch transactions arrive, get scored, turn into cases and get auto-assigned.

## How it works
1. **Features** per transaction: amount z-score vs the account's own history, 1-hour velocity, night hour, new beneficiary,
   jurisdiction risk, just-under-threshold, round amount, amount vs profile.
2. **Model**: Isolation Forest (unsupervised) blended with compliance rules (structuring, bursts, high-risk country, new payee).
3. **Explainability**: every score lists its drivers (occlusion attribution: neutralise one feature, measure the score drop) plus rule hits.
4. **Automation**: score >= 60 opens/updates a case, priority = risk + alert count + exposure, auto-assign to the least-loaded analyst,
   score >= 85 auto-escalates, SLA timer by priority.
5. **Authorisation**: token login; analysts see only their cases; only supervisors close, file SAR or reassign; auditors are read-only; full audit trail.

Data is synthetic. Replace `make_txn` / `train` in `core.py` with your real transaction feed.
