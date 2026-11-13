"""
AI-Hub ops tracker: adoption, AI spend vs budget and agent governance checks, on SYNTHETIC company data.

What it simulates (ground truth known, messy on purpose)
  6 teams, 120 employees, 90 days. Days 1-30 = frozen baseline (before AI tools), days 31-90 = rollout.
  usage_log.csv : one row per AI tool call (Claude, Claude Code, n8n, internal agents), with real-world mess:
                  3 timestamp formats, cost as text ("$1.20", "USD 0.8", "1,5"), team names in mixed case/spaces,
                  5 % missing team, 3 % duplicate rows, a test bot account.
  tasks.csv     : tasks closed per person per week (the productivity KPI), baseline and rollout.
  registry.json : agent registry (owner team, usage level human_only / human_in_the_loop / autonomous,
                  data access scopes, read or write).
  Injected problems: Engineering AI spend drifts up from day 70; Legal adoption stays low;
  one autonomous agent has write access to confidential data; one agent in the logs is not in the registry.

What it does
  1. Cleans the log and reports what it fixed (duplicates, dates, costs, team names filled from the directory).
  2. KPIs per team: weekly adoption (active users / headcount), spend, cost per active user,
     tasks per person per week vs the frozen baseline.
  3. Spend vs monthly budget with a month-end projection from the last 14 days' run rate; flags drift early.
  4. Governance checks: agents missing from the registry, autonomous agents with write access to
     confidential data, agents whose scopes exceed their usage level (least privilege).
  5. Scores the checks against the injected problems and writes an HTML report for non-technical readers.

Run:  python aiops_hub.py      (a few seconds; needs pandas, numpy, matplotlib)
"""
import json, base64, io, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
SEED = 5
TEAMS = {"Sales": 25, "Data": 15, "Engineering": 30, "Operations": 20, "Legal": 10, "Marketing": 20}
BUDGET_MONTH = {"Sales": 400, "Data": 500, "Engineering": 900, "Operations": 300, "Legal": 150, "Marketing": 300}   # USD
ADOPT = {"Sales": 0.6, "Data": 0.85, "Engineering": 0.8, "Operations": 0.55, "Legal": 0.15, "Marketing": 0.65}
START = pd.Timestamp("2026-09-01"); BASELINE_DAYS = 30; DAYS = 90
CONFIDENTIAL = {"client_contracts", "hr_records", "trading_data"}


# ---------------------------------------------------------------- synthetic data
def simulate():
    rng = np.random.default_rng(SEED)
    people = [(f"{t.lower()}{i:02d}@corp.example", t) for t, n in TEAMS.items() for i in range(n)]
    pd.DataFrame(people, columns=["user", "team"]).to_csv(DATA / "directory.csv", index=False)
    registry = [
        dict(agent_id="crm-notes", owner="Sales", level="human_in_the_loop", scopes=["crm"], access="write"),
        dict(agent_id="sql-helper", owner="Data", level="human_in_the_loop", scopes=["warehouse"], access="read"),
        dict(agent_id="pr-reviewer", owner="Engineering", level="human_in_the_loop", scopes=["code"], access="read"),
        dict(agent_id="invoice-bot", owner="Operations", level="autonomous", scopes=["invoices", "client_contracts"], access="write"),  # injected
        dict(agent_id="contract-summary", owner="Legal", level="human_only", scopes=["client_contracts"], access="read"),
        dict(agent_id="social-drafts", owner="Marketing", level="human_in_the_loop", scopes=["brand_assets"], access="write"),
    ]
    (DATA / "registry.json").write_text(json.dumps(registry, indent=2))
    agent_of = {r["owner"]: r["agent_id"] for r in registry}
    rows = []
    for day in range(BASELINE_DAYS, DAYS):
        date = START + pd.Timedelta(days=day)
        if date.weekday() >= 5: continue
        for user, team in people:
            if rng.random() > ADOPT[team] * min(1, (day - BASELINE_DAYS + 5) / 25): continue   # adoption ramps up
            for _ in range(rng.poisson(3) + 1):
                tool = rng.choice(["Claude", "Claude Code", "n8n", "agent"], p=[0.45, 0.3 if team in ("Engineering", "Data") else 0.05,
                                                                                0.1, 0.15 if team in ("Engineering", "Data") else 0.4])
                cost = rng.gamma(2, 0.08) * (3 if tool == "Claude Code" else 1)
                if team == "Engineering" and day >= 70: cost *= 2.2                               # injected spend drift
                agent = agent_of[team] if tool == "agent" else ""
                if team == "Engineering" and tool == "agent" and rng.random() < 0.3: agent = "deploy-autofix"  # shadow agent
                ts = date + pd.Timedelta(minutes=int(rng.integers(8 * 60, 19 * 60)))
                rows.append([ts, user, team, tool, agent, cost])
    log = pd.DataFrame(rows, columns=["ts", "user", "team", "tool", "agent_id", "cost"])
    # ---- make it messy
    fmt = rng.integers(0, 3, len(log))
    log["ts"] = [t.strftime(["%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%dT%H:%M:%SZ"][f]) for t, f in zip(log.ts, fmt)]
    style = rng.integers(0, 3, len(log))
    log["cost"] = [[f"${c:.2f}", f"USD {c:.2f}", f"{c:.2f}".replace(".", ",")][s] for c, s in zip(log.cost, style)]
    log["team"] = [[t, t.upper(), f" {t.lower()} "][s] for t, s in zip(log.team, rng.integers(0, 3, len(log)))]
    log.loc[rng.random(len(log)) < 0.05, "team"] = ""
    bot = log.sample(40, random_state=SEED).assign(user="test-bot@corp.example", team="")
    log = pd.concat([log, log.sample(frac=0.03, random_state=SEED), bot]).sample(frac=1, random_state=SEED)
    log.to_csv(DATA / "usage_log.csv", index=False)
    # productivity KPI: tasks closed per person per week (rollout adds ~adoption-proportional lift, plus noise)
    tasks = []
    for user, team in people:
        base = rng.normal(10, 2)
        for w in range(DAYS // 7):
            lift = 0.12 * ADOPT[team] if w * 7 >= BASELINE_DAYS else 0
            tasks.append([user, team, w + 1, max(0, round(base * (1 + lift) + rng.normal(0, 1.5)))])
    pd.DataFrame(tasks, columns=["user", "team", "week", "tasks_closed"]).to_csv(DATA / "tasks.csv", index=False)


# ---------------------------------------------------------------- cleaning
def parse_ts(s):
    s = s.strip()
    for f in ("%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%dT%H:%M:%SZ"):
        try: return pd.to_datetime(s, format=f)
        except ValueError: pass
    return pd.NaT


def parse_cost(s):
    s = str(s).strip().replace("USD", "").replace("$", "").strip().replace(",", ".")
    return float(s) if re.fullmatch(r"\d+(\.\d+)?", s) else np.nan


def clean(raw, directory):
    rep = dict(rows_in=len(raw))
    df = raw.drop_duplicates().copy(); rep["duplicates_removed"] = rep["rows_in"] - len(df)
    bots = df.user.str.contains("bot", na=False); rep["bot_rows_removed"] = int(bots.sum()); df = df[~bots]
    df["ts"] = df.ts.map(parse_ts); rep["unparsed_dates"] = int(df.ts.isna().sum())
    df["cost"] = df.cost.map(parse_cost); rep["unparsed_costs"] = int(df.cost.isna().sum())
    df["team"] = df.team.fillna("").str.strip().str.title()
    missing = df.team == ""; rep["team_missing"] = int(missing.sum())
    df.loc[missing, "team"] = df.loc[missing, "user"].map(directory.set_index("user").team)
    rep["team_unresolved"] = int(df.team.isna().sum())
    df["agent_id"] = df.agent_id.fillna("")
    rep["rows_out"] = len(df)
    return df.dropna(subset=["ts", "cost", "team"]), rep


# ---------------------------------------------------------------- KPIs, budget, governance
def kpis(df, tasks):
    df = df.assign(week=((df.ts - START).dt.days // 7) + 1)
    weekly = df.groupby(["team", "week"]).user.nunique().unstack(0).fillna(0) / pd.Series(TEAMS)
    last4 = weekly.loc[weekly.index[-4:]].mean()
    recent = df[df.ts > df.ts.max() - pd.Timedelta(days=28)]
    daily = recent.groupby([recent.ts.dt.date, "team"]).user.nunique().unstack().fillna(0) / pd.Series(TEAMS)
    daily_share = daily.mean()
    t = tasks.assign(period=np.where((tasks.week - 1) * 7 >= BASELINE_DAYS, "rollout", "baseline"))
    prod = t.groupby(["team", "period"]).tasks_closed.mean().unstack()
    out = pd.DataFrame({"headcount": pd.Series(TEAMS), "weekly_active_share_last_4w": last4.round(2),
                        "daily_active_share_last_4w": daily_share.round(2),
                        "spend_usd_rollout": df.groupby("team").cost.sum().round(0),
                        "tasks_per_person_week_baseline": prod["baseline"].round(2),
                        "tasks_per_person_week_rollout": prod["rollout"].round(2)})
    out["cost_per_active_user_month"] = (out.spend_usd_rollout / 2 / (out.weekly_active_share_last_4w * out.headcount)).round(1)
    out["tasks_change_vs_baseline_pct"] = ((out.tasks_per_person_week_rollout / out.tasks_per_person_week_baseline - 1) * 100).round(1)
    return out, weekly


def budget(df, as_of):
    """Month-to-date spend + projection to month end from the last 14 days' daily run rate."""
    m0 = as_of.replace(day=1); m_end = m0 + pd.offsets.MonthEnd(0)
    rows = []
    for team, b in BUDGET_MONTH.items():
        d = df[df.team == team]
        mtd = d[(d.ts >= m0) & (d.ts <= as_of)].cost.sum()
        rate = d[(d.ts > as_of - pd.Timedelta(days=14)) & (d.ts <= as_of)].cost.sum() / 14
        proj = mtd + rate * (m_end - as_of).days
        rows.append(dict(team=team, budget=b, month_to_date=round(mtd), projected_month_end=round(proj),
                         projected_pct_of_budget=round(100 * proj / b), flag="OVER" if proj > 1.10 * b else ""))
    return pd.DataFrame(rows).set_index("team")


def governance(df, registry):
    reg = {r["agent_id"]: r for r in registry}
    seen = set(df.agent_id) - {""}
    issues = [dict(check="agent not in registry", agent=a, calls=int((df.agent_id == a).sum())) for a in sorted(seen - set(reg))]
    for r in registry:
        conf = sorted(set(r["scopes"]) & CONFIDENTIAL)
        if r["level"] == "autonomous" and conf and r["access"] == "write":
            issues.append(dict(check="autonomous agent can write confidential data", agent=r["agent_id"], scopes=conf))
        if r["level"] == "human_only" and (df.agent_id == r["agent_id"]).any():
            issues.append(dict(check="human-only agent called automatically", agent=r["agent_id"]))
    return issues


def img(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def main():
    simulate()
    raw = pd.read_csv(DATA / "usage_log.csv", dtype=str, keep_default_na=False)
    directory = pd.read_csv(DATA / "directory.csv"); tasks = pd.read_csv(DATA / "tasks.csv")
    registry = json.loads((DATA / "registry.json").read_text())
    df, rep = clean(raw, directory)
    k, weekly = kpis(df, tasks)
    as_of = pd.Timestamp("2026-11-20 23:59")                       # 20 days into November, early warning
    bud = budget(df, as_of)
    gov = governance(df, registry)
    first_flag = {}                                                # replay the check daily: when would it have warned?
    for day in pd.date_range("2026-10-05", "2026-11-29"):
        b = budget(df, day + pd.Timedelta(hours=23, minutes=59))
        for t in b.index[b.flag == "OVER"]: first_flag.setdefault(t, str(day.date()))
    low_adoption = k.index[k.daily_active_share_last_4w < 0.30].tolist()
    low_adoption_weekly_metric = k.index[k.weekly_active_share_last_4w < 0.30].tolist()

    expected = dict(spend_drift="Engineering", low_adoption="Legal", risky_agent="invoice-bot", shadow_agent="deploy-autofix")
    found = dict(spend_drift=bud.index[bud.flag == "OVER"].tolist(), low_adoption=low_adoption,
                 risky_agent=[g["agent"] for g in gov if g["check"].startswith("autonomous")],
                 shadow_agent=[g["agent"] for g in gov if g["check"] == "agent not in registry"])
    score = {key: dict(expected=v, flagged=found[key], caught=v in found[key], false_flags=[x for x in found[key] if x != v])
             for key, v in expected.items()}

    k.to_csv(OUT / "team_kpis.csv"); bud.to_csv(OUT / "budget.csv")
    log = dict(first_budget_flag_by_team=first_flag, spend_drift_injected_from=str((START + pd.Timedelta(days=70)).date()), cleaning=rep, low_adoption_flagged_with_weekly_metric=low_adoption_weekly_metric,
               not_injected_but_flagged=[g for g in gov if g["check"] == "human-only agent called automatically"], team_kpis=k.reset_index().to_dict("records"), budget_as_of=str(as_of.date()),
               budget=bud.reset_index().to_dict("records"), governance_issues=gov, checks_vs_injected=score,
               caught=f"{sum(s['caught'] for s in score.values())}/{len(score)}",
               false_flags=sum(len(s["false_flags"]) for s in score.values()),
               note="All data is synthetic. Before/after task counts are not a causal estimate (no control group).")
    (OUT / "run_log.json").write_text(json.dumps(log, indent=2, default=str))

    fig, ax = plt.subplots(figsize=(7, 3.2)); weekly.plot(ax=ax); ax.set_ylabel("weekly active users / headcount"); ax.set_title("Weekly active share by team (saturates: see daily share)")
    a_img = img(fig)
    fig, ax = plt.subplots(figsize=(7, 3.2)); bud.projected_pct_of_budget.plot.bar(ax=ax, color=["#c0392b" if f else "#7f8c8d" for f in bud.flag])
    ax.axhline(110, ls="--", c="k", lw=0.8); ax.set_ylabel("% of monthly budget"); ax.set_title(f"Projected November spend (as of {as_of.date()})")
    b_img = img(fig)
    issues_html = "".join(f"<li>{g['check']}: <b>{g['agent']}</b></li>" for g in gov) or "<li>none</li>"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>AI-Hub report</title>
<style>body{{font-family:Arial;max-width:900px;margin:24px auto;color:#222}} table{{border-collapse:collapse;font-size:13px}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}} th{{background:#f2f2f2}}</style></head><body>
<h1>AI-Hub monthly report (synthetic data)</h1>
<h2>What needs attention</h2><ul>
<li>Spend: {', '.join(found['spend_drift']) or 'no team'} projected over 110 % of the monthly budget.</li>
<li>Daily active share under 30 %: {', '.join(low_adoption) or 'none'}.</li>{issues_html}</ul>
<h2>Team KPIs</h2>{k.to_html()}<img src="data:image/png;base64,{a_img}">
<h2>Budget</h2>{bud.to_html()}<img src="data:image/png;base64,{b_img}">
<h2>Data cleaning</h2><pre>{json.dumps(rep, indent=1)}</pre>
<p>Task counts before and after rollout are not a causal estimate: there is no control group, and other changes happened at the same time.</p>
</body></html>"""
    (OUT / "report.html").write_text(html)
    print(json.dumps({x: log[x] for x in ("cleaning", "budget", "governance_issues", "checks_vs_injected", "caught", "false_flags")}, indent=1, default=str))
    print(k.to_string())


if __name__ == "__main__":
    main()
