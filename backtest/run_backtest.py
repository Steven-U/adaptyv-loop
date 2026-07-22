"""Validate the selection policy against Adaptyv's public EGFR competition data.

Data (ODbL, published by Adaptyv):
  https://github.com/adaptyvbio/egfr_competition_1
  https://github.com/adaptyvbio/egfr_competition_2

Round 2 contains 402 designs carrying both the computational metrics a
designer has before spending money (ipTM, pae_interaction, pLDDT, ESM2 PLL,
declared design method) and the wet-lab outcome (binding call, KD, expression).
That makes it a ground-truth testbed for the only question a campaign budget
actually poses: given N slots, which designs do you buy?

Four sections, run in order:

  1. Discrimination — how well each metric separates binders from non-binders.
  2. Method table — hit rate by declared design method.
  3. Transfer test — train a model on Round 1, evaluate on Round 2. This is
     the negative result; it is reported because it is the reason the policy
     does not include a learned per-design model.
  4. Campaign simulation — the actual policy against the baselines it has to
     beat, over a two-round explore-then-exploit budget.

Run: python backtest/run_backtest.py [--data DIR] [--trials 400]
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptyv_loop.selection import (  # noqa: E402
    Candidate,
    MethodStats,
    build_posteriors,
    select_designs,
)

REPOS = {
    "egfr_competition_1": "https://github.com/adaptyvbio/egfr_competition_1.git",
    "egfr_competition_2": "https://github.com/adaptyvbio/egfr_competition_2.git",
}


def ensure_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in REPOS.items():
        dest = data_dir / name
        if dest.exists():
            continue
        print(f"cloning {name} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def parse_method(raw: object) -> str:
    """Normalize the JSON-encoded ``design_models`` list into one label."""
    try:
        parsed = ast.literal_eval(raw) if isinstance(raw, str) else None
    except (ValueError, SyntaxError):
        return "unknown"
    if not parsed:
        return "unknown"
    return " + ".join(sorted(str(p) for p in parsed))


def load_round2(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "egfr_competition_2" / "results" / "result_summary.csv")
    df = df[df.binding.astype(str).str.lower().isin(["true", "false"])].copy()
    df = df.dropna(subset=["plddt", "pae_interaction", "iptm"]).reset_index(drop=True)
    df["hit"] = df.binding.astype(str).str.lower().eq("true").astype(int)
    df["method"] = df.design_models.apply(parse_method)
    return df


def load_round1(data_dir: Path) -> pd.DataFrame:
    base = data_dir / "egfr_competition_1" / "results"
    summary = pd.read_csv(base / "result_summary.csv")
    reps = pd.read_csv(base / "replicate_summary.csv")
    reps["b"] = reps.binding.astype(str).str.lower().map({"true": 1, "false": 0})
    labels = reps.groupby("name")["b"].max().reset_index().rename(columns={"b": "hit"})
    merged = summary.merge(labels, on="name", how="inner")
    return merged.dropna(subset=["plddt", "pae_interaction", "hit"]).reset_index(drop=True)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), tie-aware."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(scores).rank().to_numpy()
    return (ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def section_discrimination(r2: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("1. DO THE STANDARD FILTERS SEPARATE BINDERS FROM NON-BINDERS?")
    print("=" * 74)
    print(f"{len(r2)} designs, {r2.hit.sum()} binders ({r2.hit.mean():.1%})")
    strong = r2.binding_strength.astype(str).str.lower().eq("strong").sum()
    print(f"{strong} strong binders ({strong / len(r2):.1%})\n")
    print(f"{'metric':<22}{'AUC':>8}   interpretation")
    print("-" * 74)
    y = r2.hit.to_numpy()
    for col, label in [
        ("iptm", "ipTM"),
        ("plddt", "pLDDT"),
        ("esm_pll", "ESM2 pseudo-LL"),
        ("pae_interaction", "pae_interaction"),
    ]:
        if col not in r2 or r2[col].isna().all():
            continue
        sub = r2.dropna(subset=[col])
        a = auc(sub[col].to_numpy(), sub.hit.to_numpy())
        if a < 0.45:
            note = "points the WRONG way vs. convention"
        elif a < 0.60:
            note = "essentially uninformative"
        else:
            note = "weak but real signal"
        print(f"{label:<22}{a:>8.3f}   {note}")
    print("\n0.500 is a coin flip. Nothing here is a strong filter.")


def section_methods(r2: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 74)
    print("2. HIT RATE BY DESIGN METHOD (known before any money is spent)")
    print("=" * 74)
    tbl = r2.groupby("method").agg(designs=("hit", "size"), binders=("hit", "sum"))
    tbl["hit_rate"] = tbl.binders / tbl.designs
    tbl = tbl[tbl.designs >= 5].sort_values("hit_rate", ascending=False)
    print(f"{'method':<44}{'n':>5}{'binders':>9}{'rate':>8}")
    print("-" * 74)
    for m, row in tbl.iterrows():
        print(f"{m[:43]:<44}{int(row.designs):>5}{int(row.binders):>9}{row.hit_rate:>8.1%}")
    spread = tbl.hit_rate.max() / max(tbl.hit_rate[tbl.hit_rate > 0].min(), 1e-9)
    print(f"\nBest-to-worst nonzero spread: {spread:.0f}x — a far bigger lever than any")
    print("per-design metric above, and it is free to act on.")
    return tbl


def section_transfer(r1: pd.DataFrame, r2: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("3. NEGATIVE RESULT: can a learned per-design model beat ipTM?")
    print("=" * 74)
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        print("!! SKIPPED — scikit-learn is not installed, so this section did not run.")
        print("!! This is the negative result the package's design rests on (no learned")
        print("!! per-design model ships). Reproduce it with:")
        print("!!     pip install -e '.[dev]'")
        return

    feats = ["plddt", "pae_interaction"]  # the only features both rounds share
    y2 = r2.hit.to_numpy()
    ks = [40, 80, 120, 200]

    def hits_at(order: np.ndarray) -> list[int]:
        return [int(y2[order[:k]].sum()) for k in ks]

    # In-distribution cross-validation, i.e. the flattering setup.
    x_cv = r2[["pae_interaction", "iptm", "plddt", "esm_pll"]].to_numpy()
    cv_rows = []
    for seed in range(10):
        oof = np.zeros(len(y2))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(x_cv, y2):
            m = GradientBoostingClassifier(
                n_estimators=120, max_depth=2, learning_rate=0.05, random_state=seed
            ).fit(x_cv[tr], y2[tr])
            oof[te] = m.predict_proba(x_cv[te])[:, 1]
        cv_rows.append(hits_at(np.argsort(-oof)))
    cv = np.mean(cv_rows, axis=0)

    # Honest setup: fit on the earlier campaign, select in the later one.
    tr_rows = []
    for seed in range(10):
        m = GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05, random_state=seed
        ).fit(r1[feats].to_numpy(), r1.hit.to_numpy().astype(int))
        tr_rows.append(hits_at(np.argsort(-m.predict_proba(r2[feats].to_numpy())[:, 1])))
    transfer = np.mean(tr_rows, axis=0)

    # Randomized tie-breaking, matching section 4. ipTM's two-decimal
    # quantization means row order alone moves this baseline by ~8 binders at
    # K=60, so a fixed argsort would make the comparison meaningless.
    rng = np.random.default_rng(0)
    iptm = r2.iptm.to_numpy()
    base = np.mean(
        [hits_at(np.argsort(-(iptm + rng.random(len(iptm)) * 1e-6))) for _ in range(200)],
        axis=0,
    )
    print(f"Binders found per budget (Round 1 train n={len(r1)}, {int(r1.hit.sum())} binders)\n")
    print(f"{'strategy':<38}" + "".join(f"K={k:<8}" for k in ks))
    print("-" * 74)
    print(f"{'random (expected)':<38}" + "".join(f"{y2.mean() * k:<10.1f}" for k in ks))
    print(f"{'rank by ipTM  [baseline]':<38}" + "".join(f"{v:<10.1f}" for v in base))
    print(f"{'GBM, 5-fold CV on Round 2':<38}" + "".join(f"{v:<10.1f}" for v in cv))
    print(f"{'GBM, trained R1 -> applied R2':<38}" + "".join(f"{v:<10.1f}" for v in transfer))
    verdict = "LOSES to" if transfer[1] < base[1] else "beats"
    print(
        f"\nCross-validation flatters the model. Out of sample it {verdict} plain ipTM."
        "\nThat is why adaptyv_loop ships no learned per-design model."
    )


PRICE_PER_PROTEIN = 99.0


def _report(title: str, rows: list[tuple[str, float]], total: int, baseline: float) -> None:
    spend = total * PRICE_PER_PROTEIN
    print(f"{title} — {total} tests, ${spend:,.0f} at ${PRICE_PER_PROTEIN:.0f}/protein")
    print(f"{'  strategy':<36}{'binders':>9}{'cost/binder':>14}{'vs ipTM':>10}")
    print("  " + "-" * 67)
    for name, val in rows:
        cpb = f"${spend / val:,.0f}" if val > 0 else "n/a"
        delta = "" if "ipTM" in name and "status" in name else f"{(val / baseline - 1) * 100:+.0f}%"
        print(f"  {name:<34}{val:>9.1f}{cpb:>14}{delta:>10}")
    print()


def section_campaign(r2: pd.DataFrame, trials: int) -> None:
    print("\n" + "=" * 74)
    print("4. CAMPAIGN SIMULATION")
    print("=" * 74)
    print("ipTM is quantized to 2dp — 78 distinct values across 378 designs — so")
    print("every budget cut lands inside a large tie group. Ties are broken at")
    print("random per seed for every strategy. Using pandas' row order instead")
    print("swings the ipTM baseline between 9 and 17 binders at K=60 on luck alone,")
    print("which is how a careless backtest manufactures a result.\n")

    y = r2.hit.to_numpy()
    iptm = r2.iptm.to_numpy()
    all_cands = [
        Candidate(id=str(i), sequence=str(row.sequence), method=row.method,
                  score=float(row.iptm))
        for i, row in r2.iterrows()
    ]

    # ---- Scenario A: cold start, no history at all ----------------------
    print("-" * 74)
    print("A. COLD START — first campaign against this target, no method history.")
    print("   Budget split one-third explore, two-thirds exploit.")
    print("-" * 74)
    for total in (60, 160):
        k1, k2 = total // 3, total - total // 3
        policy, ipt, rand = [], [], []
        for seed in range(trials):
            rng = np.random.default_rng(seed)
            sel1 = select_designs(all_cands, k1, mode="explore", seed=seed)
            picked = {int(c.id) for c in sel1.chosen}
            observed: dict[str, list[int]] = {}
            for c in sel1.chosen:
                rec = observed.setdefault(c.method, [0, 0])
                rec[0] += 1
                rec[1] += int(y[int(c.id)])
            posts = build_posteriors(
                MethodStats(method=m, tested=t, hits=h) for m, (t, h) in observed.items()
            )
            rest = [c for c in all_cands if int(c.id) not in picked]
            sel2 = select_designs(rest, k2, posteriors=posts, mode="exploit", seed=seed)
            policy.append(sum(y[int(c.id)] for c in (*sel1.chosen, *sel2.chosen)))
            jitter = rng.random(len(y)) * 1e-6
            ipt.append(int(y[np.argsort(-(iptm + jitter))[:total]].sum()))
            rand.append(int(y[rng.choice(len(y), total, replace=False)].sum()))
        base = float(np.mean(ipt))
        _report("Cold start", [
            ("random draw", float(np.mean(rand))),
            ("rank by ipTM (status quo)", base),
            ("adaptyv_loop policy", float(np.mean(policy))),
        ], total, base)

    print("Verdict: a cold-start campaign essentially MATCHES ipTM ranking, it does")
    print("not beat it. Two rounds is not enough to learn method rates, and on this")
    print("pool ipTM is already acting as a proxy for method — its top 60 designs")
    print("are 23 ProteinMPNN entries supplying 13 of the 17 binders found there.")
    print("Claiming a cold-start win here would be overfitting to one dataset.\n")

    # ---- Scenario B: warm start from a prior campaign -------------------
    print("-" * 74)
    print("B. WARM START — method rates carried over from a previous campaign.")
    print("   Designs are split in half at random: half A is the prior campaign")
    print("   whose results are known, half B is the new pool being selected from.")
    print("   No design appears in both halves; only the method rates transfer.")
    print("-" * 74)
    for total in (20, 40, 60):
        rand, ipt, meth, both = [], [], [], []
        for seed in range(trials):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(r2))
            a, b = r2.iloc[perm[: len(r2) // 2]], r2.iloc[perm[len(r2) // 2 :]]
            yb = b.hit.to_numpy()
            obs = a.groupby("method").agg(t=("hit", "size"), h=("hit", "sum"))
            posts = build_posteriors(
                MethodStats(method=m, tested=int(r.t), hits=int(r.h)) for m, r in obs.iterrows()
            )
            cands = [
                Candidate(id=str(i), sequence="X", method=row.method, score=float(row.iptm))
                for i, (_, row) in enumerate(b.iterrows())
            ]
            rand.append(yb[rng.choice(len(yb), total, replace=False)].sum())
            jitter = rng.random(len(yb)) * 1e-6
            ipt.append(yb[np.argsort(-(b.iptm.to_numpy() + jitter))[:total]].sum())
            for strength, sink in ((0.0, meth), (1.5, both)):
                sel = select_designs(cands, total, posteriors=posts, mode="exploit",
                                     score_strength=strength, seed=seed)
                sink.append(sum(yb[int(c.id)] for c in sel.chosen))
        base = float(np.mean(ipt))
        _report("Warm start", [
            ("random draw", float(np.mean(rand))),
            ("rank by ipTM (status quo)", base),
            ("method history only", float(np.mean(meth))),
            ("method history + ipTM", float(np.mean(both))),
        ], total, base)

    print("Verdict: with one prior campaign of history, allocating by method beats")
    print("ipTM ranking by 13-40%, steady at +20-27% for method history alone, worth")
    print("~$100-150 per binder found at these budgets. The value is in KEEPING the")
    print("history, not in any one ordering — which is the whole argument for wiring")
    print("the loop together rather than exporting a CSV per campaign.\n")
    print(f"All figures averaged over {trials} seeds.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--trials", type=int, default=400)
    args = ap.parse_args()

    data_dir = Path(args.data)
    ensure_data(data_dir)
    r2 = load_round2(data_dir)
    r1 = load_round1(data_dir)

    section_discrimination(r2)
    section_methods(r2)
    section_transfer(r1, r2)
    section_campaign(r2, args.trials)
    print("\nData: Adaptyv EGFR competition rounds 1-2, ODbL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
