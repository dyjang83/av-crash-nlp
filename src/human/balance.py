"""Reweight the human sample onto the ADS operating domain.

WHY THIS EXISTS AFTER THE HARD RESTRICTIONS. `human.crss_extract.restrict`
already cuts CRSS to urban, non-interstate, non-freeway, posted speed <= 45.
That removes the gross mismatch but not the fine one: within urban surface
streets, ADS fleets are concentrated on low-speed roads at particular times of
day in particular weather, and the human sample is not. IIHS measured roughly
48% of ADS crashes on roads posted <= 25 mph against roughly 8% for humans --
a six-fold difference INSIDE the restricted domain. A composition comparison
that ignores it reports the road network, not the driving.

Reweighting fixes the marginals it is given and nothing else. It is the weaker
half of the alignment; the hard restrictions are the stronger half, and the
results are reported both ways so a reader can see how much the weights are
doing.

-----------------------------------------------------------------------------
WHAT ENTROPY BALANCING DOES, AND WHY IT IS PREFERRED TO A PROPENSITY MODEL
-----------------------------------------------------------------------------
It finds new weights w that (a) exactly reproduce the target's covariate means
and (b) stay as close as possible to the original survey weights in the
Kullback-Leibler sense. Exactness matters here: a propensity model reduces
imbalance without guaranteeing it, and then the residual imbalance is an
unreported confounder. Staying close to the design weights matters because
those weights are the only thing making the human side representative at all --
this is a reweighting OF a survey weight, not a replacement for it.

The dual has a closed form up to a convex minimisation: w_i proportional to
q_i * exp(-lambda'X_i), with lambda solving the moment conditions. `q_i` is
the CRSS design weight, so the design is carried through rather than discarded.

-----------------------------------------------------------------------------
THE THREE THINGS THIS CANNOT DO, STATED RATHER THAN HOPED AWAY
-----------------------------------------------------------------------------
(1) SPEED LIMIT IS PRE-AMENDMENT ONLY. The SGO third amendment dropped
    `Posted Speed Limit (MPH)`, and a validation run established that
    narratives essentially never state it (298 of 298 extracted `unknown`).
    So a speed-limit-balanced comparison is available for the 1,384
    pre-amendment ADS incidents and for no others. `build_target` refuses to
    silently drop post-amendment rows into an `unknown` cell: pass
    `require_speed=True` and it restricts the ADS side explicitly, with the
    retained count reported.

(2) EXTREME WEIGHTS ARE THE EXPECTED OUTCOME, NOT A BUG. When a target cell is
    six times rarer in the source, the weight on that cell is six times larger,
    and a handful of CRSS vehicles end up carrying the estimate. That is
    variance, and it is why the effective sample size is reported next to every
    balanced estimate and why the trimmed version is reported alongside the
    untrimmed one. An estimate that moves a lot under trimming rests on a few
    rows.

(3) TIME OF DAY AND OCCUPANCY ARE NOT ADJUSTED. Hour is available on both
    sides and can be added to the covariate set; OCCUPANCY cannot. IIHS found
    roughly 48% of Waymo reportable crashes had no occupant, which mechanically
    lowers injury severity with no driving difference behind it. There is no
    CRSS counterpart to reweight toward, so the answer is a sensitivity run on
    occupied-only ADS crashes, not a weight.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

# Covariate cells. Deliberately coarse: entropy balancing matches the means of
# whatever indicators it is given, and a fine cross-classification produces
# cells with one or two source rows whose weights then explode. These four
# variables are matched as SEPARATE marginals plus the speed x lighting
# interaction, not as a full cross, for that reason.
DEFAULT_COVARIATES = ["speed_bin", "lighting_bin", "intersection", "weather_bin"]


def _indicators(d: pd.DataFrame, covariates: list,
                levels: Optional[dict] = None) -> tuple:
    """One-hot design matrix, dropping one level per variable.

    One level per variable is dropped because the weights are constrained to
    sum to one: with every level present the moment conditions are linearly
    dependent and the dual is singular.
    """
    levels = levels or {c: sorted(d[c].dropna().astype(str).unique())
                        for c in covariates}
    cols, names = [], []
    for c in covariates:
        lv = levels[c]
        for v in lv[1:]:                       # drop the first as reference
            cols.append((d[c].astype(str) == v).to_numpy(float))
            names.append(f"{c}={v}")
    X = np.column_stack(cols) if cols else np.zeros((len(d), 0))
    return X, names, levels


def entropy_balance(X: np.ndarray, base_w: np.ndarray, target: np.ndarray,
                    max_iter: int = 500, tol: float = 1e-8) -> tuple:
    """Weights matching `target` moments, closest to `base_w` in KL.

    Solves the convex dual with Newton steps. Returns (weights, info); the
    weights are normalised to sum to one. `info["max_imbalance"]` is the
    largest remaining absolute difference between the achieved and requested
    moments -- it should be at machine precision, and a large value means the
    target was not attainable inside the convex hull of the source, which is a
    substantive finding about domain overlap and NOT something to paper over
    with more iterations.
    """
    from scipy.optimize import minimize

    q = np.asarray(base_w, float)
    q = q / q.sum()
    if X.shape[1] == 0:
        return q, {"converged": True, "max_imbalance": 0.0, "n_moments": 0}

    def obj(lam):
        z = -X @ lam
        # Shift before exponentiating to avoid overflow, then ADD THE SHIFT
        # BACK into the objective. log(sum q e^z) = log(sum q e^(z-m)) + m, and
        # m depends on lambda -- dropping it leaves the returned objective
        # differing from the true one by a lambda-dependent constant while the
        # gradient stays correct for the true one. L-BFGS-B then walks an
        # objective whose gradient does not match it and stops early, having
        # moved the weights partway and reported convergence failure.
        m = z.max()
        w = q * np.exp(z - m)
        s = w.sum()
        # Dual: L(lam) = log(sum_i q_i exp(-lam'X_i)) + lam'target, whose
        # gradient is target - E_w[X]. The opposite sign makes L-BFGS-B report
        # convergence at the starting point and hand back the original weights
        # unchanged -- which looks like "balancing was unnecessary" rather than
        # like a failure, so it is worth getting right rather than discovering
        # downstream.
        return (np.log(s) + m + lam @ target,
                target - (X * (w / s)[:, None]).sum(0))

    res = minimize(obj, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": tol, "gtol": tol})
    z = -X @ res.x
    w = q * np.exp(z - z.max())
    w = w / w.sum()
    return w, {
        "converged": bool(res.success),
        "max_imbalance": float(np.abs((X * w[:, None]).sum(0) - target).max()),
        "n_moments": int(X.shape[1]),
        "message": str(res.message),
    }


def effective_n(w: np.ndarray) -> float:
    """Kish effective sample size.

    The number that says what a balanced estimate actually rests on. A
    reweighting that turns 168,814 vehicles into an effective 900 has not
    produced a large-sample comparison, whatever the row count says.
    """
    w = np.asarray(w, float)
    s = w.sum()
    return float(s * s / max((w * w).sum(), 1e-300))


def trim(w: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    """Cap weights at a quantile and renormalise.

    Trimming trades bias for variance: the trimmed estimator no longer matches
    the target moments exactly. Both are reported for that reason -- the
    untrimmed one is unbiased for the balanced population and possibly
    unstable, the trimmed one is stable and slightly off-target.
    """
    w = np.asarray(w, float)
    cap = np.quantile(w, quantile)
    out = np.minimum(w, cap)
    return out / out.sum()


def bin_weather(s: pd.Series) -> pd.Series:
    """Precipitation type, the distinction both sides can express.

    Mirrors `schema.crss_map._CRSS_WEATHER`: the schema has no 'cloudy'
    member, so dry-but-overcast pools with clear rather than dropping 13% of
    the human sample.
    """
    t = s.astype(str).str.strip().str.lower()
    return pd.Series(
        np.where(t.str.contains("rain|drizzle"), "rain",
                 np.where(t.str.contains("snow|sleet|hail"), "snow",
                          np.where(t.str.contains("fog|smog|smoke"), "fog",
                                   np.where(t.str.contains("clear|cloud"),
                                            "clear_or_cloudy", "other")))),
        index=s.index)


def build_target(ads: pd.DataFrame, covariates: list, levels: dict,
                 require_speed: bool = True) -> tuple:
    """Target moments from the ADS side, with the speed-limit caveat enforced.

    `require_speed=True` restricts the ADS side to incidents with a known
    posted speed limit -- which, per the module docstring, means pre-amendment
    filings. The alternative is to let post-amendment rows sit in an `unknown`
    speed cell and then balance the human side toward a cell that means "we
    stopped collecting this", which would be balancing toward a data-collection
    artifact.
    """
    a = ads.copy()
    n_all = len(a)
    if require_speed and "speed_bin" in covariates:
        a = a[a["speed_bin"].ne("unknown")]
    if a.empty:
        raise SystemExit(
            "[balance] no ADS incidents with a known posted speed limit. The "
            "third amendment dropped the column and narratives do not state it; "
            "run with --no-require-speed to balance without it.")
    X, names, _ = _indicators(a, covariates, levels)
    w = np.ones(len(a)) / len(a)             # ADS is a census: equal weights
    return (X * w[:, None]).sum(0), names, {
        "n_ads_all": int(n_all), "n_ads_used": int(len(a)),
        "speed_required": bool(require_speed and "speed_bin" in covariates),
    }


def balance_report(human: pd.DataFrame, ads: pd.DataFrame,
                   covariates: Optional[list] = None,
                   require_speed: bool = True, trim_q: float = 0.99) -> dict:
    """Balance the human weights onto the ADS domain and diagnose the result."""
    covariates = covariates or DEFAULT_COVARIATES
    covariates = [c for c in covariates if c in human.columns and c in ads.columns]

    # Shared level vocabulary, taken from the UNION so neither side can define
    # a level the other lacks and silently shift the reference category.
    levels = {c: sorted(set(human[c].dropna().astype(str))
                        | set(ads[c].dropna().astype(str)))
              for c in covariates}

    # LEVELS WITH ZERO ADS MASS BECOME HARD RESTRICTIONS, NOT ZERO WEIGHTS.
    # The ADS fleet records no snow crashes at all. Asking entropy balancing to
    # match a target of exactly zero is a boundary problem -- the dual
    # parameter diverges and the solver either fails or returns enormous
    # weights on the surviving rows. It is also the wrong description of what
    # is happening: a condition the fleet never operates in is OUTSIDE the
    # operating domain, which is a restriction, in the same way non-interstate
    # is. So those human rows are dropped and counted, and balancing runs on
    # the levels where both sides have support.
    ads_src = ads
    if require_speed and "speed_bin" in covariates:
        ads_src = ads[ads["speed_bin"].ne("unknown")]

    # THE RESTRICTION MUST BE SYMMETRIC, OR THE PROBLEM IS INFEASIBLE.
    # Dropping human rows in levels the ADS fleet never sees is only half of
    # it. The other half bites harder: the human side is hard-restricted to
    # posted speed <= 45, while the ADS side retains a handful of crashes above
    # it. Every human row then has its three speed indicators summing to
    # exactly 1, but the ADS target sums to 0.998 -- and no reweighting of rows
    # that sum to 1 can average to 0.998. The moment conditions are literally
    # unsatisfiable, the dual parameter diverges, and the solver reports
    # success while collapsing all weight onto a single cell.
    #
    # So a level absent from EITHER side is dropped from BOTH. What survives is
    # the domain where the two populations actually overlap, which is the only
    # region where a reweighted comparison means anything.
    dropped = {"human": {}, "ads": {}}
    keep_h = pd.Series(True, index=human.index)
    keep_a = pd.Series(True, index=ads.index)
    for c in covariates:
        in_ads = set(ads_src[c].dropna().astype(str))
        in_human = set(human[c].dropna().astype(str))
        shared = [v for v in levels[c] if v in in_ads and v in in_human]

        no_ads = [v for v in levels[c] if v in in_human and v not in in_ads]
        if no_ads:
            m = human[c].astype(str).isin(no_ads)
            dropped["human"][c] = {
                "levels": no_ads, "n_dropped": int(m.sum()),
                "weighted_share": float(human.loc[m, "_w"].sum()
                                        / human["_w"].sum())}
            keep_h &= ~m

        no_human = [v for v in levels[c] if v in in_ads and v not in in_human]
        if no_human:
            m = ads[c].astype(str).isin(no_human)
            dropped["ads"][c] = {"levels": no_human, "n_dropped": int(m.sum()),
                                 "share": float(m.mean())}
            keep_a &= ~m

        levels[c] = shared

    keep_mask = keep_h.to_numpy(bool).copy()   # over the ORIGINAL row order
    human = human[keep_h].copy()
    ads = ads[keep_a].copy()

    target, names, tinfo = build_target(ads, covariates, levels, require_speed)
    Xh, _, _ = _indicators(human, covariates, levels)
    q = human["_w"].to_numpy(float)

    w, info = entropy_balance(Xh, q, target)
    wt = trim(w, trim_q)

    before = (Xh * (q / q.sum())[:, None]).sum(0)
    after = (Xh * w[:, None]).sum(0)
    after_t = (Xh * wt[:, None]).sum(0)

    return {
        "covariates": covariates,
        "target_info": tinfo,
        "solver": info,
        "out_of_domain_dropped": dropped,
        "n_human": int(len(human)),
        "effective_n_before": effective_n(q),
        "effective_n_after": effective_n(w),
        "effective_n_after_trim": effective_n(wt),
        "trim_quantile": trim_q,
        "weight_ratio_max": float(w.max() / max(w.min(), 1e-300)),
        "balance": [
            {"moment": n, "ads_target": float(t), "human_before": float(b),
             "human_after": float(a_), "human_after_trim": float(at),
             "abs_gap_before": abs(float(b) - float(t)),
             "abs_gap_after": abs(float(a_) - float(t))}
            for n, t, b, a_, at in zip(names, target, before, after, after_t)
        ],
        "weights": w, "weights_trimmed": wt, "_keep_mask": keep_mask,
    }


def main():
    import argparse
    from models.composition import bin_environment

    ap = argparse.ArgumentParser(
        description="Entropy-balance the CRSS weights onto the ADS domain.")
    ap.add_argument("--ads", default=os.path.join("data", "interim",
                                                  "ads_reportability.parquet"))
    ap.add_argument("--human", default=os.path.join("data", "interim",
                                                    "crss_vehicles.parquet"))
    ap.add_argument("--driverless-only", action="store_true")
    ap.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    ap.add_argument("--no-require-speed", action="store_true",
                    help="Balance without the speed-limit moment, keeping "
                         "post-amendment ADS incidents. Reported as a separate "
                         "run, never merged with the speed-balanced one.")
    ap.add_argument("--trim-q", type=float, default=0.99)
    ap.add_argument("--out", default=os.path.join("data", "processed", "balance.json"))
    ap.add_argument("--weights-out",
                    default=os.path.join("data", "interim", "crss_balanced_weights.npz"))
    a = ap.parse_args()

    ads = bin_environment(pd.read_parquet(a.ads), "ads")
    if a.driverless_only:
        ads = ads[ads["driverless"].eq(True)]
    ads = ads[ads["incident_year"].isin(a.years)].copy()

    human = bin_environment(pd.read_parquet(a.human), "human")
    human["weather_bin"] = bin_weather(human["weather"])
    # The ADS weather columns are indicator flags, not one field; collapse them
    # into the same vocabulary the human side uses.
    wcols = {c: c.split("-", 1)[1].strip().lower()
             for c in ads.columns if c.startswith("Weather - ")}
    if wcols:
        def _ads_weather(row):
            on = [v for c, v in wcols.items()
                  if str(row.get(c, "")).strip().lower() in {"y", "yes"}]
            j = " ".join(on)
            if "rain" in j:
                return "rain"
            if "snow" in j:
                return "snow"
            if "fog" in j:
                return "fog"
            if "clear" in j or "cloudy" in j:
                return "clear_or_cloudy"
            return "other" if on else np.nan
        ads["weather_bin"] = ads.apply(_ads_weather, axis=1)

    rep = balance_report(human, ads, require_speed=not a.no_require_speed,
                         trim_q=a.trim_q)
    w, wt = rep.pop("weights"), rep.pop("weights_trimmed")

    # SAVE THE ROW SELECTION ALONGSIDE THE WEIGHTS. Balancing first drops human
    # rows in levels the ADS fleet never operates in (snow, unknown lighting),
    # so the weight vector is shorter than the frame it came from. Saving only
    # the weights makes them positionally meaningless to any consumer holding
    # the full frame -- and a consumer that aligned them anyway would attach
    # each weight to the wrong vehicle and produce a plausible, wrong answer.
    # The boolean mask is over the ORIGINAL frame's row order.
    os.makedirs(os.path.dirname(a.weights_out), exist_ok=True)
    np.savez_compressed(a.weights_out, weights=w, weights_trimmed=wt,
                        keep_mask=rep.pop("_keep_mask"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, default=str)

    t = rep["target_info"]
    print(f"[balance] covariates: {rep['covariates']}")
    print(f"[balance] ADS target from {t['n_ads_used']:,} of {t['n_ads_all']:,} "
          f"incidents (speed_limit required: {t['speed_required']})")
    print(f"[balance] solver: converged={rep['solver']['converged']}, "
          f"max residual imbalance {rep['solver']['max_imbalance']:.2e} "
          f"over {rep['solver']['n_moments']} moments")
    ood = rep["out_of_domain_dropped"]
    if ood.get("human"):
        print("[balance] levels with NO ADS support -- human rows dropped as a "
              "hard restriction, not given zero weight:")
        for c, v in ood["human"].items():
            print(f"[balance]   {c}: {v['levels']} (-{v['n_dropped']:,} rows, "
                  f"{v['weighted_share']:.1%} of weight)")
    if ood.get("ads"):
        print("[balance] levels with NO HUMAN support -- ADS rows dropped, "
              "because a target outside the human sample's convex hull makes "
              "the moment conditions unsatisfiable:")
        for c, v in ood["ads"].items():
            print(f"[balance]   {c}: {v['levels']} (-{v['n_dropped']:,} "
                  f"incidents, {v['share']:.1%})")
    print(f"[balance] human n={rep['n_human']:,}; effective n "
          f"{rep['effective_n_before']:,.0f} -> {rep['effective_n_after']:,.0f} "
          f"(trimmed at q{a.trim_q}: {rep['effective_n_after_trim']:,.0f})")
    print(f"[balance] max/min weight ratio: {rep['weight_ratio_max']:,.0f}x")
    print(f"\n{'moment':28s} {'ADS':>8s} {'human':>8s} {'balanced':>9s} "
          f"{'trimmed':>8s}")
    for b in rep["balance"]:
        print(f"{b['moment']:28s} {b['ads_target']:8.3f} {b['human_before']:8.3f} "
              f"{b['human_after']:9.3f} {b['human_after_trim']:8.3f}")
    print(f"\n[balance] wrote {a.out} and {a.weights_out}")
    print("[balance] NOT adjusted for: time of day (available, not in the "
          "default set) and occupancy (no CRSS counterpart -- use the "
          "occupied-only ADS sensitivity run instead)")


if __name__ == "__main__":
    main()
