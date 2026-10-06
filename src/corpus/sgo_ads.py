"""Assemble the ADS crash corpus from the SGO incident files.

This module is the AV side of the composition study: *given* a crash, how does
an ADS crash differ from a human one? It produces one row per DISTINCT ADS
CRASH, with the structured fields the downstream analyses need, plus the funnel
that says how many records were lost at each step and why.

It is deliberately separate from `fetch.parse_ol316`, which builds the
NARRATIVE corpus for the extraction pipeline. That module answers "which rows
carry usable narrative text"; this one answers "which rows are distinct,
in-scope crashes". The two questions have different answers -- a crash whose
narrative is fully redacted is still a crash and still belongs in a composition
denominator -- so they get different modules rather than one with a flag.

-----------------------------------------------------------------------------
FOUR PLACES WHERE THE SGO FILES DO NOT BEHAVE AS COMMONLY ASSUMED
-----------------------------------------------------------------------------

(1) THE TWO FILE GENERATIONS ARE THE REGIME MARKER, AND A DATE CUTOFF IS NOT.
    The SGO third amendment (effective 16 June 2025) added damage thresholds to
    the filing duty. The obvious way to split pre/post is on the incident date
    -- except NHTSA publishes `Incident Date` as MON-YYYY, so June 2025 cannot
    be cut at the 16th at all. The file of origin can: the "Archive" file set
    holds filings made under the pre-amendment order, the current set holds
    filings made under the amended one. `filing_regime` is therefore read off
    the source file, and `incident_regime` (a coarse month comparison, with
    June 2025 marked ambiguous) is carried alongside so the two can be checked
    against each other rather than conflated. See `regime_crosstab`.

(2) `Same Incident ID` ALREADY LINKS ACROSS REPORTING ENTITIES.
    It is widely described as entity-scoped, which would leave the co-filed
    pairs (an operator and its vehicle manufacturer both filing one crash) to
    be caught by fuzzy matching. In these files it does not: 149 Cruise/GM
    groups and 139 Waymo/Transdev groups share an ID. Fuzzy matching is still
    run -- see `corpus.dedup_audit` -- but as a MEASUREMENT of residual
    duplication rather than as the primary mechanism. Asserting the gap without
    measuring it would have inflated the dedup credit and, worse, hidden
    whatever the real residual is.

(3) THE AMENDMENT DROPPED THE OPERATING-DOMAIN COLUMNS.
    `Posted Speed Limit (MPH)`, `Lighting`, `Roadway Surface` and `Roadway
    Description` exist only in the archive generation. They are exactly the
    covariates the human-side reweighting conditions on. So the structured
    domain covariates are available for 2021-07..2025-06 filings and for no
    others, and the post-amendment rows depend on narrative extraction for the
    same fields (validated against the archive rows, where both exist). Every
    column that exists in only one generation is reported by `field_coverage`
    rather than silently arriving as NaN.

(4) THE TOW / AIRBAG FIELDS CHANGED SHAPE, NOT JUST NAME.
    Archive files carry `SV Was Vehicle Towed?` and `CP Was Vehicle Towed?` as
    two Yes/No columns. The current files carry one `Was Any Vehicle Towed?`
    whose VALUE encodes both parties ("Yes Subject Vehicle, No Crash Partner").
    A rename-based harmonization reads the current column as a Yes/No and marks
    every post-amendment crash unknown. These are parsed per generation into
    `sv_towed` / `cp_towed` booleans. The same applies to airbag deployment.
    This matters more than it looks: tow-away and airbag deployment are two of
    the three legs of the objective severity floor (Tier A) in
    `report.reportability`, and losing them post-amendment would silently make
    Tier A a pre-amendment-only analysis.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

SGO_DIR = os.path.join("data", "raw", "sgo")

# The third amendment's effective date. Used only for `incident_regime`; the
# authoritative split is the file of origin (see module docstring, note 1).
AMENDMENT_DATE = pd.Timestamp("2025-06-16")
AMENDMENT_MONTH = pd.Timestamp("2025-06-01")

# ---------------------------------------------------------------------------
# Value vocabularies, read off the files rather than from the data dictionary.
# The dictionary lists what MAY be filed; these are what WAS filed. A value
# that appears in the files and not here raises rather than falling through to
# a default -- a new SGO amendment introducing a value must be a visible
# failure, which is the same discipline schema/crss_map.py applies to CRSS.
# ---------------------------------------------------------------------------

# `Driver / Operator Type` -- AND THE OPERATORS DO NOT FILL IT THE SAME WAY.
#
# Measured over the 2021-24 window, the filing conventions are incompatible:
#
#     entity      blank   in-vehicle   remote   in-vehicle+remote
#     Waymo         465           86        0                   5
#     Cruise          0            3      121                  31
#     Zoox            1           62        0                   0
#
# Waymo leaves the field BLANK for driverless operation. Cruise never files
# blank; it files `Remote (Commercial / Test)` for operations with nobody in
# the vehicle. Treating blank as the only driverless value therefore does not
# select "driverless crashes" -- it selects "Waymo crashes". Measured: the
# driverless 2021-24 sample under that rule is 455 of 456 Waymo, which silently
# turns a multi-operator study into a Waymo study.
#
# Two definitions are carried, and every driverless result is reported under
# both, because the right answer is genuinely arguable:
#
#   STRICT      blank only. What the field literally distinguishes, and what a
#               reader comparing to IIHS's "driverless" would probably assume.
#               Biased toward Waymo by the convention above.
#   NO_ONBOARD  blank OR remote. "Nobody physically in the vehicle", which is
#               the operational property that matters for crash causation --
#               no one can take the wheel. Admits Cruise.
#
# The ambiguity that keeps this from being settled: a `Remote` operator may be
# actively teleoperating, which is arguably a human driving rather than an ADS.
# SGO does not distinguish supervision-on-standby from active teleoperation, so
# neither definition is clearly correct and both are reported.
_DRIVERLESS_STRICT = {"", "none", "nan"}
_REMOTE_ONLY = {"remote (commercial / test)"}
# Someone is aboard in all of these, so they are supervised under either rule.
_SUPERVISED = {
    "in-vehicle (commercial / test)",
    "in-vehicle and remote (commercial / test)",
    "consumer",
}
_OPERATOR_UNKNOWN = {"other, see narrative", "unknown"}

DRIVERLESS_DEFINITIONS = ("strict", "no_onboard")

# `Roadway Type`. The SGO order covers crashes on publicly accessible roads;
# parking lots are the one filed value that is not one. "Unknown" is kept and
# flagged rather than dropped: dropping it would condition the corpus on the
# road type being known, which is not independent of how minor the crash was.
_PUBLIC_ROAD = {"street", "intersection", "highway / freeway", "traffic circle",
                "rural road"}
_NOT_PUBLIC_ROAD = {"parking lot"}
_ROAD_UNKNOWN = {"", "unknown", "nan"}

# `Crash With`. Fixed objects and animals are single-vehicle crashes, not
# non-crashes; they stay in. There is no structured "this was not a crash"
# value -- that determination needs the narrative, which is why `true_crash`
# below is a structured PRE-filter only and the residual is left to the
# narrative-side exclusion in `report.reportability`.
_NON_MOTORIST_PREFIX = "non-motorist:"


def _s(v) -> str:
    """Normalize a raw SGO cell to a lowercase comparison key."""
    if v is None:
        return ""
    s = str(v).strip().lower()
    return "" if s in {"nan", "none", "n/a", "-"} else s


def _yes(v) -> Optional[bool]:
    """Tri-state Yes/No/unknown for the archive-generation Y-N columns."""
    s = _s(v)
    if s in {"yes", "y", "true"}:
        return True
    if s in {"no", "n", "false"}:
        return False
    return None


def _party_flag(v, party: str) -> Optional[bool]:
    """Decode one party out of a current-generation combined-value column.

    The amended files answer "Was Any Vehicle Towed?" with a sentence naming
    both parties, e.g. "Yes Subject Vehicle, No Crash Partner". Each party's
    answer is the word immediately before its name, so the value is split on
    the comma and the clause mentioning `party` is read.

    Returns None when the clause is absent or says Unknown, which is a real
    third state here: "Unknown Crash Partner" appears in these files and means
    the filer could not determine it, not that the answer was no.
    """
    s = _s(v)
    if not s:
        return None
    if s in {"not applicable", "unknown"}:
        return None
    for clause in s.split(","):
        c = clause.strip()
        if party not in c:
            continue
        if c.startswith("yes"):
            return True
        if c.startswith("no"):
            return False
        return None          # "Unknown Crash Partner"
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_ads_files(sgo_dir: str = SGO_DIR) -> pd.DataFrame:
    """Read the ADS incident files from both generations, tagged by regime.

    ADS files ONLY. The ADAS (SAE L2) files describe crashes in which a human
    was driving with driver assistance active, so they answer a different
    question and are not pooled here; the OTHER file covers vehicles under
    neither order. Restricting to ADS is the first of the exclusions the
    composition question requires, and it is applied at read time so no
    downstream step can reintroduce an L2 crash by accident.
    """
    from fetch.parse_ol316 import _read_sgo_csv

    paths = sorted(glob.glob(os.path.join(sgo_dir, "*Incident_Reports_ADS.csv")))
    if not paths:
        raise FileNotFoundError(
            f"no ADS incident files in {sgo_dir}/. Run `python3 -m fetch.fetch_sgo`.")

    frames = []
    for p in paths:
        df = _read_sgo_csv(p)
        if df is None:
            continue
        base = os.path.basename(p)
        # The generation is what the file IS, not what a column says. See
        # module docstring note (1).
        df = df.assign(_source_file=base,
                       filing_regime="pre" if base.startswith("Archive") else "post")
        frames.append(df)
    if not frames:
        raise RuntimeError(f"ADS files in {sgo_dir}/ exist but none could be decoded")
    return pd.concat(frames, ignore_index=True)


def collapse_versions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per Report ID, keeping the highest Report Version.

    A report amended after the third amendment appears in BOTH file sets, at a
    low version in the archive and a higher one in the current file, usually
    with different narrative text. Keeping the max version keeps the filing as
    it now stands -- which also means the retained row's `filing_regime` is the
    regime of the LATEST amendment, not of the original filing. That
    distinction is preserved as `first_filed_regime` rather than lost, because
    the reportability regime check needs to know which threshold
    was in force when the duty to file arose.
    """
    ver = pd.to_numeric(df.get("Report Version"), errors="coerce").fillna(-1)
    out = df.assign(_version=ver)

    # Which regime did this Report ID FIRST appear under? "pre" sorts before
    # "post", so the min over the group is the earliest generation seen.
    first = (out.groupby("Report ID")["filing_regime"]
                .agg(lambda s: "pre" if (s == "pre").any() else "post"))
    n_versions = out.groupby("Report ID")["_version"].nunique()

    out = (out.sort_values("_version", ascending=False)
              .drop_duplicates(subset="Report ID", keep="first")
              .reset_index(drop=True))
    out["first_filed_regime"] = out["Report ID"].map(first)
    out["n_report_versions"] = out["Report ID"].map(n_versions).astype(int)
    return out.drop(columns="_version")


# ---------------------------------------------------------------------------
# Derived fields, harmonized across the two generations
# ---------------------------------------------------------------------------
def _incident_date(df: pd.DataFrame) -> pd.Series:
    """Incident date as a month timestamp. NHTSA coarsens the day away."""
    return pd.to_datetime(df.get("Incident Date"), format="%b-%Y", errors="coerce")


def derive_fields(df: pd.DataFrame, driverless_def: str = "strict"
                  ) -> pd.DataFrame:
    """Add the harmonized columns every downstream analysis keys off.

    Every field here is built from STRUCTURED SGO columns only. Nothing in this
    function reads the narrative: the point of the composition study is that
    the narrative-derived layer can be validated against a structured key, and
    that validation is only meaningful if the key was built independently.
    """
    d = df.copy()
    d["incident_month"] = _incident_date(d)
    d["incident_year"] = d["incident_month"].dt.year
    # Submission date is published as MON-YYYY, same as the incident date.
    d["submission_date"] = pd.to_datetime(d.get("Report Submission Date"),
                                          format="%b-%Y", errors="coerce")

    # --- regime -----------------------------------------------------------
    # June 2025 straddles the 16th and the day is not published, so it is
    # neither pre nor post: marked ambiguous and excluded from the sharp
    # regime contrast rather than assigned by coin flip.
    m = d["incident_month"]
    d["incident_regime"] = np.where(
        m.isna(), "unknown",
        np.where(m < AMENDMENT_MONTH, "pre",
                 np.where(m > AMENDMENT_MONTH, "post", "ambiguous")))

    # --- operator supervision --------------------------------------------
    op = d.get("Driver / Operator Type", pd.Series("", index=d.index)).map(_s)
    d["operator_type_raw"] = op
    # Both definitions are always computed; `driverless` is an alias set by
    # `driverless_def` so downstream code has one column to read while the
    # sensitivity stays available on the same frame.
    d["driverless_strict"] = np.where(
        op.isin(_DRIVERLESS_STRICT), True,
        np.where(op.isin(_SUPERVISED | _REMOTE_ONLY), False, None)).astype("object")
    d["driverless_no_onboard"] = np.where(
        op.isin(_DRIVERLESS_STRICT | _REMOTE_ONLY), True,
        np.where(op.isin(_SUPERVISED), False, None)).astype("object")
    d["driverless"] = d[f"driverless_{driverless_def}"]
    unseen_op = (set(op.unique()) - _DRIVERLESS_STRICT - _REMOTE_ONLY
                 - _SUPERVISED - _OPERATOR_UNKNOWN)
    if unseen_op:
        raise ValueError(
            "unrecognised Driver / Operator Type value(s) "
            f"{sorted(unseen_op)!r}. Classify them in corpus.sgo_ads before "
            "running: an unclassified operator type silently becomes 'unknown' "
            "supervision and drops out of the primary driverless analysis.")

    # --- ADS engaged at impact -------------------------------------------
    # Archive generation: `Automation System Engaged?` == ADS is the only
    # signal. Current generation adds `Engagement Status`, which grades the
    # claim; 'Alleged Engaged' is the reporting entity asserting engagement it
    # has not verified, and is dropped for the same reason distant_map.py
    # drops it -- an alleged value is the entity's position, not a measurement.
    eng_sys = d.get("Automation System Engaged?", pd.Series("", index=d.index)).map(_s)
    eng_status = d.get("Engagement Status", pd.Series("", index=d.index)).map(_s)
    verified = eng_status.eq("verified engaged")
    not_engaged = eng_status.eq("verified not engaged")
    d["ads_engaged"] = np.where(
        not_engaged, False,
        np.where(verified | (eng_sys.eq("ads") & eng_status.eq("")), True, None)
    ).astype("object")

    # --- public road ------------------------------------------------------
    rt = d.get("Roadway Type", pd.Series("", index=d.index)).map(_s)
    d["roadway_type"] = rt
    d["public_road"] = np.where(rt.isin(_PUBLIC_ROAD), True,
                                np.where(rt.isin(_NOT_PUBLIC_ROAD), False,
                                         None)).astype("object")

    # --- tow-away and airbag, both generations ---------------------------
    # See module docstring note (4): two Yes/No columns pre-amendment, one
    # combined-value column post-amendment.
    def _two_gen(pre_col: str, post_col: str, party: str) -> pd.Series:
        pre = d.get(pre_col, pd.Series(np.nan, index=d.index)).map(_yes)
        post = d.get(post_col, pd.Series(np.nan, index=d.index)).map(
            lambda v: _party_flag(v, party))
        return pre.where(pre.notna(), post)

    d["sv_towed"] = _two_gen("SV Was Vehicle Towed?", "Was Any Vehicle Towed?",
                             "subject vehicle")
    d["cp_towed"] = _two_gen("CP Was Vehicle Towed?", "Was Any Vehicle Towed?",
                             "crash partner")
    d["sv_airbag"] = _two_gen("SV Any Air Bags Deployed?", "Any Air Bags Deployed?",
                              "subject vehicle")
    d["cp_airbag"] = _two_gen("CP Any Air Bags Deployed?", "Any Air Bags Deployed?",
                              "crash partner")
    # `any_*` is True if either party, False only if both are known-False.
    # A None on one side with False on the other stays None: "the AV was not
    # towed and we do not know about the other car" is not "no tow-away".
    def _either(a: pd.Series, b: pd.Series) -> pd.Series:
        both = pd.concat([a, b], axis=1)
        out = pd.Series(pd.NA, index=a.index, dtype="object")
        out[both.eq(True).any(axis=1)] = True
        out[both.eq(False).all(axis=1)] = False
        return out

    d["any_towed"] = _either(d["sv_towed"], d["cp_towed"])
    d["any_airbag"] = _either(d["sv_airbag"], d["cp_airbag"])

    # --- police involvement ----------------------------------------------
    # The one signal both generations carry is a NAMED investigating agency.
    # The archive's explicit `Law Enforcement Investigating?` is kept beside
    # it as `police_investigating_pre` so the harmonized indicator can be
    # validated against it where both exist (see `police_agreement`), rather
    # than assumed equivalent.
    agency = d.get("Investigating Agency", pd.Series("", index=d.index)).map(_s)
    named = ~agency.isin({"", "n/a", "na", "none", "not applicable", "unknown"})
    d["investigating_agency"] = agency
    d["police_named_agency"] = named
    d["police_investigating_pre"] = d.get(
        "Law Enforcement Investigating?", pd.Series(np.nan, index=d.index)).map(_yes)

    # --- operating-domain covariates (archive generation only) -----------
    # See module docstring note (3). Present as columns in every row so the
    # coverage is visible as NaN rather than as a missing column.
    # `d.get(col)` returns None for an absent column, and `pd.to_numeric(None)`
    # returns a SCALAR nan rather than a Series -- which then fails on `.where`.
    # Pooling both generations always supplies these columns, so this only
    # surfaces on a single-generation frame; the explicit default keeps that
    # case working instead of failing on a shape mismatch.
    def _num(col: str) -> pd.Series:
        s = d[col] if col in d.columns else pd.Series(np.nan, index=d.index)
        return pd.to_numeric(s, errors="coerce")

    spd = _num("Posted Speed Limit (MPH)")
    d["posted_speed_limit"] = spd.where((spd > 0) & (spd <= 85))
    # `_s` returns "" for a missing cell AND for a column that does not exist in
    # this generation. Left as "" those are indistinguishable from a filed
    # value, and `field_coverage` would report a dropped column at 100%
    # coverage -- which is exactly the failure this corpus exists to make
    # visible. Empty maps to NA for every derived string field.
    d["lighting_struct"] = d.get(
        "Lighting", pd.Series(np.nan, index=d.index)).map(_s).replace("", pd.NA)
    d["sv_precrash_speed"] = _num("SV Precrash Speed (MPH)")

    # --- occupancy --------------------------------------------------------
    # SGO HAS NO OCCUPANCY FIELD, BUT THE SEAT-BELT FIELD ANSWERS IT ANYWAY.
    # `Were All Passengers Belted?` carries an explicit "No Passenger In
    # Vehicle" value in both generations, so whether anyone was aboard is
    # recorded for 99.8% of incidents at no cost. This was nearly extracted
    # from narratives instead, which would have been slower, dearer and worse.
    #
    # It matters because roughly half of driverless ADS crashes have no
    # occupant, which mechanically suppresses first-party injury with no
    # behavioural difference behind it. Any ADS-versus-human severity
    # comparison that ignores it is partly measuring who was in the car.
    # Measured here: 52.2% of driverless crashes are unoccupied against 2.9%
    # of supervised ones, and IIHS reported ~48% for Waymo driverless.
    #
    # The two generations phrase the values differently ("No Passengers in
    # Vehicle" vs "Subject Vehicle - No Passenger In Vehicle"), so both are
    # parsed rather than one being assumed canonical.
    def _occupied(pre_val, post_val) -> Optional[bool]:
        for v in (pre_val, post_val):
            s = _s(v)
            if not s:
                continue
            if "no passenger" in s:
                return False
            if "belted" in s or s in {"yes", "no, see narrative"}:
                return True
        return None

    pre_belt = d.get("SV Were All Passengers Belted?",
                     pd.Series(np.nan, index=d.index))
    post_belt = d.get("Were All Passengers Belted?",
                      pd.Series(np.nan, index=d.index))
    d["sv_occupied"] = [_occupied(p, q) for p, q in zip(pre_belt, post_belt)]

    # --- crash partner class ---------------------------------------------
    cw = d.get("Crash With", pd.Series("", index=d.index)).map(_s)
    d["crash_with"] = cw.replace("", pd.NA)
    d["vru_partner"] = cw.str.startswith(_NON_MOTORIST_PREFIX) | cw.eq("motorcycle")
    d["no_mv_partner"] = cw.isin({"other fixed object", "pole / tree", "animal", ""})

    # --- contact areas ----------------------------------------------------
    for side in ("SV", "CP"):
        cols = [c for c in d.columns if c.startswith(f"{side} Contact Area")]
        for c in cols:
            zone = c.split("-", 1)[1].strip().lower().replace(" ", "_")
            d[f"{side.lower()}_contact_{zone}"] = d[c].map(
                lambda v: _s(v) in {"y", "yes"})
    return d


# ---------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------
# Applied in a fixed order so the funnel is reproducible, and each one records
# how many rows it removed. IIHS removed roughly 25% of SGO incidents with the
# equivalent set; publishing the per-step counts is what lets a reader see
# whether this corpus lost the same share for the same reasons.
EXCLUSIONS = [
    ("not_ads_engaged",
     lambda d: d["ads_engaged"].ne(True),
     "ADS not verified engaged at the time of the incident"),
    ("not_public_road",
     lambda d: d["public_road"].eq(False),
     "incident on a parking lot rather than a publicly accessible road"),
]


def apply_exclusions(d: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Drop out-of-scope rows, returning the frame and a per-step funnel.

    NOTE ON WHAT IS *NOT* EXCLUDED HERE. IIHS's third exclusion -- "this filing
    does not describe a crash" -- has no structured counterpart: SGO has no
    not-a-crash value, and the determination is a narrative one (a reported
    contact that turned out to be a pothole, a door opened into a stationary
    AV, a filing made out of caution). It is therefore left to the narrative
    layer in `report.reportability`, where it is coded with an evidence span
    and validated against human coding, rather than approximated here with a
    keyword rule that would look objective and not be.
    """
    funnel = [{"step": "ads_reports_deduped_by_version", "n": len(d), "removed": 0,
               "why": "one row per Report ID at the highest Report Version"}]
    out = d
    for name, pred, why in EXCLUSIONS:
        mask = pred(out)
        funnel.append({"step": name, "n": int((~mask).sum()),
                       "removed": int(mask.sum()), "why": why})
        out = out[~mask].copy()
    return out, funnel


# ---------------------------------------------------------------------------
# Incident-level deduplication
# ---------------------------------------------------------------------------
def group_incidents(d: pd.DataFrame) -> pd.DataFrame:
    """Assign `incident_key`: one value per distinct crash.

    `Same Incident ID` is the primary key and, contrary to the usual account of
    it, does link across reporting entities in these files (module docstring
    note 2). Two corrections are still needed:

      - A BLANK `Same Incident ID` is not a group. Twelve ADS rows share the
        empty string; grouping on it would merge twelve unrelated crashes into
        one. Blanks are given per-report keys.
      - Cross-entity filings that did NOT coordinate on an ID remain. Those are
        found by `corpus.dedup_audit`, which fuzzy-matches on date, city and
        narrative similarity and returns pairs to merge; this function accepts
        the resulting merge map so the fuzzy layer never has to mutate the
        frame itself.
    """
    sid = d["Same Incident ID"].map(_s)
    key = np.where(sid.eq(""), "rid:" + d["Report ID"].astype(str), "sid:" + sid)
    return d.assign(incident_key=key)


def apply_merges(d: pd.DataFrame, merges: dict[str, str]) -> pd.DataFrame:
    """Fold fuzzy-matched incident keys together.

    `merges` maps an incident_key onto the key it should join. Applied
    transitively so a chain a->b->c collapses to one group.
    """
    def resolve(k: str, seen=None) -> str:
        seen = seen or set()
        while k in merges and k not in seen:
            seen.add(k)
            k = merges[k]
        return k
    return d.assign(incident_key=d["incident_key"].map(resolve))


def collapse_to_incidents(d: pd.DataFrame) -> pd.DataFrame:
    """One row per incident_key, choosing a canonical filing per crash.

    WHICH FILING REPRESENTS A CO-FILED CRASH. When an operator and a vehicle
    manufacturer both file, the two records describe the same crash from
    different vantage points and their structured fields can disagree. The
    operator's filing is preferred, on the ground that the operator ran the
    trip and holds the telematics; ties break on the most recent submission and
    then on the longest narrative. Which record won is recorded in
    `n_co_filings` and `co_filing_entities` so a disagreement analysis remains
    possible -- the alternative, averaging the two, would invent a filing that
    neither entity made.
    """
    # Vehicle manufacturers that co-file behind an operator. Ranked lower so
    # the operator's own account is the one kept.
    manufacturer_like = {
        "general motors, llc", "ford motor company", "paccar incorporated",
        "hyundai motor america", "toyota motor engineering & manufacturing",
        "daimler trucks north america, llc", "nvidia corp", "navistar, inc.",
        "mercedes-benz usa, llc", "robert bosch, llc", "vinfast auto, llc",
        "local motors industries",
    }
    ent = d["Reporting Entity"].map(_s)
    rank = np.where(ent.isin(manufacturer_like), 1, 0)

    grp = d.groupby("incident_key")
    meta = pd.DataFrame({
        "n_co_filings": grp.size(),
        "co_filing_entities": grp["Reporting Entity"].agg(
            lambda s: "; ".join(sorted(set(s.astype(str))))),
        "n_entities": grp["Reporting Entity"].nunique(),
    })

    narr_len = d.get("Narrative", pd.Series("", index=d.index)).astype(str).str.len()
    ordered = (d.assign(_rank=rank, _nlen=narr_len)
                .sort_values(["_rank", "submission_date", "_nlen"],
                             ascending=[True, False, False]))
    out = (ordered.drop_duplicates("incident_key", keep="first")
                  .drop(columns=["_rank", "_nlen"])
                  .reset_index(drop=True))
    return out.merge(meta, left_on="incident_key", right_index=True, how="left")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def field_coverage(d: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Non-null share of each field, split by filing regime.

    Exists because the amendment dropped columns rather than blanking them
    (module docstring note 3). A field at 100% pre and 0% post is a schema
    change, not missing data, and the two must not be pooled into one
    "coverage" number that describes neither.
    """
    rows = []
    for f in fields:
        if f not in d.columns:
            rows.append({"field": f, "pre": np.nan, "post": np.nan,
                         "note": "column absent from both generations"})
            continue
        r = {"field": f}
        for reg in ("pre", "post"):
            sub = d[d["filing_regime"] == reg]
            r[reg] = float(sub[f].notna().mean()) if len(sub) else np.nan
        r["note"] = ("regime-specific column"
                     if (r["pre"] == 0) != (r["post"] == 0) else "")
        rows.append(r)
    return pd.DataFrame(rows)


def regime_crosstab(d: pd.DataFrame) -> pd.DataFrame:
    """Filing regime against incident regime.

    The off-diagonal is real and interpretable: a pre-amendment incident filed
    (or amended) after June 2025 appears as incident_regime=pre,
    filing_regime=post. `report.reportability.regime_check` needs the FILING
    regime, because that is the threshold the filer was subject to; an analysis
    that used the incident date would attribute post-amendment filing behaviour
    to crashes reported under the old rule.
    """
    return pd.crosstab(d["incident_regime"], d["filing_regime"])


def police_agreement(d: pd.DataFrame) -> dict:
    """Does the harmonized police indicator match the archive's explicit one?

    The harmonized indicator is "an investigating agency was named", because it
    is the only police signal the current generation carries. The archive
    generation also carries an explicit `Law Enforcement Investigating?`. Where
    both exist they should agree; the extent to which they do bounds how much
    of the pre/post difference in police involvement is a change in reporting
    behaviour rather than a change in what was measured.
    """
    sub = d[d["police_investigating_pre"].notna()]
    if sub.empty:
        return {"n": 0}
    a = sub["police_named_agency"].astype(bool)
    b = sub["police_investigating_pre"].astype(bool)
    return {
        "n": int(len(sub)),
        "agreement": float((a == b).mean()),
        "named_no_explicit": int((a & ~b).sum()),
        "explicit_no_named": int((~a & b).sum()),
        "named_rate": float(a.mean()),
        "explicit_rate": float(b.mean()),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build(sgo_dir: str = SGO_DIR, merges: Optional[dict] = None,
          driverless_def: str = "strict") -> tuple:
    """Full ADS corpus assembly. Returns (incident frame, funnel report)."""
    raw = load_ads_files(sgo_dir)
    n_raw = len(raw)

    d = collapse_versions(raw)
    n_reports = len(d)

    d = derive_fields(d, driverless_def=driverless_def)
    d, funnel = apply_exclusions(d)

    d = group_incidents(d)
    n_before_merge = d["incident_key"].nunique()
    if merges:
        d = apply_merges(d, merges)
    n_keys = d["incident_key"].nunique()

    inc = collapse_to_incidents(d)

    report = {
        "n_raw_rows": n_raw,
        "n_report_ids": n_reports,
        "version_collapse_removed": n_raw - n_reports,
        "exclusions": funnel,
        "n_after_exclusions": len(d),
        "n_incidents_by_same_incident_id": n_before_merge,
        "n_incidents_after_fuzzy_merge": n_keys,
        "fuzzy_merges_applied": len(merges or {}),
        "n_incidents": len(inc),
        "co_filed_incidents": int((inc["n_entities"] > 1).sum()),
        "by_regime": inc["filing_regime"].value_counts().to_dict(),
        "by_supervision": inc["driverless"].value_counts(dropna=False).to_dict(),
        "regime_crosstab": regime_crosstab(inc).to_dict(),
        "police_indicator_agreement": police_agreement(inc),
    }
    return inc, report


DOMAIN_FIELDS = ["posted_speed_limit", "lighting_struct", "sv_precrash_speed",
                 "any_towed", "any_airbag", "police_named_agency",
                 "Highest Injury Severity Alleged", "SV Pre-Crash Movement",
                 "CP Pre-Crash Movement"]


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Assemble the ADS crash corpus.")
    ap.add_argument("--sgo-dir", default=SGO_DIR)
    ap.add_argument("--merges", default=None,
                    help="JSON file of fuzzy incident-key merges from "
                         "corpus.dedup_audit. Omit to use Same Incident ID alone.")
    ap.add_argument("--out", default=os.path.join("data", "interim", "ads_incidents.parquet"))
    ap.add_argument("--report", default=os.path.join("data", "processed", "ads_corpus.json"))
    ap.add_argument("--driverless-def", default="strict",
                    choices=list(DRIVERLESS_DEFINITIONS),
                    help="'strict' = blank Driver/Operator Type only (Waymo's "
                         "convention); 'no_onboard' = blank OR remote, i.e. "
                         "nobody physically in the vehicle (admits Cruise, "
                         "which never files blank). Both columns are always "
                         "written; this selects which one `driverless` aliases.")
    a = ap.parse_args()

    merges = None
    if a.merges and os.path.exists(a.merges):
        with open(a.merges) as f:
            merges = json.load(f).get("merges", {})

    inc, rep = build(a.sgo_dir, merges, driverless_def=a.driverless_def)

    cov = field_coverage(inc, DOMAIN_FIELDS)
    rep["field_coverage"] = cov.to_dict(orient="records")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    inc.to_parquet(a.out, index=False)
    with open(a.report, "w") as f:
        json.dump(rep, f, indent=2, default=str)

    print(f"[ads] {rep['n_raw_rows']:,} raw rows -> {rep['n_report_ids']:,} Report IDs "
          f"-> {rep['n_after_exclusions']:,} in scope -> {rep['n_incidents']:,} incidents")
    for s in rep["exclusions"]:
        if s["removed"]:
            print(f"[ads]   -{s['removed']:,} {s['step']}: {s['why']}")
    print(f"[ads] co-filed by >1 entity: {rep['co_filed_incidents']:,}")
    print(f"[ads] regime: {rep['by_regime']}")
    print(f"[ads] supervision ({a.driverless_def}): {rep['by_supervision']}")
    print(f"[ads] driverless under each definition -- the operators do not fill "
          f"Driver / Operator Type the same way:")
    for defn in DRIVERLESS_DEFINITIONS:
        col = inc[f"driverless_{defn}"]
        w = inc.loc[col.eq(True), "Reporting Entity"].astype(str)
        share = w.str.contains("Waymo|Transdev").mean() if len(w) else float("nan")
        print(f"[ads]   {defn:11s} n={int(col.eq(True).sum()):5d}  "
              f"Waymo/Transdev share {share:.0%}")
    print(f"[ads] police indicator agreement (archive rows): "
          f"{rep['police_indicator_agreement']}")
    print("\n[ads] field coverage by filing regime (the amendment dropped columns):")
    print(cov.to_string(index=False))
    print(f"\n[ads] wrote {a.out} and {a.report}")


if __name__ == "__main__":
    main()
