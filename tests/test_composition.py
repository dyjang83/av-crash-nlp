"""Tests for the composition study: taxonomy, role, corpus assembly, tiers.

Synthetic fixtures only -- no number here enters the paper. What these pin are
the specific ways the composition pipeline could be silently wrong, each of
which was an actual bug caught during construction:

  - the ACC_TYPE role map inverting striking and struck
  - a vehicle-level filter destroying the quasi-induced-exposure pairing
  - comparing ADS and human shares over different category spaces
  - the two SGO file generations encoding tow/airbag differently
  - narrative similarity being used as a deduplication decision signal
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema.acc_type import (CATEGORIES, crss_acc_type, human_baseline_ratio,
                             is_two_vehicle, sgo_distant_key, verify_crss_names)
from schema.composition_schema import AccTypeCategory, CompositionExtraction
from corpus.sgo_ads import _party_flag, _yes, derive_fields
from models.composition import (REACHABLE_BY_STRUCTURED_KEY,
                                UNREACHABLE_BY_STRUCTURED_KEY, compare)
from report.reportability import _sgo_kabco_floor, tier_a


# ---------------------------------------------------------------------------
# Taxonomy and role
# ---------------------------------------------------------------------------
def test_schema_taxonomy_matches_acc_type_module():
    """The pydantic enum and the crosswalk must not drift apart."""
    assert [c.value for c in AccTypeCategory] == CATEGORIES


@pytest.mark.parametrize("code,category,role", [
    # The rear-end family reads backwards: D20 'Rear End-Stopped' is the vehicle
    # that drove into a stopped one (Going Straight 76% of the time), and D21
    # 'Stopped, Straight' is the one that was stopped (100%). See the note above
    # the D-family registrations in schema/acc_type.py.
    (20, "rear_end", "striking"),
    (21, "rear_end", "struck"),
    (29, "rear_end", "struck"),        # 'Decelerating, Going Straight'
    (86, "straight_paths_intersecting", "striking"),   # 'Striking from the Right'
    (87, "straight_paths_intersecting", "struck"),     # 'Struck on the Right'
    (92, "backing", "striking"),       # the backing vehicle
    (93, "backing", "struck"),
    (13, "pedestrian_animal", "not_applicable"),
    (33, "rear_end", "undetermined"),  # 'Specifics Unknown' keeps its family
])
def test_role_map_direction(code, category, role):
    assert crss_acc_type(code) == (category, role)


def test_role_map_covers_every_documented_code():
    """A code with no mapping would silently drop vehicles from the analysis."""
    documented = list(range(0, 17)) + list(range(20, 34)) + list(range(38, 43)) \
        + list(range(44, 56)) + list(range(58, 62)) + [63] + list(range(64, 94)) \
        + [98, 99]
    unmapped = [c for c in documented if crss_acc_type(c) is None]
    assert unmapped == [], f"unmapped ACC_TYPE codes: {unmapped}"


def test_verify_crss_names_catches_a_renumbering():
    """If NHTSA shifts the codes, the guard must fire rather than invert roles."""
    good = [(86, "L86-Intersecting Paths-Straight Paths-Striking from the Right")]
    assert verify_crss_names(good) == []
    shifted = [(86, "L87-Intersecting Paths-Straight Paths-Struck on the Right")]
    assert verify_crss_names(shifted), "a renumbered code must be reported"


def test_role_orientation_against_p_crash1():
    """Orientation needs an INDEPENDENT signal; the balance check cannot see it.

    Swapping both members of a pair leaves striking:struck at exactly 1.000, so
    the 1:1 invariant validates that the pairing survives, not that it points
    the right way. Only a variable the role map never reads -- here the observed
    pre-crash maneuver -- can establish direction.
    """
    from schema.acc_type import verify_role_orientation
    good = [(21, "Stopped in Roadway", 5000), (21, "Going Straight", 12),
            (20, "Going Straight", 4000), (20, "Decelerating in Road", 300),
            (92, "Backing Up (other than for Parking Position)", 900)]
    assert verify_role_orientation(good) == []

    # A release in which the rear-end pair is swapped must be reported. The
    # signal is the MANEUVER mismatch -- code 21 is supposed to be the stopped
    # vehicle, so seeing it going straight means either the codes moved or the
    # roles are inverted, and both warrant stopping the run.
    swapped = [(21, "Going Straight", 5000), (20, "Stopped in Roadway", 4000)]
    problems = verify_role_orientation(swapped)
    assert len(problems) == 2, f"expected both codes flagged, got {problems}"
    assert any("ACC_TYPE 21" in p for p in problems)
    assert any("ACC_TYPE 20" in p for p in problems)


def test_is_two_vehicle_excludes_partnerless_categories():
    for c in ("single_vehicle", "pedestrian_animal", "other_unknown"):
        assert not is_two_vehicle(c)
    for c in ("rear_end", "backing", "turn_across_path"):
        assert is_two_vehicle(c)


# ---------------------------------------------------------------------------
# The QIE pairing invariant
# ---------------------------------------------------------------------------
def _paired_crashes(n: int = 100) -> pd.DataFrame:
    """A synthetic census of rear-end crashes, one striking + one struck each."""
    rows = []
    for i in range(n):
        rows.append({"CASENUM": i, "acc_category": "rear_end",
                     "acc_role": "striking", "WEIGHT": 1.0,
                     "speed_limit": 35.0, "towed": True})
        rows.append({"CASENUM": i, "acc_category": "rear_end",
                     "acc_role": "struck", "WEIGHT": 1.0,
                     # The struck vehicle differs on BOTH a vehicle-level
                     # covariate and a vehicle-level outcome -- which is
                     # exactly how real rear-end crashes behave.
                     "speed_limit": 55.0 if i % 2 else 35.0,
                     "towed": False})
    return pd.DataFrame(rows)


def test_paired_census_has_unit_baseline():
    d = _paired_crashes()
    assert human_baseline_ratio(d)["rear_end"]["ratio"] == pytest.approx(1.0)


def test_vehicle_level_filter_breaks_the_pairing():
    """The bug this guards: a per-vehicle filter admits one party, not both."""
    d = _paired_crashes()
    veh = d[d["speed_limit"] <= 45]                 # vehicle-level restriction
    assert human_baseline_ratio(veh)["rear_end"]["ratio"] != pytest.approx(1.0)


def test_crash_level_filter_preserves_the_pairing():
    d = _paired_crashes()
    keep = d.assign(_ok=d["speed_limit"].le(45)).groupby("CASENUM")["_ok"].transform("all")
    crash = d[keep]
    assert human_baseline_ratio(crash)["rear_end"]["ratio"] == pytest.approx(1.0)


def test_vehicle_level_outcome_filter_also_breaks_pairing():
    """Tier A applied per vehicle, not per crash: towing differs by role."""
    d = _paired_crashes()
    assert human_baseline_ratio(d[d["towed"]])["rear_end"]["ratio"] != pytest.approx(1.0)
    crash = d[d.assign(_a=d["towed"]).groupby("CASENUM")["_a"].transform("any")]
    assert human_baseline_ratio(crash)["rear_end"]["ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Matched category spaces
# ---------------------------------------------------------------------------
def test_reachable_and_unreachable_partition_the_taxonomy():
    assert set(REACHABLE_BY_STRUCTURED_KEY) | set(UNREACHABLE_BY_STRUCTURED_KEY) \
        == set(CATEGORIES)
    assert not set(REACHABLE_BY_STRUCTURED_KEY) & set(UNREACHABLE_BY_STRUCTURED_KEY)


def test_structured_key_never_returns_an_unreachable_category():
    """If it did, the reachable/unreachable split would be a lie."""
    rng = np.random.default_rng(0)
    zones = ["front", "rear", "left", "right", "front_left", "rear_right"]
    for _ in range(400):
        row = {"crash_with": rng.choice(["passenger car", "suv", "animal",
                                         "other fixed object",
                                         "non-motorist: pedestrian"]),
               "SV Pre-Crash Movement": rng.choice(
                   ["proceeding straight", "stopped", "backing", "changing lanes"]),
               "CP Pre-Crash Movement": rng.choice(
                   ["proceeding straight", "stopped", "backing", "passing"])}
        for side in ("sv", "cp"):
            for z in zones:
                row[f"{side}_contact_{z}"] = bool(rng.random() < 0.3)
        k = sgo_distant_key(row)
        if k is not None:
            assert k[0] in REACHABLE_BY_STRUCTURED_KEY


def test_compare_restricts_human_to_the_same_levels():
    """The artifact this guards inflated every surviving ADS share."""
    ads = pd.DataFrame({"acc_category_struct": ["rear_end"] * 6 + ["backing"] * 4})
    human = pd.DataFrame({
        "acc_category": (["rear_end"] * 30 + ["backing"] * 10
                         + ["turn_across_path"] * 60),
        "_w": 1.0,
    })
    # Unrestricted: the human rear-end share is diluted by a category the ADS
    # side structurally cannot have, so the ratio is inflated.
    loose = compare(ads, human, "acc_category_struct",
                    levels=["rear_end", "backing"], n_boot_ads=20,
                    n_boot_human=20, restrict_human=False)
    tight = compare(ads, human, "acc_category_struct",
                    levels=["rear_end", "backing"], n_boot_ads=20,
                    n_boot_human=20, restrict_human=True)
    loose_rear = next(r for r in loose["rows"] if r["category"] == "rear_end")
    tight_rear = next(r for r in tight["rows"] if r["category"] == "rear_end")
    assert loose_rear["ratio"] > tight_rear["ratio"]
    assert tight_rear["human_share"] == pytest.approx(0.75)
    assert tight["human_share_of_all_categories_retained"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# The two SGO file generations
# ---------------------------------------------------------------------------
def test_party_flag_decodes_the_combined_post_amendment_value():
    v = "Yes Subject Vehicle, No Crash Partner"
    assert _party_flag(v, "subject vehicle") is True
    assert _party_flag(v, "crash partner") is False


def test_party_flag_unknown_is_not_no():
    """'Unknown Crash Partner' means the filer could not tell, not 'no'."""
    v = "No Subject Vehicle, Unknown Crash Partner"
    assert _party_flag(v, "subject vehicle") is False
    assert _party_flag(v, "crash partner") is None
    assert _party_flag("Not Applicable", "subject vehicle") is None


def test_derive_fields_harmonizes_tow_across_generations():
    df = pd.DataFrame([
        # pre-amendment: two separate Yes/No columns
        {"Report Version": 1, "filing_regime": "pre", "Incident Date": "Mar-2023",
         "Report Submission Date": "Apr-2023", "Driver / Operator Type": "",
         "Automation System Engaged?": "ADS", "Roadway Type": "Street",
         "SV Was Vehicle Towed?": "Yes", "CP Was Vehicle Towed?": "No",
         "SV Any Air Bags Deployed?": "No", "CP Any Air Bags Deployed?": "No",
         "Crash With": "Passenger Car"},
        # post-amendment: one combined-value column
        {"Report Version": 1, "filing_regime": "post", "Incident Date": "Aug-2025",
         "Report Submission Date": "Sep-2025", "Driver / Operator Type": "",
         "Automation System Engaged?": "ADS", "Engagement Status": "Verified Engaged",
         "Roadway Type": "Street",
         "Was Any Vehicle Towed?": "Yes Subject Vehicle, No Crash Partner",
         "Any Air Bags Deployed?": "No Subject Vehicle, No Crash Partner",
         "Crash With": "Passenger Car"},
    ])
    out = derive_fields(df)
    assert list(out["sv_towed"]) == [True, True]
    assert list(out["any_towed"]) == [True, True]
    assert list(out["any_airbag"]) == [False, False]
    assert list(out["driverless"]) == [True, True]
    assert list(out["ads_engaged"]) == [True, True]


def test_driverless_definitions_differ_by_operator_convention():
    """Blank-only 'driverless' selects Waymo, not driverless operation.

    Waymo leaves Driver / Operator Type blank; Cruise never does, filing
    `Remote (Commercial / Test)` for operations with nobody aboard. Measured on
    the 2021-24 window, blank-only gives 455 of 456 Waymo.
    """
    base = {"Report Version": 1, "filing_regime": "pre",
            "Incident Date": "Mar-2023", "Report Submission Date": "Apr-2023",
            "Automation System Engaged?": "ADS", "Roadway Type": "Street",
            "Crash With": "Passenger Car"}
    df = pd.DataFrame([
        dict(base, **{"Driver / Operator Type": ""}),                       # Waymo
        dict(base, **{"Driver / Operator Type": "Remote (Commercial / Test)"}),
        dict(base, **{"Driver / Operator Type": "In-Vehicle (Commercial / Test)"}),
        dict(base, **{"Driver / Operator Type":
                      "In-Vehicle and Remote (Commercial / Test)"}),
    ])
    strict = derive_fields(df, driverless_def="strict")
    assert list(strict["driverless_strict"]) == [True, False, False, False]
    assert list(strict["driverless"]) == list(strict["driverless_strict"])

    broad = derive_fields(df, driverless_def="no_onboard")
    # Remote joins driverless; anything with someone aboard stays supervised.
    assert list(broad["driverless_no_onboard"]) == [True, True, False, False]
    assert list(broad["driverless"]) == list(broad["driverless_no_onboard"])


def test_derive_fields_rejects_an_unclassified_operator_type():
    """A new SGO operator value must fail loudly, not become 'unknown'."""
    df = pd.DataFrame([{
        "Report Version": 1, "filing_regime": "post", "Incident Date": "Aug-2025",
        "Report Submission Date": "Sep-2025",
        "Driver / Operator Type": "Teleoperated Swarm",
        "Automation System Engaged?": "ADS", "Roadway Type": "Street",
        "Crash With": "Passenger Car"}])
    with pytest.raises(ValueError, match="unrecognised Driver / Operator Type"):
        derive_fields(df)


def test_june_2025_incidents_are_regime_ambiguous():
    """The day is redacted, so June 2025 cannot be cut at the 16th."""
    df = pd.DataFrame([{
        "Report Version": 1, "filing_regime": "post", "Incident Date": "Jun-2025",
        "Report Submission Date": "Jul-2025", "Driver / Operator Type": "",
        "Automation System Engaged?": "ADS", "Roadway Type": "Street",
        "Crash With": "Passenger Car"}])
    assert derive_fields(df)["incident_regime"].iloc[0] == "ambiguous"


# ---------------------------------------------------------------------------
# Tier A
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sev,expected", [
    ("Minor", True), ("Moderate W/ Hospitalization", True), ("Serious", True),
    ("Fatality", True), ("No Injuries Reported", False),
    ("Property Damage. No Injured Reported", False), ("Unknown", None), ("", None),
])
def test_sgo_kabco_floor(sev, expected):
    assert _sgo_kabco_floor(sev) is expected


def test_tier_a_is_a_disjunction_and_unknown_is_not_qualifying():
    d = pd.DataFrame([
        {"Highest Injury Severity Alleged": "No Injuries Reported",
         "any_towed": True, "any_airbag": False},      # tow alone
        {"Highest Injury Severity Alleged": "Minor",
         "any_towed": False, "any_airbag": False},     # injury alone
        {"Highest Injury Severity Alleged": "Unknown",
         "any_towed": None, "any_airbag": None},       # nothing known
        {"Highest Injury Severity Alleged": "No Injuries Reported",
         "any_towed": False, "any_airbag": False},     # qualifies on nothing
    ])
    assert list(tier_a(d)) == [True, True, False, False]


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------
def test_evidence_spans_are_required():
    """An optional audit field is one the model is free to skip -- and did."""
    base = {
        "acc_type_category": "rear_end", "striking_role": "struck",
        "sv_pre_crash_movement": "stopped_in_roadway",
        "cp_pre_crash_movement": "going_straight",
        "relation_to_junction": "intersection", "intersection_type": "four_way",
        "kabco_severity": "O", "tow_away_due_to_damage": "no",
        "airbag_deployed": "no", "damage_descriptor": "scratch_scuff",
        "sensor_only_damage": False, "posted_speed_limit_bin": "mph_30_35",
        "lighting": "daylight", "police_reportable": "no",
        "reportable_confidence": 0.8, "is_true_crash": "yes",
        "on_public_road": "yes", "ads_engaged_at_impact": "yes", "confidence": 0.7,
    }
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        CompositionExtraction.model_validate(base)          # both spans missing
    ok = dict(base, role_evidence="struck in the rear",
              reportable_evidence="no injuries were reported")
    assert CompositionExtraction.model_validate(ok).police_reportable.value == "no"


# ---------------------------------------------------------------------------
# Entropy balancing
# ---------------------------------------------------------------------------
def test_entropy_balance_hits_the_target_exactly():
    """The point of entropy balancing over a propensity model is exactness."""
    from human.balance import entropy_balance
    rng = np.random.default_rng(0)
    n = 2000
    x = (rng.random(n) < 0.2).astype(float)          # source mean ~0.2
    X = x[:, None]
    w0 = np.ones(n)
    w, info = entropy_balance(X, w0, np.array([0.7]))
    assert info["converged"]
    assert float((X * w[:, None]).sum()) == pytest.approx(0.7, abs=1e-6)
    assert info["max_imbalance"] < 1e-6


def test_entropy_balance_objective_and_gradient_agree():
    """A lambda-dependent stabilisation shift desynchronises the two.

    The symptom is subtle: L-BFGS-B reports failure or stops early, having
    moved the weights partway, and the result reads as "balancing barely
    changed anything" rather than as a bug.
    """
    from human.balance import entropy_balance
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    w0 = rng.random(300) + 0.1
    target = (X * (w0 / w0.sum())[:, None]).sum(0) + 0.25
    w, info = entropy_balance(X, w0, target)
    assert info["converged"]
    assert np.allclose((X * w[:, None]).sum(0), target, atol=1e-5)


def test_entropy_balance_preserves_base_weights():
    """It reweights the survey weight, it does not discard it.

    With a target equal to the base-weighted mean, the solution must be the
    base weights themselves -- nothing to correct.
    """
    from human.balance import entropy_balance
    rng = np.random.default_rng(2)
    X = (rng.random((500, 2)) < 0.4).astype(float)
    w0 = rng.random(500) + 0.5
    q = w0 / w0.sum()
    target = (X * q[:, None]).sum(0)
    w, info = entropy_balance(X, w0, target)
    assert np.allclose(w, q, atol=1e-6)


def test_effective_n_and_trim():
    from human.balance import effective_n, trim
    flat = np.ones(1000) / 1000
    assert effective_n(flat) == pytest.approx(1000)
    spiked = np.concatenate([np.full(999, 1e-6), [1.0]])
    spiked = spiked / spiked.sum()
    assert effective_n(spiked) < 5          # one row carries the estimate
    t = trim(spiked, 0.99)
    assert t.sum() == pytest.approx(1.0)
    assert effective_n(t) > effective_n(spiked)


def test_annotation_guide_matches_prompt():
    """The model and the human coders must be judged against ONE rubric.

    If the guide and the prompt drift apart, the human-vs-model agreement
    statistic stops measuring agreement and starts measuring the difference
    between two standards -- silently, and in a direction nobody can recover
    after the fact.
    """
    from extract.composition_prompts import REPORTABILITY_RUBRIC
    guide = (Path(__file__).resolve().parents[1]
             / "src" / "annotate" / "composition_annotation_guide.md").read_text()
    assert REPORTABILITY_RUBRIC in guide, (
        "the reportability rubric in composition_annotation_guide.md no longer "
        "matches extract/composition_prompts.REPORTABILITY_RUBRIC; regenerate "
        "the guide")
    # The census also codes crash type, role and the exclusion flags, so their
    # definitions are held to the same one-standard rule.
    from extract.composition_prompts import SYSTEM_PROMPT
    for start, end in [("CRASH TYPE (acc_type_category)", "STRIKING ROLE"),
                       ("STRIKING ROLE", "MODE AT IMPACT"),
                       ("EXCLUSION FLAGS", "REPORTABILITY.")]:
        i = SYSTEM_PROMPT.index(start)
        block = SYSTEM_PROMPT[i:SYSTEM_PROMPT.index(end, i)].rstrip()
        assert block in guide, f"guide section {start!r} no longer matches the prompt"


def test_revised_rubric_defaults_to_no_and_names_the_confounds():
    """Pin the two defects that produced a 99.6% reportable share.

    Both were failures of the rubric, not of the model: a permissive trigger
    list with no default, and no instruction that an AV tow or a bare
    'sustained damage' carries no severity information.
    """
    from extract.composition_prompts import REPORTABILITY_RUBRIC
    r = REPORTABILITY_RUBRIC.lower()
    assert 'start from "no"' in r, "the rubric must state an explicit default"
    # The AV-tow confound: 539 of 1,308 'yes' labels rested on it.
    assert "subject av being towed" in r or "av tow" in r
    assert "crash partner" in r, "only a partner tow should be a trigger"
    # The boilerplate confound: top evidence phrase in both yes and maybe.
    assert "undifferentiated" in r and "sustained damage" in r


def test_exclusion_flags_accept_unknown():
    """Regression: as booleans these lost a whole extraction when the model
    tried to abstain on `on_public_road`. Abstention is not error."""
    base = {
        "acc_type_category": "rear_end", "striking_role": "struck",
        "role_evidence": "rear contact", "sv_pre_crash_movement": "stopped_in_roadway",
        "cp_pre_crash_movement": "going_straight",
        "relation_to_junction": "unknown", "intersection_type": "unknown",
        "kabco_severity": "unknown", "tow_away_due_to_damage": "unknown",
        "airbag_deployed": "unknown", "damage_descriptor": "unknown",
        "sensor_only_damage": False, "posted_speed_limit_bin": "unknown",
        "lighting": "unknown", "police_reportable": "maybe",
        "reportable_evidence": "the narrative does not describe damage",
        "reportable_confidence": 0.4, "is_true_crash": "unknown",
        "on_public_road": "unknown", "ads_engaged_at_impact": "unknown",
        "confidence": 0.4,
    }
    o = CompositionExtraction.model_validate(base)
    assert o.on_public_road.value == "unknown"
    assert o.is_true_crash.value == "unknown"


# ---------------------------------------------------------------------------
# SWITRS reconstruction (the geographically matched comparator)
# ---------------------------------------------------------------------------
def test_crss_manner_handles_hyphenated_spellings():
    """CRSS writes `Front-to-Rear`; a space-separated match silently misses it.

    The first validation run scored rear-end recall at 0.000 and head-on at
    0.000 -- which reads as the reconstruction rule failing outright and was a
    string-matching bug. Validation caught it; this pins it.
    """
    from human.state_sf import _crss_manner
    assert _crss_manner("Front-to-Rear") == "rear_end"
    assert _crss_manner("Front-to-Front") == "head_on"
    assert _crss_manner("Sideswipe - Same Direction") == "sideswipe"
    assert _crss_manner("Angle") == "angle"
    assert _crss_manner("Not Reported") is None


def test_reconstruction_separates_the_angle_family():
    """`Broadside` pools three CRSS categories; the movement pair splits them."""
    from human.state_sf import reconstruct
    assert reconstruct("angle", ["straight", "straight"]) == \
        "straight_paths_intersecting"
    assert reconstruct("angle", ["left", "straight"]) == "turn_across_path"
    assert reconstruct("angle", ["right", "straight"]) == "turn_into_path"
    # Undetermined rather than guessed.
    assert reconstruct("angle", ["left", "right"]) is None
    assert reconstruct("angle", ["straight"]) is None


def test_reconstruction_orders_nonmotorist_and_backing_first():
    """A pedestrian crash is a pedestrian crash whatever the manner code says."""
    from human.state_sf import reconstruct
    assert reconstruct("angle", ["straight", "straight"],
                       has_nonmotorist=True) == "pedestrian_animal"
    assert reconstruct("angle", ["backing", "straight"]) == "backing"


def test_move_vocabulary_is_shared_across_both_sources():
    """The rule is only validatable on CRSS if both sources normalise the same."""
    from human.state_sf import norm_move
    assert norm_move("Proceeding Straight") == norm_move("Going Straight") == "straight"
    assert norm_move("Making Left Turn") == norm_move("Turning Left") == "left"
    assert norm_move("Backing") == norm_move(
        "Backing Up (other than for Parking Position)") == "backing"
    assert norm_move("not a real movement") is None


def test_occupancy_parses_both_generations():
    """SGO has no occupancy field; the seat-belt field answers it anyway.

    The two generations phrase the no-occupant value differently, and roughly
    half of driverless crashes are unoccupied -- so reading only one spelling
    would silently mark a large, non-random slice unknown.
    """
    base = {"Report Version": 1, "Incident Date": "Mar-2023",
            "Report Submission Date": "Apr-2023", "Driver / Operator Type": "",
            "Automation System Engaged?": "ADS", "Roadway Type": "Street",
            "Crash With": "Passenger Car", "filing_regime": "pre"}
    df = pd.DataFrame([
        dict(base, **{"SV Were All Passengers Belted?": "No Passengers in Vehicle"}),
        dict(base, **{"SV Were All Passengers Belted?": "Yes"}),
        dict(base, **{"SV Were All Passengers Belted?": "No, see Narrative"}),
        dict(base, **{"Were All Passengers Belted?":
                      "Subject Vehicle - No Passenger In Vehicle"}),
        dict(base, **{"Were All Passengers Belted?": "Subject Vehicle - All Belted"}),
        dict(base, **{"Were All Passengers Belted?": "Unknown"}),
    ])
    out = derive_fields(df)
    assert list(out["sv_occupied"]) == [False, True, True, False, True, None]


# ---------------------------------------------------------------------------
# Human census: form, blinding, scoring, gold labels
# ---------------------------------------------------------------------------
def test_census_allowed_values_match_schema():
    """The coding sheet, the census builder and the model schema must offer the
    same labels, or human and model are coding into different spaces."""
    from annotate.make_composition_census import ALLOWED
    from schema import composition_schema as cs
    assert ALLOWED["acc_type_category"] == [c.value for c in cs.AccTypeCategory]
    assert ALLOWED["striking_role"] == [c.value for c in cs.StrikingRole]
    assert ALLOWED["police_reportable"] == [c.value for c in cs.Reportability]
    for f in ("is_true_crash", "on_public_road", "ads_engaged_at_impact"):
        assert ALLOWED[f] == [c.value for c in cs.YesNoUnknown]
    assert ALLOWED["damage_descriptor"] == [c.value for c in cs.DamageDescriptor]
    assert ALLOWED["relation_to_junction"] == [c.value for c in cs.RelationToJunction]
    assert ALLOWED["intersection_type"] == [c.value for c in cs.IntersectionType]
    # every coded field has a column on the sheet
    from annotate.census_csv import CENSUS_FIELD_COLS
    assert set(ALLOWED) - {"verdict"} <= set(CENSUS_FIELD_COLS)


def test_coder_sheet_round_trip_survives_a_spreadsheet(tmp_path):
    """Excel writes TRUE for true and may pad cells; n/a cells are not fields."""
    import csv
    from annotate.census_csv import check, read_rows, write_census
    from annotate.make_composition_census import ALLOWED
    rows = [{"report_id": "r1", "batch": 1, "entity": "Waymo LLC", "narrative": "line one\nline two, with comma",
             "fields": ["acc_type_category", "police_reportable", "coder_notes"]},
            {"report_id": "r2", "batch": 3, "entity": "Zoox", "narrative": "n",
             "fields": ["acc_type_category", "sensor_only_damage", "coder_notes"]}]
    path = tmp_path / "sheet.csv"
    write_census(rows, str(path))
    with open(path, newline="", encoding="utf-8-sig") as f:
        sheet = list(csv.DictReader(f))
    assert sheet[0]["sensor_only_damage"] == "n/a" and sheet[0]["police_reportable"] == ""
    assert sheet[0]["narrative"] == "line one\nline two, with comma"
    sheet[0].update(acc_type_category=" Turn_Across_Path ", police_reportable="yes")
    sheet[1].update(acc_type_category="rear_end", sensor_only_damage="TRUE",
                    coder_notes="Mixed Case note")
    with open(path, "w", newline="", encoding="cp1252") as f:   # a legacy save
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader(); w.writerows(sheet)
    got = read_rows(str(path))
    assert got["r1"]["acc_type_category"] == "turn_across_path"
    assert got["r1"]["fields"] == ["acc_type_category", "police_reportable", "coder_notes"]
    assert got["r2"]["sensor_only_damage"] == "true" and got["r2"]["batch"] == 3
    assert got["r2"]["coder_notes"] == "Mixed Case note"      # free text kept as typed
    rep = check(str(path), ALLOWED)
    assert rep["invalid"] == [] and rep["progress"]["1"] == {"complete": 1, "rows": 1}
    sheet[0]["acc_type_category"] = "left turn"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader(); w.writerows(sheet)
    assert [x["column"] for x in check(str(path), ALLOWED)["invalid"]] == ["acc_type_category"]


def test_adjudication_through_a_csv_disagreement_sheet(tmp_path):
    import csv
    import json as _json
    from annotate.adjudicate import FIELD_SETS, diff, merge
    from annotate.census_csv import write_census
    F = ["acc_type_category", "police_reportable", "coder_notes"]
    base = [{"report_id": "c0", "batch": 0, "entity": "W", "narrative": "cal", "fields": F},
            {"report_id": "r1", "batch": 1, "entity": "W", "narrative": "n1", "fields": F}]
    for name in "AB":
        write_census(base, str(tmp_path / f"{name}.csv"))
    def fill(name, vals):
        p = tmp_path / f"{name}.csv"
        with open(p, newline="", encoding="utf-8-sig") as f:
            sh = list(csv.DictReader(f))
        for r in sh:
            r.update(vals[r["report_id"]])
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sh[0])); w.writeheader(); w.writerows(sh)
    fill("A", {"c0": {"acc_type_category": "backing"}, "r1": {"acc_type_category": "rear_end", "police_reportable": "no"}})
    fill("B", {"c0": {"acc_type_category": "head_on"}, "r1": {"acc_type_category": "turn_across_path", "police_reportable": "no"}})
    dis = tmp_path / "dis.csv"
    fs = FIELD_SETS["composition"]
    diff(str(tmp_path / "A.csv"), str(tmp_path / "B.csv"), str(dis), fs)
    with open(dis, newline="", encoding="utf-8-sig") as f:
        sh = list(csv.DictReader(f))
    assert [(r["report_id"], r["field"]) for r in sh] == [("r1", "acc_type_category")]
    with pytest.raises(SystemExit):                       # unresolved rows block the merge
        merge(str(tmp_path / "A.csv"), str(tmp_path / "B.csv"), str(dis), str(tmp_path / "g.jsonl"), fs)
    sh[0]["resolved"] = "Turn_Across_Path"
    with open(dis, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sh[0])); w.writeheader(); w.writerows(sh)
    merge(str(tmp_path / "A.csv"), str(tmp_path / "B.csv"), str(dis), str(tmp_path / "g.jsonl"), fs)
    gold = [_json.loads(l) for l in (tmp_path / "g.jsonl").read_text().splitlines()]
    assert [g["report_id"] for g in gold] == ["r1"]           # calibration dropped
    assert gold[0]["acc_type_category"] == "turn_across_path" and gold[0]["police_reportable"] == "no"


def _census_fixture(tmp_path):
    """Tiny corpus: 6 window driverless, 1 supervised, 1 out-of-window."""
    rows = []
    def add(rid, year, dl, sv="Stopped", cp="Proceeding Straight", partner="passenger car"):
        rows.append({"Report ID": rid, "incident_key": f"k{rid}", "Reporting Entity": "Waymo LLC",
                     "Narrative": "the vehicle was stopped when another car made contact with it " + rid,
                     "incident_year": year, "driverless_no_onboard": dl, "driverless_strict": dl,
                     "crash_with": partner, "SV Pre-Crash Movement": sv,
                     "CP Pre-Crash Movement": cp, "filing_regime": "pre"})
    add("w1", 2022, True, sv="Making Left Turn")
    add("w2", 2023, True, cp="Other, see Narrative")
    for i in range(3, 7):
        add(f"w{i}", 2024, True)
    add("sup", 2022, False)
    add("late", 2025, True)
    add("late2", 2026, True)      # in the stratified sample: never calibration
    inc = tmp_path / "inc.parquet"
    pd.DataFrame(rows).to_parquet(inc, index=False)
    ext = tmp_path / "ext.jsonl"
    ext.write_text("\n".join(__import__("json").dumps(
        {"report_id": r, "ok": True, "model": "m",
         "extraction": {"acc_type_category": c}})
        for r, c in [("w3", "single_vehicle"), ("w4", "rear_end"),
                     ("w5", "rear_end"), ("w6", "rear_end")]) + "\n")
    prior = tmp_path / "prior.jsonl"
    prior.write_text('{"report_id": "sup"}\n{"report_id": "w4"}\n{"report_id": "late2"}\n')
    dedup = tmp_path / "dedup.jsonl"
    dedup.write_text(__import__("json").dumps(
        {"decision": "merge", "report_i": "w1", "report_j": "w2", "narrative_sim": 0.9,
         "contact_jaccard": 1.0, "narrative_i": "a", "narrative_j": "b", "verdict": ""}) + "\n")
    return inc, ext, prior, dedup


def test_census_partitions_the_window_and_is_blind(tmp_path):
    from annotate.make_composition_census import build
    inc, ext, prior, dedup = _census_fixture(tmp_path)
    rows, dedup_rows, meta = build(str(inc), str(ext), str(prior), str(dedup),
                                   n_calibration=5)
    b = {r["report_id"]: r["batch"] for r in rows}
    frame = {f"w{i}" for i in range(1, 7)}
    # batches 1 and 2 are exactly the window frame; selection only sets order
    assert {i for i, x in b.items() if x in (1, 2)} == frame
    # left turn, vague movement and a model single-vehicle code go first
    assert {i for i, x in b.items() if x == 1} == {"w1", "w2", "w3"}
    assert b["late2"] == 3
    assert b["late"] == 0 and b["sup"] == 3 and "w4" not in {i for i, x in b.items() if x == 3}
    assert len(rows) == len(b), "every incident appears in exactly one batch"
    allowed = {"report_id", "incident_key", "entity", "batch", "fields", "narrative"}
    for r in rows:
        assert set(r) - set(r["fields"]) <= allowed, "coder rows must carry no model or SGO field"
        assert all(r[f] == "" for f in r["fields"])
    assert "decision" not in dedup_rows[0] and "narrative_sim" not in dedup_rows[0]
    assert meta["corpus_sample_weight"]["sup"] > 0


def test_agreement_is_perfect_against_itself():
    from annotate.score_composition_agreement import interannotator
    a = {str(i): {"acc_type_category": c, "police_reportable": p}
         for i, (c, p) in enumerate([("rear_end", "no"), ("turn_across_path", "yes"),
                                     ("single_vehicle", "no"), ("rear_end", "maybe")])}
    out = {r["field"]: r for r in interannotator(a, a, ["acc_type_category", "police_reportable"])}
    for f in out.values():
        assert f["pct_agree"] == 1.0 and f["kappa"] == pytest.approx(1.0)
        assert f["ac1"] == pytest.approx(1.0)


def test_weighted_per_class_precision_recall():
    from annotate.score_composition_agreement import per_class
    yt = ["tap", "tap", "re", "re"]
    yp = ["tap", "re", "re", "re"]
    res = per_class(yt, yp, [1, 1, 1, 1], n_boot=10)
    assert res["tap"]["precision"] == 1.0 and res["tap"]["recall"] == 0.5
    # Up-weighting the missed crash lowers recall accordingly.
    res = per_class(yt, yp, [1, 3, 1, 1], n_boot=10)
    assert res["tap"]["recall"] == pytest.approx(0.25)


def test_window_weights_scale_an_early_stopped_batch():
    from annotate.score_composition_agreement import window_weights
    meta = {"batch_order": {"1": ["a", "b"], "2": ["c", "d", "e", "f"]}}
    w = window_weights(meta, {"a", "b", "c", "d"})
    assert w == {"a": 1.0, "b": 1.0, "c": 2.0, "d": 2.0}


def test_adjudicate_composition_ignores_blanks_and_calibration(tmp_path):
    import json as _json
    from annotate.adjudicate import FIELD_SETS, diff
    def write(p, rows):
        p.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    base = {"fields": ["acc_type_category", "police_reportable", "coder_notes"]}
    A = [{"report_id": "c", "batch": 0, **base, "acc_type_category": "rear_end", "police_reportable": "no", "coder_notes": "x"},
         {"report_id": "r", "batch": 1, **base, "acc_type_category": "rear_end", "police_reportable": "", "coder_notes": "a"}]
    B = [{"report_id": "c", "batch": 0, **base, "acc_type_category": "backing", "police_reportable": "yes", "coder_notes": "y"},
         {"report_id": "r", "batch": 1, **base, "acc_type_category": "turn_across_path", "police_reportable": "", "coder_notes": "b"}]
    write(tmp_path / "a.jsonl", A); write(tmp_path / "b.jsonl", B)
    out = tmp_path / "dis.jsonl"
    summ = diff(str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl"), str(out),
                FIELD_SETS["composition"])
    dis = [_json.loads(l) for l in out.read_text().splitlines()]
    assert [(d["report_id"], d["field"]) for d in dis] == [("r", "acc_type_category")]
    assert summ["police_reportable"]["total"] == 0      # both blank: not a disagreement


def test_attach_gold_overrides_and_requires_full_coverage(tmp_path):
    import json as _json
    from report.reportability import GOLD_FIELDS, attach_gold
    meta = tmp_path / "meta.json"
    meta.write_text(_json.dumps({"batch_order": {"1": ["a"], "2": ["b"]}}))
    d = pd.DataFrame({"Report ID": ["a", "b", "z"],
                      "acc_type_category": ["rear_end"] * 3,
                      "police_reportable": ["no"] * 3})
    g = {f: "unknown" for f in GOLD_FIELDS}
    gold = tmp_path / "gold.jsonl"
    gold.write_text(_json.dumps({"report_id": "a", **g, "acc_type_category": "turn_across_path"}) + "\n")
    with pytest.raises(SystemExit):
        attach_gold(d, str(gold), str(meta))
    gold.write_text("\n".join(_json.dumps({"report_id": r, **g, "acc_type_category": "turn_across_path",
                                           "police_reportable": "yes"}) for r in "ab") + "\n")
    out = attach_gold(d, str(gold), str(meta))
    assert list(out["Report ID"]) == ["a", "b"]           # frame only
    assert set(out["acc_type_category"]) == {"turn_across_path"}
    assert set(out["police_reportable"]) == {"yes"}
