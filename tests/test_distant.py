"""Tests for the distant-supervision key.

The failure mode this file guards against is silent: a mapping bug does not
crash, it produces a plausible-looking accuracy number computed against the
wrong labels. Each test therefore pins a specific way the key could be wrong
rather than just exercising the code path.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fetch.parse_ol316 import _ol316_injury_flag, load_sgo_csv  # noqa: E402
from schema.distant_map import (  # noqa: E402
    REDUCERS, is_missing, reduce_av_moving, reduce_collision_single_vehicle,
    reduce_collision_vru, reduce_engagement, reduce_lighting, reduce_locality,
    reduce_maneuver, reduce_narrative_severity, reduce_other_party,
    reduce_road_class, reduce_weather,
)


# --------------------------------------------------------------------------
# Missingness
# --------------------------------------------------------------------------

def test_nan_is_missing():
    """float NaN is truthy; `value or None` does not filter it. It must not
    reach a reducer as if it were a label."""
    assert is_missing(float("nan"))
    assert is_missing(np.nan)
    assert is_missing("")
    assert is_missing("   ")
    assert is_missing("Unknown")
    assert is_missing("Other, see Narrative")
    assert not is_missing("Daylight")
    assert not is_missing(0)          # a speed of 0 is a value, not a gap


# --------------------------------------------------------------------------
# Coarsening in both directions
# --------------------------------------------------------------------------

def test_lighting_dark_unknown_collapses_both_sides():
    """The key cannot resolve lit vs unlit dark. A model answering either must
    not be scored wrong for a distinction the key cannot express."""
    row = {"struct_lighting": "Dark - Unknown Lighting"}
    assert reduce_lighting(row, "dark_lighted") == ("dark", "dark")
    assert reduce_lighting(row, "dark_unlighted") == ("dark", "dark")
    # Daylight against a dark key is still a genuine error.
    assert reduce_lighting(row, "daylight") == ("dark", "non_dark")


def test_lighting_exact_codes_are_not_coarsened():
    row = {"struct_lighting": "Dark - Lighted"}
    assert reduce_lighting(row, "dark_lighted") == ("dark_lighted", "dark_lighted")
    assert reduce_lighting(row, "dark_unlighted") == ("dark_lighted", "dark_unlighted")


def test_maneuver_longitudinal_control_folds_into_straight():
    """SGO's 'Proceeding Straight' subsumes the schema's decelerating and
    accelerating; scoring them as errors would measure the mapping."""
    row = {"struct_movement": "Proceeding Straight"}
    for pred in ("proceeding_straight", "decelerating", "accelerating"):
        assert reduce_maneuver(row, pred) == ("in_motion_straight",
                                              "in_motion_straight")
    assert reduce_maneuver(row, "backing") == ("in_motion_straight", "backing")


def test_maneuver_drops_codes_without_a_clean_counterpart():
    """'Merging' has no schema value. Collapsing it to `other` would score a
    model answering `lane_change` as wrong; the row must drop instead."""
    for code in ("Merging", "Passing", "Entering Traffic",
                 "Lane / Road Departure", "Crossing into Opposing Lane"):
        assert reduce_maneuver({"struct_movement": code}, "lane_change") is None


def test_road_class_projects_arterial_onto_surface():
    row = {"struct_roadway": "Street"}
    assert reduce_road_class(row, "arterial") == ("surface", "surface")
    assert reduce_road_class(row, "surface_street") == ("surface", "surface")
    assert reduce_road_class(row, "highway_freeway") == ("surface", "highway_freeway")


def test_road_class_drops_intersection_and_traffic_circle():
    """'Intersection' is a locality code sharing a column with road class; it
    carries no road-class information."""
    assert reduce_road_class({"struct_roadway": "Intersection"}, "surface_street") is None
    assert reduce_road_class({"struct_roadway": "Traffic Circle"}, "surface_street") is None


def test_locality_uses_only_defensible_evidence():
    """'Street' does not exclude an intersection -- the reporter picked one code
    from a single-select. Only the positive and the hard negatives are usable."""
    assert reduce_locality({"struct_roadway": "Intersection"}, "intersection") \
        == ("intersection", "intersection")
    assert reduce_locality({"struct_roadway": "Parking Lot"}, "segment") \
        == ("not_intersection", "not_intersection")
    assert reduce_locality({"struct_roadway": "Street"}, "segment") is None


def test_locality_treats_intersection_related_as_intersection():
    assert reduce_locality({"struct_roadway": "Intersection"},
                           "intersection_related") == ("intersection", "intersection")


def test_engagement_no_collapses_manual_and_disengaged_prior():
    row = {"struct_engaged": "No", "struct_report_type": "ads"}
    assert reduce_engagement(row, "manual") == ("not_engaged", "not_engaged")
    assert reduce_engagement(row, "disengaged_prior") == ("not_engaged", "not_engaged")
    assert reduce_engagement(row, "ads_engaged") == ("not_engaged", "engaged")


def test_engagement_yes_requires_report_type():
    """Yes/No alone cannot say whether an engaged system is ADS or ADAS."""
    assert reduce_engagement({"struct_engaged": "Yes"}, "ads_engaged") is None
    assert reduce_engagement({"struct_engaged": "Yes", "struct_report_type": "ads"},
                             "ads_engaged") == ("ads_engaged", "ads_engaged")
    assert reduce_engagement({"struct_engaged": "Yes", "struct_report_type": "adas"},
                             "adas_engaged") == ("adas_engaged", "adas_engaged")


# --------------------------------------------------------------------------
# Weather: the indicator-column bug
# --------------------------------------------------------------------------

def test_weather_reads_indicator_set_not_a_single_column():
    assert reduce_weather({"struct_weather_flags": {"rain": "Y"}}, "rain") \
        == ("rain", "rain")
    assert reduce_weather({"struct_weather_flags": {"snow": "Y"}}, "clear") \
        == ("snow", "clear")


def test_weather_precipitation_dominates_multiselect():
    """A 'Rain + Cloudy' filing is a rain filing."""
    row = {"struct_weather_flags": {"rain": "Y", "cloudy": "Y"}}
    assert reduce_weather(row, "rain") == ("rain", "rain")


def test_weather_cloudy_alone_drops():
    """The schema has no cloudy category. Mapping it to clear or other would
    manufacture a convention and then measure agreement with it."""
    assert reduce_weather({"struct_weather_flags": {"cloudy": "Y"}}, "clear") is None


def test_weather_no_flags_drops():
    assert reduce_weather({"struct_weather_flags": {}}, "clear") is None
    assert reduce_weather({}, "clear") is None


# --------------------------------------------------------------------------
# collision_type: the two binary decisions must stay separate
# --------------------------------------------------------------------------

def test_collision_type_binaries_are_independent():
    """Folding both one-vs-rest problems into one label vector makes the classes
    separable by which branch produced the row, inflating kappa toward 1."""
    vru_row = {"struct_crashwith": "Non-Motorist: Cyclist"}
    obj_row = {"struct_crashwith": "Pole / Tree"}
    car_row = {"struct_crashwith": "Passenger Car"}

    assert reduce_collision_vru(vru_row, "vru") == ("vru", "vru")
    assert reduce_collision_vru(car_row, "rear_end") == ("not_vru", "not_vru")
    # A fixed-object crash says nothing about VRU involvement.
    assert reduce_collision_vru(obj_row, "single_vehicle") is None

    assert reduce_collision_single_vehicle(obj_row, "single_vehicle") \
        == ("single_vehicle", "single_vehicle")
    assert reduce_collision_single_vehicle(car_row, "rear_end") \
        == ("not_single_vehicle", "not_single_vehicle")

    labels = {reduce_collision_vru(vru_row, "vru")[0],
              reduce_collision_vru(car_row, "rear_end")[0]}
    assert labels == {"vru", "not_vru"}


def test_other_party_present_drops_animal():
    """Whether a struck animal is a second road user is a convention the schema
    does not settle."""
    assert reduce_other_party({"struct_crashwith": "Animal"}, True) is None
    assert reduce_other_party({"struct_crashwith": "SUV"}, True) == ("True", "True")
    assert reduce_other_party({"struct_crashwith": "Other Fixed Object"}, True) \
        == ("False", "True")


# --------------------------------------------------------------------------
# av_moving
# --------------------------------------------------------------------------

def test_av_moving_uses_zero_nonzero_only():
    assert reduce_av_moving({"struct_speed": "0"}, False) == ("False", "False")
    assert reduce_av_moving({"struct_speed": "12"}, True) == ("True", "True")
    assert reduce_av_moving({"struct_speed": "0"}, True) == ("False", "True")


def test_av_moving_rejects_unparseable_and_negative_speed():
    assert reduce_av_moving({"struct_speed": ""}, True) is None
    assert reduce_av_moving({"struct_speed": "unknown"}, True) is None
    assert reduce_av_moving({"struct_speed": float("nan")}, True) is None
    assert reduce_av_moving({"struct_speed": "-1"}, True) is None


# --------------------------------------------------------------------------
# narrative severity concordance (diagnostic): SGO full grade vs. OL 316's
# coarser injured/deceased-only signal
# --------------------------------------------------------------------------

def test_severity_ol316_some_injury_collapses_model_grade():
    """OL 316 can't resolve minor/moderate/serious -- any of those model
    predictions must count as agreement with the coarse "some injury" key,
    the same way reduce_lighting collapses dark_lighted/dark_unlighted
    against "Dark - Unknown Lighting"."""
    row = {"struct_severity": "some injury"}
    for grade in ("minor", "moderate", "serious"):
        assert reduce_narrative_severity(row, grade) == ("some_injury", "some_injury")
    assert reduce_narrative_severity(row, "none") == ("some_injury", "none")
    assert reduce_narrative_severity(row, "fatal") == ("some_injury", "fatal")


def test_severity_ol316_none_and_fatal_are_exact():
    assert reduce_narrative_severity({"struct_severity": "none"}, "none") \
        == ("none", "none")
    assert reduce_narrative_severity({"struct_severity": "fatal"}, "minor") \
        == ("fatal", "minor")


def test_severity_sgo_full_grade_is_not_coarsened():
    """SGO rows keep the full 5-way scale; only OL 316's "some injury" key
    collapses model grades."""
    row = {"struct_severity": "Serious W Hospitalization"}
    assert reduce_narrative_severity(row, "serious") == ("serious", "serious")
    assert reduce_narrative_severity(row, "moderate") == ("serious", "moderate")


def test_severity_missing_key_drops():
    assert reduce_narrative_severity({}, "minor") is None
    assert reduce_narrative_severity({"struct_severity": None}, "minor") is None


# --------------------------------------------------------------------------
# OL 316 Section 4 injury/deceased checkbox -> struct_severity
# --------------------------------------------------------------------------

def test_ol316_injury_flag_deceased_beats_injured():
    states = {"Injured": "/Yes", "Deceased": "/Yes"}
    assert _ol316_injury_flag(states) == "fatal"


def test_ol316_injury_flag_injured_without_deceased():
    assert _ol316_injury_flag({"Injured": "/Yes"}) == "some injury"
    assert _ol316_injury_flag({"Injured_2": "/Yes"}) == "some injury"


def test_ol316_injury_flag_property_only_is_none():
    assert _ol316_injury_flag({"Proper ty": "/Yes"}) == "none"
    assert _ol316_injury_flag({"Proper ty": "/Yes", "Proper ty_2": "/Yes"}) == "none"


def test_ol316_injury_flag_blank_section_is_missing_not_none():
    """No tick at all is indistinguishable from the section being left
    blank; it must not be read as an affirmative 'no injuries'."""
    assert _ol316_injury_flag({}) is None
    assert _ol316_injury_flag({"Driver": "/Yes"}) is None  # role tick, no outcome tick


# --------------------------------------------------------------------------
# Every reducer must be total: no crash on missing input
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(REDUCERS))
def test_reducers_handle_empty_and_nan_rows(name):
    field, fn = REDUCERS[name]
    for row in ({}, {f"struct_{k}": float("nan") for k in
                     ("lighting", "roadway", "movement", "crashwith", "speed",
                      "engaged", "report_type", "severity")}):
        assert fn(row, "unknown") is None or isinstance(fn(row, "unknown"), tuple)
    assert fn({"struct_lighting": "Daylight"}, None) is None


# --------------------------------------------------------------------------
# Parser: cross-file column divergence
# --------------------------------------------------------------------------

def _concat_frame():
    """Reproduce the ADS/ADAS concat that made pandas fill 1232 rows with NaN."""
    narrative = ("The vehicle was proceeding straight through the intersection "
                 "when another vehicle entered its path and contact occurred. ") * 2
    ads = pd.DataFrame([{
        "Report ID": "A1", "Narrative": narrative, "Make": "waymo",
        "Highest Injury Severity Alleged": "Minor",
        "Lighting": "Daylight", "Roadway Type": "Street",
        "SV Precrash Movement": "Proceeding Straight",
        "Crash With": "Passenger Car", "SV Precrash Speed (MPH)": "12",
        "ADS/ADAS - Automation Engaged?": "Yes",
        "Weather - Clear": "Y",
        "_source_file": "SGO-2021-01_Incident_Reports_ADS.csv"}])
    adas = pd.DataFrame([{
        "Report ID": "B2", "Narrative": narrative, "Make": "tesla",
        "Highest Injury Severity Alleged": "No Injuries Reported",
        "Light Condition": "Dark - Lighted", "Roadway": "Highway / Freeway",
        "SV Pre-Crash Movement": "Stopped", "Crash Partner": "Pickup Truck",
        "SV Precrash Speed": "0", "Automation Engaged": "No",
        "Weather - Rain": "Y",
        "_source_file": "SGO-2021-01_Incident_Reports_ADAS.csv"}])
    return pd.concat([ads, adas], ignore_index=True)


def test_divergent_column_names_are_coalesced_not_dropped():
    """Resolving each field to a single column name silently NaN-ed out every
    row that came from the other file."""
    recs = {r["report_id"]: r for r in load_sgo_csv(_concat_frame())}
    assert len(recs) == 2
    for rid in ("A1", "B2"):
        r = recs[rid]
        for f in ("struct_lighting", "struct_roadway", "struct_movement",
                  "struct_crashwith", "struct_speed", "struct_engaged"):
            assert r[f] is not None, f"{f} lost for {rid}"
            assert not (isinstance(r[f], float) and math.isnan(r[f]))


def test_report_type_recovered_from_source_file():
    recs = {r["report_id"]: r for r in load_sgo_csv(_concat_frame())}
    assert recs["A1"]["struct_report_type"] == "ads"
    assert recs["B2"]["struct_report_type"] == "adas"


def test_weather_flags_replace_the_clear_indicator_field():
    recs = {r["report_id"]: r for r in load_sgo_csv(_concat_frame())}
    assert recs["A1"]["struct_weather_flags"] == {"clear": "Y"}
    assert recs["B2"]["struct_weather_flags"] == {"rain": "Y"}
    # The old single-column read produced 'Y'/' ' as the weather VALUE.
    assert "struct_weather" not in recs["A1"]


def test_end_to_end_reduction_on_parsed_records():
    recs = {r["report_id"]: r for r in load_sgo_csv(_concat_frame())}
    assert reduce_lighting(recs["A1"], "daylight") == ("daylight", "daylight")
    assert reduce_weather(recs["B2"], "rain") == ("rain", "rain")
    assert reduce_av_moving(recs["B2"], False) == ("False", "False")
    assert reduce_engagement(recs["A1"], "ads_engaged") == ("ads_engaged",
                                                            "ads_engaged")


# --------------------------------------------------------------------------
# OL 316 checkbox decoding
#
# Field names follow "<GROUP> <LETTER> <VEHICLE>". The letters index the
# printed option list and the trailing digit is the vehicle number. Verified
# against all 868 filings via `parse_ol316 --audit-checkboxes`.
# --------------------------------------------------------------------------

from fetch.parse_ol316 import _ol316_structured  # noqa: E402


def test_ol316_letters_decode_to_schema_values():
    out = _ol316_structured({"WEATHER A 1": "/Yes", "LIGHTING A 1": "/Yes",
                             "MOVEMENT  B 1": "/Yes", "TYPE E 1": "/Yes"})
    assert out["struct_weather_flags"] == {"clear": "Y"}
    assert out["struct_lighting"] == "daylight"
    assert out["struct_movement"] == "proceeding straight"
    assert out["struct_collision_type"] == "single_vehicle"


def test_ol316_movement_reads_subject_vehicle_only():
    """Vehicle 1 is the AV; vehicle 2 is the other party. Reading the '2' boxes
    would code the other party's maneuver as the subject vehicle's."""
    out = _ol316_structured({"MOVEMENT A 1": "/Yes", "MOVEMENT  H 2": "/Yes"})
    assert out["struct_movement"] == "stopped"       # A = stopped, veh 1
    out2 = _ol316_structured({"MOVEMENT  B 2": "/Yes"})
    assert "struct_movement" not in out2             # other party only


def test_ol316_environmental_groups_ignore_vehicle_duplication():
    """Weather and lighting are ticked redundantly for both vehicles with the
    same value; that must not read as an ambiguous double-tick."""
    out = _ol316_structured({"LIGHTING A 1": "/Yes", "LIGHTING A 2": "/Yes",
                             "WEATHER A 1": "/Yes", "WEATHER A 2": "/Yes"})
    assert out["struct_lighting"] == "daylight"
    assert out["struct_weather_flags"] == {"clear": "Y"}


def test_ol316_conflicting_ticks_yield_nothing():
    out = _ol316_structured({"LIGHTING A 1": "/Yes", "LIGHTING C 1": "/Yes"})
    assert "struct_lighting" not in out


def test_ol316_unmapped_letters_are_reported_not_forced():
    """Movement codes with no schema counterpart (e.g. R = other) must drop and
    be surfaced, not collapsed into `other`."""
    out = _ol316_structured({"MOVEMENT  R 1": "/Yes", "SOME OTHER BOX": "/Yes"})
    assert "struct_movement" not in out
    assert "SOME OTHER BOX" in out["_unmapped_checkboxes"]


def test_ol316_off_states_are_not_ticks():
    from fetch.parse_ol316 import _CHECKBOX_OFF
    assert "/off" in _CHECKBOX_OFF


def test_engagement_status_third_amendment_values():
    from schema.distant_map import reduce_engagement as re_
    assert re_({"struct_engaged": "Verified Engaged",
                "struct_report_type": "ads"}, "ads_engaged") == ("ads_engaged",
                                                                 "ads_engaged")
    assert re_({"struct_engaged": "Verified Not Engaged"}, "manual") \
        == ("not_engaged", "not_engaged")
    # An unverified allegation is not a label.
    assert re_({"struct_engaged": "Alleged Engaged",
                "struct_report_type": "ads"}, "ads_engaged") is None


def test_new_weather_categories_are_handled():
    from schema.distant_map import reduce_weather as rw
    assert rw({"struct_weather_flags": {"fog/smoke/haze": "Y"}}, "fog") == ("fog", "fog")
    assert rw({"struct_weather_flags": {"dust storm": "Y"}}, "other") == ("other", "other")
    # Location, not weather.
    assert rw({"struct_weather_flags": {"structure-indoor": "Y"}}, "clear") is None
    assert rw({"struct_weather_flags": {"partly cloudy": "Y"}}, "clear") is None
