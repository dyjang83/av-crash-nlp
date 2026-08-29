# OL 316 / SGO Narrative Coding Rubric (v1)

This rubric governs both human annotation and the LLM extractor. The model system
prompt (`src/extract/prompts.py`) is a condensed copy of these rules. If you change
one, change both.

Each narrative is coded independently by **two** annotators. Disagreements are
resolved by adjudication after agreement (Cohen's κ) is computed on the
*independent* codings. Code only what the text supports; never infer from outside
knowledge.

---

## Fields

### collision_type
The geometry of the *primary* impact.
- `rear_end` — subject strikes, or is struck in, the rear.
- `cross_path` — vehicles on crossing/turning paths at a junction.
- `sideswipe` — same-direction lateral contact.
- `head_on` — opposing-direction frontal contact.
- `backing` — contact while a vehicle is reversing.
- `vru` — **any** pedestrian, cyclist, scooter, or other vulnerable road user is
  struck, regardless of geometry. This label takes priority.
- `single_vehicle` — no second road user (object strike, curb, road departure).
- `other` / `unknown`.

### subject_pre_crash_maneuver
What the **subject AV** was doing immediately before impact: `stopped`,
`proceeding_straight`, `decelerating`, `accelerating`, `turning_left`,
`turning_right`, `lane_change`, `backing`, `parked`, `other`, `unknown`.

### other_party_present
`true` if a second road user is involved; `false` for single-vehicle events.

### contributory_party  *(narrative-described, NOT legal fault)*
Who the narrative attributes the precipitating unsafe action to.
- `av` — text describes the **AV** performing the precipitating unsafe action.
- `other_party` — text describes the **other** road user doing so (e.g. "the
  other vehicle ran the red light", "cut across its path", "rear-ended the AV").
- `shared` — both contribute.
- `ambiguous` — text does not permit a determination. **Use this freely.** Do not
  guess fault to avoid `ambiguous`.
- `not_applicable` — single-vehicle with no second party action.

> Reminder: this field is for the extraction task only. It is never a target in
> the downstream severity lift test.

### engagement_state
- `ads_engaged` — SAE L3–5 automated driving system active at the incident.
- `adas_engaged` — SAE L2 driver assistance active.
- `manual` — a human was driving.
- `disengaged_prior` — automation disengaged within seconds before impact.
- `unknown`.

### lighting
- `daylight` - full daylight
- `dark_lighted` - dark, but street lights present
- `dark_unlighted` - dark, no street lighting
- `dawn_dusk` - dawn or dusk (transitional light)
- `unknown`

### weather
- `clear`
- `rain`
- `fog`
- `snow`
- `other`
- `unknown`

### road_class
- `surface_street` - ordinary city/local street
- `arterial` - a major through-road (multi-land, higher speed, signialized corridor) but not a freeway
- `highway_freeway` - limited-access highway or freeway
- `parking_lot_private` - private lot, driveway area, or other non-public roadway
- `other`
- `unknown`

### locality
- `intersection` - the collision occurred at a junction where two roads cross
- `intersection_related` - the collision happened near a junction and was casually tied to it, but the impact point itself wasn't squarely in the intersection box
- `segment` (mid-block) - a stretch of road between intersections, with no junction involved
- `driveway` - collision involved a vehicle entering/exiting a driveway
- `other`
- `unknown`.

### narrative_injury_severity
Injuries **as described in this narrative**: `none`, `minor`, `moderate`,
`serious`, `fatal`, `unknown`. Do **not** consult the structured SGO severity
field when coding this; they must stay independent.

### av_moving
`true` if the AV was in motion at impact, `false` if stopped/parked.

### contributory_evidence
Copy the single verbatim clause that justifies the `contributory_party` label.

---

## Adjudication
1. Both annotators code independently into `data/gold/annotator_{A,B}.jsonl`.
2. Run `src/annotate/score_agreement.py` to compute per-field κ.
3. Adjudicate disagreements into `data/gold/gold.jsonl` (the evaluation set).
4. Report per-field κ in Table 2 of the paper as the label-quality ceiling.
