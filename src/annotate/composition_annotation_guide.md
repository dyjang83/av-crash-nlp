# Composition coding guide (reportability and damage)

Two coders work independently on the spreadsheets `data/gold/census_coder_A.csv`
and `census_coder_B.csv`, built by `annotate.make_composition_census` (see
"Census coding" below). They include the stratified sample drawn earlier by
`annotate.make_composition_sample` as batch 3, so the older
`composition_coder_{A,B}.jsonl` files are not coded separately. Each sheet
carries the narrative and blank fields; **no model prediction is shown**,
because a coder who can see the model is not independent of it and the
agreement statistic would be meaningless.

Six fields have no structured counterpart anywhere in SGO and are the reason
this sample exists: `police_reportable`, `damage_descriptor`,
`sensor_only_damage`, `is_true_crash`, `relation_to_junction`,
`intersection_type`. Of these, **`police_reportable` is the one every Tier B
result in the paper rests on**, and the human-vs-model agreement on it is the
headline validity number.

## Do not code to a target

An earlier version of the rubric produced a 99.6% reportable share against
IIHS's published 21.6% on the same frame. That was a defect in the rubric, not
a calibration knob. Apply the standard below as written and record what you
see; the comparison against IIHS is a *check*, and it stops being one the
moment either coder or model is steered toward their number.

## The rubric

This text is reproduced verbatim from
`src/extract/composition_prompts.REPORTABILITY_RUBRIC`, which is what the model
is given. `tests/test_composition.py::test_annotation_guide_matches_prompt`
fails if the two drift apart -- if you change one, change the other.

---

REPORTABILITY. Judge: would a reasonable person involved in this crash have reported it to the police?

START FROM "no" AND MOVE UP ONLY ON POSITIVE EVIDENCE. Most contacts between vehicles that then drive away are never reported to anyone. "No" is the default and the most common correct answer; it is not a failure to find something.

Answer "yes" ONLY when the narrative states one or more of these, and quote the words that establish it:
  (a) any injury, complaint of pain, ambulance, or hospital transport;
  (b) an airbag deployed in any vehicle;
  (c) a vehicle was towed BECAUSE it could not be driven. For the CRASH PARTNER, a stated tow is enough. For the SUBJECT AV it is not: the narrative must separately establish the vehicle was disabled or undriveable (see the note on AV tows below);
  (d) a vulnerable road user -- pedestrian, cyclist, scooter rider -- was struck;
  (e) a party left the scene without exchanging information (hit and run);
  (f) police attended, an officer responded, or a police report was made;
  (g) someone else's property was damaged and its owner was not present -- a parked car, pole, fence, or building;
  (h) damage described in terms that establish real extent: deformation, crumpling, intrusion, a panel or door that no longer functions, broken glass, a wheel or axle displaced, fluid leaking, a vehicle undriveable.

Answer "maybe" ONLY when the narrative describes damage in terms that could genuinely fall on either side of (h) -- for example "significant damage" with no further detail, or a described impact severe enough to raise the question while the outcome is left unstated. "Maybe" is a narrow category for real ambiguity, not a resting place for anything unclear.

Answer "no" in every other case, including when the narrative describes contact and damage without establishing extent.

FIVE THINGS THAT ARE NOT EVIDENCE OF REPORTABILITY. Do not let any of them move your answer up:

  1. THE SUBJECT AV BEING TOWED, ON ITS OWN. Fleet operators recover an automated vehicle that cannot continue autonomously -- after a sensor is knocked out of calibration, after a safety-system fault, or simply because policy forbids resuming the trip. So "the AV was towed from the scene" does NOT establish that the AV was damaged enough to be undriveable, and it is the single most over-read signal in these narratives. It counts only if the narrative separately says the AV was disabled, undriveable, or damaged in a way that matches trigger (h). A tow of the CRASH PARTNER's vehicle needs no such corroboration: nobody recovers a member of the public's car as fleet policy.

  2. AN UNDIFFERENTIATED STATEMENT OF DAMAGE. "Both vehicles sustained damage", "the Waymo AV sustained damage", "there was damage to both vehicles" -- these are boilerplate. They establish that contact occurred, which you already know, and say nothing about extent. On their own they support "no".

  3. DAMAGE CONFINED TO EXPOSED SENSING HARDWARE -- lidar pods, cameras, radar housings, sensor-bearing mirror units. Expensive to replace, irrelevant to whether a reasonable person calls the police.

  4. THAT THE VEHICLE IS AUTONOMOUS, or which company operates it.

  5. THAT THIS REPORT WAS FILED WITH NHTSA. Every narrative you see was filed. The filing duty is far broader than police reportability and carries no information about it.

reportable_evidence: quote the shortest verbatim span that DECIDES your answer. For "yes", it must be the text establishing the trigger you relied on. For "no", quote the clause that shows the crash was minor, or the strongest damage language present so a reader can see what you judged insufficient.
reportable_confidence: your calibrated probability in [0,1] that a careful human coder applying this same rubric would give your label. Be honest; "no" answers should often be high-confidence.

---

## The other coded fields

- **damage_descriptor** -- worst damage to ANY vehicle:
  `none`, `scratch_scuff`, `panel_dent`, `deformation`, `intrusion`,
  `sensor_only`, `unknown`. Use `unknown` freely: most narratives do not
  establish extent, and guessing here corrupts the damage comparison.
- **sensor_only_damage** -- true only when the ONLY damage described is to
  exposed sensing hardware, with nothing to structure, panels or glazing.
- **is_true_crash** -- `no` when the filing does not describe a collision at
  all (a reported contact that proved to be road debris or a pothole, a filing
  made out of caution with no contact, a third party opening a door against a
  stationary AV). `unknown` when the narrative does not say. Only an
  affirmative `no` excludes a record.
- **relation_to_junction** / **intersection_type** -- code from the narrative
  only. `unknown` is common and correct.
- **acc_type_category** / **striking_role** -- these DO have a partial
  structured key, and you are coding them so the unkeyed junction crashes can
  be checked too. The definitions are in the next section, verbatim from the
  model prompt.

## Crash type, striking role and exclusion flags

Reproduced verbatim from `src/extract/composition_prompts.SYSTEM_PROMPT`;
`tests/test_composition.py::test_annotation_guide_matches_prompt` fails if they
drift apart.

---

CRASH TYPE (acc_type_category) -- classify the FIRST harmful event:
- rear_end: front of one vehicle into the rear of another, same direction.
- sideswipe_same_direction: lateral contact, both travelling the same way.
- head_on: front-to-front, opposite directions.
- sideswipe_opposite_direction: lateral contact, opposite directions.
- turn_across_path: one vehicle turns ACROSS the other's path (the classic left turn across oncoming traffic).
- turn_into_path: one vehicle turns INTO the path of another already travelling on the road it turns onto.
- straight_paths_intersecting: both going straight on crossing roads -- the intersection T-bone.
- backing: either vehicle reversing.
- single_vehicle: no second road user; fixed object, road departure.
- pedestrian_animal: a pedestrian, cyclist, scooter rider or animal struck.
- other_unknown: none of the above, or the narrative does not say.

STRIKING ROLE (striking_role) -- of the SUBJECT vehicle, from CONTACT GEOMETRY, never from fault:
- striking: the SV's front (or, when reversing, its rear) made the contact, or the SV was the vehicle that moved into the other's path.
- struck: the other vehicle made the contact with a stationary or lawfully-proceeding SV.
- undetermined: both moved into each other, or the narrative does not say which.
- not_applicable: no second motor vehicle.
Do NOT reason about who was at fault, who was careless, or who had right of way. A vehicle stopped at a light that is rear-ended is "struck" even if the narrative blames it for stopping abruptly.

EXCLUSION FLAGS -- each is yes / no / unknown. "unknown" is a real and correct answer when the narrative does not say; do not guess, because these decide which crashes enter the analysis at all.
- is_true_crash: "no" when the narrative describes something that was not a collision at all -- a suspected contact that proved to be road debris or a pothole, a filing made out of caution with no contact described, a third party opening a door against a stationary AV.
- on_public_road: "no" for private property, parking structures, depots. "unknown" when the narrative does not locate the crash.
- ads_engaged_at_impact: was the automated driving system engaged at the moment of the first harmful event.

---

### Clarifications (not new rules)

These explain how the definitions above apply to cases that come up often.
They must not add a rule the model was not given. If you find yourself needing
one, code your best reading of the definition and explain in `coder_notes`,
so adjudication can see where the standard is unclear.

- **Which turn category.** Look at where the two vehicles came from.
  Opposite directions, one turning left across the other's lane:
  `turn_across_path`. Perpendicular approaches, one turning onto the road the
  other is travelling on: `turn_into_path`. Perpendicular approaches with both
  going straight: `straight_paths_intersecting`. If the narrative does not say
  which way the vehicles were heading, use `other_unknown` rather than
  guessing.
- **A turn that ends in lateral contact** is classified by the turn if the turn
  created the conflict: classify the FIRST harmful event.
- **`single_vehicle` means no second road user at all.** Contact with a parked
  vehicle, occupied or not, is a two-vehicle crash. Contact with debris, a
  pole, a gate or a curb is single-vehicle.
- **`striking_role` for a stationary AV** is `struck`, whatever the crash type.

## Census coding (window) and the dedup audit

Each coder gets two spreadsheets, built by `annotate.make_composition_census`:
`census_coder_X.csv` (one row per crash) and `dedup_coder_X.csv` (one row per
pair of filings). `coding_codes.csv` lists what may go in each column.

- Open the sheet in Excel, Numbers or Google Sheets. Read `narrative`, then
  type a value from `coding_codes.csv` into each **blank** cell of that row,
  spelled exactly as listed (case does not matter).
- A cell holding **`n/a`** is a field that row does not code. Leave it as it is.
- `reportable_evidence` and `coder_notes` are free text and optional.
- Do not sort, delete or add rows or columns; `report_id` is how the sheets are
  matched up.
- **Save as CSV** ("CSV UTF-8" in Excel) under the same name, and keep a copy.
- After each session, check the sheet for typos and see your progress:

      python -m annotate.census_csv check data/gold/census_coder_A.csv

  Any cell listed there holds a value that is not allowed; fix it before
  continuing.

Work in batch order (the `batch` column):

0. **Calibration** (20 rows, outside the analysis window). Code these
   together, discuss every difference, and settle how the guide applies. They
   are excluded from every statistic, and the guide is frozen afterwards.
1. **Candidates**, 2. **Remainder**, 3. **Corpus sample** -- code
   independently, and do not compare notes until both coders have finished.
   Within a batch the order is random and the same for both coders, so if
   coding has to stop early, stop both coders at the same row.

**Dedup audit** (`dedup_coder_X.csv`, 59 pairs): each row shows two filings
side by side (`_i` and `_j` columns). `verdict` is `same` (one crash, filed
twice), `different`, or `unclear`. Use the times, place, crash partner and
narratives; the pipeline's own decision is deliberately hidden.

## Adjudication

Code independently first. Then record kappa, AC1 and the majority share per
field before resolving disagreements -- the pre-adjudication numbers are the
ones that mean something:

    python -m annotate.score_composition_agreement \
        --a data/gold/census_coder_A.csv --b data/gold/census_coder_B.csv

Then list the disagreements as a spreadsheet and resolve them together:

    python -m annotate.adjudicate diff --fields composition \
        --ann-a data/gold/census_coder_A.csv --ann-b data/gold/census_coder_B.csv \
        --disagreements data/gold/census_disagreements.csv
    # fill the `resolved` column of census_disagreements.csv, save as CSV
    python -m annotate.adjudicate merge --fields composition \
        --ann-a data/gold/census_coder_A.csv --ann-b data/gold/census_coder_B.csv \
        --disagreements data/gold/census_disagreements.csv \
        --out data/gold/composition_gold.jsonl

The dedup audit works the same way with `--fields dedup`, the `dedup_coder_X.csv`
sheets and `--out data/gold/dedup_gold.jsonl`.

The corpus-sample rows are stratified: `census_meta.json` holds each row's
`corpus_sample_weight`, and a corpus-level estimate must use it. The
calibration batch never enters a statistic.
