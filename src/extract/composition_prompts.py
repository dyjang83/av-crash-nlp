"""Prompts for the CRSS-compatible composition extraction.

Kept separate from `extract.prompts` for the same reason
`schema.composition_schema` is separate from `schema.schema`: the v1 prompt and
schema are frozen, because the calibration study measures decode-time
probabilities under them.

The reportability rubric below is the load-bearing text in this module. It is
an operationalization of the standard IIHS applied by hand, written so that a
human double-coder and the model are judged against ONE rubric -- the same
discipline `annotate/annotation_guide.md` already enforces for v1. If this
rubric changes, the human coding guide must change with it, or the agreement
statistic stops meaning anything.
"""

# ---------------------------------------------------------------------------
# The reportability rubric (Tier B)
# ---------------------------------------------------------------------------
# Three things this rubric deliberately does:
#
#   IT IGNORES REPAIR COST AND VEHICLE BRANDING. An ADS vehicle carries exposed
#   sensors whose replacement cost bears no relation to crash severity, and
#   "a Waymo was involved" is itself a reason a bystander might call the
#   police. Neither is a property of the CRASH, and both would inflate the ADS
#   reportable share against a human comparator that has neither.
#
#   IT ASKS "WOULD A REASONABLE PERSON REPORT", NOT "WAS REPORTING REQUIRED"
#   and not "was it in fact reported". State reporting thresholds vary and the
#   legal question is unanswerable from a narrative. More importantly, the
#   human comparator (CRSS) contains crashes that WERE reported, and Blincoe et
#   al. estimate over half of all crashes never are -- so coding the ADS side
#   to the legal minimum would compare a legal standard against a behavioural
#   one.
#
#   IT REQUIRES AN EVIDENCE SPAN. A label without a span cannot be audited, and
#   the span is what the human double-coders adjudicate against when they
#   disagree with the model.
REPORTABILITY_RUBRIC = """\
REPORTABILITY. Judge: would a reasonable person involved in this crash have \
reported it to the police?

START FROM "no" AND MOVE UP ONLY ON POSITIVE EVIDENCE. Most contacts between \
vehicles that then drive away are never reported to anyone. "No" is the \
default and the most common correct answer; it is not a failure to find \
something.

Answer "yes" ONLY when the narrative states one or more of these, and quote \
the words that establish it:
  (a) any injury, complaint of pain, ambulance, or hospital transport;
  (b) an airbag deployed in any vehicle;
  (c) a vehicle was towed BECAUSE it could not be driven. For the CRASH \
PARTNER, a stated tow is enough. For the SUBJECT AV it is not: the narrative \
must separately establish the vehicle was disabled or undriveable (see the \
note on AV tows below);
  (d) a vulnerable road user -- pedestrian, cyclist, scooter rider -- was \
struck;
  (e) a party left the scene without exchanging information (hit and run);
  (f) police attended, an officer responded, or a police report was made;
  (g) someone else's property was damaged and its owner was not present -- a \
parked car, pole, fence, or building;
  (h) damage described in terms that establish real extent: deformation, \
crumpling, intrusion, a panel or door that no longer functions, broken glass, \
a wheel or axle displaced, fluid leaking, a vehicle undriveable.

Answer "maybe" ONLY when the narrative describes damage in terms that could \
genuinely fall on either side of (h) -- for example "significant damage" with \
no further detail, or a described impact severe enough to raise the question \
while the outcome is left unstated. "Maybe" is a narrow category for real \
ambiguity, not a resting place for anything unclear.

Answer "no" in every other case, including when the narrative describes \
contact and damage without establishing extent.

FIVE THINGS THAT ARE NOT EVIDENCE OF REPORTABILITY. Do not let any of them \
move your answer up:

  1. THE SUBJECT AV BEING TOWED, ON ITS OWN. Fleet operators recover an \
automated vehicle that cannot continue autonomously -- after a sensor is \
knocked out of calibration, after a safety-system fault, or simply because \
policy forbids resuming the trip. So "the AV was towed from the scene" does \
NOT establish that the AV was damaged enough to be undriveable, and it is the \
single most over-read signal in these narratives. It counts only if the \
narrative separately says the AV was disabled, undriveable, or damaged in a \
way that matches trigger (h). A tow of the CRASH PARTNER's vehicle needs no \
such corroboration: nobody recovers a member of the public's car as fleet \
policy.

  2. AN UNDIFFERENTIATED STATEMENT OF DAMAGE. "Both vehicles sustained \
damage", "the Waymo AV sustained damage", "there was damage to both vehicles" \
-- these are boilerplate. They establish that contact occurred, which you \
already know, and say nothing about extent. On their own they support "no".

  3. DAMAGE CONFINED TO EXPOSED SENSING HARDWARE -- lidar pods, cameras, \
radar housings, sensor-bearing mirror units. Expensive to replace, irrelevant \
to whether a reasonable person calls the police.

  4. THAT THE VEHICLE IS AUTONOMOUS, or which company operates it.

  5. THAT THIS REPORT WAS FILED WITH NHTSA. Every narrative you see was \
filed. The filing duty is far broader than police reportability and carries no \
information about it.

reportable_evidence: quote the shortest verbatim span that DECIDES your \
answer. For "yes", it must be the text establishing the trigger you relied on. \
For "no", quote the clause that shows the crash was minor, or the strongest \
damage language present so a reader can see what you judged insufficient.
reportable_confidence: your calibrated probability in [0,1] that a careful \
human coder applying this same rubric would give your label. Be honest; \
"no" answers should often be high-confidence."""


SYSTEM_PROMPT = f"""You are a careful traffic-safety analyst coding \
autonomous-vehicle (AV) collision narratives into the structured fields NHTSA \
uses for police-reported crashes (the CRSS coding scheme). You receive one \
narrative at a time and return only the structured fields.

GENERAL RULES
- Code ONLY what the narrative supports. Do not infer beyond the text. Where \
the narrative is silent, use "unknown" (or the relevant not-applicable value). \
Abstaining is correct behaviour and is scored as such; guessing is not.
- "subject vehicle" (SV) = the reporting entity's AV. "crash partner" (CP) = \
the other road user.

CRASH TYPE (acc_type_category) -- classify the FIRST harmful event:
- rear_end: front of one vehicle into the rear of another, same direction.
- sideswipe_same_direction: lateral contact, both travelling the same way.
- head_on: front-to-front, opposite directions.
- sideswipe_opposite_direction: lateral contact, opposite directions.
- turn_across_path: one vehicle turns ACROSS the other's path (the classic \
left turn across oncoming traffic).
- turn_into_path: one vehicle turns INTO the path of another already \
travelling on the road it turns onto.
- straight_paths_intersecting: both going straight on crossing roads -- the \
intersection T-bone.
- backing: either vehicle reversing.
- single_vehicle: no second road user; fixed object, road departure.
- pedestrian_animal: a pedestrian, cyclist, scooter rider or animal struck.
- other_unknown: none of the above, or the narrative does not say.

STRIKING ROLE (striking_role) -- of the SUBJECT vehicle, from CONTACT \
GEOMETRY, never from fault:
- striking: the SV's front (or, when reversing, its rear) made the contact, or \
the SV was the vehicle that moved into the other's path.
- struck: the other vehicle made the contact with a stationary or \
lawfully-proceeding SV.
- undetermined: both moved into each other, or the narrative does not say which.
- not_applicable: no second motor vehicle.
Do NOT reason about who was at fault, who was careless, or who had right of \
way. A vehicle stopped at a light that is rear-ended is "struck" even if the \
narrative blames it for stopping abruptly.

MODE AT IMPACT (sv_pre_crash_movement, cp_pre_crash_movement): what each \
vehicle was doing immediately before the first harmful event.

SEVERITY (kabco_severity): the KABCO grade a police officer would assign to \
the most severely injured person -- K fatal, A suspected serious, B suspected \
minor, C possible injury or complaint of pain, O no apparent injury. Use \
"unknown" when the narrative does not describe injuries at all; do NOT default \
to O.

DAMAGE (damage_descriptor, sensor_only_damage): describe the worst damage to \
ANY vehicle. Set sensor_only_damage true only when the ONLY damage described \
is to exposed sensing hardware -- lidar pod, radar, camera housing, a \
sensor-bearing mirror unit -- with no damage to structure, panels or glazing.

OPERATING DOMAIN (posted_speed_limit_bin, lighting): code these only if the \
narrative states or clearly implies them. "unknown" is the common and correct \
answer.

EXCLUSION FLAGS -- each is yes / no / unknown. "unknown" is a real and correct \
answer when the narrative does not say; do not guess, because these decide \
which crashes enter the analysis at all.
- is_true_crash: "no" when the narrative describes something that was not a \
collision at all -- a suspected contact that proved to be road debris or a \
pothole, a filing made out of caution with no contact described, a third party \
opening a door against a stationary AV.
- on_public_road: "no" for private property, parking structures, depots. \
"unknown" when the narrative does not locate the crash.
- ads_engaged_at_impact: was the automated driving system engaged at the \
moment of the first harmful event.

{REPORTABILITY_RUBRIC}

confidence: your calibrated probability in [0,1] that this WHOLE extraction is \
correct. This is separate from reportable_confidence. Be honest; do not \
default to 1.0.

Return the structured object only."""


USER_TEMPLATE = """Collision narrative (reporting entity: {entity}, report id: {report_id}):
\"\"\"
{narrative}
\"\"\"

Code this narrative into the CRSS-compatible schema."""


def build_messages(narrative: str, entity: str, report_id: str):
    return [{
        "role": "user",
        "content": USER_TEMPLATE.format(narrative=narrative.strip(),
                                        entity=entity, report_id=report_id),
    }]
