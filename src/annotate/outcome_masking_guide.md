# Outcome-statement coding guide

This guide covers two short coding tasks. Their results validate the masker that removes outcome information from SGO narratives before the severity lift test (`src/leakage/`). Two coders complete both tasks **independently**. Do not discuss cases until both files are finished; disagreements are adjudicated afterwards.

You will **not** be shown the structured injury severity for any record. Do not look it up.

**Do Task R before Task S.** Task R measures what can be inferred from masked text, and it is only valid if you have not first read the unmasked versions of those narratives. The two tasks use disjoint records, but the order still matters because it keeps Task S's criteria from being on your mind during Task R.

---

## Task R — Recoverability (`outcome_recover_<coder>.csv`, about 100 rows)

Each row is a narrative *after* masking. Read it and fill `injured` with one of:

| value | meaning |
|---|---|
| `yes` | the text leads you to believe someone was injured or killed |
| `no` | the text leads you to believe nobody was injured |
| `cannot_tell` | the text does not let you judge |

Go with your honest impression and do not try to reason statistically. For example, "a pedestrian was struck at 30 mph" may make you think an injury is likely; if so, answer `yes`. Crash context like that is **meant** to remain in the text, and this task measures how much it still reveals. If a row contains a leftover explicit outcome statement that the mask missed, write it in `note`.

---

## Task S — Sentence coding (`outcome_sent_<coder>.csv`, about 300 narratives, one row per sentence)

For each sentence, set `outcome` to `1` if it contains an outcome statement as defined below, otherwise `0`. If `1`, set `category` to the **first** matching code in the order O3, O2, O1, O5, O4.

| code | covers | examples → 1 |
|---|---|---|
| **O1 injury status** | whether anyone was injured, hurt, in pain, or had symptoms — **including absence statements** | "No injuries were reported." · "The passenger complained of neck pain." · "Both drivers were uninjured." |
| **O2 medical response** | EMS, ambulance, transport to or from care, hospital, treatment, medical evaluation, declining care | "The cyclist was transported to a hospital." · "The driver declined medical attention." |
| **O3 fatality** | death, fatal, killed | "The motorcyclist was pronounced dead at the scene." |
| **O4 post-crash response proxies** | police, fire or first-responder attendance; claims, claimants, allegations, attorneys, citations | "Police responded." · "Claimant alleges the system failed to brake." |
| **O5 reportability statements** | why, or under which provision, the crash is reported; mentions of the injury-severity field | "Waymo is reporting this crash under Request No. 1 of Standing General Order 2021-01 because …" |

**Code `0` for crash context, even when it correlates with severity:** speed, maneuver, what struck what, point of impact, a pedestrian or cyclist being involved or struck, vehicle damage, towing, airbag deployment, weather, lighting, road layout, and redaction markers (`[XXX]`, `[REDACTED …]`).

Edge rules:
- A sentence that mixes context and outcome is `1`. For example: "The AV was struck at 20 mph and the driver was taken to hospital."
- Implied outcomes count: "the occupant sought care the next day" is O2.
- A sentence of pure bookkeeping ("This report was updated to add information") is `0`, unless it names the injury-severity field or a reporting provision (O5).
- If the sentence boundaries look wrong, code the row as given and write `split` in `note`.

---

## What happens with your codes

- **Task S** becomes the reference for the masker. Its main reported metric is sentence-level **recall**: the share of human-coded outcome sentences the masker removes. Precision is reported too.
- **Task R** is scored against the structured SGO severity, which you never see. We report whether masked text still lets a reader recover injury status.
- Inter-coder agreement is reported for both tasks (Cohen's κ, Gwet's AC1).
