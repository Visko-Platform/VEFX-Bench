"""
Prompt templates for VEFX-Reward video editing quality evaluation.
"""

EDITREWARD_V2_SPECIAL = """You are an expert evaluator assessing the quality of AI-generated video edits. You will be provided with two videos:
- **Video 1**: The Original Video (before editing)
- **Video 2**: The Edited Video (after editing)

The editing instruction is:
"{text_prompt}"

Your task is to evaluate the Edited Video across THREE independent dimensions. Each dimension is scored on a 1–4 integer scale. **Scores across dimensions are independent** — a failure in one dimension must NOT affect scores in another.

---

## Dimension 1: Instructional Following (IF)
**Core question:** Does the edited video accurately reflect the semantic requirements of the editing instruction?

Evaluation criteria:
- Object replacement: If the instruction says "replace apple with orange," did the model actually generate an orange (not a lemon or tomato)?
- Action/attribute changes: If the instruction involves motion or attribute changes (e.g., "make it night"), was this correctly executed?
- Completeness: Were ALL parts of the instruction addressed, not just partial execution?

Scoring rubric:
- **4 (Perfect):** The edit precisely and completely executes all instructions. Object categories, attributes (color, shape), actions, and styles all match the instruction with no ambiguity.
- **3 (High):** The main instruction was executed, but minor details deviate. E.g., instruction asks for "red sports car" but a "red truck" was generated — the main concept "red car" is correct.
- **2 (Low):** The main instruction was partially executed but with significant deviations, or completely irrelevant modifications were made.
- **1 (Failed):** The edit has no relation to the instruction. E.g., instruction asks for "night scene" but the video remains daytime, or no change occurred at all.

**Important notes:**
- If the edit instruction asks for a camera perspective change (e.g., "shift to high angle") and the video shows no actual perspective change, score 1.
- If the instruction asks for adding/increasing objects and no new objects appear, score 1.
- A video that looks identical to the original (no edit happened) always scores 1 for IF.

Instructional Following score (integer 1-4): <|IF_reward|>

---

## Dimension 2: Render Quality (RQ)
**Core question:** What is the visual and temporal quality of the edited video?

Evaluation criteria:
- Naturalness and clarity: Are all parts of the video natural and sharp? Any blurriness, noise, or artifacts?
- Physical plausibility: Does object motion obey physics? Any flickering, jittering, objects disappearing/morphing unexpectedly?
- Temporal consistency: Is the video smooth frame-to-frame? Any sudden jumps, abrupt texture/color changes between frames?

Scoring rubric:
- **4 (Excellent):** Video clarity is very high with no visible defects, or only extremely minor artifacts detectable on very close inspection. Object motion fully obeys physics, smooth and natural. Visual quality is on par with or better than the original.
- **3 (Medium):** Some quality degradation exists (e.g., slight blurring, localized flickering), but all objects remain clearly identifiable. The video's overall structure is intact despite imperfections.
- **2 (Poor):** Significant quality degradation with obvious artifacts, distortion, or frame-to-frame inconsistency. Some object outlines deform, motion appears unnatural, affecting viewing experience.
- **1 (Unusable):** Video quality completely breaks down. Objects are severely deformed or unrecognizable, serious physics violations (e.g., person walking through walls, objects shattering spontaneously), heavy noise or complete blur.

**Important notes:**
- A sudden scene transition mid-video (e.g., white background abruptly becoming a construction site) counts as a physics/consistency violation — score ≤ 3.
- If the edit did NOT happen (original is preserved), RQ can still be high if the video itself looks fine — evaluate the video's visual quality independently.
- Evaluate temporal artifacts carefully: a single frame of flickering is minor (score 3), persistent warping or morphing is severe (score 1-2).

Render Quality score (integer 1-4): <|RQ_reward|>

---

## Dimension 3: Edit Exclusivity (EE)
**Core question:** Did the model ONLY perform the specified edit, without making unintended changes to other parts of the video?

Evaluation criteria:
- Over-editing: When editing a foreground object, did the background, lighting, or other unrelated objects change?
- Scene consistency: Are pixels, textures, and structures in non-edited regions preserved?
- Camera trajectory: Was the original camera movement preserved? (Changing camera motion when not instructed is over-editing.)
- Identity preservation: Do unedited people maintain their facial features, expressions, and body movements?

Scoring rubric:
- **4 (Perfect):** Strict exclusivity maintained. Only the target region specified by the instruction changed. All other regions (background, unrelated objects) remain identical to the original. Tiny pixel-level differences invisible to the eye are acceptable.
- **3 (Medium):** Visible over-editing occurred. Non-target areas show noticeable changes, but overall scene layout and unrelated object consistency are still preserved. E.g., replaced a cup on a table but the table style also changed, or a background window disappeared.
- **2 (Poor):** The overall scene or multiple unrelated objects changed significantly.
- **1 (Complete failure):** No exclusivity at all. The entire video looks like a completely new video. The surrounding scene changed drastically, or more than three unrelated objects underwent serious alterations.

**Important notes:**
- Camera trajectory changes (when not instructed) are over-editing — if the original video had camera motion and the edited video is static (or vice versa), penalize EE.
- For style transfer instructions (e.g., "turn into cyberpunk style"), it is expected that the entire visual style changes — this is NOT over-editing. But if text content or distinct object identities are destroyed during style transfer, that IS over-editing (score ≤ 3).
- If the edit failed (IF=1) but the rest of the video also changed, EE should still be scored low.

Edit Exclusivity score (integer 1-4): <|EE_reward|>
"""


def build_prompt(instruction: str) -> str:
    """Build the evaluation prompt from an editing instruction.

    Args:
        instruction: The video editing instruction text.

    Returns:
        The formatted prompt string with special reward tokens.
    """
    return EDITREWARD_V2_SPECIAL.format(text_prompt=instruction)
