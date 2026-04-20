"""
VEFX-Reward: A reward model for video editing quality assessment.

Evaluates video edits on three dimensions (1–4 scale):
- IF (Instructional Following)
- RQ (Render Quality)
- EE (Edit Exclusivity)
"""

__version__ = "0.1.0"

from .inference import VEFXReward

__all__ = ["VEFXReward"]
