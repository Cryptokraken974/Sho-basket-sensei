"""Core coaching heuristics for BasketVision Coach.

The module intentionally starts small: vision systems can pass normalized
measurements here and receive deterministic feedback that is easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShotObservation:
    """Normalized measurements for one basketball shot attempt.

    Attributes:
        release_angle: Ball release angle in degrees.
        elbow_alignment: Normalized elbow-under-ball alignment score from 0 to 1.
        follow_through: Normalized follow-through completion score from 0 to 1.
        balance: Normalized landing/stance balance score from 0 to 1.
    """

    release_angle: float
    elbow_alignment: float
    follow_through: float
    balance: float

    def __post_init__(self) -> None:
        if not 0 <= self.release_angle <= 90:
            raise ValueError("release_angle must be between 0 and 90 degrees")
        for field_name in ("elbow_alignment", "follow_through", "balance"):
            value = getattr(self, field_name)
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be normalized between 0 and 1")


@dataclass(frozen=True, slots=True)
class CoachFeedback:
    """A deterministic coaching assessment for a shot."""

    score: int
    cues: tuple[str, ...]

    @property
    def summary(self) -> str:
        """Return a compact human-readable summary."""

        if not self.cues:
            return "Shot mechanics look balanced. Keep the rhythm."
        return " ".join(self.cues)


def assess_shot(observation: ShotObservation) -> CoachFeedback:
    """Assess a shot observation and return prioritized coaching cues."""

    score = 100
    cues: list[str] = []

    if observation.release_angle < 45:
        score -= 18
        cues.append("Raise the release window; the arc is coming out too flat.")
    elif observation.release_angle > 55:
        score -= 12
        cues.append("Relax the guide hand and avoid over-lofting the release.")

    if observation.elbow_alignment < 0.75:
        score -= 20
        cues.append("Stack the shooting elbow under the ball before extension.")

    if observation.follow_through < 0.8:
        score -= 15
        cues.append("Hold the follow-through until the ball reaches the rim.")

    if observation.balance < 0.7:
        score -= 15
        cues.append("Land softly with shoulders and hips square to the target.")

    return CoachFeedback(score=max(score, 0), cues=tuple(cues))
