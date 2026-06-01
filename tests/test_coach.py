from __future__ import annotations

import pytest

from basketvision_coach import ShotObservation, assess_shot


def test_balanced_shot_returns_clean_summary() -> None:
    feedback = assess_shot(
        ShotObservation(
            release_angle=50,
            elbow_alignment=0.9,
            follow_through=0.95,
            balance=0.85,
        )
    )

    assert feedback.score == 100
    assert feedback.cues == ()
    assert feedback.summary == "Shot mechanics look balanced. Keep the rhythm."


def test_low_arc_and_unstable_base_generate_prioritized_cues() -> None:
    feedback = assess_shot(
        ShotObservation(
            release_angle=39,
            elbow_alignment=0.7,
            follow_through=0.65,
            balance=0.5,
        )
    )

    assert feedback.score == 32
    assert feedback.cues == (
        "Raise the release window; the arc is coming out too flat.",
        "Stack the shooting elbow under the ball before extension.",
        "Hold the follow-through until the ball reaches the rim.",
        "Land softly with shoulders and hips square to the target.",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"release_angle": -1, "elbow_alignment": 1, "follow_through": 1, "balance": 1},
            "release_angle",
        ),
        (
            {"release_angle": 45, "elbow_alignment": 1.2, "follow_through": 1, "balance": 1},
            "elbow_alignment",
        ),
        (
            {"release_angle": 45, "elbow_alignment": 1, "follow_through": -0.1, "balance": 1},
            "follow_through",
        ),
        (
            {"release_angle": 45, "elbow_alignment": 1, "follow_through": 1, "balance": 1.1},
            "balance",
        ),
    ],
)
def test_observation_validates_measurement_ranges(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ShotObservation(**kwargs)
