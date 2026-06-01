"""Command-line interface for quick BasketVision Coach assessments."""

from __future__ import annotations

import argparse

from basketvision_coach.coach import ShotObservation, assess_shot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assess basketball shot mechanics.")
    parser.add_argument("--release-angle", type=float, required=True)
    parser.add_argument("--elbow-alignment", type=float, required=True)
    parser.add_argument("--follow-through", type=float, required=True)
    parser.add_argument("--balance", type=float, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    feedback = assess_shot(
        ShotObservation(
            release_angle=args.release_angle,
            elbow_alignment=args.elbow_alignment,
            follow_through=args.follow_through,
            balance=args.balance,
        )
    )
    print(f"score={feedback.score}")
    print(feedback.summary)


if __name__ == "__main__":
    main()
