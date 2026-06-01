# BasketVision Coach

BasketVision Coach is a Python scaffold for turning basketball shot measurements from video or pose-estimation pipelines into concise coaching feedback.

The initial package focuses on a small, typed domain model that can be expanded as the vision pipeline matures:

- capture normalized shot observations
- score release angle, elbow alignment, follow-through, and balance
- return prioritized coaching cues
- expose a lightweight CLI for smoke testing

## Quick start

```bash
uv sync --extra dev
uv run pytest
uv run mypy src tests
uv run ruff check .
```

## CLI

```bash
uv run basketvision-coach --release-angle 49 --elbow-alignment 0.82 --follow-through 0.9 --balance 0.76
```

## Project layout

```text
src/basketvision_coach/   package code
tests/                    focused tests
```
