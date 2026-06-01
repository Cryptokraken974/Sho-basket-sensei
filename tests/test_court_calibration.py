from __future__ import annotations

import math

import pytest

from basketvision_coach.court_calibration import (
    CalibrationPointPair,
    CourtCalibrationService,
    CourtLandmark,
    FrameCapture,
    ImagePoint,
    OverlayCourtMap,
    ProjectablePoint,
    active_supported_landmark_labels,
)


def _square_pairs(scale_x: float = 94.0, scale_y: float = 50.0) -> list[CalibrationPointPair]:
    return [
        CalibrationPointPair(
            landmark=CourtLandmark.BASELINE_LEFT_SIDELINE,
            image=ImagePoint(0, 0),
            court_x=0,
            court_y=0,
        ),
        CalibrationPointPair(
            landmark=CourtLandmark.BASELINE_RIGHT_SIDELINE,
            image=ImagePoint(100, 0),
            court_x=scale_x,
            court_y=0,
        ),
        CalibrationPointPair(
            landmark=CourtLandmark.OPPOSITE_BASELINE_RIGHT_SIDELINE,
            image=ImagePoint(100, 100),
            court_x=scale_x,
            court_y=scale_y,
        ),
        CalibrationPointPair(
            landmark=CourtLandmark.OPPOSITE_BASELINE_LEFT_SIDELINE,
            image=ImagePoint(0, 100),
            court_x=0,
            court_y=scale_y,
        ),
    ]


def test_calibration_page_state_captures_frame_and_labeled_landmarks() -> None:
    capture = FrameCapture(
        video_id="game-1", frame_idx=180, ts_s=6.0, image_width=1920, image_height=1080
    )
    service = CourtCalibrationService()

    draft = service.start_ui_draft(capture)
    draft = draft.place_landmark(
        CourtLandmark.BASELINE_LEFT_SIDELINE,
        image_x=125.0,
        image_y=940.0,
        court_x=0.0,
        court_y=0.0,
    )

    assert draft.capture == capture
    assert draft.points[0].landmark is CourtLandmark.BASELINE_LEFT_SIDELINE
    assert "baseline_left_sideline" in active_supported_landmark_labels()
    assert "free_throw_left_lane_intersection" in active_supported_landmark_labels()
    assert "center_circle_top" in active_supported_landmark_labels()
    assert "three_point_arc_left_wing" in active_supported_landmark_labels()


def test_api_rejects_calibration_without_enough_point_pairs() -> None:
    service = CourtCalibrationService()

    with pytest.raises(ValueError, match="at least 4 point pairs"):
        service.create_calibration(
            video_id="game-1",
            point_pairs=_square_pairs()[:3],
            valid_from_s=0.0,
            valid_to_s=12.0,
            created_by="coach@example.com",
        )


def test_calibration_record_stores_points_homography_validity_and_creator() -> None:
    service = CourtCalibrationService()

    record = service.create_calibration(
        video_id="game-1",
        point_pairs=_square_pairs(),
        valid_from_s=0.0,
        valid_to_s=12.0,
        created_by="coach@example.com",
    )

    assert record.video_id == "game-1"
    assert record.image_points_json[0] == {
        "landmark": "baseline_left_sideline",
        "x": 0.0,
        "y": 0.0,
    }
    assert record.court_points_json[1] == {
        "landmark": "baseline_right_sideline",
        "x": 94.0,
        "y": 0.0,
    }
    assert len(record.homography_json) == 3
    assert record.valid_from_s == 0.0
    assert record.valid_to_s == 12.0
    assert record.created_by == "coach@example.com"


def test_backend_projects_image_coordinates_with_active_timestamp_calibration() -> None:
    service = CourtCalibrationService()
    service.create_calibration(
        video_id="game-1",
        point_pairs=_square_pairs(),
        valid_from_s=0.0,
        valid_to_s=12.0,
        created_by="coach@example.com",
    )

    projected = service.project_point(video_id="game-1", image_x=50.0, image_y=25.0, ts_s=5.0)

    assert math.isclose(projected.court_x, 47.0, abs_tol=1e-9)
    assert math.isclose(projected.court_y, 12.5, abs_tol=1e-9)
    assert projected.calibration_id


def test_replacement_keeps_prior_calibration_records_and_uses_new_active_range() -> None:
    service = CourtCalibrationService()
    first = service.create_calibration(
        video_id="game-1",
        point_pairs=_square_pairs(scale_x=94.0),
        valid_from_s=0.0,
        valid_to_s=10.0,
        created_by="coach-a",
    )
    second = service.create_calibration(
        video_id="game-1",
        point_pairs=_square_pairs(scale_x=47.0),
        valid_from_s=10.0,
        valid_to_s=None,
        created_by="coach-b",
    )

    assert [record.calibration_id for record in service.list_calibrations("game-1")] == [
        first.calibration_id,
        second.calibration_id,
    ]
    assert service.project_point(video_id="game-1", image_x=100, image_y=0, ts_s=5).court_x == 94.0
    assert service.project_point(video_id="game-1", image_x=100, image_y=0, ts_s=12).court_x == 47.0


def test_overlay_court_map_returns_projected_points_for_test_coordinates() -> None:
    service = CourtCalibrationService()
    record = service.create_calibration(
        video_id="game-1",
        point_pairs=_square_pairs(),
        valid_from_s=0.0,
        valid_to_s=None,
        created_by="coach@example.com",
    )
    overlay = OverlayCourtMap(width_ft=94.0, height_ft=50.0)

    payload = service.overlay_projection_payload(
        video_id="game-1",
        ts_s=2.0,
        test_points=[ProjectablePoint(label="ball", image_x=25.0, image_y=50.0)],
        court_map=overlay,
    )

    assert payload["court_map"] == {
        "width_ft": 94.0,
        "height_ft": 50.0,
        "landmarks": overlay.landmarks_json(),
    }
    assert payload["calibration_id"] == record.calibration_id
    assert payload["points"] == [
        {
            "label": "ball",
            "image_x": 25.0,
            "image_y": 50.0,
            "court_x": 23.5,
            "court_y": 25.0,
        }
    ]
