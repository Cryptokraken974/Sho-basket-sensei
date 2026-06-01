"""Court calibration and court-coordinate projection primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from uuid import uuid4


class CourtLandmark(StrEnum):
    """Supported known court landmarks for image-to-court calibration."""

    BASELINE_LEFT_SIDELINE = "baseline_left_sideline"
    BASELINE_RIGHT_SIDELINE = "baseline_right_sideline"
    OPPOSITE_BASELINE_LEFT_SIDELINE = "opposite_baseline_left_sideline"
    OPPOSITE_BASELINE_RIGHT_SIDELINE = "opposite_baseline_right_sideline"
    FREE_THROW_LEFT_LANE_INTERSECTION = "free_throw_left_lane_intersection"
    FREE_THROW_RIGHT_LANE_INTERSECTION = "free_throw_right_lane_intersection"
    OPPOSITE_FREE_THROW_LEFT_LANE_INTERSECTION = "opposite_free_throw_left_lane_intersection"
    OPPOSITE_FREE_THROW_RIGHT_LANE_INTERSECTION = "opposite_free_throw_right_lane_intersection"
    CENTER_CIRCLE_CENTER = "center_circle_center"
    CENTER_CIRCLE_TOP = "center_circle_top"
    CENTER_CIRCLE_BOTTOM = "center_circle_bottom"
    THREE_POINT_ARC_LEFT_WING = "three_point_arc_left_wing"
    THREE_POINT_ARC_RIGHT_WING = "three_point_arc_right_wing"
    THREE_POINT_ARC_TOP = "three_point_arc_top"
    OPPOSITE_THREE_POINT_ARC_LEFT_WING = "opposite_three_point_arc_left_wing"
    OPPOSITE_THREE_POINT_ARC_RIGHT_WING = "opposite_three_point_arc_right_wing"
    OPPOSITE_THREE_POINT_ARC_TOP = "opposite_three_point_arc_top"


def active_supported_landmark_labels() -> tuple[str, ...]:
    """Return API/UI labels for all supported calibration landmarks."""

    return tuple(landmark.value for landmark in CourtLandmark)


@dataclass(frozen=True, slots=True)
class ImagePoint:
    """Pixel-space calibration point on a captured video frame."""

    x: float
    y: float

    def to_json(self, landmark: CourtLandmark) -> dict[str, float | str]:
        return {"landmark": landmark.value, "x": float(self.x), "y": float(self.y)}


@dataclass(frozen=True, slots=True)
class CalibrationPointPair:
    """Known correspondence between a video pixel and standard court coordinate."""

    landmark: CourtLandmark
    image: ImagePoint
    court_x: float
    court_y: float

    def court_json(self) -> dict[str, float | str]:
        return {"landmark": self.landmark.value, "x": float(self.court_x), "y": float(self.court_y)}


@dataclass(frozen=True, slots=True)
class FrameCapture:
    """Captured video frame metadata used by the calibration UI."""

    video_id: str
    frame_idx: int
    ts_s: float
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        if self.frame_idx < 0:
            raise ValueError("frame_idx must be non-negative")
        if self.ts_s < 0:
            raise ValueError("ts_s must be non-negative")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")


@dataclass(frozen=True, slots=True)
class CalibrationDraft:
    """UI page state while a coach labels points on one captured frame."""

    capture: FrameCapture
    points: tuple[CalibrationPointPair, ...] = ()

    def place_landmark(
        self,
        landmark: CourtLandmark,
        *,
        image_x: float,
        image_y: float,
        court_x: float,
        court_y: float,
    ) -> CalibrationDraft:
        """Return updated draft with the labeled landmark placed or replaced."""

        if not 0 <= image_x <= self.capture.image_width:
            raise ValueError("image_x must be within the captured frame")
        if not 0 <= image_y <= self.capture.image_height:
            raise ValueError("image_y must be within the captured frame")
        point = CalibrationPointPair(
            landmark=landmark,
            image=ImagePoint(image_x, image_y),
            court_x=court_x,
            court_y=court_y,
        )
        kept = tuple(existing for existing in self.points if existing.landmark != landmark)
        return replace(self, points=(*kept, point))


@dataclass(frozen=True, slots=True)
class CourtCalibrationRecord:
    """Stored court_calibrations row."""

    calibration_id: str
    video_id: str
    image_points_json: list[dict[str, float | str]]
    court_points_json: list[dict[str, float | str]]
    homography_json: list[list[float]]
    valid_from_s: float
    valid_to_s: float | None
    created_by: str


@dataclass(frozen=True, slots=True)
class ProjectedCourtPoint:
    """Projected court coordinate returned by the backend API."""

    calibration_id: str
    court_x: float
    court_y: float


@dataclass(frozen=True, slots=True)
class ProjectablePoint:
    """Pixel point requested for overlay projection."""

    label: str
    image_x: float
    image_y: float


@dataclass(frozen=True, slots=True)
class OverlayCourtMap:
    """Lightweight court-map overlay payload for video playback."""

    width_ft: float = 94.0
    height_ft: float = 50.0

    def landmarks_json(self) -> list[dict[str, float | str]]:
        half_width = self.width_ft / 2
        half_height = self.height_ft / 2
        return [
            {"label": CourtLandmark.BASELINE_LEFT_SIDELINE.value, "x": 0.0, "y": 0.0},
            {"label": CourtLandmark.BASELINE_RIGHT_SIDELINE.value, "x": self.width_ft, "y": 0.0},
            {
                "label": CourtLandmark.OPPOSITE_BASELINE_LEFT_SIDELINE.value,
                "x": 0.0,
                "y": self.height_ft,
            },
            {
                "label": CourtLandmark.OPPOSITE_BASELINE_RIGHT_SIDELINE.value,
                "x": self.width_ft,
                "y": self.height_ft,
            },
            {"label": CourtLandmark.CENTER_CIRCLE_CENTER.value, "x": half_width, "y": half_height},
        ]


class CourtCalibrationService:
    """In-memory API service for court_calibrations and projection."""

    def __init__(self) -> None:
        self._calibrations: dict[str, list[CourtCalibrationRecord]] = {}

    def start_ui_draft(self, capture: FrameCapture) -> CalibrationDraft:
        return CalibrationDraft(capture=capture)

    def create_calibration(
        self,
        *,
        video_id: str,
        point_pairs: list[CalibrationPointPair] | tuple[CalibrationPointPair, ...],
        valid_from_s: float,
        valid_to_s: float | None,
        created_by: str,
    ) -> CourtCalibrationRecord:
        """Validate point pairs, compute homography, and append a calibration record."""

        if valid_from_s < 0:
            raise ValueError("valid_from_s must be non-negative")
        if valid_to_s is not None and valid_to_s <= valid_from_s:
            raise ValueError("valid_to_s must be greater than valid_from_s")
        if not created_by:
            raise ValueError("created_by is required")
        _validate_point_pairs(point_pairs)
        homography = _compute_homography(point_pairs)
        record = CourtCalibrationRecord(
            calibration_id=str(uuid4()),
            video_id=video_id,
            image_points_json=[pair.image.to_json(pair.landmark) for pair in point_pairs],
            court_points_json=[pair.court_json() for pair in point_pairs],
            homography_json=homography,
            valid_from_s=valid_from_s,
            valid_to_s=valid_to_s,
            created_by=created_by,
        )
        self._calibrations.setdefault(video_id, []).append(record)
        return record

    def list_calibrations(self, video_id: str) -> list[CourtCalibrationRecord]:
        return list(self._calibrations.get(video_id, ()))

    def active_calibration(self, video_id: str, ts_s: float) -> CourtCalibrationRecord:
        matches = [
            record
            for record in self._calibrations.get(video_id, ())
            if record.valid_from_s <= ts_s
            and (record.valid_to_s is None or ts_s < record.valid_to_s)
        ]
        if not matches:
            raise LookupError(f"no active court calibration for video {video_id!r} at {ts_s}s")
        return matches[-1]

    def project_point(
        self, *, video_id: str, image_x: float, image_y: float, ts_s: float
    ) -> ProjectedCourtPoint:
        record = self.active_calibration(video_id, ts_s)
        court_x, court_y = project_with_homography(record.homography_json, image_x, image_y)
        return ProjectedCourtPoint(
            calibration_id=record.calibration_id,
            court_x=court_x,
            court_y=court_y,
        )

    def overlay_projection_payload(
        self,
        *,
        video_id: str,
        ts_s: float,
        test_points: list[ProjectablePoint] | tuple[ProjectablePoint, ...],
        court_map: OverlayCourtMap | None = None,
    ) -> dict[str, object]:
        court_map = court_map or OverlayCourtMap()
        record = self.active_calibration(video_id, ts_s)
        points: list[dict[str, float | str]] = []
        for point in test_points:
            court_x, court_y = project_with_homography(
                record.homography_json, point.image_x, point.image_y
            )
            points.append(
                {
                    "label": point.label,
                    "image_x": point.image_x,
                    "image_y": point.image_y,
                    "court_x": court_x,
                    "court_y": court_y,
                }
            )
        return {
            "court_map": {
                "width_ft": court_map.width_ft,
                "height_ft": court_map.height_ft,
                "landmarks": court_map.landmarks_json(),
            },
            "calibration_id": record.calibration_id,
            "points": points,
        }


def project_with_homography(
    homography: list[list[float]], image_x: float, image_y: float
) -> tuple[float, float]:
    """Project pixel coordinates into court coordinates using a 3x3 homography."""

    if len(homography) != 3 or any(len(row) != 3 for row in homography):
        raise ValueError("homography must be a 3x3 matrix")
    denominator = homography[2][0] * image_x + homography[2][1] * image_y + homography[2][2]
    if abs(denominator) < 1e-12:
        raise ValueError("homography projection is undefined for this point")
    court_x = (
        homography[0][0] * image_x + homography[0][1] * image_y + homography[0][2]
    ) / denominator
    court_y = (
        homography[1][0] * image_x + homography[1][1] * image_y + homography[1][2]
    ) / denominator
    return (_clean_float(court_x), _clean_float(court_y))


def _clean_float(value: float) -> float:
    rounded = round(value, 12)
    if rounded == -0.0:
        return 0.0
    return rounded


def _validate_point_pairs(
    point_pairs: list[CalibrationPointPair] | tuple[CalibrationPointPair, ...],
) -> None:
    if len(point_pairs) < 4:
        raise ValueError("at least 4 point pairs are required to compute a homography")
    if len({pair.landmark for pair in point_pairs}) != len(point_pairs):
        raise ValueError("landmark labels must be unique within a calibration")
    if len({(pair.image.x, pair.image.y) for pair in point_pairs}) < 4:
        raise ValueError("at least 4 unique image points are required")
    if len({(pair.court_x, pair.court_y) for pair in point_pairs}) < 4:
        raise ValueError("at least 4 unique court points are required")


def _compute_homography(
    point_pairs: list[CalibrationPointPair] | tuple[CalibrationPointPair, ...],
) -> list[list[float]]:
    rows: list[list[float]] = []
    rhs: list[float] = []
    for pair in point_pairs:
        x = float(pair.image.x)
        y = float(pair.image.y)
        u = float(pair.court_x)
        v = float(pair.court_y)
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        rhs.append(v)

    normal_rows = [[sum(row[i] * row[j] for row in rows) for j in range(8)] for i in range(8)]
    normal_rhs = [
        sum(row[i] * value for row, value in zip(rows, rhs, strict=True)) for i in range(8)
    ]
    h = _solve_linear_system(normal_rows, normal_rhs)
    return [
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0],
    ]


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("point pairs are degenerate; homography cannot be computed")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        for idx in range(col, n + 1):
            augmented[col][idx] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            for idx in range(col, n + 1):
                augmented[row][idx] -= factor * augmented[col][idx]
    return [augmented[row][n] for row in range(n)]
