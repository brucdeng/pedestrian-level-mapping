from __future__ import annotations


def check_pose_calibration(metadata: dict | None) -> bool:
    """Return True only when scanner pose and calibration are both known."""
    if not isinstance(metadata, dict):
        return False
    pose = metadata.get("pose")
    calibration = metadata.get("calibration")
    if pose in (None, "", "unknown", "unverified"):
        return False
    if calibration in (None, "", "unknown", "unverified"):
        return False
    return True


def get_depth_research_plan(metadata: dict | None) -> dict:
    """Guard advanced depth reconstruction until pose/calibration are confirmed."""
    if metadata is None:
        metadata = {}
    reasons: list[str] = []
    if metadata.get("pose") in (None, "", "unknown", "unverified"):
        reasons.append("scanner pose is missing or unverified")
    if metadata.get("calibration") in (None, "", "unknown", "unverified"):
        reasons.append("scanner calibration is missing or unverified")
    if metadata.get("crs") in (None, "", "unknown", "unverified"):
        reasons.append("scene CRS is not confirmed")

    if reasons:
        return {
            "status": "deferred",
            "reasons": reasons,
            "methods": [],
            "next_action": "confirm pose, calibration, and CRS before any depth fusion or implicit-surface work",
        }

    return {
        "status": "ready",
        "reasons": [],
        "methods": [
            "original scan / range-image reconstruction",
            "pose-aware depth fusion",
            "TSDF reconstruction",
            "neural or learned implicit surfaces",
        ],
        "next_action": "run a calibrated pose-aware depth pipeline on representative blocks",
    }
