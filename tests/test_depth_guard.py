from depth_research import check_pose_calibration, get_depth_research_plan


def test_depth_research_requires_pose_and_calibration():
    metadata = {"crs": None, "pose": None, "calibration": None}
    assert check_pose_calibration(metadata) is False
    plan = get_depth_research_plan(metadata)
    assert plan["status"] == "deferred"
    assert "pose" in " ".join(plan["reasons"]).lower()
