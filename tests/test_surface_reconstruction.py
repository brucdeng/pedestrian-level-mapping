import numpy as np

from surface_reconstruction import compare_surface_methods


def test_surface_methods_produce_metrics_for_dense_cloud():
    rng = np.random.default_rng(3)
    x = np.linspace(-1, 1, 25)
    y = np.linspace(-1, 1, 25)
    xx, yy = np.meshgrid(x, y)
    z = 0.2 * np.sin(2 * xx) + 0.1 * np.cos(3 * yy)
    points = np.column_stack((xx.ravel(), yy.ravel(), z.ravel()))
    points += rng.normal(0.0, 0.01, size=points.shape)

    results = compare_surface_methods(points, method_names=["tin", "ball_pivot", "poisson", "alpha"])

    assert set(results) == {"tin", "ball_pivot", "poisson", "alpha"}
    for method_name, metrics in results.items():
        assert metrics["surface_area"] > 0
        assert metrics["empty_region_penalty"] >= 0.0
        assert metrics["method"] == method_name
