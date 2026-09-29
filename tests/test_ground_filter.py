import numpy as np

from ground_filter import separate_ground_and_objects


def test_ground_and_object_separation_on_synthetic_scene():
    rng = np.random.default_rng(0)
    x = np.linspace(-2, 2, 40)
    y = np.linspace(-2, 2, 40)
    xx, yy = np.meshgrid(x, y)
    ground = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    object_points = np.column_stack((
        rng.uniform(-1.5, 1.5, 40),
        rng.uniform(-1.5, 1.5, 40),
        rng.uniform(0.4, 1.2, 40),
    ))
    points = np.vstack([ground, object_points])

    ground_mask, object_mask = separate_ground_and_objects(points, cell_size=0.5)

    assert ground_mask.sum() > 0
    assert object_mask.sum() > 0
    assert np.all(points[ground_mask, 2] <= 0.25)
    assert np.all(points[object_mask, 2] >= 0.2)
