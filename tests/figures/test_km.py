import numpy as np

from behaviour_stat_pipeline.figures.km import km_steps


def test_km_steps_hand_computed():
    # times 1..5; animal at t=2 censored, animal at t=5 censored
    # S(1) = 4/5, censor at 2, S(3) = 4/5 * 2/3, S(4) = S(3) * 1/2, follow-up ends at 5
    t, s, ct, cs = km_steps(np.array([1, 2, 3, 4, 5.0]), np.array([1, 0, 1, 1, 0]))
    s3 = 0.8 * 2 / 3
    np.testing.assert_allclose(t, [0, 1, 3, 4, 5])
    np.testing.assert_allclose(s, [1, 0.8, s3, s3 / 2, s3 / 2])
    np.testing.assert_allclose(ct, [2, 5])
    np.testing.assert_allclose(cs, [0.8, s3 / 2])


def test_km_steps_no_events():
    t, s, ct, cs = km_steps(np.array([300.0, 302.0]), np.array([0, 0]))
    np.testing.assert_allclose(s, [1, 1])
    np.testing.assert_allclose(cs, [1, 1])
