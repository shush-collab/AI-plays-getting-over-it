import unittest

import numpy as np

from aiget.progress_signal import ProgressEstimator, ProgressSource
from aiget.reward import HeightReward


class RewardSignalTests(unittest.TestCase):
    def test_progress_estimator_uses_body_first(self) -> None:
        state = np.zeros(32, dtype=np.float32)
        state[5] = 10.0
        state[12] = 20.0
        state[22] = 1.0
        state[29] = 1.0

        sample = ProgressEstimator().estimate(state)

        self.assertTrue(sample.valid)
        self.assertEqual(sample.y, 10.0)
        self.assertEqual(sample.source, ProgressSource.MEMORY_BODY)

    def test_progress_estimator_falls_back_to_progress(self) -> None:
        state = np.zeros(32, dtype=np.float32)
        state[12] = 20.0
        state[22] = 0.0
        state[29] = 1.0

        sample = ProgressEstimator().estimate(state)

        self.assertTrue(sample.valid)
        self.assertEqual(sample.y, 20.0)
        self.assertEqual(sample.source, ProgressSource.MEMORY_PROGRESS)

    def test_reward_positive_for_new_best_height(self) -> None:
        reward = HeightReward()
        reward.reset()

        reward.compute(ProgressEstimator().estimate(make_state(10.0)))
        out1 = reward.compute(ProgressEstimator().estimate(make_state(11.0)))

        self.assertGreater(out1.reward, 0)
        self.assertGreater(out1.debug["delta_best"], 0)

    def test_reward_penalizes_invalid_progress(self) -> None:
        reward = HeightReward(max_invalid_steps=2)
        reward.reset()

        invalid = ProgressEstimator().estimate(np.zeros(32, dtype=np.float32))

        out1 = reward.compute(invalid)
        out2 = reward.compute(invalid)

        self.assertLess(out1.reward, 0)
        self.assertTrue(out2.truncated)


def make_state(body_y: float) -> np.ndarray:
    state = np.zeros(32, dtype=np.float32)
    state[5] = body_y
    state[22] = 1.0
    return state


if __name__ == "__main__":
    unittest.main()
