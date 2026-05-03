import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


@unittest.skipIf(np is None, "numpy is not installed")
class ObservationVectorTests(unittest.TestCase):
    def test_build_observation_vector_fills_values_and_masks(self) -> None:
        from aiget.observation_vector import OBS_DIM, build_observation_vector

        out = np.zeros(OBS_DIM, dtype=np.float32)
        rich_values = np.arange(11, dtype=np.float32)
        rich_mask = np.ones(11, dtype=np.float32)

        obs = build_observation_vector(
            cursor_xy=(1.0, 2.0),
            cursor_vxy=(3.0, 4.0),
            latest_rich=(rich_values, rich_mask),
            previous_action=(0.25, -0.5),
            out=out,
        )

        self.assertIs(obs, out)
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertEqual(obs[0:4].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(obs[4:15].tolist(), rich_values.tolist())
        self.assertEqual(obs[15:17].tolist(), [0.25, -0.5])
        self.assertTrue(np.all(obs[17:21] == 1.0))
        self.assertTrue(np.all(obs[21:] == 1.0))

    def test_invalid_rich_fields_are_zeroed(self) -> None:
        from aiget.observation_vector import OBS_DIM, build_observation_vector

        out = np.zeros(OBS_DIM, dtype=np.float32)
        rich_values = np.ones(11, dtype=np.float32) * 5.0
        rich_mask = np.zeros(11, dtype=np.float32)

        obs = build_observation_vector(
            cursor_xy=(1.0, 2.0),
            cursor_vxy=(0.0, 0.0),
            latest_rich=(rich_values, rich_mask),
            previous_action=(0.0, 0.0),
            out=out,
        )

        self.assertEqual(obs[4:15].tolist(), [0.0] * 11)
        self.assertEqual(obs[21:].tolist(), [0.0] * 11)


if __name__ == "__main__":
    unittest.main()
