import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.observation_schema import build_observation_schema, to_markdown


class ObservationSchemaTests(unittest.TestCase):
    def test_offsets_and_flat_dim(self) -> None:
        schema = build_observation_schema(body_ray_count=4, hammer_ray_count=6, action_dim=3)

        data = schema.to_dict()
        features = data["features"]

        self.assertEqual(data["version"], "v1")
        self.assertEqual(data["flat_dim"], 33)
        self.assertEqual(features[0]["offset"], 0)
        self.assertEqual(features[-1]["offset"], 18)
        self.assertEqual(features[-1]["size"], 15)

    def test_markdown_renders_table(self) -> None:
        schema = build_observation_schema(body_ray_count=2, hammer_ray_count=2, action_dim=2)

        rendered = to_markdown(schema)

        self.assertIn("# Observation Schema", rendered)
        self.assertIn("| Offset | Size | Name | Description |", rendered)
        self.assertIn("`cursor_position_xy`", rendered)


if __name__ == "__main__":
    unittest.main()
