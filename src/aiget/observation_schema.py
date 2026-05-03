#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    size: int
    description: str


@dataclass(frozen=True)
class ObservationSchema:
    version: str
    body_ray_count: int
    hammer_ray_count: int
    action_dim: int
    features: tuple[FeatureSpec, ...]

    @property
    def flat_dim(self) -> int:
        return sum(feature.size for feature in self.features)

    def to_dict(self) -> dict[str, object]:
        offset = 0
        entries: list[dict[str, object]] = []
        for feature in self.features:
            entries.append(
                {
                    "name": feature.name,
                    "size": feature.size,
                    "offset": offset,
                    "description": feature.description,
                }
            )
            offset += feature.size
        return {
            "version": self.version,
            "flat_dim": self.flat_dim,
            "body_ray_count": self.body_ray_count,
            "hammer_ray_count": self.hammer_ray_count,
            "action_dim": self.action_dim,
            "features": entries,
        }


def build_observation_schema(body_ray_count: int = 0, hammer_ray_count: int = 0, action_dim: int = 2) -> ObservationSchema:
    return ObservationSchema(
        version="v1",
        body_ray_count=body_ray_count,
        hammer_ray_count=hammer_ray_count,
        action_dim=action_dim,
        features=(
            FeatureSpec("cursor_position_xy", 2, "Live `fakeCursorRB` position."),
            FeatureSpec("cursor_velocity_xy", 2, "Live `fakeCursorRB` linear velocity."),
            FeatureSpec("body_position_xy", 2, "Pot/body world position when raw layout resolves it."),
            FeatureSpec("body_velocity_xy", 2, "Pot/body linear velocity derived from rich snapshots."),
            FeatureSpec("hammer_tip_xy", 2, "Hammer tip world position when raw layout resolves it."),
            FeatureSpec("hammer_direction_sin_cos", 2, "Hammer direction encoded as sin(theta), cos(theta)."),
            FeatureSpec("progress_features", 3, "Current height, best height this episode, time since last upward progress."),
            FeatureSpec("previous_action", action_dim, "Previous action sent by the policy."),
            FeatureSpec(
                "valid_mask",
                15,
                "Binary validity mask for cursor, body, hammer, and progress state values.",
            ),
        ),
    )


def to_markdown(schema: ObservationSchema) -> str:
    data = schema.to_dict()
    lines = [
        "# Observation Schema",
        "",
        f"- Version: `{data['version']}`",
        f"- Flat dimension: `{data['flat_dim']}`",
        f"- Body rays: `{data['body_ray_count']}`",
        f"- Hammer rays: `{data['hammer_ray_count']}`",
        f"- Action dimension: `{data['action_dim']}`",
        "",
        "| Offset | Size | Name | Description |",
        "| ---: | ---: | --- | --- |",
    ]
    for feature in data["features"]:
        lines.append(
            f"| {feature['offset']} | {feature['size']} | `{feature['name']}` | {feature['description']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the planned RL observation schema for AIget.")
    parser.add_argument("--body-rays", type=int, default=0, help="Reserved; LIDAR is not part of v1.")
    parser.add_argument("--hammer-rays", type=int, default=0, help="Reserved; LIDAR is not part of v1.")
    parser.add_argument("--action-dim", type=int, default=2, help="Number of action values echoed back as previous action.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json", help="Output format.")
    args = parser.parse_args()

    schema = build_observation_schema(
        body_ray_count=args.body_rays,
        hammer_ray_count=args.hammer_rays,
        action_dim=args.action_dim,
    )
    if args.format == "markdown":
        print(to_markdown(schema))
    else:
        print(json.dumps(schema.to_dict(), indent=2))


if __name__ == "__main__":
    main()
