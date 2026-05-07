from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Callable

import numpy as np


WALL_THICKNESS_BINS = np.array([0.05, 0.08], dtype=np.float64)
WALL_HALF_LENGTH_BINS = np.array([0.10, 0.50, 0.85, 1.10, 1.35, 1.65, 2.35, 2.70, 5.05], dtype=np.float64)
BOX_HALF_EXTENT_BINS = np.array([0.25, 0.35, 0.45, 0.50], dtype=np.float64)
CYLINDER_RADIUS_BINS = np.array([0.20, 0.24, 0.28, 0.32], dtype=np.float64)


def _nearest_bin(value: float, bins: np.ndarray) -> float:
    idx = int(np.argmin(np.abs(bins - float(value))))
    return float(bins[idx])


def snap_wall_size(sx: float, sy: float, z: float = 0.25) -> np.ndarray:
    sx = float(sx)
    sy = float(sy)
    if sx <= sy:
        return np.array(
            [_nearest_bin(sx, WALL_THICKNESS_BINS), _nearest_bin(sy, WALL_HALF_LENGTH_BINS), z],
            dtype=np.float64,
        )
    return np.array(
        [_nearest_bin(sx, WALL_HALF_LENGTH_BINS), _nearest_bin(sy, WALL_THICKNESS_BINS), z],
        dtype=np.float64,
    )


def snap_box_size(sx: float, sy: float, z: float = 0.18) -> np.ndarray:
    return np.array(
        [_nearest_bin(sx, BOX_HALF_EXTENT_BINS), _nearest_bin(sy, BOX_HALF_EXTENT_BINS), z],
        dtype=np.float64,
    )


def snap_cylinder_size(radius: float, half_height: float = 0.22) -> np.ndarray:
    return np.array([_nearest_bin(radius, CYLINDER_RADIUS_BINS), half_height, 0.01], dtype=np.float64)


@dataclass(frozen=True)
class BoxSpec:
    name: str
    pos: np.ndarray
    size: np.ndarray


@dataclass(frozen=True)
class CylinderSpec:
    name: str
    pos: np.ndarray
    size: np.ndarray


@dataclass(frozen=True)
class LayoutSpec:
    template_name: str
    seed: int
    start_xy: np.ndarray
    start_yaw: float
    goal_xy: np.ndarray
    walls: list[BoxSpec]
    boxes: list[BoxSpec]
    cylinders: list[CylinderSpec]
    geodesic_distance: float


class StaticHardLayoutGenerator:
    """Static hard navigation layout generator with grid connectivity checks."""

    TEMPLATE_NAMES = (
        "single-bottleneck",
        "t-junction",
        "occluded-corner",
        "clutter-room",
        "loop-with-shortcut",
    )

    def __init__(
        self,
        arena_bounds: tuple[float, float, float, float] = (-5.0, 5.0, -5.0, 5.0),
        grid_resolution: float = 0.10,
        inflation_radius: float = 0.35,
        min_path_length: float = 8.0,
        max_attempts: int = 80,
    ):
        self.arena_bounds = tuple(float(v) for v in arena_bounds)
        self.grid_resolution = float(grid_resolution)
        self.inflation_radius = float(inflation_radius)
        self.min_path_length = float(min_path_length)
        self.max_attempts = int(max_attempts)

        if self.grid_resolution <= 0.0:
            raise ValueError("grid_resolution must be positive")
        if self.inflation_radius < 0.0:
            raise ValueError("inflation_radius must be non-negative")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        xmin, xmax, ymin, ymax = self.arena_bounds
        if not (xmin < xmax and ymin < ymax):
            raise ValueError("arena_bounds must be (xmin, xmax, ymin, ymax)")

    def generate(self, seed: int, template_name: str | None = None) -> LayoutSpec:
        if template_name is None:
            template_name = self.TEMPLATE_NAMES[int(seed) % len(self.TEMPLATE_NAMES)]
        if template_name not in self.TEMPLATE_NAMES:
            raise ValueError(f"unknown template_name {template_name!r}")

        builder = self._template_builders()[template_name]
        last_reason = "no attempts made"
        for attempt in range(self.max_attempts):
            rng = np.random.default_rng(int(seed) + attempt * 1009)
            walls, boxes, cylinders, start_regions, goal_regions = builder(rng)
            start_xy = self._sample_from_regions(rng, start_regions)
            goal_xy = self._sample_from_regions(rng, goal_regions)
            start_yaw = float(np.arctan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0]))

            distance = self._grid_path_length(start_xy, goal_xy, walls, boxes, cylinders)
            if distance is None:
                last_reason = "start/goal occupied or disconnected"
                continue
            if distance < self.min_path_length:
                last_reason = f"path length {distance:.3f} < {self.min_path_length:.3f}"
                continue

            return LayoutSpec(
                template_name=template_name,
                seed=int(seed),
                start_xy=start_xy,
                start_yaw=start_yaw,
                goal_xy=goal_xy,
                walls=walls,
                boxes=boxes,
                cylinders=cylinders,
                geodesic_distance=float(distance),
            )

        raise RuntimeError(
            f"failed to generate valid {template_name!r} layout for seed {seed} "
            f"after {self.max_attempts} attempts: {last_reason}"
        )

    def _template_builders(self) -> dict[str, Callable[[np.random.Generator], tuple]]:
        return {
            "single-bottleneck": self._single_bottleneck,
            "t-junction": self._t_junction,
            "occluded-corner": self._occluded_corner,
            "clutter-room": self._clutter_room,
            "loop-with-shortcut": self._loop_with_shortcut,
        }

    def _single_bottleneck(self, rng: np.random.Generator):
        walls = self._boundary_walls()
        gap_y = float(rng.uniform(-1.0, 1.0))
        gap_half = float(rng.uniform(0.65, 0.95))
        thick = 0.08
        walls.extend(
            self._vertical_split_wall("wall", 4, 0.0, gap_y, gap_half, thick)
        )
        walls.append(self._box("wall_baffle_left", -1.8, gap_y - 1.3, 0.5, 0.85, z=0.25))
        walls.append(self._box("wall_baffle_right", 1.8, gap_y + 1.3, 0.5, 0.85, z=0.25))

        boxes = [
            self._box("box_left_low", -3.3, float(rng.uniform(-3.1, -1.8)), 0.35, 0.35, z=0.18),
            self._box("box_right_high", 3.2, float(rng.uniform(1.8, 3.2)), 0.45, 0.30, z=0.18),
        ]
        cylinders = [
            self._cylinder("cyl_left", -2.0, float(rng.uniform(1.6, 3.2)), float(rng.uniform(0.22, 0.32))),
            self._cylinder("cyl_right", 2.3, float(rng.uniform(-3.2, -1.8)), float(rng.uniform(0.22, 0.30))),
        ]
        start_regions = [(-4.35, -2.9, -4.1, 4.1)]
        goal_regions = [(2.9, 4.35, -4.1, 4.1)]
        return walls, boxes, cylinders, start_regions, goal_regions

    def _t_junction(self, rng: np.random.Generator):
        walls = self._boundary_walls()
        stem_x = float(rng.uniform(-0.35, 0.35))
        gap_half = float(rng.uniform(0.75, 1.05))
        walls.extend(self._vertical_split_wall("wall_t_stem", 4, stem_x, -2.2, gap_half, 0.08))
        walls.extend(self._horizontal_split_wall("wall_t_bar", 6, 0.8, 0.0, 1.2, 0.08))
        walls.append(self._box("wall_t_deadend", stem_x - 2.4, -2.6, 0.08, 1.0, z=0.25))

        boxes = [
            self._box("box_t_0", float(rng.uniform(1.3, 2.4)), float(rng.uniform(-3.6, -2.6)), 0.35, 0.45, z=0.18),
            self._box("box_t_1", float(rng.uniform(-3.7, -2.5)), float(rng.uniform(1.8, 3.2)), 0.40, 0.32, z=0.18),
        ]
        cylinders = [
            self._cylinder("cyl_t_0", float(rng.uniform(2.2, 3.8)), float(rng.uniform(1.8, 3.4)), 0.28),
            self._cylinder("cyl_t_1", float(rng.uniform(-1.8, -0.8)), float(rng.uniform(-3.6, -2.4)), 0.24),
        ]
        start_regions = [(-4.2, -2.5, -4.1, -2.5), (2.0, 4.1, -4.1, -2.7)]
        goal_regions = [(-4.2, -2.5, 2.0, 4.1), (2.4, 4.1, 2.0, 4.1)]
        return walls, boxes, cylinders, start_regions, goal_regions

    def _occluded_corner(self, rng: np.random.Generator):
        walls = self._boundary_walls()
        elbow_x = float(rng.uniform(-1.0, -0.3))
        elbow_y = float(rng.uniform(-0.2, 0.7))
        walls.append(self._box("wall_corner_vertical", elbow_x, -2.05, 0.08, 2.45, z=0.25))
        walls.append(self._box("wall_corner_horizontal", 1.25, elbow_y, 2.35, 0.08, z=0.25))
        walls.append(self._box("wall_corner_screen", elbow_x + 1.1, elbow_y + 1.35, 0.08, 1.1, z=0.25))
        walls.append(self._box("wall_corner_return", -3.0, 1.65, 1.05, 0.08, z=0.25))

        boxes = [
            self._box("box_corner_0", float(rng.uniform(-3.8, -2.7)), float(rng.uniform(-0.7, 0.5)), 0.35, 0.35, z=0.18),
            self._box("box_corner_1", float(rng.uniform(2.2, 3.5)), float(rng.uniform(-3.5, -2.5)), 0.45, 0.25, z=0.18),
        ]
        cylinders = [
            self._cylinder("cyl_corner_0", float(rng.uniform(1.4, 2.6)), float(rng.uniform(1.2, 2.5)), 0.25),
            self._cylinder("cyl_corner_1", float(rng.uniform(-3.7, -2.8)), float(rng.uniform(2.8, 3.7)), 0.27),
        ]
        start_regions = [(-4.2, -2.8, -4.1, -2.6)]
        goal_regions = [(2.6, 4.2, 2.4, 4.1)]
        return walls, boxes, cylinders, start_regions, goal_regions

    def _clutter_room(self, rng: np.random.Generator):
        walls = self._boundary_walls()
        walls.append(self._box("wall_room_left", -2.35, 0.4, 0.08, 2.2, z=0.25))
        walls.append(self._box("wall_room_right", 2.35, -0.4, 0.08, 2.2, z=0.25))
        walls.append(self._box("wall_room_top", 0.0, 2.55, 1.65, 0.08, z=0.25))
        walls.append(self._box("wall_room_bottom", 0.0, -2.55, 1.65, 0.08, z=0.25))

        boxes = []
        for idx in range(6):
            x = float(rng.uniform(-3.5, 3.5))
            y = float(rng.uniform(-3.5, 3.5))
            sx = float(rng.uniform(0.22, 0.45))
            sy = float(rng.uniform(0.22, 0.50))
            boxes.append(self._box(f"box_clutter_{idx}", x, y, sx, sy, z=0.18))

        cylinders = []
        for idx in range(5):
            x = float(rng.uniform(-3.7, 3.7))
            y = float(rng.uniform(-3.7, 3.7))
            r = float(rng.uniform(0.20, 0.32))
            cylinders.append(self._cylinder(f"cyl_clutter_{idx}", x, y, r))

        start_regions = [(-4.3, -3.0, -4.0, -2.2), (-4.3, -3.0, 2.2, 4.0)]
        goal_regions = [(3.0, 4.3, -4.0, -2.2), (3.0, 4.3, 2.2, 4.0)]
        return walls, boxes, cylinders, start_regions, goal_regions

    def _loop_with_shortcut(self, rng: np.random.Generator):
        walls = self._boundary_walls()
        off = float(rng.uniform(-0.25, 0.25))
        walls.append(self._box("wall_loop_inner_left", -1.65 + off, 0.0, 0.08, 2.7, z=0.25))
        walls.append(self._box("wall_loop_inner_right", 1.65 + off, 0.0, 0.08, 2.7, z=0.25))
        walls.append(self._box("wall_loop_inner_top", off, 1.95, 1.15, 0.08, z=0.25))
        walls.append(self._box("wall_loop_inner_bottom", off, -1.95, 1.15, 0.08, z=0.25))
        walls.append(self._box("wall_loop_shortcut_gate", off, 0.35, 0.55, 0.08, z=0.25))

        boxes = [
            self._box("box_loop_0", float(rng.uniform(-3.7, -2.5)), float(rng.uniform(-0.5, 1.1)), 0.35, 0.40, z=0.18),
            self._box("box_loop_1", float(rng.uniform(2.5, 3.7)), float(rng.uniform(-1.1, 0.5)), 0.40, 0.35, z=0.18),
            self._box("box_loop_2", float(rng.uniform(-0.8, 0.8)), float(rng.uniform(3.0, 3.8)), 0.35, 0.30, z=0.18),
        ]
        cylinders = [
            self._cylinder("cyl_loop_0", float(rng.uniform(-0.8, 0.8)), float(rng.uniform(-3.8, -3.0)), 0.26),
            self._cylinder("cyl_loop_1", float(rng.uniform(2.8, 3.7)), float(rng.uniform(2.7, 3.6)), 0.24),
        ]
        start_regions = [(-4.2, -2.8, -1.0, 1.0)]
        goal_regions = [(2.8, 4.2, -1.0, 1.0)]
        return walls, boxes, cylinders, start_regions, goal_regions

    def _boundary_walls(self) -> list[BoxSpec]:
        xmin, xmax, ymin, ymax = self.arena_bounds
        thickness = 0.05
        z = 0.25
        return [
            self._box("wall_boundary_north", 0.5 * (xmin + xmax), ymax + thickness, 0.5 * (xmax - xmin) + thickness, thickness, z),
            self._box("wall_boundary_south", 0.5 * (xmin + xmax), ymin - thickness, 0.5 * (xmax - xmin) + thickness, thickness, z),
            self._box("wall_boundary_east", xmax + thickness, 0.5 * (ymin + ymax), thickness, 0.5 * (ymax - ymin) + thickness, z),
            self._box("wall_boundary_west", xmin - thickness, 0.5 * (ymin + ymax), thickness, 0.5 * (ymax - ymin) + thickness, z),
        ]

    def _vertical_split_wall(self, prefix: str, start_idx: int, x: float, gap_y: float, gap_half: float, thick: float) -> list[BoxSpec]:
        _, _, ymin, ymax = self.arena_bounds
        lower_center = 0.5 * (ymin + gap_y - gap_half)
        lower_half = 0.5 * ((gap_y - gap_half) - ymin)
        upper_center = 0.5 * (gap_y + gap_half + ymax)
        upper_half = 0.5 * (ymax - (gap_y + gap_half))
        return [
            self._box(f"{prefix}_{start_idx}", x, lower_center, thick, max(0.1, lower_half), z=0.25),
            self._box(f"{prefix}_{start_idx + 1}", x, upper_center, thick, max(0.1, upper_half), z=0.25),
        ]

    def _horizontal_split_wall(self, prefix: str, start_idx: int, y: float, gap_x: float, gap_half: float, thick: float) -> list[BoxSpec]:
        xmin, xmax, _, _ = self.arena_bounds
        left_center = 0.5 * (xmin + gap_x - gap_half)
        left_half = 0.5 * ((gap_x - gap_half) - xmin)
        right_center = 0.5 * (gap_x + gap_half + xmax)
        right_half = 0.5 * (xmax - (gap_x + gap_half))
        return [
            self._box(f"{prefix}_{start_idx}", left_center, y, max(0.1, left_half), thick, z=0.25),
            self._box(f"{prefix}_{start_idx + 1}", right_center, y, max(0.1, right_half), thick, z=0.25),
        ]

    def _box(self, name: str, x: float, y: float, sx: float, sy: float, z: float) -> BoxSpec:
        size = snap_wall_size(sx, sy, z) if name.startswith("wall") else snap_box_size(sx, sy, z)
        return BoxSpec(
            name=name,
            pos=np.array([x, y, z], dtype=np.float64),
            size=size,
        )

    def _cylinder(self, name: str, x: float, y: float, radius: float) -> CylinderSpec:
        half_height = 0.22
        return CylinderSpec(
            name=name,
            pos=np.array([x, y, half_height], dtype=np.float64),
            size=snap_cylinder_size(radius, half_height),
        )

    def _sample_from_regions(
        self, rng: np.random.Generator, regions: list[tuple[float, float, float, float]]
    ) -> np.ndarray:
        region = regions[int(rng.integers(0, len(regions)))]
        xmin, xmax, ymin, ymax = region
        return np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], dtype=np.float64)

    def _grid_path_length(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        walls: list[BoxSpec],
        boxes: list[BoxSpec],
        cylinders: list[CylinderSpec],
    ) -> float | None:
        xs, ys = self._grid_axes()
        occupied = np.zeros((len(xs), len(ys)), dtype=bool)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")

        for box in [*walls, *boxes]:
            half = box.size[:2] + self.inflation_radius
            center = box.pos[:2]
            occupied |= (np.abs(xx - center[0]) <= half[0]) & (np.abs(yy - center[1]) <= half[1])

        for cylinder in cylinders:
            radius = float(cylinder.size[0] + self.inflation_radius)
            center = cylinder.pos[:2]
            occupied |= (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius * radius

        start_idx = self._xy_to_grid(start_xy, xs, ys)
        goal_idx = self._xy_to_grid(goal_xy, xs, ys)
        if start_idx is None or goal_idx is None:
            return None
        if occupied[start_idx] or occupied[goal_idx]:
            return None

        return self._dijkstra_distance(occupied, start_idx, goal_idx)

    def _grid_axes(self) -> tuple[np.ndarray, np.ndarray]:
        xmin, xmax, ymin, ymax = self.arena_bounds
        nx = int(np.floor((xmax - xmin) / self.grid_resolution)) + 1
        ny = int(np.floor((ymax - ymin) / self.grid_resolution)) + 1
        xs = xmin + np.arange(nx, dtype=np.float64) * self.grid_resolution
        ys = ymin + np.arange(ny, dtype=np.float64) * self.grid_resolution
        return xs, ys

    def _xy_to_grid(self, xy: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[int, int] | None:
        xmin, xmax, ymin, ymax = self.arena_bounds
        x, y = float(xy[0]), float(xy[1])
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return None
        ix = int(np.clip(round((x - xmin) / self.grid_resolution), 0, len(xs) - 1))
        iy = int(np.clip(round((y - ymin) / self.grid_resolution), 0, len(ys) - 1))
        return ix, iy

    def _dijkstra_distance(
        self, occupied: np.ndarray, start_idx: tuple[int, int], goal_idx: tuple[int, int]
    ) -> float | None:
        width, height = occupied.shape
        distances = np.full_like(occupied, np.inf, dtype=np.float64)
        distances[start_idx] = 0.0
        queue: list[tuple[float, tuple[int, int]]] = [(0.0, start_idx)]
        neighbors = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, np.sqrt(2.0)),
            (-1, 1, np.sqrt(2.0)),
            (1, -1, np.sqrt(2.0)),
            (1, 1, np.sqrt(2.0)),
        )

        while queue:
            distance, (ix, iy) = heappop(queue)
            if (ix, iy) == goal_idx:
                return float(distance)
            if distance > distances[ix, iy]:
                continue
            for dx, dy, multiplier in neighbors:
                nx = ix + dx
                ny = iy + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if occupied[nx, ny]:
                    continue
                step = self.grid_resolution * float(multiplier)
                new_distance = distance + step
                if new_distance < distances[nx, ny]:
                    distances[nx, ny] = new_distance
                    heappush(queue, (new_distance, (nx, ny)))
        return None


if __name__ == "__main__":
    generator = StaticHardLayoutGenerator()
    for seed in range(10):
        layout = generator.generate(seed)
        print(
            f"seed={layout.seed:02d} template={layout.template_name} "
            f"geodesic={layout.geodesic_distance:.2f}m "
            f"walls={len(layout.walls)} boxes={len(layout.boxes)} cylinders={len(layout.cylinders)}"
        )
