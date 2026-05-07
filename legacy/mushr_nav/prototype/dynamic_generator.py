from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from .layout_generator import LayoutSpec, StaticHardLayoutGenerator, _nearest_bin
except ImportError:
    from layout_generator import LayoutSpec, StaticHardLayoutGenerator, _nearest_bin


MAX_DYNAMIC_OBSTACLES = 8
DEFAULT_ARENA_BOUNDS = (-4.55, 4.55, -4.55, 4.55)
DYNAMIC_RADIUS_BINS = np.array([0.22, 0.25, 0.28, 0.31], dtype=np.float64)


@dataclass(frozen=True)
class DynamicObstacleSpec:
    """Description of one dynamic obstacle assigned to a dyn_obs_N XML slot."""

    name: str
    pattern: str
    trajectory_type: str
    center: np.ndarray
    radius: float
    size: np.ndarray
    z: float
    axis: np.ndarray
    amplitude: float
    speed: float
    speed_range: tuple[float, float]
    omega: float
    phase: float
    start_xy: np.ndarray
    end_xy: np.ndarray
    velocity: np.ndarray
    mode: str = "moving"
    trigger_xy: np.ndarray | None = None
    hold_time: float = 0.0
    note: str = ""

    def position_at(self, time: float) -> np.ndarray:
        """Return the obstacle xy position for simple standalone smoke checks."""
        if self.mode == "frozen" or self.trajectory_type == "fixed":
            return self.initial_xy.copy()
        if self.trajectory_type == "line":
            arg = self.omega * float(time) + self.phase
            return self.center + self.axis * self.amplitude * np.sin(arg)
        if self.trajectory_type == "circle":
            arg = self.omega * float(time) + self.phase
            return self.center + self.amplitude * np.array([np.cos(arg), np.sin(arg)], dtype=np.float64)
        if self.trajectory_type == "one-shot":
            return self.start_xy + self.velocity * max(0.0, float(time) - self.hold_time)
        raise ValueError(f"unknown trajectory_type {self.trajectory_type!r}")

    def velocity_at(self, time: float) -> np.ndarray:
        """Return the obstacle xy velocity for simple standalone smoke checks."""
        if self.mode == "frozen" or self.trajectory_type == "fixed":
            return np.zeros(2, dtype=np.float64)
        if self.trajectory_type == "line":
            arg = self.omega * float(time) + self.phase
            return self.axis * self.amplitude * self.omega * np.cos(arg)
        if self.trajectory_type == "circle":
            arg = self.omega * float(time) + self.phase
            return self.amplitude * self.omega * np.array([-np.sin(arg), np.cos(arg)], dtype=np.float64)
        if self.trajectory_type == "one-shot":
            return np.zeros(2, dtype=np.float64) if float(time) < self.hold_time else self.velocity.copy()
        raise ValueError(f"unknown trajectory_type {self.trajectory_type!r}")

    @property
    def initial_xy(self) -> np.ndarray:
        return self.position_at(0.0) if self.mode != "frozen" else self.start_xy.copy()

    def as_env_dict(self) -> dict:
        """Compatibility-oriented dictionary for future MuSHRNavEnv integration."""
        return {
            "name": self.name,
            "pattern": self.pattern,
            "trajectory_type": self.trajectory_type,
            "center": self.center.copy(),
            "axis": self.axis.copy(),
            "amplitude": float(self.amplitude),
            "speed": float(self.speed),
            "speed_range": tuple(float(v) for v in self.speed_range),
            "omega": float(self.omega),
            "phase": float(self.phase),
            "start_xy": self.start_xy.copy(),
            "end_xy": self.end_xy.copy(),
            "velocity": self.velocity.copy(),
            "radius": float(self.radius),
            "size": self.size.copy(),
            "z": float(self.z),
            "mode": self.mode,
            "trigger_xy": None if self.trigger_xy is None else self.trigger_xy.copy(),
            "hold_time": float(self.hold_time),
            "note": self.note,
        }


@dataclass(frozen=True)
class DynamicSceneSpec:
    """Generated dynamic scene tied to one static-hard LayoutSpec."""

    layout_seed: int
    template_name: str
    seed: int
    mode: str
    obstacles: tuple[DynamicObstacleSpec, ...]
    patterns: tuple[str, ...]

    def as_env_dicts(self) -> list[dict]:
        return [obstacle.as_env_dict() for obstacle in self.obstacles]


class DynamicHardObstacleGenerator:
    """Layout-aware first-pass generator for dynamic-hard MuSHR navigation."""

    PATTERNS = (
        "crossing-bottleneck",
        "intersection-crossing",
        "same-direction-blocking",
        "occluded-emergence",
        "goal-area-interference",
    )

    def __init__(
        self,
        arena_bounds: tuple[float, float, float, float] = DEFAULT_ARENA_BOUNDS,
        max_obstacles: int = MAX_DYNAMIC_OBSTACLES,
        radius_range: tuple[float, float] = (0.22, 0.31),
        random_phase: bool = True,
        random_speed: bool = True,
        random_amplitude: bool = True,
    ):
        self.arena_bounds = tuple(float(v) for v in arena_bounds)
        self.max_obstacles = int(max_obstacles)
        self.radius_range = tuple(float(v) for v in radius_range)
        self.random_phase = bool(random_phase)
        self.random_speed = bool(random_speed)
        self.random_amplitude = bool(random_amplitude)
        if self.max_obstacles < 1 or self.max_obstacles > MAX_DYNAMIC_OBSTACLES:
            raise ValueError(f"max_obstacles must be in [1, {MAX_DYNAMIC_OBSTACLES}]")

    def generate(self, layout: LayoutSpec, seed: int, mode: str = "moving") -> DynamicSceneSpec:
        if mode not in ("moving", "hard", "frozen"):
            raise ValueError(f"unknown dynamic obstacle mode {mode!r}")
        normalized_mode = "moving" if mode == "hard" else mode
        rng = np.random.default_rng(int(seed))

        builders = (
            self._crossing_bottleneck,
            self._intersection_crossing,
            self._same_direction_blocking,
            self._occluded_emergence,
            self._goal_area_interference,
        )
        obstacles = []
        for builder in builders:
            if len(obstacles) >= self.max_obstacles:
                break
            spec = builder(layout, rng, len(obstacles), normalized_mode)
            obstacles.append(spec)

        return DynamicSceneSpec(
            layout_seed=int(layout.seed),
            template_name=layout.template_name,
            seed=int(seed),
            mode=normalized_mode,
            obstacles=tuple(obstacles),
            patterns=tuple(obstacle.pattern for obstacle in obstacles),
        )

    def _crossing_bottleneck(
        self, layout: LayoutSpec, rng: np.random.Generator, index: int, mode: str
    ) -> DynamicObstacleSpec:
        if layout.template_name == "single-bottleneck":
            center = np.array([0.0, self._single_bottleneck_gap_y(layout)], dtype=np.float64)
            axis = np.array([1.0, 0.0], dtype=np.float64)
            amplitude = 0.78
        elif layout.template_name == "loop-with-shortcut":
            center = np.array([0.0, 0.35], dtype=np.float64)
            axis = np.array([1.0, 0.0], dtype=np.float64)
            amplitude = 0.72
        else:
            center = self._midpoint(layout.start_xy, layout.goal_xy)
            axis = self._perp(self._unit(layout.goal_xy - layout.start_xy))
            amplitude = 0.68
        return self._make_line(
            layout=layout,
            rng=rng,
            index=index,
            mode=mode,
            pattern="crossing-bottleneck",
            center=center,
            axis=axis,
            amplitude=amplitude,
            speed_range=(0.34, 0.58),
            phase=-0.5 * np.pi + self._phase(rng, -0.35, 0.35),
            note="Oscillates across the narrowest known passage or path midpoint.",
        )

    def _intersection_crossing(
        self, layout: LayoutSpec, rng: np.random.Generator, index: int, mode: str
    ) -> DynamicObstacleSpec:
        centers = {
            "t-junction": np.array([0.0, 0.8], dtype=np.float64),
            "clutter-room": np.array([0.0, 0.0], dtype=np.float64),
            "loop-with-shortcut": np.array([0.0, -1.95], dtype=np.float64),
        }
        center = centers.get(layout.template_name, self._midpoint(layout.start_xy, layout.goal_xy))
        path_axis = self._unit(layout.goal_xy - layout.start_xy)
        axis = self._perp(path_axis)
        return self._make_line(
            layout=layout,
            rng=rng,
            index=index,
            mode=mode,
            pattern="intersection-crossing",
            center=center,
            axis=axis,
            amplitude=0.82,
            speed_range=(0.30, 0.54),
            phase=self._phase(rng, 0.0, 2.0 * np.pi),
            note="Crosses an intersection-like waypoint transverse to the nominal route.",
        )

    def _same_direction_blocking(
        self, layout: LayoutSpec, rng: np.random.Generator, index: int, mode: str
    ) -> DynamicObstacleSpec:
        direction = self._unit(layout.goal_xy - layout.start_xy)
        center = layout.start_xy + 0.42 * (layout.goal_xy - layout.start_xy)
        center += self._perp(direction) * rng.uniform(-0.30, 0.30)
        return self._make_line(
            layout=layout,
            rng=rng,
            index=index,
            mode=mode,
            pattern="same-direction-blocking",
            center=center,
            axis=direction,
            amplitude=0.95,
            speed_range=(0.22, 0.42),
            phase=self._phase(rng, -0.20, 0.20),
            note="Moves along the nominal start-to-goal direction at a slower blocking speed.",
        )

    def _occluded_emergence(
        self, layout: LayoutSpec, rng: np.random.Generator, index: int, mode: str
    ) -> DynamicObstacleSpec:
        if layout.template_name == "occluded-corner":
            center = np.array([0.0, 0.75], dtype=np.float64)
            axis = self._unit(np.array([1.0, -1.0], dtype=np.float64))
            amplitude = 1.05
        elif layout.template_name == "t-junction":
            center = np.array([-1.15, -1.05], dtype=np.float64)
            axis = np.array([1.0, 0.0], dtype=np.float64)
            amplitude = 0.88
        else:
            direction = self._unit(layout.goal_xy - layout.start_xy)
            center = self._midpoint(layout.start_xy, layout.goal_xy) + self._perp(direction) * 0.75
            axis = -self._perp(direction)
            amplitude = 0.78
        return self._make_line(
            layout=layout,
            rng=rng,
            index=index,
            mode=mode,
            pattern="occluded-emergence",
            center=center,
            axis=axis,
            amplitude=amplitude,
            speed_range=(0.36, 0.66),
            phase=-0.5 * np.pi + self._phase(rng, -0.25, 0.25),
            note="Starts near a side passage or screen wall and emerges into the route.",
        )

    def _goal_area_interference(
        self, layout: LayoutSpec, rng: np.random.Generator, index: int, mode: str
    ) -> DynamicObstacleSpec:
        direction = self._unit(layout.goal_xy - layout.start_xy)
        center = layout.goal_xy - 0.75 * direction + self._perp(direction) * rng.uniform(-0.25, 0.25)
        radius = self._radius(rng)
        center = self._nudge_clear(layout, center, radius, preserve_goal_clearance=0.45)
        amplitude = self._amplitude(rng, 0.48)
        speed_range = (0.18, 0.34)
        speed = self._speed(rng, speed_range)
        omega = speed / max(amplitude, 1e-6)
        phase = self._phase(rng, 0.0, 2.0 * np.pi)
        initial = center + amplitude * np.array([np.cos(phase), np.sin(phase)], dtype=np.float64)
        velocity = amplitude * omega * np.array([-np.sin(phase), np.cos(phase)], dtype=np.float64)
        if mode == "frozen":
            velocity = np.zeros(2, dtype=np.float64)
        return DynamicObstacleSpec(
            name=f"dyn_obs_{index}",
            pattern="goal-area-interference",
            trajectory_type="circle",
            center=center,
            radius=radius,
            size=np.array([radius, 0.24, 0.01], dtype=np.float64),
            z=0.24,
            axis=np.array([1.0, 0.0], dtype=np.float64),
            amplitude=amplitude,
            speed=speed,
            speed_range=speed_range,
            omega=omega,
            phase=phase,
            start_xy=initial,
            end_xy=initial,
            velocity=velocity,
            mode=mode,
            trigger_xy=layout.goal_xy.copy(),
            note="Circles near the goal approach while preserving direct goal clearance.",
        )

    def _make_line(
        self,
        layout: LayoutSpec,
        rng: np.random.Generator,
        index: int,
        mode: str,
        pattern: str,
        center: np.ndarray,
        axis: np.ndarray,
        amplitude: float,
        speed_range: tuple[float, float],
        phase: float,
        note: str,
    ) -> DynamicObstacleSpec:
        radius = self._radius(rng)
        center = self._nudge_clear(layout, center, radius)
        axis = self._unit(axis)
        amplitude = self._amplitude(rng, amplitude)
        speed = self._speed(rng, speed_range)
        omega = speed / max(amplitude, 1e-6)
        start_xy = center - axis * amplitude
        end_xy = center + axis * amplitude
        initial = center + axis * amplitude * np.sin(phase)
        velocity = axis * amplitude * omega * np.cos(phase)
        if mode == "frozen":
            start_xy = initial
            end_xy = initial
            velocity = np.zeros(2, dtype=np.float64)
        return DynamicObstacleSpec(
            name=f"dyn_obs_{index}",
            pattern=pattern,
            trajectory_type="line",
            center=center,
            radius=radius,
            size=np.array([radius, 0.24, 0.01], dtype=np.float64),
            z=0.24,
            axis=axis,
            amplitude=amplitude,
            speed=speed,
            speed_range=tuple(float(v) for v in speed_range),
            omega=omega,
            phase=float(phase),
            start_xy=start_xy,
            end_xy=end_xy,
            velocity=velocity,
            mode=mode,
            trigger_xy=center.copy(),
            note=note,
        )

    def _radius(self, rng: np.random.Generator) -> float:
        sampled = float(rng.uniform(self.radius_range[0], self.radius_range[1]))
        valid_bins = DYNAMIC_RADIUS_BINS[
            (DYNAMIC_RADIUS_BINS >= self.radius_range[0] - 1e-9)
            & (DYNAMIC_RADIUS_BINS <= self.radius_range[1] + 1e-9)
        ]
        bins = valid_bins if len(valid_bins) else DYNAMIC_RADIUS_BINS
        return _nearest_bin(sampled, bins)

    def _phase(self, rng: np.random.Generator, low: float, high: float) -> float:
        return float(rng.uniform(low, high)) if self.random_phase else float(0.5 * (low + high))

    def _speed(self, rng: np.random.Generator, speed_range: tuple[float, float]) -> float:
        if self.random_speed:
            return float(rng.uniform(speed_range[0], speed_range[1]))
        return float(0.5 * (speed_range[0] + speed_range[1]))

    def _amplitude(self, rng: np.random.Generator, base: float) -> float:
        if self.random_amplitude:
            return float(base) * float(rng.uniform(0.9, 1.1))
        return float(base)

    def _nudge_clear(
        self,
        layout: LayoutSpec,
        point: np.ndarray,
        radius: float,
        preserve_goal_clearance: float = 0.0,
    ) -> np.ndarray:
        point = self._clamp(point)
        if self._is_clear(layout, point, radius, preserve_goal_clearance):
            return point

        offsets = [
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0.35, 0.0], dtype=np.float64),
            np.array([-0.35, 0.0], dtype=np.float64),
            np.array([0.0, 0.35], dtype=np.float64),
            np.array([0.0, -0.35], dtype=np.float64),
            np.array([0.55, 0.55], dtype=np.float64),
            np.array([-0.55, 0.55], dtype=np.float64),
            np.array([0.55, -0.55], dtype=np.float64),
            np.array([-0.55, -0.55], dtype=np.float64),
        ]
        for offset in offsets:
            candidate = self._clamp(point + offset)
            if self._is_clear(layout, candidate, radius, preserve_goal_clearance):
                return candidate
        return point

    def _is_clear(
        self,
        layout: LayoutSpec,
        point: np.ndarray,
        radius: float,
        preserve_goal_clearance: float,
    ) -> bool:
        if np.linalg.norm(point - layout.start_xy) < radius + 0.75:
            return False
        if preserve_goal_clearance > 0.0 and np.linalg.norm(point - layout.goal_xy) < radius + preserve_goal_clearance:
            return False
        for box in self._boxes(layout):
            half = box.size[:2] + radius + 0.12
            if np.all(np.abs(point - box.pos[:2]) <= half):
                return False
        for cylinder in layout.cylinders:
            if np.linalg.norm(point - cylinder.pos[:2]) <= radius + float(cylinder.size[0]) + 0.12:
                return False
        return True

    def _boxes(self, layout: LayoutSpec) -> Iterable:
        yield from layout.walls
        yield from layout.boxes

    def _clamp(self, point: np.ndarray) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.arena_bounds
        return np.array(
            [np.clip(float(point[0]), xmin, xmax), np.clip(float(point[1]), ymin, ymax)],
            dtype=np.float64,
        )

    def _single_bottleneck_gap_y(self, layout: LayoutSpec) -> float:
        split_walls = [wall for wall in layout.walls if wall.name in ("wall_4", "wall_5")]
        if len(split_walls) != 2:
            return float(self._midpoint(layout.start_xy, layout.goal_xy)[1])
        lower, upper = sorted(split_walls, key=lambda wall: wall.pos[1])
        lower_top = float(lower.pos[1] + lower.size[1])
        upper_bottom = float(upper.pos[1] - upper.size[1])
        return 0.5 * (lower_top + upper_bottom)

    def _midpoint(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return 0.5 * (np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64))

    def _unit(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            return np.array([1.0, 0.0], dtype=np.float64)
        return vector / norm

    def _perp(self, vector: np.ndarray) -> np.ndarray:
        vector = self._unit(vector)
        return np.array([-vector[1], vector[0]], dtype=np.float64)


def _smoke() -> None:
    layout_generator = StaticHardLayoutGenerator()
    dynamic_generator = DynamicHardObstacleGenerator()
    for seed, template_name in enumerate(StaticHardLayoutGenerator.TEMPLATE_NAMES):
        layout = layout_generator.generate(seed=100 + seed, template_name=template_name)
        scene = dynamic_generator.generate(layout, seed=2000 + seed, mode="moving")
        pattern_counts = {pattern: scene.patterns.count(pattern) for pattern in scene.patterns}
        summary = ", ".join(f"{pattern}={count}" for pattern, count in pattern_counts.items())
        print(
            f"layout_seed={layout.seed} template={layout.template_name} "
            f"dynamic_seed={scene.seed} obstacles={len(scene.obstacles)} {summary}"
        )

    frozen_layout = layout_generator.generate(seed=777, template_name="single-bottleneck")
    frozen_scene = dynamic_generator.generate(frozen_layout, seed=2777, mode="frozen")
    moving_velocities = [np.linalg.norm(obstacle.velocity_at(0.0)) for obstacle in frozen_scene.obstacles]
    print(
        f"layout_seed={frozen_layout.seed} template={frozen_layout.template_name} "
        f"mode=frozen obstacles={len(frozen_scene.obstacles)} "
        f"max_initial_speed={max(moving_velocities):.3f}"
    )


if __name__ == "__main__":
    _smoke()
