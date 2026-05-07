from __future__ import annotations

from dataclasses import dataclass


TRAIN_LAYOUT_SEEDS = tuple(range(0, 1000))
TRAIN_DYNAMIC_SEEDS = tuple(range(0, 1000))

TEST_SEEN_LAYOUT_SEEDS = tuple(range(1000, 1100))
TEST_SEEN_DYNAMIC_SEEDS = tuple(range(1000, 1100))

TEST_UNSEEN_LAYOUT_SEEDS = tuple(range(2000, 2100))
TEST_UNSEEN_DYNAMIC_SEEDS = tuple(range(3000, 3100))


@dataclass(frozen=True)
class EvalSeedPair:
    split: str
    layout_seed: int
    dynamic_seed: int


def make_seed_pairs(split: str, episodes: int | None = None) -> list[EvalSeedPair]:
    """Return deterministic paired layout/dynamic seeds for evaluation."""

    if split == "train-smoke":
        layout_seeds = TRAIN_LAYOUT_SEEDS[:100]
        dynamic_seeds = TRAIN_DYNAMIC_SEEDS[:100]
    elif split == "seen":
        layout_seeds = TEST_SEEN_LAYOUT_SEEDS
        dynamic_seeds = TEST_SEEN_DYNAMIC_SEEDS
    elif split == "unseen-layout":
        layout_seeds = TEST_UNSEEN_LAYOUT_SEEDS
        dynamic_seeds = TEST_SEEN_DYNAMIC_SEEDS
    elif split == "unseen-dynamic":
        layout_seeds = TEST_SEEN_LAYOUT_SEEDS
        dynamic_seeds = TEST_UNSEEN_DYNAMIC_SEEDS
    elif split == "combo":
        layout_seeds = TEST_UNSEEN_LAYOUT_SEEDS
        dynamic_seeds = TEST_UNSEEN_DYNAMIC_SEEDS
    else:
        raise ValueError(
            f"Unknown split {split!r}; choose from train-smoke, seen, "
            "unseen-layout, unseen-dynamic, combo"
        )

    pairs = [
        EvalSeedPair(split=split, layout_seed=int(layout_seed), dynamic_seed=int(dynamic_seed))
        for layout_seed, dynamic_seed in zip(layout_seeds, dynamic_seeds)
    ]
    return pairs if episodes is None else pairs[:episodes]


def split_names() -> tuple[str, ...]:
    return ("train-smoke", "seen", "unseen-layout", "unseen-dynamic", "combo")
