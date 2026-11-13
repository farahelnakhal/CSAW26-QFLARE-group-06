"""One fixed max-stall method used by both bonus comparisons."""

import numpy as np


SCHEDULES = {
    "timing_first30": tuple(range(30)),
    "timing_last30": tuple(range(30, 60)),
    "timing_alternating30": tuple(range(0, 60, 2)),
    "budget_even10": tuple(int((i + 0.5) * 60 / 10) for i in range(10)),
    "budget_even20": tuple(int((i + 0.5) * 60 / 20) for i in range(20)),
    "budget_even30": tuple(int((i + 0.5) * 60 / 30) for i in range(30)),
}


def attack_for(rounds):
    """Return the same max-stall update with only its attack rounds changed."""
    selected = frozenset(rounds)

    def attack(global_params, own_deltas, round_idx, state):
        honest = np.asarray(own_deltas[:, :24], dtype=float)
        if round_idx not in selected:
            return honest.copy(), state

        pair_mean = 0.5 * honest[0] + 0.5 * honest[1]
        length = np.linalg.norm(pair_mean)
        if length:
            direction = -pair_mean / length
        else:
            direction = np.zeros(24)
            direction[0] = 1.0

        radii = 2.0 * np.maximum(np.linalg.norm(honest, axis=1), 0.01)
        applied = honest + (radii * (1.0 - 64.0 * np.finfo(float).eps))[:, None] * direction
        if np.array_equal(applied[0], applied[1]):
            applied[1] = honest[1] + radii[1] * (1.0 - 1e-6) * direction
        return applied, state

    return attack
