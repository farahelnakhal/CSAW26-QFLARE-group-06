def attack(global_params, own_deltas, round_idx, state):
    """Stall: alternate rounds 2,4,...,58; negate previous honest updates.

    Return (two 24-entry replacement circuit deltas, state).
    Pass state=None for each new evaluation run; call once per round, 0-59.
    global_params is retained for interface compatibility but is not needed.
    The classical head is left to the evaluator. Requires NumPy.
    """
    import numpy as np

    own_deltas = np.asarray(own_deltas)
    if (own_deltas.ndim != 2 or own_deltas.shape[0] != 2
            or own_deltas.shape[1] < 24):
        raise ValueError("own_deltas must have shape (2, n) with n >= 24")
    if np.iscomplexobj(own_deltas):
        raise ValueError("own_deltas must be real")
    honest = np.array(own_deltas[:, :24], dtype=float, copy=True)
    if not np.all(np.isfinite(honest)):
        raise ValueError("Honest circuit deltas must be finite")

    if state is None:
        state = {"previous_deltas": None, "previous_round": None,
                 "attack_count": 0}

    previous = state["previous_deltas"]
    previous_round = state["previous_round"]
    # Save honest history even on startup, skipped, or exhausted-budget rounds.
    state["previous_deltas"] = honest.copy()
    state["previous_round"] = round_idx

    # Alternate attack/recovery from t=2: 2,4,...,58 (29 opportunities).
    if (round_idx < 2 or round_idx >= 60 or round_idx % 2 != 0
            or state["attack_count"] >= 30 or previous is None
            or previous_round != round_idx - 1):
        return honest, state

    magnitudes = np.linalg.norm(honest, axis=1)
    radii = 2.0 * np.maximum(magnitudes, 0.01)
    # Oppose each client's immediately previous HONEST local update.
    # A zero current update is valid; there is no cosine to compute here.
    proposals = -previous

    # Project only infeasible targets, preserving feasible targets exactly.
    # Scaled norms avoid squaring large perturbation coordinates directly.
    perturbation = proposals - honest
    scale = np.max(np.abs(perturbation), axis=1)
    direction = perturbation / np.maximum(scale, 1e-300)[:, None]
    norm = np.linalg.norm(direction, axis=1)
    allowed = radii / np.maximum(norm, 1e-300)
    outside = scale > allowed
    applied = proposals.copy()
    applied[outside] = (honest[outside]
                        + direction[outside] * allowed[outside, None])
    if not np.all(np.isfinite(applied)):
        raise ValueError("Non-finite replacement circuit deltas")
    # A changed pair consumes ONE round, after the radius projection.
    if not np.array_equal(applied, honest):
        state["attack_count"] += 1
    return applied, state

