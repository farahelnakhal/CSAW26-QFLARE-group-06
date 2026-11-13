def attack(global_params, own_deltas, round_idx, state):
    """Coordinated Stall with the original inverse-log timing.

    Oppose the arithmetic mean of the last three observed shared-model
    circuit movements. Use available history during rounds 1 and 2.
    Both clients submit the SAME vector; its length is capped by the
    stricter client's perturbation bound without rotating its direction.

    Return (replacement_circuit_deltas, state), with output shape (2, 24).
    Call once per round, 0-59; pass state=None at each new evaluation run.
    Requires NumPy. Classical-head updates remain under honest evaluation.
    """
    import numpy as np

    history_window = 3
    eps = 1e-12
    own_deltas = np.asarray(own_deltas)
    global_params = np.asarray(global_params)
    if (own_deltas.ndim != 2 or own_deltas.shape[0] != 2
            or own_deltas.shape[1] < 24):
        raise ValueError("own_deltas must have shape (2, n), n >= 24")
    if global_params.ndim != 1 or global_params.size < 24:
        raise ValueError("global_params must be a flat vector of length >= 24")
    if np.iscomplexobj(own_deltas) or np.iscomplexobj(global_params):
        raise ValueError("Parameters and deltas must be real")
    honest = np.array(own_deltas[:, :24], dtype=float, copy=True)
    current = np.array(global_params[:24], dtype=float, copy=True)
    if not np.all(np.isfinite(honest)) or not np.all(np.isfinite(current)):
        raise ValueError("Circuit parameters and deltas must be finite")

    if state is None:
        # Original inverse-log schedule, unchanged: 30 slots in rounds 1-59.
        base = np.random.default_rng(2026).uniform(2.0, 10.0)
        k = np.arange(30)
        positions = np.expm1(np.log(base) * k / 29) / (base - 1)
        rounds = 1 + k + np.floor(29 * positions).astype(int)
        rounds[0], rounds[-1] = 1, 59
        state = {
            "rounds": set(rounds.tolist()),
            "previous_params": None,
            "previous_round": None,
            "movements": [],
            "attack_count": 0,
        }

    previous = state["previous_params"]
    previous_round = state["previous_round"]
    if previous is not None and previous_round == round_idx - 1:
        movement = current - previous
        if not np.all(np.isfinite(movement)):
            raise ValueError("Non-finite shared-model movement")
        state["movements"].append(movement)
        state["movements"] = state["movements"][-history_window:]
    else:
        # A gap cannot be interpreted as one ordinary global update.
        state["movements"] = []

    # Update history on EVERY call, including non-attack and post-budget calls.
    # This is observed global movement (including past attack effects), not
    # a reconstruction of the six other clients' hidden honest updates.
    state["previous_params"] = current.copy()
    state["previous_round"] = round_idx
    if (round_idx not in state["rounds"] or state["attack_count"] >= 30
            or not state["movements"]):
        return honest, state

    # Average raw movements, not normalized directions; retain zero movements.
    past_average = np.mean(state["movements"], axis=0)
    target_length = np.hypot.reduce(past_average)
    if not np.isfinite(target_length):
        raise ValueError("Non-finite historical average")
    if target_length <= eps:
        # No reliable shared direction. Do not move this opportunity elsewhere.
        return honest, state
    unit = past_average / target_length

    magnitudes = np.hypot.reduce(honest, axis=1)
    radii = 2.0 * np.maximum(magnitudes, 0.01)
    if not np.all(np.isfinite(radii)):
        raise ValueError("Non-finite perturbation radii")

    # Both replacements have the form a_i = -length * unit.
    # Feasibility: length^2 + 2*length*(d_i @ unit) + ||d_i||^2 <= R_i^2.
    # The positive root is the greatest allowed length along this shared ray.
    # Normalize by R_i before squaring for numerical stability.
    scaled_honest = honest / radii[:, None]
    alignment = scaled_honest @ unit
    relative_magnitude = magnitudes / radii
    maximum_lengths = radii * (
        -alignment + np.sqrt(np.maximum(
            alignment**2 + 1.0 - relative_magnitude**2, 0.0)))
    common_limit = float(np.min(maximum_lengths))
    # A small inward margin protects boundary feasibility against roundoff.
    length = min(float(target_length), common_limit * (1.0 - 1e-12))
    common_update = -length * unit
    applied = np.repeat(common_update[None, :], 2, axis=0)

    # Do not project the two vectors separately: that could rotate them apart.
    # The shared length already satisfies BOTH honest-centered radius bounds.
    if not np.all(np.isfinite(applied)):
        raise ValueError("Non-finite replacement circuit deltas")
    if not np.array_equal(applied, honest):
        state["attack_count"] += 1
    return applied, state

