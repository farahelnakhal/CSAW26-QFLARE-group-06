def attack(global_params, own_deltas, round_idx, state):
    """Coordinated stall max with dynamic three-attempt timing.

    Return (replacement_circuit_updates, state), with shape (2, 24).
    Call once per round, 0..59; start each test run with state=None or {}.
    Optional initial state: {"calibration_magnitudes": [clean-run M values]}.
    M is the mean of the two honest circuit-update norms. Without calibration,
    the initial threshold is 0.025. There is no logarithmic schedule or switch.

    Both added errors oppose the honest sum and use the full allowed radius,
    up to a floating-point guard. Identical candidate submissions reduce
    client 8's error magnitude by 1e-6 times its radius. A near-zero sum uses
    the last reliable direction, or the first coordinate if none exists.
    global_params is retained for template compatibility; it is not needed.
    """
    import numpy as np

    def norm(values):
        return np.hypot.reduce(np.abs(values), axis=-1)

    raw = np.asarray(own_deltas)
    if raw.ndim != 2 or raw.shape[0] != 2 or raw.shape[1] < 24:
        raise ValueError("own_deltas must have shape (2, n), n >= 24")
    if np.iscomplexobj(raw):
        raise ValueError("own_deltas must contain real values")
    honest = np.array(raw[:, :24], dtype=float, copy=True)
    if not np.isfinite(honest).all():
        raise ValueError("Circuit updates must be finite")
    if (isinstance(round_idx, (bool, np.bool_))
            or not isinstance(round_idx, (int, np.integer)) or round_idx < 0):
        raise ValueError("round_idx must be a nonnegative integer")
    t = int(round_idx)

    if state is None:
        state = {}
    if not isinstance(state, dict):
        raise ValueError("state must be None or the returned dictionary")
    state = dict(state)
    if "stall_max_version" not in state:
        if "attack_count" in state or "previous_round" in state:
            raise ValueError("Start this strategy with state=None")
        raw_calibration = np.asarray(state.get("calibration_magnitudes", []))
        if np.iscomplexobj(raw_calibration):
            raise ValueError("Calibration magnitudes must be real")
        calibration = np.asarray(raw_calibration, dtype=float)
        if (calibration.ndim != 1 or not np.isfinite(calibration).all()
                or np.any(calibration < 0)):
            raise ValueError("Calibration must be a finite, nonnegative vector")
        threshold = 0.025
        if calibration.size:
            largest = float(calibration.max())
            threshold = (largest * float(np.mean(calibration / largest))
                         if largest > 0.0 else 0.0)
        state.update({
            "stall_max_version": 1,
            "attack_count": 0,
            "threshold": max(1e-12, threshold),
            "magnitude_history": [],
            "last_direction": None,
            "previous_round": None,
        })
    elif state["stall_max_version"] != 1:
        raise ValueError("Incompatible state; start with state=None")

    previous_round = state["previous_round"]
    if previous_round is not None and t <= previous_round:
        raise ValueError("Rounds must advance; reset state for each test run")

    with np.errstate(over="ignore", invalid="ignore"):
        magnitudes = norm(honest)
        radii = 2.0 * np.maximum(magnitudes, 0.01)
    if not np.isfinite(radii).all():
        raise ValueError("Update magnitudes exceed the supported numeric range")
    magnitude = float(0.5 * magnitudes[0] + 0.5 * magnitudes[1])

    # Half the sum has the same direction and avoids overflowing d7 + d8.
    half_sum = 0.5 * honest[0] + 0.5 * honest[1]
    half_norm = float(norm(half_sum))
    if half_norm > 0.5e-12:
        direction = -half_sum / half_norm
        state["last_direction"] = direction.copy()
        direction_source = "current_sum"
    elif state["last_direction"] is not None:
        direction = state["last_direction"].copy()
        direction_source = "saved_direction"
    else:
        direction = np.zeros(24)
        direction[0] = 1.0
        direction_source = "first_coordinate"

    threshold = float(state["threshold"])
    remaining_attacks = 30 - state["attack_count"]
    eligible = 1 <= t < 60 and remaining_attacks > 0
    forced = bool(eligible and remaining_attacks >= 60 - t)
    attack_now = bool(eligible and (magnitude >= threshold or forced))
    applied = honest.copy()
    tie_adjusted = False

    if attack_now:
        initial = honest + radii[:, None] * direction
        tie_adjusted = np.array_equal(initial[0], initial[1])

        # This tiny guard prevents roundoff from exceeding either hard radius.
        lengths = radii * (1.0 - 64.0 * np.finfo(float).eps)
        applied = honest + lengths[:, None] * direction
        tie_adjusted = bool(tie_adjusted or np.array_equal(applied[0], applied[1]))
        if tie_adjusted:
            lengths[1] -= 1e-6 * radii[1]
            applied[1] = honest[1] + lengths[1] * direction

        if (not np.isfinite(applied).all()
                or np.any(norm(applied - honest) > radii)):
            raise ArithmeticError("Could not construct updates within both radii")
        if (np.any(np.all(applied == honest, axis=1))
                or np.array_equal(applied[0], applied[1])):
            raise ArithmeticError("Both clients must change with distinct submissions")
        state["attack_count"] += 1

    # Decide first, then adapt from every eligible-time attempt, including skips.
    # Round 0 and calls outside the 60-round run do not enter this memory.
    if 1 <= t < 60:
        history = (state["magnitude_history"] + [magnitude])[-3:]
        state["magnitude_history"] = history
        if len(history) == 3:
            ordered = sorted(history)
            if sum(value > threshold for value in ordered) >= 2:
                threshold_next = ordered[1]
            elif ordered[2] < threshold:
                threshold_next = 0.5 * ordered[1] + 0.5 * ordered[2]
            else:
                threshold_next = threshold
            state["threshold"] = max(1e-12, threshold_next)

    state["previous_round"] = t
    state["last_decision"] = {
        "round": t, "attack": attack_now, "forced": forced,
        "magnitude": magnitude, "threshold_used": threshold,
        "tie_adjusted": tie_adjusted, "direction_source": direction_source,
    }
    return applied, state

