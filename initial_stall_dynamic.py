def attack(global_params, own_deltas, round_idx, state):
    """Original Stall with dynamic attack timing.

    Target each client's previous HONEST update with the opposite sign:
        target[i] = -previous_honest[i]

    Return (replacement_updates, state), with shape (2, 24).
    Call once per round, 0..59; start each run with state=None or {}.

    Optional initial state:
        {"calibration_magnitudes": [clean_run_M_values]}

    Round 0 is honest. At most 30 rounds may actually change.
    global_params is retained for template compatibility but is not needed.
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

    if (
        isinstance(round_idx, (bool, np.bool_))
        or not isinstance(round_idx, (int, np.integer))
        or round_idx < 0
    ):
        raise ValueError("round_idx must be a nonnegative integer")

    t = int(round_idx)

    if state is None:
        state = {}

    if not isinstance(state, dict):
        raise ValueError("state must be None or the returned dictionary")

    state = dict(state)

    if "initial_stall_dynamic_version" not in state:
        if "attack_count" in state or "previous_round" in state:
            raise ValueError("Start this strategy with state=None")

        raw_calibration = np.asarray(
            state.get("calibration_magnitudes", [])
        )

        if np.iscomplexobj(raw_calibration):
            raise ValueError("Calibration magnitudes must be real")

        calibration = np.asarray(raw_calibration, dtype=float)

        if (
            calibration.ndim != 1
            or not np.isfinite(calibration).all()
            or np.any(calibration < 0)
        ):
            raise ValueError(
                "Calibration must be a finite, nonnegative vector"
            )

        threshold = 0.025

        if calibration.size:
            largest = float(calibration.max())
            threshold = (
                largest * float(np.mean(calibration / largest))
                if largest > 0.0
                else 0.0
            )

        state.update({
            "initial_stall_dynamic_version": 1,
            "attack_count": 0,
            "threshold": max(1e-12, threshold),
            "magnitude_history": [],
            "previous_deltas": None,
            "previous_round": None,
        })

    elif state["initial_stall_dynamic_version"] != 1:
        raise ValueError("Incompatible state; start with state=None")

    previous = state["previous_deltas"]
    previous_round = state["previous_round"]

    if previous_round is not None and t <= previous_round:
        raise ValueError(
            "Rounds must advance; reset state for each test run"
        )

    history_ready = (
        previous is not None
        and previous_round == t - 1
    )

    with np.errstate(over="ignore", invalid="ignore"):
        magnitudes = norm(honest)
        radii = 2.0 * np.maximum(magnitudes, 0.01)

    if not np.isfinite(radii).all():
        raise ValueError("Update magnitudes exceed numeric range")

    magnitude = float(
        0.5 * magnitudes[0] + 0.5 * magnitudes[1]
    )
    threshold = float(state["threshold"])
    remaining_attacks = 30 - state["attack_count"]

    eligible = (
        1 <= t < 60
        and history_ready
        and remaining_attacks > 0
    )

    forced = bool(
        eligible and remaining_attacks >= 60 - t
    )

    attempt = bool(
        eligible and (magnitude >= threshold or forced)
    )

    applied = honest.copy()
    changed = False

    if attempt:
        # Original Stall: oppose each client's previous HONEST update.
        targets = -previous

        with np.errstate(over="ignore", invalid="ignore"):
            perturbation = targets - honest
            distances = norm(perturbation)

        if not np.isfinite(distances).all():
            raise ValueError(
                "Distance to the Stall target exceeds numeric range"
            )

        # Preserve feasible targets exactly.
        applied = targets.copy()
        outside = distances > radii

        # Project only targets outside the permitted ball.
        if np.any(outside):
            directions = (
                perturbation[outside]
                / distances[outside, None]
            )
            safe_radii = radii[outside] * (
                1.0 - 64.0 * np.finfo(float).eps
            )
            applied[outside] = (
                honest[outside]
                + safe_radii[:, None] * directions
            )

        if (
            not np.isfinite(applied).all()
            or np.any(norm(applied - honest) > radii)
        ):
            raise ArithmeticError(
                "Could not project Stall targets within the radii"
            )

        # Count once per changed round, even if both clients change.
        changed = not np.array_equal(applied, honest)

        if changed:
            state["attack_count"] += 1
        else:
            applied = honest.copy()

    # Decide first, then adapt using the latest three decision rounds.
    # Skipped rounds and no-op attempts also enter this memory.
    if 1 <= t < 60:
        recent = (
            state["magnitude_history"] + [magnitude]
        )[-3:]
        state["magnitude_history"] = recent

        if len(recent) == 3:
            ordered = sorted(recent)

            # At least two above the threshold: raise to the median.
            if sum(value > threshold for value in ordered) >= 2:
                threshold_next = ordered[1]

            # All three below: average the two highest.
            elif ordered[2] < threshold:
                threshold_next = (
                    0.5 * ordered[1] + 0.5 * ordered[2]
                )

            else:
                threshold_next = threshold

            state["threshold"] = max(1e-12, threshold_next)

    # Always save HONEST updates, including skipped and post-budget rounds.
    state["previous_deltas"] = honest.copy()
    state["previous_round"] = t

    state["last_decision"] = {
        "round": t,
        "attempt": attempt,
        "changed": changed,
        "forced": forced,
        "history_ready": history_ready,
        "magnitude": magnitude,
        "threshold_used": threshold,
    }

    return applied, state
