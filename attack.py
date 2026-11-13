def attack(global_params, own_deltas, round_idx, state):
    """Client 7: original Stall. Client 8: full-radius orthogonal update.

    Returns (updates, state), with updates.shape == (2, 24).
    Start each run with state=None or {}; call once per round, 0..59.
    Round 0 is honest. Dynamic timing permits at most 30 changed rounds.

    Optional initial state:
        {"calibration_magnitudes": [clean_run_pair_magnitudes],
         "fedavg_weights": [0.125, 0.125]}

    Client 8's submitted update is perpendicular to both current honest
    updates and up to three corrected past model changes. History subtracts
    BOTH clients' entire previously applied weighted updates. This recovery
    is exact for linear FedAvg with correct weights; otherwise it is a proxy.
    Client 7 may remain unchanged if its original Stall target is a no-op.
    """
    import numpy as np

    def norm(values):
        return np.hypot.reduce(np.abs(values), axis=-1)

    raw_d = np.asarray(own_deltas)
    raw_q = np.asarray(global_params)

    if raw_d.ndim != 2 or raw_d.shape[0] != 2 or raw_d.shape[1] < 24:
        raise ValueError("own_deltas must have shape (2, n), n >= 24")

    if raw_q.ndim != 1 or raw_q.size < 24:
        raise ValueError("global_params must be a vector with >= 24 entries")

    if np.iscomplexobj(raw_d) or np.iscomplexobj(raw_q):
        raise ValueError("Parameters and updates must be real")

    honest = np.array(raw_d[:, :24], dtype=float, copy=True)
    current = np.array(raw_q[:24], dtype=float, copy=True)

    if not (np.isfinite(honest).all() and np.isfinite(current).all()):
        raise ValueError("Circuit parameters and updates must be finite")

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

    if "stall_orthogonal_version" not in state:
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
            raise ValueError("Calibration must be finite and nonnegative")

        threshold = 0.025

        if calibration.size:
            largest = float(calibration.max())
            threshold = (
                largest * float(np.mean(calibration / largest))
                if largest > 0.0
                else 0.0
            )

        state.update({
            "stall_orthogonal_version": 1,
            "attack_count": 0,
            "threshold": max(1e-12, threshold),
            "magnitude_history": [],
            "corrected_history": [],
            "previous_round": None,
            "previous_params": None,
            "previous_honest": None,
            "previous_applied": None,
            "previous_weights": None,
        })

    elif state["stall_orthogonal_version"] != 1:
        raise ValueError("Incompatible state; start with state=None")

    raw_weights = np.asarray(
        state.get("fedavg_weights", [0.125, 0.125])
    )

    if np.iscomplexobj(raw_weights):
        raise ValueError("fedavg_weights must be real")

    weights = np.asarray(raw_weights, dtype=float)

    if (
        weights.shape != (2,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or float(weights.sum()) > 1.0
    ):
        raise ValueError(
            "fedavg_weights must be nonnegative with sum <= 1"
        )

    previous_round = state["previous_round"]

    if previous_round is not None and t <= previous_round:
        raise ValueError(
            "Rounds must advance; reset state for each test run"
        )

    history_ready = (
        previous_round is not None
        and t == previous_round + 1
    )
    history = list(state["corrected_history"])

    if history_ready:
        with np.errstate(over="ignore", invalid="ignore"):
            movement = current - state["previous_params"]
            own_contribution = (
                state["previous_weights"] @ state["previous_applied"]
            )
            corrected = movement - own_contribution

        if not np.isfinite(corrected).all():
            raise ValueError(
                "Corrected model change exceeds numeric range"
            )

        history = (history + [corrected.copy()])[-3:]

    elif previous_round is not None:
        history = []
        state["magnitude_history"] = []

    state["corrected_history"] = history

    with np.errstate(over="ignore", invalid="ignore"):
        magnitudes = norm(honest)
        radii = 2.0 * np.maximum(magnitudes, 0.01)

    if not np.isfinite(radii).all():
        raise ValueError("Update magnitudes exceed numeric range")

    magnitude = float(
        0.5 * magnitudes[0] + 0.5 * magnitudes[1]
    )
    threshold = float(state["threshold"])
    remaining = 30 - state["attack_count"]

    eligible = (
        1 <= t < 60
        and history_ready
        and remaining > 0
    )
    forced = bool(eligible and remaining >= 60 - t)
    attack_now = bool(
        eligible and (magnitude >= threshold or forced)
    )

    applied = honest.copy()

    if attack_now:
        guard = 1.0 - 64.0 * np.finfo(float).eps

        # Client 7: original Stall.
        target = -state["previous_honest"][0]

        with np.errstate(over="ignore", invalid="ignore"):
            difference = target - honest[0]
            distance = float(norm(difference))

        if not np.isfinite(distance):
            raise ValueError(
                "Distance to the Stall target exceeds numeric range"
            )

        if distance <= radii[0]:
            applied[0] = target
        else:
            applied[0] = (
                honest[0]
                + (radii[0] * guard) * (difference / distance)
            )

        # Client 8: orthogonal to current honest updates and corrected history.
        columns = []

        for vector in [honest[0], honest[1]] + history:
            scale = float(np.max(np.abs(vector)))

            if scale > 0.0:
                scaled = vector / scale
                columns.append(scaled / float(norm(scaled)))

        if columns:
            matrix = np.column_stack(columns)
            u, singular_values, _ = np.linalg.svd(
                matrix, full_matrices=False
            )
            tolerance = (
                max(matrix.shape)
                * np.finfo(float).eps
                * singular_values[0]
            )
            basis = u[:, singular_values > tolerance]
        else:
            basis = np.empty((24, 0))

        # Choose the coordinate with the largest nullspace projection.
        index = int(
            np.argmax(1.0 - np.sum(basis * basis, axis=1))
        )
        direction = np.eye(24)[index] - basis @ basis[index]

        # Reproject to reduce floating-point residuals.
        direction -= basis @ (basis.T @ direction)
        length = float(norm(direction))

        if not np.isfinite(length) or length <= 1e-12:
            raise ArithmeticError(
                "Could not construct an orthogonal direction"
            )

        direction /= length

        # a8 = rho*v; delta8 = -d8 + rho*v.
        # The perturbation uses the full permitted radius to roundoff.
        safe_radius = radii[1] * guard
        ratio = float(magnitudes[1] / safe_radius)
        rho = safe_radius * np.sqrt(
            max(0.0, 1.0 - ratio * ratio)
        )
        applied[1] = rho * direction

        if (
            not np.isfinite(applied).all()
            or np.any(norm(applied - honest) > radii)
        ):
            raise ArithmeticError(
                "Could not construct updates within both radii"
            )

        if np.array_equal(applied[1], honest[1]):
            raise ArithmeticError(
                "The orthogonal client must change"
            )

        # Count the round once, regardless of how many clients changed.
        state["attack_count"] += 1

    # Update timing AFTER deciding, including skipped rounds.
    if 1 <= t < 60:
        recent = (
            state["magnitude_history"] + [magnitude]
        )[-3:]
        state["magnitude_history"] = recent

        if len(recent) == 3:
            ordered = sorted(recent)

            if sum(value > threshold for value in ordered) >= 2:
                threshold_next = ordered[1]
            elif ordered[2] < threshold:
                threshold_next = (
                    0.5 * ordered[1] + 0.5 * ordered[2]
                )
            else:
                threshold_next = threshold

            state["threshold"] = max(1e-12, threshold_next)

    # Honest history feeds Stall; applied history removes our contribution.
    state["previous_params"] = current.copy()
    state["previous_honest"] = honest.copy()
    state["previous_applied"] = applied.copy()
    state["previous_weights"] = weights.copy()
    state["previous_round"] = t

    state["last_decision"] = {
        "round": t,
        "attack": attack_now,
        "forced": forced,
        "magnitude": magnitude,
        "threshold_used": threshold,
    }

    return applied, state
