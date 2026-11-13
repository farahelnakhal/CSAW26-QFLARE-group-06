def attack(global_params, own_deltas, round_idx, state):
    """Adaptive casewise / max-Stall / orthogonal attack.

    Dynamic timing only. Returns (updates, state), updates.shape == (2, 24).
    Call once per round, 0..59, starting with state=None.
    Round 0 is honest; at most 30 rounds are changed.

    Optional initial state:
        {"calibration_magnitudes": [...],
         "fedavg_weights": [0.125, 0.125]}

    History subtraction is exact only for linear FedAvg with correct weights.
    For other aggregators, recovered history and strategy feedback are proxies,
    not estimates of the evaluation score.
    """
    import numpy as np

    def norm(x):
        return np.hypot.reduce(np.abs(x), axis=-1)

    def unit(x):
        scale = float(np.max(np.abs(x)))
        if scale == 0.0:
            return np.zeros_like(x)
        scaled = x / scale
        return scaled / float(norm(scaled))

    raw_d = np.asarray(own_deltas)
    raw_q = np.asarray(global_params)

    if raw_d.ndim != 2 or raw_d.shape[0] != 2 or raw_d.shape[1] < 24:
        raise ValueError("own_deltas must have shape (2, n), n >= 24")
    if raw_q.ndim != 1 or raw_q.size < 24:
        raise ValueError("global_params must have >= 24 entries")
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

    if "adaptive_hybrid_version" not in state:
        if "attack_count" in state or "previous_round" in state:
            raise ValueError("Start this strategy with state=None")

        calibration = np.asarray(state.get("calibration_magnitudes", []))
        if np.iscomplexobj(calibration):
            raise ValueError("Calibration magnitudes must be real")
        calibration = np.asarray(calibration, dtype=float)
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
                if largest > 0.0 else 0.0
            )

        state.update({
            "adaptive_hybrid_version": 1,
            "attack_count": 0,
            "threshold": max(1e-12, threshold),
            "magnitude_history": [],
            "corrected_history": [],
            "previous_round": None,
            "previous_params": None,
            "previous_applied": None,
            "previous_weights": None,
            "pending_feedback": None,
            "mode_trials": [0, 0, 0],
            "mode_observations": [0, 0, 0],
            "mode_scores": [0.0, 0.0, 0.0],
            "last_orthogonal_direction": None,
            "last_max_direction": None,
        })
    elif state["adaptive_hybrid_version"] != 1:
        raise ValueError("Incompatible state; start with state=None")

    raw_weights = np.asarray(state.get("fedavg_weights", [0.125, 0.125]))
    if np.iscomplexobj(raw_weights):
        raise ValueError("fedavg_weights must be real")
    weights = np.asarray(raw_weights, dtype=float)
    if (
        weights.shape != (2,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or float(weights.sum()) > 1.0
    ):
        raise ValueError("fedavg_weights must be nonnegative with sum <= 1")

    trials = np.array(state["mode_trials"], dtype=int, copy=True)
    observations = np.array(state["mode_observations"], dtype=int, copy=True)
    scores = np.array(state["mode_scores"], dtype=float, copy=True)
    history = list(state["corrected_history"])

    previous_round = state["previous_round"]
    if previous_round is not None and t <= previous_round:
        raise ValueError("Rounds must advance; reset state for each test run")
    consecutive = previous_round is not None and t == previous_round + 1

    if consecutive:
        with np.errstate(over="ignore", invalid="ignore"):
            movement = current - state["previous_params"]
            corrected = movement - (
                state["previous_weights"] @ state["previous_applied"]
            )
        if not np.isfinite(corrected).all():
            raise ValueError("Corrected model change exceeds numeric range")

        pending = state["pending_feedback"]
        if pending is not None:
            # Noisy feedback: lost forward progress plus sideways displacement.
            with np.errstate(over="ignore", invalid="ignore"):
                error = (
                    movement - pending["baseline"]
                ) / pending["capacity"]

            if np.isfinite(error).all():
                forward = float(error @ pending["reference"])
                sideways = float(norm(
                    error - forward * pending["reference"]
                ))
                reward = float(np.clip(
                    -forward + 0.5 * sideways, -1.0, 1.0
                ))
                mode = pending["mode"]
                rate = 1.0 if observations[mode] == 0 else 0.3
                scores[mode] += rate * (reward - scores[mode])
                observations[mode] += 1

        history = (history + [corrected.copy()])[-3:]

    elif previous_round is not None:
        # Do not assign a multi-round movement to one attack.
        history = []
        state["magnitude_history"] = []

    state["pending_feedback"] = None
    state["corrected_history"] = history

    with np.errstate(over="ignore", invalid="ignore"):
        magnitudes = norm(honest)
        radii = 2.0 * np.maximum(magnitudes, 0.01)
    if not np.isfinite(radii).all():
        raise ValueError("Update magnitudes exceed numeric range")

    magnitude = float(0.5 * magnitudes[0] + 0.5 * magnitudes[1])
    threshold = float(state["threshold"])
    remaining = 30 - int(state["attack_count"])

    eligible = 1 <= t < 60 and remaining > 0
    forced = bool(eligible and remaining >= 60 - t)
    attack_now = bool(eligible and (magnitude >= threshold or forced))

    applied = honest.copy()
    selected_name = "honest"

    if attack_now:
        guard = 1.0 - 64.0 * np.finfo(float).eps
        safe_radii = radii * guard
        pair_mean = 0.5 * honest[0] + 0.5 * honest[1]

        if history:
            baseline = np.sum(
                np.asarray(history) / len(history), axis=0
            )
            baseline = baseline + weights @ honest
        else:
            baseline = pair_mean.copy()

        if not np.isfinite(baseline).all():
            raise ValueError("Estimated model movement exceeds numeric range")

        reference = unit(baseline)
        if not np.any(reference):
            reference = unit(pair_mean)

        def project(proposal):
            difference = proposal - honest
            distances = norm(difference)
            result = proposal.copy()

            for i in range(2):
                if distances[i] > radii[i]:
                    result[i] = (
                        honest[i] + safe_radii[i] * unit(difference[i])
                    )

            if (
                not np.isfinite(result).all()
                or np.any(norm(result - honest) > radii)
            ):
                raise ArithmeticError("Unable to construct feasible updates")
            return result

        def casewise():
            # Original casewise actions; timing is handled separately.
            if np.any(magnitudes <= 1e-12):
                return honest.copy()

            rng = np.random.default_rng(2027 + t)
            direction_threshold = float(rng.choice([0.0, 0.25, 0.5]))
            magnitude_threshold = float(rng.choice([0.1, 0.25, 0.5]))
            size_threshold = float(rng.choice([0.01, 0.025, 0.05]))
            reverse_both = int(rng.integers(2))
            selected_client = int(rng.integers(2))
            action = int(rng.integers(3))

            units = honest / magnitudes[:, None]
            agree = float(units[0] @ units[1]) >= direction_threshold
            similar = (
                abs(magnitudes[0] - magnitudes[1]) / magnitudes.max()
                <= magnitude_threshold
            )
            large = magnitude >= size_threshold
            h = unit(history[-1]) if history else np.zeros(24)

            if not agree and not similar:
                return honest.copy()

            if not agree:
                proposal = np.zeros_like(honest)
                if large:
                    i = int(np.argmin(magnitudes))
                    if magnitudes[0] == magnitudes[1] and np.any(h):
                        i = int(np.argmin(units @ h))
                    proposal[i] = 0.5 * honest[i]
                else:
                    i = selected_client
                    proposal[i] = (
                        -(radii[i] - magnitudes[i]) * units[i]
                    )
                return proposal

            if large:
                if similar or not reverse_both:
                    return np.zeros_like(honest)
                return -honest

            if action == 0 or not np.any(h):
                return -(radii - magnitudes)[:, None] * units

            if action == 1:
                return honest - radii[:, None] * h

            alignment = (honest @ h) / radii
            lengths = radii * (
                -alignment
                + np.sqrt(np.maximum(
                    alignment * alignment
                    + 1.0
                    - (magnitudes / radii) ** 2,
                    0.0,
                ))
            )
            return -lengths[:, None] * h

        # Max-Stall: both perturbations oppose the honest sum.
        max_direction = -unit(pair_mean)
        if not np.any(max_direction):
            max_direction = -reference
        if not np.any(max_direction):
            previous_direction = state["last_max_direction"]
            max_direction = (
                np.eye(24)[0]
                if previous_direction is None
                else np.array(previous_direction, copy=True)
            )

        max_lengths = safe_radii.copy()
        max_proposal = honest + max_lengths[:, None] * max_direction
        if np.array_equal(max_proposal[0], max_proposal[1]):
            max_lengths[1] *= 1.0 - 1e-6
            max_proposal = honest + max_lengths[:, None] * max_direction

        # Orthogonal: both submitted updates share a direction perpendicular
        # to the honest updates and up to three corrected model changes.
        columns = [
            unit(v)
            for v in [honest[0], honest[1]] + history
            if np.any(v)
        ]
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

        old_direction = state["last_orthogonal_direction"]
        orth_direction = np.zeros(24)

        if old_direction is not None:
            old_direction = np.asarray(old_direction)
            persisting = (
                not history
                or float(unit(history[-1]) @ old_direction) >= -0.1
            )
            if persisting:
                orth_direction = (
                    old_direction - basis @ (basis.T @ old_direction)
                )
                orth_direction -= basis @ (basis.T @ orth_direction)

        if float(norm(orth_direction)) <= 1e-10:
            index = int(np.argmax(
                1.0 - np.sum(basis * basis, axis=1)
            ))
            orth_direction = np.eye(24)[index] - basis @ basis[index]
            orth_direction -= basis @ (basis.T @ orth_direction)

        orth_direction = unit(orth_direction)
        rho = safe_radii * np.sqrt(np.maximum(
            1.0 - (magnitudes / safe_radii) ** 2, 0.0
        ))
        orth_proposal = rho[:, None] * orth_direction

        candidates = [
            project(casewise()),
            project(max_proposal),
            project(orth_proposal),
        ]
        available = [
            i for i, candidate in enumerate(candidates)
            if not np.array_equal(candidate, honest)
        ]
        untried = [i for i in available if trials[i] == 0]

        if untried:
            selected = untried[0]
        else:
            # Exploration selects HOW; it does not schedule attack rounds.
            bonuses = 0.4 * np.sqrt(
                np.log(1.0 + trials.sum()) / np.maximum(trials, 1)
            )
            selected = max(
                available, key=lambda i: scores[i] + bonuses[i]
            )

        applied = candidates[selected]
        selected_name = (
            "casewise", "max_stall", "orthogonal"
        )[selected]

        trials[selected] += 1
        state["attack_count"] += 1

        if selected == 1:
            state["last_max_direction"] = max_direction.copy()
        if selected == 2:
            state["last_orthogonal_direction"] = orth_direction.copy()

        capacity = float(weights @ radii)
        if capacity > 1e-12:
            state["pending_feedback"] = {
                "mode": selected,
                "baseline": baseline.copy(),
                "reference": reference.copy(),
                "capacity": capacity,
            }

    # Adapt AFTER deciding. Include skipped decision-round observations.
    if 1 <= t < 60:
        recent = (
            list(state["magnitude_history"]) + [magnitude]
        )[-3:]
        state["magnitude_history"] = recent

        if len(recent) == 3:
            ordered = sorted(recent)
            next_threshold = threshold

            if sum(value > threshold for value in ordered) >= 2:
                next_threshold = ordered[1]
            elif ordered[2] < threshold:
                next_threshold = 0.5 * ordered[1] + 0.5 * ordered[2]

            state["threshold"] = max(1e-12, next_threshold)

    state["mode_trials"] = trials.tolist()
    state["mode_observations"] = observations.tolist()
    state["mode_scores"] = scores.tolist()
    state["previous_params"] = current.copy()
    state["previous_applied"] = applied.copy()
    state["previous_weights"] = weights.copy()
    state["previous_round"] = t
    state["last_decision"] = {
        "round": t,
        "attack": not np.array_equal(applied, honest),
        "forced": forced,
        "strategy": selected_name,
        "magnitude": magnitude,
        "threshold_used": threshold,
    }

    return applied, state
