
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from sim_config import (
    ExperimentScenario,
    build_default_experiments,
    build_mechanism_specs,
    count_shelf_complexity,
)

STATE_NAMES = [
    "S0_Home_Primary",
    "S1_Drift_Adjacent",
    "S2_Dump_Eye_Drawer",
    "S3_Dump_Wire_Bin",
    "S4_Dump_Pegboard",
    "S5_Blocked_Or_Buried",
]


@dataclass
class MarkovAnalysisResult:
    """Analytical DTMC outputs and Monte Carlo validation metrics."""

    exp_id: str
    regime: str
    transition_matrix: List[List[float]]
    max_row_sum_error: float
    is_irreducible: bool
    period: int
    communication_classes: List[Tuple[List[int], str]]
    stationary_dist: List[float]
    stationary_residual: float
    spectral_gap: float
    mixing_time_95: float
    mfpt_matrix: List[List[float]]
    mfpt_home_to_any_misplaced: float
    mfpt_misplaced_to_home: float
    mfpt_home_to_buried: float
    sim_stationary_dist: List[float]
    total_variation_distance: float
    sim_mfpt_home_to_misplaced: float
    sim_mfpt_misplaced_to_home: float


def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Multiply two square/rectangular matrices."""
    n = len(a)
    m = len(b[0])
    k_len = len(b)
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        row_a = a[i]
        row_out = out[i]
        for k in range(k_len):
            aik = row_a[k]
            if aik == 0.0:
                continue
            row_b = b[k]
            for j in range(m):
                row_out[j] += aik * row_b[j]
    return out


def solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(a)
    aug = [a[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-14:
            raise ValueError("Singular matrix encountered in linear solver.")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        for r in range(col + 1, n):
            factor = aug[r][col] / pivot
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / aug[i][i]
    return x


def _tarjan_scc(adj: List[List[int]]) -> List[List[int]]:
    """Find strongly connected components using Tarjan's algorithm."""
    n = len(adj)
    index = 0
    indices = [-1] * n
    lowlink = [0] * n
    on_stack = [False] * n
    stack: List[int] = []
    sccs: List[List[int]] = []

    def strongconnect(v: int) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True

        for w in adj[v]:
            if indices[w] == -1:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            sccs.append(sorted(comp))

    for i in range(n):
        if indices[i] == -1:
            strongconnect(i)
    return sccs


def classify_markov_chain(
    p_mat: List[List[float]], tol: float = 1e-12
) -> Tuple[bool, int, List[Tuple[List[int], str]]]:
    """Classify communication classes, irreducibility, and periodicity."""
    n = len(p_mat)
    adj = [[j for j in range(n) if p_mat[i][j] > tol] for i in range(n)]
    sccs = _tarjan_scc(adj)

    comp_of = {}
    for cid, comp in enumerate(sccs):
        for node in comp:
            comp_of[node] = cid

    classes: List[Tuple[List[int], str]] = []
    for cid, comp in enumerate(sccs):
        has_outgoing = any(comp_of[j] != cid for i in comp for j in adj[i])
        if len(comp) == 1 and not has_outgoing and abs(p_mat[comp[0]][comp[0]] - 1.0) <= tol:
            kind = "absorbing"
        elif not has_outgoing:
            kind = "recurrent"
        else:
            kind = "transient"
        classes.append((comp, kind))

    is_irreducible = len(sccs) == 1

    period = 1
    recurrent_comp = next((c for c, k in classes if k in ("recurrent", "absorbing")), sccs[0])
    if len(recurrent_comp) > 1 or any(p_mat[u][u] <= tol for u in recurrent_comp):
        allowed = set(recurrent_comp)
        start = recurrent_comp[0]
        dist = {start: 0}
        queue = [start]
        g = 0
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if v not in allowed:
                    continue
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    cycle_diff = abs(dist[u] + 1 - dist[v])
                    if cycle_diff > 0:
                        g = math.gcd(g, cycle_diff)
        period = g if g > 0 else 1

    return is_irreducible, period, classes


def compute_stationary_distribution(p_mat: List[List[float]]) -> Tuple[List[float], float]:
    """Solve pi * P = pi, sum(pi) = 1 via linear system, returning (pi, max_residual)."""
    n = len(p_mat)
    sys_a = [[p_mat[j][i] - (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
    sys_b = [0.0] * n
    sys_a[n - 1] = [1.0] * n
    sys_b[n - 1] = 1.0

    pi = solve_linear_system(sys_a, sys_b)
    pi = [max(0.0, v) for v in pi]
    s = sum(pi)
    pi = [v / s for v in pi]

    pi_p = [sum(pi[i] * p_mat[i][j] for i in range(n)) for j in range(n)]
    residual = max(abs(pi_p[j] - pi[j]) for j in range(n))
    return pi, residual


def compute_second_eigenvalue_magnitude(
    p_mat: List[List[float]], pi: Sequence[float]
) -> float:
    """Compute |lambda_2| using Wielandt deflation P_def = P - 1 * pi^T."""
    n = len(p_mat)
    p_def = [[p_mat[i][j] - pi[j] for j in range(n)] for i in range(n)]
    v = [math.sin(i + 1.0) for i in range(n)]
    norm = math.sqrt(sum(x * x for x in v))
    v = [x / norm for x in v]

    lam2 = 0.0
    for _ in range(300):
        w1 = [sum(p_def[i][j] * v[j] for j in range(n)) for i in range(n)]
        n1 = math.sqrt(sum(x * x for x in w1))
        if n1 < 1e-14:
            return 0.0
        w2 = [sum(p_def[i][j] * w1[j] for j in range(n)) for i in range(n)]
        n2 = math.sqrt(sum(x * x for x in w2))
        lam2 = math.sqrt(n2)
        v = [x / max(1e-15, n2) for x in w2]
    return min(0.999999, lam2)


def compute_mfpt_matrix(
    p_mat: List[List[float]], pi: Sequence[float]
) -> List[List[float]]:
    """Compute exact n x n Mean First Passage Time matrix M."""
    n = len(p_mat)
    mfpt = [[0.0] * n for _ in range(n)]

    for target in range(n):
        mfpt[target][target] = 1.0 / pi[target] if pi[target] > 1e-14 else float("inf")
        others = [idx for idx in range(n) if idx != target]
        m = len(others)
        sys_a = [
            [(1.0 if r == c else 0.0) - p_mat[others[r]][others[c]] for c in range(m)]
            for r in range(m)
        ]
        sys_b = [1.0] * m
        sol = solve_linear_system(sys_a, sys_b)
        for idx, orig_i in enumerate(others):
            mfpt[orig_i][target] = sol[idx]
    return mfpt


def compute_absorption_time_from_set(
    p_mat: List[List[float]], transient_states: Sequence[int]
) -> Dict[int, float]:
    """Compute expected steps to exit a set of transient states via N = (I - Q_t)^-1."""
    t_list = list(transient_states)
    m = len(t_list)
    sys_a = [
        [(1.0 if r == c else 0.0) - p_mat[t_list[r]][t_list[c]] for c in range(m)]
        for r in range(m)
    ]
    sol = solve_linear_system(sys_a, [1.0] * m)
    return {t_list[idx]: sol[idx] for idx in range(m)}


def build_scenario_transition_matrix(
    scenario: ExperimentScenario, regime: str = "rush"
) -> List[List[float]]:
    """Construct the 6-state DTMC transition matrix P for a scenario and shift regime.

    States:
      0: Home_Primary
      1: Drift_Adjacent
      2: Dump_Eye_Drawer
      3: Dump_Wire_Bin
      4: Dump_Pegboard
      5: Blocked_Or_Buried
    """
    specs = build_mechanism_specs(scenario.topology)
    beh = scenario.behavior

    if regime == "rush":
        tau = 0.5 * (scenario.rush.rush_tau_range[0] + scenario.rush.rush_tau_range[1])
        pick_rate = 0.34
    else:
        tau = 0.5 * (scenario.rush.calm_tau_range[0] + scenario.rush.calm_tau_range[1])
        pick_rate = 0.16

    if scenario.topology == "M_modular" and scenario.row_configs is not None:
        r1, r2, r3 = scenario.row_configs
        mechs: List[str] = []
        weights: List[float] = []
        row_w = {1: 0.56 if scenario.overflow_mode == "hybrid_reserve" else 0.62, 2: 0.28, 3: 0.16}
        has_bulk_peg_mismatch = False
        for tier_idx, rcode in enumerate((r1, r2, r3), start=1):
            t_tag = {1: "eye", 2: "mid", 3: "3"}[tier_idx]
            w = row_w[tier_idx]
            if rcode in ("7D_4P", "7D_3P", "4D4P_3D3P"):
                mechs.append(f"split_drawer_{t_tag}")
                weights.append(w)
            elif rcode in ("1B_4D_4P", "1B_4D_3P", "1B_2D4P_2D3P"):
                mechs.append(f"split_drawer_{t_tag}")
                weights.append(w * 0.76)
                mechs.append(f"wire_bin_{t_tag}")
                weights.append(w * 0.24)
            elif rcode in ("1D_2B", "1B_1D_1B", "2B_1D"):
                mechs.append(f"split_drawer_{t_tag}")
                weights.append(w * 0.22)
                mechs.append(f"wire_bin_{t_tag}")
                weights.append(w * 0.78)
            elif rcode == "2B":
                mechs.append(f"wire_bin_{t_tag}")
                weights.append(w)
            elif rcode == "SLIDING_PEG":
                if scenario.overflow_mode == "all_colocated" or scenario.abc_colocation_mode == "cross_tier_height":
                    has_bulk_peg_mismatch = True
                mechs.append("pegboard_ovh")
                weights.append(w)
            else:
                raise ValueError(f"Unsupported modular row config: {rcode}")
        tot_w = sum(weights)
        tier_shares = {}
        for m, w in zip(mechs, weights):
            tier_shares[m] = tier_shares.get(m, 0.0) + w / max(1e-9, tot_w)
    else:
        raise ValueError(f"Unsupported topology or missing row_configs: {scenario.topology}")

    avg_ergo = sum(tier_shares[m] * specs[m].reach_actuation_cost for m in tier_shares)
    avg_vis_open = sum(tier_shares[m] * specs[m].visibility_open for m in tier_shares)
    has_pull_down = any(specs[m].is_pull_down for m in tier_shares)

    if scenario.abc_colocation_mode in ("same_slot", "adjacent_partition"):
        cross_tier_split = 0.08
    elif scenario.abc_colocation_mode == "same_tier_bay":
        cross_tier_split = 0.16
    else:
        cross_tier_split = 0.58

    if has_pull_down and regime == "rush":
        if scenario.abc_colocation_mode in ("adjacent_partition", "same_slot") and scenario.overflow_mode == "hybrid_reserve":
            occlusion_extra = 0.02 * tau
        else:
            occlusion_extra = 0.24 * tau
    else:
        occlusion_extra = 0.0

    phi_rush = 1.0 + beh.alpha_rush * (tau ** 2)
    phi_ergo = 1.0 + beh.alpha_ergo * avg_ergo * cross_tier_split + occlusion_extra
    bulk_peg_extra = 0.10 if has_bulk_peg_mismatch else 0.0
    crowd_excess = max(0.0, (scenario.target_fill_ratio + bulk_peg_extra - 0.75) / (1.05 - min(0.99, scenario.target_fill_ratio + 0.5 * bulk_peg_extra)))
    phi_crowd = 1.0 + beh.alpha_crowd * crowd_excess
    n_bins, extra_styles = count_shelf_complexity(scenario.row_configs)
    bin_dump_blur = beh.bin_dump_error_penalty * n_bins
    cog_load_blur = beh.mech_diversity_penalty * extra_styles
    partition_blur = 0.16 if scenario.drawer_partitions == 3 else 0.06
    phi_blur = 1.0 + beh.alpha_blur * (
        (1.0 - scenario.cue_salience) * 0.25 + partition_blur + bin_dump_blur + cog_load_blur
    )

    p_err = min(0.85, max(0.005, beh.base_error_rate * phi_rush * phi_ergo * phi_crowd * phi_blur))
    p_identify = min(0.98, max(0.08, scenario.cue_salience * (1.0 - 0.55 * (tau ** 2))))
    p_fix = min(0.90, max(0.01, beh.base_correction_rate * avg_vis_open * scenario.cue_salience * (1.0 - 0.7 * tau)))

    w_drift = 0.34 + (0.18 if scenario.drawer_partitions == 3 else 0.0) + 0.14 * crowd_excess
    w_eye_dump = 0.22 + (0.16 if has_pull_down else 0.0)
    w_wire_dump = (
        (0.10 + 0.50 * beh.bin_dump_routing_weight * n_bins + 0.06 * (1.0 - cross_tier_split))
        if n_bins > 0
        else 0.02
    )
    w_peg_dump = 0.10
    w_buried = 0.08 + 0.18 * crowd_excess + (0.10 if regime == "rush" else 0.0)

    w_sum = w_drift + w_eye_dump + w_wire_dump + w_peg_dump + w_buried
    route = [w / w_sum for w in (w_drift, w_eye_dump, w_wire_dump, w_peg_dump, w_buried)]

    n = 6
    p = [[0.0] * n for _ in range(n)]

    p_leave_home = min(0.75, pick_rate * p_err)
    p[0][0] = 1.0 - p_leave_home
    for idx, r_prob in enumerate(route, start=1):
        p[0][idx] = p_leave_home * r_prob

    state_vis = {1: 0.68, 2: 0.42, 3: 0.28, 4: 0.78}
    for s in (1, 2, 3, 4):
        encounter_prob = min(0.85, pick_rate * (0.65 + 0.55 * state_vis[s]))
        fix_prob = min(0.90, p_fix * (0.60 + 0.40 * state_vis[s]))
        recover_direct = encounter_prob * p_identify * (1.0 - 0.55 * p_err)
        recover_spontaneous = (1.0 - encounter_prob) * fix_prob * 0.45
        p_to_home = min(0.92, recover_direct + recover_spontaneous)

        rem = 1.0 - p_to_home
        p_worsen_buried = rem * min(0.32, 0.06 + 0.14 * crowd_excess + 0.10 * tau)
        p_migrate_eye = rem * 0.04
        p_stay = rem - p_worsen_buried - p_migrate_eye

        p[s][0] = p_to_home
        p[s][s] += p_stay
        p[s][2] += p_migrate_eye
        p[s][5] += p_worsen_buried

    p_spot_buried = min(0.65, pick_rate * 0.42 * p_identify + p_fix * 0.22)
    p_unbury_to_eye = (1.0 - p_spot_buried) * min(0.28, 0.10 + 0.12 * tau)
    p_stay_buried = 1.0 - p_spot_buried - p_unbury_to_eye
    p[5][0] = p_spot_buried
    p[5][2] = p_unbury_to_eye
    p[5][5] = p_stay_buried

    return p


def simulate_dtmc_validation(
    p_mat: List[List[float]],
    steps: int = 35000,
    burn_in: int = 2500,
    replications: int = 20,
    seed: int = 42,
) -> Tuple[List[float], float, float]:
    """Validate analytical stationary distribution and MFPTs via sample-path simulation."""
    rng = random.Random(seed)
    n = len(p_mat)
    counts = [0] * n
    cdfs = []
    for row in p_mat:
        cum = []
        acc = 0.0
        for v in row:
            acc += v
            cum.append(acc)
        cum[-1] = 1.0
        cdfs.append(cum)

    total_samples = 0
    ht_home_to_misplaced = []
    ht_misplaced_to_home = []

    for _ in range(replications):
        state = 0
        steps_in_home = 0
        steps_in_misplaced = 0
        for step in range(steps):
            prev_state = state
            r = rng.random()
            row_cdf = cdfs[state]
            nxt = 0
            while nxt < n - 1 and r > row_cdf[nxt]:
                nxt += 1
            state = nxt

            if prev_state == 0:
                steps_in_home += 1
                if state != 0:
                    ht_home_to_misplaced.append(steps_in_home)
                    steps_in_home = 0
            else:
                steps_in_misplaced += 1
                if state == 0:
                    ht_misplaced_to_home.append(steps_in_misplaced)
                    steps_in_misplaced = 0

            if step >= burn_in:
                counts[state] += 1
                total_samples += 1

    sim_pi = [c / total_samples for c in counts]
    mean_h2m = sum(ht_home_to_misplaced) / max(1, len(ht_home_to_misplaced))
    mean_m2h = sum(ht_misplaced_to_home) / max(1, len(ht_misplaced_to_home))
    return sim_pi, mean_h2m, mean_m2h


def analyze_scenario_markov(
    scenario: ExperimentScenario, regime: str = "shift_composite"
) -> MarkovAnalysisResult:
    """Run full Markov classification, stationary solve, MFPTs, and validation."""
    if regime == "shift_composite":
        p_rush = build_scenario_transition_matrix(scenario, "rush")
        p_calm = build_scenario_transition_matrix(scenario, "calm")
        w_r = scenario.rush.rush_job_share
        n = 6
        p_mat = [
            [w_r * p_rush[i][j] + (1.0 - w_r) * p_calm[i][j] for j in range(n)]
            for i in range(n)
        ]
    else:
        p_mat = build_scenario_transition_matrix(scenario, regime)

    max_row_err = max(abs(sum(row) - 1.0) for row in p_mat)
    is_irred, period, classes = classify_markov_chain(p_mat)
    pi, residual = compute_stationary_distribution(p_mat)
    lam2 = compute_second_eigenvalue_magnitude(p_mat, pi)
    spectral_gap = 1.0 - lam2
    mixing_time_95 = math.log(0.05) / math.log(max(1e-9, lam2)) if lam2 > 1e-9 else 1.0

    mfpt = compute_mfpt_matrix(p_mat, pi)
    exit_home = compute_absorption_time_from_set(p_mat, [0])[0]
    return_from_misplaced = compute_absorption_time_from_set(p_mat, [1, 2, 3, 4, 5])
    misplaced_weights = sum(pi[s] for s in (1, 2, 3, 4, 5))
    weighted_return_mfpt = (
        sum(pi[s] * return_from_misplaced[s] for s in (1, 2, 3, 4, 5)) / misplaced_weights
        if misplaced_weights > 1e-12
        else return_from_misplaced[2]
    )

    sim_pi, sim_h2m, sim_m2h = simulate_dtmc_validation(p_mat)
    tvd = 0.5 * sum(abs(sim_pi[i] - pi[i]) for i in range(6))

    return MarkovAnalysisResult(
        exp_id=scenario.exp_id,
        regime=regime,
        transition_matrix=p_mat,
        max_row_sum_error=max_row_err,
        is_irreducible=is_irred,
        period=period,
        communication_classes=classes,
        stationary_dist=pi,
        stationary_residual=residual,
        spectral_gap=spectral_gap,
        mixing_time_95=mixing_time_95,
        mfpt_matrix=mfpt,
        mfpt_home_to_any_misplaced=exit_home,
        mfpt_misplaced_to_home=weighted_return_mfpt,
        mfpt_home_to_buried=mfpt[0][5],
        sim_stationary_dist=sim_pi,
        total_variation_distance=tvd,
        sim_mfpt_home_to_misplaced=sim_h2m,
        sim_mfpt_misplaced_to_home=sim_m2h,
    )


def run_all_markov_analyses(
    experiments: Sequence[ExperimentScenario] | None = None,
) -> List[MarkovAnalysisResult]:
    """Analyze all experiments under composite, rush, and calm regimes."""
    exps = list(experiments) if experiments is not None else build_default_experiments()
    results = []
    for exp in exps:
        results.append(analyze_scenario_markov(exp, regime="shift_composite"))
    return results
