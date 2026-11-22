from std.collections import Dict, List
from std.math import gcd, log, sin, sqrt

from src.rng import Rng
from src.sim_config import (
    MECH_PEGBOARD_OVH,
    MECH_SPLIT_DRAWER_EYE,
    MECH_SPLIT_DRAWER_MID,
    MECH_WIRE_BIN_EYE,
    MECH_WIRE_BIN_MID,
    NUM_MECHS,
    ExperimentScenario,
    MechanismSpec,
    count_drawer_slots,
    count_mixed_bin_drawer_rows,
    count_shelf_complexity,
    has_mixed_drawer_sizes,
    has_row_width_mismatch,
)


@fieldwise_init
struct CommClass(Copyable, Movable):
    var nodes: List[Int]
    var kind: String


@fieldwise_init
struct MarkovAnalysisResult(Copyable, Movable):
    """Analytical DTMC outputs and sample-path validation metrics."""

    var exp_id: String
    var regime: String
    var transition_matrix: List[List[Float64]]
    var max_row_sum_error: Float64
    var is_irreducible: Bool
    var period: Int
    var communication_classes: List[CommClass]
    var stationary_dist: List[Float64]
    var stationary_residual: Float64
    var spectral_gap: Float64
    var mixing_time_95: Float64
    var mfpt_matrix: List[List[Float64]]
    var mfpt_home_to_any_misplaced: Float64
    var mfpt_misplaced_to_home: Float64
    var mfpt_home_to_buried: Float64
    var sim_stationary_dist: List[Float64]
    var total_variation_distance: Float64
    var sim_mfpt_home_to_misplaced: Float64
    var sim_mfpt_misplaced_to_home: Float64


def solve_linear_system(
    a: List[List[Float64]], b: List[Float64]
) raises -> List[Float64]:
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    var n = len(a)
    var aug = List[List[Float64]](capacity=n)
    for i in range(n):
        var row = List[Float64](capacity=n + 1)
        for j in range(n):
            row.append(a[i][j])
        row.append(b[i])
        aug.append(row^)

    for col in range(n):
        var pivot_row = col
        var best_val = abs(aug[col][col])
        for r in range(col + 1, n):
            var v = abs(aug[r][col])
            if v > best_val:
                best_val = v
                pivot_row = r
        if best_val < 1e-14:
            raise Error("Singular matrix encountered in linear solver.")
        if pivot_row != col:
            var tmp_row = aug[col].copy()
            aug[col] = aug[pivot_row].copy()
            aug[pivot_row] = tmp_row^

        var pivot = aug[col][col]
        for r in range(col + 1, n):
            var factor = aug[r][col] / pivot
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]

    var x = List[Float64](capacity=n)
    for _ in range(n):
        x.append(0.0)
    for i in range(n - 1, -1, -1):
        var s = aug[i][n]
        for j in range(i + 1, n):
            s -= aug[i][j] * x[j]
        x[i] = s / aug[i][i]
    return x^


def _strongconnect(
    v: Int,
    adj: List[List[Int]],
    mut index: Int,
    mut indices: List[Int],
    mut lowlink: List[Int],
    mut on_stack: List[Bool],
    mut stack: List[Int],
    mut sccs: List[List[Int]],
):
    indices[v] = index
    lowlink[v] = index
    index += 1
    stack.append(v)
    on_stack[v] = True

    for w in adj[v]:
        if indices[w] == -1:
            _strongconnect(w, adj, index, indices, lowlink, on_stack, stack, sccs)
            if lowlink[w] < lowlink[v]:
                lowlink[v] = lowlink[w]
        elif on_stack[w]:
            if indices[w] < lowlink[v]:
                lowlink[v] = indices[w]

    if lowlink[v] == indices[v]:
        var comp = List[Int]()
        while True:
            var w = stack.pop()
            on_stack[w] = False
            comp.append(w)
            if w == v:
                break
        sort(comp)
        sccs.append(comp^)


def _tarjan_scc(adj: List[List[Int]]) -> List[List[Int]]:
    var n = len(adj)
    var index = 0
    var indices = List[Int](capacity=n)
    var lowlink = List[Int](capacity=n)
    var on_stack = List[Bool](capacity=n)
    for _ in range(n):
        indices.append(-1)
        lowlink.append(0)
        on_stack.append(False)
    var stack = List[Int]()
    var sccs = List[List[Int]]()

    for i in range(n):
        if indices[i] == -1:
            _strongconnect(i, adj, index, indices, lowlink, on_stack, stack, sccs)
    return sccs^


def classify_markov_chain(
    p_mat: List[List[Float64]], tol: Float64 = 1e-12
) -> Tuple[Bool, Int, List[CommClass]]:
    """Classify communication classes, irreducibility, and periodicity."""
    var n = len(p_mat)
    var adj = List[List[Int]](capacity=n)
    for i in range(n):
        var row_adj = List[Int]()
        for j in range(n):
            if p_mat[i][j] > tol:
                row_adj.append(j)
        adj.append(row_adj^)

    var sccs = _tarjan_scc(adj)
    var comp_of = List[Int](capacity=n)
    for _ in range(n):
        comp_of.append(-1)
    for cid in range(len(sccs)):
        for node in sccs[cid]:
            comp_of[node] = cid

    var classes = List[CommClass](capacity=len(sccs))
    var recurrent_idx = 0
    var found_rec = False
    for cid in range(len(sccs)):
        var has_outgoing = False
        for u in sccs[cid]:
            for v in adj[u]:
                if comp_of[v] != cid:
                    has_outgoing = True
                    break
            if has_outgoing:
                break
        var kind: String
        if (
            len(sccs[cid]) == 1
            and not has_outgoing
            and abs(p_mat[sccs[cid][0]][sccs[cid][0]] - 1.0) <= tol
        ):
            kind = "absorbing"
        elif not has_outgoing:
            kind = "recurrent"
        else:
            kind = "transient"
        if not found_rec and (kind == "recurrent" or kind == "absorbing"):
            recurrent_idx = cid
            found_rec = True
        classes.append(CommClass(sccs[cid].copy(), kind))

    var is_irreducible = len(sccs) == 1
    var period = 1
    var rec_comp = sccs[recurrent_idx].copy()
    var need_bfs = len(rec_comp) > 1
    if not need_bfs:
        for u in rec_comp:
            if p_mat[u][u] <= tol:
                need_bfs = True
                break

    if need_bfs:
        var in_rec = List[Bool](capacity=n)
        var dist = List[Int](capacity=n)
        for _ in range(n):
            in_rec.append(False)
            dist.append(-1)
        for u in rec_comp:
            in_rec[u] = True
        var start = rec_comp[0]
        dist[start] = 0
        var queue = List[Int]()
        queue.append(start)
        var q_head = 0
        var g = 0
        while q_head < len(queue):
            var u = queue[q_head]
            q_head += 1
            for v in adj[u]:
                if not in_rec[v]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    var cycle_diff = abs(dist[u] + 1 - dist[v])
                    if cycle_diff > 0:
                        g = gcd(g, cycle_diff)
        if g > 0:
            period = g

    return (is_irreducible, period, classes^)


def compute_stationary_distribution(
    p_mat: List[List[Float64]]
) raises -> Tuple[List[Float64], Float64]:
    """Solve pi * P = pi, sum(pi) = 1 via linear system."""
    var n = len(p_mat)
    var sys_a = List[List[Float64]](capacity=n)
    var sys_b = List[Float64](capacity=n)
    for i in range(n):
        var row = List[Float64](capacity=n)
        for j in range(n):
            var diag = 1.0 if i == j else 0.0
            row.append(p_mat[j][i] - diag)
        sys_a.append(row^)
        sys_b.append(0.0)
    for j in range(n):
        sys_a[n - 1][j] = 1.0
    sys_b[n - 1] = 1.0

    var pi = solve_linear_system(sys_a, sys_b)
    var s: Float64 = 0.0
    for i in range(n):
        if pi[i] < 0.0:
            pi[i] = 0.0
        s += pi[i]
    for i in range(n):
        pi[i] /= s

    var residual: Float64 = 0.0
    for j in range(n):
        var pj: Float64 = 0.0
        for i in range(n):
            pj += pi[i] * p_mat[i][j]
        var diff = abs(pj - pi[j])
        if diff > residual:
            residual = diff

    return (pi^, residual)


def compute_second_eigenvalue_magnitude(
    p_mat: List[List[Float64]], pi: List[Float64]
) -> Float64:
    """Compute |lambda_2| using Wielandt deflation P_def = P - 1 * pi^T."""
    var n = len(p_mat)
    var p_def = List[List[Float64]](capacity=n)
    for i in range(n):
        var row = List[Float64](capacity=n)
        for j in range(n):
            row.append(p_mat[i][j] - pi[j])
        p_def.append(row^)

    var v = List[Float64](capacity=n)
    var norm_sq: Float64 = 0.0
    for i in range(n):
        var x = sin(Float64(i) + 1.0)
        v.append(x)
        norm_sq += x * x
    var norm = sqrt(norm_sq)
    for i in range(n):
        v[i] /= norm

    var lam2: Float64 = 0.0
    for _ in range(300):
        var w1 = List[Float64](capacity=n)
        var n1_sq: Float64 = 0.0
        for i in range(n):
            var acc: Float64 = 0.0
            for j in range(n):
                acc += p_def[i][j] * v[j]
            w1.append(acc)
            n1_sq += acc * acc
        if sqrt(n1_sq) < 1e-14:
            return 0.0

        var w2 = List[Float64](capacity=n)
        var n2_sq: Float64 = 0.0
        for i in range(n):
            var acc: Float64 = 0.0
            for j in range(n):
                acc += p_def[i][j] * w1[j]
            w2.append(acc)
            n2_sq += acc * acc
        var n2 = sqrt(n2_sq)
        lam2 = sqrt(n2)
        var denom = n2 if n2 > 1e-15 else 1e-15
        for i in range(n):
            v[i] = w2[i] / denom

    return lam2 if lam2 < 0.999999 else 0.999999


def compute_mfpt_matrix(
    p_mat: List[List[Float64]], pi: List[Float64]
) raises -> List[List[Float64]]:
    """Compute exact n x n Mean First Passage Time matrix M."""
    var n = len(p_mat)
    var mfpt = List[List[Float64]](capacity=n)
    for _ in range(n):
        var row = List[Float64](capacity=n)
        for _ in range(n):
            row.append(0.0)
        mfpt.append(row^)

    for target in range(n):
        mfpt[target][target] = (1.0 / pi[target]) if pi[target] > 1e-14 else 1e300
        var others = List[Int]()
        for idx in range(n):
            if idx != target:
                others.append(idx)
        var m = len(others)
        var sys_a = List[List[Float64]](capacity=m)
        var sys_b = List[Float64](capacity=m)
        for r in range(m):
            var row = List[Float64](capacity=m)
            for c in range(m):
                var eye = 1.0 if r == c else 0.0
                row.append(eye - p_mat[others[r]][others[c]])
            sys_a.append(row^)
            sys_b.append(1.0)
        var sol = solve_linear_system(sys_a, sys_b)
        for idx in range(m):
            mfpt[others[idx]][target] = sol[idx]
    return mfpt^


def compute_absorption_time_from_set(
    p_mat: List[List[Float64]], transient_states: List[Int]
) raises -> List[Float64]:
    """Return expected steps to exit transient_states indexed identically to p_mat."""
    var n = len(p_mat)
    var m = len(transient_states)
    var sys_a = List[List[Float64]](capacity=m)
    var sys_b = List[Float64](capacity=m)
    for r in range(m):
        var row = List[Float64](capacity=m)
        for c in range(m):
            var eye = 1.0 if r == c else 0.0
            row.append(eye - p_mat[transient_states[r]][transient_states[c]])
        sys_a.append(row^)
        sys_b.append(1.0)
    var sol = solve_linear_system(sys_a, sys_b)
    var out = List[Float64](capacity=n)
    for _ in range(n):
        out.append(0.0)
    for idx in range(m):
        out[transient_states[idx]] = sol[idx]
    return out^


def _add_row_mechs(
    rcode: String,
    tier_idx: Int,
    w: Float64,
    abc_mode: String,
    overflow_mode: String,
    mut tier_shares: List[Float64],
    mut has_bulk_peg_mismatch: Bool,
) raises:
    var is_eye = tier_idx == 1
    var drw_id = MECH_SPLIT_DRAWER_EYE if is_eye else MECH_SPLIT_DRAWER_MID
    var bin_id = MECH_WIRE_BIN_EYE if is_eye else MECH_WIRE_BIN_MID

    if (
        rcode == "7D6_4P"
        or rcode == "7D6_3P"
        or rcode == "5D9_4P"
        or rcode == "5D9_3P"
        or rcode == "3D9_3D6_4P"
        or rcode == "7D_4P"
        or rcode == "7D_3P"
        or rcode == "4D4P_3D3P"
    ):
        tier_shares[drw_id] += w
    elif (
        rcode == "1B_4D6_4P"
        or rcode == "1B_4D6_3P"
        or rcode == "1B_3D9_4P"
        or rcode == "1B_3D9_3P"
        or rcode == "1B_4D_4P"
        or rcode == "1B_4D_3P"
        or rcode == "1B_2D4P_2D3P"
    ):
        tier_shares[drw_id] += w * 0.76
        tier_shares[bin_id] += w * 0.24
    elif (
        rcode == "1D6_2B"
        or rcode == "1B_1D6_1B"
        or rcode == "2B_1D6"
        or rcode == "1D_2B"
        or rcode == "1B_1D_1B"
        or rcode == "2B_1D"
    ):
        tier_shares[drw_id] += w * 0.22
        tier_shares[bin_id] += w * 0.78
    elif rcode == "2B":
        tier_shares[bin_id] += w
    elif rcode == "SLIDING_PEG":
        if overflow_mode == "all_colocated" or abc_mode == "cross_tier_height":
            has_bulk_peg_mismatch = True
        tier_shares[MECH_PEGBOARD_OVH] += w
    else:
        raise Error(t"Unsupported modular row config: {rcode}")


def build_scenario_transition_matrix(
    scenario: ExperimentScenario,
    specs: List[MechanismSpec],
    regime: String = "rush",
) raises -> List[List[Float64]]:
    """Construct the 6-state DTMC transition matrix P for a scenario and shift regime."""
    if scenario.topology != "M_modular":
        raise Error(t"Unsupported topology: {scenario.topology}")

    var beh = scenario.behavior.copy()
    var tau: Float64
    var pick_rate: Float64
    if regime == "rush":
        tau = 0.5 * (scenario.rush.rush_tau_lo + scenario.rush.rush_tau_hi)
        pick_rate = 0.34
    else:
        tau = 0.5 * (scenario.rush.calm_tau_lo + scenario.rush.calm_tau_hi)
        pick_rate = 0.16

    var tier_shares = List[Float64](capacity=NUM_MECHS)
    for _ in range(NUM_MECHS):
        tier_shares.append(0.0)
    var has_bulk_peg_mismatch = False

    var w1 = 0.56 if scenario.overflow_mode == "hybrid_reserve" else 0.62
    _add_row_mechs(scenario.row0, 1, w1, scenario.abc_colocation_mode, scenario.overflow_mode, tier_shares, has_bulk_peg_mismatch)
    _add_row_mechs(scenario.row1, 2, 0.28, scenario.abc_colocation_mode, scenario.overflow_mode, tier_shares, has_bulk_peg_mismatch)
    _add_row_mechs(scenario.row2, 3, 0.16, scenario.abc_colocation_mode, scenario.overflow_mode, tier_shares, has_bulk_peg_mismatch)

    var tot_w: Float64 = 0.0
    for i in range(NUM_MECHS):
        tot_w += tier_shares[i]
    var denom = tot_w if tot_w > 1e-9 else 1e-9
    for i in range(NUM_MECHS):
        tier_shares[i] /= denom

    var avg_ergo: Float64 = 0.0
    var avg_vis_open: Float64 = 0.0
    var has_pull_down = False
    for m in range(NUM_MECHS):
        if tier_shares[m] > 0.0:
            avg_ergo += tier_shares[m] * specs[m].reach_actuation_cost
            avg_vis_open += tier_shares[m] * specs[m].visibility_open
            if specs[m].is_pull_down:
                has_pull_down = True

    var cross_tier_split: Float64
    if scenario.abc_colocation_mode == "same_slot" or scenario.abc_colocation_mode == "adjacent_partition":
        cross_tier_split = 0.08
    elif scenario.abc_colocation_mode == "same_tier_bay":
        cross_tier_split = 0.16
    else:
        cross_tier_split = 0.58

    var occlusion_extra: Float64 = 0.0
    if has_pull_down and regime == "rush":
        if (
            (scenario.abc_colocation_mode == "adjacent_partition" or scenario.abc_colocation_mode == "same_slot")
            and scenario.overflow_mode == "hybrid_reserve"
        ):
            occlusion_extra = 0.02 * tau
        else:
            occlusion_extra = 0.24 * tau

    var phi_rush = 1.0 + beh.alpha_rush * (tau * tau)
    var phi_ergo = 1.0 + beh.alpha_ergo * avg_ergo * cross_tier_split + occlusion_extra
    var bulk_peg_extra = 0.10 if has_bulk_peg_mismatch else 0.0
    var fill_denom = 1.05 - min(0.99, scenario.target_fill_ratio + 0.5 * bulk_peg_extra)
    var crowd_excess = max(0.0, (scenario.target_fill_ratio + bulk_peg_extra - 0.75) / fill_denom)
    var phi_crowd = 1.0 + beh.alpha_crowd * crowd_excess

    var cpx = count_shelf_complexity(scenario.row0, scenario.row1, scenario.row2)
    var n_bins = cpx[0]
    var extra_styles = cpx[1]
    var mixed_bin_drw_rows = count_mixed_bin_drawer_rows(scenario.row0, scenario.row1, scenario.row2)
    var bin_dump_blur = beh.bin_dump_error_penalty * Float64(n_bins)
    var size_mix_blur = beh.drawer_size_mix_penalty if has_mixed_drawer_sizes(scenario.row0, scenario.row1, scenario.row2) else 0.0
    var n_drawer_slots = count_drawer_slots(scenario.row0, scenario.row1, scenario.row2)
    var deficit_frac = max(0.0, Float64(48 - n_drawer_slots) / 48.0)
    var slot_deficit_blur = beh.slot_deficit_penalty * deficit_frac
    var row_width_blur = beh.row_width_mismatch_penalty if has_row_width_mismatch(scenario.row0, scenario.row1, scenario.row2) else 0.0
    var cog_load_blur = (
        beh.mech_diversity_penalty * Float64(extra_styles)
        + beh.bin_drawer_same_row_penalty * Float64(mixed_bin_drw_rows)
        + size_mix_blur
        + slot_deficit_blur
        + row_width_blur
    )
    var partition_blur = 0.16 if scenario.drawer_partitions == 3 else 0.06
    var phi_blur = 1.0 + beh.alpha_blur * (
        (1.0 - scenario.cue_salience) * 0.25 + partition_blur + bin_dump_blur + cog_load_blur
    )

    var p_err = min(0.85, max(0.005, beh.base_error_rate * phi_rush * phi_ergo * phi_crowd * phi_blur))
    var p_identify = min(0.98, max(0.08, scenario.cue_salience * (1.0 - 0.55 * (tau * tau))))
    var p_fix = min(0.90, max(0.01, beh.base_correction_rate * avg_vis_open * scenario.cue_salience * (1.0 - 0.7 * tau)))

    var w_drift = 0.34 + (0.18 if scenario.drawer_partitions == 3 else 0.0) + 0.14 * crowd_excess
    var w_eye_dump = 0.22 + (0.16 if has_pull_down else 0.0)
    var w_wire_dump = (
        (0.10 + 0.50 * beh.bin_dump_routing_weight * Float64(n_bins) + 0.06 * (1.0 - cross_tier_split))
        if n_bins > 0
        else 0.02
    )
    var w_peg_dump = 0.10
    var w_buried = 0.08 + 0.18 * crowd_excess + (0.10 if regime == "rush" else 0.0)
    var w_sum = w_drift + w_eye_dump + w_wire_dump + w_peg_dump + w_buried

    var n = 6
    var p = List[List[Float64]](capacity=n)
    for _ in range(n):
        var row = List[Float64](capacity=n)
        for _ in range(n):
            row.append(0.0)
        p.append(row^)

    var p_leave_home = min(0.75, pick_rate * p_err)
    p[0][0] = 1.0 - p_leave_home
    p[0][1] = p_leave_home * (w_drift / w_sum)
    p[0][2] = p_leave_home * (w_eye_dump / w_sum)
    p[0][3] = p_leave_home * (w_wire_dump / w_sum)
    p[0][4] = p_leave_home * (w_peg_dump / w_sum)
    p[0][5] = p_leave_home * (w_buried / w_sum)

    var state_vis = [0.0, 0.68, 0.42, 0.28, 0.78]
    for s in range(1, 5):
        var sv = state_vis[s]
        var encounter_prob = min(0.85, pick_rate * (0.65 + 0.55 * sv))
        var fix_prob = min(0.90, p_fix * (0.60 + 0.40 * sv))
        var recover_direct = encounter_prob * p_identify * (1.0 - 0.55 * p_err)
        var recover_spontaneous = (1.0 - encounter_prob) * fix_prob * 0.45
        var p_to_home = min(0.92, recover_direct + recover_spontaneous)

        var rem = 1.0 - p_to_home
        var p_worsen_buried = rem * min(0.32, 0.06 + 0.14 * crowd_excess + 0.10 * tau)
        var p_migrate_eye = rem * 0.04
        var p_stay = rem - p_worsen_buried - p_migrate_eye

        p[s][0] = p_to_home
        p[s][s] += p_stay
        p[s][2] += p_migrate_eye
        p[s][5] += p_worsen_buried

    var p_spot_buried = min(0.65, pick_rate * 0.42 * p_identify + p_fix * 0.22)
    var p_unbury_to_eye = (1.0 - p_spot_buried) * min(0.28, 0.10 + 0.12 * tau)
    var p_stay_buried = 1.0 - p_spot_buried - p_unbury_to_eye
    p[5][0] = p_spot_buried
    p[5][2] = p_unbury_to_eye
    p[5][5] = p_stay_buried

    return p^


def simulate_dtmc_validation(
    p_mat: List[List[Float64]],
    steps: Int = 35000,
    burn_in: Int = 2500,
    replications: Int = 20,
    seed: Int = 42,
) -> Tuple[List[Float64], Float64, Float64]:
    """Validate analytical stationary distribution and MFPTs via sample-path simulation."""
    var rng = Rng(seed)
    var n = len(p_mat)
    var counts = List[Int](capacity=n)
    for _ in range(n):
        counts.append(0)

    var cdfs = List[List[Float64]](capacity=n)
    for i in range(n):
        var cum = List[Float64](capacity=n)
        var acc: Float64 = 0.0
        for j in range(n):
            acc += p_mat[i][j]
            cum.append(acc)
        cum[n - 1] = 1.0
        cdfs.append(cum^)

    var total_samples = 0
    var sum_h2m = 0
    var cnt_h2m = 0
    var sum_m2h = 0
    var cnt_m2h = 0

    for _ in range(replications):
        var state = 0
        var steps_in_home = 0
        var steps_in_misplaced = 0
        for step in range(steps):
            var prev_state = state
            var r = rng.random()
            var nxt = 0
            while nxt < n - 1 and r > cdfs[state][nxt]:
                nxt += 1
            state = nxt

            if prev_state == 0:
                steps_in_home += 1
                if state != 0:
                    sum_h2m += steps_in_home
                    cnt_h2m += 1
                    steps_in_home = 0
            else:
                steps_in_misplaced += 1
                if state == 0:
                    sum_m2h += steps_in_misplaced
                    cnt_m2h += 1
                    steps_in_misplaced = 0

            if step >= burn_in:
                counts[state] += 1
                total_samples += 1

    var sim_pi = List[Float64](capacity=n)
    for i in range(n):
        sim_pi.append(Float64(counts[i]) / Float64(total_samples))
    var mean_h2m = Float64(sum_h2m) / Float64(cnt_h2m if cnt_h2m > 0 else 1)
    var mean_m2h = Float64(sum_m2h) / Float64(cnt_m2h if cnt_m2h > 0 else 1)
    return (sim_pi^, mean_h2m, mean_m2h)


def analyze_scenario_markov(
    scenario: ExperimentScenario,
    specs: List[MechanismSpec],
    regime: String = "shift_composite",
) raises -> MarkovAnalysisResult:
    """Run full Markov classification, stationary solve, MFPTs, and validation."""
    var p_mat: List[List[Float64]]
    var n = 6
    if regime == "shift_composite":
        var p_rush = build_scenario_transition_matrix(scenario, specs, "rush")
        var p_calm = build_scenario_transition_matrix(scenario, specs, "calm")
        var w_r = scenario.rush.rush_job_share
        p_mat = List[List[Float64]](capacity=n)
        for i in range(n):
            var row = List[Float64](capacity=n)
            for j in range(n):
                row.append(w_r * p_rush[i][j] + (1.0 - w_r) * p_calm[i][j])
            p_mat.append(row^)
    else:
        p_mat = build_scenario_transition_matrix(scenario, specs, regime)

    var max_row_err: Float64 = 0.0
    for i in range(n):
        var s: Float64 = 0.0
        for j in range(n):
            s += p_mat[i][j]
        var e = abs(s - 1.0)
        if e > max_row_err:
            max_row_err = e

    var cls_res = classify_markov_chain(p_mat)
    var is_irred = cls_res[0]
    var period = cls_res[1]
    var classes = cls_res[2].copy()

    var pi_res = compute_stationary_distribution(p_mat)
    var pi = pi_res[0].copy()
    var residual = pi_res[1]

    var lam2 = compute_second_eigenvalue_magnitude(p_mat, pi)
    var spectral_gap = 1.0 - lam2
    var mixing_time_95 = (
        log(0.05) / log(lam2 if lam2 > 1e-9 else 1e-9)
        if lam2 > 1e-9
        else 1.0
    )

    var mfpt = compute_mfpt_matrix(p_mat, pi)
    var home_exit = compute_absorption_time_from_set(p_mat, [0])
    var exit_home = home_exit[0]
    var ret_misplaced = compute_absorption_time_from_set(p_mat, [1, 2, 3, 4, 5])
    var misplaced_w: Float64 = 0.0
    var num_w: Float64 = 0.0
    for s in range(1, 6):
        misplaced_w += pi[s]
        num_w += pi[s] * ret_misplaced[s]
    var weighted_ret_mfpt = (num_w / misplaced_w) if misplaced_w > 1e-12 else ret_misplaced[2]

    var sim_res = simulate_dtmc_validation(p_mat)
    var sim_pi = sim_res[0].copy()
    var sim_h2m = sim_res[1]
    var sim_m2h = sim_res[2]

    var tvd: Float64 = 0.0
    for i in range(6):
        tvd += abs(sim_pi[i] - pi[i])
    tvd *= 0.5

    return MarkovAnalysisResult(
        scenario.exp_id,
        regime,
        p_mat^,
        max_row_err,
        is_irred,
        period,
        classes^,
        pi^,
        residual,
        spectral_gap,
        mixing_time_95,
        mfpt^,
        exit_home,
        weighted_ret_mfpt,
        mfpt[0][5],
        sim_pi^,
        tvd,
        sim_h2m,
        sim_m2h,
    )


def run_all_markov_analyses(
    experiments: List[ExperimentScenario],
    specs: List[MechanismSpec],
) raises -> List[MarkovAnalysisResult]:
    var results = List[MarkovAnalysisResult](capacity=len(experiments))
    for i in range(len(experiments)):
        results.append(analyze_scenario_markov(experiments[i], specs, "shift_composite"))
    return results^
