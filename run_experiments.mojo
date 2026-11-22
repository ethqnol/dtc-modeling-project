from std.collections import List
from std.pathlib import Path
from std.python import Python, PythonObject
from std.sys import argv
from std.time import perf_counter_ns

from src.generate_charts import SweepSeries, generate_all_charts
from src.format_utils import csv_cell, fmt_float, fmt_sci2
from src.markov_analysis import MarkovAnalysisResult, run_all_markov_analyses
from src.orientation_search import (
    build_modular_topologies,
    build_positioning_policies,
    run_orientation_search,
)
from src.sim_config import (
    AppConfig,
    ExperimentScenario,
    InventoryData,
    RollItem,
    build_default_experiments,
    load_app_config,
    load_roll_inventory,
)
from src.simulation_engine import (
    ExperimentSummary,
    allocate_home_slots,
    compute_layout_metrics,
    run_experiment_monte_carlo,
)


def _resolve_workspace_dir() -> String:
    if Path("data/roll-data.csv").exists():
        return "."
    return "/home/ewu/Code/dtc-simulation"


def _run_capacity_sweep(
    cfg: AppConfig,
    inv: InventoryData,
    n_replications: Int,
) raises -> List[SweepSeries]:
    var fill_ratios = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]
    var labels = [
        "TOPO053_Mid3D93D64P_Btm3D93D64P",
        "TOPO007_Mid7D64P_Btm2B",
        "TOPO007_BaselineCOI",
    ]
    var r0s = ["3D9_3D6_4P", "7D6_4P", "7D6_4P"]
    var r1s = ["3D9_3D6_4P", "2B", "2B"]
    var abcs = ["adjacent_partition", "adjacent_partition", "cross_tier_height"]
    var ovfs = ["hybrid_reserve", "hybrid_reserve", "all_colocated"]
    var sps = ["center_out", "center_out", "random_perm"]
    var cues = [0.90, 0.90, 0.20]

    var out = List[SweepSeries](capacity=3)
    for ci in range(3):
        var rhos = List[Float64](capacity=len(fill_ratios))
        var accs = List[Float64](capacity=len(fill_ratios))
        var lats = List[Float64](capacity=len(fill_ratios))
        for ri in range(len(fill_ratios)):
            var rho = fill_ratios[ri]
            var scen = ExperimentScenario(
                String(t"SWEEP-{labels[ci]}-{Int(rho * 100.0)}"),
                labels[ci],
                "M_modular",
                "C_hybrid" if ovfs[ci] == "hybrid_reserve" else "B_kit",
                4,
                cues[ci],
                rho,
                "",
                r0s[ci],
                r1s[ci],
                "SLIDING_PEG",
                cfg.geometry.copy(),
                cfg.rush.copy(),
                cfg.behavior.copy(),
                abcs[ci],
                ovfs[ci],
                sps[ci],
                1000,
            )
            var res = run_experiment_monte_carlo(
                scen,
                inv.variants,
                inv.rolls,
                inv.freq_calm,
                inv.freq_rush,
                inv.co_matrix,
                inv.num_variants,
                inv.high_assoc_siblings,
                cfg.mechanisms,
                n_replications,
                5000,
            )
            rhos.append(rho)
            accs.append(res.final_acc_mean)
            lats.append(res.overall_latency_mean)
        out.append(SweepSeries(labels[ci], rhos^, accs^, lats^))
    return out^


def _cmp_slotting_roll(a: RollItem, b: RollItem) -> Bool:
    if a.home_mechanism != b.home_mechanism:
        return a.home_mechanism < b.home_mechanism
    if a.home_slot != b.home_slot:
        return a.home_slot < b.home_slot
    return a.roll_id < b.roll_id


def _export_slotting_plan(
    winning_scenario: ExperimentScenario,
    inv: InventoryData,
    out_csv: String,
) raises:
    var rolls = inv.rolls.copy()
    var slots = allocate_home_slots(winning_scenario, inv.variants, rolls)
    sort(rolls, _cmp_slotting_roll)

    var lines = List[String](capacity=len(rolls) + 1)
    lines.append(
        "roll_id,sku_code,product_name,variant_code,copy_index,category,"
        "outer_diameter_cm,roll_height_cm,has_core,is_rush_priority,"
        "assigned_mechanism,assigned_slot,home_tier,bay_side,x_center_cm"
    )
    for i in range(len(rolls)):
        var r = rolls[i].copy()
        var s = slots[r.home_slot_idx].copy()
        var bay = "Left(0-60cm)" if s.bay_side == 0 else "Right(60-120cm)"
        var core_s = "True" if r.has_core else "False"
        var prio_s = "True" if r.is_rush_priority else "False"
        lines.append(
            String(r.roll_id)
            + "," + csv_cell(r.sku_code)
            + "," + csv_cell(r.product_name)
            + "," + csv_cell(r.variant_code)
            + "," + String(r.copy_index + 1)
            + "," + csv_cell(r.category)
            + "," + String(r.outer_diameter_cm)
            + "," + String(r.roll_height_cm)
            + "," + core_s
            + "," + prio_s
            + "," + csv_cell(r.home_mechanism)
            + "," + csv_cell(r.home_slot)
            + "," + String(s.tier_index)
            + "," + csv_cell(bay)
            + "," + String(s.x_center_cm)
        )
    Path(out_csv).write_text("\n".join(lines) + "\n")


def _export_markov_csv(results: List[MarkovAnalysisResult], out_csv: String) raises:
    var lines = List[String](capacity=len(results) + 1)
    lines.append(
        "exp_id,is_irreducible,period,max_row_sum_error,"
        "pi_home,pi_drift_adj,pi_eye_dump,pi_wire_dump,pi_peg_dump,pi_buried,"
        "spectral_gap,mixing_time_95,mfpt_home_to_misplaced,mfpt_misplaced_to_home,"
        "mfpt_home_to_buried,sim_tvd"
    )
    for i in range(len(results)):
        var mr = results[i].copy()
        var irr_s = "True" if mr.is_irreducible else "False"
        lines.append(
            csv_cell(mr.exp_id)
            + "," + irr_s
            + "," + String(mr.period)
            + "," + fmt_sci2(mr.max_row_sum_error)
            + "," + fmt_float(mr.stationary_dist[0], 5)
            + "," + fmt_float(mr.stationary_dist[1], 5)
            + "," + fmt_float(mr.stationary_dist[2], 5)
            + "," + fmt_float(mr.stationary_dist[3], 5)
            + "," + fmt_float(mr.stationary_dist[4], 5)
            + "," + fmt_float(mr.stationary_dist[5], 5)
            + "," + fmt_float(mr.spectral_gap, 5)
            + "," + fmt_float(mr.mixing_time_95, 2)
            + "," + fmt_float(mr.mfpt_home_to_any_misplaced, 2)
            + "," + fmt_float(mr.mfpt_misplaced_to_home, 2)
            + "," + fmt_float(mr.mfpt_home_to_buried, 2)
            + "," + fmt_float(mr.total_variation_distance, 5)
        )
    Path(out_csv).write_text("\n".join(lines) + "\n")


def _export_experiment_csv(summaries: List[ExperimentSummary], out_csv: String) raises:
    var lines = List[String](capacity=len(summaries) + 1)
    lines.append(
        "exp_id,name,topology,strategy,drawer_partitions,"
        "cue_salience,target_fill_ratio,final_acc_mean,final_acc_std,"
        "final_entropy,final_cross_tier_contam,mean_job_latency_sec,"
        "rush_latency_sec,calm_latency_sec,mean_t_crit,collapse_rate"
    )
    for i in range(len(summaries)):
        var res = summaries[i].copy()
        var sc = res.scenario.copy()
        lines.append(
            csv_cell(sc.exp_id)
            + "," + csv_cell(sc.name)
            + "," + csv_cell(sc.topology)
            + "," + csv_cell(sc.strategy)
            + "," + String(sc.drawer_partitions)
            + "," + String(sc.cue_salience)
            + "," + String(sc.target_fill_ratio)
            + "," + fmt_float(res.final_acc_mean, 4)
            + "," + fmt_float(res.final_acc_std, 4)
            + "," + fmt_float(res.final_entropy_mean, 4)
            + "," + fmt_float(res.final_contam_mean, 4)
            + "," + fmt_float(res.overall_latency_mean, 2)
            + "," + fmt_float(res.rush_latency_mean, 2)
            + "," + fmt_float(res.calm_latency_mean, 2)
            + "," + fmt_float(res.t_crit_mean, 1)
            + "," + fmt_float(res.collapse_rate, 2)
        )
    Path(out_csv).write_text("\n".join(lines) + "\n")


def _export_sweep_csv(sweeps: List[SweepSeries], out_csv: String) raises:
    var lines = List[String]()
    lines.append("layout,fill_ratio,final_accuracy,mean_latency_sec")
    for si in range(len(sweeps)):
        var sw = sweeps[si].copy()
        for pi in range(len(sw.rhos)):
            lines.append(
                csv_cell(sw.label)
                + "," + String(sw.rhos[pi])
                + "," + fmt_float(sw.accs[pi], 4)
                + "," + fmt_float(sw.lats[pi], 2)
            )
    Path(out_csv).write_text("\n".join(lines) + "\n")


def _py_float_list(vals: List[Float64]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(vals)):
        out.append(PythonObject(vals[i]))
    return out^


def _export_notebook_data(
    cfg: AppConfig,
    inv: InventoryData,
    all_scenarios: List[ExperimentScenario],
    markov_results: List[MarkovAnalysisResult],
    top_ori_id: String,
    out_json: String,
) raises:
    var json_mod = Python.import_module("json")
    var payload = Python.dict()
    payload["top_ori_id"] = PythonObject(top_ori_id)

    var search_meta = Python.dict()
    search_meta["num_topologies"] = PythonObject(len(build_modular_topologies()))
    search_meta["num_policies"] = PythonObject(len(build_positioning_policies()))
    payload["search_meta"] = search_meta

    var g = cfg.geometry.copy()
    var py_geom = Python.dict()
    py_geom["shelf_width_cm"] = PythonObject(g.shelf_width_cm)
    py_geom["shelf_depth_cm"] = PythonObject(g.shelf_depth_cm)
    py_geom["shelf_row_height_cm"] = PythonObject(g.shelf_row_height_cm)
    py_geom["eye_level_height_cm"] = PythonObject(g.eye_level_height_cm)
    py_geom["bottom_level_height_cm"] = PythonObject(g.bottom_level_height_cm)
    py_geom["mid_level_height_cm"] = PythonObject(g.mid_level_height_cm)
    py_geom["top_overhead_height_cm"] = PythonObject(g.top_overhead_height_cm)
    py_geom["drawer_outer_width_cm"] = PythonObject(g.drawer_outer_width_cm)
    py_geom["drawer_inner_width_cm"] = PythonObject(g.drawer_inner_width_cm)
    py_geom["drawers_per_full_row"] = PythonObject(g.drawers_per_full_row)
    py_geom["drawers_per_half_bay"] = PythonObject(g.drawers_per_half_bay)
    py_geom["drawer_9in_outer_width_cm"] = PythonObject(g.drawer_9in_outer_width_cm)
    py_geom["drawer_9in_inner_width_cm"] = PythonObject(g.drawer_9in_inner_width_cm)
    py_geom["drawers_9in_per_full_row"] = PythonObject(g.drawers_9in_per_full_row)
    py_geom["drawers_9in_per_half_bay"] = PythonObject(g.drawers_9in_per_half_bay)
    py_geom["partition_6in_4p_cm"] = PythonObject(g.partition_width_cm(4, 0))
    py_geom["partition_6in_3p_top_cm"] = PythonObject(g.partition_width_cm(3, 0))
    py_geom["partition_9in_4p_cm"] = PythonObject(
        g.partition_width_cm(4, 0, g.drawer_9in_inner_width_cm)
    )
    py_geom["partition_9in_3p_top_cm"] = PythonObject(
        g.partition_width_cm(3, 0, g.drawer_9in_inner_width_cm)
    )
    py_geom["wire_bin_bay_width_cm"] = PythonObject(g.wire_bin_bay_width_cm)
    py_geom["wire_bin_subzones"] = PythonObject(g.wire_bin_subzones)
    py_geom["wire_bin_subzone_hard_cap"] = PythonObject(g.wire_bin_subzone_hard_cap)
    py_geom["peg_rows"] = PythonObject(g.peg_rows)
    py_geom["pegs_per_row"] = PythonObject(g.pegs_per_row)
    py_geom["peg_max_od_cm"] = PythonObject(g.peg_max_od_cm)
    payload["geometry"] = py_geom

    var py_mechs = Python.list()
    for i in range(len(cfg.mechanisms)):
        var sp = cfg.mechanisms[i].copy()
        var md = Python.dict()
        md["mechanism_id"] = PythonObject(sp.mechanism_name)
        md["reach_actuation_cost"] = PythonObject(sp.reach_actuation_cost)
        md["visibility_closed"] = PythonObject(sp.visibility_closed)
        md["visibility_open"] = PythonObject(sp.visibility_open)
        md["clearance_score"] = PythonObject(sp.clearance_score)
        md["base_retrieval_sec"] = PythonObject(sp.base_retrieval_sec)
        md["base_return_sec"] = PythonObject(sp.base_return_sec)
        md["supports_coreless"] = PythonObject(sp.supports_coreless)
        md["is_pull_down"] = PythonObject(sp.is_pull_down)
        py_mechs.append(md)
    payload["mechanisms"] = py_mechs

    var rc = cfg.rush.copy()
    var py_rush = Python.dict()
    py_rush["num_shifts"] = PythonObject(rc.num_shifts)
    py_rush["shift_cycles"] = PythonObject(rc.shift_cycles)
    py_rush["total_cycles"] = PythonObject(rc.total_cycles())
    py_rush["rush_job_share"] = PythonObject(rc.rush_job_share)
    var r_tau = Python.list()
    r_tau.append(PythonObject(rc.rush_tau_lo))
    r_tau.append(PythonObject(rc.rush_tau_hi))
    py_rush["rush_tau_range"] = r_tau
    var c_tau = Python.list()
    c_tau.append(PythonObject(rc.calm_tau_lo))
    c_tau.append(PythonObject(rc.calm_tau_hi))
    py_rush["calm_tau_range"] = c_tau
    py_rush["rush_sku_boost"] = PythonObject(rc.rush_sku_boost)
    var py_cats = Python.list()
    for i in range(len(rc.rush_priority_categories)):
        py_cats.append(PythonObject(rc.rush_priority_categories[i]))
    py_rush["rush_priority_categories"] = py_cats
    py_rush["same_product_co_retrieval"] = PythonObject(rc.same_product_co_retrieval)
    payload["rush"] = py_rush

    var bc = cfg.behavior.copy()
    var py_beh = Python.dict()
    py_beh["base_error_rate"] = PythonObject(bc.base_error_rate)
    py_beh["base_correction_rate"] = PythonObject(bc.base_correction_rate)
    py_beh["alpha_rush"] = PythonObject(bc.alpha_rush)
    py_beh["alpha_ergo"] = PythonObject(bc.alpha_ergo)
    py_beh["alpha_crowd"] = PythonObject(bc.alpha_crowd)
    py_beh["alpha_blur"] = PythonObject(bc.alpha_blur)
    py_beh["bin_dump_error_penalty"] = PythonObject(bc.bin_dump_error_penalty)
    py_beh["bin_dump_routing_weight"] = PythonObject(bc.bin_dump_routing_weight)
    py_beh["mech_diversity_penalty"] = PythonObject(bc.mech_diversity_penalty)
    py_beh["bin_drawer_same_row_penalty"] = PythonObject(bc.bin_drawer_same_row_penalty)
    py_beh["multi_sku_slot_penalty"] = PythonObject(bc.multi_sku_slot_penalty)
    py_beh["slot_deficit_penalty"] = PythonObject(bc.slot_deficit_penalty)
    py_beh["row_width_mismatch_penalty"] = PythonObject(bc.row_width_mismatch_penalty)
    py_beh["unlatch_rush_prob"] = PythonObject(bc.unlatch_rush_prob)
    py_beh["usability_accuracy_threshold"] = PythonObject(bc.usability_accuracy_threshold)
    payload["behavior"] = py_beh

    var n_rush_rolls = 0
    for i in range(len(inv.rolls)):
        if inv.rolls[i].is_rush_priority:
            n_rush_rolls += 1
    var abc_pairs = 0
    for i in range(inv.num_variants):
        for j in range(i + 1, inv.num_variants):
            if (
                inv.variants[i].product_name == inv.variants[j].product_name
                and inv.co(i, j) >= 0.85
            ):
                abc_pairs += 1
    var py_vars = Python.list()
    for i in range(len(inv.variants)):
        var v = inv.variants[i].copy()
        var vd = Python.dict()
        vd["sku_code"] = PythonObject(v.sku_code)
        vd["product_name"] = PythonObject(v.product_name)
        vd["variant_code"] = PythonObject(v.variant_code)
        vd["category"] = PythonObject(v.category)
        vd["quantity"] = PythonObject(v.quantity)
        vd["outer_diameter_cm"] = PythonObject(v.outer_diameter_cm)
        vd["roll_height_cm"] = PythonObject(v.roll_height_cm)
        vd["has_core"] = PythonObject(v.has_core)
        vd["is_rush_priority"] = PythonObject(v.is_rush_priority)
        vd["baseline_freq"] = PythonObject(v.baseline_freq)
        vd["rush_freq"] = PythonObject(v.rush_freq)
        py_vars.append(vd)
    var py_inv = Python.dict()
    py_inv["num_rolls"] = PythonObject(len(inv.rolls))
    py_inv["num_rush_rolls"] = PythonObject(n_rush_rolls)
    py_inv["abc_pairs"] = PythonObject(abc_pairs)
    py_inv["variants"] = py_vars
    payload["inventory"] = py_inv

    var py_benches = Python.list()
    for i in range(len(all_scenarios)):
        var sc = all_scenarios[i].copy()
        var rolls_copy = inv.rolls.copy()
        var slots = allocate_home_slots(sc, inv.variants, rolls_copy)
        var lm = compute_layout_metrics(rolls_copy, slots)
        var bd = Python.dict()
        bd["exp_id"] = PythonObject(sc.exp_id)
        bd["name"] = PythonObject(sc.name)
        var py_rows = Python.list()
        py_rows.append(PythonObject(sc.row0))
        py_rows.append(PythonObject(sc.row1))
        py_rows.append(PythonObject(sc.row2))
        bd["row_configs"] = py_rows
        bd["abc_colocation_mode"] = PythonObject(sc.abc_colocation_mode)
        bd["overflow_mode"] = PythonObject(sc.overflow_mode)
        bd["cue_salience"] = PythonObject(sc.cue_salience)
        bd["target_fill_ratio"] = PythonObject(sc.target_fill_ratio)
        bd["abc_same_drawer_pct"] = PythonObject(lm.abc_same_drawer_pct)
        py_benches.append(bd)
    payload["benchmarks"] = py_benches

    var py_mk = Python.dict()
    for i in range(len(markov_results)):
        var mr = markov_results[i].copy()
        var md = Python.dict()
        var py_p = Python.list()
        for r in range(len(mr.transition_matrix)):
            py_p.append(_py_float_list(mr.transition_matrix[r]))
        md["transition_matrix"] = py_p
        md["stationary_dist"] = _py_float_list(mr.stationary_dist)
        md["sim_stationary_dist"] = _py_float_list(mr.sim_stationary_dist)
        md["spectral_gap"] = PythonObject(mr.spectral_gap)
        md["mixing_time_95"] = PythonObject(mr.mixing_time_95)
        md["mfpt_home_to_any_misplaced"] = PythonObject(mr.mfpt_home_to_any_misplaced)
        md["mfpt_misplaced_to_home"] = PythonObject(mr.mfpt_misplaced_to_home)
        md["total_variation_distance"] = PythonObject(mr.total_variation_distance)
        py_mk[mr.exp_id] = md
    payload["markov_details"] = py_mk

    var json_str = String(py=json_mod.dumps(payload, indent=2))
    Path(out_json).write_text(json_str + "\n")


def main() raises:
    var args = argv()
    var ws_dir = _resolve_workspace_dir()
    var cfg_path = ws_dir + "/config/shelf_config.yaml"
    var geom_path = ws_dir + "/config/shelf_geometry.yaml"
    var csv_path = ws_dir + "/data/roll-data.csv"
    var out_dir = ws_dir + "/output"
    var fig_dir = ws_dir + "/figures"

    var quick = False
    var no_charts = False
    var ai = 1
    while ai < len(args):
        var a = String(args[ai])
        if a == "--quick":
            quick = True
        elif a == "--no-charts":
            no_charts = True
        elif (a == "-c" or a == "--config") and (ai + 1 < len(args)):
            ai += 1
            cfg_path = String(args[ai])
        ai += 1

    var cfg = load_app_config(geom_path, cfg_path)
    var inv = load_roll_inventory(csv_path, cfg.rush)
    var bench_reps = 10 if quick else cfg.execution.benchmark_replications
    var sweep_reps = 8 if quick else cfg.execution.sweep_replications
    var search_reps = 2 if quick else cfg.execution.search_replications
    var lat_w = cfg.execution.composite_score_latency_weight

    var t0 = perf_counter_ns()
    var experiments = build_default_experiments(cfg)

    print("Running Markov analysis (Mojo)...")
    var markov_results = run_all_markov_analyses(experiments, cfg.mechanisms)

    print(String(t"Running Monte Carlo ({bench_reps} replications, Mojo)..."))
    var summaries = List[ExperimentSummary](capacity=len(experiments) + 1)
    for i in range(len(experiments)):
        summaries.append(
            run_experiment_monte_carlo(
                experiments[i],
                inv.variants,
                inv.rolls,
                inv.freq_calm,
                inv.freq_rush,
                inv.co_matrix,
                inv.num_variants,
                inv.high_assoc_siblings,
                cfg.mechanisms,
                bench_reps,
                2026,
            )
        )

    print("Running capacity sweep (Mojo)...")
    var sweep = _run_capacity_sweep(cfg, inv, sweep_reps)

    print("Running layout search (825 orientations across 165 feasible topologies, Mojo)...")
    var search_res = run_orientation_search(
        cfg,
        inv,
        search_reps,
        lat_w,
        out_dir + "/orientation_search_400.csv",
        out_dir + "/topology_summary_100.csv",
    )
    var ori_records = search_res[0].copy()

    var champ_idx = 0
    for i in range(1, len(ori_records)):
        if ori_records[i].composite_score > ori_records[champ_idx].composite_score:
            champ_idx = i
    var cr = ori_records[champ_idx].copy()
    var c_scen_orig = cr.scenario.copy()
    var champ_scen = ExperimentScenario(
        "ORI-CHAMP",
        String(t"Champion {cr.ori_id} ({cr.topo_id})"),
        c_scen_orig.topology,
        c_scen_orig.strategy,
        c_scen_orig.drawer_partitions,
        c_scen_orig.cue_salience,
        c_scen_orig.target_fill_ratio,
        c_scen_orig.description,
        c_scen_orig.row0,
        c_scen_orig.row1,
        c_scen_orig.row2,
        c_scen_orig.geometry.copy(),
        c_scen_orig.rush.copy(),
        c_scen_orig.behavior.copy(),
        c_scen_orig.abc_colocation_mode,
        c_scen_orig.overflow_mode,
        c_scen_orig.spatial_order_mode,
        c_scen_orig.layout_seed,
    )

    print(String(t"Verifying {cr.ori_id} ({bench_reps} replications, Mojo)..."))
    var champ_mc = run_experiment_monte_carlo(
        champ_scen,
        inv.variants,
        inv.rolls,
        inv.freq_calm,
        inv.freq_rush,
        inv.co_matrix,
        inv.num_variants,
        inv.high_assoc_siblings,
        cfg.mechanisms,
        bench_reps,
        2026,
    )
    summaries.append(champ_mc.copy())
    var champ_mr = run_all_markov_analyses([champ_scen.copy()], cfg.mechanisms)[0].copy()
    markov_results.append(champ_mr.copy())

    print(
        String(
            t"Best: {cr.ori_id} ({cr.topo_id}) -> Acc1000={fmt_float(champ_mc.final_acc_mean * 100.0, 1)}%, Latency={fmt_float(champ_mc.overall_latency_mean, 2)}s, pi_Home={fmt_float(champ_mr.stationary_dist[0] * 100.0, 1)}%"
        )
    )

    _export_markov_csv(markov_results, out_dir + "/markov_summary.csv")
    _export_experiment_csv(summaries, out_dir + "/experiment_summary.csv")
    _export_sweep_csv(sweep, out_dir + "/capacity_sweep.csv")
    _export_slotting_plan(champ_scen, inv, out_dir + "/optimal_slotting_120x60x60.csv")

    var all_benchmarks = List[ExperimentScenario](capacity=len(experiments) + 1)
    for i in range(len(experiments)):
        all_benchmarks.append(experiments[i].copy())
    all_benchmarks.append(champ_scen.copy())
    _export_notebook_data(
        cfg, inv, all_benchmarks, markov_results, cr.ori_id, out_dir + "/notebook_data.json"
    )

    if not no_charts:
        print("Generating figures (Mojo dataviz)...")
        generate_all_charts(summaries, sweep, ori_records, fig_dir)

    var elapsed = Float64(perf_counter_ns() - t0) / 1e9
    print(String(t"Done in {fmt_float(elapsed, 2)}s."))
