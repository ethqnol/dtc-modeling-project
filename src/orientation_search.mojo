from std.collections import List
from std.pathlib import Path

from src.format_utils import csv_cell, fmt_float
from src.sim_config import (
    AppConfig,
    ExperimentScenario,
    InventoryData,
)
from src.simulation_engine import (
    allocate_home_slots,
    compute_layout_metrics,
    run_experiment_monte_carlo,
)


def _zpad3(n: Int) -> String:
    if n < 10:
        return "00" + String(n)
    if n < 100:
        return "0" + String(n)
    return String(n)


def lower_shelf_layouts() -> List[String]:
    return [
        "7D6_4P",
        "7D6_3P",
        "5D9_4P",
        "5D9_3P",
        "3D9_3D6_4P",
        "1B_4D6_4P",
        "1B_4D6_3P",
        "1B_3D9_4P",
        "1B_3D9_3P",
        "1D6_2B",
        "1B_1D6_1B",
        "2B_1D6",
        "2B",
    ]


def _fits_all_labels(rc: String) -> Bool:
    return (
        rc == "5D9_4P"
        or rc == "5D9_3P"
        or rc == "3D9_3D6_4P"
        or rc == "1B_4D6_4P"
        or rc == "1B_4D6_3P"
        or rc == "1B_3D9_4P"
        or rc == "1B_3D9_3P"
        or rc == "1D6_2B"
        or rc == "1B_1D6_1B"
        or rc == "2B_1D6"
        or rc == "2B"
    )


@fieldwise_init
struct ModularTopology(Copyable, Movable):
    var topo_id: String
    var topo_label: String
    var row0: String
    var row1: String
    var row2: String


def build_modular_topologies() -> List[ModularTopology]:
    var layouts = lower_shelf_layouts()
    var out = List[ModularTopology]()
    var idx = 1
    for mid in layouts:
        for btm in layouts:
            if _fits_all_labels(mid) or _fits_all_labels(btm):
                var tid = "TOPO-" + _zpad3(idx)
                var lbl = String(t"Top: Peg | Mid: {mid} | Btm: {btm}")
                out.append(ModularTopology(tid, lbl, mid, btm, "SLIDING_PEG"))
                idx += 1
    return out^


@fieldwise_init
struct PositioningPolicy(Copyable, Movable):
    var abc_mode: String
    var overflow_mode: String
    var spatial_mode: String


def build_positioning_policies() -> List[PositioningPolicy]:
    var policies = List[PositioningPolicy](capacity=5)
    policies.append(PositioningPolicy("adjacent_partition", "hybrid_reserve", "center_out"))
    policies.append(PositioningPolicy("same_slot", "hybrid_reserve", "center_out"))
    policies.append(PositioningPolicy("same_tier_bay", "hybrid_reserve", "center_out"))
    policies.append(PositioningPolicy("cross_tier_height", "hybrid_reserve", "center_out"))
    policies.append(PositioningPolicy("cross_tier_height", "all_colocated", "random_perm"))
    return policies^


@fieldwise_init
struct OrientationCandidate(Copyable, Movable):
    var topo_id: String
    var topo_label: String
    var scenario: ExperimentScenario


@fieldwise_init
struct OrientationRecord(Copyable, Movable):
    var ori_id: String
    var topo_id: String
    var topo_label: String
    var row0: String
    var row1: String
    var row2: String
    var abc_colocation_mode: String
    var overflow_mode: String
    var spatial_order_mode: String
    var layout_seed: Int
    var abc_same_drawer_pct: Float64
    var abc_same_tier_pct: Float64
    var active_eye_share: Float64
    var mean_eye_fill_ratio: Float64
    var final_acc_mean: Float64
    var final_acc_std: Float64
    var final_entropy_mean: Float64
    var final_contam_mean: Float64
    var overall_latency_mean: Float64
    var rush_latency_mean: Float64
    var calm_latency_mean: Float64
    var collapse_rate: Float64
    var composite_score: Float64
    var is_pareto_optimal: Bool
    var scenario: ExperimentScenario


def build_orientation_scenarios(cfg: AppConfig) -> List[OrientationCandidate]:
    """Build all feasible (165 topologies x 5 policies = 825) candidate scenarios."""
    var topos = build_modular_topologies()
    var policies = build_positioning_policies()
    var out = List[OrientationCandidate](capacity=len(topos) * len(policies))
    var ori_num = 1

    for t_idx in range(len(topos)):
        var tp = topos[t_idx].copy()
        var p_eye = 3 if tp.row0.endswith("_3P") else 4
        for p_idx in range(len(policies)):
            var pol = policies[p_idx].copy()
            var ori_id = "ORI-" + _zpad3(ori_num)
            var seed = 1000 + t_idx * 31 + p_idx * 7
            var cue_sal = (
                0.20
                if (pol.spatial_mode == "random_perm" and pol.abc_mode == "cross_tier_height")
                else (
                    0.35
                    if pol.spatial_mode == "random_perm"
                    else (
                        0.76
                        if pol.spatial_mode == "vertical_column"
                        else (0.84 if pol.spatial_mode == "category_zones" else 0.90)
                    )
                )
            )
            var fill_ratio = (
                0.90
                if (pol.spatial_mode == "random_perm" and pol.abc_mode == "cross_tier_height")
                else (
                    0.88
                    if pol.spatial_mode == "random_perm"
                    else (0.86 if pol.overflow_mode == "all_colocated" else 0.80)
                )
            )
            var scen = ExperimentScenario(
                ori_id,
                String(t"{tp.topo_id} | {pol.abc_mode} | {pol.overflow_mode} | {pol.spatial_mode}"),
                "M_modular",
                "C_hybrid" if pol.overflow_mode == "hybrid_reserve" else "B_kit",
                p_eye,
                cue_sal,
                fill_ratio,
                String(t"{tp.topo_label} ({pol.abc_mode}, {pol.overflow_mode}, {pol.spatial_mode})"),
                tp.row0,
                tp.row1,
                tp.row2,
                cfg.geometry.copy(),
                cfg.rush.copy(),
                cfg.behavior.copy(),
                pol.abc_mode,
                pol.overflow_mode,
                pol.spatial_mode,
                seed,
            )
            out.append(OrientationCandidate(tp.topo_id, tp.topo_label, scen^))
            ori_num += 1
    return out^


def _cmp_pareto_lat(a: OrientationRecord, b: OrientationRecord) -> Bool:
    return a.overall_latency_mean < b.overall_latency_mean


def _cmp_score_desc(a: OrientationRecord, b: OrientationRecord) -> Bool:
    if a.composite_score != b.composite_score:
        return a.composite_score > b.composite_score
    return a.overall_latency_mean < b.overall_latency_mean


def mark_pareto_frontier(mut records: List[OrientationRecord]) -> List[OrientationRecord]:
    var n = len(records)
    var frontier = List[OrientationRecord]()
    for i in range(n):
        var dominated = False
        var acc_i = records[i].final_acc_mean
        var lat_i = records[i].overall_latency_mean
        for j in range(n):
            var acc_j = records[j].final_acc_mean
            var lat_j = records[j].overall_latency_mean
            if (acc_j >= acc_i and lat_j <= lat_i) and (acc_j > acc_i or lat_j < lat_i):
                dominated = True
                break
        records[i].is_pareto_optimal = not dominated
        if not dominated:
            frontier.append(records[i].copy())
    sort(frontier, _cmp_pareto_lat)
    return frontier^


@fieldwise_init
struct _TopoAgg(Copyable, Movable):
    var best_rec: OrientationRecord
    var mean_acc: Float64
    var mean_lat: Float64


def _cmp_topo_agg(a: _TopoAgg, b: _TopoAgg) -> Bool:
    if a.best_rec.composite_score != b.best_rec.composite_score:
        return a.best_rec.composite_score > b.best_rec.composite_score
    return a.best_rec.overall_latency_mean < b.best_rec.overall_latency_mean


def export_orientation_csv(records: List[OrientationRecord], out_csv: String) raises:
    var ranked = records.copy()
    sort(ranked, _cmp_score_desc)
    var lines = List[String](capacity=len(ranked) + 1)
    lines.append(
        "rank,ori_id,topo_id,topo_label,"
        "row1_eye,row2_mid,row3_low_top,"
        "abc_colocation_mode,overflow_mode,spatial_order_mode,"
        "abc_same_drawer_pct,abc_same_tier_pct,active_eye_share,mean_eye_fill_ratio,"
        "final_acc_mean,final_acc_std,final_entropy_mean,final_contam_mean,"
        "overall_latency_sec,rush_latency_sec,calm_latency_sec,"
        "collapse_rate,composite_score,is_pareto_optimal"
    )
    for i in range(len(ranked)):
        var r = ranked[i].copy()
        var pareto_str = "True" if r.is_pareto_optimal else "False"
        var row = (
            String(i + 1)
            + "," + csv_cell(r.ori_id)
            + "," + csv_cell(r.topo_id)
            + "," + csv_cell(r.topo_label)
            + "," + csv_cell(r.row0)
            + "," + csv_cell(r.row1)
            + "," + csv_cell(r.row2)
            + "," + csv_cell(r.abc_colocation_mode)
            + "," + csv_cell(r.overflow_mode)
            + "," + csv_cell(r.spatial_order_mode)
            + "," + fmt_float(r.abc_same_drawer_pct, 3)
            + "," + fmt_float(r.abc_same_tier_pct, 3)
            + "," + fmt_float(r.active_eye_share, 3)
            + "," + fmt_float(r.mean_eye_fill_ratio, 3)
            + "," + fmt_float(r.final_acc_mean, 4)
            + "," + fmt_float(r.final_acc_std, 4)
            + "," + fmt_float(r.final_entropy_mean, 4)
            + "," + fmt_float(r.final_contam_mean, 4)
            + "," + fmt_float(r.overall_latency_mean, 2)
            + "," + fmt_float(r.rush_latency_mean, 2)
            + "," + fmt_float(r.calm_latency_mean, 2)
            + "," + fmt_float(r.collapse_rate, 2)
            + "," + fmt_float(r.composite_score, 2)
            + "," + pareto_str
        )
        lines.append(row)
    Path(out_csv).write_text("\n".join(lines) + "\n")


def export_topology_summary_csv(records: List[OrientationRecord], out_csv: String) raises:
    var topo_ids = List[String]()
    var topo_groups = List[List[Int]]()
    for i in range(len(records)):
        var tid = records[i].topo_id
        var pos = -1
        for k in range(len(topo_ids)):
            if topo_ids[k] == tid:
                pos = k
                break
        if pos < 0:
            pos = len(topo_ids)
            topo_ids.append(tid)
            topo_groups.append(List[Int]())
        topo_groups[pos].append(i)

    var aggs = List[_TopoAgg](capacity=len(topo_ids))
    for grp in topo_groups:
        var best_idx = grp[0]
        var sum_acc: Float64 = 0.0
        var sum_lat: Float64 = 0.0
        for idx in grp:
            var r = records[idx].copy()
            sum_acc += r.final_acc_mean
            sum_lat += r.overall_latency_mean
            if r.composite_score > records[best_idx].composite_score:
                best_idx = idx
        var n_g = Float64(len(grp))
        aggs.append(_TopoAgg(records[best_idx].copy(), sum_acc / n_g, sum_lat / n_g))

    sort(aggs, _cmp_topo_agg)
    var lines = List[String](capacity=len(aggs) + 1)
    lines.append(
        "rank,topo_id,topo_label,"
        "mid_shelf,btm_shelf,top_shelf,"
        "best_ori_id,best_abc_mode,"
        "best_acc_mean,best_latency_sec,"
        "mean_acc_across_policies,mean_latency_sec,"
        "best_composite_score"
    )
    for i in range(len(aggs)):
        var ag = aggs[i].copy()
        var br = ag.best_rec.copy()
        var row = (
            String(i + 1)
            + "," + csv_cell(br.topo_id)
            + "," + csv_cell(br.topo_label)
            + "," + csv_cell(br.row0)
            + "," + csv_cell(br.row1)
            + "," + csv_cell(br.row2)
            + "," + csv_cell(br.ori_id)
            + "," + csv_cell(br.abc_colocation_mode)
            + "," + fmt_float(br.final_acc_mean, 4)
            + "," + fmt_float(br.overall_latency_mean, 2)
            + "," + fmt_float(ag.mean_acc, 4)
            + "," + fmt_float(ag.mean_lat, 2)
            + "," + fmt_float(br.composite_score, 2)
        )
        lines.append(row)
    Path(out_csv).write_text("\n".join(lines) + "\n")


def run_orientation_search(
    cfg: AppConfig,
    inv: InventoryData,
    n_replications: Int = -1,
    latency_weight: Float64 = -1.0,
    out_ori_csv: String = "",
    out_topo_csv: String = "",
) raises -> Tuple[List[OrientationRecord], List[OrientationRecord]]:
    """Run all 825 shelf topology x positioning orientations and export CSV summaries."""
    var eff_reps = n_replications if n_replications > 0 else cfg.execution.search_replications
    var eff_lat_w = latency_weight if latency_weight >= 0.0 else cfg.execution.composite_score_latency_weight
    var candidates = build_orientation_scenarios(cfg)
    var records = List[OrientationRecord](capacity=len(candidates))

    for idx in range(len(candidates)):
        var cand = candidates[idx].copy()
        var scen = cand.scenario.copy()
        var rolls_copy = inv.rolls.copy()
        var slots = allocate_home_slots(scen, inv.variants, rolls_copy)
        var lm = compute_layout_metrics(rolls_copy, slots)

        var summary = run_experiment_monte_carlo(
            scen,
            inv.variants,
            inv.rolls,
            inv.freq_calm,
            inv.freq_rush,
            inv.co_matrix,
            inv.num_variants,
            inv.high_assoc_siblings,
            cfg.mechanisms,
            eff_reps,
            9000 + idx * 13,
        )
        var score = 100.0 * summary.final_acc_mean - eff_lat_w * summary.overall_latency_mean
        records.append(
            OrientationRecord(
                scen.exp_id,
                cand.topo_id,
                cand.topo_label,
                scen.row0,
                scen.row1,
                scen.row2,
                scen.abc_colocation_mode,
                scen.overflow_mode,
                scen.spatial_order_mode,
                scen.layout_seed,
                lm.abc_same_drawer_pct,
                lm.abc_same_tier_pct,
                lm.active_eye_share,
                lm.mean_eye_fill_ratio,
                summary.final_acc_mean,
                summary.final_acc_std,
                summary.final_entropy_mean,
                summary.final_contam_mean,
                summary.overall_latency_mean,
                summary.rush_latency_mean,
                summary.calm_latency_mean,
                summary.collapse_rate,
                score,
                False,
                scen^,
            )
        )

    var pareto = mark_pareto_frontier(records)
    if out_ori_csv.byte_length() > 0:
        export_orientation_csv(records, out_ori_csv)
    if out_topo_csv.byte_length() > 0:
        export_topology_summary_csv(records, out_topo_csv)
    return (records^, pareto^)
