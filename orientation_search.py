from __future__ import annotations

import csv
import os
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sim_config import (
    DEFAULT_CSV_PATH,
    ExperimentScenario,
    RollItem,
    RollVariantSKU,
    get_execution_config,
    load_roll_inventory,
)
from simulation_engine import (
    ExperimentSummary,
    allocate_home_slots,
    compute_layout_metrics,
    run_experiment_monte_carlo,
)

LOWER_SHELF_LAYOUTS: Tuple[str, ...] = (
    "7D_4P",
    "7D_3P",
    "4D4P_3D3P",
    "1B_4D_4P",
    "1B_4D_3P",
    "1B_2D4P_2D3P",
    "1D_2B",
    "1B_1D_1B",
    "2B_1D",
    "2B",
)

MODULAR_TOPOLOGIES: List[Tuple[str, str, Tuple[str, str, str]]] = [
    (
        f"TOPO-{idx:03d}",
        f"Top: Peg | Mid: {mid} | Btm: {btm}",
        (mid, btm, "SLIDING_PEG"),
    )
    for idx, (mid, btm) in enumerate(
        [(m, b) for m in LOWER_SHELF_LAYOUTS for b in LOWER_SHELF_LAYOUTS],
        start=1,
    )
]

POSITIONING_POLICIES: List[Tuple[str, str, str]] = [
    ("adjacent_partition", "hybrid_reserve", "center_out"),
    ("same_slot", "hybrid_reserve", "center_out"),
    ("same_tier_bay", "hybrid_reserve", "center_out"),
    ("cross_tier_height", "hybrid_reserve", "center_out"),
    ("cross_tier_height", "all_colocated", "random_perm"),
]


@dataclass
class OrientationRecord:
    """Single evaluated shelf topology + label positioning orientation."""

    ori_id: str
    topo_id: str
    topo_label: str
    row_configs: Tuple[str, str, str]
    abc_colocation_mode: str
    overflow_mode: str
    spatial_order_mode: str
    layout_seed: int
    abc_same_drawer_pct: float
    abc_same_tier_pct: float
    active_eye_share: float
    mean_eye_fill_ratio: float
    final_acc_mean: float
    final_acc_std: float
    final_entropy_mean: float
    final_contam_mean: float
    overall_latency_mean: float
    rush_latency_mean: float
    calm_latency_mean: float
    collapse_rate: float
    composite_score: float
    is_pareto_optimal: bool = False
    scenario: Optional[ExperimentScenario] = None


_WORKER_INVENTORY: Optional[
    Tuple[List[RollVariantSKU], List[RollItem], List[float], List[float], List[List[float]]]
] = None


def _init_search_worker() -> None:
    global _WORKER_INVENTORY
    _WORKER_INVENTORY = load_roll_inventory(DEFAULT_CSV_PATH)


def build_orientation_scenarios() -> List[Tuple[str, str, ExperimentScenario]]:
    """Build all (100 topologies x 5 policies = 500) candidate scenarios."""
    candidates: List[Tuple[str, str, ExperimentScenario]] = []
    ori_num = 1
    for t_idx, (topo_id, topo_label, row_cfgs) in enumerate(MODULAR_TOPOLOGIES):
        p_eye = 3 if row_cfgs[0] in ("7D_3P", "1B_4D_3P") else 4
        for p_idx, (abc_mode, ov_mode, sp_mode) in enumerate(POSITIONING_POLICIES):
            ori_id = f"ORI-{ori_num:03d}"
            seed = 1000 + t_idx * 31 + p_idx * 7
            scen = ExperimentScenario(
                exp_id=ori_id,
                name=f"{topo_id} | {abc_mode} | {ov_mode} | {sp_mode}",
                topology="M_modular",
                strategy="C_hybrid" if ov_mode == "hybrid_reserve" else "B_kit",
                drawer_partitions=p_eye,
                cue_salience=(
                    0.20
                    if (sp_mode == "random_perm" and abc_mode == "cross_tier_height")
                    else (
                        0.35
                        if sp_mode == "random_perm"
                        else (
                            0.76
                            if sp_mode == "vertical_column"
                            else (0.84 if sp_mode == "category_zones" else 0.90)
                        )
                    )
                ),
                target_fill_ratio=(
                    0.90
                    if (sp_mode == "random_perm" and abc_mode == "cross_tier_height")
                    else (
                        0.88
                        if sp_mode == "random_perm"
                        else (0.86 if ov_mode == "all_colocated" else 0.80)
                    )
                ),
                description=f"{topo_label} ({abc_mode}, {ov_mode}, {sp_mode})",
                row_configs=row_cfgs,
                abc_colocation_mode=abc_mode,
                overflow_mode=ov_mode,
                spatial_order_mode=sp_mode,
                layout_seed=seed,
            )
            candidates.append((topo_id, topo_label, scen))
            ori_num += 1
    return candidates


def _eval_orientation_task(
    args: Tuple[str, str, ExperimentScenario, int, int, float]
) -> OrientationRecord:
    topo_id, topo_label, scen, n_reps, base_seed, lat_weight = args
    global _WORKER_INVENTORY
    if _WORKER_INVENTORY is None:
        _WORKER_INVENTORY = load_roll_inventory(DEFAULT_CSV_PATH, scen.rush)
    variants, base_rolls, _, _, _ = _WORKER_INVENTORY

    rolls_copy = [
        RollItem(
            roll_id=r.roll_id, variant_id=r.variant_id, sku_code=r.sku_code,
            product_name=r.product_name, variant_code=r.variant_code, copy_index=r.copy_index,
            outer_diameter_cm=r.outer_diameter_cm, roll_height_cm=r.roll_height_cm,
            has_core=r.has_core, category=r.category, is_rush_priority=r.is_rush_priority,
            visual_similarity_group=r.visual_similarity_group,
        )
        for r in base_rolls
    ]
    slots = allocate_home_slots(scen, variants, rolls_copy)
    struct = compute_layout_metrics(rolls_copy, slots)

    summary: ExperimentSummary = run_experiment_monte_carlo(
        scen,
        n_replications=n_reps,
        base_seed=base_seed,
        preloaded_inventory=_WORKER_INVENTORY,
    )
    score = 100.0 * summary.final_acc_mean - lat_weight * summary.overall_latency_mean
    assert scen.row_configs is not None
    return OrientationRecord(
        ori_id=scen.exp_id,
        topo_id=topo_id,
        topo_label=topo_label,
        row_configs=scen.row_configs,
        abc_colocation_mode=scen.abc_colocation_mode,
        overflow_mode=scen.overflow_mode,
        spatial_order_mode=scen.spatial_order_mode,
        layout_seed=scen.layout_seed,
        abc_same_drawer_pct=struct["abc_same_drawer_pct"],
        abc_same_tier_pct=struct["abc_same_tier_pct"],
        active_eye_share=struct["active_eye_share"],
        mean_eye_fill_ratio=struct["mean_eye_fill_ratio"],
        final_acc_mean=summary.final_acc_mean,
        final_acc_std=summary.final_acc_std,
        final_entropy_mean=summary.final_entropy_mean,
        final_contam_mean=summary.final_contam_mean,
        overall_latency_mean=summary.overall_latency_mean,
        rush_latency_mean=summary.rush_latency_mean,
        calm_latency_mean=summary.calm_latency_mean,
        collapse_rate=summary.collapse_rate,
        composite_score=score,
        is_pareto_optimal=False,
        scenario=scen,
    )


def mark_pareto_frontier(records: List[OrientationRecord]) -> List[OrientationRecord]:
    """Identify all non-dominated Pareto-optimal points maximizing Accuracy and minimizing Latency."""
    for rec in records:
        dominated = any(
            (other.final_acc_mean >= rec.final_acc_mean and other.overall_latency_mean <= rec.overall_latency_mean)
            and (other.final_acc_mean > rec.final_acc_mean or other.overall_latency_mean < rec.overall_latency_mean)
            for other in records
        )
        rec.is_pareto_optimal = not dominated
    frontier = [r for r in records if r.is_pareto_optimal]
    frontier.sort(key=lambda r: r.overall_latency_mean)
    return frontier


def run_orientation_search(
    n_replications: Optional[int] = None,
    out_csv: Optional[Path] = None,
    pool: Optional[ProcessPoolExecutor] = None,
    latency_weight: Optional[float] = None,
) -> Tuple[List[OrientationRecord], List[OrientationRecord]]:
    """Run all 400 shelf topology + label positioning orientations in parallel."""
    exec_cfg = get_execution_config()
    eff_reps = int(n_replications if n_replications is not None else exec_cfg["search_replications"])
    eff_lat_w = float(
        latency_weight if latency_weight is not None else exec_cfg["composite_score_latency_weight"]
    )
    candidates = build_orientation_scenarios()
    tasks = [
        (topo_id, topo_label, scen, eff_reps, 9000 + idx * 13, eff_lat_w)
        for idx, (topo_id, topo_label, scen) in enumerate(candidates)
    ]

    n_workers = (getattr(pool, "_max_workers", None) if pool is not None else None) or min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_search_worker) as search_pool:
        records = list(search_pool.map(_eval_orientation_task, tasks, chunksize=8))

    pareto = mark_pareto_frontier(records)

    if out_csv is not None:
        export_orientation_csv(records, out_csv)
        export_topology_summary_csv(records, out_csv.parent / "topology_summary_100.csv")

    return records, pareto


def export_topology_summary_csv(records: Sequence[OrientationRecord], out_csv: Path) -> None:
    """Write ranked performance summary for every evaluated (Mid, Btm) hardware topology."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    by_topo: Dict[str, List[OrientationRecord]] = {}
    for r in records:
        by_topo.setdefault(r.topo_id, []).append(r)

    rows = []
    for topo_id, items in by_topo.items():
        best_r = max(items, key=lambda x: x.composite_score)
        mean_acc = statistics.mean(x.final_acc_mean for x in items)
        mean_lat = statistics.mean(x.overall_latency_mean for x in items)
        rows.append((best_r, mean_acc, mean_lat))

    rows.sort(key=lambda t: (-t[0].composite_score, t[0].overall_latency_mean))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "topo_id", "topo_label",
            "mid_shelf", "btm_shelf", "top_shelf",
            "best_ori_id", "best_abc_mode",
            "best_acc_mean", "best_latency_sec",
            "mean_acc_across_policies", "mean_latency_sec",
            "best_composite_score",
        ])
        for idx, (best_r, mean_acc, mean_lat) in enumerate(rows, start=1):
            w.writerow([
                idx,
                best_r.topo_id,
                best_r.topo_label,
                best_r.row_configs[0],
                best_r.row_configs[1],
                best_r.row_configs[2],
                best_r.ori_id,
                best_r.abc_colocation_mode,
                f"{best_r.final_acc_mean:.4f}",
                f"{best_r.overall_latency_mean:.2f}",
                f"{mean_acc:.4f}",
                f"{mean_lat:.2f}",
                f"{best_r.composite_score:.2f}",
            ])


def export_orientation_csv(records: Sequence[OrientationRecord], out_csv: Path) -> None:
    """Write all evaluated orientations sorted by composite score descending."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(records, key=lambda r: -r.composite_score)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "ori_id", "topo_id", "topo_label",
            "row1_eye", "row2_mid", "row3_low_top",
            "abc_colocation_mode", "overflow_mode", "spatial_order_mode",
            "abc_same_drawer_pct", "abc_same_tier_pct", "active_eye_share", "mean_eye_fill_ratio",
            "final_acc_mean", "final_acc_std", "final_entropy_mean", "final_contam_mean",
            "overall_latency_sec", "rush_latency_sec", "calm_latency_sec",
            "collapse_rate", "composite_score", "is_pareto_optimal",
        ])
        for idx, r in enumerate(ranked, start=1):
            w.writerow([
                idx, r.ori_id, r.topo_id, r.topo_label,
                r.row_configs[0], r.row_configs[1], r.row_configs[2],
                r.abc_colocation_mode, r.overflow_mode, r.spatial_order_mode,
                f"{r.abc_same_drawer_pct:.3f}", f"{r.abc_same_tier_pct:.3f}",
                f"{r.active_eye_share:.3f}", f"{r.mean_eye_fill_ratio:.3f}",
                f"{r.final_acc_mean:.4f}", f"{r.final_acc_std:.4f}",
                f"{r.final_entropy_mean:.4f}", f"{r.final_contam_mean:.4f}",
                f"{r.overall_latency_mean:.2f}", f"{r.rush_latency_mean:.2f}",
                f"{r.calm_latency_mean:.2f}", f"{r.collapse_rate:.2f}",
                f"{r.composite_score:.2f}", r.is_pareto_optimal,
            ])


def summarize_by_dimension(records: Sequence[OrientationRecord]) -> Dict[str, List[Tuple[str, int, float, float, float]]]:
    """Aggregate mean accuracy, latency, and max accuracy by positioning mode and topology."""
    by_abc: Dict[str, List[OrientationRecord]] = {}
    by_topo: Dict[str, List[OrientationRecord]] = {}
    by_ovf: Dict[str, List[OrientationRecord]] = {}
    for r in records:
        by_abc.setdefault(r.abc_colocation_mode, []).append(r)
        by_topo.setdefault(f"{r.topo_id}: {r.topo_label}", []).append(r)
        by_ovf.setdefault(r.overflow_mode, []).append(r)

    def _agg(d: Dict[str, List[OrientationRecord]]) -> List[Tuple[str, int, float, float, float]]:
        out = []
        for k, items in d.items():
            mean_acc = statistics.mean(x.final_acc_mean for x in items)
            mean_lat = statistics.mean(x.overall_latency_mean for x in items)
            best_acc = max(x.final_acc_mean for x in items)
            out.append((k, len(items), mean_acc, mean_lat, best_acc))
        out.sort(key=lambda t: (-t[2], t[3]))
        return out

    return {
        "abc_mode": _agg(by_abc),
        "topology": _agg(by_topo),
        "overflow_mode": _agg(by_ovf),
    }
