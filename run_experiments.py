from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from generate_charts import (
    plot_fig1_entropy_decay,
    plot_fig2_cross_tier_contamination,
    plot_fig3_slot_heatmap,
    plot_fig4_capacity_and_partitions,
    plot_fig5_orientation_search_pareto,
)
from markov_analysis import run_all_markov_analyses
from orientation_search import run_orientation_search, summarize_by_dimension
from sim_config import (
    DEFAULT_CSV_PATH,
    ExperimentScenario,
    build_default_experiments,
    get_execution_config,
    load_roll_inventory,
    save_config_file,
    set_active_config,
)
from simulation_engine import (
    ExperimentSummary,
    allocate_home_slots,
    run_experiment_monte_carlo,
)

WORKSPACE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = WORKSPACE_DIR / "output"
FIGURES_DIR = WORKSPACE_DIR / "figures"


def _run_mc_task(args: Tuple[ExperimentScenario, int, int]) -> ExperimentSummary:
    scen, n_reps, seed = args
    return run_experiment_monte_carlo(scen, n_replications=n_reps, base_seed=seed)


def run_capacity_sweep(
    fill_ratios: Tuple[float, ...] = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98),
    n_replications: int = 25,
    pool: ProcessPoolExecutor | None = None,
) -> Dict[str, List[Tuple[float, float, float]]]:
    """Sweep target fill ratio rho across core modular 120x60x60 cm layout architectures."""
    configs = {
        "TOPO01_Mid7D4P_Btm1B4D4P": (
            ("7D_4P", "1B_4D_4P", "SLIDING_PEG"),
            "adjacent_partition",
            "hybrid_reserve",
            "center_out",
            4,
            0.90,
        ),
        "TOPO07_Mid7D4P_Btm2B": (
            ("7D_4P", "2B", "SLIDING_PEG"),
            "adjacent_partition",
            "hybrid_reserve",
            "center_out",
            4,
            0.90,
        ),
        "TOPO07_BaselineCOI": (
            ("7D_4P", "2B", "SLIDING_PEG"),
            "cross_tier_height",
            "all_colocated",
            "random_perm",
            4,
            0.20,
        ),
    }
    tasks: List[Tuple[str, float, Tuple[ExperimentScenario, int, int]]] = []
    for label, (row_cfgs, abc_mode, ov_mode, sp_mode, parts, sigma) in configs.items():
        for rho in fill_ratios:
            scen = ExperimentScenario(
                exp_id=f"SWEEP-{label}-{int(rho*100)}",
                name=label,
                topology="M_modular",
                strategy="C_hybrid" if ov_mode == "hybrid_reserve" else "B_kit",
                drawer_partitions=parts,
                cue_salience=sigma,
                target_fill_ratio=rho,
                row_configs=row_cfgs,
                abc_colocation_mode=abc_mode,
                overflow_mode=ov_mode,
                spatial_order_mode=sp_mode,
                layout_seed=1000,
            )
            tasks.append((label, rho, (scen, n_replications, 5000)))

    if pool is not None:
        results = list(pool.map(_run_mc_task, [t[2] for t in tasks]))
    else:
        results = [_run_mc_task(t[2]) for t in tasks]

    sweep_out: Dict[str, List[Tuple[float, float, float]]] = {k: [] for k in configs}
    for (label, rho, _), res in zip(tasks, results):
        sweep_out[label].append((rho, res.final_acc_mean, res.overall_latency_mean))
    return sweep_out


def export_slotting_plan(
    winning_scenario: ExperimentScenario, out_csv: Path
) -> None:
    """Export the exact 128-roll physical slotting schematic for the winning layout."""
    variants, rolls, _, _, _ = load_roll_inventory(DEFAULT_CSV_PATH, winning_scenario.rush)
    slots = allocate_home_slots(winning_scenario, variants, rolls)
    slots_by_id = {s.slot_id: s for s in slots}

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "roll_id",
            "sku_code",
            "product_name",
            "variant_code",
            "copy_index",
            "category",
            "outer_diameter_cm",
            "roll_height_cm",
            "has_core",
            "is_rush_priority",
            "assigned_mechanism",
            "assigned_slot",
            "home_tier",
            "bay_side",
            "x_center_cm",
        ])
        for r in sorted(rolls, key=lambda x: (x.home_mechanism, x.home_slot, x.roll_id)):
            s = slots_by_id[r.home_slot]
            writer.writerow([
                r.roll_id,
                r.sku_code,
                r.product_name,
                r.variant_code,
                r.copy_index + 1,
                r.category,
                r.outer_diameter_cm,
                r.roll_height_cm,
                r.has_core,
                r.is_rush_priority,
                r.home_mechanism,
                r.home_slot,
                s.tier_index,
                "Left(0-60cm)" if s.bay_side == 0 else "Right(60-120cm)",
                s.x_center_cm,
            ])


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run shelf label storage Markov analysis, Monte Carlo experiments, and chart generation."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to a .yaml/.yml, .json, or .toon configuration file (defaults to shelf_config.yaml).",
    )
    parser.add_argument(
        "--dump-config",
        type=str,
        default=None,
        help="Write the current active configuration to a .yaml, .json, or .toon file and exit.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast iteration mode (10 benchmark reps, 8 sweep reps, 2 search reps).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.config:
        set_active_config(Path(args.config))
    if args.dump_config:
        written = save_config_file(Path(args.dump_config))
        print(f"Wrote configuration template to {written}")
        return

    exec_cfg = get_execution_config()
    if args.quick:
        bench_reps, sweep_reps, search_reps = 10, 8, 2
    else:
        bench_reps = int(exec_cfg["benchmark_replications"])
        sweep_reps = int(exec_cfg["sweep_replications"])
        search_reps = int(exec_cfg["search_replications"])
    lat_weight = float(exec_cfg["composite_score_latency_weight"])

    t0 = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    experiments = build_default_experiments()

    print("Running Markov analysis...")
    markov_results = run_all_markov_analyses(experiments)

    print(f"Running Monte Carlo ({bench_reps} replications)...")
    summaries: Dict[str, ExperimentSummary] = {}
    n_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        exp_results = list(pool.map(_run_mc_task, [(exp, bench_reps, 2026) for exp in experiments]))
        for exp, res in zip(experiments, exp_results):
            summaries[exp.exp_id] = res

        print("Running capacity sweep...")
        sweep = run_capacity_sweep(n_replications=sweep_reps, pool=pool)

    print("Running layout search (500 orientations)...")
    ori_records, _ = run_orientation_search(
        n_replications=search_reps,
        out_csv=OUTPUT_DIR / "orientation_search_400.csv",
        latency_weight=lat_weight,
    )

    champion_rec = max(ori_records, key=lambda r: r.composite_score)
    assert champion_rec.scenario is not None
    champ_scen = ExperimentScenario(
        exp_id="ORI-CHAMP",
        name=f"Champion {champion_rec.ori_id} ({champion_rec.topo_id})",
        topology=champion_rec.scenario.topology,
        strategy=champion_rec.scenario.strategy,
        drawer_partitions=champion_rec.scenario.drawer_partitions,
        cue_salience=champion_rec.scenario.cue_salience,
        target_fill_ratio=champion_rec.scenario.target_fill_ratio,
        description=champion_rec.scenario.description,
        row_configs=champion_rec.scenario.row_configs,
        abc_colocation_mode=champion_rec.scenario.abc_colocation_mode,
        overflow_mode=champion_rec.scenario.overflow_mode,
        spatial_order_mode=champion_rec.scenario.spatial_order_mode,
        layout_seed=champion_rec.scenario.layout_seed,
    )
    print(f"Verifying {champion_rec.ori_id} ({bench_reps} replications)...")
    champ_mc = run_experiment_monte_carlo(champ_scen, n_replications=bench_reps, base_seed=2026)
    summaries["ORI-CHAMP"] = champ_mc
    champ_mr = run_all_markov_analyses([champ_scen])[0]
    markov_results.append(champ_mr)
    print(
        f"Best: {champion_rec.ori_id} ({champion_rec.topo_id}) -> "
        f"Acc1000={champ_mc.final_acc_mean*100:.1f}%, Latency={champ_mc.overall_latency_mean:.2f}s, "
        f"pi_Home={champ_mr.stationary_dist[0]*100:.1f}%"
    )

    with open(OUTPUT_DIR / "markov_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "exp_id", "is_irreducible", "period", "max_row_sum_error",
            "pi_home", "pi_drift_adj", "pi_eye_dump", "pi_wire_dump", "pi_peg_dump", "pi_buried",
            "spectral_gap", "mixing_time_95", "mfpt_home_to_misplaced", "mfpt_misplaced_to_home",
            "mfpt_home_to_buried", "sim_tvd",
        ])
        for mr in markov_results:
            w.writerow([
                mr.exp_id, mr.is_irreducible, mr.period, f"{mr.max_row_sum_error:.2e}",
                *[f"{v:.5f}" for v in mr.stationary_dist],
                f"{mr.spectral_gap:.5f}", f"{mr.mixing_time_95:.2f}",
                f"{mr.mfpt_home_to_any_misplaced:.2f}", f"{mr.mfpt_misplaced_to_home:.2f}",
                f"{mr.mfpt_home_to_buried:.2f}", f"{mr.total_variation_distance:.5f}",
            ])

    with open(OUTPUT_DIR / "experiment_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "exp_id", "name", "topology", "strategy", "drawer_partitions",
            "cue_salience", "target_fill_ratio", "final_acc_mean", "final_acc_std",
            "final_entropy", "final_cross_tier_contam", "mean_job_latency_sec",
            "rush_latency_sec", "calm_latency_sec", "mean_t_crit", "collapse_rate",
        ])
        for exp in list(experiments) + [champ_scen]:
            res = summaries[exp.exp_id]
            w.writerow([
                exp.exp_id, exp.name, exp.topology, exp.strategy, exp.drawer_partitions,
                exp.cue_salience, exp.target_fill_ratio,
                f"{res.final_acc_mean:.4f}", f"{res.final_acc_std:.4f}",
                f"{res.final_entropy_mean:.4f}", f"{res.final_contam_mean:.4f}",
                f"{res.overall_latency_mean:.2f}", f"{res.rush_latency_mean:.2f}",
                f"{res.calm_latency_mean:.2f}", f"{res.t_crit_mean:.1f}",
                f"{res.collapse_rate:.2f}",
            ])

    with open(OUTPUT_DIR / "capacity_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layout", "fill_ratio", "final_accuracy", "mean_latency_sec"])
        for label, pts in sweep.items():
            for rho, acc, lat in pts:
                w.writerow([label, rho, f"{acc:.4f}", f"{lat:.2f}"])

    export_slotting_plan(champ_scen, OUTPUT_DIR / "optimal_slotting_120x60x60.csv")

    print("Generating figures...")
    plot_fig1_entropy_decay(summaries, FIGURES_DIR)
    plot_fig2_cross_tier_contamination(summaries, FIGURES_DIR)
    plot_fig3_slot_heatmap(summaries, FIGURES_DIR)
    plot_fig4_capacity_and_partitions(sweep, summaries, FIGURES_DIR)
    plot_fig5_orientation_search_pareto(ori_records, FIGURES_DIR)

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()
