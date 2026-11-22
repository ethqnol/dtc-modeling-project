import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import copy
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    root_dir = Path(__file__).resolve().parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from src.markov_analysis import analyze_scenario_markov
    from src.orientation_search import (
        MODULAR_TOPOLOGIES,
        POSITIONING_POLICIES,
        build_orientation_scenarios,
    )
    from src.sim_config import (
        DEFAULT_CSV_PATH,
        ExperimentScenario,
        build_default_experiments,
        build_mechanism_specs,
        load_roll_inventory,
        make_behavioral_config,
        make_geometry_config,
        make_rush_config,
    )
    from src.simulation_engine import allocate_home_slots, compute_layout_metrics

    LAYOUT_NAMES = {
        "7D6_4P": "7D6_4P (7 x 6in drawers, 2x2)",
        "7D6_3P": "7D6_3P (7 x 6in drawers, 2:1:1)",
        "5D9_4P": "5D9_4P (5 x 9in drawers, 2x2)",
        "5D9_3P": "5D9_3P (5 x 9in drawers, 2:1:1)",
        "3D9_3D6_4P": "3D9_3D6_4P (3 x 9in + 3 x 6in, 2x2)",
        "1B_4D6_4P": "1B_4D6_4P (1 bin + 4 x 6in drawers)",
        "1B_4D6_3P": "1B_4D6_3P (1 bin + 4 x 6in drawers)",
        "1B_3D9_4P": "1B_3D9_4P (1 bin + 3 x 9in drawers)",
        "1B_3D9_3P": "1B_3D9_3P (1 bin + 3 x 9in drawers)",
        "1D6_2B": "1D6_2B (1 x 6in drawer + 2 bins)",
        "1B_1D6_1B": "1B_1D6_1B (bin + 1 x 6in drawer + bin)",
        "2B_1D6": "2B_1D6 (2 bins + 1 x 6in drawer)",
        "2B": "2B (2 wire bins)",
        "SLIDING_PEG": "Sliding peg (32 pegs)",
    }

    VARIANT_RULE_NAMES = {
        "adjacent_partition": "Adjacent lanes (same drawer)",
        "same_slot": "Shared lane",
        "same_tier_bay": "Same shelf bay",
        "cross_tier_height": "Split by roll height",
    }

    BACKUP_RULE_NAMES = {
        "hybrid_reserve": "Reserve on bottom shelf",
        "all_colocated": "All copies together",
    }

    ORDER_RULE_NAMES = {
        "center_out": "Center priority",
        "bark_sticker_kit": "Kit pairing (bark + sticker)",
        "category_zones": "Grouped by category",
        "vertical_column": "Aligned vertical columns",
        "random_perm": "Random placement",
    }

    MODULE_NAMES = {
        "split_drawer_eye": ("Mid Shelf Split Drawers", "Middle Shelf (140 cm)"),
        "split_drawer_mid": ("Bottom Shelf Split Drawers", "Bottom Shelf (25-80 cm)"),
        "wire_bin_eye": ("Mid Shelf Wire Bin", "Middle Shelf (140 cm)"),
        "wire_bin_mid": ("Bottom Shelf Wire Bin", "Bottom Shelf (25-80 cm)"),
        "pegboard_ovh": ("Top Shelf Sliding Peg", "Top Shelf (195 cm)"),
    }

    TIER_LABELS = {
        1: "Mid Shelf (140 cm)",
        2: "Bottom Shelf (25-80 cm)",
        3: "Top Shelf (195 cm)",
    }
    return (
        BACKUP_RULE_NAMES,
        DEFAULT_CSV_PATH,
        ExperimentScenario,
        LAYOUT_NAMES,
        MODULAR_TOPOLOGIES,
        MODULE_NAMES,
        ORDER_RULE_NAMES,
        POSITIONING_POLICIES,
        TIER_LABELS,
        VARIANT_RULE_NAMES,
        allocate_home_slots,
        analyze_scenario_markov,
        build_default_experiments,
        build_mechanism_specs,
        build_orientation_scenarios,
        compute_layout_metrics,
        copy,
        load_roll_inventory,
        make_behavioral_config,
        make_geometry_config,
        make_rush_config,
        mo,
        pd,
        root_dir,
    )


@app.cell(hide_code=True)
def _(mo, root_dir):
    _synthesis_text = (root_dir / "docs" / "synthesis.md").read_text(encoding="utf-8")
    mo.md(_synthesis_text)
    return


@app.cell(hide_code=True)
def _(
    MODULE_NAMES,
    build_mechanism_specs,
    make_behavioral_config,
    make_geometry_config,
    make_rush_config,
    mo,
    pd,
):
    geom = make_geometry_config()
    rush_cfg = make_rush_config()
    beh_cfg = make_behavioral_config()
    mech_specs = build_mechanism_specs("M_modular")

    _geom_df = pd.DataFrame([
        {"Parameter": "Shelf envelope (W x D x H cm)", "Value": f"{geom.shelf_width_cm:.0f} x {geom.shelf_depth_cm:.0f} x {geom.shelf_row_height_cm:.0f}"},
        {"Parameter": "Shelf heights (Mid / Btm / Top cm)", "Value": f"{geom.eye_level_height_cm:.0f} / {geom.bottom_level_height_cm:.0f}-{geom.mid_level_height_cm:.0f} / {geom.top_overhead_height_cm:.0f}"},
        {"Parameter": "6in drawer outer / inner width (cm)", "Value": f"{geom.drawer_outer_width_cm:.2f} / {geom.drawer_inner_width_cm:.1f} ({geom.drawers_per_full_row} per full row, {geom.drawers_per_half_bay} beside 1 bin)"},
        {"Parameter": "9in drawer outer / inner width (cm)", "Value": f"{geom.drawer_9in_outer_width_cm:.2f} / {geom.drawer_9in_inner_width_cm:.1f} ({geom.drawers_9in_per_full_row} per full row, {geom.drawers_9in_per_half_bay} beside 1 bin)"},
        {"Parameter": "4P (2x2) / 3P (2:1:1) clear width (cm)", "Value": f"6in: {geom.partition_width_cm(4, 0):.1f} / {geom.partition_width_cm(3, 0):.1f} top; 9in: {geom.partition_width_cm(4, 0, geom.drawer_9in_inner_width_cm):.1f} / {geom.partition_width_cm(3, 0, geom.drawer_9in_inner_width_cm):.1f} top"},
        {"Parameter": "Wire bin width and capacity", "Value": f"{geom.wire_bin_bay_width_cm:.1f} cm ({geom.wire_bin_subzones} corner zones, {geom.wire_bin_subzone_hard_cap} rolls/zone max)"},
        {"Parameter": "Top sliding pegboard", "Value": f"{geom.peg_rows} rows x {geom.pegs_per_row} pins = {geom.peg_rows * geom.pegs_per_row} pegs (<= {geom.peg_max_od_cm:.2f} cm / 6 in OD, 1 tall or 2 short/peg)"},
    ])

    _mech_df = pd.DataFrame([
        {
            "Module": MODULE_NAMES[_spec.mechanism_id][0],
            "Location": MODULE_NAMES[_spec.mechanism_id][1],
            "Reach Cost": f"{_spec.reach_actuation_cost:.2f}",
            "Closed Vis.": f"{_spec.visibility_closed:.2f}",
            "Open Vis.": f"{_spec.visibility_open:.2f}",
            "Clearance (W)": f"{_spec.clearance_score:.2f}",
            "Relative Fetch Effort": f"{_spec.base_retrieval_sec / 3.0:.2f}x",
            "Relative Return Effort": f"{_spec.base_return_sec / 2.5:.2f}x",
            "Coreless OK": "Yes" if _spec.supports_coreless else "No",
            "Pull-Down": "Yes" if _spec.is_pull_down else "No",
        }
        for _spec in mech_specs.values()
    ])

    _beh_df = pd.DataFrame([
        {
            "Group": "Shift",
            "Setting": "Simulated duration",
            "Value": f"{rush_cfg.num_shifts} shifts x {rush_cfg.shift_cycles} cycles ({rush_cfg.total_cycles} total)",
            "Literature / Physical Basis": "8-hour shift horizon across 20 shifts to capture steady-state drift [1]",
        },
        {
            "Group": "Shift",
            "Setting": "Morning rush share",
            "Value": f"{rush_cfg.rush_job_share:.0%} of each shift",
            "Literature / Physical Basis": "45% batch volume concentrated in 8:00-10:00 AM production window",
        },
        {
            "Group": "Shift",
            "Setting": "Time pressure range (rush / calm)",
            "Value": f"{rush_cfg.rush_tau_range} / {rush_cfg.calm_tau_range}",
            "Literature / Physical Basis": "Normalized pacing stress index tau driving speed-accuracy pressure [2, 6]",
        },
        {
            "Group": "Demand",
            "Setting": "Priority batch boost",
            "Value": f"{rush_cfg.rush_sku_boost}x on {', '.join(rush_cfg.rush_priority_categories)}",
            "Literature / Physical Basis": "Higher pick skew on Bites, Salad, and Breakfast Cookie rush SKUs",
        },
        {
            "Group": "Demand",
            "Setting": "Same-product variant pairing",
            "Value": f"{rush_cfg.same_product_co_retrieval:.0%} probability for A/B/C siblings",
            "Literature / Physical Basis": "90% empirical joint pull rate across matched A/B/C product sets [5]",
        },
        {
            "Group": "Behavior",
            "Setting": "Baseline mistake rate / calm fix rate",
            "Value": f"{beh_cfg.base_error_rate:.3f} / {beh_cfg.base_correction_rate:.2f}",
            "Literature / Physical Basis": "1%-3% nominal manual HEP [6, 7] and 15%-30% incidental anomaly detection [3, 6]",
        },
        {
            "Group": "Behavior",
            "Setting": "Weights (rush / reach / crowding / confusion)",
            "Value": f"{beh_cfg.alpha_rush} / {beh_cfg.alpha_ergo} / {beh_cfg.alpha_crowd} / {beh_cfg.alpha_blur}",
            "Literature / Physical Basis": "THERP rush stress [6], reach zone strain [5, 7], 75% fill knee [2, 3, 5], Category Adjustment [4]",
        },
        {
            "Group": "Behavior",
            "Setting": "Wire bin dump penalty / draw",
            "Value": f"{beh_cfg.bin_dump_error_penalty:.2f} error / {beh_cfg.bin_dump_routing_weight:.2f} draw per bin",
            "Literature / Physical Basis": "Unwalled wide drop target behavior and lower boundary discipline [4, 7]",
        },
        {
            "Group": "Behavior",
            "Setting": "Mixed-style / bin+drawer row load",
            "Value": f"{beh_cfg.mech_diversity_penalty:.2f} (>2 styles) / {beh_cfg.bin_drawer_same_row_penalty:.2f} (same row)",
            "Literature / Physical Basis": "Working-memory switching across storage rules [7] and same-row bin+drawer friction",
        },
        {
            "Group": "Behavior",
            "Setting": "Multi-SKU slot / slot deficit / width mismatch",
            "Value": f"{beh_cfg.multi_sku_slot_penalty:.2f} / {beh_cfg.slot_deficit_penalty:.2f} / {beh_cfg.row_width_mismatch_penalty:.2f}",
            "Literature / Physical Basis": "Workspace constraints for <48 drawer slots, shared slots, and 42in vs. 45in row alignment",
        },
        {
            "Group": "Behavior",
            "Setting": "Peg row left unlatched probability",
            "Value": f"{beh_cfg.unlatch_rush_prob} x tau",
            "Literature / Physical Basis": "Hurried bypass of secondary pull-down re-latching under time pressure [6, 7]",
        },
        {
            "Group": "Behavior",
            "Setting": "Usability accuracy floor",
            "Value": f"{beh_cfg.usability_accuracy_threshold:.0%}",
            "Literature / Physical Basis": "Minimum home-slot reliability floor before routine search failures compound",
        },
    ])

    mo.vstack([
        mo.md("## Wire shelf geometry and model parameters"),
        mo.md(
            "Tunable weights load from `config/shelf_config.yaml` (override with `--config <.yaml/.json/.toon>` in `run_experiments.py`), while fixed physical dimensions, shift schedules, and SKU lists live in `config/shelf_geometry.yaml`. "
            "Core human-error parameters map to human-reliability and order-picking benchmarks in [2-7], paired with explicit workspace constraints for compartment count and row-width alignment."
        ),
        mo.md("### Shelf dimensions"),
        mo.ui.table(
            _geom_df,
            selection=None,
            show_column_summaries=False,
            pagination=False,
            show_search=False,
            text_justify_columns={"Parameter": "left", "Value": "right"},
        ),
        mo.md(
            "Drawers come in two outer widths sharing the same drawer mechanism (`58.0 cm` deep x `27.0 cm` tall inside): `6in` (`15.24 cm` outer / `14.0 cm` inner, fitting `7` per shelf = `106.7 cm` or `4` beside a `51.4 cm` bin = `112.4 cm`) and `9in` (`22.86 cm` outer / `21.6 cm` inner, fitting `5` per shelf = `114.3 cm`, `3` beside a bin = `120.0 cm`, or `3` x `9in` + `3` x `6in` = `114.3 cm`). Mixing `6in` and `9in` drawers incurs zero mechanism-switching penalty (`drawer_size_mix_penalty = 0.0`) because both use identical drawer actuation."
        ),
        mo.md("### Storage modules"),
        mo.ui.table(
            _mech_df,
            selection=None,
            show_column_summaries=False,
            pagination=False,
            show_search=False,
            text_justify_columns={
                "Module": "left",
                "Location": "left",
                "Reach Cost": "right",
                "Closed Vis.": "right",
                "Open Vis.": "right",
                "Clearance (W)": "right",
                "Relative Fetch Effort": "right",
                "Relative Return Effort": "right",
                "Coreless OK": "center",
                "Pull-Down": "center",
            },
        ),
        mo.md(
            "`Reach Cost` scores ergonomic bending or overhead strain relative to waist-height access. `Closed Vis.` and `Open Vis.` control how often workers notice out-of-place rolls when walking past vs. when opening a drawer or pulling a peg row down. `Clearance (W)` captures how wide a compartment opening feels during fast returns, and `Relative Fetch / Return Effort` indexes single-roll handling effort to middle-shelf split drawers (`1.00x`)."
        ),
        mo.md("### Shift and behavioral settings"),
        mo.ui.table(
            _beh_df,
            selection=None,
            show_column_summaries=False,
            pagination=False,
            show_search=False,
            text_justify_columns={
                "Group": "left",
                "Setting": "left",
                "Value": "right",
                "Literature / Physical Basis": "left",
            },
        ),
        mo.md(
            "**Basis for behavioral weights.** Core error and recovery rates sit within published human-reliability and warehouse-picking bounds, while hardware penalties encode explicit shop-floor constraints:\n\n"
            "- **Baseline error (`0.028`) and calm fix rate (`0.24`):** THERP [6] puts routine selection mistakes among similar compartments at `1%` to `3%`, with incidental scanning catching roughly `15%` to `30%` of misplaced items [3, 6].\n"
            "- **Rush (`2.8`), reach (`2.3`), crowding (`3.2`), and boundary confusion (`1.75`):** Morning pacing (`tau` from `0.75` to `1.00`) lifts error odds by `2.58x` to `3.80x` [2, 6]. Waist-to-eye reach (`140 cm`, `0.14`) sits in the comfortable zone, whereas stooping to bottom bins (`25-80 cm`, `0.38-0.42`) or pulling overhead pegs (`195 cm`, `0.22`) increases return shortcuts [5, 7]. Past `75%` compartment fill, drop clearance shrinks under Fitts's Law [2, 3]. Undivided `3P` lanes, weak labels (`1 - sigma`), and neighbor drift widen spatial recall scatter [4].\n"
            "- **Wire-bin dump bias (`0.11` error, `0.30` draw), mixed styles (`0.10`), and bin+drawer row (`0.16`):** Unwalled wire baskets attract hurried drop-ins (`0.30` routing weight per bin) and add `+0.11` local slip risk [4, 7]. Mixing a wire bin and drawers in one row adds `+0.16` cognitive load and `0.55 s` hesitation. Mixing more than two storage styles adds `+0.10` per extra mechanism, whereas mixing `6in` and `9in` drawers adds `0.00` [7].\n"
            "- **Organizational slot deficit (`1.20`), multi-SKU sharing (`0.16`), and row-width alignment (`0.25`):** Dropping below `48` drawer compartments (`12` `2x2` drawers) forces unrelated SKUs into shared compartments (`+0.16` blur per extra SKU in a slot) and adds rack-wide organizational blur (`1.20` scaled by the deficit below `48`). Pairing a `42 in` (`7D6`) row with a `45 in` (`5D9` or `3D9_3D6`) row misaligns bays by `3 in` across tiers (`+0.25`).\n\n"
            "All 825 feasible layouts run against this fixed vector. Literature-grounded parameters set the baseline advantage of drawer grouping over wire bins and height splitting, while the slot-deficit and row-width weights reflect explicit workspace preferences for `48` compartments and uniform `45 in` bay alignment."
        ),
    ])
    return (rush_cfg,)


@app.cell(hide_code=True)
def _(DEFAULT_CSV_PATH, load_roll_inventory, mo, pd, rush_cfg):
    variants, base_rolls, freq_calm, freq_rush, co_matrix = load_roll_inventory(
        DEFAULT_CSV_PATH, rush_cfg
    )

    _n_products = len({_v.product_name for _v in variants})
    _n_rush_rolls = sum(1 for _r in base_rolls if _r.is_rush_priority)
    _coreless_skus = [_v.sku_code for _v in variants if not _v.has_core]
    _abc_pairs = sum(
        1
        for _i in range(len(variants))
        for _j in range(_i + 1, len(variants))
        if variants[_i].product_name == variants[_j].product_name
        and co_matrix[_i][_j] >= 0.85
    )

    _variants_df = pd.DataFrame([
        {
            "SKU": _v.sku_code,
            "Product": _v.product_name,
            "Variant": _v.variant_code,
            "Category": _v.category,
            "Rolls": _v.quantity,
            "Diameter (cm)": round(_v.outer_diameter_cm, 1),
            "Height (cm)": round(_v.roll_height_cm, 1),
            "Cored": "Yes" if _v.has_core else "No",
            "Rush Priority": "Yes" if _v.is_rush_priority else "No",
            "Calm Pick Share": f"{_v.baseline_freq:.2%}",
            "Rush Pick Share": f"{_v.rush_freq:.2%}",
        }
        for _v in sorted(variants, key=lambda _x: (-_x.rush_freq, -_x.quantity, _x.sku_code))
    ])

    _stats_row = mo.hstack([
        mo.stat(value=str(_n_products), label="Food products", bordered=True),
        mo.stat(value=str(len(variants)), label="SKU variants", bordered=True),
        mo.stat(value=str(len(base_rolls)), label="Physical rolls", bordered=True),
        mo.stat(value=f"{_n_rush_rolls} ({_n_rush_rolls / len(base_rolls):.1%})", label="Rush-priority rolls", bordered=True),
        mo.stat(value=f"{_abc_pairs} pairs", label="90% A/B/C sibling links", bordered=True),
        mo.stat(value=str(len(_coreless_skus)), label="Coreless SKUs (no pegs)", bordered=True),
    ], widths="equal")

    mo.vstack([
        mo.md("## Roll inventory"),
        mo.md(
            f"All {len(base_rolls)} physical rolls from `data/roll-data.csv` across {len(variants)} SKU variants "
            f"and {_n_products} food products. One SKU (`{', '.join(_coreless_skus)}`) has no inner core and must sit in a drawer or wire bin."
        ),
        _stats_row,
        mo.ui.table(
            _variants_df,
            selection=None,
            show_column_summaries=False,
            page_size=12,
            text_justify_columns={
                "SKU": "left",
                "Product": "left",
                "Variant": "center",
                "Category": "left",
                "Rolls": "right",
                "Diameter (cm)": "right",
                "Height (cm)": "right",
                "Cored": "center",
                "Rush Priority": "center",
                "Calm Pick Share": "right",
                "Rush Pick Share": "right",
            },
        ),
        mo.md(
            "Rows are sorted by `Rush Pick Share` so high-volume 8:00-10:00 AM items (`Bites`, `Salad`, and `Breakfast Cookies`) appear first. Multi-copy `Rolls` counts (`1` to `6` per SKU) drive why duplicate reserve copies move to the bottom shelf in `hybrid_reserve` layouts so active middle lanes stay near `80%` fill."
        ),
    ])
    return base_rolls, variants


@app.cell(hide_code=True)
def _(
    BACKUP_RULE_NAMES,
    ExperimentScenario,
    LAYOUT_NAMES,
    VARIANT_RULE_NAMES,
    allocate_home_slots,
    base_rolls,
    build_default_experiments,
    build_orientation_scenarios,
    compute_layout_metrics,
    copy,
    mo,
    pd,
    root_dir,
    variants,
):
    exp_summary_df = pd.read_csv(root_dir / "output" / "experiment_summary.csv")
    cap_sweep_df = pd.read_csv(root_dir / "output" / "capacity_sweep.csv")
    _ori_raw_df = pd.read_csv(root_dir / "output" / "orientation_search_400.csv")

    _default_exps = build_default_experiments()
    _top_ori_id = str(_ori_raw_df.sort_values("composite_score", ascending=False).iloc[0]["ori_id"])
    _ori_map = {_scen.exp_id: _scen for _, _, _scen in build_orientation_scenarios()}
    _top_base = _ori_map[_top_ori_id]
    _champ_scen = ExperimentScenario(
        exp_id="ORI-CHAMP",
        name=f"Best layout {_top_ori_id}",
        topology=_top_base.topology,
        strategy=_top_base.strategy,
        drawer_partitions=_top_base.drawer_partitions,
        cue_salience=_top_base.cue_salience,
        target_fill_ratio=_top_base.target_fill_ratio,
        description=_top_base.description,
        row_configs=_top_base.row_configs,
        abc_colocation_mode=_top_base.abc_colocation_mode,
        overflow_mode=_top_base.overflow_mode,
        spatial_order_mode=_top_base.spatial_order_mode,
        layout_seed=_top_base.layout_seed,
    )

    _all_scenarios = list(_default_exps) + [_champ_scen]
    all_benchmarks_by_id = {_s.exp_id: _s for _s in _all_scenarios}
    _sum_by_id = {str(_r["exp_id"]): _r for _, _r in exp_summary_df.iterrows()}
    _base_sr = _sum_by_id["ORI-258"]
    _base_acc = float(_base_sr["final_acc_mean"])
    _base_lat = float(_base_sr["mean_job_latency_sec"])

    _combined_rows = []
    for _scen in _all_scenarios:
        _rolls_copy = copy.deepcopy(base_rolls)
        _slots = allocate_home_slots(_scen, variants, _rolls_copy)
        _metrics = compute_layout_metrics(_rolls_copy, _slots)
        _sr = _sum_by_id[_scen.exp_id]
        _acc = float(_sr["final_acc_mean"])
        _std = float(_sr["final_acc_std"])
        _lat = float(_sr["mean_job_latency_sec"])
        _disp_id = f"{_top_ori_id} (Best)" if _scen.exp_id == "ORI-CHAMP" else _scen.exp_id
        _vs_base = "0.0%" if _scen.exp_id == "ORI-258" else f"{(_acc - _base_acc) * 100:+.1f}%"
        _combined_rows.append({
            "_sort_acc": _acc,
            "Layout": _disp_id,
            "Mid Shelf": LAYOUT_NAMES.get(_scen.row_configs[0], _scen.row_configs[0]) if _scen.row_configs else "",
            "Bottom Shelf": LAYOUT_NAMES.get(_scen.row_configs[1], _scen.row_configs[1]) if _scen.row_configs else "",
            "Top Shelf": LAYOUT_NAMES.get(_scen.row_configs[2], _scen.row_configs[2]) if _scen.row_configs else "",
            "Variant Rule": VARIANT_RULE_NAMES.get(_scen.abc_colocation_mode, _scen.abc_colocation_mode),
            "Backup Rule": BACKUP_RULE_NAMES.get(_scen.overflow_mode, _scen.overflow_mode),
            "Label Clarity": f"{_scen.cue_salience:.0%}",
            "Fill Target": f"{_scen.target_fill_ratio:.0%}",
            "Same-Drawer A/B/C": f"{_metrics['abc_same_drawer_pct']:.0%}",
            "Cycle 1,000 Accuracy": f"{_acc * 100:.1f}% \u00b1 {_std * 100:.1f}%",
            "Vs. Baseline Acc.": _vs_base,
            "Cross-Shelf Mixup": f"{float(_sr['final_cross_tier_contam']) * 100:.1f}%",
            "Relative Effort": f"{_lat / _base_lat:.2f}x",
        })

    _bench_df = (
        pd.DataFrame(_combined_rows)
        .sort_values("_sort_acc", ascending=False)
        .drop(columns=["_sort_acc"])
        .reset_index(drop=True)
    )

    _acc_piv = cap_sweep_df.pivot(index="fill_ratio", columns="layout", values="final_accuracy")
    _lat_piv = cap_sweep_df.pivot(index="fill_ratio", columns="layout", values="mean_latency_sec")
    _sweep_rows = []
    for _rho in sorted(_acc_piv.index):
        _sweep_rows.append({
            "Fill Target": f"{_rho:.0%}",
            "Mid 3D9+3D6(4P) + Btm 3D9+3D6(4P) Acc": f"{_acc_piv.loc[_rho, 'TOPO053_Mid3D93D64P_Btm3D93D64P']:.1%}",
            "Mid 3D9+3D6(4P) + Btm 3D9+3D6(4P) Effort": f"{_lat_piv.loc[_rho, 'TOPO053_Mid3D93D64P_Btm3D93D64P'] / _base_lat:.2f}x",
            "Mid 7D6(4P) + Btm 2B Acc": f"{_acc_piv.loc[_rho, 'TOPO007_Mid7D64P_Btm2B']:.1%}",
            "Mid 7D6(4P) + Btm 2B Effort": f"{_lat_piv.loc[_rho, 'TOPO007_Mid7D64P_Btm2B'] / _base_lat:.2f}x",
            "Baseline COI Acc": f"{_acc_piv.loc[_rho, 'TOPO007_BaselineCOI']:.1%}",
            "Baseline COI Effort": f"{_lat_piv.loc[_rho, 'TOPO007_BaselineCOI'] / _base_lat:.2f}x",
        })
    _sweep_df = pd.DataFrame(_sweep_rows)

    _fig_dir = root_dir / "figures"

    mo.vstack([
        mo.md("## Benchmark shelf experiments"),
        mo.md("Fifty simulated runs of 1,000 work cycles each (20 shifts with daily rush periods). Effort is scaled relative to the unorganized `ORI-258` baseline (`1.00x`)."),
        mo.ui.table(
            _bench_df,
            selection=None,
            show_column_summaries=False,
            pagination=False,
            show_search=False,
            text_justify_columns={
                "Layout": "left",
                "Mid Shelf": "left",
                "Bottom Shelf": "left",
                "Top Shelf": "left",
                "Variant Rule": "left",
                "Backup Rule": "left",
                "Label Clarity": "right",
                "Fill Target": "right",
                "Same-Drawer A/B/C": "right",
                "Cycle 1,000 Accuracy": "right",
                "Vs. Baseline Acc.": "right",
                "Cross-Shelf Mixup": "right",
                "Relative Effort": "right",
            },
        ),
        mo.md(
            "`Same-Drawer A/B/C` gives the share of co-retrieved product variants placed in one drawer, while `Cross-Shelf Mixup` tracks rolls ending up on the wrong vertical shelf tier by Cycle 1,000. Combining `3 x 9in` (`21.6 cm` inner) and `3 x 6in` (`14.0 cm` inner) drawers on each lower row (`ORI-261`: `Top: Peg | Mid: 3D9_3D6_4P | Btm: 3D9_3D6_4P`) gives `48` drawer slots (`45 in` uniform width) with room for every roll up to `23.0 cm` OD inside enclosed drawers, reaching `89.5%` accuracy at `0.31x` effort (`2.0%` cross-shelf mixup) and beating `5D9_4P | 5D9_4P` (`ORI-5D9`, `88.3%`, `0.33x`), which only has `40` drawer slots."
        ),
        mo.image(
            src=_fig_dir / "fig1_entropy_decay.png",
            alt="Home slot accuracy over 1,000 work cycles",
            width=960,
        ),
        mo.md(
            "*Figure 1: Home-slot accuracy across 1,000 work cycles for key benchmark configurations.* "
            "Each sawtooth dip marks a morning rush (cycles 1 to 22 of each 50-cycle shift), and each rise marks afternoon cleanup. Only the two bin-free `9in` drawer layouts (`ORI-261` at `89.5%`, `ORI-5D9` at `88.3%`) stay above the `80%` usability floor across all 20 shifts."
        ),
        mo.image(
            src=_fig_dir / "fig2_cross_tier_contamination.png",
            alt=f"Roll location share over 1,000 cycles for ORI-258 vs {_top_ori_id}",
            width=960,
        ),
        mo.md(
            "*Figure 2: Roll location state distribution over time comparing ORI-258 against ORI-261.* "
            "Height-splitting `A`, `B`, and `C` variants across shelves in `ORI-258` leaves `20.8%` of rolls at home (`42.3%` cross-shelf mixup), whereas `ORI-261` holds `89.5%` at home (`2.0%` cross-shelf mixup)."
        ),
        mo.image(
            src=_fig_dir / "fig3_slot_heatmap.png",
            alt="Final-cycle misplaced rolls per compartment",
            width=960,
        ),
        mo.md(
            "*Figure 3: Compartment-level misplaced roll count and total roll occupancy at Cycle 1,000.* "
            "Bottom wire-bin zones in `ORI-258` accumulate `4.92` to `5.50` foreign rolls each and mid drawers carry up to `2.54`, while `ORI-261` stays below `0.48` misplaced rolls per compartment across all `80` slots."
        ),
        mo.md("### Capacity tipping point (70% to 98% target fill)"),
        mo.ui.table(
            _sweep_df,
            selection=None,
            show_column_summaries=False,
            pagination=False,
            show_search=False,
            text_justify_columns={c: "right" for c in _sweep_df.columns},
        ),
        mo.image(
            src=_fig_dir / "fig4_capacity_and_partitions.png",
            alt="Fill ratio sweep and benchmark layout comparison",
            width=960,
        ),
        mo.md(
            "*Figure 4: Target fill sweep (70% to 98%, 25 replications per point) and benchmark layout comparison.* "
            "`Mid: 3D9_3D6_4P | Btm: 3D9_3D6_4P` stays between `89.3%` and `87.9%` across the `75%` to `85%` buffer (`84.2%` at `98%` fill), while `Mid: 7D6_4P | Btm: 2B` falls from `72.3%` to `64.8%` and the `COI` baseline stays below `24%`."
        ),
    ])
    return (all_benchmarks_by_id,)


@app.cell(hide_code=True)
def _(LAYOUT_NAMES, all_benchmarks_by_id, mo, pd, root_dir):
    markov_summary_df = pd.read_csv(root_dir / "output" / "markov_summary.csv")
    _champ_scen = all_benchmarks_by_id.get("ORI-CHAMP")
    _champ_ori_id = _champ_scen.name.split()[-1] if _champ_scen is not None else "ORI-CHAMP"
    _champ_rows_desc = (
        " + ".join([
            LAYOUT_NAMES.get(_champ_scen.row_configs[0], _champ_scen.row_configs[0]),
            LAYOUT_NAMES.get(_champ_scen.row_configs[1], _champ_scen.row_configs[1]),
            LAYOUT_NAMES.get(_champ_scen.row_configs[2], _champ_scen.row_configs[2]),
        ])
        if _champ_scen and _champ_scen.row_configs
        else "Modular layout"
    )

    _mk_display_df = pd.DataFrame([
        {
            "Layout": f"{_champ_ori_id} (Top Search)" if str(_r["exp_id"]) == "ORI-CHAMP" else str(_r["exp_id"]),
            "Home (S0)": f"{float(_r['pi_home']):.1%}",
            "Adjacent Drift (S1)": f"{float(_r['pi_drift_adj']):.1%}",
            "Mid Drawer Dump (S2)": f"{float(_r['pi_eye_dump']):.1%}",
            "Wire Bin Dump (S3)": f"{float(_r['pi_wire_dump']):.1%}",
            "Peg Dump (S4)": f"{float(_r['pi_peg_dump']):.1%}",
            "Buried (S5)": f"{float(_r['pi_buried']):.1%}",
            "Cycles Until Misplaced": f"{float(_r['mfpt_home_to_misplaced']):.1f}",
            "Cycles to Recover": f"{float(_r['mfpt_misplaced_to_home']):.1f}",
            "Sim vs. Math Diff (TVD)": f"{float(_r['sim_tvd']):.4f}",
        }
        for _, _r in markov_summary_df.iterrows()
    ])

    _dropdown_options = {
        (f"{_champ_ori_id} (Top Search): {_champ_rows_desc}" if _k == "ORI-CHAMP" else _k): _k
        for _k in all_benchmarks_by_id.keys()
    }
    _default_label = next(
        (_lbl for _lbl, _val in _dropdown_options.items() if _val == "ORI-CHAMP"),
        next(iter(_dropdown_options.keys())),
    )
    markov_selector = mo.ui.dropdown(
        options=_dropdown_options,
        value=_default_label,
        label="Inspect 6-state transition matrix P for layout:",
    )

    mo.vstack([
        mo.md("## Six-state Markov chain results"),
        mo.md("Exact long-run steady-state occupancy, expected cycles before a roll leaves home, expected cycles to recover, and sample-path Total Variation Distance (TVD)."),
        mo.ui.table(
            _mk_display_df,
            selection=None,
            show_column_summaries=False,
            pagination=False,
            show_search=False,
            text_justify_columns={
                "Layout": "left",
                "Home (S0)": "right",
                "Adjacent Drift (S1)": "right",
                "Mid Drawer Dump (S2)": "right",
                "Wire Bin Dump (S3)": "right",
                "Peg Dump (S4)": "right",
                "Buried (S5)": "right",
                "Cycles Until Misplaced": "right",
                "Cycles to Recover": "right",
                "Sim vs. Math Diff (TVD)": "right",
            },
        ),
        mo.md(
            "The first six columns show the exact stationary occupancy share $\\pi_0 \\dots \\pi_5$ across shelf states (rows sum to `100%`). `Cycles Until Misplaced` is the expected cycles a roll stays in `Home (S0)` before slipping (`30.9` cycles for `ORI-261` vs. `4.9` for `ORI-258`), and `Cycles to Recover` is the expected absorption time back to `S0` (`7.7` cycles vs. `47.1`). `Sim vs. Math Diff (TVD)` checks agreement between exact Markov algebra and simulated random walks (`0.0006` for `ORI-261`, `<= 0.0033` across all benchmarks)."
        ),
        markov_selector,
    ])
    return (markov_selector,)


@app.cell(hide_code=True)
def _(all_benchmarks_by_id, analyze_scenario_markov, markov_selector, mo, pd):
    _clean_states = [
        "S0: Home",
        "S1: Adjacent lane",
        "S2: Mid drawer",
        "S3: Wire bin",
        "S4: Top peg",
        "S5: Buried",
    ]
    _col_headers = ["S0", "S1", "S2", "S3", "S4", "S5"]

    _selected_id = markov_selector.value
    _sel_scen = all_benchmarks_by_id[_selected_id]
    _display_id = f"{_sel_scen.name.split()[-1]} (Top Search)" if _selected_id == "ORI-CHAMP" else _selected_id
    _res = analyze_scenario_markov(_sel_scen, regime="shift_composite")

    _p_df = pd.DataFrame(
        [[round(_val, 4) for _val in _row] for _row in _res.transition_matrix],
        columns=_col_headers,
    )
    _p_df.insert(0, "From \\ To", _clean_states)

    _pi_df = pd.DataFrame([
        {
            "State": _name,
            "Exact Steady-State Share": f"{_res.stationary_dist[_i]:.2%}",
            "Sample-Path Share": f"{_res.sim_stationary_dist[_i]:.2%}",
        }
        for _i, _name in enumerate(_clean_states)
    ])

    mo.vstack([
        mo.md(
            f"**Selected `{_display_id}`:** Stationary home accuracy $\\pi_0 = {_res.stationary_dist[0] * 100:.2f}\\%$, "
            f"spectral gap $= {_res.spectral_gap:.4f}$, 95% mixing horizon $t_{{95}} = {_res.mixing_time_95:.1f}$ cycles, "
            f"cycles until misplaced $= {_res.mfpt_home_to_any_misplaced:.1f}$, "
            f"cycles to recover $= {_res.mfpt_misplaced_to_home:.1f}$, "
            f"$\\text{{TVD}} = {_res.total_variation_distance:.4f}$."
        ),
        mo.md(
            "The spectral gap ($1 - |\\lambda_2|$) controls how quickly transient misplacements damp out; $t_{95}$ estimates the number of work cycles required for any starting state to settle within 5% of steady state."
        ),
        mo.md("### Single-cycle transition matrix ($P$)"),
        mo.ui.table(
            _p_df,
            selection=None,
            show_column_summaries=False,
            pagination=False,
            show_search=False,
            text_justify_columns={"From \\ To": "left", **{c: "right" for c in _col_headers}},
        ),
        mo.md(
            f"Each row (`From`) sums to `1.0000` and shows the one-cycle probability of moving to each column (`To`) under the `45%` rush / `55%` calm shift mix. "
            f"For `{_display_id}`, `S0 -> S0` (`{_res.transition_matrix[0][0]:.2%}`) is the chance a roll at home remains home after one cycle, while the off-diagonal cells in row `S0` split the remaining slip probability across misplaced states. "
            f"Column `S0` gives the per-cycle recovery rate from each misplaced state back to home (`{_res.transition_matrix[1][0]:.2%}` from adjacent lane `S1` down to `{_res.transition_matrix[5][0]:.2%}` from buried state `S5`), and diagonal cell `S5 -> S5` (`{_res.transition_matrix[5][5]:.2%}`) shows how sticky buried stock stays once trapped."
        ),
        mo.md("### Steady-state share: exact solve vs. sample path"),
        mo.ui.table(
            _pi_df,
            selection=None,
            show_column_summaries=False,
            pagination=False,
            show_search=False,
            text_justify_columns={
                "State": "left",
                "Exact Steady-State Share": "right",
                "Sample-Path Share": "right",
            },
        ),
        mo.md(
            f"`Exact Steady-State Share` solves the stationary balance equation $\\pi P = \\pi$ directly. `Sample-Path Share` averages `20` random-walk chains of `35,000` cycles on $P$ (discarding the first `2,500` burn-in cycles per chain), matching the analytical distribution with `TVD = {_res.total_variation_distance:.4f}`."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(
    BACKUP_RULE_NAMES,
    LAYOUT_NAMES,
    MODULAR_TOPOLOGIES,
    MODULE_NAMES,
    ORDER_RULE_NAMES,
    POSITIONING_POLICIES,
    TIER_LABELS,
    VARIANT_RULE_NAMES,
    mo,
    pd,
    root_dir,
):
    ori_df = pd.read_csv(root_dir / "output" / "orientation_search_400.csv")
    topo_df = pd.read_csv(root_dir / "output" / "topology_summary_100.csv")
    slotting_df = pd.read_csv(root_dir / "output" / "optimal_slotting_120x60x60.csv")

    _ori_base_rows = ori_df[
        (ori_df["row1_eye"] == "7D_4P")
        & (ori_df["row2_mid"] == "2B")
        & (ori_df["abc_colocation_mode"] == "cross_tier_height")
        & (ori_df["spatial_order_mode"] == "random_perm")
    ]
    if len(_ori_base_rows) > 0:
        _ori_base_id = str(_ori_base_rows.iloc[0]["ori_id"])
        _ori_base_acc = float(_ori_base_rows.iloc[0]["final_acc_mean"])
        _ori_base_lat = float(_ori_base_rows.iloc[0]["overall_latency_sec"])
    else:
        _ori_base_id = "ORI-258"
        _ori_base_acc = float(ori_df["final_acc_mean"].min())
        _ori_base_lat = float(ori_df["overall_latency_sec"].max())

    _top_ori_row = ori_df.sort_values("composite_score", ascending=False).iloc[0]
    _top_ori_label = str(_top_ori_row["ori_id"])
    _top_mid_name = LAYOUT_NAMES.get(str(_top_ori_row["row1_eye"]), str(_top_ori_row["row1_eye"]))
    _top_btm_name = LAYOUT_NAMES.get(str(_top_ori_row["row2_mid"]), str(_top_ori_row["row2_mid"]))
    _top_var_name = VARIANT_RULE_NAMES.get(str(_top_ori_row["abc_colocation_mode"]), str(_top_ori_row["abc_colocation_mode"])).lower()

    _abc_agg_raw = (
        ori_df.groupby("abc_colocation_mode", as_index=False)
        .agg(
            orientations=("ori_id", "count"),
            mean_acc=("final_acc_mean", "mean"),
            max_acc=("final_acc_mean", "max"),
            mean_latency_s=("overall_latency_sec", "mean"),
            best_score=("composite_score", "max"),
        )
        .sort_values("best_score", ascending=False)
    )
    _abc_agg_df = pd.DataFrame([
        {
            "Variant Rule": VARIANT_RULE_NAMES.get(str(_r["abc_colocation_mode"]), str(_r["abc_colocation_mode"])),
            "Tested Layouts": int(_r["orientations"]),
            "Mean Accuracy": f"{float(_r['mean_acc']):.1%}",
            "Best Accuracy": f"{float(_r['max_acc']):.1%}",
            "Best vs. Baseline": f"{(float(_r['max_acc']) - _ori_base_acc) * 100:+.1f}%",
            "Mean Effort": f"{float(_r['mean_latency_s']) / _ori_base_lat:.2f}x",
            "Best Score": f"{float(_r['best_score']):.2f}",
        }
        for _, _r in _abc_agg_raw.iterrows()
    ])

    def _format_ori_table(_df_in):
        return pd.DataFrame([
            {
                "Rank": int(_r["rank"]),
                "ID": str(_r["ori_id"]),
                "Topology": str(_r["topo_id"]),
                "Mid Shelf": LAYOUT_NAMES.get(str(_r["row1_eye"]), str(_r["row1_eye"])),
                "Bottom Shelf": LAYOUT_NAMES.get(str(_r["row2_mid"]), str(_r["row2_mid"])),
                "Top Shelf": LAYOUT_NAMES.get(str(_r["row3_low_top"]), str(_r["row3_low_top"])),
                "Variant Rule": VARIANT_RULE_NAMES.get(str(_r["abc_colocation_mode"]), str(_r["abc_colocation_mode"])),
                "Backup Rule": BACKUP_RULE_NAMES.get(str(_r["overflow_mode"]), str(_r["overflow_mode"])),
                "Shelf Order": ORDER_RULE_NAMES.get(str(_r["spatial_order_mode"]), str(_r["spatial_order_mode"])),
                "Same-Drawer A/B/C": f"{float(_r['abc_same_drawer_pct']):.0%}",
                "Accuracy": f"{float(_r['final_acc_mean']):.1%}",
                "Vs. Baseline Acc.": "0.0%" if str(_r["ori_id"]) == _ori_base_id else f"{(float(_r['final_acc_mean']) - _ori_base_acc) * 100:+.1f}%",
                "Relative Effort": f"{float(_r['overall_latency_sec']) / _ori_base_lat:.2f}x",
                "Score": f"{float(_r['composite_score']):.2f}",
            }
            for _, _r in _df_in.iterrows()
        ])

    _topo_summary_df = pd.DataFrame([
        {
            "Rank": int(_r["rank"]),
            "Topology": str(_r["topo_id"]),
            "Mid Shelf": LAYOUT_NAMES.get(str(_r["mid_shelf"]), str(_r["mid_shelf"])),
            "Bottom Shelf": LAYOUT_NAMES.get(str(_r["btm_shelf"]), str(_r["btm_shelf"])),
            "Top Shelf": LAYOUT_NAMES.get(str(_r["top_shelf"]), str(_r["top_shelf"])),
            "Best ID": str(_r["best_ori_id"]),
            "Best Variant Rule": VARIANT_RULE_NAMES.get(str(_r["best_abc_mode"]), str(_r["best_abc_mode"])),
            "Best Accuracy": f"{float(_r['best_acc_mean']):.1%}",
            "Best vs. Baseline": f"{(float(_r['best_acc_mean']) - _ori_base_acc) * 100:+.1f}%",
            "Mean Accuracy": f"{float(_r['mean_acc_across_policies']):.1%}",
            "Best Effort": f"{float(_r['best_latency_sec']) / _ori_base_lat:.2f}x",
            "Best Score": f"{float(_r['best_composite_score']):.2f}",
        }
        for _, _r in topo_df.iterrows()
    ])

    _pareto_df = _format_ori_table(
        ori_df[ori_df["is_pareto_optimal"] == True].sort_values("composite_score", ascending=False)
    )
    _all_ori_clean_df = _format_ori_table(ori_df)

    _slot_clean_df = pd.DataFrame([
        {
            "Shelf Level": TIER_LABELS.get(int(_r["home_tier"]), f"Row {_r['home_tier']}"),
            "Storage Module": MODULE_NAMES.get(str(_r["assigned_mechanism"]), (str(_r["assigned_mechanism"]), ""))[0],
            "Compartment": str(_r["assigned_slot"]),
            "Bay Position": str(_r["bay_side"]),
            "Center (cm)": round(float(_r["x_center_cm"]), 1),
            "SKU": str(_r["sku_code"]),
            "Product": str(_r["product_name"]),
            "Copy": int(_r["copy_index"]),
            "Diameter (cm)": round(float(_r["outer_diameter_cm"]), 1),
            "Height (cm)": round(float(_r["roll_height_cm"]), 1),
            "Cored": "Yes" if bool(_r["has_core"]) else "No",
            "Rush Priority": "Yes" if bool(_r["is_rush_priority"]) else "No",
        }
        for _, _r in slotting_df.iterrows()
    ])

    _ori_justify = {
        "Rank": "right",
        "ID": "left",
        "Topology": "left",
        "Mid Shelf": "left",
        "Bottom Shelf": "left",
        "Top Shelf": "left",
        "Variant Rule": "left",
        "Backup Rule": "left",
        "Shelf Order": "left",
        "Same-Drawer A/B/C": "right",
        "Accuracy": "right",
        "Vs. Baseline Acc.": "right",
        "Relative Effort": "right",
        "Score": "right",
    }

    mo.vstack([
        mo.md(
            f"## {len(MODULAR_TOPOLOGIES)}-topology ({len(ori_df)}-layout) search and recommended roll placement\n\n"
            f"Crossed all {len(MODULAR_TOPOLOGIES)} physically feasible three-shelf hardware topologies (requiring at least one `9in` drawer row or wire bin so the widest `23.0 cm` OD roll fits) with {len(POSITIONING_POLICIES)} roll placement policies ({len(ori_df)} total layouts). "
            f"Effort is scaled relative to `{_ori_base_id}` (`1.00x`). The highest-scoring layout in the search is `{_top_ori_label}` "
            f"({_top_mid_name} on Mid Shelf, {_top_btm_name} on Bottom Shelf, sliding pegboard on Top Shelf, `{_top_var_name}` grouping, and backup copies moved to the bottom shelf)."
        ),
        mo.image(
            src=root_dir / "figures" / "fig5_orientation_search_pareto.png",
            alt="Pareto frontier and top hardware topologies across 165 feasible shelf topologies",
            width=960,
        ),
        mo.md(
            "*Figure 5: Pareto frontier of retrieval effort vs. accuracy across 165 feasible hardware topologies and 825 orientations, alongside top 12 architectures.*"
        ),
        mo.md(
            "Plotting all 825 feasible combinations (`165` hardware topologies x `5` placement rules) shows uniform `45 in` (`114.3 cm`) drawer topologies dominating the top-left Pareto frontier. The top overall layout is `TOPO-053` (`ORI-261`: `Top: Peg | Mid: 3D9_3D6_4P | Btm: 3D9_3D6_4P`, `90.5%` search accuracy at `0.25x` search effort), followed closely by `TOPO-051` (`Mid: 3D9_3D6_4P | Btm: 5D9_4P`, `90.4%`), `TOPO-027` (`Mid: 5D9_4P | Btm: 3D9_3D6_4P`, `88.5%`), and `TOPO-025` (`Mid: 5D9_4P | Btm: 5D9_4P`, `88.4%`)."
        ),
        mo.md(f"### All {len(MODULAR_TOPOLOGIES)} feasible `(Mid, Btm)` hardware topologies"),
        mo.ui.table(
            _topo_summary_df,
            selection=None,
            show_column_summaries=False,
            page_size=15,
            text_justify_columns={
                "Rank": "right",
                "Topology": "left",
                "Mid Shelf": "left",
                "Bottom Shelf": "left",
                "Top Shelf": "left",
                "Best ID": "left",
                "Best Variant Rule": "left",
                "Best Accuracy": "right",
                "Best vs. Baseline": "right",
                "Mean Accuracy": "right",
                "Best Effort": "right",
                "Best Score": "right",
            },
        ),
        mo.md(
            "Ranks every `(Mid, Btm)` hardware pair out of all 165 feasible combinations by `Best Score`. `Best Accuracy` shows the top placement policy achievable on that hardware, whereas `Mean Accuracy` averages all 5 tested placement rules; the gap between those two columns measures how sensitive each hardware frame is to proper SKU slotting."
        ),
        mo.md("### Summary by variant placement rule"),
        mo.ui.table(
            _abc_agg_df,
            selection=None,
            show_column_summaries=False,
            pagination=False,
            show_search=False,
            text_justify_columns={
                "Variant Rule": "left",
                "Tested Layouts": "right",
                "Mean Accuracy": "right",
                "Best Accuracy": "right",
                "Best vs. Baseline": "right",
                "Mean Effort": "right",
                "Best Score": "right",
            },
        ),
        mo.md(
            "Compares roll-grouping policies across all 165 topologies. `Adjacent lanes (same drawer)`, `Shared lane`, and `Same shelf bay` each test `165` layouts (`1` policy per topology), while `Split by roll height` covers `330` (`center_out` plus the `random_perm` baseline per topology). Keeping `A/B/C` variants in the same drawer (`68.0%` shared lane, `66.6%` adjacent lanes) beats height-splitting (`42.9%`) by nearly `24` percentage points on average."
        ),
        mo.md("### Pareto-optimal layouts"),
        mo.ui.table(
            _pareto_df,
            selection=None,
            show_column_summaries=False,
            pagination=False,
            show_search=False,
            text_justify_columns=_ori_justify,
        ),
        mo.md(
            "A configuration is Pareto-optimal when no other tested layout beats it on both accuracy and retrieval/return effort at once. `ORI-261` (`TOPO-053`: `Mid: 3D9_3D6_4P | Btm: 3D9_3D6_4P`, `adjacent_partition`, `90.5%` at `16.66 s`) is the sole Pareto-optimal layout across all `825` tested configurations, achieving both the highest accuracy and lowest retrieval/return latency."
        ),
        mo.md(f"### All {len(ori_df)} tested orientations"),
        mo.ui.table(
            _all_ori_clean_df,
            selection=None,
            show_column_summaries=False,
            page_size=15,
            text_justify_columns=_ori_justify,
        ),
        mo.md(
            f"Full ranked table of all {len(ori_df)} combinations sorted by `Score` (`100 x Accuracy - 0.85 x FetchTime`), letting you compare every `(Mid, Btm)` shelf pair under each individual variant and backup placement rule."
        ),
        mo.md(f"### Recommended 128-roll placement for `{_top_ori_label}`"),
        mo.ui.table(
            _slot_clean_df,
            selection=None,
            show_column_summaries=False,
            page_size=15,
            text_justify_columns={
                "Shelf Level": "left",
                "Storage Module": "left",
                "Compartment": "left",
                "Bay Position": "left",
                "Center (cm)": "right",
                "SKU": "left",
                "Product": "left",
                "Copy": "right",
                "Diameter (cm)": "right",
                "Height (cm)": "right",
                "Cored": "center",
                "Rush Priority": "center",
            },
        ),
        mo.md(
            "Compartment IDs encode shelf level and address: `R1-DRW-d-Pp` denotes Middle Shelf drawer `d` partition `p`, `R2-DRW-d-Pp` denotes Bottom Shelf drawer `d` partition `p`, and `R3-PEG-Rr-Cc` denotes Top Shelf peg row `r` pin `c`. `Center (cm)` gives the horizontal slot offset along the `120 cm` rack, and `Copy` separates active primary rolls (`#1`) from duplicate reserve rolls (`#2+`) routed to bottom drawers."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Design direction

    Under the configured physical limits and workspace weights, the highest-scoring layout is `TOPO-053` / `ORI-261` (`Top: Peg | Mid: 3D9_3D6_4P | Btm: 3D9_3D6_4P`, `89.5%` accuracy at `0.31x` effort), ahead of `TOPO-051` (`Mid: 3D9_3D6_4P | Btm: 5D9_4P`, `90.4%` search) and `TOPO-025` (`Mid: 5D9_4P | Btm: 5D9_4P`, `88.3%` benchmark). Pairing `3 x 9in` (`21.6 cm` inner) and `3 x 6in` (`14.0 cm` inner) `2x2` drawers on both lower tiers keeps `48` drawer compartments (`8` more than `10` `9in` drawers) while fitting every SKU up to `23.0 cm` OD inside enclosed drawers, avoiding wire-bin dump drift, mixed bin+drawer row friction, and `42 in` vs. `45 in` shelf-width misalignment. Grouping `A`, `B`, and `C` variants together inside `2x2` drawers, routing duplicate reserve copies to the bottom shelf (`hybrid_reserve`), hanging lower-frequency cored stock (`OD <= 15.24 cm`) on the 32-peg top sliding pegboard, and maintaining crisp lane labels keeps accuracy above the `80%` floor.

    ## Model limitations

    Effort is evaluated as a ratio relative to `ORI-258` (`0.31x` vs. `1.00x`) using Fitts's movement and rummaging terms rather than calibrated shop-floor stopwatch data. Core error and recovery parameters come from published human-reliability handbooks (`THERP`), ergonomic reach data, and warehouse/retail studies (`[2-7]`), while the slot-deficit, row-width alignment, and bin-penalty weights reflect explicit workspace design constraints rather than plant error logs; exact Cycle 1,000 percentages (`89.5%` vs. `73.2%`) should be read as comparative model estimates. Inventory stays fixed at `128` physical rolls across `60` SKUs over `1,000` cycles without partial roll depletion, mid-shift restocking, or operator contention in front of the `120 cm` bay. Finally, the six-state Markov chain uses tier-level averages (`TVD <= 0.0033`) and relies on the Monte Carlo engine (`50` runs for benchmarks, `6` runs per candidate across `825` combinations) to capture lane-level capacities, `OD` limits (`14.0 cm` 6in drawers, `21.6 cm` 9in drawers, `15.24 cm` pegs), and `3P` vs. `4P` geometry.
    """)
    return


if __name__ == "__main__":
    app.run()
