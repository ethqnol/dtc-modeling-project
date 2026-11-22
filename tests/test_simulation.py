import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.markov_analysis import analyze_scenario_markov, build_scenario_transition_matrix, run_all_markov_analyses
from src.orientation_search import MODULAR_TOPOLOGIES, build_orientation_scenarios
from src.sim_config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_CSV_PATH,
    BehavioralConfig,
    ExperimentScenario,
    RollItem,
    build_default_experiments,
    count_shelf_complexity,
    load_config_dict,
    load_roll_inventory,
    make_behavioral_config,
    save_config_file,
    set_active_config,
)
from src.simulation_engine import (
    allocate_home_slots,
    compute_layout_metrics,
    create_shelf_slots,
    run_single_replication,
)


class TestLabelShelfSimulation(unittest.TestCase):
    def test_inventory_loading_matches_csv(self) -> None:
        variants, rolls, f_calm, f_rush, co_mat = load_roll_inventory(DEFAULT_CSV_PATH)
        self.assertEqual(len(variants), 60)
        self.assertEqual(len({v.product_name for v in variants}), 40)
        self.assertEqual(len(rolls), 128)
        self.assertAlmostEqual(sum(f_calm), 1.0, places=9)
        self.assertAlmostEqual(sum(f_rush), 1.0, places=9)
        self.assertEqual(len(co_mat), 60)

        coreless = [v for v in variants if not v.has_core]
        self.assertEqual(len(coreless), 1)
        self.assertIn("Choco Chip Pumpkin Pie Clean Bites", coreless[0].product_name)
        self.assertEqual(coreless[0].variant_code, "B")

        prio_variants = [v for v in variants if v.is_rush_priority]
        prio_rolls = [r for r in rolls if r.is_rush_priority]
        self.assertEqual(len(prio_variants), 30)
        self.assertEqual(len(prio_rolls), 75)
        self.assertEqual(len({v.product_name for v in prio_variants}), 18)

    def test_allocation_respects_coreless_and_all_rolls_placed(self) -> None:
        variants, base_rolls, _, _, _ = load_roll_inventory(DEFAULT_CSV_PATH)
        for exp in build_default_experiments():
            rolls = [
                RollItem(
                    roll_id=r.roll_id, variant_id=r.variant_id, sku_code=r.sku_code,
                    product_name=r.product_name, variant_code=r.variant_code, copy_index=r.copy_index,
                    outer_diameter_cm=r.outer_diameter_cm, roll_height_cm=r.roll_height_cm,
                    has_core=r.has_core, category=r.category, is_rush_priority=r.is_rush_priority,
                    visual_similarity_group=r.visual_similarity_group,
                )
                for r in base_rolls
            ]
            rolls_by_id = {r.roll_id: r for r in rolls}
            slots = allocate_home_slots(exp, variants, rolls)
            self.assertEqual(sum(len(s.occupant_roll_ids) for s in slots), 128)
            for s in slots:
                if "peg" in s.mechanism_id and len(s.occupant_roll_ids) > 1:
                    self.assertLessEqual(len(s.occupant_roll_ids), 2)
                    for rid in s.occupant_roll_ids:
                        self.assertLessEqual(rolls_by_id[rid].roll_height_cm, 10.0)
            for r in rolls:
                self.assertTrue(r.home_slot != "")
                if not r.has_core:
                    self.assertNotIn("peg", r.home_mechanism)

    def test_markov_chain_invariants(self) -> None:
        results = run_all_markov_analyses()
        self.assertEqual(len(results), 8)
        for res in results:
            self.assertLess(res.max_row_sum_error, 1e-12)
            self.assertTrue(res.is_irreducible)
            self.assertEqual(res.period, 1)
            self.assertAlmostEqual(sum(res.stationary_dist), 1.0, places=10)
            self.assertTrue(all(v > 0.0 for v in res.stationary_dist))
            self.assertLess(res.stationary_residual, 1e-10)
            self.assertGreater(res.spectral_gap, 0.0)
            self.assertLess(res.total_variation_distance, 0.01)
            for idx in range(6):
                self.assertAlmostEqual(
                    res.mfpt_matrix[idx][idx], 1.0 / res.stationary_dist[idx], places=6
                )

    def test_simulation_short_run_and_coreless_safety(self) -> None:
        variants, base_rolls, f_calm, f_rush, co_mat = load_roll_inventory(DEFAULT_CSV_PATH)
        exps = build_default_experiments()
        ori258 = [e for e in exps if e.exp_id == "ORI-258"][0]
        ori258.rush.num_shifts = 2  # 100 cycles quick check
        trace = run_single_replication(ori258, variants, base_rolls, f_calm, f_rush, co_mat, seed=99)
        self.assertEqual(len(trace.accuracy_t), 100)
        self.assertTrue(all(0.0 <= a <= 1.0 for a in trace.accuracy_t))
        self.assertTrue(all(0.0 <= e <= 1.0 for e in trace.entropy_t))
        self.assertTrue(all(l > 0.0 for l in trace.latency_t))

    def test_90pct_abc_association_and_400_orientations(self) -> None:
        variants, base_rolls, _, _, co_mat = load_roll_inventory(DEFAULT_CSV_PATH)
        # Verify 0.90 same-product A/B/C conditional association
        same_prod_pairs = [
            co_mat[i][j]
            for i in range(len(variants))
            for j in range(i + 1, len(variants))
            if variants[i].product_name == variants[j].product_name
        ]
        self.assertGreater(len(same_prod_pairs), 0)
        for val in same_prod_pairs:
            self.assertAlmostEqual(val, 0.90, places=6)

        valid_lower_rows = {
            "7D6_4P", "7D6_3P", "5D9_4P", "5D9_3P", "3D9_3D6_4P",
            "1B_4D6_4P", "1B_4D6_3P", "1B_3D9_4P", "1B_3D9_3P",
            "1D6_2B", "1B_1D6_1B", "2B_1D6", "2B",
        }
        candidates = build_orientation_scenarios()
        self.assertEqual(len(MODULAR_TOPOLOGIES), 165)
        for _, _, row_cfgs in MODULAR_TOPOLOGIES:
            self.assertIn(row_cfgs[0], valid_lower_rows)
            self.assertIn(row_cfgs[1], valid_lower_rows)
            self.assertEqual(row_cfgs[2], "SLIDING_PEG")
        self.assertEqual(len(candidates), 825)

        # Verify 6in 7-drawer shelf, tight 1B+4D6 bay, & 32-peg Top Sliding Peg capacity
        ori_bin = next(s for _, _, s in candidates if s.row_configs == ("7D6_4P", "1B_4D6_4P", "SLIDING_PEG"))
        slots_001 = create_shelf_slots(ori_bin)
        mid_drw_slots = [s for s in slots_001 if s.tier_index == 1 and "drawer" in s.mechanism_id]
        btm_drw_slots = [s for s in slots_001 if s.tier_index == 2 and "drawer" in s.mechanism_id]
        btm_bin_slots = [s for s in slots_001 if s.tier_index == 2 and "wire_bin" in s.mechanism_id]
        peg_slots = [s for s in slots_001 if "pegboard" in s.mechanism_id]
        self.assertEqual(len(mid_drw_slots), 28)   # 7 x 6in drawers x 4 partitions
        self.assertEqual(len(btm_drw_slots), 16)   # 4 x 6in drawers x 4 partitions
        self.assertEqual(len(btm_bin_slots), 4)    # 1 wire bin x 4 sub-zones
        self.assertEqual(len(peg_slots), 32)       # 8 rows x 4 pins
        self.assertEqual(sum(s.max_rolls_hard for s in peg_slots), 64)

        # Verify 9in drawer configuration: 5 drawers on Mid (20 slots) + 3 drawers beside 1 bin on Btm (12 slots), fits all labels (23.0 cm OD)
        ori_9in = next(s for _, _, s in candidates if s.row_configs == ("5D9_4P", "1B_3D9_4P", "SLIDING_PEG"))
        slots_9in = create_shelf_slots(ori_9in)
        mid_9_slots = [s for s in slots_9in if s.tier_index == 1 and "drawer" in s.mechanism_id]
        btm_9_slots = [s for s in slots_9in if s.tier_index == 2 and "drawer" in s.mechanism_id]
        self.assertEqual(len(mid_9_slots), 20)
        self.assertEqual(len(btm_9_slots), 12)
        for s in mid_9_slots + btm_9_slots:
            self.assertAlmostEqual(s.max_od_cm, 23.0)
            self.assertEqual(s.max_rolls_hard, 8)

        # Verify 3D9_3D6_4P intra-row mix: 3 x 9in (12 slots, 23.0 cm) + 3 x 6in (12 slots, 14.0 cm)
        ori_mix = next(s for _, _, s in candidates if s.row_configs == ("3D9_3D6_4P", "1B_3D9_4P", "SLIDING_PEG"))
        mid_mix_slots = [s for s in create_shelf_slots(ori_mix) if s.tier_index == 1]
        self.assertEqual(len(mid_mix_slots), 24)
        self.assertEqual(sum(1 for s in mid_mix_slots if abs(s.max_od_cm - 23.0) < 1e-6), 12)
        self.assertEqual(sum(1 for s in mid_mix_slots if abs(s.max_od_cm - 14.0) < 1e-6), 12)

        # Verify 3P (2:1:1) drawer topology: 4 x 3 = 12 Btm drawer compartments with 2:1:1 capacity split beside 1 bin
        ori_3p = next(s for _, _, s in candidates if s.row_configs == ("7D6_4P", "1B_4D6_3P", "SLIDING_PEG"))
        slots_061 = create_shelf_slots(ori_3p)
        btm_3p_slots = [s for s in slots_061 if s.tier_index == 2 and "drawer" in s.mechanism_id]
        self.assertEqual(len(btm_3p_slots), 12)
        for s in btm_3p_slots:
            self.assertEqual(s.partition_count, 3)
            self.assertEqual(s.max_rolls_hard, 10 if s.sub_idx == 0 else 6)
            self.assertAlmostEqual(s.max_od_cm, 14.0)

        # Verify 6in-only drawer topology fails because >15.24 cm OD rolls cannot fit 6in drawers (14.0 cm) or pegs (15.24 cm)
        no_bin = ExperimentScenario(
            exp_id="NO-BIN", name="NO-BIN", topology="M_modular", strategy="C_hybrid",
            drawer_partitions=4, cue_salience=0.9, target_fill_ratio=0.8,
            row_configs=("7D6_4P", "7D6_4P", "SLIDING_PEG"),
        )
        with self.assertRaises(RuntimeError):
            allocate_home_slots(no_bin, variants, [
                RollItem(
                    roll_id=r.roll_id, variant_id=r.variant_id, sku_code=r.sku_code,
                    product_name=r.product_name, variant_code=r.variant_code, copy_index=r.copy_index,
                    outer_diameter_cm=r.outer_diameter_cm, roll_height_cm=r.roll_height_cm,
                    has_core=r.has_core, category=r.category, is_rush_priority=r.is_rush_priority,
                    visual_similarity_group=r.visual_similarity_group,
                )
                for r in base_rolls
            ])

        # Verify flush 1D_2B, 1B_1D_1B, and 2B_1D bottom configurations produce 4 drawer + 8 bin slots with unique IDs
        for btm_code in ("1D_2B", "1B_1D_1B", "2B_1D"):
            sc = ExperimentScenario(
                exp_id=f"TEST-{btm_code}", name=btm_code, topology="M_modular", strategy="C_hybrid",
                drawer_partitions=4, cue_salience=0.9, target_fill_ratio=0.8,
                row_configs=("7D_4P", btm_code, "SLIDING_PEG"),
            )
            btm_slots = [s for s in create_shelf_slots(sc) if s.tier_index == 2]
            self.assertEqual(len(btm_slots), 12)
            self.assertEqual(sum(1 for s in btm_slots if "drawer" in s.mechanism_id), 4)
            self.assertEqual(sum(1 for s in btm_slots if "wire_bin" in s.mechanism_id), 8)
            self.assertEqual(len({s.slot_id for s in btm_slots}), 12)

        # Confirm Top shelf rejects non-peg modules
        bad_top = ExperimentScenario(
            exp_id="BAD-TOP", name="BAD-TOP", topology="M_modular", strategy="C_hybrid",
            drawer_partitions=4, cue_salience=0.9, target_fill_ratio=0.8,
            row_configs=("7D_4P", "1B_4D_4P", "2B"),
        )
        with self.assertRaises(ValueError):
            create_shelf_slots(bad_top)

        # Spot check one from every topology + adjacent_partition and same_slot modes
        seen_topos = set()
        for topo_id, _, scen in candidates:
            if topo_id in seen_topos and scen.abc_colocation_mode != "adjacent_partition":
                continue
            seen_topos.add(topo_id)
            rolls = [
                RollItem(
                    roll_id=r.roll_id, variant_id=r.variant_id, sku_code=r.sku_code,
                    product_name=r.product_name, variant_code=r.variant_code, copy_index=r.copy_index,
                    outer_diameter_cm=r.outer_diameter_cm, roll_height_cm=r.roll_height_cm,
                    has_core=r.has_core, category=r.category, is_rush_priority=r.is_rush_priority,
                    visual_similarity_group=r.visual_similarity_group,
                )
                for r in base_rolls
            ]
            rolls_by_id = {r.roll_id: r for r in rolls}
            slots = allocate_home_slots(scen, variants, rolls)
            self.assertEqual(sum(len(s.occupant_roll_ids) for s in slots), 128)
            for s in slots:
                if "peg" in s.mechanism_id and len(s.occupant_roll_ids) > 1:
                    self.assertLessEqual(len(s.occupant_roll_ids), 2)
                    for rid in s.occupant_roll_ids:
                        self.assertLessEqual(rolls_by_id[rid].roll_height_cm, 10.0)
            metrics = compute_layout_metrics(rolls, slots)
            self.assertGreaterEqual(metrics["abc_same_drawer_pct"], 0.0)
            for r in rolls:
                if not r.has_core:
                    self.assertNotIn("peg", r.home_mechanism)

        # Verify modular champion scenario is valid for both simulation and Markov chain
        champ = next(s for _, _, s in candidates if s.row_configs == ("7D6_4P", "1B_4D6_4P", "SLIDING_PEG"))
        mr = run_all_markov_analyses([champ])[0]
        self.assertLess(mr.max_row_sum_error, 1e-12)
        self.assertLess(mr.total_variation_distance, 0.01)

    def test_config_loading_and_complexity_penalties(self) -> None:
        # 1) Default YAML weights & separate geometry YAML load cleanly
        yaml_cfg = load_config_dict(DEFAULT_CONFIG_PATH)
        self.assertIn("behavioral_weights", yaml_cfg)
        self.assertNotIn("geometry", yaml_cfg)
        self.assertAlmostEqual(yaml_cfg["behavioral_weights"]["bin_dump_error_penalty"], 0.11)
        self.assertAlmostEqual(yaml_cfg["behavioral_weights"]["mech_diversity_penalty"], 0.10)

        # 2) Temporary JSON & TOON overrides load cleanly and restore default cleanly
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "override.json"
            json_path.write_text(
                json.dumps({"behavioral_weights": {"bin_dump_error_penalty": 0.25}}),
                encoding="utf-8",
            )
            toon_path = Path(tmp) / "export.toon"
            try:
                set_active_config(json_path)
                self.assertAlmostEqual(make_behavioral_config().bin_dump_error_penalty, 0.25)
                save_config_file(toon_path)
                toon_cfg = load_config_dict(toon_path)
                self.assertAlmostEqual(toon_cfg["behavioral_weights"]["bin_dump_error_penalty"], 0.25)
                set_active_config(toon_path)
                self.assertAlmostEqual(make_behavioral_config().mech_diversity_penalty, 0.10)
            finally:
                set_active_config(None)
        self.assertAlmostEqual(make_behavioral_config().bin_dump_error_penalty, 0.11)

        # 3) Complexity counters across 2-style uniform vs mixed shelves
        self.assertEqual(count_shelf_complexity(("7D_4P", "7D_4P", "SLIDING_PEG")), (0, 0))
        self.assertEqual(count_shelf_complexity(("7D_4P", "7D_3P", "SLIDING_PEG")), (0, 1))
        self.assertEqual(count_shelf_complexity(("7D_4P", "1B_4D_4P", "SLIDING_PEG")), (1, 1))
        self.assertEqual(count_shelf_complexity(("7D_4P", "1B_2D4P_2D3P", "SLIDING_PEG")), (1, 2))

        # 4) Penalties reduce steady-state pi_0 for mixed/bin layouts while leaving uniform 7D(4P)+7D(4P)+Peg invariant
        def _mk_scen(row_cfgs, beh):
            return ExperimentScenario(
                exp_id="T", name="T", topology="M_modular", strategy="C_hybrid",
                drawer_partitions=4, cue_salience=0.90, target_fill_ratio=0.80,
                behavior=beh, row_configs=row_cfgs,
                abc_colocation_mode="adjacent_partition", overflow_mode="hybrid_reserve",
            )

        low_pen = BehavioralConfig(bin_dump_error_penalty=0.0, bin_dump_routing_weight=0.0, mech_diversity_penalty=0.0)
        high_pen = BehavioralConfig(bin_dump_error_penalty=0.20, bin_dump_routing_weight=0.50, mech_diversity_penalty=0.18)

        pi_uniform_low = analyze_scenario_markov(_mk_scen(("7D_4P", "7D_4P", "SLIDING_PEG"), low_pen)).stationary_dist[0]
        pi_uniform_high = analyze_scenario_markov(_mk_scen(("7D_4P", "7D_4P", "SLIDING_PEG"), high_pen)).stationary_dist[0]
        self.assertAlmostEqual(pi_uniform_low, pi_uniform_high, places=9)

        pi_mix_low = analyze_scenario_markov(_mk_scen(("7D_4P", "1B_2D4P_2D3P", "SLIDING_PEG"), low_pen)).stationary_dist[0]
        pi_mix_high = analyze_scenario_markov(_mk_scen(("7D_4P", "1B_2D4P_2D3P", "SLIDING_PEG"), high_pen)).stationary_dist[0]
        self.assertGreater(pi_mix_low, pi_mix_high)

        p_bin_low = build_scenario_transition_matrix(_mk_scen(("7D_4P", "1B_4D_4P", "SLIDING_PEG"), low_pen), "rush")[0][3]
        p_bin_high = build_scenario_transition_matrix(_mk_scen(("7D_4P", "1B_4D_4P", "SLIDING_PEG"), high_pen), "rush")[0][3]
        self.assertGreater(p_bin_high, p_bin_low)


if __name__ == "__main__":
    unittest.main()
