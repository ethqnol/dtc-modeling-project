from std.collections import List
from std.pathlib import Path
from std.testing import TestSuite, assert_almost_equal, assert_equal, assert_true

from src.markov_analysis import analyze_scenario_markov, run_all_markov_analyses
from src.orientation_search import build_modular_topologies, build_orientation_scenarios
from src.sim_config import (
    AppConfig,
    ExperimentScenario,
    InventoryData,
    build_default_experiments,
    count_drawer_slots,
    count_mixed_bin_drawer_rows,
    count_shelf_complexity,
    has_row_width_mismatch,
    load_app_config,
    load_roll_inventory,
)
from src.simulation_engine import (
    allocate_home_slots,
    compute_layout_metrics,
    create_shelf_slots,
    run_experiment_monte_carlo,
)


def _ws_dir() -> String:
    if Path("data/roll-data.csv").exists():
        return "."
    return "/home/ewu/Code/dtc-simulation"


def _cfg_and_inv() raises -> Tuple[AppConfig, InventoryData]:
    var ws = _ws_dir()
    var cfg = load_app_config(ws + "/config/shelf_geometry.yaml", ws + "/config/shelf_config.yaml")
    var inv = load_roll_inventory(ws + "/data/roll-data.csv", cfg.rush)
    return (cfg^, inv^)


def test_inventory_loading_matches_csv() raises:
    var data = _cfg_and_inv()
    var inv = data[1].copy()
    assert_equal(len(inv.variants), 60)
    assert_equal(len(inv.rolls), 128)

    var prods = List[String]()
    var coreless_cnt = 0
    for v in inv.variants:
        var seen = False
        for p in prods:
            if p == v.product_name:
                seen = True
                break
        if not seen:
            prods.append(v.product_name)
        if not v.has_core:
            coreless_cnt += 1
    assert_equal(len(prods), 40)
    assert_equal(coreless_cnt, 1)

    var sum_c: Float64 = 0.0
    var sum_r: Float64 = 0.0
    for f in inv.freq_calm:
        sum_c += f
    for f in inv.freq_rush:
        sum_r += f
    assert_almost_equal(sum_c, 1.0, atol=1e-9)
    assert_almost_equal(sum_r, 1.0, atol=1e-9)

    for i in range(inv.num_variants):
        for j in range(inv.num_variants):
            assert_almost_equal(inv.co(i, j), inv.co(j, i), atol=1e-12)
            if i != j and inv.variants[i].product_name == inv.variants[j].product_name:
                assert_almost_equal(inv.co(i, j), 0.90, atol=1e-9)


def test_allocation_respects_coreless_and_all_rolls_placed() raises:
    var data = _cfg_and_inv()
    var cfg = data[0].copy()
    var inv = data[1].copy()
    var experiments = build_default_experiments(cfg)

    for i in range(len(experiments)):
        var exp = experiments[i].copy()
        var rolls = inv.rolls.copy()
        var slots = allocate_home_slots(exp, inv.variants, rolls)
        assert_equal(len(rolls), 128)
        for r in rolls:
            assert_true(r.home_slot_idx >= 0)
            var s = slots[r.home_slot_idx].copy()
            if not r.has_core:
                assert_true(not s.is_peg())
            if s.is_drawer() or s.is_peg():
                assert_true(r.outer_diameter_cm <= s.max_od_cm)

        if exp.exp_id == "ORI-121":
            for s in slots:
                if s.is_drawer():
                    assert_almost_equal(s.max_od_cm, 23.0, atol=1e-6)

    var mix_scen = experiments[0].copy()
    mix_scen.row0 = "3D9_3D6_4P"
    mix_scen.row1 = "3D9_3D6_4P"
    mix_scen.row2 = "SLIDING_PEG"
    var mix_slots = create_shelf_slots(mix_scen)
    var mid_9 = 0
    var mid_6 = 0
    for s in mix_slots:
        if s.tier_index == 1 and s.is_drawer():
            if s.max_od_cm >= 22.0:
                mid_9 += 1
            else:
                mid_6 += 1
    assert_equal(mid_9, 12)
    assert_equal(mid_6, 12)


def test_markov_chain_invariants() raises:
    var data = _cfg_and_inv()
    var cfg = data[0].copy()
    var experiments = build_default_experiments(cfg)
    var results = run_all_markov_analyses(experiments, cfg.mechanisms)
    assert_equal(len(results), 8)

    for res in results:
        assert_true(res.max_row_sum_error < 1e-12)
        assert_true(res.is_irreducible)
        assert_equal(res.period, 1)
        assert_true(res.stationary_residual < 1e-10)
        assert_true(res.total_variation_distance < 0.02)
        if res.exp_id == "ORI-258":
            assert_almost_equal(res.stationary_dist[0], 0.09440, atol=1e-4)

    var champ = experiments[5].copy()
    champ.exp_id = "ORI-261"
    champ.row0 = "3D9_3D6_4P"
    champ.row1 = "3D9_3D6_4P"
    var mr_champ = analyze_scenario_markov(champ, cfg.mechanisms, "shift_composite")
    assert_almost_equal(mr_champ.stationary_dist[0], 0.81105, atol=1e-4)


def test_simulation_short_run_and_coreless_safety() raises:
    var data = _cfg_and_inv()
    var cfg = data[0].copy()
    var inv = data[1].copy()
    var experiments = build_default_experiments(cfg)

    var s_base = run_experiment_monte_carlo(
        experiments[0],
        inv.variants,
        inv.rolls,
        inv.freq_calm,
        inv.freq_rush,
        inv.co_matrix,
        inv.num_variants,
        inv.high_assoc_siblings,
        cfg.mechanisms,
        2,
        101,
    )
    var s_121 = run_experiment_monte_carlo(
        experiments[5],
        inv.variants,
        inv.rolls,
        inv.freq_calm,
        inv.freq_rush,
        inv.co_matrix,
        inv.num_variants,
        inv.high_assoc_siblings,
        cfg.mechanisms,
        2,
        101,
    )
    assert_true(s_base.final_acc_mean >= 0.0 and s_base.final_acc_mean <= 1.0)
    assert_true(s_121.final_acc_mean > s_base.final_acc_mean)
    assert_true(s_121.overall_latency_mean < s_base.overall_latency_mean)


def test_90pct_abc_association_and_825_orientations() raises:
    var data = _cfg_and_inv()
    var cfg = data[0].copy()
    var inv = data[1].copy()
    var topos = build_modular_topologies()
    var cands = build_orientation_scenarios(cfg)
    assert_equal(len(topos), 165)
    assert_equal(len(cands), 825)

    var found_53 = False
    for tp in topos:
        if tp.topo_id == "TOPO-053" and tp.row0 == "3D9_3D6_4P" and tp.row1 == "3D9_3D6_4P":
            found_53 = True
    assert_true(found_53)

    var ori261 = cands[0].scenario.copy()
    for c in cands:
        if c.scenario.exp_id == "ORI-261":
            ori261 = c.scenario.copy()
            break
    var rolls = inv.rolls.copy()
    var slots = allocate_home_slots(ori261, inv.variants, rolls)
    var lm = compute_layout_metrics(rolls, slots)
    assert_true(lm.abc_same_drawer_pct >= 0.85)


def test_config_loading_and_complexity_penalties() raises:
    var data = _cfg_and_inv()
    var cfg = data[0].copy()
    assert_almost_equal(cfg.behavior.bin_dump_error_penalty, 0.11, atol=1e-9)
    assert_almost_equal(cfg.behavior.mech_diversity_penalty, 0.10, atol=1e-9)
    assert_almost_equal(cfg.behavior.slot_deficit_penalty, 1.20, atol=1e-9)
    assert_almost_equal(cfg.behavior.row_width_mismatch_penalty, 0.25, atol=1e-9)

    var c1 = count_shelf_complexity("7D6_4P", "2B", "SLIDING_PEG")
    assert_equal(c1[0], 2)
    assert_equal(c1[1], 1)
    assert_equal(count_mixed_bin_drawer_rows("3D9_3D6_4P", "1B_3D9_4P", "SLIDING_PEG"), 1)
    assert_true(has_row_width_mismatch("7D6_4P", "5D9_4P", "SLIDING_PEG"))
    assert_true(not has_row_width_mismatch("3D9_3D6_4P", "3D9_3D6_4P", "SLIDING_PEG"))
    assert_equal(count_drawer_slots("3D9_3D6_4P", "3D9_3D6_4P", "SLIDING_PEG"), 48)
    assert_equal(count_drawer_slots("5D9_4P", "5D9_4P", "SLIDING_PEG"), 40)

    var exps = build_default_experiments(cfg)
    var s_uniform = exps[5].copy()
    s_uniform.row0 = "3D9_3D6_4P"
    s_uniform.row1 = "3D9_3D6_4P"
    var s_mismatch = exps[5].copy()
    s_mismatch.row0 = "7D6_4P"
    s_mismatch.row1 = "5D9_4P"
    var pi_u = analyze_scenario_markov(s_uniform, cfg.mechanisms).stationary_dist[0]
    var pi_m = analyze_scenario_markov(s_mismatch, cfg.mechanisms).stationary_dist[0]
    assert_true(pi_u > pi_m)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
