from std.collections import Dict, List
from std.math import log2, sqrt

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
    RollItem,
    RollVariantSKU,
    count_drawer_slots,
    count_mixed_bin_drawer_rows,
    count_shelf_complexity,
    has_mixed_drawer_sizes,
    has_row_width_mismatch,
    mech_id_to_name,
)


def _list_contains_int(lst: List[Int], val: Int) -> Bool:
    for x in lst:
        if x == val:
            return True
    return False


def _remove_int_first(mut lst: List[Int], val: Int):
    for i in range(len(lst)):
        if lst[i] == val:
            _ = lst.pop(i)
            return


@fieldwise_init
struct PhysicalSlot(Copyable, Movable):
    """A single physical peg, drawer partition, or wire container slot."""

    var slot_idx: Int
    var slot_id: String
    var mech_id: Int
    var mechanism_id: String
    var tier_index: Int
    var group_idx: Int
    var sub_idx: Int
    var max_rolls_hard: Int
    var max_stack_height_cm: Float64
    var clear_width_cm: Float64
    var supports_coreless: Bool
    var is_pull_down: Bool
    var occupant_roll_ids: List[Int]
    var assigned_variant_ids: List[Int]
    var nominal_capacity: Float64
    var bay_side: Int
    var x_center_cm: Float64
    var partition_count: Int
    var max_od_cm: Float64

    def is_peg(self) -> Bool:
        return self.mech_id == MECH_PEGBOARD_OVH

    def is_drawer(self) -> Bool:
        return self.mech_id == MECH_SPLIT_DRAWER_EYE or self.mech_id == MECH_SPLIT_DRAWER_MID

    def is_wire_bin(self) -> Bool:
        return self.mech_id == MECH_WIRE_BIN_EYE or self.mech_id == MECH_WIRE_BIN_MID

    def can_physically_accept(self, roll: RollItem, rolls: List[RollItem]) -> Bool:
        if roll.outer_diameter_cm > self.max_od_cm:
            return False
        if not self.supports_coreless and not roll.has_core:
            return False
        if len(self.occupant_roll_ids) >= self.max_rolls_hard:
            return False
        if self.is_peg():
            if len(self.occupant_roll_ids) > 0:
                if roll.roll_height_cm > 10.0:
                    return False
                for rid in self.occupant_roll_ids:
                    if rolls[rid].roll_height_cm > 10.0:
                        return False
            var used_h: Float64 = 0.0
            for rid in self.occupant_roll_ids:
                used_h += rolls[rid].roll_height_cm
            if used_h + roll.roll_height_cm > self.max_stack_height_cm:
                return False
        return True

    def fill_ratio(self, rolls: List[RollItem], target_fill_ratio: Float64) -> Float64:
        var n_occ = len(self.occupant_roll_ids)
        if n_occ == 0:
            return 0.0
        var scale = target_fill_ratio / 0.80
        var base_rho: Float64
        if self.is_peg():
            if n_occ == 1:
                base_rho = 0.55
            else:
                var used_h: Float64 = 0.0
                for rid in self.occupant_roll_ids:
                    used_h += rolls[rid].roll_height_cm
                var max_h = self.max_stack_height_cm if self.max_stack_height_cm > 1.0 else 1.0
                base_rho = max(Float64(n_occ) / 2.0, used_h / max_h)
        else:
            var nom = self.nominal_capacity if self.nominal_capacity > 1.0 else 1.0
            base_rho = Float64(n_occ) / nom
        var r = base_rho * scale
        return r if r < 1.01 else 1.01


@fieldwise_init
struct LayoutMetrics(Copyable, Movable):
    var abc_same_drawer_pct: Float64
    var abc_same_tier_pct: Float64
    var active_eye_share: Float64
    var mean_eye_fill_ratio: Float64


@fieldwise_init
struct SlotHeatmapEntry(Copyable, Movable):
    var slot_id: String
    var mechanism_id: String
    var tier_index: Int
    var group_idx: Int
    var sub_idx: Int
    var home_rolls: Int
    var mean_final_occupants: Float64
    var mean_final_misplaced: Float64
    var misplace_density: Float64


@fieldwise_init
struct ReplicationTrace(Copyable, Movable):
    var accuracy_t: List[Float64]
    var entropy_t: List[Float64]
    var cross_tier_contam_t: List[Float64]
    var eye_drawer_dump_share_t: List[Float64]
    var latency_t: List[Float64]
    var per_mech_accuracy_final: List[Float64]
    var final_slot_misplace_counts: List[Int]
    var final_slot_occupant_counts: List[Int]
    var t_crit: Int


@fieldwise_init
struct ExperimentSummary(Copyable, Movable):
    var scenario: ExperimentScenario
    var n_replications: Int
    var cycles: Int
    var acc_mean: List[Float64]
    var acc_std: List[Float64]
    var entropy_mean: List[Float64]
    var contam_mean: List[Float64]
    var eye_dump_mean: List[Float64]
    var latency_mean: List[Float64]
    var latency_std: List[Float64]
    var final_acc_mean: Float64
    var final_acc_std: Float64
    var final_entropy_mean: Float64
    var final_contam_mean: Float64
    var overall_latency_mean: Float64
    var rush_latency_mean: Float64
    var calm_latency_mean: Float64
    var t_crit_mean: Float64
    var collapse_rate: Float64
    var per_mech_acc_final: List[Float64]
    var slot_heatmap_data: List[SlotHeatmapEntry]
    var home_t: List[Float64]
    var within_drift_t: List[Float64]
    var other_contam_t: List[Float64]


def _append_drawers(
    mut slots: List[PhysicalSlot],
    scenario: ExperimentScenario,
    tier_idx: Int,
    part_counts: List[Int],
    total_w_cm: Float64,
    x_start_cm: Float64,
    d_offset: Int,
    is_9in: Bool,
):
    var geom = scenario.geometry.copy()
    var mech_id = MECH_SPLIT_DRAWER_EYE if tier_idx == 1 else MECH_SPLIT_DRAWER_MID
    var mech_name = mech_id_to_name(mech_id)
    var n_drw = len(part_counts)
    var slot_w = total_w_cm / Float64(n_drw if n_drw > 1 else 1)
    var inner_w = geom.drawer_9in_inner_width_cm if is_9in else geom.drawer_inner_width_cm

    for d in range(n_drw):
        var p_cnt = part_counts[d]
        var d_idx = d + d_offset
        var x_center = x_start_cm + (Float64(d) + 0.5) * slot_w
        var bay_side = 0 if x_center < 60.0 else 1
        for p in range(p_cnt):
            var p_width = geom.partition_width_cm(p_cnt, p, inner_w)
            var is_wide_top = (p_cnt == 3) and (p == 0)
            var hard_cap: Int
            var nom_cap: Float64
            if is_9in:
                hard_cap = 13 if is_wide_top else 8
                nom_cap = 9.0 if is_wide_top else 5.0
            else:
                hard_cap = 10 if is_wide_top else 6
                nom_cap = 6.6 if is_wide_top else 3.5
            var s_idx = len(slots)
            var s_id = String(t"R{tier_idx}-DRW-{d_idx + 1}-P{p + 1}")
            slots.append(
                PhysicalSlot(
                    s_idx,
                    s_id,
                    mech_id,
                    mech_name,
                    tier_idx,
                    d_idx,
                    p,
                    hard_cap,
                    geom.drawer_inner_depth_cm,
                    p_width,
                    True,
                    False,
                    List[Int](),
                    List[Int](),
                    nom_cap,
                    bay_side,
                    x_center,
                    p_cnt,
                    23.0 if is_9in else inner_w,
                )
            )


def _append_bins(
    mut slots: List[PhysicalSlot],
    scenario: ExperimentScenario,
    tier_idx: Int,
    n_bins: Int,
    x_start_cm: Float64,
    bay_side_fixed: Int,
    b_offset: Int,
):
    var geom = scenario.geometry.copy()
    var mech_id = MECH_WIRE_BIN_EYE if tier_idx == 1 else MECH_WIRE_BIN_MID
    var mech_name = mech_id_to_name(mech_id)
    var sub_w = geom.wire_bin_bay_width_cm / 2.0
    var total_sub = n_bins * geom.wire_bin_subzones
    for b in range(total_sub):
        var b_idx = b + b_offset
        var bin_idx = b // geom.wire_bin_subzones
        var x_center = (
            x_start_cm
            + Float64(bin_idx) * geom.wire_bin_bay_width_cm
            + Float64(b % 2) * sub_w
            + sub_w / 2.0
        )
        var bay_side = bay_side_fixed if bay_side_fixed >= 0 else (0 if x_center < 60.0 else 1)
        var s_idx = len(slots)
        var s_id = String(t"R{tier_idx}-BIN-{b_idx + 1}")
        slots.append(
            PhysicalSlot(
                s_idx,
                s_id,
                mech_id,
                mech_name,
                tier_idx,
                b_idx,
                0,
                geom.wire_bin_subzone_hard_cap,
                geom.wire_bin_depth_cm,
                geom.wire_bin_width_cm,
                True,
                False,
                List[Int](),
                List[Int](),
                6.2,
                bay_side,
                x_center,
                1,
                geom.wire_bin_width_cm,
            )
        )


def create_shelf_slots(scenario: ExperimentScenario) raises -> List[PhysicalSlot]:
    """Instantiate modular 120 x 60 x 60 cm shelf row slots across 3 vertical levels."""
    if scenario.topology != "M_modular":
        raise Error(t"Unsupported topology: {scenario.topology}")

    var geom = scenario.geometry.copy()
    var row_codes = [scenario.row0, scenario.row1, scenario.row2]
    var peg_rows_count = 0
    for rc in row_codes:
        if "PEG" in rc:
            peg_rows_count += 1
    if peg_rows_count > 1:
        raise Error("Sliding pegboard can only occupy at most one full shelf")

    var slots = List[PhysicalSlot]()
    for t0 in range(3):
        var tier_idx = t0 + 1
        var rc = row_codes[t0]
        if tier_idx == 3 and rc != "SLIDING_PEG":
            raise Error("Top shelf (Row 3) only supports SLIDING_PEG")
        if tier_idx != 3 and rc == "SLIDING_PEG":
            raise Error("SLIDING_PEG is only supported on Top shelf (Row 3)")

        if rc == "7D6_4P" or rc == "7D6_3P" or rc == "7D_4P" or rc == "7D_3P" or rc == "4D4P_3D3P":
            var parts_list = List[Int]()
            if rc == "4D4P_3D3P":
                parts_list = [4, 4, 4, 4, 3, 3, 3]
            else:
                var p_cnt = 4 if rc.endswith("4P") else 3
                for _ in range(geom.drawers_per_full_row):
                    parts_list.append(p_cnt)
            _append_drawers(
                slots,
                scenario,
                tier_idx,
                parts_list,
                Float64(len(parts_list)) * geom.drawer_outer_width_cm,
                0.0,
                0,
                False,
            )
        elif rc == "5D9_4P" or rc == "5D9_3P":
            var p_cnt = 4 if rc.endswith("4P") else 3
            var parts_list = List[Int]()
            for _ in range(geom.drawers_9in_per_full_row):
                parts_list.append(p_cnt)
            _append_drawers(
                slots,
                scenario,
                tier_idx,
                parts_list,
                Float64(len(parts_list)) * geom.drawer_9in_outer_width_cm,
                0.0,
                0,
                True,
            )
        elif rc == "3D9_3D6_4P":
            var span_9 = 3.0 * geom.drawer_9in_outer_width_cm
            var span_6 = 3.0 * geom.drawer_outer_width_cm
            _append_drawers(slots, scenario, tier_idx, [4, 4, 4], span_9, 0.0, 0, True)
            _append_drawers(slots, scenario, tier_idx, [4, 4, 4], span_6, span_9, 3, False)
        elif (
            rc == "1B_4D6_4P"
            or rc == "1B_4D6_3P"
            or rc == "1B_4D_4P"
            or rc == "1B_4D_3P"
            or rc == "1B_2D4P_2D3P"
        ):
            var parts_list = List[Int]()
            if rc == "1B_2D4P_2D3P":
                parts_list = [4, 4, 3, 3]
            else:
                var p_cnt = 4 if rc.endswith("4P") else 3
                for _ in range(geom.drawers_per_half_bay):
                    parts_list.append(p_cnt)
            var drw_span = Float64(len(parts_list)) * geom.drawer_outer_width_cm
            _append_drawers(slots, scenario, tier_idx, parts_list, drw_span, 0.0, 0, False)
            _append_bins(slots, scenario, tier_idx, 1, drw_span, 1, 0)
        elif rc == "1B_3D9_4P" or rc == "1B_3D9_3P":
            var p_cnt = 4 if rc.endswith("4P") else 3
            var parts_list = List[Int]()
            for _ in range(geom.drawers_9in_per_half_bay):
                parts_list.append(p_cnt)
            var drw_span = Float64(len(parts_list)) * geom.drawer_9in_outer_width_cm
            _append_drawers(slots, scenario, tier_idx, parts_list, drw_span, 0.0, 0, True)
            _append_bins(slots, scenario, tier_idx, 1, drw_span, 1, 0)
        elif rc == "1D6_2B" or rc == "1D_2B":
            var drw_span = geom.drawer_outer_width_cm
            _append_drawers(slots, scenario, tier_idx, [4], drw_span, 0.0, 0, False)
            _append_bins(slots, scenario, tier_idx, 2, drw_span, -1, 0)
        elif rc == "1B_1D6_1B" or rc == "1B_1D_1B":
            var bin_w = geom.wire_bin_bay_width_cm
            var drw_w = geom.drawer_outer_width_cm
            _append_bins(slots, scenario, tier_idx, 1, 0.0, 0, 0)
            _append_drawers(slots, scenario, tier_idx, [4], drw_w, bin_w, 0, False)
            _append_bins(slots, scenario, tier_idx, 1, bin_w + drw_w, 1, geom.wire_bin_subzones)
        elif rc == "2B_1D6" or rc == "2B_1D":
            var bin_span = 2.0 * geom.wire_bin_bay_width_cm
            var drw_w = geom.drawer_outer_width_cm
            _append_bins(slots, scenario, tier_idx, 2, 0.0, -1, 0)
            _append_drawers(slots, scenario, tier_idx, [4], drw_w, bin_span, 0, False)
        elif rc == "2B":
            _append_bins(slots, scenario, tier_idx, geom.wire_bins_per_full_row, 0.0, -1, 0)
        elif rc == "SLIDING_PEG":
            var p_div = Float64(geom.pegs_per_row if geom.pegs_per_row > 1 else 1)
            var peg_x_step = geom.shelf_width_cm / p_div
            var peg_name = mech_id_to_name(MECH_PEGBOARD_OVH)
            for r in range(geom.peg_rows):
                for c in range(geom.pegs_per_row):
                    var bay_side = 0 if c < (geom.pegs_per_row // 2) else 1
                    var s_idx = len(slots)
                    var s_id = String(t"R{tier_idx}-PEG-R{r + 1}-C{c + 1}")
                    slots.append(
                        PhysicalSlot(
                            s_idx,
                            s_id,
                            MECH_PEGBOARD_OVH,
                            peg_name,
                            tier_idx,
                            r,
                            c,
                            2,
                            geom.peg_usable_length_cm,
                            geom.peg_horizontal_spacing_cm,
                            False,
                            True,
                            List[Int](),
                            List[Int](),
                            1.65,
                            bay_side,
                            (Float64(c) + 0.5) * peg_x_step,
                            1,
                            geom.peg_max_od_cm,
                        )
                    )
        else:
            raise Error(t"Unknown modular row config: {rc}")

    return slots^


def _assign_roll_to_slot(
    mut roll: RollItem, mut slot: PhysicalSlot, is_home: Bool = True
):
    slot.occupant_roll_ids.append(roll.roll_id)
    roll.current_slot_idx = slot.slot_idx
    roll.current_slot = slot.slot_id
    roll.current_mech_id = slot.mech_id
    roll.current_mechanism = slot.mechanism_id
    if is_home:
        roll.home_slot_idx = slot.slot_idx
        roll.home_slot = slot.slot_id
        roll.home_mech_id = slot.mech_id
        roll.home_mechanism = slot.mechanism_id
        if not _list_contains_int(slot.assigned_variant_ids, roll.variant_id):
            slot.assigned_variant_ids.append(roll.variant_id)


def _move_roll(
    mut roll: RollItem,
    mut slots: List[PhysicalSlot],
    target_slot_idx: Int,
):
    var old_idx = roll.current_slot_idx
    if old_idx >= 0 and old_idx < len(slots):
        _remove_int_first(slots[old_idx].occupant_roll_ids, roll.roll_id)
    slots[target_slot_idx].occupant_roll_ids.append(roll.roll_id)
    roll.current_slot_idx = target_slot_idx
    roll.current_slot = slots[target_slot_idx].slot_id
    roll.current_mech_id = slots[target_slot_idx].mech_id
    roll.current_mechanism = slots[target_slot_idx].mechanism_id


@fieldwise_init
struct _SlotSortKey(Copyable, Movable):
    var slot_idx: Int
    var k0: Int
    var k1: Float64
    var k2: Int
    var k3: Int


def _cmp_slot_sort_key(a: _SlotSortKey, b: _SlotSortKey) -> Bool:
    if a.k0 != b.k0:
        return a.k0 < b.k0
    if a.k1 != b.k1:
        return a.k1 < b.k1
    if a.k2 != b.k2:
        return a.k2 < b.k2
    if a.k3 != b.k3:
        return a.k3 < b.k3
    return a.slot_idx < b.slot_idx


def _order_slot_indices_for_mode(
    slots: List[PhysicalSlot], spatial_mode: String, mut rng: Rng
) -> List[Int]:
    var n = len(slots)
    if spatial_mode == "random_perm":
        var indices = List[Int](capacity=n)
        for i in range(n):
            indices.append(i)
        rng.shuffle_ints(indices)
        return indices^

    var keys = List[_SlotSortKey](capacity=n)
    for i in range(n):
        var s = slots[i].copy()
        if spatial_mode == "center_out" or spatial_mode == "bark_sticker_kit":
            keys.append(_SlotSortKey(i, s.tier_index, abs(s.x_center_cm - 60.0), s.group_idx, s.sub_idx))
        elif spatial_mode == "category_zones":
            keys.append(_SlotSortKey(i, s.tier_index, s.x_center_cm, s.group_idx, s.sub_idx))
        elif spatial_mode == "vertical_column":
            var k1 = Float64(Int(s.x_center_cm // 30.0))
            keys.append(_SlotSortKey(i, s.bay_side, k1, s.tier_index * 100 + s.group_idx, s.sub_idx))
        else:
            keys.append(_SlotSortKey(i, s.tier_index, Float64(s.group_idx), s.sub_idx, 0))

    sort(keys, _cmp_slot_sort_key)
    var out = List[Int](capacity=n)
    for k in keys:
        out.append(k.slot_idx)
    return out^


@fieldwise_init
struct _ProductFamily(Copyable, Movable):
    var product_name: String
    var roll_ids: List[Int]
    var rush_flag: Int
    var peak_freq: Float64
    var category: String
    var kit_flag: Int
    var kit_sub: Int


def _cmp_roll_family_item(
    a_roll_id: Int,
    b_roll_id: Int,
    rolls: List[RollItem],
) -> Bool:
    var ra = rolls[a_roll_id].copy()
    var rb = rolls[b_roll_id].copy()
    if ra.copy_index != rb.copy_index:
        return ra.copy_index < rb.copy_index
    if ra.variant_code != rb.variant_code:
        return ra.variant_code < rb.variant_code
    return ra.roll_id < rb.roll_id


def _sort_roll_ids_family(mut rids: List[Int], rolls: List[RollItem]):
    var n = len(rids)
    for i in range(1, n):
        var key = rids[i]
        var j = i - 1
        while j >= 0 and _cmp_roll_family_item(key, rids[j], rolls):
            rids[j + 1] = rids[j]
            j -= 1
        rids[j + 1] = key


def _cmp_fam_center(a: _ProductFamily, b: _ProductFamily) -> Bool:
    if a.peak_freq != b.peak_freq:
        return a.peak_freq > b.peak_freq
    return a.product_name < b.product_name


def _cmp_fam_category(a: _ProductFamily, b: _ProductFamily) -> Bool:
    if a.category != b.category:
        return a.category < b.category
    if a.peak_freq != b.peak_freq:
        return a.peak_freq > b.peak_freq
    return a.product_name < b.product_name


def _cmp_fam_vertical(a: _ProductFamily, b: _ProductFamily) -> Bool:
    if a.rush_flag != b.rush_flag:
        return a.rush_flag < b.rush_flag
    if a.peak_freq != b.peak_freq:
        return a.peak_freq > b.peak_freq
    return a.product_name < b.product_name


def _cmp_fam_kit(a: _ProductFamily, b: _ProductFamily) -> Bool:
    if a.kit_flag != b.kit_flag:
        return a.kit_flag < b.kit_flag
    if a.kit_sub != b.kit_sub:
        return a.kit_sub < b.kit_sub
    if a.peak_freq != b.peak_freq:
        return a.peak_freq > b.peak_freq
    return a.product_name < b.product_name


def _order_product_families(
    rolls: List[RollItem],
    variants: List[RollVariantSKU],
    spatial_mode: String,
    mut rng: Rng,
) -> List[_ProductFamily]:
    var prod_to_idx = Dict[String, Int]()
    var fams = List[_ProductFamily]()
    for r in rolls:
        var p_opt = prod_to_idx.get(r.product_name)
        var p_idx: Int
        if p_opt:
            p_idx = p_opt.value()
        else:
            p_idx = len(fams)
            prod_to_idx[r.product_name] = p_idx
            var is_kit = 0 if (r.category == "Chocolate bark" or r.category == "Ingredient sticker") else 1
            var sub = 0 if ("Peppermint" in r.product_name) else (1 if ("Almond" in r.product_name) else 2)
            fams.append(_ProductFamily(r.product_name, List[Int](), 1, 0.0, r.category, is_kit, sub))
        fams[p_idx].roll_ids.append(r.roll_id)
        if r.is_rush_priority:
            fams[p_idx].rush_flag = 0
        var rf = variants[r.variant_id].rush_freq
        if rf > fams[p_idx].peak_freq:
            fams[p_idx].peak_freq = rf

    for i in range(len(fams)):
        _sort_roll_ids_family(fams[i].roll_ids, rolls)

    if spatial_mode == "random_perm":
        var ord = List[Int](capacity=len(fams))
        for i in range(len(fams)):
            ord.append(i)
        rng.shuffle_ints(ord)
        var shuffled = List[_ProductFamily](capacity=len(fams))
        for idx in ord:
            shuffled.append(fams[idx].copy())
        return shuffled^
    elif spatial_mode == "category_zones":
        sort(fams, _cmp_fam_category)
    elif spatial_mode == "vertical_column":
        sort(fams, _cmp_fam_vertical)
    elif spatial_mode == "bark_sticker_kit":
        sort(fams, _cmp_fam_kit)
    else:
        sort(fams, _cmp_fam_center)
    return fams^


def _pick_first_valid_slot(
    roll: RollItem,
    candidate_indices: List[Int],
    slots: List[PhysicalSlot],
    rolls: List[RollItem],
    prefer_empty: Bool = True,
) -> Int:
    var valid = List[Int]()
    for s_idx in candidate_indices:
        if slots[s_idx].can_physically_accept(roll, rolls):
            valid.append(s_idx)
    if len(valid) == 0:
        return -1
    if prefer_empty:
        for s_idx in valid:
            if len(slots[s_idx].occupant_roll_ids) == 0:
                return s_idx
    var best = valid[0]
    var best_v = len(slots[best].assigned_variant_ids)
    var best_o = len(slots[best].occupant_roll_ids)
    for i in range(1, len(valid)):
        var s_idx = valid[i]
        var cv = len(slots[s_idx].assigned_variant_ids)
        var co = len(slots[s_idx].occupant_roll_ids)
        if cv < best_v or (cv == best_v and co < best_o):
            best = s_idx
            best_v = cv
            best_o = co
    return best


def _find_best_slot_in_mechs(
    roll: RollItem,
    mech_order: List[Int],
    slots_by_mech: List[List[Int]],
    mut slots: List[PhysicalSlot],
    rolls: List[RollItem],
    prefer_same_variant: Bool = True,
) -> Int:
    for m_id in mech_order:
        var m_slots = slots_by_mech[m_id].copy()
        if prefer_same_variant:
            for s_idx in m_slots:
                if (
                    _list_contains_int(slots[s_idx].assigned_variant_ids, roll.variant_id)
                    and slots[s_idx].can_physically_accept(roll, rolls)
                ):
                    return s_idx
        var valid = List[Int]()
        for s_idx in m_slots:
            if slots[s_idx].can_physically_accept(roll, rolls):
                valid.append(s_idx)
        if len(valid) > 0:
            var best = valid[0]
            var best_v = len(slots[best].assigned_variant_ids)
            var best_o = len(slots[best].occupant_roll_ids)
            for i in range(1, len(valid)):
                var s_idx = valid[i]
                var cv = len(slots[s_idx].assigned_variant_ids)
                var co = len(slots[s_idx].occupant_roll_ids)
                if cv < best_v or (cv == best_v and co < best_o):
                    best = s_idx
                    best_v = cv
                    best_o = co
            return best

    for i in range(len(slots)):
        if slots[i].supports_coreless or roll.has_core:
            return i
    return -1


def allocate_home_slots(
    scenario: ExperimentScenario,
    variants: List[RollVariantSKU],
    mut rolls: List[RollItem],
) raises -> List[PhysicalSlot]:
    """Assign initial home slots for all 128 rolls across a modular layout."""
    var slots = create_shelf_slots(scenario)
    var rng = Rng(scenario.layout_seed + 137)
    var ordered_slots = _order_slot_indices_for_mode(slots, scenario.spatial_order_mode, rng)
    var families = _order_product_families(rolls, variants, scenario.spatial_order_mode, rng)

    var drawer_slots = List[Int]()
    var peg_slots = List[Int]()
    var bin_slots = List[Int]()
    for s_idx in ordered_slots:
        if slots[s_idx].is_drawer():
            drawer_slots.append(s_idx)
        elif slots[s_idx].is_peg():
            peg_slots.append(s_idx)
        elif slots[s_idx].is_wire_bin():
            bin_slots.append(s_idx)

    # Drawer groups preserving first-seen group order, sorted within each group by sub_idx
    var group_keys = List[Int]()
    var drawer_groups = List[List[Int]]()
    for s_idx in drawer_slots:
        var s = slots[s_idx].copy()
        var key = (s.tier_index * 10 + s.mech_id) * 1000 + s.group_idx
        var g_pos = -1
        for gi in range(len(group_keys)):
            if group_keys[gi] == key:
                g_pos = gi
                break
        if g_pos < 0:
            g_pos = len(group_keys)
            group_keys.append(key)
            drawer_groups.append(List[Int]())
        drawer_groups[g_pos].append(s_idx)

    for gi in range(len(drawer_groups)):
        var gl = drawer_groups[gi].copy()
        var n_g = len(gl)
        for i in range(1, n_g):
            var k = gl[i]
            var ks = slots[k].sub_idx
            var j = i - 1
            while j >= 0 and slots[gl[j]].sub_idx > ks:
                gl[j + 1] = gl[j]
                j -= 1
            gl[j + 1] = k
        drawer_groups[gi] = gl^

    # non_peg_by_tier, reserve_slots, all_by_tier using stable insertion sort
    var non_peg_by_tier = List[Int]()
    for s_idx in drawer_slots:
        non_peg_by_tier.append(s_idx)
    for s_idx in bin_slots:
        non_peg_by_tier.append(s_idx)
    for i in range(1, len(non_peg_by_tier)):
        var k = non_peg_by_tier[i]
        var kt = slots[k].tier_index
        var km = 0 if slots[k].is_drawer() else 1
        var j = i - 1
        while j >= 0:
            var jt = slots[non_peg_by_tier[j]].tier_index
            var jm = 0 if slots[non_peg_by_tier[j]].is_drawer() else 1
            if kt < jt or (kt == jt and km < jm):
                non_peg_by_tier[j + 1] = non_peg_by_tier[j]
                j -= 1
            else:
                break
        non_peg_by_tier[j + 1] = k

    var reserve_slots = List[Int]()
    for s_idx in bin_slots:
        reserve_slots.append(s_idx)
    for s_idx in drawer_slots:
        if slots[s_idx].tier_index >= 2:
            reserve_slots.append(s_idx)

    def res_less(a_idx: Int, b_idx: Int, slts: List[PhysicalSlot]) -> Bool:
        var sa = slts[a_idx].copy()
        var sb = slts[b_idx].copy()
        var ka0 = 0 if sa.is_wire_bin() else 1
        var kb0 = 0 if sb.is_wire_bin() else 1
        if ka0 != kb0:
            return ka0 < kb0
        var ka1 = 0 if sa.tier_index == 2 else 1
        var kb1 = 0 if sb.tier_index == 2 else 1
        if ka1 != kb1:
            return ka1 < kb1
        return (-sa.tier_index) < (-sb.tier_index)

    for i in range(1, len(reserve_slots)):
        var k = reserve_slots[i]
        var j = i - 1
        while j >= 0 and res_less(k, reserve_slots[j], slots):
            reserve_slots[j + 1] = reserve_slots[j]
            j -= 1
        reserve_slots[j + 1] = k

    var all_by_tier = ordered_slots.copy()
    for i in range(1, len(all_by_tier)):
        var k = all_by_tier[i]
        var kt = slots[k].tier_index
        var j = i - 1
        while j >= 0 and slots[all_by_tier[j]].tier_index > kt:
            all_by_tier[j + 1] = all_by_tier[j]
            j -= 1
        all_by_tier[j + 1] = k

    var peg_row_keys = List[Int]()
    var peg_rows_ordered = List[List[Int]]()
    for s_idx in peg_slots:
        var pkey = (slots[s_idx].tier_index * 10 + slots[s_idx].mech_id) * 1000 + slots[s_idx].group_idx
        var p_pos = -1
        for pi in range(len(peg_row_keys)):
            if peg_row_keys[pi] == pkey:
                p_pos = pi
                break
        if p_pos < 0:
            p_pos = len(peg_row_keys)
            peg_row_keys.append(pkey)
            peg_rows_ordered.append(List[Int]())
        peg_rows_ordered[p_pos].append(s_idx)

    for pi in range(len(peg_rows_ordered)):
        var pl = peg_rows_ordered[pi].copy()
        for i in range(1, len(pl)):
            var k = pl[i]
            var ks = slots[k].sub_idx
            var j = i - 1
            while j >= 0 and slots[pl[j]].sub_idx > ks:
                pl[j + 1] = pl[j]
                j -= 1
            pl[j + 1] = k
        peg_rows_ordered[pi] = pl^

    var abc_mode = scenario.abc_colocation_mode
    var ov_mode = scenario.overflow_mode

    var active_fams = List[_ProductFamily]()
    var overflow_rids = List[Int]()
    for fam in families:
        var act_ids = List[Int]()
        for rid in fam.roll_ids:
            var is_ovf = False
            if ov_mode != "all_colocated":
                var r = rolls[rid].copy()
                is_ovf = (r.copy_index >= 2) or ((not r.is_rush_priority) and r.copy_index >= 1)
            if is_ovf:
                overflow_rids.append(rid)
            else:
                act_ids.append(rid)
        if len(act_ids) > 0:
            var nf = fam.copy()
            nf.roll_ids = act_ids^
            active_fams.append(nf^)

    def peg_rem(s_idx: Int, slts: List[PhysicalSlot], rls: List[RollItem]) -> Int:
        var occ = slts[s_idx].occupant_roll_ids.copy()
        if len(occ) > 0:
            for rid in occ:
                if rls[rid].roll_height_cm > 10.0:
                    return 0
        return slts[s_idx].max_rolls_hard - len(occ)

    def assign_peg_family(
        act_items: List[Int],
        peg_slots_arg: List[Int],
        peg_rows_arg: List[List[Int]],
        mut slts: List[PhysicalSlot],
        mut rls: List[RollItem],
    ) -> Bool:
        if len(peg_slots_arg) == 0:
            return False
        for rid in act_items:
            if not rls[rid].has_core:
                return False
        var tot_cap = 0
        for s_idx in peg_slots_arg:
            tot_cap += peg_rem(s_idx, slts, rls)
        if tot_cap < len(act_items):
            return False

        var chosen_row = List[Int]()
        var found_row = False
        for pr in peg_rows_arg:
            var empty_pegs = List[Int]()
            for s_idx in pr:
                if len(slts[s_idx].occupant_roll_ids) == 0:
                    empty_pegs.append(s_idx)
            if len(empty_pegs) >= len(act_items):
                chosen_row = empty_pegs^
                found_row = True
                break
        if not found_row:
            for pr in peg_rows_arg:
                var row_cap = 0
                for s_idx in pr:
                    row_cap += peg_rem(s_idx, slts, rls)
                if row_cap >= len(act_items):
                    chosen_row = pr.copy()
                    found_row = True
                    break

        var pool_seq = List[Int]()
        if found_row:
            for s_idx in chosen_row:
                pool_seq.append(s_idx)
        for s_idx in peg_slots_arg:
            pool_seq.append(s_idx)

        var ordered_items = act_items.copy()
        for i in range(1, len(ordered_items)):
            var k = ordered_items[i]
            var rk = rls[k].copy()
            var k_tall = 0 if rk.roll_height_cm > 10.0 else 1
            var j = i - 1
            while j >= 0:
                var rj = rls[ordered_items[j]].copy()
                var j_tall = 0 if rj.roll_height_cm > 10.0 else 1
                var less = (
                    k_tall < j_tall
                    or (k_tall == j_tall and rk.variant_code < rj.variant_code)
                    or (k_tall == j_tall and rk.variant_code == rj.variant_code and rk.copy_index < rj.copy_index)
                )
                if less:
                    ordered_items[j + 1] = ordered_items[j]
                    j -= 1
                else:
                    break
            ordered_items[j + 1] = k

        var placed_rids = List[Int]()
        var placed_sids = List[Int]()
        for rid in ordered_items:
            var t_idx = _pick_first_valid_slot(rls[rid], pool_seq, slts, rls, True)
            if t_idx < 0:
                for pi in range(len(placed_rids)):
                    var pr_id = placed_rids[pi]
                    var ps_idx = placed_sids[pi]
                    _remove_int_first(slts[ps_idx].occupant_roll_ids, pr_id)
                    slts[ps_idx].assigned_variant_ids = List[Int]()
                    for oid in slts[ps_idx].occupant_roll_ids:
                        var vid = rls[oid].variant_id
                        if not _list_contains_int(slts[ps_idx].assigned_variant_ids, vid):
                            slts[ps_idx].assigned_variant_ids.append(vid)
                    rls[pr_id].current_slot = ""
                    rls[pr_id].current_slot_idx = -1
                    rls[pr_id].home_slot = ""
                    rls[pr_id].home_slot_idx = -1
                    rls[pr_id].current_mechanism = ""
                    rls[pr_id].current_mech_id = -1
                    rls[pr_id].home_mechanism = ""
                    rls[pr_id].home_mech_id = -1
                return False
            _assign_roll_to_slot(rls[rid], slts[t_idx], True)
            placed_rids.append(rid)
            placed_sids.append(t_idx)
        return True

    def split_drawer_vs_peg(
        act_list: List[_ProductFamily],
        peg_slots_arg: List[Int],
        drawer_slots_arg: List[Int],
        bin_slots_arg: List[Int],
        slts: List[PhysicalSlot],
        rls: List[RollItem],
    ) -> Tuple[List[_ProductFamily], List[_ProductFamily]]:
        if len(peg_slots_arg) == 0:
            return (act_list.copy(), List[_ProductFamily]())
        var n_pegs = len(peg_slots_arg)
        var peg_hard_budget = Float64(n_pegs)
        var mid_drw = 0
        var btm_drw = 0
        for s_idx in drawer_slots_arg:
            if slts[s_idx].tier_index == 1:
                mid_drw += 1
            elif slts[s_idx].tier_index >= 2:
                btm_drw += 1
        var eff_btm_relief = btm_drw if len(bin_slots_arg) > 0 else max(0, btm_drw - 21)
        var peg_budget = (
            max(0.5 * Float64(n_pegs), peg_hard_budget - 0.5 * Float64(eff_btm_relief))
            if mid_drw >= 22
            else peg_hard_budget
        )
        var drw_fams = List[_ProductFamily]()
        var peg_fams = List[_ProductFamily]()
        var peg_used: Float64 = 0.0
        for fam in act_list:
            var is_prio = False
            var all_cored = True
            var has_bulk = False
            var peg_cost: Float64 = 0.0
            for rid in fam.roll_ids:
                var r = rls[rid].copy()
                if r.is_rush_priority:
                    is_prio = True
                if not r.has_core:
                    all_cored = False
                if r.copy_index >= 1:
                    has_bulk = True
                peg_cost += 1.0 if r.roll_height_cm > 10.0 else 0.5
            if (not is_prio) and all_cored and (not has_bulk) and (peg_used + peg_cost <= peg_budget):
                peg_fams.append(fam.copy())
                peg_used += peg_cost
            else:
                drw_fams.append(fam.copy())
        return (drw_fams^, peg_fams^)

    var fb_pool = non_peg_by_tier.copy()
    for s_idx in all_by_tier:
        fb_pool.append(s_idx)

    if abc_mode == "same_slot":
        var split_res = split_drawer_vs_peg(active_fams, peg_slots, drawer_slots, bin_slots, slots, rolls)
        var drw_fams = split_res[0].copy()
        var peg_fams = split_res[1].copy()

        for fam in drw_fams:
            var by_var = fam.roll_ids.copy()
            for i in range(1, len(by_var)):
                var k = by_var[i]
                var rk = rolls[k].copy()
                var j = i - 1
                while j >= 0:
                    var rj = rolls[by_var[j]].copy()
                    if rk.variant_code < rj.variant_code or (rk.variant_code == rj.variant_code and rk.copy_index < rj.copy_index):
                        by_var[j + 1] = by_var[j]
                        j -= 1
                    else:
                        break
                by_var[j + 1] = k

            var empty_non_peg = List[Int]()
            for s_idx in non_peg_by_tier:
                if len(slots[s_idx].occupant_roll_ids) == 0:
                    empty_non_peg.append(s_idx)

            var has_coreless = False
            var has_tall = False
            var add_h: Float64 = 0.0
            for rid in by_var:
                if not rolls[rid].has_core:
                    has_coreless = True
                if rolls[rid].roll_height_cm > 10.0:
                    has_tall = True
                add_h += rolls[rid].roll_height_cm

            var target_idx = -1
            var pools = [empty_non_peg^, non_peg_by_tier.copy(), all_by_tier.copy()]
            for pool_i in range(3):
                for s_idx in pools[pool_i]:
                    if (not slots[s_idx].supports_coreless) and has_coreless:
                        continue
                    if slots[s_idx].is_peg():
                        var n_cur = len(slots[s_idx].occupant_roll_ids)
                        var occ_has_tall = False
                        var used_h: Float64 = 0.0
                        for oid in slots[s_idx].occupant_roll_ids:
                            if rolls[oid].roll_height_cm > 10.0:
                                occ_has_tall = True
                            used_h += rolls[oid].roll_height_cm
                        if (n_cur + len(by_var) > 1) and (has_tall or occ_has_tall):
                            continue
                        if used_h + add_h > slots[s_idx].max_stack_height_cm:
                            continue
                    if len(slots[s_idx].occupant_roll_ids) + len(by_var) <= slots[s_idx].max_rolls_hard:
                        target_idx = s_idx
                        break
                if target_idx >= 0:
                    break

            for rid in by_var:
                if target_idx >= 0 and slots[target_idx].can_physically_accept(rolls[rid], rolls):
                    _assign_roll_to_slot(rolls[rid], slots[target_idx], True)
                else:
                    var fb = _pick_first_valid_slot(rolls[rid], fb_pool, slots, rolls, False)
                    if fb < 0:
                        raise Error(t"No valid slot found for roll {rolls[rid].sku_code}")
                    _assign_roll_to_slot(rolls[rid], slots[fb], True)

        for fam in peg_fams:
            if not assign_peg_family(fam.roll_ids, peg_slots, peg_rows_ordered, slots, rolls):
                for rid in fam.roll_ids:
                    var fb = _pick_first_valid_slot(rolls[rid], fb_pool, slots, rolls, False)
                    if fb < 0:
                        raise Error(t"No valid slot found for roll {rolls[rid].sku_code}")
                    _assign_roll_to_slot(rolls[rid], slots[fb], True)

    elif abc_mode == "adjacent_partition":
        var split_res = split_drawer_vs_peg(active_fams, peg_slots, drawer_slots, bin_slots, slots, rolls)
        var drw_fams = split_res[0].copy()
        var peg_fams = split_res[1].copy()

        for fam in drw_fams:
            var sorted_act = fam.roll_ids.copy()
            for i in range(1, len(sorted_act)):
                var k = sorted_act[i]
                var rk = rolls[k].copy()
                var j = i - 1
                while j >= 0:
                    var rj = rolls[sorted_act[j]].copy()
                    if rk.variant_code < rj.variant_code or (rk.variant_code == rj.variant_code and rk.copy_index < rj.copy_index):
                        sorted_act[j + 1] = sorted_act[j]
                        j -= 1
                    else:
                        break
                sorted_act[j + 1] = k

            var var_ids = List[Int]()
            var var_rolls = List[List[Int]]()
            for rid in sorted_act:
                var vid = rolls[rid].variant_id
                var vpos = -1
                for vi in range(len(var_ids)):
                    if var_ids[vi] == vid:
                        vpos = vi
                        break
                if vpos < 0:
                    vpos = len(var_ids)
                    var_ids.append(vid)
                    var_rolls.append(List[Int]())
                var_rolls[vpos].append(rid)

            var n_vars = len(var_ids)
            var chosen_grp = List[Int]()
            var found_grp = False
            var passes = [n_vars, 1, 0]
            for req in passes:
                for grp in drawer_groups:
                    var empty_p = 0
                    var avail_p = 0
                    var cap_left = 0
                    for s_idx in grp:
                        var occ_n = len(slots[s_idx].occupant_roll_ids)
                        if occ_n == 0:
                            empty_p += 1
                        if occ_n < slots[s_idx].max_rolls_hard:
                            avail_p += 1
                        cap_left += slots[s_idx].max_rolls_hard - occ_n
                    if empty_p >= req and avail_p >= 1 and cap_left >= len(sorted_act):
                        chosen_grp = grp.copy()
                        found_grp = True
                        break
                if found_grp:
                    break

            if found_grp:
                var ord_grp = chosen_grp.copy()
                for i in range(1, len(ord_grp)):
                    var k = ord_grp[i]
                    var ko = 1 if len(slots[k].occupant_roll_ids) > 0 else 0
                    var ks = slots[k].sub_idx
                    var j = i - 1
                    while j >= 0:
                        var jo = 1 if len(slots[ord_grp[j]].occupant_roll_ids) > 0 else 0
                        var js = slots[ord_grp[j]].sub_idx
                        if ko < jo or (ko == jo and ks < js):
                            ord_grp[j + 1] = ord_grp[j]
                            j -= 1
                        else:
                            break
                    ord_grp[j + 1] = k

                var grp_plus_fb = ord_grp.copy()
                for s_idx in fb_pool:
                    grp_plus_fb.append(s_idx)

                for vi in range(n_vars):
                    var p_idx = ord_grp[vi % len(ord_grp)]
                    for rid in var_rolls[vi]:
                        if slots[p_idx].can_physically_accept(rolls[rid], rolls):
                            _assign_roll_to_slot(rolls[rid], slots[p_idx], True)
                        else:
                            var fb = _pick_first_valid_slot(rolls[rid], grp_plus_fb, slots, rolls, True)
                            if fb < 0:
                                raise Error(t"No valid slot found for roll {rolls[rid].sku_code}")
                            _assign_roll_to_slot(rolls[rid], slots[fb], True)
            else:
                for rid in sorted_act:
                    var fb = _pick_first_valid_slot(rolls[rid], fb_pool, slots, rolls, True)
                    if fb < 0:
                        raise Error(t"No valid slot found for roll {rolls[rid].sku_code}")
                    _assign_roll_to_slot(rolls[rid], slots[fb], True)

        for fam in peg_fams:
            if not assign_peg_family(fam.roll_ids, peg_slots, peg_rows_ordered, slots, rolls):
                for rid in fam.roll_ids:
                    var fb = _pick_first_valid_slot(rolls[rid], fb_pool, slots, rolls, True)
                    if fb < 0:
                        raise Error(t"No valid slot found for roll {rolls[rid].sku_code}")
                    _assign_roll_to_slot(rolls[rid], slots[fb], True)

    elif abc_mode == "same_tier_bay":
        for fam in active_fams:
            var short_or_coreless = List[Int]()
            var tall_cored = List[Int]()
            for rid in fam.roll_ids:
                if rolls[rid].roll_height_cm > 11.5 and rolls[rid].has_core:
                    tall_cored.append(rid)
                else:
                    short_or_coreless.append(rid)
            var target_tier = -1

            for rid in short_or_coreless:
                var cands = List[Int]()
                for s_idx in non_peg_by_tier:
                    if target_tier < 0 or slots[s_idx].tier_index == target_tier:
                        cands.append(s_idx)
                for s_idx in all_by_tier:
                    cands.append(s_idx)
                var s_idx = _pick_first_valid_slot(rolls[rid], cands, slots, rolls, True)
                if s_idx < 0:
                    raise Error(t"No valid slot found for roll {rolls[rid].sku_code}")
                if target_tier < 0:
                    target_tier = slots[s_idx].tier_index
                _assign_roll_to_slot(rolls[rid], slots[s_idx], True)

            for rid in tall_cored:
                var cands = List[Int]()
                for s_idx in peg_slots:
                    if target_tier < 0 or slots[s_idx].tier_index == target_tier:
                        cands.append(s_idx)
                for s_idx in non_peg_by_tier:
                    if target_tier < 0 or slots[s_idx].tier_index == target_tier:
                        cands.append(s_idx)
                for s_idx in all_by_tier:
                    cands.append(s_idx)
                var s_idx = _pick_first_valid_slot(rolls[rid], cands, slots, rolls, True)
                if s_idx < 0:
                    raise Error(t"No valid slot found for roll {rolls[rid].sku_code}")
                if target_tier < 0:
                    target_tier = slots[s_idx].tier_index
                _assign_roll_to_slot(rolls[rid], slots[s_idx], True)

    else:
        var sorted_peg_desc = peg_slots.copy()
        for i in range(1, len(sorted_peg_desc)):
            var k = sorted_peg_desc[i]
            var kt = slots[k].tier_index
            var j = i - 1
            while j >= 0 and slots[sorted_peg_desc[j]].tier_index < kt:
                sorted_peg_desc[j + 1] = sorted_peg_desc[j]
                j -= 1
            sorted_peg_desc[j + 1] = k

        var sorted_drw_asc = drawer_slots.copy()
        for i in range(1, len(sorted_drw_asc)):
            var k = sorted_drw_asc[i]
            var kt = slots[k].tier_index
            var j = i - 1
            while j >= 0 and slots[sorted_drw_asc[j]].tier_index > kt:
                sorted_drw_asc[j + 1] = sorted_drw_asc[j]
                j -= 1
            sorted_drw_asc[j + 1] = k

        for fam in active_fams:
            for rid in fam.roll_ids:
                var r = rolls[rid].copy()
                var cands = List[Int]()
                if r.roll_height_cm > 10.0 and r.has_core:
                    for s_idx in sorted_peg_desc:
                        cands.append(s_idx)
                    for s_idx in ordered_slots:
                        if slots[s_idx].tier_index >= 2 and slots[s_idx].supports_coreless:
                            cands.append(s_idx)
                    for s_idx in all_by_tier:
                        cands.append(s_idx)
                elif ov_mode == "all_colocated" and (not r.is_rush_priority):
                    for s_idx in bin_slots:
                        cands.append(s_idx)
                    for s_idx in sorted_drw_asc:
                        cands.append(s_idx)
                    for s_idx in all_by_tier:
                        cands.append(s_idx)
                else:
                    for s_idx in sorted_drw_asc:
                        cands.append(s_idx)
                    for s_idx in non_peg_by_tier:
                        cands.append(s_idx)
                    for s_idx in all_by_tier:
                        cands.append(s_idx)
                var s_idx = _pick_first_valid_slot(r, cands, slots, rolls, True)
                if s_idx < 0:
                    raise Error(t"No valid slot found for roll {r.sku_code}")
                _assign_roll_to_slot(rolls[rid], slots[s_idx], True)

    # Sort overflow_rids by (copy_index, -rush_freq, variant_id)
    for i in range(1, len(overflow_rids)):
        var k = overflow_rids[i]
        var rk = rolls[k].copy()
        var fk = variants[rk.variant_id].rush_freq
        var j = i - 1
        while j >= 0:
            var rj = rolls[overflow_rids[j]].copy()
            var fj = variants[rj.variant_id].rush_freq
            var less = (
                rk.copy_index < rj.copy_index
                or (rk.copy_index == rj.copy_index and fk > fj)
                or (rk.copy_index == rj.copy_index and fk == fj and rk.variant_id < rj.variant_id)
            )
            if less:
                overflow_rids[j + 1] = overflow_rids[j]
                j -= 1
            else:
                break
        overflow_rids[j + 1] = k

    var ovf_fb = reserve_slots.copy()
    for s_idx in all_by_tier:
        ovf_fb.append(s_idx)

    for rid in overflow_rids:
        var r = rolls[rid].copy()
        var placed = False
        for s_idx in reserve_slots:
            if (
                _list_contains_int(slots[s_idx].assigned_variant_ids, r.variant_id)
                and slots[s_idx].can_physically_accept(r, rolls)
            ):
                _assign_roll_to_slot(rolls[rid], slots[s_idx], True)
                placed = True
                break
        if not placed:
            var s_idx = _pick_first_valid_slot(r, ovf_fb, slots, rolls, False)
            if s_idx < 0:
                raise Error(t"No valid overflow slot found for roll {r.sku_code}")
            _assign_roll_to_slot(rolls[rid], slots[s_idx], True)

    return slots^


def compute_layout_metrics(
    rolls: List[RollItem], slots: List[PhysicalSlot]
) -> LayoutMetrics:
    """Compute structural A/B/C co-location and capacity metrics."""
    var prod_names = List[String]()
    var prod_var_rolls = List[List[Int]]()
    for r in rolls:
        if r.copy_index != 0:
            continue
        var p_pos = -1
        for i in range(len(prod_names)):
            if prod_names[i] == r.product_name:
                p_pos = i
                break
        if p_pos < 0:
            p_pos = len(prod_names)
            prod_names.append(r.product_name)
            prod_var_rolls.append(List[Int]())
        # Keep first copy per variant_code
        var seen_v = False
        for rid_exist in prod_var_rolls[p_pos]:
            if rolls[rid_exist].variant_code == r.variant_code:
                seen_v = True
                break
        if not seen_v:
            prod_var_rolls[p_pos].append(r.roll_id)

    var multi_cnt = 0
    var same_drawer_cnt = 0
    var same_tier_cnt = 0
    for vr in prod_var_rolls:
        if len(vr) < 2:
            continue
        multi_cnt += 1
        var first_s = slots[rolls[vr[0]].home_slot_idx].copy()
        var all_same_drw = True
        var all_same_tier = True
        for k in range(1, len(vr)):
            var sk = slots[rolls[vr[k]].home_slot_idx].copy()
            if sk.mech_id != first_s.mech_id or sk.group_idx != first_s.group_idx:
                all_same_drw = False
            if sk.tier_index != first_s.tier_index:
                all_same_tier = False
        if all_same_drw:
            same_drawer_cnt += 1
        if all_same_tier:
            same_tier_cnt += 1

    var active_cnt = 0
    var eye_active = 0
    for r in rolls:
        if r.copy_index == 0:
            active_cnt += 1
            if slots[r.home_slot_idx].tier_index == 1:
                eye_active += 1

    var eye_non_empty = 0
    var eye_fill_sum: Float64 = 0.0
    for s in slots:
        if s.tier_index == 1 and len(s.occupant_roll_ids) > 0:
            eye_non_empty += 1
            var nom = s.nominal_capacity if s.nominal_capacity > 1.0 else 1.0
            eye_fill_sum += Float64(len(s.occupant_roll_ids)) / nom

    var n_multi = multi_cnt if multi_cnt > 1 else 1
    var n_act = active_cnt if active_cnt > 1 else 1
    var n_eye = eye_non_empty if eye_non_empty > 1 else 1
    return LayoutMetrics(
        Float64(same_drawer_cnt) / Float64(n_multi),
        Float64(same_tier_cnt) / Float64(n_multi),
        Float64(eye_active) / Float64(n_act),
        eye_fill_sum / Float64(n_eye),
    )


def _weighted_choice(mut rng: Rng, weights: List[Float64]) -> Int:
    var total: Float64 = 0.0
    for w in weights:
        total += w
    var r = rng.random() * total
    var acc: Float64 = 0.0
    for idx in range(len(weights)):
        acc += weights[idx]
        if r <= acc:
            return idx
    return len(weights) - 1


def _sample_job_variants(
    mut rng: Rng,
    is_rush: Bool,
    freq_calm: List[Float64],
    freq_rush: List[Float64],
    co_matrix: List[Float64],
    n_variants: Int,
    high_assoc_siblings: List[List[Int]],
) -> List[Int]:
    var freqs = freq_rush.copy() if is_rush else freq_calm.copy()
    var r_val = rng.random()
    var target_size = 1 if r_val < 0.25 else (2 if r_val < 0.65 else (3 if r_val < 0.90 else 4))
    var anchor = _weighted_choice(rng, freqs)
    var picked = List[Int]()
    picked.append(anchor)

    for sib in high_assoc_siblings[anchor]:
        if len(picked) >= 4:
            break
        if rng.random() < co_matrix[anchor * n_variants + sib]:
            picked.append(sib)

    var job_size = max(target_size, len(picked))
    if job_size > 4:
        job_size = 4
    while len(picked) < job_size:
        var weights = List[Float64](capacity=n_variants)
        var w_sum: Float64 = 0.0
        for j in range(n_variants):
            if _list_contains_int(picked, j):
                weights.append(0.0)
            else:
                var w = co_matrix[anchor * n_variants + j] * 5.0 + freqs[j]
                weights.append(w)
                w_sum += w
        if w_sum <= 0.0:
            break
        picked.append(_weighted_choice(rng, weights))
    return picked^


def _compute_shannon_entropy(
    rolls: List[RollItem], slots: List[PhysicalSlot], n_variants: Int
) -> Float64:
    var max_zones_log = log2(8.0)
    var total_h: Float64 = 0.0
    var total_w = 0
    var displaced = 0

    # Group roll zone keys per variant
    var v_zones = List[List[Int]](capacity=n_variants)
    var v_counts = List[List[Int]](capacity=n_variants)
    var v_totals = List[Int](capacity=n_variants)
    for _ in range(n_variants):
        v_zones.append(List[Int]())
        v_counts.append(List[Int]())
        v_totals.append(0)

    for r in rolls:
        if r.current_slot_idx != r.home_slot_idx:
            displaced += 1
        var s = slots[r.current_slot_idx].copy()
        var zkey = s.mech_id * 1000 + s.group_idx
        var v = r.variant_id
        v_totals[v] += 1
        var pos = -1
        for k in range(len(v_zones[v])):
            if v_zones[v][k] == zkey:
                pos = k
                break
        if pos < 0:
            v_zones[v].append(zkey)
            v_counts[v].append(1)
        else:
            v_counts[v][pos] += 1

    for v in range(n_variants):
        var m = v_totals[v]
        if m <= 1:
            continue
        var h: Float64 = 0.0
        for c in v_counts[v]:
            var p = Float64(c) / Float64(m)
            if p > 0.0:
                h -= p * log2(p)
        var norm_h = h / max_zones_log
        if norm_h > 1.0:
            norm_h = 1.0
        total_h += Float64(m) * norm_h
        total_w += m

    var displaced_singles = Float64(displaced) / Float64(len(rolls))
    var base_entropy = (total_h / Float64(total_w)) if total_w > 0 else 0.0
    var combined = 0.55 * base_entropy + 0.45 * displaced_singles
    return combined if combined < 1.0 else 1.0


def run_single_replication(
    scenario: ExperimentScenario,
    variants: List[RollVariantSKU],
    base_rolls: List[RollItem],
    freq_calm: List[Float64],
    freq_rush: List[Float64],
    co_matrix: List[Float64],
    n_variants: Int,
    high_assoc_siblings: List[List[Int]],
    specs: List[MechanismSpec],
    seed: Int,
) raises -> ReplicationTrace:
    """Execute one 1,000-cycle Monte Carlo replication."""
    var rng = Rng(seed)
    var beh = scenario.behavior.copy()
    var rolls = base_rolls.copy()
    var slots = allocate_home_slots(scenario, variants, rolls)

    var slots_by_mech = List[List[Int]](capacity=NUM_MECHS)
    for _ in range(NUM_MECHS):
        slots_by_mech.append(List[Int]())
    var all_bin_slots = List[Int]()
    var n_drawer_slots = 0
    for s in slots:
        slots_by_mech[s.mech_id].append(s.slot_idx)
        if s.is_wire_bin():
            all_bin_slots.append(s.slot_idx)
        if s.is_drawer():
            n_drawer_slots += 1

    # Precompute sibling group slot indices for every slot
    var neighbors_of = List[List[Int]](capacity=len(slots))
    for i in range(len(slots)):
        var nb = List[Int]()
        var mi = slots[i].mech_id
        var gi = slots[i].group_idx
        for j in slots_by_mech[mi]:
            if slots[j].group_idx == gi:
                nb.append(j)
        neighbors_of.append(nb^)

    var rolls_by_variant = List[List[Int]](capacity=n_variants)
    for _ in range(n_variants):
        rolls_by_variant.append(List[Int]())
    for r in rolls:
        rolls_by_variant[r.variant_id].append(r.roll_id)

    var cpx = count_shelf_complexity(scenario.row0, scenario.row1, scenario.row2)
    var n_bins = cpx[0]
    var extra_styles = cpx[1]
    var mixed_bin_drw_rows = count_mixed_bin_drawer_rows(scenario.row0, scenario.row1, scenario.row2)
    var size_mix_blur = beh.drawer_size_mix_penalty if has_mixed_drawer_sizes(scenario.row0, scenario.row1, scenario.row2) else 0.0
    var slot_deficit_blur = beh.slot_deficit_penalty * max(0.0, Float64(48 - n_drawer_slots) / 48.0)
    var row_width_blur = beh.row_width_mismatch_penalty if has_row_width_mismatch(scenario.row0, scenario.row1, scenario.row2) else 0.0

    var has_double_bin = False
    for rc in [scenario.row0, scenario.row1, scenario.row2]:
        if (
            rc == "2B"
            or rc == "1D6_2B"
            or rc == "1B_1D6_1B"
            or rc == "2B_1D6"
            or rc == "1D_2B"
            or rc == "1B_1D_1B"
            or rc == "2B_1D"
        ):
            has_double_bin = True

    var eye_mech_default = -1
    if len(slots_by_mech[MECH_SPLIT_DRAWER_EYE]) > 0:
        eye_mech_default = MECH_SPLIT_DRAWER_EYE
    elif len(slots_by_mech[MECH_WIRE_BIN_EYE]) > 0:
        eye_mech_default = MECH_WIRE_BIN_EYE
    else:
        for s in slots:
            if s.tier_index == 1:
                eye_mech_default = s.mech_id
                break
        if eye_mech_default < 0:
            eye_mech_default = slots[0].mech_id

    var unlatched_peg_rows = List[Int]()
    var total_cycles = scenario.rush.total_cycles()
    var shift_cycles = scenario.rush.shift_cycles
    var rush_cutoff = Int(round(Float64(shift_cycles) * scenario.rush.rush_job_share))

    var acc_t = List[Float64](capacity=total_cycles)
    var entropy_t = List[Float64](capacity=total_cycles)
    var contam_t = List[Float64](capacity=total_cycles)
    var eye_dump_t = List[Float64](capacity=total_cycles)
    var latency_t = List[Float64](capacity=total_cycles)
    var t_crit = -1

    var row_cfgs = [scenario.row0, scenario.row1, scenario.row2]
    var n_rolls = len(rolls)

    for t in range(total_cycles):
        var cycle_in_shift = t % shift_cycles
        var is_rush = cycle_in_shift < rush_cutoff
        var tau = (
            rng.uniform(scenario.rush.rush_tau_lo, scenario.rush.rush_tau_hi)
            if is_rush
            else rng.uniform(scenario.rush.calm_tau_lo, scenario.rush.calm_tau_hi)
        )

        if (not is_rush) and len(unlatched_peg_rows) > 0:
            sort(unlatched_peg_rows)
            var kept = List[Int]()
            for row_idx in unlatched_peg_rows:
                if not (rng.random() < 0.70):
                    kept.append(row_idx)
            unlatched_peg_rows = kept^

        var picked_vids = _sample_job_variants(
            rng, is_rush, freq_calm, freq_rush, co_matrix, n_variants, high_assoc_siblings
        )

        var ret_rids = List[Int]()
        var ret_was_home = List[Bool]()
        var ret_origin_slot = List[Int]()
        var opened_groups = List[Int]()
        var job_latency: Float64 = 0.0

        for vid in picked_vids:
            var cands = rolls_by_variant[vid].copy()
            var chosen_rid = -1
            var best_t = 9999
            var best_ci = 9999
            for rid in cands:
                if rolls[rid].current_slot_idx == rolls[rid].home_slot_idx:
                    var ht = slots[rolls[rid].home_slot_idx].tier_index
                    var ci = rolls[rid].copy_index
                    if ht < best_t or (ht == best_t and ci < best_ci):
                        best_t = ht
                        best_ci = ci
                        chosen_rid = rid

            var was_home: Bool
            if chosen_rid >= 0:
                was_home = True
            else:
                chosen_rid = cands[rng.randint(len(cands))]
                was_home = False

            var origin_idx = rolls[chosen_rid].current_slot_idx
            var origin_mech = rolls[chosen_rid].current_mech_id
            var spec = specs[origin_mech].copy()
            var slot_obj = slots[origin_idx].copy()
            var grp_key = origin_mech * 1000 + slot_obj.group_idx
            var drawer_already_open = (
                (origin_mech == MECH_SPLIT_DRAWER_EYE or origin_mech == MECH_SPLIT_DRAWER_MID)
                and _list_contains_int(opened_groups, grp_key)
            )
            if not _list_contains_int(opened_groups, grp_key):
                opened_groups.append(grp_key)

            var edge_factor = abs(slot_obj.x_center_cm - 60.0) / 60.0
            var t_step = spec.base_retrieval_sec * (1.0 + 0.15 * (1.0 - tau)) + 0.35 * edge_factor
            if len(slot_obj.assigned_variant_ids) > 1:
                t_step += 0.25 * Float64(len(slot_obj.assigned_variant_ids) - 1)
            if drawer_already_open:
                t_step = max(0.8, t_step - 1.35)
            if spec.mech_id == MECH_PEGBOARD_OVH and len(slot_obj.occupant_roll_ids) >= 2:
                t_step += 1.45 * Float64(len(slot_obj.occupant_roll_ids) - 1)
            if slot_obj.is_wire_bin() and has_double_bin:
                t_step += 0.45
            if extra_styles > 0:
                t_step += 0.18 * Float64(extra_styles)
            if slot_obj.tier_index == 1 or slot_obj.tier_index == 2:
                var rcfg = row_cfgs[slot_obj.tier_index - 1]
                if "B" in rcfg and "D" in rcfg:
                    t_step += 0.55

            if spec.is_pull_down:
                var heavy = rolls[chosen_rid].is_rush_priority or len(slot_obj.occupant_roll_ids) >= 2
                var unlatch_mult = 1.0 if heavy else 0.10
                if is_rush and rng.random() < beh.unlatch_rush_prob * tau * unlatch_mult:
                    if not _list_contains_int(unlatched_peg_rows, slot_obj.group_idx):
                        unlatched_peg_rows.append(slot_obj.group_idx)
            elif len(unlatched_peg_rows) > 0 and slot_obj.tier_index == 1:
                t_step += 2.2

            if not was_home:
                var rho = slot_obj.fill_ratio(rolls, scenario.target_fill_ratio)
                t_step += beh.search_miss_penalty_sec * (1.0 + 0.6 * rho) * (1.15 - 0.3 * scenario.cue_salience)

            job_latency += t_step
            ret_rids.append(chosen_rid)
            ret_was_home.append(was_home)
            ret_origin_slot.append(origin_idx)

        # Active cleanup when drawers/pegs are opened
        sort(opened_groups)
        for grp_key in opened_groups:
            var m_id = grp_key // 1000
            var g_idx = grp_key % 1000
            var v_active = specs[m_id].visibility_open
            var p_fix = beh.base_correction_rate * v_active * scenario.cue_salience * (1.0 - 0.7 * tau)
            for s_idx in slots_by_mech[m_id]:
                if slots[s_idx].group_idx != g_idx:
                    continue
                var occ_snap = slots[s_idx].occupant_roll_ids.copy()
                for rid in occ_snap:
                    if rolls[rid].current_slot_idx != rolls[rid].home_slot_idx and rng.random() < p_fix:
                        var h_idx = rolls[rid].home_slot_idx
                        if slots[h_idx].can_physically_accept(rolls[rid], rolls):
                            _move_roll(rolls[rid], slots, h_idx)

        # Calm-period passive visual scan
        if (not is_rush) and scenario.cue_salience >= 0.50:
            for s_idx in range(len(slots)):
                var v_closed = specs[slots[s_idx].mech_id].visibility_closed
                if v_closed < 0.25:
                    continue
                var p_passive = 0.025 * v_closed * scenario.cue_salience
                var occ_snap = slots[s_idx].occupant_roll_ids.copy()
                for rid in occ_snap:
                    if rolls[rid].current_slot_idx != rolls[rid].home_slot_idx and rng.random() < p_passive:
                        var h_idx = rolls[rid].home_slot_idx
                        if slots[h_idx].can_physically_accept(rolls[rid], rolls):
                            _move_roll(rolls[rid], slots, h_idx)

        # Return phase
        var dist_tiers = List[Int]()
        var dist_mechs = List[Int]()
        var dist_drawers = List[Int]()
        var companion_slots = List[Int]()
        for rid in ret_rids:
            var h_idx = rolls[rid].home_slot_idx
            var hs = slots[h_idx].copy()
            if not _list_contains_int(dist_tiers, hs.tier_index):
                dist_tiers.append(hs.tier_index)
            if not _list_contains_int(dist_mechs, hs.mech_id):
                dist_mechs.append(hs.mech_id)
            var d_key = hs.mech_id * 1000 + hs.group_idx
            if not _list_contains_int(dist_drawers, d_key):
                dist_drawers.append(d_key)
            companion_slots.append(h_idx)

        var n_tiers = len(dist_tiers)
        var n_mechs = len(dist_mechs)
        var n_drawers = len(dist_drawers)
        var returned_groups = List[Int]()

        for ri in range(len(ret_rids)):
            var rid = ret_rids[ri]
            var was_home = ret_was_home[ri]
            var origin_idx = ret_origin_slot[ri]
            var home_idx = rolls[rid].home_slot_idx
            var home_mech = rolls[rid].home_mech_id
            var home_spec = specs[home_mech].copy()
            var home_slot = slots[home_idx].copy()
            var home_group = home_mech * 1000 + home_slot.group_idx

            var edge_factor_ret = abs(home_slot.x_center_cm - 60.0) / 60.0
            var t_ret = home_spec.base_return_sec + 0.25 * edge_factor_ret
            if home_slot.is_drawer() and _list_contains_int(returned_groups, home_group):
                t_ret = max(0.7, t_ret - 1.05)
            if home_mech == MECH_PEGBOARD_OVH and len(home_slot.occupant_roll_ids) >= 2:
                t_ret += 1.15 * Float64(len(home_slot.occupant_roll_ids) - 1)
            if not _list_contains_int(returned_groups, home_group):
                returned_groups.append(home_group)
            job_latency += t_ret

            if not was_home:
                var p_id = min(0.98, max(0.08, scenario.cue_salience * (1.0 - 0.55 * (tau * tau))))
                if rng.random() > p_id:
                    var t_sl_idx: Int
                    if rng.random() < 0.65:
                        t_sl_idx = origin_idx
                    else:
                        t_sl_idx = _find_best_slot_in_mechs(
                            rolls[rid], [eye_mech_default], slots_by_mech, slots, rolls, False
                        )
                    if t_sl_idx >= 0 and slots[t_sl_idx].can_physically_accept(rolls[rid], rolls):
                        _move_roll(rolls[rid], slots, t_sl_idx)
                        continue

            var rho_slot = home_slot.fill_ratio(rolls, scenario.target_fill_ratio)
            var tot_n = 0
            var mis_n = 0
            for n_idx in neighbors_of[home_idx]:
                for oid in slots[n_idx].occupant_roll_ids:
                    tot_n += 1
                    if rolls[oid].current_slot_idx != rolls[oid].home_slot_idx:
                        mis_n += 1
            var h_local = (Float64(mis_n) / Float64(tot_n)) if tot_n > 0 else 0.0

            var phi_rush = 1.0 + beh.alpha_rush * (tau * tau)
            var occ_pen = beh.occlusion_friction_penalty if (len(unlatched_peg_rows) > 0 and home_slot.tier_index == 1) else 0.0
            var cross_tier_factor = 1.0 + 0.42 * Float64(max(0, n_tiers - 1))

            var co_loc_bonus: Float64 = 0.0
            if len(ret_rids) > 1 and n_drawers == 1:
                co_loc_bonus = -0.22
            elif len(ret_rids) > 1:
                for r2_id in ret_rids:
                    if r2_id != rid:
                        var s2 = slots[rolls[r2_id].home_slot_idx].copy()
                        if (s2.mech_id * 1000 + s2.group_idx) == home_group:
                            co_loc_bonus = -0.14
                            break
            var spread_pen = 0.12 if (n_tiers == 1 and n_mechs > 1) else 0.0
            var edge_ergo = 0.08 * edge_factor_ret

            var phi_ergo = max(
                0.55,
                1.0
                + beh.alpha_ergo * home_spec.reach_actuation_cost * cross_tier_factor
                + occ_pen
                + spread_pen
                + co_loc_bonus
                + edge_ergo,
            )
            var crowd_term = max(0.0, (rho_slot - 0.75) / (1.05 - rho_slot))
            var peg_bulk_pen = (
                0.45 * Float64(max(0, len(home_slot.occupant_roll_ids) - 1))
                if home_mech == MECH_PEGBOARD_OVH
                else 0.0
            )
            var phi_crowd = 1.0 + beh.alpha_crowd * crowd_term + peg_bulk_pen
            var part_blur = (
                (0.18 if home_slot.sub_idx == 0 else 0.08)
                if (home_slot.is_drawer() and home_slot.partition_count == 3)
                else 0.0
            )
            var same_slot_mix = beh.multi_sku_slot_penalty * Float64(max(0, len(home_slot.assigned_variant_ids) - 1))
            var bin_dump_blur = beh.bin_dump_error_penalty * Float64(n_bins) + (0.06 if home_slot.is_wire_bin() else 0.0)
            var cog_load_blur = (
                beh.mech_diversity_penalty * Float64(extra_styles)
                + beh.bin_drawer_same_row_penalty * Float64(mixed_bin_drw_rows)
                + size_mix_blur
                + slot_deficit_blur
                + row_width_blur
            )
            var phi_blur = max(
                0.65,
                1.0
                + beh.alpha_blur
                * max(
                    0.0,
                    h_local
                    + part_blur
                    + same_slot_mix
                    + bin_dump_blur
                    + cog_load_blur
                    + (1.0 - scenario.cue_salience) * 0.25
                    + co_loc_bonus,
                ),
            )
            var p_err = min(0.95, max(0.005, beh.base_error_rate * phi_rush * phi_ergo * phi_crowd * phi_blur))

            if rng.random() < p_err:
                var part_w = (0.25 if home_slot.sub_idx == 0 else 0.12) if home_slot.partition_count == 3 else 0.0
                var w_adj = 0.42 + part_w + 0.15 * h_local
                var w_eye = 0.45 * home_spec.reach_actuation_cost + (0.20 if home_spec.is_pull_down else 0.0)
                var w_comp = 0.22 if len(companion_slots) > 1 else 0.02
                var w_bin = (beh.bin_dump_routing_weight * Float64(n_bins)) if len(all_bin_slots) > 0 else 0.0
                var mode = _weighted_choice(rng, [w_adj, w_eye, w_comp, w_bin])

                var cand_slots = List[Int]()
                if mode == 0:
                    for s_idx in neighbors_of[home_idx]:
                        if s_idx != home_idx and slots[s_idx].can_physically_accept(rolls[rid], rolls):
                            cand_slots.append(s_idx)
                    if len(cand_slots) == 0:
                        for s_idx in slots_by_mech[home_mech]:
                            if s_idx != home_idx and slots[s_idx].can_physically_accept(rolls[rid], rolls):
                                cand_slots.append(s_idx)
                elif mode == 1:
                    var dump_mech = eye_mech_default
                    for s_idx in slots_by_mech[dump_mech]:
                        if s_idx != home_idx and slots[s_idx].can_physically_accept(rolls[rid], rolls):
                            cand_slots.append(s_idx)
                    if len(cand_slots) == 0:
                        for s_idx in range(len(slots)):
                            if (
                                (slots[s_idx].tier_index == 1 or slots[s_idx].tier_index == 2)
                                and slots[s_idx].supports_coreless
                                and s_idx != home_idx
                                and slots[s_idx].can_physically_accept(rolls[rid], rolls)
                            ):
                                cand_slots.append(s_idx)
                elif mode == 2:
                    var comp_cands = List[Int]()
                    for cs_idx in companion_slots:
                        if cs_idx != home_idx:
                            comp_cands.append(cs_idx)
                    if len(comp_cands) > 0:
                        var chosen_cs = comp_cands[rng.randint(len(comp_cands))]
                        for s_idx in neighbors_of[chosen_cs]:
                            if slots[s_idx].can_physically_accept(rolls[rid], rolls):
                                cand_slots.append(s_idx)
                else:
                    for s_idx in all_bin_slots:
                        if s_idx != home_idx and slots[s_idx].can_physically_accept(rolls[rid], rolls):
                            cand_slots.append(s_idx)

                if len(cand_slots) > 0:
                    var chosen = cand_slots[rng.randint(len(cand_slots))]
                    _move_roll(rolls[rid], slots, chosen)
                    rolls[rid].misplaced_history_count += 1
                    continue

            if slots[home_idx].can_physically_accept(rolls[rid], rolls) or rolls[rid].current_slot_idx == home_idx:
                if rolls[rid].current_slot_idx != home_idx:
                    _move_roll(rolls[rid], slots, home_idx)
            else:
                var spill = List[Int]()
                for s_idx in neighbors_of[home_idx]:
                    if slots[s_idx].can_physically_accept(rolls[rid], rolls):
                        spill.append(s_idx)
                if len(spill) == 0:
                    for s_idx in slots_by_mech[eye_mech_default]:
                        if slots[s_idx].can_physically_accept(rolls[rid], rolls):
                            spill.append(s_idx)
                if len(spill) == 0:
                    for s_idx in range(len(slots)):
                        if slots[s_idx].supports_coreless and slots[s_idx].can_physically_accept(rolls[rid], rolls):
                            spill.append(s_idx)
                if len(spill) > 0:
                    var chosen_sp = spill[rng.randint(len(spill))]
                    _move_roll(rolls[rid], slots, chosen_sp)

        var home_cnt = 0
        var contam_cnt = 0
        var eye_dump_cnt = 0
        for r in rolls:
            if r.current_slot_idx == r.home_slot_idx:
                home_cnt += 1
            if r.current_mech_id != r.home_mech_id:
                contam_cnt += 1
                if slots[r.current_slot_idx].tier_index == 1:
                    eye_dump_cnt += 1

        var acc = Float64(home_cnt) / Float64(n_rolls)
        var ent = _compute_shannon_entropy(rolls, slots, n_variants)
        var contam = Float64(contam_cnt) / Float64(n_rolls)
        var eye_dump = Float64(eye_dump_cnt) / Float64(n_rolls)

        acc_t.append(acc)
        entropy_t.append(ent)
        contam_t.append(contam)
        eye_dump_t.append(eye_dump)
        latency_t.append(job_latency)

        if t_crit < 0 and len(acc_t) >= 10:
            var w10: Float64 = 0.0
            var l_acc = len(acc_t)
            for k in range(l_acc - 10, l_acc):
                w10 += acc_t[k]
            if (w10 / 10.0) < beh.usability_accuracy_threshold:
                t_crit = t + 1

    var per_mech_acc = List[Float64](capacity=NUM_MECHS)
    for m_id in range(NUM_MECHS):
        var tot_m = 0
        var hom_m = 0
        for r in rolls:
            if r.home_mech_id == m_id:
                tot_m += 1
                if r.current_slot_idx == r.home_slot_idx:
                    hom_m += 1
        per_mech_acc.append((Float64(hom_m) / Float64(tot_m)) if tot_m > 0 else 1.0)

    var slot_occ = List[Int](capacity=len(slots))
    var slot_mis = List[Int](capacity=len(slots))
    for s in slots:
        slot_occ.append(len(s.occupant_roll_ids))
        var mc = 0
        for rid in s.occupant_roll_ids:
            if rolls[rid].home_slot_idx != s.slot_idx:
                mc += 1
        slot_mis.append(mc)

    return ReplicationTrace(
        acc_t^,
        entropy_t^,
        contam_t^,
        eye_dump_t^,
        latency_t^,
        per_mech_acc^,
        slot_mis^,
        slot_occ^,
        t_crit,
    )


def run_experiment_monte_carlo(
    scenario: ExperimentScenario,
    variants: List[RollVariantSKU],
    base_rolls: List[RollItem],
    freq_calm: List[Float64],
    freq_rush: List[Float64],
    co_matrix: List[Float64],
    n_variants: Int,
    high_assoc_siblings: List[List[Int]],
    specs: List[MechanismSpec],
    n_replications: Int = 50,
    base_seed: Int = 1000,
) raises -> ExperimentSummary:
    """Run N Monte Carlo replications for a scenario and aggregate KPIs."""
    var total_cycles = scenario.rush.total_cycles()
    var shift_cycles = scenario.rush.shift_cycles
    var rush_cutoff = Int(round(Float64(shift_cycles) * scenario.rush.rush_job_share))

    var traces = List[ReplicationTrace](capacity=n_replications)
    for rep in range(n_replications):
        traces.append(
            run_single_replication(
                scenario,
                variants,
                base_rolls,
                freq_calm,
                freq_rush,
                co_matrix,
                n_variants,
                high_assoc_siblings,
                specs,
                base_seed + rep * 17,
            )
        )

    var n_rep_f = Float64(n_replications)
    var acc_mean = List[Float64](capacity=total_cycles)
    var acc_std = List[Float64](capacity=total_cycles)
    var entropy_mean = List[Float64](capacity=total_cycles)
    var contam_mean = List[Float64](capacity=total_cycles)
    var eye_dump_mean = List[Float64](capacity=total_cycles)
    var latency_mean = List[Float64](capacity=total_cycles)
    var latency_std = List[Float64](capacity=total_cycles)

    for t in range(total_cycles):
        var sum_a: Float64 = 0.0
        var sum_e: Float64 = 0.0
        var sum_c: Float64 = 0.0
        var sum_ed: Float64 = 0.0
        var sum_l: Float64 = 0.0
        for rep in range(n_replications):
            sum_a += traces[rep].accuracy_t[t]
            sum_e += traces[rep].entropy_t[t]
            sum_c += traces[rep].cross_tier_contam_t[t]
            sum_ed += traces[rep].eye_drawer_dump_share_t[t]
            sum_l += traces[rep].latency_t[t]
        var ma = sum_a / n_rep_f
        var ml = sum_l / n_rep_f
        var va: Float64 = 0.0
        var vl: Float64 = 0.0
        for rep in range(n_replications):
            var da = traces[rep].accuracy_t[t] - ma
            var dl = traces[rep].latency_t[t] - ml
            va += da * da
            vl += dl * dl
        acc_mean.append(ma)
        acc_std.append(sqrt(va / n_rep_f))
        entropy_mean.append(sum_e / n_rep_f)
        contam_mean.append(sum_c / n_rep_f)
        eye_dump_mean.append(sum_ed / n_rep_f)
        latency_mean.append(ml)
        latency_std.append(sqrt(vl / n_rep_f))

    var home_t = acc_mean.copy()
    var within_drift_t = List[Float64](capacity=total_cycles)
    var other_contam_t = List[Float64](capacity=total_cycles)
    var sum_all_lat: Float64 = 0.0
    var sum_rush_lat: Float64 = 0.0
    var cnt_rush_lat = 0
    var sum_calm_lat: Float64 = 0.0
    var cnt_calm_lat = 0

    for t in range(total_cycles):
        within_drift_t.append(max(0.0, 1.0 - acc_mean[t] - contam_mean[t]))
        other_contam_t.append(max(0.0, contam_mean[t] - eye_dump_mean[t]))
        var lt = latency_mean[t]
        sum_all_lat += lt
        if (t % shift_cycles) < rush_cutoff:
            sum_rush_lat += lt
            cnt_rush_lat += 1
        else:
            sum_calm_lat += lt
            cnt_calm_lat += 1

    var collapse_cnt = 0
    var t_crit_sum = 0
    for rep in range(n_replications):
        if traces[rep].t_crit >= 0:
            collapse_cnt += 1
            t_crit_sum += traces[rep].t_crit
        else:
            t_crit_sum += total_cycles

    var per_mech_final = List[Float64](capacity=NUM_MECHS)
    for m_id in range(NUM_MECHS):
        var ms: Float64 = 0.0
        for rep in range(n_replications):
            ms += traces[rep].per_mech_accuracy_final[m_id]
        per_mech_final.append(ms / n_rep_f)

    var sample_rolls = base_rolls.copy()
    var sample_slots = allocate_home_slots(scenario, variants, sample_rolls)
    var heatmap = List[SlotHeatmapEntry](capacity=len(sample_slots))
    for s_idx in range(len(sample_slots)):
        var s = sample_slots[s_idx].copy()
        var m_sum = 0
        var o_sum = 0
        for rep in range(n_replications):
            m_sum += traces[rep].final_slot_misplace_counts[s_idx]
            o_sum += traces[rep].final_slot_occupant_counts[s_idx]
        var mean_mis = Float64(m_sum) / n_rep_f
        var mean_occ = Float64(o_sum) / n_rep_f
        var r_mis = round(mean_mis * 100.0) / 100.0
        var r_occ = round(mean_occ * 100.0) / 100.0
        var denom_occ = mean_occ if mean_occ > 1.0 else 1.0
        var r_dens = round((mean_mis / denom_occ) * 1000.0) / 1000.0
        heatmap.append(
            SlotHeatmapEntry(
                s.slot_id,
                s.mechanism_id,
                s.tier_index,
                s.group_idx,
                s.sub_idx,
                len(s.occupant_roll_ids),
                r_occ,
                r_mis,
                r_dens,
            )
        )

    var last = total_cycles - 1
    return ExperimentSummary(
        scenario.copy(),
        n_replications,
        total_cycles,
        acc_mean.copy(),
        acc_std.copy(),
        entropy_mean.copy(),
        contam_mean.copy(),
        eye_dump_mean.copy(),
        latency_mean.copy(),
        latency_std.copy(),
        acc_mean[last],
        acc_std[last],
        entropy_mean[last],
        contam_mean[last],
        sum_all_lat / Float64(total_cycles),
        sum_rush_lat / Float64(cnt_rush_lat if cnt_rush_lat > 0 else 1),
        sum_calm_lat / Float64(cnt_calm_lat if cnt_calm_lat > 0 else 1),
        Float64(t_crit_sum) / n_rep_f,
        Float64(collapse_cnt) / n_rep_f,
        per_mech_final^,
        heatmap^,
        home_t^,
        within_drift_t^,
        other_contam_t^,
    )
