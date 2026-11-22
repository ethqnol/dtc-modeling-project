"""Agent-based Monte Carlo simulation of the 128-roll wire shelf."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from src.sim_config import (
    DEFAULT_CSV_PATH,
    BehavioralConfig,
    ExperimentScenario,
    MechanismSpec,
    RollItem,
    RollVariantSKU,
    build_mechanism_specs,
    count_mixed_bin_drawer_rows,
    has_row_width_mismatch,
    count_shelf_complexity,
    has_mixed_drawer_sizes,
    load_roll_inventory,
)


@dataclass
class PhysicalSlot:
    """A single physical peg, drawer partition, or wire container slot."""

    slot_id: str
    mechanism_id: str
    tier_index: int
    group_idx: int
    sub_idx: int
    max_rolls_hard: int
    max_stack_height_cm: float
    clear_width_cm: float
    supports_coreless: bool
    is_pull_down: bool
    occupant_roll_ids: List[int] = field(default_factory=list)
    assigned_variant_ids: List[int] = field(default_factory=list)
    nominal_capacity: float = 3.0
    bay_side: int = 0
    x_center_cm: float = 60.0
    partition_count: int = 4
    max_od_cm: float = 999.0

    def can_physically_accept(
        self, roll: RollItem, rolls_by_id: Dict[int, RollItem]
    ) -> bool:
        """Check hard geometric constraints (OD limit, coreless, height stack, hard roll count)."""
        if roll.outer_diameter_cm > self.max_od_cm:
            return False
        if not self.supports_coreless and not roll.has_core:
            return False
        if len(self.occupant_roll_ids) >= self.max_rolls_hard:
            return False
        if "peg" in self.mechanism_id:
            if self.occupant_roll_ids:
                if roll.roll_height_cm > 10.0:
                    return False
                if any(rolls_by_id[rid].roll_height_cm > 10.0 for rid in self.occupant_roll_ids):
                    return False
            used_h = sum(rolls_by_id[rid].roll_height_cm for rid in self.occupant_roll_ids)
            if used_h + roll.roll_height_cm > self.max_stack_height_cm:
                return False
        return True

    def fill_ratio(
        self, rolls_by_id: Dict[int, RollItem], target_fill_ratio: float
    ) -> float:
        """Compute effective crowding ratio rho_slot scaled by target fill ratio."""
        if not self.occupant_roll_ids:
            return 0.0
        scale = target_fill_ratio / 0.80
        if "peg" in self.mechanism_id:
            if len(self.occupant_roll_ids) == 1:
                base_rho = 0.55
            else:
                used_h = sum(rolls_by_id[rid].roll_height_cm for rid in self.occupant_roll_ids)
                base_rho = max(
                    len(self.occupant_roll_ids) / 2.0,
                    used_h / max(1.0, self.max_stack_height_cm),
                )
        else:
            base_rho = len(self.occupant_roll_ids) / max(1.0, self.nominal_capacity)
        return min(1.01, base_rho * scale)


@dataclass
class ReplicationTrace:
    """Per-cycle KPI time series for a single Monte Carlo replication."""

    accuracy_t: List[float]
    entropy_t: List[float]
    cross_tier_contam_t: List[float]
    eye_drawer_dump_share_t: List[float]
    latency_t: List[float]
    per_mech_accuracy_final: Dict[str, float]
    final_slot_misplace_counts: Dict[str, int]
    final_slot_occupant_counts: Dict[str, int]
    t_crit: Optional[int]


@dataclass
class ExperimentSummary:
    """Aggregated statistics across N Monte Carlo replications."""

    scenario: ExperimentScenario
    n_replications: int
    cycles: int
    acc_mean: List[float]
    acc_std: List[float]
    entropy_mean: List[float]
    contam_mean: List[float]
    eye_dump_mean: List[float]
    latency_mean: List[float]
    latency_std: List[float]
    final_acc_mean: float
    final_acc_std: float
    final_entropy_mean: float
    final_contam_mean: float
    overall_latency_mean: float
    rush_latency_mean: float
    calm_latency_mean: float
    t_crit_mean: float
    collapse_rate: float
    per_mech_acc_final: Dict[str, float]
    slot_heatmap_data: List[Dict[str, object]]
    contam_breakdown_t: Dict[str, List[float]]


def _create_modular_slots(scenario: ExperimentScenario) -> List[PhysicalSlot]:
    """Instantiate modular 120 x 60 x 60 cm shelf row slots across 3 vertical levels."""
    geom = scenario.geometry
    assert scenario.row_configs is not None
    if sum(1 for rc in scenario.row_configs if "PEG" in rc) > 1:
        raise ValueError("Sliding pegboard can only occupy at most one full shelf")
    slots: List[PhysicalSlot] = []
    tier_tags = {1: "eye", 2: "mid", 3: "3"}

    for tier_idx, row_code in enumerate(scenario.row_configs, start=1):
        t_tag = tier_tags[tier_idx]
        if tier_idx == 3 and row_code != "SLIDING_PEG":
            raise ValueError(
                "Top shelf (Row 3) is too high for drawers or wire bins; only SLIDING_PEG is allowed on Top."
            )
        if tier_idx != 3 and row_code == "SLIDING_PEG":
            raise ValueError("SLIDING_PEG is only supported on Top shelf (Row 3).")

        def _append_drawers(
            part_counts: Sequence[int],
            total_w_cm: float,
            x_start_cm: float = 0.0,
            d_offset: int = 0,
            is_9in: bool = False,
        ) -> None:
            mech_id = f"split_drawer_{t_tag}"
            n_drw = len(part_counts)
            slot_w = total_w_cm / max(1, n_drw)
            inner_w = geom.drawer_9in_inner_width_cm if is_9in else geom.drawer_inner_width_cm
            for d, p_cnt in enumerate(part_counts):
                d_idx = d + d_offset
                x_center = x_start_cm + (d + 0.5) * slot_w
                bay_side = 0 if x_center < 60.0 else 1
                for p in range(p_cnt):
                    p_width = geom.partition_width_cm(p_cnt, p, inner_width_cm=inner_w)
                    is_wide_top = p_cnt == 3 and p == 0
                    if is_9in:
                        hard_cap = 13 if is_wide_top else 8
                        nom_cap = 9.0 if is_wide_top else 5.0
                    else:
                        hard_cap = 10 if is_wide_top else 6
                        nom_cap = 6.6 if is_wide_top else 3.5
                    slots.append(
                        PhysicalSlot(
                            slot_id=f"R{tier_idx}-DRW-{d_idx+1}-P{p+1}",
                            mechanism_id=mech_id,
                            tier_index=tier_idx,
                            group_idx=d_idx,
                            sub_idx=p,
                            max_rolls_hard=hard_cap,
                            max_stack_height_cm=geom.drawer_inner_depth_cm,
                            clear_width_cm=p_width,
                            supports_coreless=True,
                            is_pull_down=False,
                            nominal_capacity=nom_cap,
                            bay_side=bay_side,
                            x_center_cm=x_center,
                            partition_count=p_cnt,
                            max_od_cm=23.0 if is_9in else inner_w,
                        )
                    )

        def _append_bins(
            n_bins: int,
            x_start_cm: float,
            bay_side_fixed: Optional[int] = None,
            b_offset: int = 0,
        ) -> None:
            bin_mech = "wire_bin_eye" if tier_idx == 1 else "wire_bin_mid"
            sub_w = geom.wire_bin_bay_width_cm / 2.0
            for b in range(n_bins * geom.wire_bin_subzones):
                b_idx = b + b_offset
                bin_idx = b // geom.wire_bin_subzones
                x_center = x_start_cm + bin_idx * geom.wire_bin_bay_width_cm + (b % 2) * sub_w + sub_w / 2.0
                bay_side = bay_side_fixed if bay_side_fixed is not None else (0 if x_center < 60.0 else 1)
                slots.append(
                    PhysicalSlot(
                        slot_id=f"R{tier_idx}-BIN-{b_idx+1}",
                        mechanism_id=bin_mech,
                        tier_index=tier_idx,
                        group_idx=b_idx,
                        sub_idx=0,
                        max_rolls_hard=geom.wire_bin_subzone_hard_cap,
                        max_stack_height_cm=geom.wire_bin_depth_cm,
                        clear_width_cm=geom.wire_bin_width_cm,
                        supports_coreless=True,
                        is_pull_down=False,
                        nominal_capacity=6.2,
                        bay_side=bay_side,
                        x_center_cm=x_center,
                        partition_count=1,
                        max_od_cm=geom.wire_bin_width_cm,
                    )
                )

        if row_code in ("7D6_4P", "7D6_3P", "7D_4P", "7D_3P", "4D4P_3D3P"):
            if row_code == "4D4P_3D3P":
                parts_list = [4, 4, 4, 4, 3, 3, 3]
            else:
                p_cnt = 4 if row_code.endswith("4P") else 3
                parts_list = [p_cnt] * geom.drawers_per_full_row
            _append_drawers(parts_list, len(parts_list) * geom.drawer_outer_width_cm, is_9in=False)

        elif row_code in ("5D9_4P", "5D9_3P"):
            p_cnt = 4 if row_code.endswith("4P") else 3
            parts_list = [p_cnt] * geom.drawers_9in_per_full_row
            _append_drawers(parts_list, len(parts_list) * geom.drawer_9in_outer_width_cm, is_9in=True)

        elif row_code == "3D9_3D6_4P":
            span_9in = 3 * geom.drawer_9in_outer_width_cm
            span_6in = 3 * geom.drawer_outer_width_cm
            _append_drawers([4, 4, 4], span_9in, x_start_cm=0.0, d_offset=0, is_9in=True)
            _append_drawers([4, 4, 4], span_6in, x_start_cm=span_9in, d_offset=3, is_9in=False)

        elif row_code in ("1B_4D6_4P", "1B_4D6_3P", "1B_4D_4P", "1B_4D_3P", "1B_2D4P_2D3P"):
            if row_code == "1B_2D4P_2D3P":
                parts_list = [4, 4, 3, 3]
            else:
                p_cnt = 4 if row_code.endswith("4P") else 3
                parts_list = [p_cnt] * geom.drawers_per_half_bay
            drw_span = len(parts_list) * geom.drawer_outer_width_cm
            _append_drawers(parts_list, drw_span, is_9in=False)
            _append_bins(1, drw_span, bay_side_fixed=1)

        elif row_code in ("1B_3D9_4P", "1B_3D9_3P"):
            p_cnt = 4 if row_code.endswith("4P") else 3
            parts_list = [p_cnt] * geom.drawers_9in_per_half_bay
            drw_span = len(parts_list) * geom.drawer_9in_outer_width_cm
            _append_drawers(parts_list, drw_span, is_9in=True)
            _append_bins(1, drw_span, bay_side_fixed=1)

        elif row_code in ("1D6_2B", "1D_2B"):
            drw_span = geom.drawer_outer_width_cm
            _append_drawers([4], drw_span, x_start_cm=0.0, is_9in=False)
            _append_bins(2, x_start_cm=drw_span, bay_side_fixed=None)

        elif row_code in ("1B_1D6_1B", "1B_1D_1B"):
            bin_w = geom.wire_bin_bay_width_cm
            drw_w = geom.drawer_outer_width_cm
            _append_bins(1, x_start_cm=0.0, bay_side_fixed=0, b_offset=0)
            _append_drawers([4], drw_w, x_start_cm=bin_w, d_offset=0, is_9in=False)
            _append_bins(1, x_start_cm=bin_w + drw_w, bay_side_fixed=1, b_offset=geom.wire_bin_subzones)

        elif row_code in ("2B_1D6", "2B_1D"):
            bin_span = 2 * geom.wire_bin_bay_width_cm
            drw_w = geom.drawer_outer_width_cm
            _append_bins(2, x_start_cm=0.0, bay_side_fixed=None, b_offset=0)
            _append_drawers([4], drw_w, x_start_cm=bin_span, d_offset=0, is_9in=False)

        elif row_code == "2B":
            _append_bins(geom.wire_bins_per_full_row, 0.0, bay_side_fixed=None)

        elif row_code == "SLIDING_PEG":
            peg_x_step = geom.shelf_width_cm / max(1, geom.pegs_per_row)
            for r in range(geom.peg_rows):
                for c in range(geom.pegs_per_row):
                    bay_side = 0 if c < (geom.pegs_per_row // 2) else 1
                    slots.append(
                        PhysicalSlot(
                            slot_id=f"R{tier_idx}-PEG-R{r+1}-C{c+1}",
                            mechanism_id="pegboard_ovh",
                            tier_index=tier_idx,
                            group_idx=r,
                            sub_idx=c,
                            max_rolls_hard=2,
                            max_stack_height_cm=geom.peg_usable_length_cm,
                            clear_width_cm=geom.peg_horizontal_spacing_cm,
                            supports_coreless=False,
                            is_pull_down=True,
                            nominal_capacity=1.65,
                            bay_side=bay_side,
                            x_center_cm=(c + 0.5) * peg_x_step,
                            partition_count=1,
                            max_od_cm=geom.peg_max_od_cm,
                        )
                    )
        else:
            raise ValueError(f"Unknown modular row config: {row_code}")
    return slots


def create_shelf_slots(scenario: ExperimentScenario) -> List[PhysicalSlot]:
    """Instantiate physical slots matching the scenario's modular layout."""
    if scenario.row_configs is None:
        raise ValueError(f"Unsupported topology or missing row_configs: {scenario.topology}")
    return _create_modular_slots(scenario)


def _assign_roll_to_slot(
    roll: RollItem, slot: PhysicalSlot, is_home: bool = True
) -> None:
    """Place a roll into a slot and update its location attributes."""
    slot.occupant_roll_ids.append(roll.roll_id)
    roll.current_slot = slot.slot_id
    roll.current_mechanism = slot.mechanism_id
    if is_home:
        roll.home_slot = slot.slot_id
        roll.home_mechanism = slot.mechanism_id
        if roll.variant_id not in slot.assigned_variant_ids:
            slot.assigned_variant_ids.append(roll.variant_id)


def _find_best_slot_in_mechanisms(
    roll: RollItem,
    mech_order: Sequence[str],
    slots_by_mech: Dict[str, List[PhysicalSlot]],
    rolls_by_id: Dict[int, RollItem],
    prefer_same_variant: bool = True,
) -> PhysicalSlot:
    """Find a physically valid slot within the highest-priority available mechanism."""
    for mech in mech_order:
        mech_slots = slots_by_mech.get(mech, [])
        if prefer_same_variant:
            for slot in mech_slots:
                if (
                    roll.variant_id in slot.assigned_variant_ids
                    and slot.can_physically_accept(roll, rolls_by_id)
                ):
                    return slot

        candidates = [
            s for s in mech_slots if s.can_physically_accept(roll, rolls_by_id)
        ]
        if candidates:
            candidates.sort(key=lambda s: (len(s.assigned_variant_ids), len(s.occupant_roll_ids)))
            return candidates[0]

    for mech_slots in slots_by_mech.values():
        for slot in mech_slots:
            if slot.supports_coreless or roll.has_core:
                return slot
    raise RuntimeError(f"No valid slot found for roll {roll.sku_code}")


def _order_slots_for_mode(
    slots: Sequence[PhysicalSlot], spatial_mode: str, rng: random.Random
) -> List[PhysicalSlot]:
    """Order physical slots along the 120 cm shelf based on spatial_order_mode."""
    out = list(slots)
    if spatial_mode in ("center_out", "bark_sticker_kit"):
        out.sort(key=lambda s: (s.tier_index, abs(s.x_center_cm - 60.0), s.group_idx, s.sub_idx))
    elif spatial_mode == "category_zones":
        out.sort(key=lambda s: (s.tier_index, s.x_center_cm, s.group_idx, s.sub_idx))
    elif spatial_mode == "vertical_column":
        out.sort(key=lambda s: (s.bay_side, int(s.x_center_cm // 30), s.tier_index, s.group_idx, s.sub_idx))
    elif spatial_mode == "random_perm":
        rng.shuffle(out)
    else:
        out.sort(key=lambda s: (s.tier_index, s.group_idx, s.sub_idx))
    return out


def _order_product_families(
    rolls: Sequence[RollItem],
    variants_by_id: Dict[int, RollVariantSKU],
    spatial_mode: str,
    rng: random.Random,
) -> List[Tuple[str, List[RollItem]]]:
    """Group rolls by product_name and order families by spatial_order_mode."""
    by_prod: Dict[str, List[RollItem]] = {}
    for r in rolls:
        by_prod.setdefault(r.product_name, []).append(r)
    for plist in by_prod.values():
        plist.sort(key=lambda r: (r.copy_index, r.variant_code, r.roll_id))

    def family_meta(prod_name: str, items: List[RollItem]) -> Tuple[int, float, str, str]:
        rush_flag = 0 if any(r.is_rush_priority for r in items) else 1
        peak_freq = max(variants_by_id[r.variant_id].rush_freq for r in items)
        cat = items[0].category
        return rush_flag, peak_freq, cat, prod_name

    families = list(by_prod.items())
    if spatial_mode == "center_out":
        families.sort(key=lambda kv: (-family_meta(kv[0], kv[1])[1], kv[0]))
    elif spatial_mode == "category_zones":
        families.sort(key=lambda kv: (family_meta(kv[0], kv[1])[2], -family_meta(kv[0], kv[1])[1], kv[0]))
    elif spatial_mode == "vertical_column":
        families.sort(key=lambda kv: (family_meta(kv[0], kv[1])[0], -family_meta(kv[0], kv[1])[1], kv[0]))
    elif spatial_mode == "bark_sticker_kit":
        def bark_rank(kv: Tuple[str, List[RollItem]]) -> Tuple[int, int, float, str]:
            name, items = kv
            cat = items[0].category
            is_kit = 0 if cat in ("Chocolate bark", "Ingredient sticker") else 1
            sub = 0 if "Peppermint" in name else (1 if "Almond" in name else 2)
            peak_freq = max(variants_by_id[r.variant_id].rush_freq for r in items)
            return (is_kit, sub, -peak_freq, name)
        families.sort(key=bark_rank)
    elif spatial_mode == "random_perm":
        rng.shuffle(families)
    else:
        families.sort(key=lambda kv: (-family_meta(kv[0], kv[1])[1], kv[0]))
    return families


def _pick_first_valid_slot(
    roll: RollItem,
    candidate_slots: Sequence[PhysicalSlot],
    rolls_by_id: Dict[int, RollItem],
    prefer_empty: bool = True,
) -> Optional[PhysicalSlot]:
    """Return the first physically valid slot from an ordered candidate list."""
    valid = [s for s in candidate_slots if s.can_physically_accept(roll, rolls_by_id)]
    if not valid:
        return None
    if prefer_empty:
        for s in valid:
            if not s.occupant_roll_ids:
                return s
    return min(valid, key=lambda s: (len(s.assigned_variant_ids), len(s.occupant_roll_ids)))


def allocate_modular_orientation(
    scenario: ExperimentScenario,
    variants: Sequence[RollVariantSKU],
    rolls: List[RollItem],
    slots: List[PhysicalSlot],
) -> List[PhysicalSlot]:
    """Assign all 128 rolls across a modular 120x60x60 cm layout orientation."""
    rng = random.Random(scenario.layout_seed + 137)
    rolls_by_id = {r.roll_id: r for r in rolls}
    variants_by_id = {v.variant_id: v for v in variants}
    ordered_slots = _order_slots_for_mode(slots, scenario.spatial_order_mode, rng)
    families = _order_product_families(rolls, variants_by_id, scenario.spatial_order_mode, rng)

    drawer_slots = [s for s in ordered_slots if s.mechanism_id.startswith("split_drawer")]
    peg_slots = [s for s in ordered_slots if "pegboard" in s.mechanism_id]
    bin_slots = [s for s in ordered_slots if "wire_bin" in s.mechanism_id]

    drawer_groups: List[Tuple[Tuple[int, str, int], List[PhysicalSlot]]] = []
    seen_grp: Dict[Tuple[int, str, int], List[PhysicalSlot]] = {}
    for s in drawer_slots:
        key = (s.tier_index, s.mechanism_id, s.group_idx)
        if key not in seen_grp:
            seen_grp[key] = []
            drawer_groups.append((key, seen_grp[key]))
        seen_grp[key].append(s)
    for _, slist in drawer_groups:
        slist.sort(key=lambda s: s.sub_idx)

    non_peg_by_tier = sorted(
        drawer_slots + bin_slots,
        key=lambda s: (s.tier_index, 0 if "drawer" in s.mechanism_id else 1),
    )
    reserve_slots = sorted(
        bin_slots + [s for s in drawer_slots if s.tier_index >= 2],
        key=lambda s: (
            0 if "wire_bin" in s.mechanism_id else 1,
            0 if s.tier_index == 2 else 1,
            -s.tier_index,
        ),
    )
    all_by_tier = sorted(ordered_slots, key=lambda s: s.tier_index)

    peg_rows_ordered: List[Tuple[Tuple[int, str, int], List[PhysicalSlot]]] = []
    seen_peg_row: Dict[Tuple[int, str, int], List[PhysicalSlot]] = {}
    for s in peg_slots:
        pkey = (s.tier_index, s.mechanism_id, s.group_idx)
        if pkey not in seen_peg_row:
            seen_peg_row[pkey] = []
            peg_rows_ordered.append((pkey, seen_peg_row[pkey]))
        seen_peg_row[pkey].append(s)
    for _, plist in peg_rows_ordered:
        plist.sort(key=lambda s: s.sub_idx)

    abc_mode = scenario.abc_colocation_mode
    ov_mode = scenario.overflow_mode

    def is_overflow_roll(roll: RollItem) -> bool:
        if ov_mode == "all_colocated":
            return False
        return roll.copy_index >= 2 or (not roll.is_rush_priority and roll.copy_index >= 1)

    active_rolls_list: List[Tuple[str, List[RollItem]]] = []
    overflow_rolls: List[RollItem] = []
    for prod_name, items in families:
        act = [r for r in items if not is_overflow_roll(r)]
        ovf = [r for r in items if is_overflow_roll(r)]
        if act:
            active_rolls_list.append((prod_name, act))
        overflow_rolls.extend(ovf)

    def _assign_peg_family(act_items: List[RollItem]) -> bool:
        """Try placing a cored non-bulk active family onto sliding pegs (1 tall or 2 short/peg)."""
        if not peg_slots or any(not r.has_core for r in act_items):
            return False

        def _peg_rem(slot: PhysicalSlot) -> int:
            if slot.occupant_roll_ids and any(
                rolls_by_id[rid].roll_height_cm > 10.0 for rid in slot.occupant_roll_ids
            ):
                return 0
            return slot.max_rolls_hard - len(slot.occupant_roll_ids)

        total_peg_cap_left = sum(_peg_rem(s) for s in peg_slots)
        if total_peg_cap_left < len(act_items):
            return False
        chosen_row: Optional[List[PhysicalSlot]] = None
        for _, r_slots in peg_rows_ordered:
            empty_pegs = [s for s in r_slots if not s.occupant_roll_ids]
            if len(empty_pegs) >= len(act_items):
                chosen_row = empty_pegs
                break
        if chosen_row is None:
            for _, r_slots in peg_rows_ordered:
                if sum(_peg_rem(s) for s in r_slots) >= len(act_items):
                    chosen_row = r_slots
                    break
        pool_seq = (chosen_row + peg_slots) if chosen_row is not None else peg_slots
        placed: List[Tuple[RollItem, PhysicalSlot]] = []
        ordered_items = sorted(
            act_items,
            key=lambda r: (0 if r.roll_height_cm > 10.0 else 1, r.variant_code, r.copy_index),
        )
        for r in ordered_items:
            target = _pick_first_valid_slot(r, pool_seq, rolls_by_id, prefer_empty=True)
            if target is None:
                for pr, ps in placed:
                    ps.occupant_roll_ids.remove(pr.roll_id)
                    ps.assigned_variant_ids[:] = [
                        rolls_by_id[rid].variant_id for rid in ps.occupant_roll_ids
                    ]
                    pr.current_slot = ""
                    pr.home_slot = ""
                    pr.current_mechanism = ""
                    pr.home_mechanism = ""
                return False
            _assign_roll_to_slot(r, target, is_home=True)
            placed.append((r, target))
        return True

    def _split_drawer_vs_peg_families(
        act_list: List[Tuple[str, List[RollItem]]]
    ) -> Tuple[List[Tuple[str, List[RollItem]]], List[Tuple[str, List[RollItem]]]]:
        """Route single-copy non-rush cored SKUs to Top Sliding Peg within the 32-peg budget."""
        if not peg_slots:
            return act_list, []
        n_pegs = len(peg_slots)
        peg_hard_budget = float(n_pegs)
        mid_drawer_slots = sum(1 for s in drawer_slots if s.tier_index == 1)
        btm_drawer_slots = sum(1 for s in drawer_slots if s.tier_index >= 2)
        eff_btm_relief = btm_drawer_slots if bin_slots else max(0, btm_drawer_slots - 21)
        if mid_drawer_slots >= 22:
            peg_budget = max(0.5 * n_pegs, peg_hard_budget - 0.5 * eff_btm_relief)
        else:
            peg_budget = peg_hard_budget
        drawer_fams: List[Tuple[str, List[RollItem]]] = []
        peg_fams: List[Tuple[str, List[RollItem]]] = []
        peg_used = 0.0
        for prod_name, items in act_list:
            is_prio = any(r.is_rush_priority for r in items)
            all_cored = all(r.has_core for r in items)
            has_bulk_dups = any(r.copy_index >= 1 for r in items)
            peg_cost = sum(1.0 if r.roll_height_cm > 10.0 else 0.5 for r in items)
            if (
                not is_prio
                and all_cored
                and not has_bulk_dups
                and (peg_used + peg_cost) <= peg_budget
            ):
                peg_fams.append((prod_name, items))
                peg_used += peg_cost
            else:
                drawer_fams.append((prod_name, items))
        return drawer_fams, peg_fams

    if abc_mode == "same_slot":
        drawer_fams, peg_fams = _split_drawer_vs_peg_families(active_rolls_list)
        for _, act_items in drawer_fams:
            by_var_code = sorted(act_items, key=lambda r: (r.variant_code, r.copy_index))
            target_slot: Optional[PhysicalSlot] = None
            for cand_pool in (
                [s for s in non_peg_by_tier if not s.occupant_roll_ids],
                non_peg_by_tier,
                all_by_tier,
            ):
                for s in cand_pool:
                    if not s.supports_coreless and any(not r.has_core for r in by_var_code):
                        continue
                    if "pegboard" in s.mechanism_id:
                        if len(s.occupant_roll_ids) + len(by_var_code) > 1 and (
                            any(r.roll_height_cm > 10.0 for r in by_var_code)
                            or any(rolls_by_id[rid].roll_height_cm > 10.0 for rid in s.occupant_roll_ids)
                        ):
                            continue
                        used_h = sum(rolls_by_id[rid].roll_height_cm for rid in s.occupant_roll_ids)
                        add_h = sum(r.roll_height_cm for r in by_var_code)
                        if used_h + add_h > s.max_stack_height_cm:
                            continue
                    if len(s.occupant_roll_ids) + len(by_var_code) <= s.max_rolls_hard:
                        target_slot = s
                        break
                if target_slot is not None:
                    break

            for r in by_var_code:
                if target_slot is not None and target_slot.can_physically_accept(r, rolls_by_id):
                    _assign_roll_to_slot(r, target_slot, is_home=True)
                else:
                    fb = _pick_first_valid_slot(r, non_peg_by_tier + all_by_tier, rolls_by_id, prefer_empty=False)
                    if fb is None:
                        raise RuntimeError(f"No valid slot found for roll {r.sku_code}")
                    _assign_roll_to_slot(r, fb, is_home=True)

        for _, act_items in peg_fams:
            if not _assign_peg_family(act_items):
                for r in act_items:
                    fb = _pick_first_valid_slot(r, non_peg_by_tier + all_by_tier, rolls_by_id, prefer_empty=False)
                    if fb is None:
                        raise RuntimeError(f"No valid slot found for roll {r.sku_code}")
                    _assign_roll_to_slot(r, fb, is_home=True)

    elif abc_mode == "adjacent_partition":
        drawer_fams, peg_fams = _split_drawer_vs_peg_families(active_rolls_list)
        for _, act_items in drawer_fams:
            variants_in_prod: Dict[int, List[RollItem]] = {}
            for r in sorted(act_items, key=lambda x: (x.variant_code, x.copy_index)):
                variants_in_prod.setdefault(r.variant_id, []).append(r)
            n_vars = len(variants_in_prod)

            chosen_group: Optional[List[PhysicalSlot]] = None
            for pass_empty_req in (n_vars, 1, 0):
                for (_, _, _), g_slots in drawer_groups:
                    empty_parts = [s for s in g_slots if not s.occupant_roll_ids]
                    avail_slots = [s for s in g_slots if len(s.occupant_roll_ids) < s.max_rolls_hard]
                    if len(empty_parts) >= pass_empty_req and len(avail_slots) >= 1:
                        cap_left = sum(s.max_rolls_hard - len(s.occupant_roll_ids) for s in g_slots)
                        if cap_left >= len(act_items):
                            chosen_group = g_slots
                            break
                if chosen_group is not None:
                    break

            if chosen_group is not None:
                ordered_group = sorted(
                    chosen_group,
                    key=lambda s: (1 if s.occupant_roll_ids else 0, s.sub_idx),
                )
                for idx, v_rolls in enumerate(variants_in_prod.values()):
                    part_slot = ordered_group[idx % len(ordered_group)]
                    for r in v_rolls:
                        if part_slot.can_physically_accept(r, rolls_by_id):
                            _assign_roll_to_slot(r, part_slot, is_home=True)
                        else:
                            fb = _pick_first_valid_slot(r, chosen_group + non_peg_by_tier + all_by_tier, rolls_by_id)
                            if fb is None:
                                raise RuntimeError(f"No valid slot found for roll {r.sku_code}")
                            _assign_roll_to_slot(r, fb, is_home=True)
            else:
                for r in act_items:
                    fb = _pick_first_valid_slot(r, non_peg_by_tier + all_by_tier, rolls_by_id)
                    if fb is None:
                        raise RuntimeError(f"No valid slot found for roll {r.sku_code}")
                    _assign_roll_to_slot(r, fb, is_home=True)

        for _, act_items in peg_fams:
            if not _assign_peg_family(act_items):
                for r in act_items:
                    fb = _pick_first_valid_slot(r, non_peg_by_tier + all_by_tier, rolls_by_id)
                    if fb is None:
                        raise RuntimeError(f"No valid slot found for roll {r.sku_code}")
                    _assign_roll_to_slot(r, fb, is_home=True)

    elif abc_mode == "same_tier_bay":
        for _, act_items in active_rolls_list:
            short_or_coreless = [r for r in act_items if not (r.roll_height_cm > 11.5 and r.has_core)]
            tall_cored = [r for r in act_items if r.roll_height_cm > 11.5 and r.has_core]
            target_tier: Optional[int] = None

            for r in short_or_coreless:
                cands = [s for s in non_peg_by_tier if (target_tier is None or s.tier_index == target_tier)]
                slot = _pick_first_valid_slot(r, cands + all_by_tier, rolls_by_id)
                if slot is None:
                    raise RuntimeError(f"No valid slot found for roll {r.sku_code}")
                if target_tier is None:
                    target_tier = slot.tier_index
                _assign_roll_to_slot(r, slot, is_home=True)

            for r in tall_cored:
                same_tier_pegs = [
                    s for s in peg_slots
                    if target_tier is None or s.tier_index == target_tier
                ]
                same_tier_other = [
                    s for s in non_peg_by_tier
                    if target_tier is None or s.tier_index == target_tier
                ]
                slot = _pick_first_valid_slot(
                    r,
                    same_tier_pegs + same_tier_other + all_by_tier,
                    rolls_by_id,
                    prefer_empty=True,
                )
                if slot is None:
                    raise RuntimeError(f"No valid slot found for roll {r.sku_code}")
                if target_tier is None:
                    target_tier = slot.tier_index
                _assign_roll_to_slot(r, slot, is_home=True)

    else:
        for _, act_items in active_rolls_list:
            for r in act_items:
                if r.roll_height_cm > 10.0 and r.has_core:
                    secondary = [s for s in ordered_slots if s.tier_index >= 2 and s.supports_coreless]
                    cands = sorted(peg_slots, key=lambda s: -s.tier_index) + secondary + all_by_tier
                elif ov_mode == "all_colocated" and not r.is_rush_priority:
                    cands = bin_slots + sorted(drawer_slots, key=lambda s: s.tier_index) + all_by_tier
                else:
                    cands = sorted(drawer_slots, key=lambda s: s.tier_index) + non_peg_by_tier + all_by_tier
                slot = _pick_first_valid_slot(r, cands, rolls_by_id, prefer_empty=True)
                if slot is None:
                    raise RuntimeError(f"No valid slot found for roll {r.sku_code}")
                _assign_roll_to_slot(r, slot, is_home=True)

    overflow_rolls.sort(key=lambda r: (r.copy_index, -variants_by_id[r.variant_id].rush_freq, r.variant_id))
    for r in overflow_rolls:
        same_var_reserve = [
            s for s in reserve_slots
            if r.variant_id in s.assigned_variant_ids and s.can_physically_accept(r, rolls_by_id)
        ]
        if same_var_reserve:
            _assign_roll_to_slot(r, same_var_reserve[0], is_home=True)
            continue
        fallback_pool = reserve_slots + all_by_tier
        slot = _pick_first_valid_slot(r, fallback_pool, rolls_by_id, prefer_empty=False)
        if slot is None:
            raise RuntimeError(f"No valid overflow slot found for roll {r.sku_code}")
        _assign_roll_to_slot(r, slot, is_home=True)

    return slots


def compute_layout_metrics(
    rolls: Sequence[RollItem], slots: Sequence[PhysicalSlot]
) -> Dict[str, float]:
    """Compute structural A/B/C co-location and capacity metrics for a placed layout."""
    slots_by_id = {s.slot_id: s for s in slots}
    active_by_prod: Dict[str, Dict[str, RollItem]] = {}
    for r in rolls:
        if r.copy_index == 0:
            active_by_prod.setdefault(r.product_name, {})[r.variant_code] = r

    multi_prods = [v for v in active_by_prod.values() if len(v) >= 2]
    same_drawer_count = 0
    same_tier_count = 0
    for var_map in multi_prods:
        r_slots = [slots_by_id[r.home_slot] for r in var_map.values()]
        if len({(s.mechanism_id, s.group_idx) for s in r_slots}) == 1:
            same_drawer_count += 1
        if len({s.tier_index for s in r_slots}) == 1:
            same_tier_count += 1

    active_rolls = [r for r in rolls if r.copy_index == 0]
    eye_active = sum(1 for r in active_rolls if slots_by_id[r.home_slot].tier_index == 1)
    eye_non_empty = [s for s in slots if s.tier_index == 1 and s.occupant_roll_ids]
    mean_eye_fill = (
        sum(len(s.occupant_roll_ids) / max(1.0, s.nominal_capacity) for s in eye_non_empty)
        / max(1, len(eye_non_empty))
    )

    n_multi = max(1, len(multi_prods))
    return {
        "abc_same_drawer_pct": same_drawer_count / n_multi,
        "abc_same_tier_pct": same_tier_count / n_multi,
        "active_eye_share": eye_active / max(1, len(active_rolls)),
        "mean_eye_fill_ratio": mean_eye_fill,
    }


def allocate_home_slots(
    scenario: ExperimentScenario,
    variants: Sequence[RollVariantSKU],
    rolls: List[RollItem],
) -> List[PhysicalSlot]:
    """Assign initial home slots for all 128 rolls across a modular layout."""
    slots = create_shelf_slots(scenario)
    return allocate_modular_orientation(scenario, variants, rolls, slots)


def _weighted_choice(rng: random.Random, weights: Sequence[float]) -> int:
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for idx, w in enumerate(weights):
        acc += w
        if r <= acc:
            return idx
    return len(weights) - 1


def _sample_job_variants(
    rng: random.Random,
    is_rush: bool,
    freq_calm: Sequence[float],
    freq_rush: Sequence[float],
    co_matrix: Sequence[Sequence[float]],
    high_assoc_siblings: Optional[Sequence[Sequence[int]]] = None,
) -> List[int]:
    """Sample 1 to 4 co-retrieved SKU variants with explicit 90% A/B/C association."""
    freqs = freq_rush if is_rush else freq_calm
    r_val = rng.random()
    target_size = 1 if r_val < 0.25 else (2 if r_val < 0.65 else (3 if r_val < 0.90 else 4))

    anchor = _weighted_choice(rng, freqs)
    picked = [anchor]
    n = len(freqs)

    sibs = (
        high_assoc_siblings[anchor]
        if high_assoc_siblings is not None
        else [j for j in range(n) if j != anchor and co_matrix[anchor][j] >= 0.85]
    )
    for sib in sibs:
        if len(picked) >= 4:
            break
        if rng.random() < co_matrix[anchor][sib]:
            picked.append(sib)

    job_size = min(4, max(target_size, len(picked)))
    while len(picked) < job_size:
        weights = [
            (co_matrix[anchor][j] * 5.0 + freqs[j]) if j not in picked else 0.0
            for j in range(n)
        ]
        if sum(weights) <= 0.0:
            break
        picked.append(_weighted_choice(rng, weights))
    return picked


def _compute_shannon_entropy(
    rolls: Sequence[RollItem], slots_by_id: Dict[str, PhysicalSlot]
) -> float:
    """Normalized spatial entropy measuring SKU scattering across mechanisms/slots."""
    by_variant: Dict[int, Dict[str, int]] = {}
    for r in rolls:
        by_variant.setdefault(r.variant_id, {})
        slot = slots_by_id[r.current_slot]
        zone = f"{slot.mechanism_id}:{slot.group_idx}"
        by_variant[r.variant_id][zone] = by_variant[r.variant_id].get(zone, 0) + 1

    total_h = 0.0
    total_w = 0
    max_zones_log = math.log2(8.0)
    for counts in by_variant.values():
        m = sum(counts.values())
        if m <= 1:
            continue
        h = 0.0
        for c in counts.values():
            p = c / m
            if p > 0.0:
                h -= p * math.log2(p)
        total_h += m * min(1.0, h / max_zones_log)
        total_w += m

    displaced_singles = sum(1 for r in rolls if r.current_slot != r.home_slot) / len(rolls)
    base_entropy = (total_h / max(1, total_w)) if total_w > 0 else 0.0
    return min(1.0, 0.55 * base_entropy + 0.45 * displaced_singles)


def _move_roll(
    roll: RollItem,
    target_slot: PhysicalSlot,
    slots_by_id: Dict[str, PhysicalSlot],
) -> None:
    if roll.current_slot and roll.current_slot in slots_by_id:
        old = slots_by_id[roll.current_slot]
        if roll.roll_id in old.occupant_roll_ids:
            old.occupant_roll_ids.remove(roll.roll_id)
    target_slot.occupant_roll_ids.append(roll.roll_id)
    roll.current_slot = target_slot.slot_id
    roll.current_mechanism = target_slot.mechanism_id


def run_single_replication(
    scenario: ExperimentScenario,
    variants: Sequence[RollVariantSKU],
    base_rolls: Sequence[RollItem],
    freq_calm: Sequence[float],
    freq_rush: Sequence[float],
    co_matrix: Sequence[List[float]],
    seed: int,
    high_assoc_siblings: Optional[Sequence[Sequence[int]]] = None,
) -> ReplicationTrace:
    """Execute one 1,000-cycle Monte Carlo replication."""
    rng = random.Random(seed)
    beh: BehavioralConfig = scenario.behavior
    specs: Dict[str, MechanismSpec] = build_mechanism_specs(scenario.topology)
    if high_assoc_siblings is None:
        high_assoc_siblings = [
            [j for j in range(len(variants)) if j != i and co_matrix[i][j] >= 0.85]
            for i in range(len(variants))
        ]

    rolls = [
        RollItem(
            roll_id=r.roll_id,
            variant_id=r.variant_id,
            sku_code=r.sku_code,
            product_name=r.product_name,
            variant_code=r.variant_code,
            copy_index=r.copy_index,
            outer_diameter_cm=r.outer_diameter_cm,
            roll_height_cm=r.roll_height_cm,
            has_core=r.has_core,
            category=r.category,
            is_rush_priority=r.is_rush_priority,
            visual_similarity_group=r.visual_similarity_group,
        )
        for r in base_rolls
    ]
    slots = allocate_home_slots(scenario, variants, rolls)
    slots_by_id = {s.slot_id: s for s in slots}
    slots_by_mech: Dict[str, List[PhysicalSlot]] = {}
    for s in slots:
        slots_by_mech.setdefault(s.mechanism_id, []).append(s)
    rolls_by_id = {r.roll_id: r for r in rolls}
    rolls_by_variant: Dict[int, List[RollItem]] = {}
    for r in rolls:
        rolls_by_variant.setdefault(r.variant_id, []).append(r)

    n_bins, extra_styles = count_shelf_complexity(scenario.row_configs)
    mixed_bin_drw_rows = count_mixed_bin_drawer_rows(scenario.row_configs)
    size_mix_blur = beh.drawer_size_mix_penalty if has_mixed_drawer_sizes(scenario.row_configs) else 0.0
    n_drawer_slots = sum(1 for s in slots if s.mechanism_id.startswith("split_drawer"))
    slot_deficit_blur = beh.slot_deficit_penalty * max(0.0, (48 - n_drawer_slots) / 48.0)
    row_width_blur = beh.row_width_mismatch_penalty if has_row_width_mismatch(scenario.row_configs) else 0.0
    all_bin_slots = [s for s in slots if "wire_bin" in s.mechanism_id]

    eye_mechs = [s.mechanism_id for s in slots if s.tier_index == 1]
    if "split_drawer_eye" in slots_by_mech:
        eye_mech_default = "split_drawer_eye"
    elif "split_drawer" in slots_by_mech:
        eye_mech_default = "split_drawer"
    elif "wire_bin_eye" in slots_by_mech:
        eye_mech_default = "wire_bin_eye"
    else:
        eye_mech_default = eye_mechs[0] if eye_mechs else next(iter(slots_by_mech))

    unlatched_peg_rows: set[int] = set()
    total_cycles = scenario.rush.total_cycles
    shift_cycles = scenario.rush.shift_cycles
    rush_cutoff = int(round(shift_cycles * scenario.rush.rush_job_share))

    acc_t: List[float] = []
    entropy_t: List[float] = []
    contam_t: List[float] = []
    eye_dump_t: List[float] = []
    latency_t: List[float] = []
    t_crit: Optional[int] = None

    for t in range(total_cycles):
        cycle_in_shift = t % shift_cycles
        is_rush = cycle_in_shift < rush_cutoff
        tau = rng.uniform(
            *(scenario.rush.rush_tau_range if is_rush else scenario.rush.calm_tau_range)
        )

        if not is_rush and unlatched_peg_rows:
            for row_idx in sorted(unlatched_peg_rows):
                if rng.random() < 0.70:
                    unlatched_peg_rows.discard(row_idx)

        picked_vids = _sample_job_variants(
            rng, is_rush, freq_calm, freq_rush, co_matrix, high_assoc_siblings=high_assoc_siblings
        )
        retrieved: List[Tuple[RollItem, bool, str]] = []
        opened_groups: set[Tuple[str, int]] = set()
        job_latency = 0.0

        for vid in picked_vids:
            candidates = rolls_by_variant[vid]
            home_candidates = sorted(
                [r for r in candidates if r.current_slot == r.home_slot],
                key=lambda r: (slots_by_id[r.home_slot].tier_index, r.copy_index),
            )
            if home_candidates:
                roll = home_candidates[0]
                was_home = True
            else:
                roll = rng.choice(candidates)
                was_home = False

            origin_slot = roll.current_slot
            origin_mech = roll.current_mechanism
            spec = specs[origin_mech]
            slot_obj = slots_by_id[origin_slot]
            grp_key = (origin_mech, slot_obj.group_idx)
            drawer_already_open = origin_mech.startswith("split_drawer") and grp_key in opened_groups
            opened_groups.add(grp_key)

            edge_factor = abs(slot_obj.x_center_cm - 60.0) / 60.0
            t_step = spec.base_retrieval_sec * (1.0 + 0.15 * (1.0 - tau)) + 0.35 * edge_factor
            if len(slot_obj.assigned_variant_ids) > 1:
                t_step += 0.25 * (len(slot_obj.assigned_variant_ids) - 1)
            if drawer_already_open:
                t_step = max(0.8, t_step - 1.35)
            if "pegboard" in origin_mech and len(slot_obj.occupant_roll_ids) >= 2:
                t_step += 1.45 * (len(slot_obj.occupant_roll_ids) - 1)
            if (
                "wire_bin" in origin_mech
                and scenario.row_configs
                and any(rc in ("2B", "1D6_2B", "1B_1D6_1B", "2B_1D6", "1D_2B", "1B_1D_1B", "2B_1D") for rc in scenario.row_configs)
            ):
                t_step += 0.45
            if extra_styles > 0:
                t_step += 0.18 * extra_styles
            if (
                scenario.row_configs
                and 1 <= slot_obj.tier_index <= 2
                and "B" in scenario.row_configs[slot_obj.tier_index - 1]
                and "D" in scenario.row_configs[slot_obj.tier_index - 1]
            ):
                t_step += 0.55

            if spec.is_pull_down:
                is_heavy_peg_use = roll.is_rush_priority or len(slot_obj.occupant_roll_ids) >= 2
                unlatch_mult = 1.0 if is_heavy_peg_use else 0.10
                if is_rush and rng.random() < beh.unlatch_rush_prob * tau * unlatch_mult:
                    unlatched_peg_rows.add(slot_obj.group_idx)
            elif unlatched_peg_rows and slot_obj.tier_index == 1:
                t_step += 2.2

            if not was_home:
                rho = slot_obj.fill_ratio(rolls_by_id, scenario.target_fill_ratio)
                t_step += beh.search_miss_penalty_sec * (1.0 + 0.6 * rho) * (1.15 - 0.3 * scenario.cue_salience)
            job_latency += t_step
            retrieved.append((roll, was_home, origin_slot))

        for mech_id, group_idx in sorted(opened_groups):
            v_active = specs[mech_id].visibility_open
            p_fix = beh.base_correction_rate * v_active * scenario.cue_salience * (1.0 - 0.7 * tau)
            group_slots = [
                s for s in slots_by_mech[mech_id] if s.group_idx == group_idx
            ]
            for s in group_slots:
                for rid in list(s.occupant_roll_ids):
                    r_obs = rolls_by_id[rid]
                    if r_obs.current_slot != r_obs.home_slot and rng.random() < p_fix:
                        home_s = slots_by_id[r_obs.home_slot]
                        if home_s.can_physically_accept(r_obs, rolls_by_id):
                            _move_roll(r_obs, home_s, slots_by_id)

        if not is_rush and scenario.cue_salience >= 0.50:
            for s in slots:
                v_closed = specs[s.mechanism_id].visibility_closed
                if v_closed < 0.25:
                    continue
                p_passive = 0.025 * v_closed * scenario.cue_salience
                for rid in list(s.occupant_roll_ids):
                    r_obs = rolls_by_id[rid]
                    if r_obs.current_slot != r_obs.home_slot and rng.random() < p_passive:
                        home_s = slots_by_id[r_obs.home_slot]
                        if home_s.can_physically_accept(r_obs, rolls_by_id):
                            _move_roll(r_obs, home_s, slots_by_id)

        distinct_tiers = len({slots_by_id[r.home_slot].tier_index for r, _, _ in retrieved})
        distinct_mechs = len({slots_by_id[r.home_slot].mechanism_id for r, _, _ in retrieved})
        distinct_drawers = len({
            (slots_by_id[r.home_slot].mechanism_id, slots_by_id[r.home_slot].group_idx)
            for r, _, _ in retrieved
        })
        companion_slots = [r.home_slot for r, _, _ in retrieved]
        returned_groups: set[Tuple[str, int]] = set()

        for roll, was_home, origin_slot in retrieved:
            home_slot = slots_by_id[roll.home_slot]
            home_spec = specs[home_slot.mechanism_id]
            home_group = (home_slot.mechanism_id, home_slot.group_idx)

            edge_factor_ret = abs(home_slot.x_center_cm - 60.0) / 60.0
            t_ret = home_spec.base_return_sec + 0.25 * edge_factor_ret
            if home_slot.mechanism_id.startswith("split_drawer") and home_group in returned_groups:
                t_ret = max(0.7, t_ret - 1.05)
            if "pegboard" in home_slot.mechanism_id and len(home_slot.occupant_roll_ids) >= 2:
                t_ret += 1.15 * (len(home_slot.occupant_roll_ids) - 1)
            returned_groups.add(home_group)
            job_latency += t_ret

            target_slot = home_slot
            if not was_home:
                p_identify = min(0.98, max(0.08, scenario.cue_salience * (1.0 - 0.55 * (tau ** 2))))
                if rng.random() > p_identify:
                    if rng.random() < 0.65:
                        target_slot = slots_by_id[origin_slot]
                    else:
                        target_slot = _find_best_slot_in_mechanisms(
                            roll, [eye_mech_default], slots_by_mech, rolls_by_id, prefer_same_variant=False
                        )
                    if target_slot.can_physically_accept(roll, rolls_by_id):
                        _move_roll(roll, target_slot, slots_by_id)
                        continue

            rho_slot = home_slot.fill_ratio(rolls_by_id, scenario.target_fill_ratio)
            neighbors = [
                s for s in slots_by_mech[home_slot.mechanism_id]
                if s.group_idx == home_slot.group_idx
            ]
            tot_n = sum(len(s.occupant_roll_ids) for s in neighbors)
            mis_n = sum(
                1 for s in neighbors for rid in s.occupant_roll_ids
                if rolls_by_id[rid].current_slot != rolls_by_id[rid].home_slot
            )
            h_local = (mis_n / tot_n) if tot_n > 0 else 0.0

            phi_rush = 1.0 + beh.alpha_rush * (tau ** 2)
            occ_pen = beh.occlusion_friction_penalty if (unlatched_peg_rows and home_slot.tier_index == 1) else 0.0
            cross_tier_factor = 1.0 + (0.42 * max(0, distinct_tiers - 1))

            if len(retrieved) > 1 and distinct_drawers == 1:
                co_location_bonus = -0.22
            elif len(retrieved) > 1 and any(
                r2.roll_id != roll.roll_id
                and (slots_by_id[r2.home_slot].mechanism_id, slots_by_id[r2.home_slot].group_idx) == home_group
                for r2, _, _ in retrieved
            ):
                co_location_bonus = -0.14
            else:
                co_location_bonus = 0.0
            spread_penalty = 0.12 if (distinct_tiers == 1 and distinct_mechs > 1) else 0.0
            edge_ergo = 0.08 * edge_factor_ret

            phi_ergo = max(
                0.55,
                1.0
                + beh.alpha_ergo * home_spec.reach_actuation_cost * cross_tier_factor
                + occ_pen
                + spread_penalty
                + co_location_bonus
                + edge_ergo,
            )
            crowd_term = max(0.0, (rho_slot - 0.75) / (1.05 - rho_slot))
            peg_bulk_pen = (
                0.45 * max(0, len(home_slot.occupant_roll_ids) - 1)
                if "pegboard" in home_slot.mechanism_id
                else 0.0
            )
            phi_crowd = 1.0 + beh.alpha_crowd * crowd_term + peg_bulk_pen
            part_blur_bias = (
                (0.18 if home_slot.sub_idx == 0 else 0.08)
                if (home_slot.mechanism_id.startswith("split_drawer") and home_slot.partition_count == 3)
                else 0.0
            )
            same_slot_mix = beh.multi_sku_slot_penalty * max(0, len(home_slot.assigned_variant_ids) - 1)
            bin_dump_blur = beh.bin_dump_error_penalty * n_bins + (
                0.06 if "wire_bin" in home_slot.mechanism_id else 0.0
            )
            cog_load_blur = (
                beh.mech_diversity_penalty * extra_styles
                + beh.bin_drawer_same_row_penalty * mixed_bin_drw_rows
                + size_mix_blur
                + slot_deficit_blur
                + row_width_blur
            )
            phi_blur = max(
                0.65,
                1.0
                + beh.alpha_blur
                * max(
                    0.0,
                    h_local
                    + part_blur_bias
                    + same_slot_mix
                    + bin_dump_blur
                    + cog_load_blur
                    + (1.0 - scenario.cue_salience) * 0.25
                    + co_location_bonus,
                ),
            )

            p_err = min(0.95, max(0.005, beh.base_error_rate * phi_rush * phi_ergo * phi_crowd * phi_blur))

            if rng.random() < p_err:
                w_adj = 0.42 + ((0.25 if home_slot.sub_idx == 0 else 0.12) if home_slot.partition_count == 3 else 0.0) + 0.15 * h_local
                w_eye = 0.45 * home_spec.reach_actuation_cost + (0.20 if home_spec.is_pull_down else 0.0)
                w_comp = 0.22 if len(companion_slots) > 1 else 0.02
                w_bin = (beh.bin_dump_routing_weight * n_bins) if all_bin_slots else 0.0
                mode = _weighted_choice(rng, [w_adj, w_eye, w_comp, w_bin])

                candidate_slots: List[PhysicalSlot] = []
                if mode == 0:
                    candidate_slots = [
                        s for s in slots_by_mech[home_slot.mechanism_id]
                        if s.group_idx == home_slot.group_idx and s.slot_id != home_slot.slot_id
                        and s.can_physically_accept(roll, rolls_by_id)
                    ]
                    if not candidate_slots:
                        candidate_slots = [
                            s for s in slots_by_mech[home_slot.mechanism_id]
                            if s.slot_id != home_slot.slot_id and s.can_physically_accept(roll, rolls_by_id)
                        ]
                elif mode == 1:
                    dump_mech = "split_drawer" if "split_drawer" in slots_by_mech else eye_mech_default
                    candidate_slots = [
                        s for s in slots_by_mech[dump_mech]
                        if s.slot_id != home_slot.slot_id and s.can_physically_accept(roll, rolls_by_id)
                    ]
                    if not candidate_slots:
                        candidate_slots = [
                            s for s in slots
                            if s.tier_index in (1, 2)
                            and s.supports_coreless
                            and s.slot_id != home_slot.slot_id
                            and s.can_physically_accept(roll, rolls_by_id)
                        ]
                elif mode == 2:
                    comp_sids = [sid for sid in companion_slots if sid != home_slot.slot_id]
                    if comp_sids:
                        cs = slots_by_id[rng.choice(comp_sids)]
                        adj_comp = [
                            s for s in slots_by_mech[cs.mechanism_id]
                            if s.group_idx == cs.group_idx and s.can_physically_accept(roll, rolls_by_id)
                        ]
                        candidate_slots = adj_comp
                else:
                    candidate_slots = [
                        s for s in all_bin_slots
                        if s.slot_id != home_slot.slot_id and s.can_physically_accept(roll, rolls_by_id)
                    ]

                if candidate_slots:
                    chosen = rng.choice(candidate_slots)
                    _move_roll(roll, chosen, slots_by_id)
                    roll.misplaced_history_count += 1
                    continue

            if home_slot.can_physically_accept(roll, rolls_by_id) or roll.current_slot == home_slot.slot_id:
                if roll.current_slot != home_slot.slot_id:
                    _move_roll(roll, home_slot, slots_by_id)
            else:
                spill = [
                    s for s in slots_by_mech[home_slot.mechanism_id]
                    if s.group_idx == home_slot.group_idx and s.can_physically_accept(roll, rolls_by_id)
                ]
                if not spill:
                    spill = [
                        s for s in slots_by_mech[eye_mech_default]
                        if s.can_physically_accept(roll, rolls_by_id)
                    ]
                if not spill:
                    spill = [
                        s for s in slots
                        if s.supports_coreless and s.can_physically_accept(roll, rolls_by_id)
                    ]
                if spill:
                    _move_roll(roll, rng.choice(spill), slots_by_id)

        n_rolls = len(rolls)
        acc = sum(1 for r in rolls if r.current_slot == r.home_slot) / n_rolls
        ent = _compute_shannon_entropy(rolls, slots_by_id)
        contam = sum(1 for r in rolls if r.current_mechanism != r.home_mechanism) / n_rolls
        eye_dump = sum(
            1 for r in rolls
            if r.current_mechanism != r.home_mechanism and slots_by_id[r.current_slot].tier_index == 1
        ) / n_rolls

        acc_t.append(acc)
        entropy_t.append(ent)
        contam_t.append(contam)
        eye_dump_t.append(eye_dump)
        latency_t.append(job_latency)

        if t_crit is None and len(acc_t) >= 10:
            if sum(acc_t[-10:]) / 10.0 < beh.usability_accuracy_threshold:
                t_crit = t + 1

    per_mech_acc: Dict[str, float] = {}
    for mech_id in slots_by_mech:
        mech_rolls = [r for r in rolls if r.home_mechanism == mech_id]
        if mech_rolls:
            per_mech_acc[mech_id] = sum(
                1 for r in mech_rolls if r.current_slot == r.home_slot
            ) / len(mech_rolls)
        else:
            per_mech_acc[mech_id] = 1.0

    slot_misplace: Dict[str, int] = {}
    slot_occ: Dict[str, int] = {}
    for s in slots:
        slot_occ[s.slot_id] = len(s.occupant_roll_ids)
        slot_misplace[s.slot_id] = sum(
            1 for rid in s.occupant_roll_ids if rolls_by_id[rid].home_slot != s.slot_id
        )

    return ReplicationTrace(
        accuracy_t=acc_t,
        entropy_t=entropy_t,
        cross_tier_contam_t=contam_t,
        eye_drawer_dump_share_t=eye_dump_t,
        latency_t=latency_t,
        per_mech_accuracy_final=per_mech_acc,
        final_slot_misplace_counts=slot_misplace,
        final_slot_occupant_counts=slot_occ,
        t_crit=t_crit,
    )


def run_experiment_monte_carlo(
    scenario: ExperimentScenario,
    n_replications: int = 50,
    base_seed: int = 1000,
    preloaded_inventory: Optional[
        Tuple[List[RollVariantSKU], List[RollItem], List[float], List[float], List[List[float]]]
    ] = None,
) -> ExperimentSummary:
    """Run N Monte Carlo replications for an experimental scenario and aggregate KPIs."""
    if preloaded_inventory is not None:
        variants, base_rolls, freq_calm, freq_rush, co_matrix = preloaded_inventory
    else:
        variants, base_rolls, freq_calm, freq_rush, co_matrix = load_roll_inventory(
            DEFAULT_CSV_PATH, scenario.rush
        )
    high_assoc_siblings = [
        [j for j in range(len(variants)) if j != i and co_matrix[i][j] >= 0.85]
        for i in range(len(variants))
    ]
    total_cycles = scenario.rush.total_cycles
    shift_cycles = scenario.rush.shift_cycles
    rush_cutoff = int(round(shift_cycles * scenario.rush.rush_job_share))

    traces: List[ReplicationTrace] = []
    for rep in range(n_replications):
        traces.append(
            run_single_replication(
                scenario,
                variants,
                base_rolls,
                freq_calm,
                freq_rush,
                co_matrix,
                seed=base_seed + rep * 17,
                high_assoc_siblings=high_assoc_siblings,
            )
        )

    acc_mean, acc_std = [], []
    entropy_mean, contam_mean, eye_dump_mean = [], [], []
    latency_mean, latency_std = [], []

    for t in range(total_cycles):
        accs = [tr.accuracy_t[t] for tr in traces]
        lats = [tr.latency_t[t] for tr in traces]
        acc_mean.append(statistics.mean(accs))
        acc_std.append(statistics.pstdev(accs))
        entropy_mean.append(statistics.mean(tr.entropy_t[t] for tr in traces))
        contam_mean.append(statistics.mean(tr.cross_tier_contam_t[t] for tr in traces))
        eye_dump_mean.append(statistics.mean(tr.eye_drawer_dump_share_t[t] for tr in traces))
        latency_mean.append(statistics.mean(lats))
        latency_std.append(statistics.pstdev(lats))

    home_t = acc_mean[:]
    within_drift_t = [max(0.0, 1.0 - acc_mean[t] - contam_mean[t]) for t in range(total_cycles)]
    eye_dump_t = eye_dump_mean[:]
    other_contam_t = [max(0.0, contam_mean[t] - eye_dump_mean[t]) for t in range(total_cycles)]

    rush_lats = [latency_mean[t] for t in range(total_cycles) if (t % shift_cycles) < rush_cutoff]
    calm_lats = [latency_mean[t] for t in range(total_cycles) if (t % shift_cycles) >= rush_cutoff]

    t_crits = [tr.t_crit for tr in traces if tr.t_crit is not None]
    collapse_rate = len(t_crits) / n_replications
    t_crit_mean = statistics.mean(
        tr.t_crit if tr.t_crit is not None else total_cycles for tr in traces
    )

    mech_keys = list(traces[0].per_mech_accuracy_final.keys())
    per_mech_final = {
        k: statistics.mean(tr.per_mech_accuracy_final[k] for tr in traces) for k in mech_keys
    }

    sample_slots = allocate_home_slots(scenario, variants, [
        RollItem(
            roll_id=r.roll_id, variant_id=r.variant_id, sku_code=r.sku_code,
            product_name=r.product_name, variant_code=r.variant_code, copy_index=r.copy_index,
            outer_diameter_cm=r.outer_diameter_cm, roll_height_cm=r.roll_height_cm,
            has_core=r.has_core, category=r.category, is_rush_priority=r.is_rush_priority,
            visual_similarity_group=r.visual_similarity_group,
        ) for r in base_rolls
    ])
    heatmap_rows = []
    for s in sample_slots:
        mean_mis = statistics.mean(tr.final_slot_misplace_counts[s.slot_id] for tr in traces)
        mean_occ = statistics.mean(tr.final_slot_occupant_counts[s.slot_id] for tr in traces)
        heatmap_rows.append(
            {
                "slot_id": s.slot_id,
                "mechanism_id": s.mechanism_id,
                "tier_index": s.tier_index,
                "group_idx": s.group_idx,
                "sub_idx": s.sub_idx,
                "home_rolls": len(s.occupant_roll_ids),
                "mean_final_occupants": round(mean_occ, 2),
                "mean_final_misplaced": round(mean_mis, 2),
                "misplace_density": round(mean_mis / max(1.0, mean_occ), 3),
            }
        )

    return ExperimentSummary(
        scenario=scenario,
        n_replications=n_replications,
        cycles=total_cycles,
        acc_mean=acc_mean,
        acc_std=acc_std,
        entropy_mean=entropy_mean,
        contam_mean=contam_mean,
        eye_dump_mean=eye_dump_mean,
        latency_mean=latency_mean,
        latency_std=latency_std,
        final_acc_mean=acc_mean[-1],
        final_acc_std=acc_std[-1],
        final_entropy_mean=entropy_mean[-1],
        final_contam_mean=contam_mean[-1],
        overall_latency_mean=statistics.mean(latency_mean),
        rush_latency_mean=statistics.mean(rush_lats),
        calm_latency_mean=statistics.mean(calm_lats),
        t_crit_mean=t_crit_mean,
        collapse_rate=collapse_rate,
        per_mech_acc_final=per_mech_final,
        slot_heatmap_data=heatmap_rows,
        contam_breakdown_t={
            "Home": home_t,
            "Within-Mechanism Drift": within_drift_t,
            "Eye-Level Drawer Dump": eye_dump_t,
            "Other Cross-Tier Dump": other_contam_t,
        },
    )
