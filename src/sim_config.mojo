from std.collections import List
from std.pathlib import Path
from std.python import Python, PythonObject


comptime MECH_SPLIT_DRAWER_EYE = 0
comptime MECH_SPLIT_DRAWER_MID = 1
comptime MECH_WIRE_BIN_EYE = 2
comptime MECH_WIRE_BIN_MID = 3
comptime MECH_PEGBOARD_OVH = 4
comptime NUM_MECHS = 5


def mech_id_to_name(mech_id: Int) -> String:
    if mech_id == MECH_SPLIT_DRAWER_EYE:
        return "split_drawer_eye"
    if mech_id == MECH_SPLIT_DRAWER_MID:
        return "split_drawer_mid"
    if mech_id == MECH_WIRE_BIN_EYE:
        return "wire_bin_eye"
    if mech_id == MECH_WIRE_BIN_MID:
        return "wire_bin_mid"
    return "pegboard_ovh"


def mech_name_to_id(name: String) raises -> Int:
    if name == "split_drawer_eye":
        return MECH_SPLIT_DRAWER_EYE
    if name == "split_drawer_mid":
        return MECH_SPLIT_DRAWER_MID
    if name == "wire_bin_eye":
        return MECH_WIRE_BIN_EYE
    if name == "wire_bin_mid":
        return MECH_WIRE_BIN_MID
    if name == "pegboard_ovh":
        return MECH_PEGBOARD_OVH
    raise Error(t"Unknown mechanism name: {name}")


def str_in_list(needle: String, haystack: List[String]) -> Bool:
    for item in haystack:
        if item == needle:
            return True
    return False


struct ShelfGeometryConfig(Copyable, Movable):
    """Physical wire shelf dimensions (cm) and slot parameters."""

    var shelf_width_cm: Float64
    var shelf_depth_cm: Float64
    var shelf_row_height_cm: Float64
    var eye_level_height_cm: Float64
    var mid_level_height_cm: Float64
    var bottom_level_height_cm: Float64
    var top_overhead_height_cm: Float64

    var wire_bins_per_full_row: Int
    var drawers_per_full_row: Int
    var drawers_per_half_bay: Int
    var drawer_outer_width_cm: Float64
    var drawer_outer_depth_cm: Float64
    var drawer_outer_height_cm: Float64
    var drawers_9in_per_full_row: Int
    var drawers_9in_per_half_bay: Int
    var drawer_9in_outer_width_cm: Float64
    var drawer_9in_inner_width_cm: Float64
    var wire_bin_bay_width_cm: Float64
    var wire_bin_bay_depth_cm: Float64
    var wire_bin_bay_height_cm: Float64
    var wire_bin_subzones: Int
    var wire_bin_subzone_hard_cap: Int

    var drawer_count: Int
    var partitions_per_drawer: Int
    var drawer_inner_width_cm: Float64
    var drawer_inner_depth_cm: Float64
    var drawer_inner_height_cm: Float64
    var partition_divider_thickness_cm: Float64

    var wire_bin_width_cm: Float64
    var wire_bin_depth_cm: Float64
    var wire_bin_height_cm: Float64
    var wire_bin_max_rolls: Int

    var peg_rows: Int
    var pegs_per_row: Int
    var peg_usable_length_cm: Float64
    var peg_horizontal_spacing_cm: Float64
    var peg_pull_down_drop_cm: Float64
    var peg_max_od_cm: Float64

    def __init__(out self):
        self.shelf_width_cm = 120.0
        self.shelf_depth_cm = 60.0
        self.shelf_row_height_cm = 60.0
        self.eye_level_height_cm = 140.0
        self.mid_level_height_cm = 80.0
        self.bottom_level_height_cm = 25.0
        self.top_overhead_height_cm = 195.0
        self.wire_bins_per_full_row = 2
        self.drawers_per_full_row = 7
        self.drawers_per_half_bay = 4
        self.drawer_outer_width_cm = 15.24
        self.drawer_outer_depth_cm = 60.0
        self.drawer_outer_height_cm = 30.0
        self.drawers_9in_per_full_row = 5
        self.drawers_9in_per_half_bay = 3
        self.drawer_9in_outer_width_cm = 22.86
        self.drawer_9in_inner_width_cm = 21.6
        self.wire_bin_bay_width_cm = 51.4
        self.wire_bin_bay_depth_cm = 60.0
        self.wire_bin_bay_height_cm = 60.0
        self.wire_bin_subzones = 4
        self.wire_bin_subzone_hard_cap = 9
        self.drawer_count = 7
        self.partitions_per_drawer = 4
        self.drawer_inner_width_cm = 14.0
        self.drawer_inner_depth_cm = 58.0
        self.drawer_inner_height_cm = 27.0
        self.partition_divider_thickness_cm = 0.4
        self.wire_bin_width_cm = 25.0
        self.wire_bin_depth_cm = 58.0
        self.wire_bin_height_cm = 58.0
        self.wire_bin_max_rolls = 10
        self.peg_rows = 8
        self.pegs_per_row = 4
        self.peg_usable_length_cm = 25.0
        self.peg_horizontal_spacing_cm = 15.0
        self.peg_pull_down_drop_cm = 35.0
        self.peg_max_od_cm = 15.24

    def partition_width_cm(
        self,
        partitions: Int = -1,
        sub_idx: Int = 0,
        inner_width_cm: Float64 = -1.0,
    ) -> Float64:
        var n = partitions if partitions >= 0 else self.partitions_per_drawer
        var w = inner_width_cm if inner_width_cm > 0.0 else self.drawer_inner_width_cm
        if n == 3 and sub_idx == 0:
            return w
        if n <= 1:
            return w
        return (w - self.partition_divider_thickness_cm) / 2.0


@fieldwise_init
struct MechanismSpec(Copyable, Movable):
    """Physical and human-factors profile for a storage mechanism."""

    var mech_id: Int
    var mechanism_name: String
    var label: String
    var reach_actuation_cost: Float64
    var visibility_closed: Float64
    var visibility_open: Float64
    var clearance_score: Float64
    var base_retrieval_sec: Float64
    var base_return_sec: Float64
    var supports_coreless: Bool
    var is_pull_down: Bool


def build_default_mechanisms() -> List[MechanismSpec]:
    var specs = List[MechanismSpec](capacity=NUM_MECHS)
    specs.append(
        MechanismSpec(
            MECH_SPLIT_DRAWER_EYE,
            "split_drawer_eye",
            "Mid Shelf: Modular Split Drawers",
            0.14,
            0.18,
            0.88,
            0.75,
            3.0,
            2.5,
            True,
            False,
        )
    )
    specs.append(
        MechanismSpec(
            MECH_SPLIT_DRAWER_MID,
            "split_drawer_mid",
            "Btm Shelf: Modular Split Drawers",
            0.38,
            0.15,
            0.78,
            0.76,
            4.3,
            3.5,
            True,
            False,
        )
    )
    specs.append(
        MechanismSpec(
            MECH_WIRE_BIN_EYE,
            "wire_bin_eye",
            "Mid Shelf: Wire Bin",
            0.28,
            0.45,
            0.68,
            0.88,
            4.2,
            3.4,
            True,
            False,
        )
    )
    specs.append(
        MechanismSpec(
            MECH_WIRE_BIN_MID,
            "wire_bin_mid",
            "Btm Shelf: Wire Bin",
            0.42,
            0.40,
            0.62,
            0.88,
            4.9,
            3.9,
            True,
            False,
        )
    )
    specs.append(
        MechanismSpec(
            MECH_PEGBOARD_OVH,
            "pegboard_ovh",
            "Top Shelf: Sliding Peg (32 Pegs)",
            0.22,
            0.84,
            0.94,
            0.62,
            3.6,
            3.3,
            False,
            True,
        )
    )
    return specs^


struct RushHourConfig(Copyable, Movable):
    """Shift schedule and 8:00 AM - 10:00 AM rush window demand parameters."""

    var shift_cycles: Int
    var num_shifts: Int
    var rush_job_share: Float64
    var rush_tau_lo: Float64
    var rush_tau_hi: Float64
    var calm_tau_lo: Float64
    var calm_tau_hi: Float64
    var rush_sku_boost: Float64
    var same_product_co_retrieval: Float64
    var rush_priority_categories: List[String]
    var rush_priority_products: List[String]

    def __init__(out self):
        self.shift_cycles = 50
        self.num_shifts = 20
        self.rush_job_share = 0.45
        self.rush_tau_lo = 0.75
        self.rush_tau_hi = 1.00
        self.calm_tau_lo = 0.10
        self.calm_tau_hi = 0.35
        self.rush_sku_boost = 3.6
        self.same_product_co_retrieval = 0.90
        self.rush_priority_categories = ["Bites", "Salad", "Breakfast Cookies"]
        self.rush_priority_products = [
            "Peppermint Clean Bites",
            "Sunshine Cookie Dough Bites",
            "Lemon and Rosemary Clean Bites",
            "Choco Chip Pumpkin Pie Clean Bites",
            "Better Buckeyes Bites",
            "Cherry Bomb Clean Bites",
            "Brain Boost Brownie Bites",
            "PB & J Clean Bites",
            "Cookie Dough Clean Bites",
            "Orange Lavendar Clean Bites",
            "Grain Free Brownie Bites",
            "Carrot Cake Clean Bites",
            "Peppermint Kisses",
            "Chia Leah Fresh Salad",
            "Pumpkin & Chocolate Chip Breakfast Cookie",
            "Original Breakfast Cookie",
            "Zucchini Breakfast Cookie",
            "Sunny Chocolate Chip Breakfast Cookie",
        ]

    def total_cycles(self) -> Int:
        return self.shift_cycles * self.num_shifts


struct BehavioralConfig(Copyable, Movable):
    """Human-factors parameters governing search, misplacement, and cleanup."""

    var base_error_rate: Float64
    var base_correction_rate: Float64
    var alpha_rush: Float64
    var alpha_crowd: Float64
    var alpha_blur: Float64
    var alpha_ergo: Float64
    var unlatch_rush_prob: Float64
    var occlusion_friction_penalty: Float64
    var search_miss_penalty_sec: Float64
    var usability_accuracy_threshold: Float64
    var bin_dump_error_penalty: Float64
    var bin_dump_routing_weight: Float64
    var mech_diversity_penalty: Float64
    var bin_drawer_same_row_penalty: Float64
    var drawer_size_mix_penalty: Float64
    var multi_sku_slot_penalty: Float64
    var slot_deficit_penalty: Float64
    var row_width_mismatch_penalty: Float64

    def __init__(out self):
        self.base_error_rate = 0.028
        self.base_correction_rate = 0.24
        self.alpha_rush = 2.8
        self.alpha_crowd = 3.2
        self.alpha_blur = 1.75
        self.alpha_ergo = 2.3
        self.unlatch_rush_prob = 0.38
        self.occlusion_friction_penalty = 0.35
        self.search_miss_penalty_sec = 14.0
        self.usability_accuracy_threshold = 0.80
        self.bin_dump_error_penalty = 0.11
        self.bin_dump_routing_weight = 0.30
        self.mech_diversity_penalty = 0.10
        self.bin_drawer_same_row_penalty = 0.16
        self.drawer_size_mix_penalty = 0.0
        self.multi_sku_slot_penalty = 0.16
        self.slot_deficit_penalty = 1.20
        self.row_width_mismatch_penalty = 0.25


struct ExecutionConfig(Copyable, Movable):
    var benchmark_replications: Int
    var sweep_replications: Int
    var search_replications: Int
    var composite_score_latency_weight: Float64

    def __init__(out self):
        self.benchmark_replications = 50
        self.sweep_replications = 25
        self.search_replications = 6
        self.composite_score_latency_weight = 0.85


struct AppConfig(Copyable, Movable):
    var geometry: ShelfGeometryConfig
    var rush: RushHourConfig
    var behavior: BehavioralConfig
    var execution: ExecutionConfig
    var mechanisms: List[MechanismSpec]

    def __init__(out self):
        self.geometry = ShelfGeometryConfig()
        self.rush = RushHourConfig()
        self.behavior = BehavioralConfig()
        self.execution = ExecutionConfig()
        self.mechanisms = build_default_mechanisms()


def _py_has(obj: PythonObject, key: String) raises -> Bool:
    return Bool(obj.__contains__(key))


def _py_float(obj: PythonObject, key: String, default: Float64) raises -> Float64:
    if _py_has(obj, key):
        return Float64(py=obj[key])
    return default


def _py_int(obj: PythonObject, key: String, default: Int) raises -> Int:
    if _py_has(obj, key):
        return Int(py=obj[key])
    return default


def _py_bool(obj: PythonObject, key: String, default: Bool) raises -> Bool:
    if _py_has(obj, key):
        return Bool(py=obj[key])
    return default


def _py_str(obj: PythonObject, key: String, default: String) raises -> String:
    if _py_has(obj, key):
        return String(py=obj[key])
    return default


def _merge_py_dict(dst: PythonObject, src: PythonObject) raises:
    var builtins = Python.import_module("builtins")
    for k in src:
        var k_str = String(py=k)
        var v = src[k]
        if Bool(py=dst.__contains__(k)) and Bool(py=builtins.isinstance(dst[k], builtins.dict)) and Bool(py=builtins.isinstance(v, builtins.dict)):
            _merge_py_dict(dst[k], v)
        else:
            dst[k] = v


def load_app_config(geom_path: String, cfg_path: String) raises -> AppConfig:
    """Load and merge shelf_geometry.yaml and shelf_config.yaml via PyYAML."""
    var yaml_mod = Python.import_module("yaml")
    var json_mod = Python.import_module("json")
    var builtins = Python.import_module("builtins")

    var data = Python.dict()
    if Path(geom_path).exists():
        var g_text = Path(geom_path).read_text()
        var g_obj = yaml_mod.safe_load(g_text)
        if Bool(py=builtins.isinstance(g_obj, builtins.dict)):
            _merge_py_dict(data, g_obj)

    if Path(cfg_path).exists():
        var c_text = Path(cfg_path).read_text()
        var c_obj = json_mod.loads(c_text) if cfg_path.endswith(".json") else yaml_mod.safe_load(c_text)
        if Bool(py=builtins.isinstance(c_obj, builtins.dict)):
            _merge_py_dict(data, c_obj)

    var cfg = AppConfig()

    if _py_has(data, "geometry"):
        var g = data["geometry"]
        cfg.geometry.shelf_width_cm = _py_float(g, "shelf_width_cm", cfg.geometry.shelf_width_cm)
        cfg.geometry.shelf_depth_cm = _py_float(g, "shelf_depth_cm", cfg.geometry.shelf_depth_cm)
        cfg.geometry.shelf_row_height_cm = _py_float(g, "shelf_row_height_cm", cfg.geometry.shelf_row_height_cm)
        cfg.geometry.eye_level_height_cm = _py_float(g, "eye_level_height_cm", cfg.geometry.eye_level_height_cm)
        cfg.geometry.mid_level_height_cm = _py_float(g, "mid_level_height_cm", cfg.geometry.mid_level_height_cm)
        cfg.geometry.bottom_level_height_cm = _py_float(g, "bottom_level_height_cm", cfg.geometry.bottom_level_height_cm)
        cfg.geometry.top_overhead_height_cm = _py_float(g, "top_overhead_height_cm", cfg.geometry.top_overhead_height_cm)
        cfg.geometry.wire_bins_per_full_row = _py_int(g, "wire_bins_per_full_row", cfg.geometry.wire_bins_per_full_row)
        cfg.geometry.drawers_per_full_row = _py_int(g, "drawers_per_full_row", cfg.geometry.drawers_per_full_row)
        cfg.geometry.drawers_per_half_bay = _py_int(g, "drawers_per_half_bay", cfg.geometry.drawers_per_half_bay)
        cfg.geometry.drawer_outer_width_cm = _py_float(g, "drawer_outer_width_cm", cfg.geometry.drawer_outer_width_cm)
        cfg.geometry.drawer_outer_depth_cm = _py_float(g, "drawer_outer_depth_cm", cfg.geometry.drawer_outer_depth_cm)
        cfg.geometry.drawer_outer_height_cm = _py_float(g, "drawer_outer_height_cm", cfg.geometry.drawer_outer_height_cm)
        cfg.geometry.drawers_9in_per_full_row = _py_int(g, "drawers_9in_per_full_row", cfg.geometry.drawers_9in_per_full_row)
        cfg.geometry.drawers_9in_per_half_bay = _py_int(g, "drawers_9in_per_half_bay", cfg.geometry.drawers_9in_per_half_bay)
        cfg.geometry.drawer_9in_outer_width_cm = _py_float(g, "drawer_9in_outer_width_cm", cfg.geometry.drawer_9in_outer_width_cm)
        cfg.geometry.drawer_9in_inner_width_cm = _py_float(g, "drawer_9in_inner_width_cm", cfg.geometry.drawer_9in_inner_width_cm)
        cfg.geometry.wire_bin_bay_width_cm = _py_float(g, "wire_bin_bay_width_cm", cfg.geometry.wire_bin_bay_width_cm)
        cfg.geometry.wire_bin_bay_depth_cm = _py_float(g, "wire_bin_bay_depth_cm", cfg.geometry.wire_bin_bay_depth_cm)
        cfg.geometry.wire_bin_bay_height_cm = _py_float(g, "wire_bin_bay_height_cm", cfg.geometry.wire_bin_bay_height_cm)
        cfg.geometry.wire_bin_subzones = _py_int(g, "wire_bin_subzones", cfg.geometry.wire_bin_subzones)
        cfg.geometry.wire_bin_subzone_hard_cap = _py_int(g, "wire_bin_subzone_hard_cap", cfg.geometry.wire_bin_subzone_hard_cap)
        cfg.geometry.drawer_count = _py_int(g, "drawer_count", cfg.geometry.drawer_count)
        cfg.geometry.partitions_per_drawer = _py_int(g, "partitions_per_drawer", cfg.geometry.partitions_per_drawer)
        cfg.geometry.drawer_inner_width_cm = _py_float(g, "drawer_inner_width_cm", cfg.geometry.drawer_inner_width_cm)
        cfg.geometry.drawer_inner_depth_cm = _py_float(g, "drawer_inner_depth_cm", cfg.geometry.drawer_inner_depth_cm)
        cfg.geometry.drawer_inner_height_cm = _py_float(g, "drawer_inner_height_cm", cfg.geometry.drawer_inner_height_cm)
        cfg.geometry.partition_divider_thickness_cm = _py_float(g, "partition_divider_thickness_cm", cfg.geometry.partition_divider_thickness_cm)
        cfg.geometry.wire_bin_width_cm = _py_float(g, "wire_bin_width_cm", cfg.geometry.wire_bin_width_cm)
        cfg.geometry.wire_bin_depth_cm = _py_float(g, "wire_bin_depth_cm", cfg.geometry.wire_bin_depth_cm)
        cfg.geometry.wire_bin_height_cm = _py_float(g, "wire_bin_height_cm", cfg.geometry.wire_bin_height_cm)
        cfg.geometry.wire_bin_max_rolls = _py_int(g, "wire_bin_max_rolls", cfg.geometry.wire_bin_max_rolls)
        cfg.geometry.peg_rows = _py_int(g, "peg_rows", cfg.geometry.peg_rows)
        cfg.geometry.pegs_per_row = _py_int(g, "pegs_per_row", cfg.geometry.pegs_per_row)
        cfg.geometry.peg_usable_length_cm = _py_float(g, "peg_usable_length_cm", cfg.geometry.peg_usable_length_cm)
        cfg.geometry.peg_horizontal_spacing_cm = _py_float(g, "peg_horizontal_spacing_cm", cfg.geometry.peg_horizontal_spacing_cm)
        cfg.geometry.peg_pull_down_drop_cm = _py_float(g, "peg_pull_down_drop_cm", cfg.geometry.peg_pull_down_drop_cm)
        cfg.geometry.peg_max_od_cm = _py_float(g, "peg_max_od_cm", cfg.geometry.peg_max_od_cm)

    if _py_has(data, "shift_schedule"):
        var ss = data["shift_schedule"]
        cfg.rush.shift_cycles = _py_int(ss, "shift_cycles", cfg.rush.shift_cycles)
        cfg.rush.num_shifts = _py_int(ss, "num_shifts", cfg.rush.num_shifts)
        if _py_has(ss, "rush_priority_categories"):
            var cats = List[String]()
            for c_item in ss["rush_priority_categories"]:
                cats.append(String(py=c_item))
            if len(cats) > 0:
                cfg.rush.rush_priority_categories = cats^
        if _py_has(ss, "rush_priority_products"):
            var prods = List[String]()
            for p_item in ss["rush_priority_products"]:
                prods.append(String(py=p_item))
            if len(prods) > 0:
                cfg.rush.rush_priority_products = prods^

    if _py_has(data, "demand_weights"):
        var dw = data["demand_weights"]
        cfg.rush.rush_job_share = _py_float(dw, "rush_job_share", cfg.rush.rush_job_share)
        cfg.rush.rush_sku_boost = _py_float(dw, "rush_sku_boost", cfg.rush.rush_sku_boost)
        cfg.rush.same_product_co_retrieval = _py_float(dw, "same_product_co_retrieval", cfg.rush.same_product_co_retrieval)
        if _py_has(dw, "rush_tau_range"):
            var r_rng = dw["rush_tau_range"]
            cfg.rush.rush_tau_lo = Float64(py=r_rng[0])
            cfg.rush.rush_tau_hi = Float64(py=r_rng[1])
        if _py_has(dw, "calm_tau_range"):
            var c_rng = dw["calm_tau_range"]
            cfg.rush.calm_tau_lo = Float64(py=c_rng[0])
            cfg.rush.calm_tau_hi = Float64(py=c_rng[1])

    if _py_has(data, "behavioral_weights"):
        var bw = data["behavioral_weights"]
        cfg.behavior.base_error_rate = _py_float(bw, "base_error_rate", cfg.behavior.base_error_rate)
        cfg.behavior.base_correction_rate = _py_float(bw, "base_correction_rate", cfg.behavior.base_correction_rate)
        cfg.behavior.alpha_rush = _py_float(bw, "alpha_rush", cfg.behavior.alpha_rush)
        cfg.behavior.alpha_crowd = _py_float(bw, "alpha_crowd", cfg.behavior.alpha_crowd)
        cfg.behavior.alpha_blur = _py_float(bw, "alpha_blur", cfg.behavior.alpha_blur)
        cfg.behavior.alpha_ergo = _py_float(bw, "alpha_ergo", cfg.behavior.alpha_ergo)
        cfg.behavior.unlatch_rush_prob = _py_float(bw, "unlatch_rush_prob", cfg.behavior.unlatch_rush_prob)
        cfg.behavior.occlusion_friction_penalty = _py_float(bw, "occlusion_friction_penalty", cfg.behavior.occlusion_friction_penalty)
        cfg.behavior.search_miss_penalty_sec = _py_float(bw, "search_miss_penalty_sec", cfg.behavior.search_miss_penalty_sec)
        cfg.behavior.usability_accuracy_threshold = _py_float(bw, "usability_accuracy_threshold", cfg.behavior.usability_accuracy_threshold)
        cfg.behavior.bin_dump_error_penalty = _py_float(bw, "bin_dump_error_penalty", cfg.behavior.bin_dump_error_penalty)
        cfg.behavior.bin_dump_routing_weight = _py_float(bw, "bin_dump_routing_weight", cfg.behavior.bin_dump_routing_weight)
        cfg.behavior.mech_diversity_penalty = _py_float(bw, "mech_diversity_penalty", cfg.behavior.mech_diversity_penalty)
        cfg.behavior.bin_drawer_same_row_penalty = _py_float(bw, "bin_drawer_same_row_penalty", cfg.behavior.bin_drawer_same_row_penalty)
        cfg.behavior.drawer_size_mix_penalty = _py_float(bw, "drawer_size_mix_penalty", cfg.behavior.drawer_size_mix_penalty)
        cfg.behavior.multi_sku_slot_penalty = _py_float(bw, "multi_sku_slot_penalty", cfg.behavior.multi_sku_slot_penalty)
        cfg.behavior.slot_deficit_penalty = _py_float(bw, "slot_deficit_penalty", cfg.behavior.slot_deficit_penalty)
        cfg.behavior.row_width_mismatch_penalty = _py_float(bw, "row_width_mismatch_penalty", cfg.behavior.row_width_mismatch_penalty)

    if _py_has(data, "execution"):
        var ex = data["execution"]
        cfg.execution.benchmark_replications = _py_int(ex, "benchmark_replications", cfg.execution.benchmark_replications)
        cfg.execution.sweep_replications = _py_int(ex, "sweep_replications", cfg.execution.sweep_replications)
        cfg.execution.search_replications = _py_int(ex, "search_replications", cfg.execution.search_replications)
        cfg.execution.composite_score_latency_weight = _py_float(ex, "composite_score_latency_weight", cfg.execution.composite_score_latency_weight)

    if _py_has(data, "mechanisms"):
        var ms = data["mechanisms"]
        for m_idx in range(NUM_MECHS):
            var m_name = cfg.mechanisms[m_idx].mechanism_name
            if _py_has(ms, m_name):
                var patch = ms[m_name]
                cfg.mechanisms[m_idx].label = _py_str(patch, "label", cfg.mechanisms[m_idx].label)
                cfg.mechanisms[m_idx].reach_actuation_cost = _py_float(patch, "reach_actuation_cost", cfg.mechanisms[m_idx].reach_actuation_cost)
                cfg.mechanisms[m_idx].visibility_closed = _py_float(patch, "visibility_closed", cfg.mechanisms[m_idx].visibility_closed)
                cfg.mechanisms[m_idx].visibility_open = _py_float(patch, "visibility_open", cfg.mechanisms[m_idx].visibility_open)
                cfg.mechanisms[m_idx].clearance_score = _py_float(patch, "clearance_score", cfg.mechanisms[m_idx].clearance_score)
                cfg.mechanisms[m_idx].base_retrieval_sec = _py_float(patch, "base_retrieval_sec", cfg.mechanisms[m_idx].base_retrieval_sec)
                cfg.mechanisms[m_idx].base_return_sec = _py_float(patch, "base_return_sec", cfg.mechanisms[m_idx].base_return_sec)
                cfg.mechanisms[m_idx].supports_coreless = _py_bool(patch, "supports_coreless", cfg.mechanisms[m_idx].supports_coreless)
                cfg.mechanisms[m_idx].is_pull_down = _py_bool(patch, "is_pull_down", cfg.mechanisms[m_idx].is_pull_down)

    return cfg^


def _is_two_bin_row(r: String) -> Bool:
    return (
        r == "2B"
        or r == "1D6_2B"
        or r == "1B_1D6_1B"
        or r == "2B_1D6"
        or r == "1D_2B"
        or r == "1B_1D_1B"
        or r == "2B_1D"
    )


def _is_one_bin_prefix(r: String) -> Bool:
    return r.startswith("1B_") and not _is_two_bin_row(r)


def count_shelf_complexity(r0: String, r1: String, r2: String) -> Tuple[Int, Int]:
    """Return (n_bins, extra_styles) across {peg, 4P_drawer, 3P_drawer, wire_bin}."""
    var rows = [r0, r1, r2]
    var n_bins = 0
    var has_peg = False
    var has_4p = False
    var has_3p = False
    var has_bin = False

    for r in rows:
        if _is_two_bin_row(r):
            n_bins += 2
            has_bin = True
        elif _is_one_bin_prefix(r):
            n_bins += 1
            has_bin = True

        if "PEG" in r:
            has_peg = True
        if (
            "4P" in r
            or r == "1D6_2B"
            or r == "1B_1D6_1B"
            or r == "2B_1D6"
            or r == "1D_2B"
            or r == "1B_1D_1B"
            or r == "2B_1D"
        ):
            has_4p = True
        if "3P" in r:
            has_3p = True

    var n_styles = (
        (1 if has_peg else 0)
        + (1 if has_4p else 0)
        + (1 if has_3p else 0)
        + (1 if has_bin else 0)
    )
    var extra = n_styles - 2
    if extra < 0:
        extra = 0
    return (n_bins, extra)


def has_mixed_drawer_sizes(r0: String, r1: String, r2: String) -> Bool:
    var has_6 = False
    var has_9 = False
    var rows = [r0, r1, r2]
    for r in rows:
        if (
            "D6" in r
            or r == "7D_4P"
            or r == "7D_3P"
            or r == "1B_4D_4P"
            or r == "1B_4D_3P"
            or r == "1D_2B"
            or r == "1B_1D_1B"
            or r == "2B_1D"
        ):
            has_6 = True
        if "D9" in r:
            has_9 = True
    return has_6 and has_9


def count_mixed_bin_drawer_rows(r0: String, r1: String, r2: String) -> Int:
    var count = 0
    var rows = [r0, r1, r2]
    for r in rows:
        if "B" in r and "D" in r:
            count += 1
    return count


def has_row_width_mismatch(r0: String, r1: String, r2: String) -> Bool:
    var has_42 = False
    var has_45 = False
    var rows = [r0, r1, r2]
    for r in rows:
        if r == "7D6_4P" or r == "7D6_3P" or r == "7D_4P" or r == "7D_3P" or r == "4D4P_3D3P":
            has_42 = True
        elif r == "5D9_4P" or r == "5D9_3P" or r == "3D9_3D6_4P":
            has_45 = True
    return has_42 and has_45


def count_drawer_slots(r0: String, r1: String, r2: String) -> Int:
    var rows = [r0, r1, r2]
    var total = 0
    for r in rows:
        if r == "7D6_4P" or r == "7D_4P":
            total += 28
        elif r == "7D6_3P" or r == "7D_3P":
            total += 21
        elif r == "4D4P_3D3P":
            total += 25
        elif r == "5D9_4P":
            total += 20
        elif r == "5D9_3P":
            total += 15
        elif r == "3D9_3D6_4P":
            total += 24
        elif r == "1B_4D6_4P" or r == "1B_4D_4P":
            total += 16
        elif r == "1B_4D6_3P" or r == "1B_4D_3P":
            total += 12
        elif r == "1B_3D9_4P":
            total += 12
        elif r == "1B_3D9_3P":
            total += 9
        elif r == "1D6_2B" or r == "1B_1D6_1B" or r == "2B_1D6" or r == "1D_2B" or r == "1B_1D_1B" or r == "2B_1D":
            total += 4
    return total


@fieldwise_init
struct ExperimentScenario(Copyable, Movable):
    """Parameter vector specifying a single simulation experiment."""

    var exp_id: String
    var name: String
    var topology: String
    var strategy: String
    var drawer_partitions: Int
    var cue_salience: Float64
    var target_fill_ratio: Float64
    var description: String
    var row0: String
    var row1: String
    var row2: String
    var geometry: ShelfGeometryConfig
    var rush: RushHourConfig
    var behavior: BehavioralConfig
    var abc_colocation_mode: String
    var overflow_mode: String
    var spatial_order_mode: String
    var layout_seed: Int


@fieldwise_init
struct RollVariantSKU(Copyable, Movable):
    var variant_id: Int
    var sku_code: String
    var product_name: String
    var variant_code: String
    var outer_diameter_cm: Float64
    var roll_height_cm: Float64
    var quantity: Int
    var category: String
    var notes: String
    var has_core: Bool
    var is_rush_priority: Bool
    var visual_similarity_group: String
    var baseline_freq: Float64
    var rush_freq: Float64


@fieldwise_init
struct RollItem(Copyable, Movable):
    var roll_id: Int
    var variant_id: Int
    var sku_code: String
    var product_name: String
    var variant_code: String
    var copy_index: Int
    var outer_diameter_cm: Float64
    var roll_height_cm: Float64
    var has_core: Bool
    var category: String
    var is_rush_priority: Bool
    var visual_similarity_group: String
    var home_mech_id: Int
    var home_mechanism: String
    var home_slot_idx: Int
    var home_slot: String
    var current_mech_id: Int
    var current_mechanism: String
    var current_slot_idx: Int
    var current_slot: String
    var misplaced_history_count: Int


struct InventoryData(Copyable, Movable):
    var variants: List[RollVariantSKU]
    var rolls: List[RollItem]
    var freq_calm: List[Float64]
    var freq_rush: List[Float64]
    var co_matrix: List[Float64]
    var num_variants: Int
    var high_assoc_siblings: List[List[Int]]

    def __init__(
        out self,
        var variants: List[RollVariantSKU],
        var rolls: List[RollItem],
        var freq_calm: List[Float64],
        var freq_rush: List[Float64],
        var co_matrix: List[Float64],
        num_variants: Int,
        var high_assoc_siblings: List[List[Int]],
    ):
        self.variants = variants^
        self.rolls = rolls^
        self.freq_calm = freq_calm^
        self.freq_rush = freq_rush^
        self.co_matrix = co_matrix^
        self.num_variants = num_variants
        self.high_assoc_siblings = high_assoc_siblings^

    def co(self, i: Int, j: Int) -> Float64:
        return self.co_matrix[i * self.num_variants + j]


def _split_csv_row(line: String) -> List[String]:
    var fields = List[String]()
    var cur = String("")
    var in_quote = False
    for ch in line.codepoint_slices():
        var s = String(ch)
        if s == "\"":
            in_quote = not in_quote
        elif s == "," and not in_quote:
            fields.append(cur)
            cur = String("")
        else:
            cur += s
    fields.append(cur)
    return fields^


def load_roll_inventory(
    csv_path: String, rush_cfg: RushHourConfig
) raises -> InventoryData:
    """Parse roll-data.csv and construct SKU variants, 128 rolls, f, and C."""
    var content = Path(csv_path).read_text()
    var lines = content.splitlines()
    if len(lines) < 2:
        raise Error(t"Empty CSV at {csv_path}")

    var variants = List[RollVariantSKU]()
    var rolls = List[RollItem]()
    var vid = 0
    var rid = 0

    for line_idx in range(1, len(lines)):
        var raw = String(lines[line_idx])
        if raw.byte_length() == 0:
            continue
        var cols = _split_csv_row(raw)
        if len(cols) < 6:
            continue
        var label = String(cols[0].strip())
        if label.byte_length() == 0 or label.startswith("*"):
            continue
        var var_code = String(cols[1].strip())
        if var_code.byte_length() == 0:
            var_code = "A"
        var od = Float64(String(cols[2].strip()))
        var ht = Float64(String(cols[3].strip()))
        var qty = Int(String(cols[4].strip()))
        var category = String(cols[5].strip())
        if category.byte_length() == 0:
            category = "Other"
        var notes = String(cols[6].strip()) if len(cols) > 6 else String("")
        var has_core = "no inner roll" not in notes.lower()
        var is_prio = (
            str_in_list(label, rush_cfg.rush_priority_products)
            or category == "Bites"
            or category == "Salad"
            or ("breakfast cookie" in label.lower())
        )
        var height_band = "short" if ht <= 10.0 else "tall"
        var vis_group = String(t"{category}::{height_band}")
        var sku_code = String(t"{label} [{var_code}]")

        variants.append(
            RollVariantSKU(
                vid,
                sku_code,
                label,
                var_code,
                od,
                ht,
                qty,
                category,
                notes,
                has_core,
                is_prio,
                vis_group,
                0.0,
                0.0,
            )
        )
        for c_idx in range(qty):
            rolls.append(
                RollItem(
                    rid,
                    vid,
                    sku_code,
                    label,
                    var_code,
                    c_idx,
                    od,
                    ht,
                    has_core,
                    category,
                    is_prio,
                    vis_group,
                    -1,
                    "",
                    -1,
                    "",
                    -1,
                    "",
                    -1,
                    "",
                    0,
                )
            )
            rid += 1
        vid += 1

    var n = len(variants)
    var raw_calm = List[Float64](capacity=n)
    var raw_rush = List[Float64](capacity=n)
    var sum_calm: Float64 = 0.0
    var sum_rush: Float64 = 0.0

    for i in range(n):
        var base_w = 0.6 + 0.55 * Float64(variants[i].quantity)
        if variants[i].category == "Ingredient sticker":
            base_w += 0.8
        raw_calm.append(base_w)
        var r_mult = rush_cfg.rush_sku_boost if variants[i].is_rush_priority else 0.70
        var rush_w = base_w * r_mult
        raw_rush.append(rush_w)
        sum_calm += base_w
        sum_rush += rush_w

    var freq_calm = List[Float64](capacity=n)
    var freq_rush = List[Float64](capacity=n)
    for i in range(n):
        var fc = raw_calm[i] / sum_calm
        var fr = raw_rush[i] / sum_rush
        freq_calm.append(fc)
        freq_rush.append(fr)
        variants[i].baseline_freq = fc
        variants[i].rush_freq = fr

    var co_matrix = List[Float64](capacity=n * n)
    for _ in range(n * n):
        co_matrix.append(0.0)

    for i in range(n):
        for j in range(i + 1, n):
            var w: Float64 = 0.01
            if variants[i].product_name == variants[j].product_name:
                w = rush_cfg.same_product_co_retrieval
            elif (
                (variants[i].category == "Chocolate bark" and variants[j].category == "Ingredient sticker")
                or (variants[i].category == "Ingredient sticker" and variants[j].category == "Chocolate bark")
            ):
                var pep_i = "Peppermint" in variants[i].product_name
                var pep_j = "Peppermint" in variants[j].product_name
                if pep_i and pep_j:
                    w = 0.78
                elif (not pep_i) and (not pep_j):
                    w = 0.52
                else:
                    w = 0.15
            elif variants[i].category == variants[j].category:
                w = 0.24 if (variants[i].is_rush_priority and variants[j].is_rush_priority) else 0.10

            co_matrix[i * n + j] = w
            co_matrix[j * n + i] = w

    var high_assoc = List[List[Int]](capacity=n)
    for i in range(n):
        var sibs = List[Int]()
        for j in range(n):
            if j != i and co_matrix[i * n + j] >= 0.85:
                sibs.append(j)
        high_assoc.append(sibs^)

    return InventoryData(
        variants^,
        rolls^,
        freq_calm^,
        freq_rush^,
        co_matrix^,
        n,
        high_assoc^,
    )


def build_default_experiments(cfg: AppConfig) -> List[ExperimentScenario]:
    var out = List[ExperimentScenario](capacity=8)
    out.append(
        ExperimentScenario(
            "ORI-258",
            "Top: Peg | Mid: 7D6_4P | Btm: 2B (90% fill, low cue)",
            "M_modular",
            "B_kit",
            4,
            0.20,
            0.90,
            "Top: Peg | Mid: 7D6_4P | Btm: 2B at 90% fill with low label clarity and height-split COI.",
            "7D6_4P",
            "2B",
            "SLIDING_PEG",
            cfg.geometry.copy(),
            cfg.rush.copy(),
            cfg.behavior.copy(),
            "cross_tier_height",
            "all_colocated",
            "random_perm",
            1491,
        )
    )
    out.append(
        ExperimentScenario(
            "ORI-255",
            "Top: Peg | Mid: 7D6_4P | Btm: 2B (height split)",
            "M_modular",
            "C_hybrid",
            4,
            0.90,
            0.80,
            "Top: Peg | Mid: 7D6_4P | Btm: 2B at 80% fill with clear labels and height-split grouping.",
            "7D6_4P",
            "2B",
            "SLIDING_PEG",
            cfg.geometry.copy(),
            cfg.rush.copy(),
            cfg.behavior.copy(),
            "cross_tier_height",
            "hybrid_reserve",
            "center_out",
            1470,
        )
    )
    out.append(
        ExperimentScenario(
            "ORI-099",
            "Top: Peg | Mid: 7D6_3P | Btm: 1B_4D6_3P (low cue)",
            "M_modular",
            "C_hybrid",
            3,
            0.20,
            0.80,
            "Top: Peg | Mid: 7D6_3P | Btm: 1B_4D6_3P at 80% fill with low label clarity and random placement.",
            "7D6_3P",
            "1B_4D6_3P",
            "SLIDING_PEG",
            cfg.geometry.copy(),
            cfg.rush.copy(),
            cfg.behavior.copy(),
            "adjacent_partition",
            "hybrid_reserve",
            "random_perm",
            1250,
        )
    )
    out.append(
        ExperimentScenario(
            "ORI-085",
            "Top: Peg | Mid: 7D6_3P | Btm: 1B_4D6_3P",
            "M_modular",
            "C_hybrid",
            3,
            0.90,
            0.80,
            "Top: Peg | Mid: 7D6_3P | Btm: 1B_4D6_3P at 80% fill with clear labels and adjacent A/B/C grouping.",
            "7D6_3P",
            "1B_4D6_3P",
            "SLIDING_PEG",
            cfg.geometry.copy(),
            cfg.rush.copy(),
            cfg.behavior.copy(),
            "adjacent_partition",
            "hybrid_reserve",
            "center_out",
            1152,
        )
    )
    out.append(
        ExperimentScenario(
            "ORI-245",
            "Top: Peg | Mid: 7D6_4P | Btm: 1D6_2B",
            "M_modular",
            "C_hybrid",
            4,
            0.90,
            0.80,
            "Top: Peg | Mid: 7D6_4P | Btm: 1D6_2B (1 6in drawer on left + 2 wire bins).",
            "7D6_4P",
            "1D6_2B",
            "SLIDING_PEG",
            cfg.geometry.copy(),
            cfg.rush.copy(),
            cfg.behavior.copy(),
            "adjacent_partition",
            "hybrid_reserve",
            "center_out",
            1400,
        )
    )
    out.append(
        ExperimentScenario(
            "ORI-121",
            "Top: Peg | Mid: 5D9_4P | Btm: 5D9_4P",
            "M_modular",
            "C_hybrid",
            4,
            0.90,
            0.80,
            "Top: Peg | Mid: 5D9_4P | Btm: 5D9_4P (all 9in outer drawers, bin-free).",
            "5D9_4P",
            "5D9_4P",
            "SLIDING_PEG",
            cfg.geometry.copy(),
            cfg.rush.copy(),
            cfg.behavior.copy(),
            "adjacent_partition",
            "hybrid_reserve",
            "center_out",
            1407,
        )
    )
    out.append(
        ExperimentScenario(
            "ORI-325",
            "Top: Peg | Mid: 1B_4D6_4P | Btm: 2B",
            "M_modular",
            "C_hybrid",
            4,
            0.90,
            0.80,
            "Top: Peg | Mid: 1B_4D6_4P | Btm: 2B at 80% fill.",
            "1B_4D6_4P",
            "2B",
            "SLIDING_PEG",
            cfg.geometry.copy(),
            cfg.rush.copy(),
            cfg.behavior.copy(),
            "adjacent_partition",
            "hybrid_reserve",
            "center_out",
            1524,
        )
    )
    out.append(
        ExperimentScenario(
            "ORI-021",
            "Top: Peg | Mid: 7D6_4P | Btm: 1B_4D6_3P",
            "M_modular",
            "C_hybrid",
            4,
            0.90,
            0.80,
            "Top: Peg | Mid: 7D6_4P | Btm: 1B_4D6_3P at 80% fill.",
            "7D6_4P",
            "1B_4D6_3P",
            "SLIDING_PEG",
            cfg.geometry.copy(),
            cfg.rush.copy(),
            cfg.behavior.copy(),
            "adjacent_partition",
            "hybrid_reserve",
            "center_out",
            1040,
        )
    )
    return out^
