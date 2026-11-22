"""Parametric configuration and CSV inventory loader for the wire shelf label storage model.

All physical dimensions, drawer split counts, peg capacities, 8:00-10:00 AM rush
profiles, and SKU priority/co-retrieval weights are exposed as explicit variables
so empirical measurements can be dropped in without changing simulation logic.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = ROOT_DIR / "data" / "roll-data.csv"
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "shelf_config.yaml"
DEFAULT_GEOMETRY_PATH = ROOT_DIR / "config" / "shelf_geometry.yaml"

_ACTIVE_CONFIG_PATH: Optional[Path] = None
_ACTIVE_CONFIG_CACHE: Optional[Tuple[Optional[str], Dict[str, Any]]] = None


@dataclass
class ShelfGeometryConfig:
    """Physical wire shelf dimensions (cm) and slot parameters.

    Three-tier modular shelf (120 x 60 x 60 cm per row):
      - Mid (eye level, 140 cm) and Btm (25-80 cm): 7 drawers (4P/3P or mixed) or 1 bin + 4 drawers, or 2 bins.
      - Top (195 cm): Sliding Peg (8 rows x 4 pins = 32 pegs; 1 tall or 2 short rolls/peg).
    """

    shelf_width_cm: float = 120.0
    shelf_depth_cm: float = 60.0
    shelf_row_height_cm: float = 60.0
    eye_level_height_cm: float = 140.0
    mid_level_height_cm: float = 80.0
    bottom_level_height_cm: float = 25.0
    top_overhead_height_cm: float = 195.0

    wire_bins_per_full_row: int = 2
    drawers_per_full_row: int = 7
    drawers_per_half_bay: int = 4
    drawer_outer_width_cm: float = 15.24
    drawer_outer_depth_cm: float = 60.0
    drawer_outer_height_cm: float = 30.0
    drawers_9in_per_full_row: int = 5
    drawers_9in_per_half_bay: int = 3
    drawer_9in_outer_width_cm: float = 22.86
    drawer_9in_inner_width_cm: float = 21.6
    wire_bin_bay_width_cm: float = 51.4
    wire_bin_bay_depth_cm: float = 60.0
    wire_bin_bay_height_cm: float = 60.0
    wire_bin_subzones: int = 4
    wire_bin_subzone_hard_cap: int = 9

    drawer_count: int = 7
    partitions_per_drawer: int = 4
    drawer_inner_width_cm: float = 14.0
    drawer_inner_depth_cm: float = 58.0
    drawer_inner_height_cm: float = 27.0
    partition_divider_thickness_cm: float = 0.4

    wire_bin_width_cm: float = 25.0
    wire_bin_depth_cm: float = 58.0
    wire_bin_height_cm: float = 58.0
    wire_bin_max_rolls: int = 10

    peg_rows: int = 8
    pegs_per_row: int = 4
    peg_usable_length_cm: float = 25.0
    peg_horizontal_spacing_cm: float = 15.0
    peg_pull_down_drop_cm: float = 35.0
    peg_max_od_cm: float = 15.24

    def partition_width_cm(
        self,
        partitions: Optional[int] = None,
        sub_idx: int = 0,
        inner_width_cm: Optional[float] = None,
    ) -> float:
        """Return clear inner width (cm) of a single drawer partition (4P 2x2 or 3P 2:1:1)."""
        n = partitions if partitions is not None else self.partitions_per_drawer
        w = inner_width_cm if inner_width_cm is not None else self.drawer_inner_width_cm
        if n == 3 and sub_idx == 0:
            return w
        return (w - self.partition_divider_thickness_cm) / 2.0


@dataclass
class MechanismSpec:
    """Ergonomic, visibility, and clearance parameters for a storage mechanism."""

    mechanism_id: str
    label: str
    reach_actuation_cost: float
    visibility_closed: float
    visibility_open: float
    clearance_score: float
    base_retrieval_sec: float
    base_return_sec: float
    supports_coreless: bool = True
    is_pull_down: bool = False


def build_mechanism_specs(topology: str) -> Dict[str, MechanismSpec]:
    """Build tier/mechanism physical parameters for a given shelf topology."""
    if topology != "M_modular":
        raise ValueError(f"Unknown topology: {topology}")
    specs = {
        "split_drawer_eye": MechanismSpec(
            mechanism_id="split_drawer_eye",
            label="Mid Shelf: 7 Modular Split Drawers",
            reach_actuation_cost=0.14,
            visibility_closed=0.18,
            visibility_open=0.88,
            clearance_score=0.75,
            base_retrieval_sec=3.0,
            base_return_sec=2.5,
            supports_coreless=True,
            is_pull_down=False,
        ),
        "split_drawer_mid": MechanismSpec(
            mechanism_id="split_drawer_mid",
            label="Btm Shelf: Modular Split Drawers",
            reach_actuation_cost=0.38,
            visibility_closed=0.15,
            visibility_open=0.78,
            clearance_score=0.76,
            base_retrieval_sec=4.3,
            base_return_sec=3.5,
            supports_coreless=True,
            is_pull_down=False,
        ),
        "wire_bin_eye": MechanismSpec(
            mechanism_id="wire_bin_eye",
            label="Mid Shelf: Wire Container",
            reach_actuation_cost=0.28,
            visibility_closed=0.45,
            visibility_open=0.68,
            clearance_score=0.88,
            base_retrieval_sec=4.2,
            base_return_sec=3.4,
            supports_coreless=True,
            is_pull_down=False,
        ),
        "wire_bin_mid": MechanismSpec(
            mechanism_id="wire_bin_mid",
            label="Btm Shelf: Bulk Wire Container",
            reach_actuation_cost=0.42,
            visibility_closed=0.40,
            visibility_open=0.62,
            clearance_score=0.88,
            base_retrieval_sec=4.9,
            base_return_sec=3.9,
            supports_coreless=True,
            is_pull_down=False,
        ),
        "pegboard_ovh": MechanismSpec(
            mechanism_id="pegboard_ovh",
            label="Top Shelf: Sliding Peg (32 Pegs)",
            reach_actuation_cost=0.22,
            visibility_closed=0.84,
            visibility_open=0.94,
            clearance_score=0.62,
            base_retrieval_sec=3.6,
            base_return_sec=3.3,
            supports_coreless=False,
            is_pull_down=True,
        ),
    }
    overrides = _get_active_config().get("mechanisms") or {}
    for mech_id, patch in overrides.items():
        if mech_id in specs and isinstance(patch, dict):
            spec = specs[mech_id]
            for f in fields(MechanismSpec):
                if f.name in patch and f.name != "mechanism_id":
                    setattr(spec, f.name, _coerce_like(patch[f.name], getattr(spec, f.name)))
    return specs


@dataclass
class RushHourConfig:
    """Shift schedule and 8:00 AM - 10:00 AM rush window demand parameters."""

    shift_cycles: int = 50
    num_shifts: int = 20
    rush_window_hours: Tuple[float, float] = (8.0, 10.0)
    shift_hours: Tuple[float, float] = (8.0, 16.0)
    rush_job_share: float = 0.45
    rush_tau_range: Tuple[float, float] = (0.75, 1.00)
    calm_tau_range: Tuple[float, float] = (0.10, 0.35)
    rush_sku_boost: float = 3.6
    same_product_co_retrieval: float = 0.90

    rush_priority_categories: Tuple[str, ...] = ("Bites", "Salad", "Breakfast Cookies")
    rush_priority_products: Tuple[str, ...] = (
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
    )

    @property
    def total_cycles(self) -> int:
        return self.shift_cycles * self.num_shifts


@dataclass
class BehavioralConfig:
    """Human-factors parameters governing search, misplacement, and cleanup."""

    base_error_rate: float = 0.028
    base_correction_rate: float = 0.24
    alpha_rush: float = 2.8
    alpha_crowd: float = 3.2
    alpha_blur: float = 1.75
    alpha_ergo: float = 2.3
    unlatch_rush_prob: float = 0.38
    occlusion_friction_penalty: float = 0.35
    search_miss_penalty_sec: float = 14.0
    usability_accuracy_threshold: float = 0.80
    bin_dump_error_penalty: float = 0.11
    bin_dump_routing_weight: float = 0.30
    mech_diversity_penalty: float = 0.10
    bin_drawer_same_row_penalty: float = 0.16
    drawer_size_mix_penalty: float = 0.0
    multi_sku_slot_penalty: float = 0.16
    slot_deficit_penalty: float = 1.20
    row_width_mismatch_penalty: float = 0.25


def count_shelf_complexity(row_configs: Optional[Sequence[str]]) -> Tuple[int, int]:
    """Return (n_bins, extra_styles) across {peg, 4P_drawer, 3P_drawer, wire_bin}."""
    if not row_configs:
        return 0, 0
    n_bins = sum(
        2 if r in ("2B", "1D6_2B", "1B_1D6_1B", "2B_1D6", "1D_2B", "1B_1D_1B", "2B_1D")
        else (1 if r.startswith("1B_") else 0)
        for r in row_configs
    )
    styles = set()
    for r in row_configs:
        if "PEG" in r:
            styles.add("peg")
        if "4P" in r or r in ("1D6_2B", "1B_1D6_1B", "2B_1D6", "1D_2B", "1B_1D_1B", "2B_1D"):
            styles.add("4P_drawer")
        if "3P" in r:
            styles.add("3P_drawer")
        if r in ("2B", "1D6_2B", "1B_1D6_1B", "2B_1D6", "1D_2B", "1B_1D_1B", "2B_1D") or r.startswith("1B_"):
            styles.add("wire_bin")
    return n_bins, max(0, len(styles) - 2)


def has_mixed_drawer_sizes(row_configs: Optional[Sequence[str]]) -> bool:
    """Return True if a rack mixes both 6in and 9in outer drawer sizes."""
    if not row_configs:
        return False
    has_6in = any(("D6" in r or r in ("7D_4P", "7D_3P", "1B_4D_4P", "1B_4D_3P", "1D_2B", "1B_1D_1B", "2B_1D")) for r in row_configs)
    has_9in = any("D9" in r for r in row_configs)
    return has_6in and has_9in


def count_mixed_bin_drawer_rows(row_configs: Optional[Sequence[str]]) -> int:
    """Return number of shelf rows that mix a wire bin and drawers in the same row."""
    if not row_configs:
        return 0
    return sum(1 for r in row_configs if ("B" in r and "D" in r))


def has_row_width_mismatch(row_configs: Optional[Sequence[str]]) -> bool:
    """Return True if full-drawer rows on the rack have misaligned outer widths (e.g. 42in vs 45in)."""
    if not row_configs:
        return False
    widths_in = []
    for r in row_configs:
        if r in ("7D6_4P", "7D6_3P", "7D_4P", "7D_3P", "4D4P_3D3P"):
            widths_in.append(42.0)
        elif r in ("5D9_4P", "5D9_3P", "3D9_3D6_4P"):
            widths_in.append(45.0)
    return len(set(widths_in)) > 1


def _strip_toon_comment(raw: str) -> str:
    """Remove unquoted # comments from a TOON line."""
    in_quote: Optional[str] = None
    escaped = False
    for idx, ch in enumerate(raw):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch in ('"', "'"):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
        elif ch == "#" and in_quote is None:
            return raw[:idx].rstrip()
    return raw.rstrip()


def _split_csv_values(raw: str) -> List[str]:
    """Split a comma-separated TOON value row while honoring quotes."""
    parts: List[str] = []
    cur: List[str] = []
    in_quote: Optional[str] = None
    for ch in raw:
        if ch in ('"', "'"):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
            cur.append(ch)
        elif ch == "," and in_quote is None:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur or raw.endswith(","):
        parts.append("".join(cur).strip())
    return parts


def _coerce_toon_scalar(token: str) -> Any:
    """Convert a TOON scalar string into bool, int, float, None, or str."""
    t = token.strip()
    if t == "" or t in ("null", "~"):
        return None
    if t == "true":
        return True
    if t == "false":
        return False
    if (len(t) >= 2 and t[0] == t[-1]) and t[0] in ('"', "'"):
        return t[1:-1]
    if re.fullmatch(r"[-+]?\d+", t):
        return int(t)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+(?:[eE][-+]?\d+))", t):
        return float(t)
    return t


def _parse_toon(text: str) -> Dict[str, Any]:
    """Parse TOON (Token-Oriented Object Notation) text into nested Python objects."""
    lines: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        cleaned = _strip_toon_comment(raw)
        if cleaned.strip():
            indent = len(cleaned) - len(cleaned.lstrip(" "))
            lines.append((indent, cleaned.strip()))

    def _parse_block(idx: int, min_indent: int) -> Tuple[Any, int]:
        out: Any = None
        while idx < len(lines):
            indent, content = lines[idx]
            if indent < min_indent:
                break
            if content.startswith("- "):
                if out is None:
                    out = []
                item_body = content[2:].strip()
                item_dict: Dict[str, Any] = {}
                if ":" in item_body:
                    k_part, rest = _split_key_val(item_body)
                    if k_part is not None:
                        _assign_toon_entry(item_dict, k_part, rest, lines, idx, indent + 2)
                idx += 1
                while idx < len(lines) and lines[idx][0] > indent:
                    sub_obj, idx = _parse_block(idx, indent + 1)
                    if isinstance(sub_obj, dict):
                        item_dict.update(sub_obj)
                out.append(item_dict if item_dict else _coerce_toon_scalar(item_body))
                continue

            if out is None:
                out = {}
            k_part, rest = _split_key_val(content)
            if k_part is None:
                idx += 1
                continue
            idx = _assign_toon_entry(out, k_part, rest, lines, idx, indent)
        return ({} if out is None else out), idx

    def _split_key_val(content: str) -> Tuple[Optional[str], str]:
        if ":" not in content:
            return None, ""
        key, rest = content.split(":", 1)
        return key.strip(), rest.strip()

    def _assign_toon_entry(
        target: Dict[str, Any],
        raw_key: str,
        rest: str,
        all_lines: List[Tuple[int, str]],
        cur_idx: int,
        base_indent: int,
    ) -> int:
        tab_match = re.fullmatch(r"([^\[\{]+)\[\d+\]\{([^\}]+)\}", raw_key)
        arr_match = re.fullmatch(r"([^\[\{]+)\[\d+\]", raw_key)
        if tab_match:
            key = tab_match.group(1).strip()
            cols = [c.strip() for c in tab_match.group(2).split(",")]
            rows: List[Dict[str, Any]] = []
            nxt = cur_idx + 1
            while nxt < len(all_lines) and all_lines[nxt][0] > base_indent:
                vals = [_coerce_toon_scalar(v) for v in _split_csv_values(all_lines[nxt][1])]
                rows.append(dict(zip(cols, vals)))
                nxt += 1
            target[key] = rows
            return nxt
        if arr_match:
            key = arr_match.group(1).strip()
            if rest:
                target[key] = [_coerce_toon_scalar(v) for v in _split_csv_values(rest)]
                return cur_idx + 1
            nxt = cur_idx + 1
            if nxt < len(all_lines) and all_lines[nxt][0] > base_indent:
                child, nxt = _parse_block(nxt, all_lines[nxt][0])
                target[key] = child if isinstance(child, list) else [child]
                return nxt
            target[key] = []
            return cur_idx + 1
        key = raw_key.strip()
        if rest:
            target[key] = _coerce_toon_scalar(rest)
            return cur_idx + 1
        nxt = cur_idx + 1
        if nxt < len(all_lines) and all_lines[nxt][0] > base_indent:
            child, nxt = _parse_block(nxt, all_lines[nxt][0])
            target[key] = child
            return nxt
        target[key] = {}
        return cur_idx + 1

    parsed, _ = _parse_block(0, 0)
    return parsed if isinstance(parsed, dict) else {}


def _fmt_toon_scalar(val: Any) -> str:
    """Format a Python scalar cleanly for TOON output."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if val is None:
        return "null"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val)
    if s == "" or any(c in s for c in (",", "#", "\n")) or s in ("true", "false", "null"):
        return json.dumps(s)
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
        return json.dumps(s)
    return s


def _dump_toon(data: Any, indent: int = 0) -> str:
    """Serialize a nested config dict into TOON format."""
    pad = " " * indent
    out: List[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                out.append(f"{pad}{k}:")
                out.append(_dump_toon(v, indent + 2))
            elif isinstance(v, (list, tuple)):
                if all(not isinstance(item, (dict, list, tuple)) for item in v):
                    joined = ",".join(_fmt_toon_scalar(item) for item in v)
                    out.append(f"{pad}{k}[{len(v)}]: {joined}")
                elif v and all(
                    isinstance(item, dict)
                    and all(not isinstance(fv, (dict, list, tuple)) for fv in item.values())
                    for item in v
                ):
                    cols = list(v[0].keys())
                    out.append(f"{pad}{k}[{len(v)}]{{{','.join(cols)}}}:")
                    for row in v:
                        row_vals = ",".join(_fmt_toon_scalar(row.get(c)) for c in cols)
                        out.append(f"{pad}  {row_vals}")
                else:
                    out.append(f"{pad}{k}[{len(v)}]:")
                    for item in v:
                        item_lines = _dump_toon(item, indent + 4).splitlines()
                        if item_lines:
                            first = item_lines[0].lstrip()
                            out.append(f"{pad}  - {first}")
                            out.extend(item_lines[1:])
            else:
                out.append(f"{pad}{k}: {_fmt_toon_scalar(v)}")
    return "\n".join(line for line in out if line != "") + ("\n" if indent == 0 else "")


def _coerce_like(raw_val: Any, template: Any) -> Any:
    """Coerce loaded config values to match dataclass field types."""
    if isinstance(template, tuple):
        if isinstance(raw_val, (list, tuple)):
            return tuple(raw_val)
        return template
    if isinstance(template, bool):
        return bool(raw_val)
    if isinstance(template, int) and not isinstance(template, bool):
        return int(raw_val)
    if isinstance(template, float):
        return float(raw_val)
    if isinstance(template, str):
        return str(raw_val)
    return raw_val


def _resolve_active_config_path() -> Optional[Path]:
    """Return active config path from override, env var, or default shelf_config.yaml."""
    if _ACTIVE_CONFIG_PATH is not None:
        return _ACTIVE_CONFIG_PATH
    env_val = os.environ.get("DTC_SIM_CONFIG", "").strip()
    if env_val:
        return Path(env_val)
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    return None


def load_config_dict(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration dict from a .yaml/.yml, .json, or .toon file."""
    target = Path(path) if path is not None else _resolve_active_config_path()
    if target is None or not target.exists():
        return {}
    text = target.read_text(encoding="utf-8")
    suffix = target.suffix.lower()
    if suffix == ".json":
        loaded = json.loads(text)
    elif suffix == ".toon":
        loaded = _parse_toon(text)
    else:
        loaded = yaml.safe_load(text) or {}
    return loaded if isinstance(loaded, dict) else {}


def _deep_merge_dicts(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge patch keys onto base dict."""
    merged = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged


def _normalize_config_sections(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Combine geometry, shift_schedule, and demand_weights cleanly alongside weights."""
    out = dict(raw)
    combined_rush: Dict[str, Any] = {}
    for key in ("shift_schedule", "demand_weights", "rush_schedule"):
        sec = out.get(key)
        if isinstance(sec, dict):
            combined_rush.update(sec)
    if combined_rush:
        out["rush_schedule"] = combined_rush
    return out


def _get_active_config() -> Dict[str, Any]:
    """Return cached configuration merged from shelf_geometry.yaml and active config file."""
    global _ACTIVE_CONFIG_CACHE
    target = _resolve_active_config_path()
    key = str(target.resolve()) if target is not None and target.exists() else None
    geom_key = str(DEFAULT_GEOMETRY_PATH.resolve()) if DEFAULT_GEOMETRY_PATH.exists() else ""
    cache_key = f"{geom_key}::{key or ''}"
    if _ACTIVE_CONFIG_CACHE is not None and _ACTIVE_CONFIG_CACHE[0] == cache_key:
        return _ACTIVE_CONFIG_CACHE[1]
    base_data = load_config_dict(DEFAULT_GEOMETRY_PATH) if DEFAULT_GEOMETRY_PATH.exists() else {}
    over_data = load_config_dict(target) if key is not None else {}
    merged = _normalize_config_sections(_deep_merge_dicts(base_data, over_data))
    _ACTIVE_CONFIG_CACHE = (cache_key, merged)
    return merged


def set_active_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Set the active YAML/JSON/TOON config file (propagated to worker subprocesses)."""
    global _ACTIVE_CONFIG_PATH, _ACTIVE_CONFIG_CACHE
    if path is None:
        _ACTIVE_CONFIG_PATH = None
        os.environ.pop("DTC_SIM_CONFIG", None)
        _ACTIVE_CONFIG_CACHE = None
        return _get_active_config()
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    _ACTIVE_CONFIG_PATH = resolved
    os.environ["DTC_SIM_CONFIG"] = str(resolved)
    _ACTIVE_CONFIG_CACHE = None
    return _get_active_config()


def _build_dataclass_with_overrides(cls: Any, section_key: str) -> Any:
    """Instantiate a config dataclass and apply overrides from the active config section."""
    obj = cls()
    section = _get_active_config().get(section_key) or {}
    if isinstance(section, dict):
        for f in fields(cls):
            if f.name in section:
                setattr(obj, f.name, _coerce_like(section[f.name], getattr(obj, f.name)))
    return obj


def make_geometry_config() -> ShelfGeometryConfig:
    return _build_dataclass_with_overrides(ShelfGeometryConfig, "geometry")


def make_rush_config() -> RushHourConfig:
    return _build_dataclass_with_overrides(RushHourConfig, "rush_schedule")


def make_behavioral_config() -> BehavioralConfig:
    return _build_dataclass_with_overrides(BehavioralConfig, "behavioral_weights")


def get_execution_config() -> Dict[str, Any]:
    """Return execution replication counts and score weights with active config overrides."""
    defaults: Dict[str, Any] = {
        "benchmark_replications": 50,
        "sweep_replications": 25,
        "search_replications": 4,
        "composite_score_latency_weight": 0.85,
    }
    section = _get_active_config().get("execution") or {}
    if isinstance(section, dict):
        for k, v in defaults.items():
            if k in section:
                defaults[k] = _coerce_like(section[k], v)
    return defaults


def export_active_config_dict() -> Dict[str, Any]:
    """Build a complete serializable dictionary of all active weights, geometry, and scenarios."""
    mech_specs = build_mechanism_specs("M_modular")
    scenarios_list = []
    for exp in build_default_experiments():
        scenarios_list.append({
            "exp_id": exp.exp_id,
            "name": exp.name,
            "topology": exp.topology,
            "strategy": exp.strategy,
            "drawer_partitions": exp.drawer_partitions,
            "cue_salience": exp.cue_salience,
            "target_fill_ratio": exp.target_fill_ratio,
            "description": exp.description,
            "row_configs": list(exp.row_configs) if exp.row_configs else [],
            "abc_colocation_mode": exp.abc_colocation_mode,
            "overflow_mode": exp.overflow_mode,
            "spatial_order_mode": exp.spatial_order_mode,
            "layout_seed": exp.layout_seed,
        })
    return {
        "execution": get_execution_config(),
        "behavioral_weights": asdict(make_behavioral_config()),
        "geometry": asdict(make_geometry_config()),
        "rush_schedule": {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in asdict(make_rush_config()).items()
        },
        "mechanisms": {k: asdict(v) for k, v in mech_specs.items()},
        "benchmark_scenarios": scenarios_list,
    }


def save_config_file(path: Path, config_data: Optional[Dict[str, Any]] = None) -> Path:
    """Write configuration dictionary to .yaml/.yml, .json, or .toon."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = config_data if config_data is not None else export_active_config_dict()
    suffix = out_path.suffix.lower()
    if suffix == ".json":
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    elif suffix == ".toon":
        out_path.write_text(_dump_toon(payload), encoding="utf-8")
    else:
        out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out_path


@dataclass
class ExperimentScenario:
    """Complete parameter bundle for one experimental run."""

    exp_id: str
    name: str
    topology: str
    strategy: str
    drawer_partitions: int
    cue_salience: float
    target_fill_ratio: float
    description: str = ""
    geometry: ShelfGeometryConfig = field(default_factory=make_geometry_config)
    rush: RushHourConfig = field(default_factory=make_rush_config)
    behavior: BehavioralConfig = field(default_factory=make_behavioral_config)
    row_configs: Optional[Tuple[str, str, str]] = None
    abc_colocation_mode: str = "default"
    overflow_mode: str = "hybrid"
    spatial_order_mode: str = "center_out"
    layout_seed: int = 0


@dataclass
class RollVariantSKU:
    """A distinct (Label type, Roll variant) SKU from roll-data.csv."""

    variant_id: int
    sku_code: str
    product_name: str
    variant_code: str
    outer_diameter_cm: float
    roll_height_cm: float
    quantity: int
    category: str
    notes: str
    has_core: bool
    is_rush_priority: bool
    visual_similarity_group: str
    baseline_freq: float = 0.0
    rush_freq: float = 0.0


@dataclass
class RollItem:
    """A single physical roll tracked across slots over time."""

    roll_id: int
    variant_id: int
    sku_code: str
    product_name: str
    variant_code: str
    copy_index: int
    outer_diameter_cm: float
    roll_height_cm: float
    has_core: bool
    category: str
    is_rush_priority: bool
    visual_similarity_group: str
    home_mechanism: str = ""
    home_slot: str = ""
    current_mechanism: str = ""
    current_slot: str = ""
    misplaced_history_count: int = 0


def load_roll_inventory(
    csv_path: Path = DEFAULT_CSV_PATH,
    rush_config: Optional[RushHourConfig] = None,
    priority_overrides: Optional[Dict[str, float]] = None,
    co_retrieval_overrides: Optional[Dict[Tuple[str, str], float]] = None,
) -> Tuple[List[RollVariantSKU], List[RollItem], List[float], List[float], List[List[float]]]:
    """Load roll-data.csv and construct SKU variants, 128 physical rolls, f, and C.

    Args:
        csv_path: Path to roll-data.csv.
        rush_config: Rush-hour priority definitions.
        priority_overrides: Optional dict mapping "Product|Variant" -> demand weight.
        co_retrieval_overrides: Optional dict mapping (sku_a, sku_b) -> cond. prob.

    Returns:
        (variants, rolls, freq_calm, freq_rush, co_matrix_60x60)
    """
    rc = rush_config or RushHourConfig()
    variants: List[RollVariantSKU] = []
    rolls: List[RollItem] = []

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        vid = 0
        rid = 0
        for row in reader:
            label = (row.get("Label type") or "").strip()
            if not label or label.startswith("*"):
                continue
            var_code = (row.get("Roll variant") or "A").strip()
            od = float(row["Outer diameter (cm)"])
            ht = float(row["Roll height (cm)"])
            qty = int(row["Quantity"])
            category = (row.get("Category") or "Other").strip()
            notes = (row.get("Notes") or "").strip()
            has_core = "no inner roll" not in notes.lower()
            is_priority = (
                label in rc.rush_priority_products
                or category in ("Bites", "Salad")
                or "breakfast cookie" in label.lower()
            )
            height_band = "short" if ht <= 10.0 else "tall"
            vis_group = f"{category}::{height_band}"
            sku_code = f"{label} [{var_code}]"

            sku = RollVariantSKU(
                variant_id=vid,
                sku_code=sku_code,
                product_name=label,
                variant_code=var_code,
                outer_diameter_cm=od,
                roll_height_cm=ht,
                quantity=qty,
                category=category,
                notes=notes,
                has_core=has_core,
                is_rush_priority=is_priority,
                visual_similarity_group=vis_group,
            )
            variants.append(sku)

            for c_idx in range(qty):
                rolls.append(
                    RollItem(
                        roll_id=rid,
                        variant_id=vid,
                        sku_code=sku_code,
                        product_name=label,
                        variant_code=var_code,
                        copy_index=c_idx,
                        outer_diameter_cm=od,
                        roll_height_cm=ht,
                        has_core=has_core,
                        category=category,
                        is_rush_priority=is_priority,
                        visual_similarity_group=vis_group,
                    )
                )
                rid += 1
            vid += 1

    freq_calm, freq_rush, co_matrix = build_demand_model(
        variants,
        rc,
        priority_overrides=priority_overrides,
        co_retrieval_overrides=co_retrieval_overrides,
    )
    for idx, sku in enumerate(variants):
        sku.baseline_freq = freq_calm[idx]
        sku.rush_freq = freq_rush[idx]

    return variants, rolls, freq_calm, freq_rush, co_matrix


def build_demand_model(
    variants: Sequence[RollVariantSKU],
    rush_config: RushHourConfig,
    priority_overrides: Optional[Dict[str, float]] = None,
    co_retrieval_overrides: Optional[Dict[Tuple[str, str], float]] = None,
) -> Tuple[List[float], List[float], List[List[float]]]:
    """Construct calm/rush pick probability vectors f and 60x60 co-retrieval matrix C."""
    n = len(variants)
    raw_calm = []
    raw_rush = []

    for v in variants:
        base_w = 0.6 + 0.55 * v.quantity
        if v.category == "Ingredient sticker":
            base_w += 0.8
        if priority_overrides and v.sku_code in priority_overrides:
            base_w = max(0.05, float(priority_overrides[v.sku_code]))
        raw_calm.append(base_w)

        rush_w = base_w * (rush_config.rush_sku_boost if v.is_rush_priority else 0.70)
        raw_rush.append(rush_w)

    sum_calm = sum(raw_calm)
    sum_rush = sum(raw_rush)
    freq_calm = [w / sum_calm for w in raw_calm]
    freq_rush = [w / sum_rush for w in raw_rush]

    co_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            vi = variants[i]
            vj = variants[j]
            weight = 0.01
            if vi.product_name == vj.product_name:
                weight = rush_config.same_product_co_retrieval
            elif {vi.category, vj.category} == {"Chocolate bark", "Ingredient sticker"}:
                if "Peppermint" in vi.product_name and "Peppermint" in vj.product_name:
                    weight = 0.78
                elif "Peppermint" not in vi.product_name and "Peppermint" not in vj.product_name:
                    weight = 0.52
                else:
                    weight = 0.15
            elif vi.category == vj.category:
                weight = 0.24 if (vi.is_rush_priority and vj.is_rush_priority) else 0.10

            if co_retrieval_overrides:
                pair_a = (vi.sku_code, vj.sku_code)
                pair_b = (vj.sku_code, vi.sku_code)
                if pair_a in co_retrieval_overrides:
                    weight = co_retrieval_overrides[pair_a]
                elif pair_b in co_retrieval_overrides:
                    weight = co_retrieval_overrides[pair_b]

            co_matrix[i][j] = weight
            co_matrix[j][i] = weight

    return freq_calm, freq_rush, co_matrix


def build_default_experiments() -> List[ExperimentScenario]:
    """Return modular shelf benchmark scenarios (overridable via active config)."""
    cfg_list = _get_active_config().get("benchmark_scenarios")
    if isinstance(cfg_list, list) and cfg_list:
        out: List[ExperimentScenario] = []
        for item in cfg_list:
            if not isinstance(item, dict):
                continue
            rc = item.get("row_configs")
            row_tuple = tuple(str(x) for x in rc) if isinstance(rc, (list, tuple)) else None
            out.append(
                ExperimentScenario(
                    exp_id=str(item["exp_id"]),
                    name=str(item["name"]),
                    topology=str(item.get("topology", "M_modular")),
                    strategy=str(item.get("strategy", "C_hybrid")),
                    drawer_partitions=int(item.get("drawer_partitions", 4)),
                    cue_salience=float(item.get("cue_salience", 0.90)),
                    target_fill_ratio=float(item.get("target_fill_ratio", 0.80)),
                    description=str(item.get("description", "")),
                    row_configs=row_tuple,
                    abc_colocation_mode=str(item.get("abc_colocation_mode", "adjacent_partition")),
                    overflow_mode=str(item.get("overflow_mode", "hybrid_reserve")),
                    spatial_order_mode=str(item.get("spatial_order_mode", "center_out")),
                    layout_seed=int(item.get("layout_seed", 0)),
                )
            )
        if out:
            return out

    return [
        ExperimentScenario(
            exp_id="ORI-258",
            name="Top: Peg | Mid: 7D6_4P | Btm: 2B (90% fill, low cue)",
            topology="M_modular",
            strategy="B_kit",
            drawer_partitions=4,
            cue_salience=0.20,
            target_fill_ratio=0.90,
            description="Top: Peg | Mid: 7D6_4P | Btm: 2B at 90% fill with low label clarity and height-split COI.",
            row_configs=("7D6_4P", "2B", "SLIDING_PEG"),
            abc_colocation_mode="cross_tier_height",
            overflow_mode="all_colocated",
            spatial_order_mode="random_perm",
            layout_seed=1491,
        ),
        ExperimentScenario(
            exp_id="ORI-255",
            name="Top: Peg | Mid: 7D6_4P | Btm: 2B (height split)",
            topology="M_modular",
            strategy="C_hybrid",
            drawer_partitions=4,
            cue_salience=0.90,
            target_fill_ratio=0.80,
            description="Top: Peg | Mid: 7D6_4P | Btm: 2B at 80% fill with clear labels and height-split grouping.",
            row_configs=("7D6_4P", "2B", "SLIDING_PEG"),
            abc_colocation_mode="cross_tier_height",
            overflow_mode="hybrid_reserve",
            spatial_order_mode="center_out",
            layout_seed=1470,
        ),
        ExperimentScenario(
            exp_id="ORI-099",
            name="Top: Peg | Mid: 7D6_3P | Btm: 1B_4D6_3P (low cue)",
            topology="M_modular",
            strategy="C_hybrid",
            drawer_partitions=3,
            cue_salience=0.20,
            target_fill_ratio=0.80,
            description="Top: Peg | Mid: 7D6_3P | Btm: 1B_4D6_3P at 80% fill with low label clarity and random placement.",
            row_configs=("7D6_3P", "1B_4D6_3P", "SLIDING_PEG"),
            abc_colocation_mode="adjacent_partition",
            overflow_mode="hybrid_reserve",
            spatial_order_mode="random_perm",
            layout_seed=1250,
        ),
        ExperimentScenario(
            exp_id="ORI-085",
            name="Top: Peg | Mid: 7D6_3P | Btm: 1B_4D6_3P",
            topology="M_modular",
            strategy="C_hybrid",
            drawer_partitions=3,
            cue_salience=0.90,
            target_fill_ratio=0.80,
            description="Top: Peg | Mid: 7D6_3P | Btm: 1B_4D6_3P at 80% fill with clear labels and adjacent A/B/C grouping.",
            row_configs=("7D6_3P", "1B_4D6_3P", "SLIDING_PEG"),
            abc_colocation_mode="adjacent_partition",
            overflow_mode="hybrid_reserve",
            spatial_order_mode="center_out",
            layout_seed=1152,
        ),
        ExperimentScenario(
            exp_id="ORI-245",
            name="Top: Peg | Mid: 7D6_4P | Btm: 1D6_2B",
            topology="M_modular",
            strategy="C_hybrid",
            drawer_partitions=4,
            cue_salience=0.90,
            target_fill_ratio=0.80,
            description="Top: Peg | Mid: 7D6_4P | Btm: 1D6_2B (1 6in drawer on left + 2 wire bins).",
            row_configs=("7D6_4P", "1D6_2B", "SLIDING_PEG"),
            abc_colocation_mode="adjacent_partition",
            overflow_mode="hybrid_reserve",
            spatial_order_mode="center_out",
            layout_seed=1400,
        ),
        ExperimentScenario(
            exp_id="ORI-5D9",
            name="Top: Peg | Mid: 5D9_4P | Btm: 5D9_4P",
            topology="M_modular",
            strategy="C_hybrid",
            drawer_partitions=4,
            cue_salience=0.90,
            target_fill_ratio=0.80,
            description="Top: Peg | Mid: 5D9_4P | Btm: 5D9_4P (all 9in outer drawers, bin-free).",
            row_configs=("5D9_4P", "5D9_4P", "SLIDING_PEG"),
            abc_colocation_mode="adjacent_partition",
            overflow_mode="hybrid_reserve",
            spatial_order_mode="center_out",
            layout_seed=1407,
        ),
        ExperimentScenario(
            exp_id="ORI-325",
            name="Top: Peg | Mid: 1B_4D6_4P | Btm: 2B",
            topology="M_modular",
            strategy="C_hybrid",
            drawer_partitions=4,
            cue_salience=0.90,
            target_fill_ratio=0.80,
            description="Top: Peg | Mid: 1B_4D6_4P | Btm: 2B at 80% fill.",
            row_configs=("1B_4D6_4P", "2B", "SLIDING_PEG"),
            abc_colocation_mode="adjacent_partition",
            overflow_mode="hybrid_reserve",
            spatial_order_mode="center_out",
            layout_seed=1524,
        ),
        ExperimentScenario(
            exp_id="ORI-021",
            name="Top: Peg | Mid: 7D6_4P | Btm: 1B_4D6_3P",
            topology="M_modular",
            strategy="C_hybrid",
            drawer_partitions=4,
            cue_salience=0.90,
            target_fill_ratio=0.80,
            description="Top: Peg | Mid: 7D6_4P | Btm: 1B_4D6_3P at 80% fill.",
            row_configs=("7D6_4P", "1B_4D6_3P", "SLIDING_PEG"),
            abc_colocation_mode="adjacent_partition",
            overflow_mode="hybrid_reserve",
            spatial_order_mode="center_out",
            layout_seed=1040,
        ),
    ]
