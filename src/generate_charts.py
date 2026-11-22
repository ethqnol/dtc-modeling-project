"""Publication-quality chart generator using pycairo (300 DPI PNG + vector SVG)."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Sequence, Tuple

import cairo

from src.simulation_engine import ExperimentSummary

if TYPE_CHECKING:
    from src.orientation_search import OrientationRecord


PALETTE = {
    "ORI-258": (0.769, 0.306, 0.322),
    "ORI-255": (0.506, 0.447, 0.702),
    "ORI-099": (0.576, 0.471, 0.376),
    "ORI-085": (0.867, 0.518, 0.322),
    "ORI-245": (0.333, 0.659, 0.408),
    "ORI-5D9": (0.800, 0.475, 0.655),
    "ORI-325": (0.392, 0.710, 0.804),
    "ORI-021": (0.298, 0.447, 0.690),
    "ORI-CHAMP": (0.090, 0.545, 0.355),
}

STACK_COLORS = {
    "Home": (0.298, 0.447, 0.690),
    "Within-Mechanism Drift": (0.867, 0.518, 0.322),
    "Eye-Level Drawer Dump": (0.769, 0.306, 0.322),
    "Other Cross-Tier Dump": (0.506, 0.447, 0.702),
}


def _render_dual(
    out_stem: Path,
    width: int,
    height: int,
    draw_fn: Callable[[cairo.Context, int, int], None],
    scale: float = 2.0,
) -> Tuple[Path, Path]:
    """Render a drawing function to both high-DPI PNG and vector SVG."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_stem.with_suffix(".png")
    svg_path = out_stem.with_suffix(".svg")

    img = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(width * scale), int(height * scale))
    ctx_img = cairo.Context(img)
    ctx_img.scale(scale, scale)
    draw_fn(ctx_img, width, height)
    img.write_to_png(str(png_path))

    svg = cairo.SVGSurface(str(svg_path), width, height)
    ctx_svg = cairo.Context(svg)
    draw_fn(ctx_svg, width, height)
    svg.finish()
    return png_path, svg_path


def _rounded_rect(
    ctx: cairo.Context, x: float, y: float, w: float, h: float, r: float = 3.5
) -> None:
    r = min(r, w / 2.0, h / 2.0)
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2.0, 0.0)
    ctx.arc(x + w - r, y + h - r, r, 0.0, math.pi / 2.0)
    ctx.arc(x + r, y + h - r, r, math.pi / 2.0, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3.0 * math.pi / 2.0)
    ctx.close_path()


def _text(
    ctx: cairo.Context,
    x: float,
    y: float,
    txt: str,
    size: float = 12.0,
    bold: bool = False,
    color: Tuple[float, float, float] = (0.12, 0.14, 0.18),
    align: str = "left",
    valign: str = "baseline",
) -> None:
    ctx.save()
    weight = cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL
    ctx.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, weight)
    ctx.set_font_size(size)
    ctx.set_source_rgb(*color)
    ext = ctx.text_extents(txt)
    tx = x
    if align == "center":
        tx = x - ext.width / 2.0 - ext.x_bearing
    elif align == "right":
        tx = x - ext.width - ext.x_bearing
    ty = y
    if valign == "middle":
        ty = y - ext.y_bearing - ext.height / 2.0
    elif valign == "top":
        ty = y - ext.y_bearing
    ctx.move_to(tx, ty)
    ctx.show_text(txt)
    ctx.new_path()
    ctx.restore()


def _bg(ctx: cairo.Context, w: int, h: int) -> None:
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.rectangle(0, 0, w, h)
    ctx.fill()


def _heat_color(val_norm: float) -> Tuple[float, float, float]:
    """Sequential colorblind-safe ramp from cream -> amber -> orange -> deep red."""
    v = max(0.0, min(1.0, val_norm))
    stops = [
        (0.00, (0.95, 0.97, 0.99)),
        (0.30, (0.99, 0.91, 0.65)),
        (0.65, (0.98, 0.56, 0.24)),
        (1.00, (0.74, 0.08, 0.15)),
    ]
    for idx in range(len(stops) - 1):
        t0, c0 = stops[idx]
        t1, c1 = stops[idx + 1]
        if v <= t1:
            u = (v - t0) / max(1e-9, t1 - t0)
            return tuple(c0[k] + u * (c1[k] - c0[k]) for k in range(3))
    return stops[-1][1]


def _format_champ_labels(summary: ExperimentSummary) -> Tuple[str, str]:
    """Build dynamic plain labels for the top 400-search layout."""
    scen = summary.scenario
    ori_id = scen.name.split()[1] if scen.name.startswith("Champion ORI-") else scen.exp_id
    row_map = {
        "7D_4P": "7D_4P",
        "7D_3P": "7D_3P",
        "4D4P_3D3P": "4D4P_3D3P",
        "1B_4D_4P": "1B_4D_4P",
        "1B_4D_3P": "1B_4D_3P",
        "1B_2D4P_2D3P": "1B_2D4P_2D3P",
        "1D_2B": "1D_2B",
        "1B_1D_1B": "1B_1D_1B",
        "2B_1D": "2B_1D",
        "2B": "2B",
        "SLIDING_PEG": "Peg",
    }
    cfgs = scen.row_configs or ()
    if len(cfgs) == 3:
        desc = f"Top: {row_map.get(cfgs[2], cfgs[2])} | Mid: {row_map.get(cfgs[0], cfgs[0])} | Btm: {row_map.get(cfgs[1], cfgs[1])}"
    else:
        desc = "Modular Layout"
    return (
        f"{ori_id}: {desc}",
        f"{ori_id}: {desc}",
    )


def plot_fig1_entropy_decay(
    summaries: Dict[str, ExperimentSummary], out_dir: Path
) -> Tuple[Path, Path]:
    """Chart 1: Accuracy decay curves with +-1 SD bands and 80% usability threshold."""
    base_ids = ["ORI-5D9", "ORI-021", "ORI-245", "ORI-325", "ORI-099", "ORI-258"]
    selected_ids = (["ORI-CHAMP"] if "ORI-CHAMP" in summaries else []) + base_ids
    champ_long = _format_champ_labels(summaries["ORI-CHAMP"])[0] if "ORI-CHAMP" in summaries else "ORI-001"
    short_labels = {
        "ORI-CHAMP": champ_long,
        "ORI-5D9": "ORI-5D9: Top: Peg | Mid: 5D9_4P | Btm: 5D9_4P",
        "ORI-021": "ORI-021: Top: Peg | Mid: 7D6_4P | Btm: 1B_4D6_3P",
        "ORI-245": "ORI-245: Top: Peg | Mid: 7D6_4P | Btm: 1D6_2B",
        "ORI-325": "ORI-325: Top: Peg | Mid: 1B_4D6_4P | Btm: 2B",
        "ORI-099": "ORI-099: Top: Peg | Mid: 7D6_3P | Btm: 1B_4D6_3P (low cue)",
        "ORI-258": "ORI-258: Top: Peg | Mid: 7D6_4P | Btm: 2B (90% fill, low cue)",
    }

    def draw(ctx: cairo.Context, w: int, h: int) -> None:
        _bg(ctx, w, h)
        ml, mr, mt, mb = 78, 42, 86, 68
        pw = w - ml - mr
        ph = h - mt - mb

        _text(
            ctx,
            ml,
            30,
            "Home Slot Accuracy Over 1,000 Work Cycles",
            size=16.5,
            bold=True,
        )
        _text(
            ctx,
            ml,
            52,
            "Share of 128 rolls in their assigned home slot across 20 shifts (50-run mean, shaded band = +/- 1 SD)",
            size=11.5,
            color=(0.35, 0.39, 0.45),
        )

        y_min, y_max = 0.15, 1.02
        cycles = summaries[selected_ids[0]].cycles

        def px(t: float) -> float:
            return ml + (t / max(1, cycles - 1)) * pw

        def py(val: float) -> float:
            return mt + ph - ((val - y_min) / (y_max - y_min)) * ph

        for pct in (20, 30, 40, 50, 60, 70, 80, 90, 100):
            yv = py(pct / 100.0)
            ctx.set_source_rgba(0.85, 0.87, 0.90, 0.8)
            ctx.set_line_width(1.0)
            ctx.move_to(ml, yv)
            ctx.line_to(ml + pw, yv)
            ctx.stroke()
            _text(ctx, ml - 10, yv, f"{pct}%", size=10.5, align="right", valign="middle", color=(0.35, 0.38, 0.42))

        for c in range(0, cycles + 1, 200):
            xv = px(min(c, cycles - 1))
            ctx.set_source_rgba(0.88, 0.90, 0.92, 0.7)
            ctx.set_line_width(1.0)
            ctx.move_to(xv, mt)
            ctx.line_to(xv, mt + ph)
            ctx.stroke()
            _text(ctx, xv, mt + ph + 18, f"{c}", size=10.5, align="center", color=(0.35, 0.38, 0.42))

        ctx.set_source_rgb(0.25, 0.28, 0.32)
        ctx.set_line_width(1.4)
        ctx.move_to(ml, mt)
        ctx.line_to(ml, mt + ph)
        ctx.line_to(ml + pw, mt + ph)
        ctx.stroke()

        y80 = py(0.80)
        ctx.save()
        ctx.set_source_rgb(0.82, 0.18, 0.20)
        ctx.set_line_width(1.8)
        ctx.set_dash([7.0, 5.0])
        ctx.move_to(ml, y80)
        ctx.line_to(ml + pw, y80)
        ctx.stroke()
        ctx.restore()

        step = 4
        indices = list(range(0, cycles, step))
        if indices[-1] != cycles - 1:
            indices.append(cycles - 1)

        for eid in selected_ids:
            s = summaries[eid]
            col = PALETTE[eid]
            ctx.set_source_rgba(col[0], col[1], col[2], 0.14)
            ctx.move_to(px(indices[0]), py(min(1.0, s.acc_mean[indices[0]] + s.acc_std[indices[0]])))
            for idx in indices[1:]:
                ctx.line_to(px(idx), py(min(1.0, s.acc_mean[idx] + s.acc_std[idx])))
            for idx in reversed(indices):
                ctx.line_to(px(idx), py(max(y_min, s.acc_mean[idx] - s.acc_std[idx])))
            ctx.close_path()
            ctx.fill()

        for eid in selected_ids:
            s = summaries[eid]
            col = PALETTE[eid]
            ctx.set_source_rgb(*col)
            ctx.set_line_width(2.4)
            ctx.move_to(px(indices[0]), py(s.acc_mean[indices[0]]))
            for idx in indices[1:]:
                ctx.line_to(px(idx), py(s.acc_mean[idx]))
            ctx.stroke()

            end_x = px(cycles - 1)
            end_y = py(s.final_acc_mean)
            ctx.new_sub_path()
            ctx.arc(end_x, end_y, 3.6, 0, 2 * math.pi)
            ctx.fill()

        ctx.set_source_rgba(1.0, 1.0, 1.0, 0.90)
        ctx.rectangle(ml + pw - 216, y80 - 18, 210, 16)
        ctx.fill()
        _text(
            ctx,
            ml + pw - 8,
            y80 - 6,
            "80% Usability Threshold (t_crit)",
            size=10.5,
            bold=True,
            color=(0.78, 0.16, 0.18),
            align="right",
        )

        lx, ly = ml + 250, py(0.58)
        lw, lh = 515, 16 + len(selected_ids) * 20
        ctx.set_source_rgba(1.0, 1.0, 1.0, 0.93)
        ctx.rectangle(lx, ly, lw, lh)
        ctx.fill()
        ctx.set_source_rgb(0.80, 0.83, 0.87)
        ctx.set_line_width(1.0)
        ctx.rectangle(lx, ly, lw, lh)
        ctx.stroke()

        for i, eid in enumerate(selected_ids):
            ry = ly + 17 + i * 20
            col = PALETTE[eid]
            ctx.set_source_rgb(*col)
            ctx.set_line_width(2.6)
            ctx.move_to(lx + 10, ry)
            ctx.line_to(lx + 30, ry)
            ctx.stroke()
            final_pct = summaries[eid].final_acc_mean * 100.0
            _text(
                ctx,
                lx + 38,
                ry,
                f"{short_labels[eid]} ({final_pct:.1f}%)",
                size=9.8,
                bold=(eid == "ORI-CHAMP"),
                valign="middle",
            )

        _text(ctx, ml + pw / 2.0, h - 18, "Work Cycle t (50 cycles = 1 eight-hour shift)", size=12.0, bold=True, align="center")
        ctx.save()
        ctx.translate(22, mt + ph / 2.0)
        ctx.rotate(-math.pi / 2.0)
        _text(ctx, 0, 0, "Global System Accuracy (% rolls in home slot)", size=12.0, bold=True, align="center")
        ctx.restore()

    return _render_dual(out_dir / "fig1_entropy_decay", 1050, 600, draw)


def plot_fig2_cross_tier_contamination(
    summaries: Dict[str, ExperimentSummary], out_dir: Path
) -> Tuple[Path, Path]:
    """Chart 2: Stacked area comparison exposing Gravity Dump into Mid-shelf drawers."""
    stack_order = [
        "Home",
        "Within-Mechanism Drift",
        "Eye-Level Drawer Dump",
        "Other Cross-Tier Dump",
    ]
    disp_names = {
        "Eye-Level Drawer Dump": "Mid-Shelf Drawer Dump",
    }
    champ_long = _format_champ_labels(summaries["ORI-CHAMP"])[0]
    champ_id = champ_long.split(" ", 1)[0].rstrip(":")
    champ_sub = champ_long
    pair = [
        ("ORI-258", "ORI-258: Top: Peg | Mid: 7D_4P | Btm: 2B (90% fill, low cue)"),
        ("ORI-CHAMP", champ_sub),
    ]

    def draw(ctx: cairo.Context, w: int, h: int) -> None:
        _bg(ctx, w, h)
        ml, mr, mt, mb = 72, 32, 110, 68
        gap = 52
        pw = (w - ml - mr - gap) / 2.0
        ph = h - mt - mb

        _text(
            ctx,
            ml,
            30,
            f"Roll Location Breakdown Over Time: Baseline (ORI-258) vs. Top Pick ({champ_id})",
            size=15.5,
            bold=True,
        )
        _text(
            ctx,
            ml,
            52,
            "Share of 128 physical rolls in each location state across 1,000 work cycles (50-run mean)",
            size=11.5,
            color=(0.35, 0.39, 0.45),
        )

        lx = ml
        for key in stack_order:
            col = STACK_COLORS[key]
            ctx.set_source_rgb(*col)
            ctx.rectangle(lx, 67, 14, 12)
            ctx.fill()
            _text(ctx, lx + 20, 73, disp_names.get(key, key), size=10.8, valign="middle")
            lx += 205 if key != "Home" else 95

        for p_idx, (eid, subtitle) in enumerate(pair):
            s = summaries[eid]
            x0 = ml + p_idx * (pw + gap)
            cycles = s.cycles
            step = 5
            indices = list(range(0, cycles, step))
            if indices[-1] != cycles - 1:
                indices.append(cycles - 1)

            def px(t: float) -> float:
                return x0 + (t / max(1, cycles - 1)) * pw

            def py(val: float) -> float:
                return mt + ph - val * ph

            _text(ctx, x0 + pw / 2.0, mt - 8, subtitle, size=10.8, bold=True, align="center")

            cum_series: Dict[str, List[float]] = {}
            prev = [0.0] * cycles
            for key in stack_order:
                vals = s.contam_breakdown_t[key]
                nxt = [min(1.0, prev[t] + vals[t]) for t in range(cycles)]
                cum_series[key] = nxt
                col = STACK_COLORS[key]
                ctx.set_source_rgba(col[0], col[1], col[2], 0.86)
                ctx.move_to(px(indices[0]), py(prev[indices[0]]))
                for idx in indices[1:]:
                    ctx.line_to(px(idx), py(prev[idx]))
                for idx in reversed(indices):
                    ctx.line_to(px(idx), py(nxt[idx]))
                ctx.close_path()
                ctx.fill()
                prev = nxt

            for pct in (20, 40, 60, 80, 100):
                yv = py(pct / 100.0)
                ctx.set_source_rgba(1.0, 1.0, 1.0, 0.35)
                ctx.set_line_width(0.8)
                ctx.move_to(x0, yv)
                ctx.line_to(x0 + pw, yv)
                ctx.stroke()
                if p_idx == 0:
                    _text(ctx, x0 - 8, yv, f"{pct}%", size=10.5, align="right", valign="middle", color=(0.35, 0.38, 0.42))

            for c in range(0, cycles + 1, 250):
                xv = px(min(c, cycles - 1))
                _text(ctx, xv, mt + ph + 18, f"{c}", size=10.5, align="center", color=(0.35, 0.38, 0.42))

            ctx.set_source_rgb(0.25, 0.28, 0.32)
            ctx.set_line_width(1.3)
            ctx.move_to(x0, mt)
            ctx.line_to(x0, mt + ph)
            ctx.line_to(x0 + pw, mt + ph)
            ctx.stroke()

            home_end = s.contam_breakdown_t["Home"][-1] * 100.0
            dump_end = s.contam_breakdown_t["Eye-Level Drawer Dump"][-1] * 100.0
            _text(ctx, x0 + pw - 8, py(home_end / 200.0), f"Home: {home_end:.1f}%", size=10.5, bold=True, color=(1, 1, 1), align="right", valign="middle")
            if dump_end >= 5.0:
                mid_dump = (s.contam_breakdown_t["Home"][-1] + s.contam_breakdown_t["Within-Mechanism Drift"][-1] + 0.5 * s.contam_breakdown_t["Eye-Level Drawer Dump"][-1])
                _text(ctx, x0 + pw - 8, py(mid_dump), f"Mid Dump: {dump_end:.1f}%", size=10.5, bold=True, color=(1, 1, 1), align="right", valign="middle")

        _text(ctx, ml + pw + gap / 2.0, h - 18, "Work Cycle t", size=12.0, bold=True, align="center")
        ctx.save()
        ctx.translate(20, mt + ph / 2.0)
        ctx.rotate(-math.pi / 2.0)
        _text(ctx, 0, 0, "Share of 128 Physical Rolls (%)", size=12.0, bold=True, align="center")
        ctx.restore()

    return _render_dual(out_dir / "fig2_cross_tier_contamination", 1080, 560, draw)


def plot_fig3_slot_heatmap(
    summaries: Dict[str, ExperimentSummary], out_dir: Path
) -> Tuple[Path, Path]:
    """Chart 3: Visual slot misplacement heatmap at Cycle 1,000 (ORI-258 vs ORI-CHAMP)."""
    exp_left = summaries["ORI-258"]
    exp_right = summaries["ORI-CHAMP"]

    def draw_panel(
        ctx: cairo.Context,
        x0: float,
        y0: float,
        pw: float,
        summary: ExperimentSummary,
        title: str,
        max_mis: float,
    ) -> None:
        _text(ctx, x0, y0, title, size=12.2, bold=True)
        cur_y = y0 + 22
        by_mech: Dict[str, List[Dict[str, object]]] = {}
        for row in summary.slot_heatmap_data:
            by_mech.setdefault(str(row["mechanism_id"]), []).append(row)

        mech_titles = {
            "split_drawer_eye": "Mid Shelf (Eye-Level): Modular Split Drawers",
            "split_drawer_mid": "Btm Shelf: Modular Split Drawers",
            "wire_bin_eye": "Mid Shelf (Eye-Level): Wire Container Subzones",
            "wire_bin_mid": "Btm Shelf: Bulk Wire Container Subzones",
            "pegboard_ovh": "Top Shelf: Sliding Peg (32 Pegs, 1 Tall or 2 Short/Peg)",
        }

        for mech_id, items in by_mech.items():
            _text(ctx, x0, cur_y, f"{mech_titles.get(mech_id, mech_id)} ({len(items)} slots)", size=10.3, bold=True, color=(0.28, 0.32, 0.38))
            cur_y += 10
            if mech_id.startswith("pegboard"):
                cols = max(1, max(int(r["sub_idx"]) for r in items) + 1)
                rows_count = len({int(r["group_idx"]) for r in items})
            elif len(items) > 4:
                cols = 8
                rows_count = math.ceil(len(items) / cols)
            else:
                cols = max(1, len(items))
                rows_count = 1

            cell_gap = 4.0
            cell_w = min(68.0, (pw - (cols - 1) * cell_gap) / max(1, cols))
            cell_h = 26.0 if rows_count >= 4 else 30.0

            for idx, item in enumerate(items):
                if mech_id.startswith("pegboard"):
                    r_i = int(item["group_idx"])
                    c_i = int(item["sub_idx"])
                elif len(items) > 4:
                    r_i = idx // cols
                    c_i = idx % cols
                else:
                    r_i = 0
                    c_i = idx
                cx = x0 + c_i * (cell_w + cell_gap)
                cy = cur_y + r_i * (cell_h + cell_gap)
                mis = float(item["mean_final_misplaced"])
                occ = float(item["mean_final_occupants"])
                col = _heat_color(mis / max(0.5, max_mis))

                ctx.set_source_rgb(*col)
                ctx.rectangle(cx, cy, cell_w, cell_h)
                ctx.fill()
                ctx.set_source_rgb(0.65, 0.68, 0.72)
                ctx.set_line_width(0.8)
                ctx.rectangle(cx, cy, cell_w, cell_h)
                ctx.stroke()

                txt_col = (1.0, 1.0, 1.0) if (mis / max(0.5, max_mis)) > 0.62 else (0.14, 0.16, 0.20)
                raw_id = str(item["slot_id"])
                short_id = (
                    (raw_id.split("-", 1)[1] if raw_id.startswith(("R1-", "R2-", "R3-")) else raw_id)
                    .replace("DRW-", "D")
                    .replace("PEG-", "")
                )
                _text(
                    ctx,
                    cx + cell_w / 2.0,
                    cy + cell_h / 2.0 - 4.2,
                    short_id,
                    size=7.4,
                    color=txt_col,
                    align="center",
                    valign="middle",
                )
                _text(
                    ctx,
                    cx + cell_w / 2.0,
                    cy + cell_h / 2.0 + 5.2,
                    f"{mis:.2f} / {occ:.1f}",
                    size=8.0,
                    bold=True,
                    color=txt_col,
                    align="center",
                    valign="middle",
                )
            cur_y += rows_count * (cell_h + cell_gap) + 18

    def draw(ctx: cairo.Context, w: int, h: int) -> None:
        _bg(ctx, w, h)
        _text(
            ctx,
            34,
            30,
            "Misplaced Rolls per Compartment at Cycle 1,000",
            size=15.4,
            bold=True,
        )
        _text(
            ctx,
            34,
            51,
            "Each cell shows compartment ID and mean misplaced rolls / total rolls at Cycle 1,000 (50 runs)",
            size=11.0,
            color=(0.35, 0.39, 0.45),
        )

        max_mis = max(
            float(r["mean_final_misplaced"])
            for r in exp_left.slot_heatmap_data + exp_right.slot_heatmap_data
        )
        champ_hdr = _format_champ_labels(exp_right)[0]
        draw_panel(ctx, 34, 82, 505, exp_left, "ORI-258: Top: Peg | Mid: 7D6_4P | Btm: 2B (90% fill, low cue)", max_mis)
        draw_panel(ctx, 575, 82, 505, exp_right, champ_hdr, max_mis)

        cbx, cby, cbw, cbh = 34, h - 38, 280, 12
        for i in range(100):
            u = i / 99.0
            ctx.set_source_rgb(*_heat_color(u))
            ctx.rectangle(cbx + u * cbw, cby, cbw / 99.0 + 0.5, cbh)
            ctx.fill()
        ctx.set_source_rgb(0.45, 0.48, 0.52)
        ctx.set_line_width(0.8)
        ctx.rectangle(cbx, cby, cbw, cbh)
        ctx.stroke()
        _text(ctx, cbx, cby + 24, "0.0 misplaced", size=9.5)
        _text(ctx, cbx + cbw, cby + 24, f"{max_mis:.2f} misplaced", size=9.5, align="right")

    return _render_dual(out_dir / "fig3_slot_heatmap", 1120, 790, draw)


def plot_fig4_capacity_and_partitions(
    sweep_results: Dict[str, List[Tuple[float, float, float]]],
    summaries: Dict[str, ExperimentSummary],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Chart 4: Capacity tipping-point sweep (70%-98%) + Modular benchmark orientation comparison."""

    def draw(ctx: cairo.Context, w: int, h: int) -> None:
        _bg(ctx, w, h)
        _text(
            ctx,
            40,
            30,
            "Effect of Slot Fill Ratio and Shelf Layout on Accuracy and Relative Effort",
            size=15.4,
            bold=True,
        )
        _text(
            ctx,
            40,
            51,
            "Left: Cycle 1,000 accuracy across 70%-98% target fill | Right: Benchmark layouts ranked by accuracy (effort relative to ORI-258 = 1.00x)",
            size=11.0,
            color=(0.35, 0.39, 0.45),
        )

        ml, mt, pw, ph = 72, 92, 430, 380
        _text(ctx, ml + pw / 2.0, mt - 12, "Capacity Fill Sweep (70% to 98% Target Fill)", size=12.2, bold=True, align="center")

        x_min, x_max = 0.68, 0.99
        y_min, y_max = 0.15, 1.00

        def px(rho: float) -> float:
            return ml + ((rho - x_min) / (x_max - x_min)) * pw

        def py(acc: float) -> float:
            return mt + ph - ((acc - y_min) / (y_max - y_min)) * ph

        ctx.set_source_rgba(0.33, 0.66, 0.41, 0.12)
        ctx.rectangle(px(0.75), mt, px(0.85) - px(0.75), ph)
        ctx.fill()
        _text(ctx, px(0.80), mt + 14, "Target Buffer", size=9.5, bold=True, color=(0.20, 0.48, 0.27), align="center")
        _text(ctx, px(0.80), mt + 27, "(15-25% free)", size=9.0, color=(0.20, 0.48, 0.27), align="center")

        for pct in (20, 30, 40, 50, 60, 70, 80, 90, 100):
            yv = py(pct / 100.0)
            ctx.set_source_rgba(0.86, 0.88, 0.90, 0.8)
            ctx.set_line_width(1.0)
            ctx.move_to(ml, yv)
            ctx.line_to(ml + pw, yv)
            ctx.stroke()
            _text(ctx, ml - 8, yv, f"{pct}%", size=10.0, align="right", valign="middle")

        for rho_pct in (70, 75, 80, 85, 90, 95):
            xv = px(rho_pct / 100.0)
            ctx.set_source_rgba(0.88, 0.90, 0.92, 0.6)
            ctx.move_to(xv, mt)
            ctx.line_to(xv, mt + ph)
            ctx.stroke()
            _text(ctx, xv, mt + ph + 16, f"{rho_pct}%", size=10.0, align="center")

        ctx.save()
        ctx.set_source_rgb(0.82, 0.18, 0.20)
        ctx.set_line_width(1.5)
        ctx.set_dash([6.0, 4.0])
        ctx.move_to(ml, py(0.80))
        ctx.line_to(ml + pw, py(0.80))
        ctx.stroke()
        ctx.restore()

        sweep_styles = [
            ("TOPO053_Mid3D93D64P_Btm3D93D64P", "Top: Peg | Mid: 3D9_3D6_4P | Btm: 3D9_3D6_4P", PALETTE["ORI-CHAMP"]),
            ("TOPO007_Mid7D64P_Btm2B", "Top: Peg | Mid: 7D6_4P | Btm: 2B", PALETTE["ORI-245"]),
            ("TOPO007_BaselineCOI", "Top: Peg | Mid: 7D6_4P | Btm: 2B (COI baseline)", PALETTE["ORI-258"]),
        ]
        for key, label, col in sweep_styles:
            pts = sweep_results[key]
            ctx.set_source_rgb(*col)
            ctx.set_line_width(2.5)
            ctx.move_to(px(pts[0][0]), py(pts[0][1]))
            for rho, acc, _ in pts[1:]:
                ctx.line_to(px(rho), py(acc))
            ctx.stroke()
            for rho, acc, _ in pts:
                ctx.new_sub_path()
                ctx.arc(px(rho), py(acc), 3.8, 0, 2 * math.pi)
                ctx.fill()

        lx, ly = ml + 14, py(0.58)
        ctx.set_source_rgba(1.0, 1.0, 1.0, 0.92)
        ctx.rectangle(lx, ly, 265, 64)
        ctx.fill()
        ctx.set_source_rgb(0.82, 0.84, 0.88)
        ctx.set_line_width(0.9)
        ctx.rectangle(lx, ly, 265, 64)
        ctx.stroke()
        for idx, (_, label, col) in enumerate(sweep_styles):
            ry = ly + 14 + idx * 18
            ctx.set_source_rgb(*col)
            ctx.set_line_width(2.5)
            ctx.move_to(lx + 8, ry)
            ctx.line_to(lx + 24, ry)
            ctx.stroke()
            _text(ctx, lx + 30, ry, label, size=9.5, valign="middle")

        ctx.set_source_rgb(0.25, 0.28, 0.32)
        ctx.set_line_width(1.3)
        ctx.move_to(ml, mt)
        ctx.line_to(ml, mt + ph)
        ctx.line_to(ml + pw, mt + ph)
        ctx.stroke()
        _text(ctx, ml + pw / 2.0, mt + ph + 38, "Target Capacity Fill Ratio (rho)", size=11.5, bold=True, align="center")
        ctx.save()
        ctx.translate(22, mt + ph / 2.0)
        ctx.rotate(-math.pi / 2.0)
        _text(ctx, 0, 0, "Cycle 1,000 Accuracy (%)", size=11.5, bold=True, align="center")
        ctx.restore()

        rx0, ry0, rpw = 560, 92, 580
        _text(ctx, rx0 + rpw / 2.0, ry0 - 12, "Benchmark Layouts: Accuracy & Relative Effort", size=11.8, bold=True, align="center")

        base_lat = (
            summaries["ORI-258"].overall_latency_mean
            if "ORI-258" in summaries
            else max(s.overall_latency_mean for s in summaries.values())
        )
        ranked = sorted(summaries.values(), key=lambda s: s.final_acc_mean)
        n_bars = len(ranked)
        step_y = (ph - 12.0) / max(1, n_bars)
        bar_h = min(28.0, step_y * 0.65)
        bar_gap = step_y - bar_h
        bx_start = rx0 + 255
        bw_max = rpw - 330
        bar_label_map = {
            "ORI-258": "ORI-258: Top: Peg | Mid: 7D6_4P | Btm: 2B (90%, low cue)",
            "ORI-255": "ORI-255: Top: Peg | Mid: 7D6_4P | Btm: 2B (height split)",
            "ORI-099": "ORI-099: Top: Peg | Mid: 7D6_3P | Btm: 1B_4D6_3P (low cue)",
            "ORI-085": "ORI-085: Top: Peg | Mid: 7D6_3P | Btm: 1B_4D6_3P",
            "ORI-245": "ORI-245: Top: Peg | Mid: 7D6_4P | Btm: 1D6_2B",
            "ORI-5D9": "ORI-5D9: Top: Peg | Mid: 5D9_4P | Btm: 5D9_4P",
            "ORI-325": "ORI-325: Top: Peg | Mid: 1B_4D6_4P | Btm: 2B",
            "ORI-021": "ORI-021: Top: Peg | Mid: 7D6_4P | Btm: 1B_4D6_3P",
        }

        for idx, s in enumerate(ranked):
            by = ry0 + 8 + idx * (bar_h + bar_gap)
            eid = s.scenario.exp_id
            col = PALETTE.get(eid, (0.298, 0.447, 0.690))
            if eid == "ORI-CHAMP":
                label = _format_champ_labels(s)[1]
            else:
                label = bar_label_map.get(
                    eid,
                    f"{eid} ({s.scenario.drawer_partitions}p, {int(s.scenario.target_fill_ratio*100)}%)",
                )
            _text(
                ctx,
                bx_start - 8,
                by + bar_h / 2.0,
                label,
                size=9.5,
                align="right",
                valign="middle",
                bold=(eid == "ORI-CHAMP"),
            )

            bar_w = s.final_acc_mean * bw_max
            ctx.set_source_rgb(*col)
            ctx.rectangle(bx_start, by, bar_w, bar_h)
            ctx.fill()

        x80_bar = bx_start + 0.80 * bw_max
        ctx.save()
        ctx.set_source_rgb(0.82, 0.18, 0.20)
        ctx.set_line_width(1.4)
        ctx.set_dash([5.0, 4.0])
        ctx.move_to(x80_bar, ry0 + 4)
        ctx.line_to(x80_bar, ry0 + 8 + n_bars * (bar_h + bar_gap) - bar_gap + 4)
        ctx.stroke()
        ctx.restore()
        _text(ctx, x80_bar, mt + ph + 16, "80%", size=9.8, bold=True, color=(0.78, 0.16, 0.18), align="center")
        _text(ctx, bx_start + bw_max / 2.0, mt + ph + 38, "Final Accuracy (%) | Relative Effort (vs. ORI-258 = 1.00x)", size=11.0, bold=True, align="center")

        for idx, s in enumerate(ranked):
            by = ry0 + 8 + idx * (bar_h + bar_gap)
            bar_w = s.final_acc_mean * bw_max
            val_txt = f"{s.final_acc_mean*100:.1f}% | {s.overall_latency_mean / base_lat:.2f}x"
            ctx.save()
            ctx.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(9.6)
            ext = ctx.text_extents(val_txt)
            ctx.restore()
            tx = bx_start + bar_w + 6
            pill_w = ext.width + 8.0
            pill_h = ext.height + 6.0
            ctx.set_source_rgba(1.0, 1.0, 1.0, 0.94)
            _rounded_rect(ctx, tx - 3.0, by + bar_h / 2.0 - pill_h / 2.0, pill_w, pill_h, 3.0)
            ctx.fill()
            _text(
                ctx,
                tx,
                by + bar_h / 2.0,
                val_txt,
                size=9.6,
                bold=True,
                valign="middle",
            )

    return _render_dual(out_dir / "fig4_capacity_and_partitions", 1180, 540, draw)


def plot_fig5_orientation_search_pareto(
    orientation_records: Sequence[OrientationRecord],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Chart 5: 100-topology (500-orientation) Pareto frontier & top shelf architectures."""
    abc_palette = {
        "same_slot": (0.18, 0.39, 0.68),
        "adjacent_partition": (0.16, 0.60, 0.40),
        "same_tier_bay": (0.24, 0.65, 0.78),
        "cross_tier_height": (0.86, 0.48, 0.22),
        "random_perm": (0.58, 0.52, 0.62),
    }
    abc_labels = [
        ("same_slot", "Same-Slot (58 cm Depth)"),
        ("adjacent_partition", "Adjacent Partitions (Same Drawer)"),
        ("same_tier_bay", "Same-Tier Split Bay"),
        ("cross_tier_height", "Cross-Tier Height Segregated"),
        ("random_perm", "Random Permutations (Baseline Cloud)"),
    ]

    base_lat = next(
        (
            r.overall_latency_mean
            for r in orientation_records
            if r.row_configs == ("7D_4P", "2B", "SLIDING_PEG")
            and r.abc_colocation_mode == "cross_tier_height"
            and r.spatial_order_mode == "random_perm"
        ),
        max(r.overall_latency_mean for r in orientation_records),
    )

    def rel_eff(r: OrientationRecord) -> float:
        return r.overall_latency_mean / max(1e-9, base_lat)

    effs = [rel_eff(r) for r in orientation_records]
    accs = [r.final_acc_mean for r in orientation_records]
    x_min = min(0.30, math.floor(min(effs) * 10.0) / 10.0)
    x_max = max(1.10, math.ceil(max(effs) * 10.0) / 10.0)
    y_min = max(0.15, math.floor(min(accs) * 20.0 - 1.0) / 20.0)
    y_max = 1.00

    pareto = sorted(
        [r for r in orientation_records if r.is_pareto_optimal],
        key=lambda r: rel_eff(r),
    )
    champion = max(orientation_records, key=lambda r: r.composite_score)

    by_topo: Dict[str, List[OrientationRecord]] = {}
    for r in orientation_records:
        key = f"{r.topo_id}: {r.topo_label}"
        by_topo.setdefault(key, []).append(r)

    topo_stats = []
    for k, items in by_topo.items():
        best_r = max(items, key=lambda x: x.composite_score)
        mean_acc = statistics.mean(x.final_acc_mean for x in items)
        topo_stats.append((k, mean_acc, best_r.final_acc_mean, best_r.overall_latency_mean, best_r.ori_id))
    topo_stats.sort(key=lambda t: (-t[2], t[3]))
    top_n = topo_stats[:12]

    def draw(ctx: cairo.Context, w: int, h: int) -> None:
        _bg(ctx, w, h)
        _text(
            ctx,
            38,
            30,
            "Layout Search Across 165 Feasible Wire Shelf Topologies (825 Configurations)",
            size=15.0,
            bold=True,
        )
        _text(
            ctx,
            38,
            51,
            "165 physically feasible setups (requiring >=1 9in drawer row or wire bin) x 5 policies",
            size=11.0,
            color=(0.35, 0.39, 0.45),
        )

        ml, mt, pw, ph = 72, 92, 500, 415
        _text(
            ctx,
            ml + pw / 2.0,
            mt - 12,
            "Pareto Frontier: Cycle 1,000 Accuracy vs. Relative Effort (165 Topologies)",
            size=11.8,
            bold=True,
            align="center",
        )

        def px(eff: float) -> float:
            return ml + ((eff - x_min) / max(1e-6, x_max - x_min)) * pw

        def py(acc: float) -> float:
            return mt + ph - ((acc - y_min) / max(1e-6, y_max - y_min)) * ph

        pct_step = 10 if (y_max - y_min) >= 0.45 else 5
        start_pct = int(math.ceil((y_min * 100.0) / pct_step) * pct_step)
        for pct in range(start_pct, 101, pct_step):
            yv = py(pct / 100.0)
            ctx.set_source_rgba(0.86, 0.88, 0.90, 0.8)
            ctx.set_line_width(1.0)
            ctx.move_to(ml, yv)
            ctx.line_to(ml + pw, yv)
            ctx.stroke()
            _text(ctx, ml - 8, yv, f"{pct}%", size=9.8, align="right", valign="middle", color=(0.35, 0.38, 0.42))

        start_tick = int(math.ceil(x_min * 10.0))
        end_tick = int(math.floor(x_max * 10.0))
        for tick_i in range(start_tick, end_tick + 1):
            xv_val = tick_i / 10.0
            xv = px(xv_val)
            ctx.set_source_rgba(0.88, 0.90, 0.92, 0.65)
            ctx.set_line_width(1.0)
            ctx.move_to(xv, mt)
            ctx.line_to(xv, mt + ph)
            ctx.stroke()
            _text(ctx, xv, mt + ph + 16, f"{xv_val:.1f}x", size=9.5, align="center", color=(0.35, 0.38, 0.42))

        if y_min < 0.80:
            y80 = py(0.80)
            ctx.save()
            ctx.set_source_rgb(0.82, 0.18, 0.20)
            ctx.set_line_width(1.4)
            ctx.set_dash([6.0, 4.0])
            ctx.move_to(ml, y80)
            ctx.line_to(ml + pw, y80)
            ctx.stroke()
            ctx.restore()
            _text(ctx, ml + pw - 6, y80 + 11, "80% Usability Floor", size=9.0, bold=True, color=(0.78, 0.16, 0.18), align="right")

        for rec in orientation_records:
            cx = px(rel_eff(rec))
            cy = py(rec.final_acc_mean)
            if rec.spatial_order_mode == "random_perm":
                col = abc_palette["random_perm"]
                ctx.set_source_rgba(col[0], col[1], col[2], 0.55)
                ctx.set_line_width(1.3)
                ctx.new_sub_path()
                ctx.arc(cx, cy, 3.3, 0, 2 * math.pi)
                ctx.stroke()
            else:
                col = abc_palette.get(rec.abc_colocation_mode, (0.4, 0.4, 0.4))
                alpha = 0.78 if rec.overflow_mode == "hybrid_reserve" else 0.48
                ctx.set_source_rgba(col[0], col[1], col[2], alpha)
                ctx.new_sub_path()
                ctx.arc(cx, cy, 3.6, 0, 2 * math.pi)
                ctx.fill()

        if pareto:
            ctx.save()
            ctx.set_source_rgb(0.10, 0.12, 0.16)
            ctx.set_line_width(1.8)
            ctx.set_dash([5.0, 3.0])
            ctx.move_to(px(rel_eff(pareto[0])), py(pareto[0].final_acc_mean))
            for pr in pareto[1:]:
                ctx.line_to(px(rel_eff(pr)), py(pr.final_acc_mean))
            ctx.stroke()
            ctx.restore()
            for pr in pareto:
                ctx.set_source_rgb(1.0, 1.0, 1.0)
                ctx.new_sub_path()
                ctx.arc(px(rel_eff(pr)), py(pr.final_acc_mean), 5.2, 0, 2 * math.pi)
                ctx.fill()
                ctx.set_source_rgb(0.10, 0.12, 0.16)
                ctx.set_line_width(1.6)
                ctx.new_sub_path()
                ctx.arc(px(rel_eff(pr)), py(pr.final_acc_mean), 5.2, 0, 2 * math.pi)
                ctx.stroke()

        ch_x = px(rel_eff(champion))
        ch_y = py(champion.final_acc_mean)
        ctx.set_source_rgb(0.85, 0.15, 0.22)
        ctx.set_line_width(2.2)
        ctx.new_sub_path()
        ctx.arc(ch_x, ch_y, 7.5, 0, 2 * math.pi)
        ctx.stroke()
        ctx.new_sub_path()
        ctx.arc(ch_x, ch_y, 3.8, 0, 2 * math.pi)
        ctx.fill()

        box_x = ml + 266
        box_y = mt + 8
        box_w = 230
        box_h = 44
        ctx.save()
        ctx.set_source_rgb(0.85, 0.15, 0.22)
        ctx.set_line_width(1.2)
        ctx.move_to(ch_x + 7.5, ch_y - 2.0)
        ctx.line_to(box_x, box_y + box_h / 2.0)
        ctx.stroke()
        ctx.restore()

        ctx.set_source_rgba(1.0, 1.0, 1.0, 0.96)
        ctx.rectangle(box_x, box_y, box_w, box_h)
        ctx.fill()
        ctx.set_source_rgb(0.85, 0.15, 0.22)
        ctx.set_line_width(1.2)
        ctx.rectangle(box_x, box_y, box_w, box_h)
        ctx.stroke()
        _text(
            ctx,
            box_x + 8,
            box_y + 13,
            f"Best layout: {champion.ori_id} ({champion.topo_id})",
            size=9.0,
            bold=True,
            color=(0.75, 0.12, 0.18),
        )
        _text(
            ctx,
            box_x + 8,
            box_y + 26,
            champion.topo_label,
            size=8.2,
            color=(0.18, 0.20, 0.24),
        )
        _text(
            ctx,
            box_x + 8,
            box_y + 39,
            f"{champion.abc_colocation_mode} | {champion.final_acc_mean*100:.1f}% Acc | {rel_eff(champion):.2f}x Effort",
            size=8.2,
            bold=True,
            color=(0.14, 0.45, 0.30),
        )

        ctx.set_source_rgb(0.25, 0.28, 0.32)
        ctx.set_line_width(1.3)
        ctx.move_to(ml, mt)
        ctx.line_to(ml, mt + ph)
        ctx.line_to(ml + pw, mt + ph)
        ctx.stroke()

        lx, ly = ml + pw - 238, mt + ph - 110
        ctx.set_source_rgba(1.0, 1.0, 1.0, 0.93)
        ctx.rectangle(lx, ly, 230, 102)
        ctx.fill()
        ctx.set_source_rgb(0.80, 0.83, 0.87)
        ctx.set_line_width(0.9)
        ctx.rectangle(lx, ly, 230, 102)
        ctx.stroke()
        for idx, (mode_k, label_txt) in enumerate(abc_labels):
            ry = ly + 14 + idx * 17.5
            col = abc_palette[mode_k]
            ctx.set_source_rgb(*col)
            if mode_k == "random_perm":
                ctx.set_line_width(1.5)
                ctx.new_sub_path()
                ctx.arc(lx + 12, ry, 3.8, 0, 2 * math.pi)
                ctx.stroke()
            else:
                ctx.new_sub_path()
                ctx.arc(lx + 12, ry, 4.0, 0, 2 * math.pi)
                ctx.fill()
            _text(ctx, lx + 22, ry, label_txt, size=9.0, valign="middle")

        _text(ctx, ml + pw / 2.0, mt + ph + 38, "Relative Retrieval & Return Effort (Baseline ORI-258 = 1.00x)", size=11.0, bold=True, align="center")
        ctx.save()
        ctx.translate(22, mt + ph / 2.0)
        ctx.rotate(-math.pi / 2.0)
        _text(ctx, 0, 0, "Cycle 1,000 Accuracy (% rolls at home)", size=11.5, bold=True, align="center")
        ctx.restore()

        rx0, ry0, rpw = 615, 92, 585
        _text(ctx, rx0 + rpw / 2.0, ry0 - 12, "Top 12 Modular Shelf Architectures: Mean vs. Best Orientation", size=12.0, bold=True, align="center")

        bx_start = rx0 + 268
        bw_max = rpw - 348
        row_h = 23.0
        row_gap = 10.5

        for idx, (tlabel, mean_acc, best_acc, best_lat, _) in enumerate(top_n):
            by = ry0 + 10 + idx * (row_h + row_gap)
            _text(
                ctx,
                bx_start - 8,
                by + row_h / 2.0,
                tlabel,
                size=8.6,
                bold=(idx == 0),
                align="right",
                valign="middle",
                color=(0.14, 0.16, 0.20) if idx < 3 else (0.30, 0.33, 0.38),
            )
            def bx(a: float) -> float:
                u = max(0.0, min(1.0, (a - 0.30) / 0.70))
                return bx_start + u * bw_max

            ctx.set_source_rgb(0.93, 0.95, 0.97)
            ctx.rectangle(bx_start, by + 3, bw_max, row_h - 6)
            ctx.fill()

            w_mean = max(2.0, bx(mean_acc) - bx_start)
            ctx.set_source_rgba(0.45, 0.60, 0.78, 0.72)
            ctx.rectangle(bx_start, by + 5, w_mean, row_h - 10)
            ctx.fill()

            x_mean = bx(mean_acc)
            x_best = bx(best_acc)
            ctx.set_source_rgb(0.16, 0.60, 0.40) if idx < 3 else ctx.set_source_rgb(0.24, 0.45, 0.70)
            ctx.set_line_width(2.0)
            ctx.move_to(x_mean, by + row_h / 2.0)
            ctx.line_to(x_best, by + row_h / 2.0)
            ctx.stroke()
            ctx.new_sub_path()
            ctx.arc(x_best, by + row_h / 2.0, 4.5, 0, 2 * math.pi)
            ctx.fill()

            _text(
                ctx,
                bx_start + bw_max + 8,
                by + row_h / 2.0,
                f"Best {best_acc*100:.1f}% ({best_lat / max(1e-9, base_lat):.2f}x)",
                size=8.6,
                bold=(idx == 0),
                valign="middle",
            )

        x80_r = bx_start + ((0.80 - 0.30) / 0.70) * bw_max
        ctx.save()
        ctx.set_source_rgb(0.82, 0.18, 0.20)
        ctx.set_line_width(1.3)
        ctx.set_dash([5.0, 4.0])
        ctx.move_to(x80_r, ry0 + 4)
        ctx.line_to(x80_r, ry0 + 10 + len(top_n) * (row_h + row_gap) - 4)
        ctx.stroke()
        ctx.restore()
        for pct in (30, 50, 70, 80, 100):
            xt = bx_start + ((pct / 100.0 - 0.30) / 0.70) * bw_max
            _text(ctx, xt, ry0 + 10 + len(top_n) * (row_h + row_gap) + 12, f"{pct}%", size=9.2, align="center", color=(0.35, 0.38, 0.42))

        _text(
            ctx,
            bx_start + bw_max / 2.0,
            mt + ph + 38,
            "Cycle 1,000 Accuracy (Bar = Hardware Mean, Dot = Best Positioning)",
            size=10.8,
            bold=True,
            align="center",
        )

    return _render_dual(out_dir / "fig5_orientation_search_pareto", 1225, 580, draw)
