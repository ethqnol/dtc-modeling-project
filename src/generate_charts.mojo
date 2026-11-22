from std.collections import List
from std.math import ceil, floor, pi
from std.os import makedirs
from std.pathlib import Path

from canvas import (
    Canvas,
    Color,
    FillRule,
    FontCache,
    FontSlant,
    FontWeight,
    LinearGradient,
    Path as CanvasPath,
    SvgCanvas,
    TextAlign,
    measure_text,
    write_png,
    write_svg,
)
from canvas.vector.draw_target import DrawTarget
from dataviz import AnyChart, LegendPosition, LineStyle, Plot, Theme, save_layers
from dataviz.core.colors import WHITE
from dataviz.core.scale import TickFormat
from dataviz.core.text import _replay_text_requests
from dataviz.layers import _render_layers_generic
from dataviz.rendering import _all_at_dpi

from src.format_utils import fmt_float
from src.orientation_search import OrientationRecord
from src.simulation_engine import ExperimentSummary, SlotHeatmapEntry


@fieldwise_init
struct SweepSeries(Copyable, Movable):
    var label: String
    var rhos: List[Float64]
    var accs: List[Float64]
    var lats: List[Float64]


@fieldwise_init
struct _PanelBox(Copyable, Movable):
    var px0: Int
    var py0: Int
    var px1: Int
    var py1: Int


@fieldwise_init
struct _TopoStat(Copyable, Movable):
    var label: String
    var mean_acc: Float64
    var best_acc: Float64
    var best_lat: Float64


def _rgb(r: Float64, g: Float64, b: Float64, a: Float64 = 1.0) -> Color:
    """Convert normalized [0, 1] RGBA channels into an 8-bit Color."""
    return Color(
        UInt8(max(0.0, min(255.0, r * 255.0 + 0.5))),
        UInt8(max(0.0, min(255.0, g * 255.0 + 0.5))),
        UInt8(max(0.0, min(255.0, b * 255.0 + 0.5))),
        UInt8(max(0.0, min(255.0, a * 255.0 + 0.5))),
    )


def _blend_white(c: Color, alpha: Float64) -> Color:
    """Pre-blend an RGBA color over solid white to obtain an opaque tint."""
    return Color(
        UInt8(alpha * Float64(c.r) + (1.0 - alpha) * 255.0 + 0.5),
        UInt8(alpha * Float64(c.g) + (1.0 - alpha) * 255.0 + 0.5),
        UInt8(alpha * Float64(c.b) + (1.0 - alpha) * 255.0 + 0.5),
        255,
    )


def _palette_color(exp_id: String) -> Color:
    """Return the fixed series color assigned to each benchmark layout."""
    if exp_id == "ORI-258":
        return Color(196, 78, 82)
    if exp_id == "ORI-255":
        return Color(129, 114, 179)
    if exp_id == "ORI-099":
        return Color(147, 120, 96)
    if exp_id == "ORI-085":
        return Color(221, 132, 82)
    if exp_id == "ORI-245":
        return Color(85, 168, 104)
    if exp_id == "ORI-121":
        return Color(204, 121, 167)
    if exp_id == "ORI-325":
        return Color(100, 181, 205)
    if exp_id == "ORI-021":
        return Color(76, 114, 176)
    if exp_id == "ORI-CHAMP":
        return Color(23, 139, 91)
    return Color(76, 114, 176)


def _abc_palette_color(mode: String, alpha: UInt8 = 255) -> Color:
    """Return the placement policy scatter color with explicit alpha."""
    var base = Color(102, 102, 102)
    if mode == "same_slot":
        base = Color(46, 99, 173)
    elif mode == "adjacent_partition":
        base = Color(41, 153, 102)
    elif mode == "same_tier_bay":
        base = Color(61, 166, 199)
    elif mode == "cross_tier_height":
        base = Color(219, 122, 56)
    elif mode == "random_perm":
        base = Color(148, 133, 158)
    return Color(base.r, base.g, base.b, alpha)


def _heat_color(val_norm: Float64) -> Color:
    """Sequential colorblind-safe ramp from pale slate to deep red."""
    var v = max(0.0, min(1.0, val_norm))
    var offsets: List[Float64] = [0.00, 0.30, 0.65, 1.00]
    var rs: List[Float64] = [242.25, 252.45, 249.90, 188.70]
    var gs: List[Float64] = [247.35, 232.05, 142.80, 20.40]
    var bs: List[Float64] = [252.45, 165.75, 61.20, 38.25]
    for i in range(3):
        if v <= offsets[i + 1]:
            var u = (v - offsets[i]) / max(1e-9, offsets[i + 1] - offsets[i])
            return Color(
                UInt8(rs[i] + u * (rs[i + 1] - rs[i]) + 0.5),
                UInt8(gs[i] + u * (gs[i + 1] - gs[i]) + 0.5),
                UInt8(bs[i] + u * (bs[i + 1] - bs[i]) + 0.5),
            )
    return Color(189, 20, 38)


def _find_summary(summaries: List[ExperimentSummary], exp_id: String) -> Int:
    """Return index of exp_id in summaries, or -1 if absent."""
    for i in range(len(summaries)):
        if summaries[i].scenario.exp_id == exp_id:
            return i
    return -1


def _find_sweep(sweeps: List[SweepSeries], label: String) -> Int:
    """Return index of label in sweeps, or -1 if absent."""
    for i in range(len(sweeps)):
        if sweeps[i].label == label:
            return i
    return -1


def _row_short_name(cfg_name: String) -> String:
    if cfg_name == "SLIDING_PEG":
        return "Peg"
    return cfg_name


def _champ_ori_id(summary: ExperimentSummary) -> String:
    var name = summary.scenario.name
    if name.startswith("Champion ORI-"):
        var parts = name.split(" ")
        if len(parts) >= 2:
            return String(parts[1])
    return summary.scenario.exp_id


def _format_champ_label(summary: ExperimentSummary) -> String:
    var ori_id = _champ_ori_id(summary)
    var sc = summary.scenario.copy()
    return (
        ori_id
        + ": Top: "
        + _row_short_name(sc.row2)
        + " | Mid: "
        + _row_short_name(sc.row0)
        + " | Btm: "
        + _row_short_name(sc.row1)
    )


def _bench_short_label(summary: ExperimentSummary, champ_label: String) -> String:
    var eid = summary.scenario.exp_id
    if eid == "ORI-CHAMP":
        return champ_label
    if eid == "ORI-121":
        return "ORI-121: Top: Peg | Mid: 5D9_4P | Btm: 5D9_4P"
    if eid == "ORI-021":
        return "ORI-021: Top: Peg | Mid: 7D6_4P | Btm: 1B_4D6_3P"
    if eid == "ORI-245":
        return "ORI-245: Top: Peg | Mid: 7D6_4P | Btm: 1D6_2B"
    if eid == "ORI-325":
        return "ORI-325: Top: Peg | Mid: 1B_4D6_4P | Btm: 2B"
    if eid == "ORI-099":
        return "ORI-099: Top: Peg | Mid: 7D6_3P | Btm: 1B_4D6_3P (low cue)"
    if eid == "ORI-258":
        return "ORI-258: Top: Peg | Mid: 7D6_4P | Btm: 2B (90% fill, low cue)"
    return eid


def _bench_bar_label(summary: ExperimentSummary, champ_label: String) -> String:
    var eid = summary.scenario.exp_id
    if eid == "ORI-CHAMP":
        return champ_label
    if eid == "ORI-258":
        return "ORI-258: Top: Peg | Mid: 7D6_4P | Btm: 2B (90%, low cue)"
    if eid == "ORI-255":
        return "ORI-255: Top: Peg | Mid: 7D6_4P | Btm: 2B (height split)"
    if eid == "ORI-099":
        return "ORI-099: Top: Peg | Mid: 7D6_3P | Btm: 1B_4D6_3P (low cue)"
    if eid == "ORI-085":
        return "ORI-085: Top: Peg | Mid: 7D6_3P | Btm: 1B_4D6_3P"
    if eid == "ORI-245":
        return "ORI-245: Top: Peg | Mid: 7D6_4P | Btm: 1D6_2B"
    if eid == "ORI-121":
        return "ORI-121: Top: Peg | Mid: 5D9_4P | Btm: 5D9_4P"
    if eid == "ORI-325":
        return "ORI-325: Top: Peg | Mid: 1B_4D6_4P | Btm: 2B"
    if eid == "ORI-021":
        return "ORI-021: Top: Peg | Mid: 7D6_4P | Btm: 1B_4D6_3P"
    return (
        eid
        + " ("
        + String(summary.scenario.drawer_partitions)
        + "p, "
        + String(Int(summary.scenario.target_fill_ratio * 100.0 + 0.5))
        + "%)"
    )


def _mech_title(mech_id: String) -> String:
    if mech_id == "split_drawer_eye":
        return "Mid Shelf (Eye-Level): Modular Split Drawers"
    if mech_id == "split_drawer_mid":
        return "Btm Shelf: Modular Split Drawers"
    if mech_id == "wire_bin_eye":
        return "Mid Shelf (Eye-Level): Wire Container Subzones"
    if mech_id == "wire_bin_mid":
        return "Btm Shelf: Bulk Wire Container Subzones"
    if mech_id == "pegboard_ovh":
        return "Top Shelf: Sliding Peg (32 Pegs, 1 Tall or 2 Short/Peg)"
    return mech_id


def _short_slot_id(raw_id: String) -> String:
    var out = raw_id
    if raw_id.startswith("R1-") or raw_id.startswith("R2-") or raw_id.startswith("R3-"):
        out = String(raw_id[byte=3:])
    return out.replace("DRW-", "D").replace("PEG-", "")


def _stroke_rect[
    T: DrawTarget
](
    mut target: T,
    x: Float64,
    y: Float64,
    w: Float64,
    h: Float64,
    color: Color,
    width: Float64,
):
    """Outline an axis-aligned rectangle via four anti-aliased edges."""
    target.draw_line_aa(x, y, x + w, y, color, width=width)
    target.draw_line_aa(x + w, y, x + w, y + h, color, width=width)
    target.draw_line_aa(x + w, y + h, x, y + h, color, width=width)
    target.draw_line_aa(x, y + h, x, y, color, width=width)


def _fill_rounded_rect[
    T: DrawTarget
](
    mut target: T,
    x: Float64,
    y: Float64,
    w: Float64,
    h: Float64,
    r_in: Float64,
    color: Color,
) raises:
    """Fill a rounded rectangle cleanly across both raster and SVG targets."""
    var r = min(r_in, min(w * 0.5, h * 0.5))
    var p = CanvasPath()
    p.move_to(x + r, y)
    p.line_to(x + w - r, y)
    p.quad_curve_to(x + w, y, x + w, y + r)
    p.line_to(x + w, y + h - r)
    p.quad_curve_to(x + w, y + h, x + w - r, y + h)
    p.line_to(x + r, y + h)
    p.quad_curve_to(x, y + h, x, y + h - r)
    p.line_to(x, y + r)
    p.quad_curve_to(x, y, x + r, y)
    p.close()
    target.fill_path_aa(p, color, fill_rule=FillRule.NONZERO)


def _text[
    T: DrawTarget
](
    mut target: T,
    x: Float64,
    y: Float64,
    txt: String,
    size: Float64,
    color: Color = Color(31, 36, 46),
    bold: Bool = False,
    align: TextAlign = TextAlign.LEFT,
    valign: String = "baseline",
    rotation: Float64 = 0.0,
    *,
    mut cache: FontCache,
) raises:
    """Unified DrawTarget text helper supporting middle/top/baseline anchoring."""
    var ty = y
    if valign == "middle":
        ty = y + size * 0.35
    elif valign == "top":
        ty = y + size * 0.80
    target.draw_text(
        x,
        ty,
        txt,
        color,
        size,
        family="Sans",
        slant=FontSlant.NORMAL,
        weight=FontWeight.BOLD if bold else FontWeight.NORMAL,
        rotation=rotation,
        align=align,
        cache=cache,
    )


def _draw_header[
    T: DrawTarget
](
    mut target: T,
    w: Int,
    h: Int,
    x: Float64,
    title: String,
    subtitle: String,
    title_size: Float64,
    s: Float64,
    mut cache: FontCache,
) raises:
    target.fill_rect(0, 0, w, h, WHITE)
    _text(
        target,
        x * s,
        30.0 * s,
        title,
        size=title_size * s,
        bold=True,
        cache=cache,
    )
    _text(
        target,
        x * s,
        51.0 * s,
        subtitle,
        size=11.0 * s,
        color=Color(89, 99, 115),
        cache=cache,
    )


def _render_panel_layers[
    T: DrawTarget
](
    mut target: T,
    plots: List[AnyChart],
    ox0: Int,
    oy0: Int,
    ox1: Int,
    oy1: Int,
    dpi: Float64,
    mut cache: FontCache,
) raises -> _PanelBox:
    """Render dataviz Plot layers into a sub-region and replay their text requests."""
    var scaled = _all_at_dpi(plots, dpi, "generate_charts") if dpi != 72.0 else plots.copy()
    var res = _render_layers_generic(target, scaled, ox0, oy0, ox1, oy1, cache=cache)
    _replay_text_requests(target, res.text_requests, cache)
    return _PanelBox(res.px0, res.py0, res.px1, res.py1)


def _sample_indices(cycles: Int, step: Int) -> List[Int]:
    var out = List[Int](capacity=(cycles // step) + 2)
    var t = 0
    while t < cycles:
        out.append(t)
        t += step
    if out[len(out) - 1] != cycles - 1:
        out.append(cycles - 1)
    return out^


def _plot_fig1_entropy_decay(
    summaries: List[ExperimentSummary], out_dir: String
) raises:
    """Chart 1: Accuracy decay curves with +/-1 SD bands and 80% usability line."""
    var candidate_ids: List[String] = [
        "ORI-CHAMP",
        "ORI-121",
        "ORI-021",
        "ORI-245",
        "ORI-325",
        "ORI-099",
        "ORI-258",
    ]
    var selected = List[ExperimentSummary]()
    for i in range(len(candidate_ids)):
        var idx = _find_summary(summaries, candidate_ids[i])
        if idx >= 0:
            selected.append(summaries[idx].copy())
    if len(selected) == 0:
        return

    var champ_idx = _find_summary(summaries, "ORI-CHAMP")
    var champ_label = _format_champ_label(summaries[champ_idx]) if champ_idx >= 0 else "ORI-CHAMP"
    var cycles = selected[0].cycles
    var indices = _sample_indices(cycles, 4)

    var plots = List[AnyChart](capacity=len(selected) * 2)
    var w = 1240
    var h = 580

    for i in range(len(selected)):
        var s = selected[i].copy()
        var eid = s.scenario.exp_id
        var col = _palette_color(eid)
        var band_col = Color(col.r, col.g, col.b, 36)

        var xs = List[Float64](capacity=len(indices))
        var ys = List[Float64](capacity=len(indices))
        var y_lo = List[Float64](capacity=len(indices))
        var y_hi = List[Float64](capacity=len(indices))
        for k in range(len(indices)):
            var idx = indices[k]
            var m = s.acc_mean[idx]
            var sd = s.acc_std[idx]
            var lo = max(0.15, min(1.0, m - sd))
            var hi = max(lo, min(1.0, m + sd))
            xs.append(Float64(idx))
            ys.append(m)
            y_lo.append(lo)
            y_hi.append(hi)

        var series_label = (
            _bench_short_label(s, champ_label)
            + " ("
            + fmt_float(s.final_acc_mean * 100.0, 1)
            + "%)"
        )
        var th = Theme(
            background=WHITE,
            mark_color=col,
            axis_color=Color(64, 71, 82),
            gridline_color=Color(220, 224, 230),
            text_color=Color(31, 36, 46),
            font_size=11.0,
            point_radius=4.0,
            line_width=2.4,
            margin_left=68,
            margin_right=18,
            margin_top=24,
            margin_bottom=44,
            title_font_size=16.5,
            subtitle_font_size=11.5,
            subtitle_color=Color(89, 99, 115),
            axis_title_font_size=12.0,
            annotation_color=Color(209, 46, 51),
            annotation_area_color=band_col,
            legend_row_gap=10,
            annotation_line_style=LineStyle.DASHED,
            legend_position=LegendPosition.RIGHT,
            y_tick_format=TickFormat.PERCENT,
        )
        var p = (
            Plot()
            .mark_line()
            .theme(th)
            .encode(x=xs, y=ys)
            .annotate_band(xs, y_lo, y_hi)
            .series_name(series_label)
            .size(w, h)
        )
        if i == 0:
            p = (
                p^
                .labels(
                    title="Home Slot Accuracy Over 1,000 Work Cycles",
                    subtitle="Share of 128 rolls in their assigned home slot across 20 shifts (50-run mean, shaded band = +/- 1 SD)",
                    x_title="Work Cycle t (50 cycles = 1 eight-hour shift)",
                    y_title="Global System Accuracy (% rolls in home slot)",
                )
                .scale_x_domain(0.0, Float64(cycles - 1))
                .scale_y_domain(0.15, 1.02)
                .scale_x_ticks([0.0, 200.0, 400.0, 600.0, 800.0, 1000.0])
                .scale_y_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                .annotate_line(0.80, label="80% Usability Threshold (t_crit)")
            )
        plots.append(p.erased())

    for i in range(len(selected)):
        var s = selected[i].copy()
        var col = _palette_color(s.scenario.exp_id)
        var ex: List[Float64] = [Float64(cycles - 1)]
        var ey: List[Float64] = [s.final_acc_mean]
        var ep = (
            Plot()
            .mark_point()
            .theme(Theme(mark_color=col, point_radius=4.0, show_legend=False))
            .encode(x=ex, y=ey)
            .size(w, h)
        )
        plots.append(ep.erased())

    save_layers(plots, out_dir + "/fig1_entropy_decay.svg")
    save_layers(plots, out_dir + "/fig1_entropy_decay.png", dpi=144.0)


def _draw_fig2[
    T: DrawTarget
](
    mut target: T,
    left_s: ExperimentSummary,
    right_s: ExperimentSummary,
    w_log: Int,
    h_log: Int,
    s: Float64,
    mut cache: FontCache,
) raises:
    var w = Int(Float64(w_log) * s + 0.5)
    var h = Int(Float64(h_log) * s + 0.5)
    var champ_id = _champ_ori_id(right_s)
    var champ_sub = _format_champ_label(right_s)

    _draw_header(
        target,
        w,
        h,
        40.0,
        "Roll Location Breakdown Over Time: Baseline (ORI-258) vs. Top Pick (" + champ_id + ")",
        "Share of 128 physical rolls in each location state across 1,000 work cycles (50-run mean)",
        15.5,
        s,
        cache,
    )

    var legend_names: List[String] = [
        "Home",
        "Within-Mechanism Drift",
        "Mid-Shelf Drawer Dump",
        "Other Cross-Tier Dump",
    ]
    var stack_colors: List[Color] = [
        Color(76, 114, 176),
        Color(221, 132, 82),
        Color(196, 78, 82),
        Color(129, 114, 179),
    ]
    var lx = 40.0 * s
    for i in range(4):
        target.fill_rect(lx, 67.0 * s, 14.0 * s, 12.0 * s, _blend_white(stack_colors[i], 0.86))
        _text(
            target,
            lx + 20.0 * s,
            73.0 * s,
            legend_names[i],
            size=10.8 * s,
            valign="middle",
            cache=cache,
        )
        lx += (95.0 if i == 0 else 205.0) * s

    var dpi = 72.0 * s
    var panel_x0s: List[Float64] = [30.0, 548.0]
    var panel_x1s: List[Float64] = [528.0, 1052.0]
    var subtitles: List[String] = [
        "ORI-258: Top: Peg | Mid: 7D6_4P | Btm: 2B (90% fill, low cue)",
        champ_sub,
    ]
    var grid_col = Color(255, 255, 255, 92)

    for p_idx in range(2):
        var sum_ref = left_s.copy() if p_idx == 0 else right_s.copy()
        var cycles = sum_ref.cycles
        var indices = _sample_indices(cycles, 5)
        var xs = List[Float64](capacity=len(indices))
        var c0 = List[Float64](capacity=len(indices))
        var c1 = List[Float64](capacity=len(indices))
        var c2 = List[Float64](capacity=len(indices))
        var c3 = List[Float64](capacity=len(indices))
        for k in range(len(indices)):
            var t = indices[k]
            var v0 = sum_ref.home_t[t]
            var v1 = min(1.0, v0 + sum_ref.within_drift_t[t])
            var v2 = min(1.0, v1 + sum_ref.eye_dump_mean[t])
            var v3 = min(1.0, v2 + sum_ref.other_contam_t[t])
            xs.append(Float64(t))
            c0.append(v0)
            c1.append(v1)
            c2.append(v2)
            c3.append(v3)

        var ox0 = Int(panel_x0s[p_idx] * s + 0.5)
        var oy0 = Int(100.0 * s + 0.5)
        var ox1 = Int(panel_x1s[p_idx] * s + 0.5)
        var oy1 = Int(526.0 * s + 0.5)

        var plots = List[AnyChart](capacity=4)
        for layer_k in range(4):
            var rev = 3 - layer_k
            var ys = c3.copy()
            if rev == 0:
                ys = c0.copy()
            elif rev == 1:
                ys = c1.copy()
            elif rev == 2:
                ys = c2.copy()

            var th = Theme(
                background=WHITE,
                mark_color=_blend_white(stack_colors[rev], 0.86),
                axis_color=Color(64, 71, 82),
                text_color=Color(89, 97, 107),
                font_size=10.5,
                margin_left=44 if p_idx == 0 else 28,
                margin_right=8,
                margin_top=12,
                margin_bottom=30,
                show_gridlines=False,
                show_legend=False,
                y_tick_format=TickFormat.PERCENT,
            )
            var ap = (
                Plot()
                .mark_area()
                .theme(th)
                .encode(x=xs, y=ys)
                .size(ox1 - ox0, oy1 - oy0)
            )
            if layer_k == 0:
                ap = (
                    ap^
                    .scale_x_domain(0.0, Float64(cycles - 1))
                    .scale_y_domain(0.0, 1.0)
                )
            plots.append(ap.erased())

        var box = _render_panel_layers(target, plots, ox0, oy0, ox1, oy1, dpi, cache)

        _text(
            target,
            0.5 * Float64(box.px0 + box.px1),
            Float64(box.py0) - 10.0 * s,
            subtitles[p_idx],
            size=10.8 * s,
            bold=True,
            align=TextAlign.CENTER,
            cache=cache,
        )

        var ph = Float64(box.py1 - box.py0)
        for pct_idx in range(1, 6):
            var frac = Float64(pct_idx) * 0.20
            var yv = Float64(box.py1) - frac * ph
            target.draw_line_aa(Float64(box.px0), yv, Float64(box.px1), yv, grid_col, width=0.8 * s)

        var end_home = sum_ref.home_t[cycles - 1]
        var end_within = sum_ref.within_drift_t[cycles - 1]
        var end_dump = sum_ref.eye_dump_mean[cycles - 1]
        _text(
            target,
            Float64(box.px1) - 8.0 * s,
            Float64(box.py1) - (end_home * 0.5) * ph,
            "Home: " + fmt_float(end_home * 100.0, 1) + "%",
            size=10.5 * s,
            bold=True,
            color=WHITE,
            align=TextAlign.RIGHT,
            valign="middle",
            cache=cache,
        )
        if end_dump * 100.0 >= 5.0:
            var mid_dump = end_home + end_within + 0.5 * end_dump
            _text(
                target,
                Float64(box.px1) - 8.0 * s,
                Float64(box.py1) - mid_dump * ph,
                "Mid Dump: " + fmt_float(end_dump * 100.0, 1) + "%",
                size=10.5 * s,
                bold=True,
                color=WHITE,
                align=TextAlign.RIGHT,
                valign="middle",
                cache=cache,
            )

    _text(
        target,
        0.5 * Float64(w),
        Float64(h) - 16.0 * s,
        "Work Cycle t",
        size=12.0 * s,
        bold=True,
        align=TextAlign.CENTER,
        cache=cache,
    )
    _text(
        target,
        22.0 * s,
        300.0 * s,
        "Share of 128 Physical Rolls (%)",
        size=12.0 * s,
        bold=True,
        align=TextAlign.CENTER,
        rotation=-pi / 2.0,
        cache=cache,
    )


def _plot_fig2_cross_tier_contamination(
    summaries: List[ExperimentSummary], out_dir: String
) raises:
    var idx_l = _find_summary(summaries, "ORI-258")
    var idx_r = _find_summary(summaries, "ORI-CHAMP")
    if idx_l < 0 or idx_r < 0:
        return
    var left_s = summaries[idx_l].copy()
    var right_s = summaries[idx_r].copy()
    var w = 1080
    var h = 560

    var svg = SvgCanvas(w, h)
    var c_svg = FontCache()
    _draw_fig2(svg, left_s, right_s, w, h, 1.0, c_svg)
    write_svg(svg, out_dir + "/fig2_cross_tier_contamination.svg")

    var can = Canvas(w * 2, h * 2, WHITE)
    var c_png = FontCache()
    _draw_fig2(can, left_s, right_s, w, h, 2.0, c_png)
    write_png(can, out_dir + "/fig2_cross_tier_contamination.png")


def _draw_heatmap_panel[
    T: DrawTarget
](
    mut target: T,
    x0: Float64,
    y0: Float64,
    pw: Float64,
    summary: ExperimentSummary,
    title: String,
    max_mis: Float64,
    s: Float64,
    mut cache: FontCache,
) raises:
    _text(target, x0 * s, y0 * s, title, size=12.2 * s, bold=True, cache=cache)
    var cur_y = y0 + 22.0

    var mech_order = List[String]()
    for i in range(len(summary.slot_heatmap_data)):
        var mid = summary.slot_heatmap_data[i].mechanism_id
        var seen = False
        for k in range(len(mech_order)):
            if mech_order[k] == mid:
                seen = True
                break
        if not seen:
            mech_order.append(mid)

    var border_col = Color(166, 173, 184)
    var dark_ink = Color(36, 41, 51)
    var sub_col = Color(71, 82, 97)
    var denom = max(0.5, max_mis)

    for m_idx in range(len(mech_order)):
        var mech_id = mech_order[m_idx]
        var items = List[SlotHeatmapEntry]()
        for i in range(len(summary.slot_heatmap_data)):
            if summary.slot_heatmap_data[i].mechanism_id == mech_id:
                items.append(summary.slot_heatmap_data[i].copy())

        _text(
            target,
            x0 * s,
            cur_y * s,
            _mech_title(mech_id) + " (" + String(len(items)) + " slots)",
            size=10.3 * s,
            bold=True,
            color=sub_col,
            cache=cache,
        )
        cur_y += 10.0

        var cols: Int
        var rows_count: Int
        if mech_id.startswith("pegboard"):
            var max_sub = 0
            var groups = List[Int]()
            for i in range(len(items)):
                if items[i].sub_idx > max_sub:
                    max_sub = items[i].sub_idx
                var g = items[i].group_idx
                var g_seen = False
                for k in range(len(groups)):
                    if groups[k] == g:
                        g_seen = True
                        break
                if not g_seen:
                    groups.append(g)
            cols = max(1, max_sub + 1)
            rows_count = max(1, len(groups))
        elif len(items) > 4:
            cols = 8
            rows_count = (len(items) + cols - 1) // cols
        else:
            cols = max(1, len(items))
            rows_count = 1

        var cell_gap = 4.0
        var cell_w = min(68.0, (pw - Float64(cols - 1) * cell_gap) / Float64(max(1, cols)))
        var cell_h = 26.0 if rows_count >= 4 else 30.0

        for idx in range(len(items)):
            var item = items[idx].copy()
            var r_i = 0
            var c_i = idx
            if mech_id.startswith("pegboard"):
                r_i = item.group_idx
                c_i = item.sub_idx
            elif len(items) > 4:
                r_i = idx // cols
                c_i = idx % cols

            var cx = (x0 + Float64(c_i) * (cell_w + cell_gap)) * s
            var cy = (cur_y + Float64(r_i) * (cell_h + cell_gap)) * s
            var mis = item.mean_final_misplaced
            var occ = item.mean_final_occupants
            var ratio = mis / denom
            var col = _heat_color(ratio)

            target.fill_rect(cx, cy, cell_w * s, cell_h * s, col)
            _stroke_rect(target, cx, cy, cell_w * s, cell_h * s, border_col, 0.8 * s)

            var txt_col = WHITE if ratio > 0.62 else dark_ink
            var short_id = _short_slot_id(item.slot_id)
            _text(
                target,
                cx + 0.5 * cell_w * s,
                cy + 0.5 * cell_h * s - 4.2 * s,
                short_id,
                size=7.4 * s,
                color=txt_col,
                align=TextAlign.CENTER,
                valign="middle",
                cache=cache,
            )
            _text(
                target,
                cx + 0.5 * cell_w * s,
                cy + 0.5 * cell_h * s + 5.2 * s,
                fmt_float(mis, 2) + " / " + fmt_float(occ, 1),
                size=8.0 * s,
                bold=True,
                color=txt_col,
                align=TextAlign.CENTER,
                valign="middle",
                cache=cache,
            )

        cur_y += Float64(rows_count) * (cell_h + cell_gap) + 18.0


def _draw_fig3[
    T: DrawTarget
](
    mut target: T,
    exp_left: ExperimentSummary,
    exp_right: ExperimentSummary,
    w_log: Int,
    h_log: Int,
    s: Float64,
    mut cache: FontCache,
) raises:
    var w = Int(Float64(w_log) * s + 0.5)
    var h = Int(Float64(h_log) * s + 0.5)
    _draw_header(
        target,
        w,
        h,
        34.0,
        "Misplaced Rolls per Compartment at Cycle 1,000",
        "Each cell shows compartment ID and mean misplaced rolls / total rolls at Cycle 1,000 (50 runs)",
        15.4,
        s,
        cache,
    )

    var max_mis = 0.0
    for i in range(len(exp_left.slot_heatmap_data)):
        if exp_left.slot_heatmap_data[i].mean_final_misplaced > max_mis:
            max_mis = exp_left.slot_heatmap_data[i].mean_final_misplaced
    for i in range(len(exp_right.slot_heatmap_data)):
        if exp_right.slot_heatmap_data[i].mean_final_misplaced > max_mis:
            max_mis = exp_right.slot_heatmap_data[i].mean_final_misplaced

    var champ_hdr = _format_champ_label(exp_right)
    _draw_heatmap_panel(
        target,
        34.0,
        82.0,
        505.0,
        exp_left,
        "ORI-258: Top: Peg | Mid: 7D6_4P | Btm: 2B (90% fill, low cue)",
        max_mis,
        s,
        cache,
    )
    _draw_heatmap_panel(
        target,
        575.0,
        82.0,
        505.0,
        exp_right,
        champ_hdr,
        max_mis,
        s,
        cache,
    )

    var cbx = 34.0 * s
    var cby = Float64(h_log - 38) * s
    var cbw = 280.0 * s
    var cbh = 12.0 * s
    var grad = LinearGradient(cbx, cby, cbx + cbw, cby)
    grad.add_stop(0.00, _heat_color(0.00))
    grad.add_stop(0.30, _heat_color(0.30))
    grad.add_stop(0.65, _heat_color(0.65))
    grad.add_stop(1.00, _heat_color(1.00))
    target.fill_rect_gradient(cbx, cby, cbw, cbh, grad)
    _stroke_rect(target, cbx, cby, cbw, cbh, Color(115, 122, 133), 0.8 * s)

    _text(target, cbx, cby + 24.0 * s, "0.0 misplaced", size=9.5 * s, cache=cache)
    _text(
        target,
        cbx + cbw,
        cby + 24.0 * s,
        fmt_float(max_mis, 2) + " misplaced",
        size=9.5 * s,
        align=TextAlign.RIGHT,
        cache=cache,
    )


def _plot_fig3_slot_heatmap(
    summaries: List[ExperimentSummary], out_dir: String
) raises:
    var idx_l = _find_summary(summaries, "ORI-258")
    var idx_r = _find_summary(summaries, "ORI-CHAMP")
    if idx_l < 0 or idx_r < 0:
        return
    var exp_left = summaries[idx_l].copy()
    var exp_right = summaries[idx_r].copy()
    var w = 1120
    var h = 790

    var svg = SvgCanvas(w, h)
    var c_svg = FontCache()
    _draw_fig3(svg, exp_left, exp_right, w, h, 1.0, c_svg)
    write_svg(svg, out_dir + "/fig3_slot_heatmap.svg")

    var can = Canvas(w * 2, h * 2, WHITE)
    var c_png = FontCache()
    _draw_fig3(can, exp_left, exp_right, w, h, 2.0, c_png)
    write_png(can, out_dir + "/fig3_slot_heatmap.png")


def _sort_summaries_asc(mut items: List[ExperimentSummary]):
    """In-place ascending sort of benchmark summaries by final_acc_mean."""
    var n = len(items)
    for i in range(n):
        var min_i = i
        for j in range(i + 1, n):
            if items[j].final_acc_mean < items[min_i].final_acc_mean:
                min_i = j
        if min_i != i:
            var tmp = items[i].copy()
            items[i] = items[min_i].copy()
            items[min_i] = tmp^


def _draw_fig4[
    T: DrawTarget
](
    mut target: T,
    sweeps: List[SweepSeries],
    summaries: List[ExperimentSummary],
    w_log: Int,
    h_log: Int,
    s: Float64,
    mut cache: FontCache,
) raises:
    var w = Int(Float64(w_log) * s + 0.5)
    var h = Int(Float64(h_log) * s + 0.5)
    _draw_header(
        target,
        w,
        h,
        40.0,
        "Effect of Slot Fill Ratio and Shelf Layout on Accuracy and Relative Effort",
        "Left: Cycle 1,000 accuracy across 70%-98% target fill | Right: Benchmark layouts ranked by accuracy (effort relative to ORI-258 = 1.00x)",
        15.4,
        s,
        cache,
    )

    var sweep_keys: List[String] = [
        "TOPO053_Mid3D93D64P_Btm3D93D64P",
        "TOPO007_Mid7D64P_Btm2B",
        "TOPO007_BaselineCOI",
    ]
    var sweep_labels: List[String] = [
        "Top: Peg | Mid: 3D9_3D6_4P | Btm: 3D9_3D6_4P",
        "Top: Peg | Mid: 7D6_4P | Btm: 2B",
        "Top: Peg | Mid: 7D6_4P | Btm: 2B (COI baseline)",
    ]
    var sweep_cols: List[Color] = [
        _palette_color("ORI-CHAMP"),
        _palette_color("ORI-245"),
        _palette_color("ORI-258"),
    ]

    var plots = List[AnyChart](capacity=6)
    var ox0 = Int(32.0 * s + 0.5)
    var oy0 = Int(92.0 * s + 0.5)
    var ox1 = Int(495.0 * s + 0.5)
    var oy1 = Int(500.0 * s + 0.5)

    var is_first = True
    var bx: List[Float64] = [0.75, 0.85]
    var blo: List[Float64] = [0.15, 0.15]
    var bhi: List[Float64] = [1.00, 1.00]

    for i in range(3):
        var s_idx = _find_sweep(sweeps, sweep_keys[i])
        if s_idx < 0:
            continue
        var sw = sweeps[s_idx].copy()
        var th = Theme(
            background=WHITE,
            mark_color=sweep_cols[i],
            axis_color=Color(64, 71, 82),
            gridline_color=Color(220, 224, 230),
            text_color=Color(31, 36, 46),
            font_size=10.0,
            line_width=2.5,
            point_radius=3.8,
            margin_left=44,
            margin_right=10,
            margin_top=6,
            margin_bottom=26,
            show_legend=False,
            annotation_color=Color(209, 46, 51),
            annotation_area_color=Color(84, 168, 104, 32),
            annotation_line_style=LineStyle.DASHED,
            x_tick_format=TickFormat.PERCENT,
            y_tick_format=TickFormat.PERCENT,
        )
        var lp = (
            Plot()
            .mark_line()
            .theme(th)
            .encode(x=sw.rhos, y=sw.accs)
            .size(ox1 - ox0, oy1 - oy0)
        )
        if is_first:
            lp = (
                lp^
                .scale_x_domain(0.68, 0.99)
                .scale_y_domain(0.15, 1.00)
                .annotate_band(bx, blo, bhi)
                .annotate_line(0.80)
            )
            is_first = False
        plots.append(lp.erased())

        var pp = (
            Plot()
            .mark_point()
            .theme(Theme(mark_color=sweep_cols[i], point_radius=3.8, show_legend=False))
            .encode(x=sw.rhos, y=sw.accs)
            .size(ox1 - ox0, oy1 - oy0)
        )
        plots.append(pp.erased())

    var lbox = _render_panel_layers(
        target, plots, ox0, oy0, ox1, oy1, 72.0 * s, cache
    )
    var pw_l = Float64(lbox.px1 - lbox.px0)
    var ph_l = Float64(lbox.py1 - lbox.py0)

    _text(
        target,
        Float64(lbox.px0) + 0.5 * pw_l,
        82.0 * s,
        "Capacity Fill Sweep (70% to 98% Target Fill)",
        size=12.0 * s,
        bold=True,
        align=TextAlign.CENTER,
        cache=cache,
    )

    var buf_x = Float64(lbox.px0) + ((0.80 - 0.68) / 0.31) * pw_l
    _text(
        target,
        buf_x,
        Float64(lbox.py0) + 15.0 * s,
        "Target Buffer",
        size=9.5 * s,
        bold=True,
        color=Color(51, 122, 69),
        align=TextAlign.CENTER,
        cache=cache,
    )
    _text(
        target,
        buf_x,
        Float64(lbox.py0) + 27.0 * s,
        "(15-25% free)",
        size=9.0 * s,
        color=Color(51, 122, 69),
        align=TextAlign.CENTER,
        cache=cache,
    )

    var max_sw_w = 0.0
    for i in range(3):
        var tm_sw = measure_text(sweep_labels[i], 9.3 * s, family="Sans", cache=cache)
        if tm_sw.width > max_sw_w:
            max_sw_w = tm_sw.width
    var lx = Float64(lbox.px0) + 10.0 * s
    var ly = Float64(lbox.py1) - ((0.58 - 0.15) / 0.85) * ph_l
    var lw = max(260.0 * s, max_sw_w + 42.0 * s)
    var lh = 64.0 * s
    target.fill_rect(lx, ly, lw, lh, Color(255, 255, 255, 236))
    _stroke_rect(target, lx, ly, lw, lh, Color(209, 214, 224), 0.9 * s)
    for i in range(3):
        var ry = ly + (14.0 + Float64(i) * 18.0) * s
        target.draw_line_aa(lx + 8.0 * s, ry, lx + 24.0 * s, ry, sweep_cols[i], width=2.5 * s)
        _text(target, lx + 30.0 * s, ry, sweep_labels[i], size=9.3 * s, valign="middle", cache=cache)

    _text(
        target,
        Float64(lbox.px0) + 0.5 * pw_l,
        514.0 * s,
        "Target Capacity Fill Ratio (rho)",
        size=11.5 * s,
        bold=True,
        align=TextAlign.CENTER,
        cache=cache,
    )
    _text(
        target,
        20.0 * s,
        0.5 * Float64(lbox.py0 + lbox.py1),
        "Cycle 1,000 Accuracy (%)",
        size=11.5 * s,
        bold=True,
        align=TextAlign.CENTER,
        rotation=-pi / 2.0,
        cache=cache,
    )

    # Right panel: ranked benchmark horizontal bars
    var champ_idx = _find_summary(summaries, "ORI-CHAMP")
    var champ_lbl = _format_champ_label(summaries[champ_idx]) if champ_idx >= 0 else "ORI-CHAMP"
    var idx_258 = _find_summary(summaries, "ORI-258")
    var base_lat = 1.0
    if idx_258 >= 0:
        base_lat = summaries[idx_258].overall_latency_mean
    else:
        for i in range(len(summaries)):
            if summaries[i].overall_latency_mean > base_lat:
                base_lat = summaries[i].overall_latency_mean

    var ranked = summaries.copy()
    _sort_summaries_asc(ranked)
    var n_bars = len(ranked)
    var max_lbl_w = 0.0
    for i in range(n_bars):
        var lbl = _bench_bar_label(ranked[i], champ_lbl)
        var tm = measure_text(
            lbl,
            9.4 * s,
            family="Sans",
            weight=FontWeight.BOLD if ranked[i].scenario.exp_id == "ORI-CHAMP" else FontWeight.NORMAL,
            cache=cache,
        )
        if tm.width > max_lbl_w:
            max_lbl_w = tm.width

    var pill_reserve = (
        measure_text("100.0% | 1.00x", 9.5 * s, family="Sans", weight=FontWeight.BOLD, cache=cache).width
        + 20.0 * s
    )
    var rx_left = max(512.0 * s, Float64(lbox.px1) + 16.0 * s)
    var bx_start = max(rx_left + 255.0 * s, rx_left + max_lbl_w + 10.0 * s)
    var bw_max = max(140.0 * s, Float64(w_log) * s - pill_reserve - bx_start)
    var ry0 = Float64(lbox.py0)
    var ph_r = Float64(lbox.py1 - lbox.py0)
    var step_y = (ph_r - 12.0 * s) / Float64(max(1, n_bars))
    var bar_h = min(28.0 * s, step_y * 0.65)
    var bar_gap = step_y - bar_h

    _text(
        target,
        rx_left + 0.5 * (Float64(w_log) * s - 20.0 * s - rx_left),
        82.0 * s,
        "Benchmark Layouts: Accuracy & Relative Effort",
        size=11.8 * s,
        bold=True,
        align=TextAlign.CENTER,
        cache=cache,
    )

    for i in range(n_bars):
        var item = ranked[i].copy()
        var eid = item.scenario.exp_id
        var by = ry0 + 6.0 * s + Float64(i) * (bar_h + bar_gap)
        var lbl = _bench_bar_label(item, champ_lbl)
        _text(
            target,
            bx_start - 8.0 * s,
            by + 0.5 * bar_h,
            lbl,
            size=9.4 * s,
            bold=(eid == "ORI-CHAMP"),
            align=TextAlign.RIGHT,
            valign="middle",
            cache=cache,
        )
        var bar_w = item.final_acc_mean * bw_max
        target.fill_rect(bx_start, by, bar_w, bar_h, _palette_color(eid))

    var x80_bar = bx_start + 0.80 * bw_max
    var dash80: List[Float64] = [5.0 * s, 4.0 * s]
    target.draw_line_aa(
        x80_bar,
        ry0 + 2.0 * s,
        x80_bar,
        ry0 + 6.0 * s + Float64(n_bars) * (bar_h + bar_gap) - bar_gap + 4.0 * s,
        Color(209, 46, 51),
        width=1.4 * s,
        dashes=dash80,
    )
    _text(
        target,
        x80_bar,
        Float64(lbox.py1) + 16.0 * s,
        "80%",
        size=9.8 * s,
        bold=True,
        color=Color(199, 41, 46),
        align=TextAlign.CENTER,
        cache=cache,
    )
    _text(
        target,
        bx_start + 0.5 * bw_max,
        514.0 * s,
        "Final Accuracy (%) | Relative Effort (vs. ORI-258 = 1.00x)",
        size=10.8 * s,
        bold=True,
        align=TextAlign.CENTER,
        cache=cache,
    )

    for i in range(n_bars):
        var item = ranked[i].copy()
        var by = ry0 + 6.0 * s + Float64(i) * (bar_h + bar_gap)
        var bar_w = item.final_acc_mean * bw_max
        var val_txt = (
            fmt_float(item.final_acc_mean * 100.0, 1)
            + "% | "
            + fmt_float(item.overall_latency_mean / max(1e-9, base_lat), 2)
            + "x"
        )
        var tm = measure_text(val_txt, 9.5 * s, family="Sans", weight=FontWeight.BOLD, cache=cache)
        var tx = bx_start + bar_w + 6.0 * s
        var pill_w = tm.width + 8.0 * s
        var pill_h = tm.height + 6.0 * s
        _fill_rounded_rect(
            target,
            tx - 3.0 * s,
            by + 0.5 * bar_h - 0.5 * pill_h,
            pill_w,
            pill_h,
            3.0 * s,
            Color(255, 255, 255, 240),
        )
        _text(
            target,
            tx,
            by + 0.5 * bar_h,
            val_txt,
            size=9.5 * s,
            bold=True,
            valign="middle",
            cache=cache,
        )


def _plot_fig4_capacity_and_partitions(
    sweeps: List[SweepSeries],
    summaries: List[ExperimentSummary],
    out_dir: String,
) raises:
    var w = 1180
    var h = 540

    var svg = SvgCanvas(w, h)
    var c_svg = FontCache()
    _draw_fig4(svg, sweeps, summaries, w, h, 1.0, c_svg)
    write_svg(svg, out_dir + "/fig4_capacity_and_partitions.svg")

    var can = Canvas(w * 2, h * 2, WHITE)
    var c_png = FontCache()
    _draw_fig4(can, sweeps, summaries, w, h, 2.0, c_png)
    write_png(can, out_dir + "/fig4_capacity_and_partitions.png")


def _sort_records_by_eff(
    mut recs: List[OrientationRecord], base_lat: Float64
):
    var n = len(recs)
    var denom = max(1e-9, base_lat)
    for i in range(n):
        var min_i = i
        for j in range(i + 1, n):
            if (recs[j].overall_latency_mean / denom) < (recs[min_i].overall_latency_mean / denom):
                min_i = j
        if min_i != i:
            var tmp = recs[i].copy()
            recs[i] = recs[min_i].copy()
            recs[min_i] = tmp^


def _sort_topo_stats(mut stats: List[_TopoStat]):
    var n = len(stats)
    for i in range(n):
        var best_i = i
        for j in range(i + 1, n):
            var better = False
            if stats[j].best_acc > stats[best_i].best_acc + 1e-12:
                better = True
            elif abs(stats[j].best_acc - stats[best_i].best_acc) <= 1e-12 and stats[j].best_lat < stats[best_i].best_lat:
                better = True
            if better:
                best_i = j
        if best_i != i:
            var tmp = stats[i].copy()
            stats[i] = stats[best_i].copy()
            stats[best_i] = tmp^


def _draw_fig5[
    T: DrawTarget
](
    mut target: T,
    ori_records: List[OrientationRecord],
    w_log: Int,
    h_log: Int,
    s: Float64,
    mut cache: FontCache,
) raises:
    var w = Int(Float64(w_log) * s + 0.5)
    var h = Int(Float64(h_log) * s + 0.5)
    _draw_header(
        target,
        w,
        h,
        38.0,
        "Layout Search Across 165 Feasible Wire Shelf Topologies (825 Configurations)",
        "165 physically feasible setups (requiring >=1 9in drawer row or wire bin) x 5 policies",
        15.0,
        s,
        cache,
    )

    if len(ori_records) == 0:
        return

    var base_lat = -1.0
    var max_lat = 0.0
    for i in range(len(ori_records)):
        var r = ori_records[i].copy()
        if r.overall_latency_mean > max_lat:
            max_lat = r.overall_latency_mean
        if (
            (r.row0 == "7D6_4P" or r.row0 == "7D_4P")
            and r.row1 == "2B"
            and r.row2 == "SLIDING_PEG"
            and r.abc_colocation_mode == "cross_tier_height"
            and r.spatial_order_mode == "random_perm"
        ):
            base_lat = r.overall_latency_mean
            break
    if base_lat <= 0.0:
        base_lat = max_lat
    var lat_denom = max(1e-9, base_lat)

    var min_eff = 1e9
    var max_eff = 0.0
    var min_acc = 1.0
    var champ_i = 0
    for i in range(len(ori_records)):
        var eff = ori_records[i].overall_latency_mean / lat_denom
        var acc = ori_records[i].final_acc_mean
        if eff < min_eff:
            min_eff = eff
        if eff > max_eff:
            max_eff = eff
        if acc < min_acc:
            min_acc = acc
        if ori_records[i].composite_score > ori_records[champ_i].composite_score:
            champ_i = i

    var x_min = min(0.30, floor(min_eff * 10.0) / 10.0) - 0.02
    var x_max = max(1.10, ceil(max_eff * 10.0) / 10.0) + 0.02
    var y_min = max(0.05, floor(min_acc * 20.0 - 1.0) / 20.0)
    var y_max = 1.01

    var pareto = List[OrientationRecord]()
    for i in range(len(ori_records)):
        if ori_records[i].is_pareto_optimal:
            pareto.append(ori_records[i].copy())
    _sort_records_by_eff(pareto, lat_denom)
    var champion = ori_records[champ_i].copy()

    # Left panel: dataviz scatter + pareto frontier line
    var ox0 = Int(28.0 * s + 0.5)
    var oy0 = Int(90.0 * s + 0.5)
    var ox1 = Int(570.0 * s + 0.5)
    var oy1 = Int(532.0 * s + 0.5)

    var abc_modes: List[String] = [
        "same_slot",
        "adjacent_partition",
        "same_tier_bay",
        "cross_tier_height",
    ]
    var ovf_flags: List[Bool] = [True, False]
    var plots = List[AnyChart]()
    var is_first = True

    for m_i in range(4):
        var mode = abc_modes[m_i]
        for o_i in range(2):
            var is_hyb = ovf_flags[o_i]
            var xs = List[Float64]()
            var ys = List[Float64]()
            for i in range(len(ori_records)):
                var r = ori_records[i].copy()
                if r.spatial_order_mode == "random_perm":
                    continue
                var hyb = r.overflow_mode == "hybrid_reserve"
                if r.abc_colocation_mode == mode and hyb == is_hyb:
                    xs.append(r.overall_latency_mean / lat_denom)
                    ys.append(r.final_acc_mean)
            if len(xs) == 0:
                continue
            var alpha: UInt8 = 199 if is_hyb else 122
            var th = Theme(
                background=WHITE,
                mark_color=_abc_palette_color(mode, alpha),
                axis_color=Color(64, 71, 82),
                gridline_color=Color(220, 224, 230),
                text_color=Color(31, 36, 46),
                font_size=9.6,
                point_radius=3.6,
                margin_left=44,
                margin_right=8,
                margin_top=6,
                margin_bottom=26,
                show_legend=False,
                annotation_color=Color(209, 46, 51),
                annotation_line_style=LineStyle.DASHED,
                x_tick_format=TickFormat.FIXED(1).with_affixes(suffix="x"),
                y_tick_format=TickFormat.PERCENT,
            )
            var sp = (
                Plot()
                .mark_point()
                .theme(th)
                .encode(x=xs, y=ys)
                .size(ox1 - ox0, oy1 - oy0)
            )
            if is_first:
                sp = sp^.scale_x_domain(x_min, x_max).scale_y_domain(y_min, y_max)
                if y_min < 0.80:
                    sp = sp^.annotate_line(0.80)
                is_first = False
            plots.append(sp.erased())

    if len(pareto) >= 2:
        var pxs = List[Float64](capacity=len(pareto))
        var pys = List[Float64](capacity=len(pareto))
        for i in range(len(pareto)):
            pxs.append(pareto[i].overall_latency_mean / lat_denom)
            pys.append(pareto[i].final_acc_mean)
        var pl = (
            Plot()
            .mark_line(style=LineStyle.DASHED)
            .theme(Theme(mark_color=Color(26, 31, 41), line_width=1.8, show_legend=False))
            .encode(x=pxs, y=pys)
            .size(ox1 - ox0, oy1 - oy0)
        )
        if is_first:
            pl = pl^.scale_x_domain(x_min, x_max).scale_y_domain(y_min, y_max)
        plots.append(pl.erased())

    var lbox = _render_panel_layers(
        target, plots, ox0, oy0, ox1, oy1, 72.0 * s, cache
    )
    var pw_l = Float64(lbox.px1 - lbox.px0)
    var ph_l = Float64(lbox.py1 - lbox.py0)

    _text(
        target,
        Float64(lbox.px0) + 0.5 * pw_l,
        80.0 * s,
        "Pareto Frontier: Cycle 1,000 Accuracy vs. Relative Effort (165 Topologies)",
        size=11.8 * s,
        bold=True,
        align=TextAlign.CENTER,
        cache=cache,
    )

    if y_min < 0.80:
        var y80 = Float64(lbox.py1) - ((0.80 - y_min) / (y_max - y_min)) * ph_l
        _text(
            target,
            Float64(lbox.px1) - 6.0 * s,
            y80 + 12.0 * s,
            "80% Usability Floor",
            size=9.0 * s,
            bold=True,
            color=Color(199, 41, 46),
            align=TextAlign.RIGHT,
            cache=cache,
        )

    # Hollow circles for random_perm baseline cloud (clipped inside plot frame)
    target.push_clip(lbox.px0, lbox.py0, lbox.px1 - lbox.px0, lbox.py1 - lbox.py0)
    var rand_col = _abc_palette_color("random_perm", 140)
    for i in range(len(ori_records)):
        if ori_records[i].spatial_order_mode == "random_perm":
            var eff = ori_records[i].overall_latency_mean / lat_denom
            var acc = ori_records[i].final_acc_mean
            var cx = Float64(lbox.px0) + ((eff - x_min) / (x_max - x_min)) * pw_l
            var cy = Float64(lbox.py1) - ((acc - y_min) / (y_max - y_min)) * ph_l
            target.draw_circle_aa(cx, cy, 3.3 * s, rand_col, width=1.3 * s)
    target.pop_clip()

    # White-filled dark rings on Pareto frontier points
    for i in range(len(pareto)):
        var eff = pareto[i].overall_latency_mean / lat_denom
        var acc = pareto[i].final_acc_mean
        var cx = Float64(lbox.px0) + ((eff - x_min) / (x_max - x_min)) * pw_l
        var cy = Float64(lbox.py1) - ((acc - y_min) / (y_max - y_min)) * ph_l
        target.fill_circle_aa(cx, cy, 5.2 * s, WHITE)
        target.draw_circle_aa(cx, cy, 5.2 * s, Color(26, 31, 41), width=1.6 * s)

    # Red champion bullseye + leader line + callout box
    var ch_eff = champion.overall_latency_mean / lat_denom
    var ch_acc = champion.final_acc_mean
    var ch_x = Float64(lbox.px0) + ((ch_eff - x_min) / (x_max - x_min)) * pw_l
    var ch_y = Float64(lbox.py1) - ((ch_acc - y_min) / (y_max - y_min)) * ph_l
    var ch_red = Color(217, 38, 56)
    target.draw_circle_aa(ch_x, ch_y, 7.5 * s, ch_red, width=2.2 * s)
    target.fill_circle_aa(ch_x, ch_y, 3.8 * s, ch_red)

    var box_w = 232.0 * s
    var box_h = 44.0 * s
    var box_x = Float64(lbox.px1) - box_w - 8.0 * s
    var box_y = Float64(lbox.py0) + 8.0 * s
    target.draw_line_aa(ch_x + 7.5 * s, ch_y - 2.0 * s, box_x, box_y + 0.5 * box_h, ch_red, width=1.2 * s)
    target.fill_rect(box_x, box_y, box_w, box_h, Color(255, 255, 255, 245))
    _stroke_rect(target, box_x, box_y, box_w, box_h, ch_red, 1.2 * s)
    _text(
        target,
        box_x + 8.0 * s,
        box_y + 13.0 * s,
        "Best layout: " + champion.ori_id + " (" + champion.topo_id + ")",
        size=9.0 * s,
        bold=True,
        color=Color(191, 31, 46),
        cache=cache,
    )
    _text(
        target,
        box_x + 8.0 * s,
        box_y + 26.0 * s,
        champion.topo_label,
        size=8.0 * s,
        color=Color(46, 51, 61),
        cache=cache,
    )
    _text(
        target,
        box_x + 8.0 * s,
        box_y + 39.0 * s,
        champion.abc_colocation_mode
        + " | "
        + fmt_float(ch_acc * 100.0, 1)
        + "% Acc | "
        + fmt_float(ch_eff, 2)
        + "x Effort",
        size=8.2 * s,
        bold=True,
        color=Color(36, 115, 77),
        cache=cache,
    )

    # Floating lower-right legend box inside scatter frame
    var leg_w = 232.0 * s
    var leg_h = 102.0 * s
    var lx = Float64(lbox.px1) - leg_w - 8.0 * s
    var ly = Float64(lbox.py1) - leg_h - 8.0 * s
    target.fill_rect(lx, ly, leg_w, leg_h, Color(255, 255, 255, 246))
    _stroke_rect(target, lx, ly, leg_w, leg_h, Color(204, 212, 222), 0.9 * s)
    var leg_modes: List[String] = [
        "same_slot",
        "adjacent_partition",
        "same_tier_bay",
        "cross_tier_height",
        "random_perm",
    ]
    var leg_titles: List[String] = [
        "Same-Slot (58 cm Depth)",
        "Adjacent Partitions (Same Drawer)",
        "Same-Tier Split Bay",
        "Cross-Tier Height Segregated",
        "Random Permutations (Baseline Cloud)",
    ]
    for i in range(5):
        var ry = ly + (14.0 + Float64(i) * 17.5) * s
        var col = _abc_palette_color(leg_modes[i])
        if leg_modes[i] == "random_perm":
            target.draw_circle_aa(lx + 12.0 * s, ry, 3.8 * s, col, width=1.5 * s)
        else:
            target.fill_circle_aa(lx + 12.0 * s, ry, 4.0 * s, col)
        _text(target, lx + 22.0 * s, ry, leg_titles[i], size=9.0 * s, valign="middle", cache=cache)

    _text(
        target,
        Float64(lbox.px0) + 0.5 * pw_l,
        544.0 * s,
        "Relative Retrieval & Return Effort (Baseline ORI-258 = 1.00x)",
        size=11.0 * s,
        bold=True,
        align=TextAlign.CENTER,
        cache=cache,
    )
    _text(
        target,
        20.0 * s,
        0.5 * Float64(lbox.py0 + lbox.py1),
        "Cycle 1,000 Accuracy (% rolls at home)",
        size=11.5 * s,
        bold=True,
        align=TextAlign.CENTER,
        rotation=-pi / 2.0,
        cache=cache,
    )

    # Right panel: Top 12 hardware mean bars + best orientation lollipops
    var uniq_keys = List[String]()
    for i in range(len(ori_records)):
        var key = ori_records[i].topo_id + ": " + ori_records[i].topo_label
        var seen = False
        for k in range(len(uniq_keys)):
            if uniq_keys[k] == key:
                seen = True
                break
        if not seen:
            uniq_keys.append(key)

    var topo_stats = List[_TopoStat](capacity=len(uniq_keys))
    for k in range(len(uniq_keys)):
        var key = uniq_keys[k]
        var sum_acc = 0.0
        var cnt = 0
        var best_score = -1e9
        var best_acc = 0.0
        var best_lat = 0.0
        for i in range(len(ori_records)):
            var r = ori_records[i].copy()
            if (r.topo_id + ": " + r.topo_label) == key:
                sum_acc += r.final_acc_mean
                cnt += 1
                if r.composite_score > best_score:
                    best_score = r.composite_score
                    best_acc = r.final_acc_mean
                    best_lat = r.overall_latency_mean
        topo_stats.append(_TopoStat(key, sum_acc / Float64(max(1, cnt)), best_acc, best_lat))
    _sort_topo_stats(topo_stats)

    var top_cnt = min(12, len(topo_stats))
    var rx0 = max(594.0 * s, Float64(lbox.px1) + 22.0 * s)
    var bx_start = rx0 + 272.0 * s
    var bw_max = Float64(w_log) * s - 108.0 * s - bx_start
    var ry0 = Float64(lbox.py0)
    var row_h = 23.0 * s
    var row_gap = 10.5 * s

    _text(
        target,
        rx0 + 0.5 * (Float64(w_log) * s - 22.0 * s - rx0),
        80.0 * s,
        "Top 12 Modular Shelf Architectures: Mean vs. Best Orientation",
        size=12.0 * s,
        bold=True,
        align=TextAlign.CENTER,
        cache=cache,
    )

    for i in range(top_cnt):
        var st = topo_stats[i].copy()
        var by = ry0 + 6.0 * s + Float64(i) * (row_h + row_gap)
        _text(
            target,
            bx_start - 8.0 * s,
            by + 0.5 * row_h,
            st.label,
            size=8.6 * s,
            bold=(i == 0),
            color=Color(36, 41, 51) if i < 3 else Color(77, 84, 97),
            align=TextAlign.RIGHT,
            valign="middle",
            cache=cache,
        )

        var u_mean = max(0.0, min(1.0, (st.mean_acc - 0.30) / 0.70))
        var u_best = max(0.0, min(1.0, (st.best_acc - 0.30) / 0.70))
        var x_mean = bx_start + u_mean * bw_max
        var x_best = bx_start + u_best * bw_max

        target.fill_rect(bx_start, by + 3.0 * s, bw_max, row_h - 6.0 * s, Color(237, 242, 247))
        target.fill_rect(
            bx_start,
            by + 5.0 * s,
            max(2.0 * s, x_mean - bx_start),
            row_h - 10.0 * s,
            Color(115, 153, 199, 184),
        )
        var stem_col = Color(41, 153, 102) if i < 3 else Color(61, 115, 179)
        target.draw_line_aa(x_mean, by + 0.5 * row_h, x_best, by + 0.5 * row_h, stem_col, width=2.0 * s)
        target.fill_circle_aa(x_best, by + 0.5 * row_h, 4.5 * s, stem_col)

        _text(
            target,
            bx_start + bw_max + 8.0 * s,
            by + 0.5 * row_h,
            "Best " + fmt_float(st.best_acc * 100.0, 1) + "% (" + fmt_float(st.best_lat / lat_denom, 2) + "x)",
            size=8.6 * s,
            bold=(i == 0),
            valign="middle",
            cache=cache,
        )

    var x80_r = bx_start + ((0.80 - 0.30) / 0.70) * bw_max
    var dash_r: List[Float64] = [5.0 * s, 4.0 * s]
    target.draw_line_aa(
        x80_r,
        ry0 + 2.0 * s,
        x80_r,
        ry0 + 6.0 * s + Float64(top_cnt) * (row_h + row_gap) - 4.0 * s,
        Color(209, 46, 51),
        width=1.3 * s,
        dashes=dash_r,
    )
    var r_ticks: List[Int] = [30, 50, 70, 80, 100]
    var tick_y = ry0 + 10.0 * s + Float64(top_cnt) * (row_h + row_gap) + 12.0 * s
    for i in range(5):
        var pct = r_ticks[i]
        var xt = bx_start + ((Float64(pct) / 100.0 - 0.30) / 0.70) * bw_max
        _text(
            target,
            xt,
            tick_y,
            String(pct) + "%",
            size=9.2 * s,
            color=Color(89, 97, 107),
            align=TextAlign.CENTER,
            cache=cache,
        )

    _text(
        target,
        bx_start + 0.5 * bw_max,
        544.0 * s,
        "Cycle 1,000 Accuracy (Bar = Hardware Mean, Dot = Best Positioning)",
        size=10.8 * s,
        bold=True,
        align=TextAlign.CENTER,
        cache=cache,
    )


def _plot_fig5_orientation_search_pareto(
    ori_records: List[OrientationRecord], out_dir: String
) raises:
    var w = 1225
    var h = 580

    var svg = SvgCanvas(w, h)
    var c_svg = FontCache()
    _draw_fig5(svg, ori_records, w, h, 1.0, c_svg)
    write_svg(svg, out_dir + "/fig5_orientation_search_pareto.svg")

    var can = Canvas(w * 2, h * 2, WHITE)
    var c_png = FontCache()
    _draw_fig5(can, ori_records, w, h, 2.0, c_png)
    write_png(can, out_dir + "/fig5_orientation_search_pareto.png")


def generate_all_charts(
    summaries: List[ExperimentSummary],
    sweeps: List[SweepSeries],
    ori_records: List[OrientationRecord],
    figures_dir: String,
) raises:
    """Emit all 5 simulation figures in PNG (@2x / 144 DPI) and SVG using dataviz & canvas."""
    if not Path(figures_dir).exists():
        makedirs(figures_dir, exist_ok=True)
    _plot_fig1_entropy_decay(summaries, figures_dir)
    _plot_fig2_cross_tier_contamination(summaries, figures_dir)
    _plot_fig3_slot_heatmap(summaries, figures_dir)
    _plot_fig4_capacity_and_partitions(sweeps, summaries, figures_dir)
    _plot_fig5_orientation_search_pareto(ori_records, figures_dir)
