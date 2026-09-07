#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the Quasar support flow: who does what, in which repo.

  docs/source/imgs/compiler_arch/quasar-bringup-phases.drawio.svg

An SVG that is also a diagrams.net document, so it renders in the docs and opens
editable at app.diagrams.net.

A grid: rows are repos, columns are phases in order. Drawn this way because for work
split across three repos "who owns this?" is the question actually asked in the room,
and a per-repo grid answers it while keeping the ordering as the column order.

The empty cells carry as much as the full ones -- forge has nothing to do in phases 3
and 4 -- so they are drawn as a dash rather than a box, to read at a glance.

One accent: the single cross-lane arrow, tt-mlir's dispatch into ttnn's Quasar op
library. That seam is the only place the arch-neutral stack meets Quasar-specific code.

    python scripts/gen_quasar_phase_diagram.py
    python scripts/gen_quasar_phase_diagram.py --check
"""

import argparse
import os
import sys
from xml.sax.saxutils import escape, quoteattr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_pipeline_diagram import (  # noqa: E402
    BADGE_STYLE,
    BLOCKED,
    NEUTRAL,
    PATCHED,
    REPO,
    SYSDESC,
    check_anchors,
    svg_text,
)

OUT = os.path.join(
    REPO, "docs", "source", "imgs", "compiler_arch",
    "quasar-bringup-phases.drawio.svg",
)

# --- geometry -------------------------------------------------------------
LANE_W = 232          # the row-label column
CELL_W = 372
CELL_H = 128
GAP_X, GAP_Y = 22, 18
X0, Y0 = 40, 196      # top-left of the first cell
HDR_H = 34
W, H = 1900, 760

MAX_CELL_LINES = 3    # the cleanliness rule, asserted below

TITLE = "Quasar support: what each repo has to do"
SUB = ("Rows are repos, columns are phases in order. Status as of 2026-09-07. "
       "The empty cells matter as much as the full ones.")

DONE, WORK, EXT, NONE = "done", "work", "external", "none"
TONE = {DONE: SYSDESC, WORK: PATCHED, EXT: BLOCKED, NONE: NEUTRAL}
TAG = {DONE: "DONE", WORK: "THE WORK", EXT: "METAL OWNS", NONE: ""}

PHASES = [
    "1 · Descriptor",
    "2 · Execute",
    "3 · Dispatch",
    "4 · Ops & kernels",
]

LANES = [
    dict(
        id="forge", name="tt-forge-onnx",
        note="12 source files · plumb an arch through config, no per-op work",
        cells=[
            dict(status=DONE, head="target_arch / system_desc_path", lines=[
                "conditional ttcore.system_desc stamp",
                "lower_to_mlir.cpp:194  mlir_config.cpp:167",
            ]),
            dict(status=DONE, head="bf16 override + cwd fixture", lines=[
                "default_df_override = Float16_b",
                "test_quasar_sim.py  _tt_metal_cwd",
            ]),
            dict(status=NONE, head="nothing — arch-neutral", lines=[
                "the frontend, TVM path and graph",
                "passes never ask which chip",
            ]),
            dict(status=NONE, head="nothing", lines=[]),
        ],
    ),
    dict(
        id="ttmlir", name="tt-mlir",
        note="16 files · 13 of them runtime op dispatch — this is the bulk",
        cells=[
            dict(status=DONE, head="mock + live descriptor", lines=[
                "both now query the tt-metal API",
                "TTCoreOpsTypes.cpp:85  system_desc.cpp:181",
            ]),
            dict(status=NONE, head="nothing", lines=[]),
            dict(status=WORK, head="isQuasar() branch, per op", lines=[
                "10 of 28 families reached · 18 available",
                "runtime/lib/ttnn/operations/**",
            ]),
            dict(status=NONE, head="nothing", lines=[
                "no compiler pass needs a change:",
                "the two arch pipelines are identical",
            ]),
        ],
    ),
    dict(
        id="ttnn", name="ttnn / tt-metal",
        note="2 local files · the Quasar op library, HAL and SoC YAMLs are upstream",
        cells=[
            dict(status=DONE, head="is_data_format_supported", lines=[
                "already existed — we just call it",
                "tt_backend_api_types.hpp:72",
            ]),
            dict(status=EXT, head="kernel-include convention", lines=[
                "paths resolve against the cwd, not",
                "TT_METAL_HOME · kernel.cpp:100-112",
            ]),
            dict(status=DONE, head="experimental/quasar/** — upstream", lines=[
                "28 op families ship with tt-metal",
                "binary/binary.hpp:183 is add's target",
            ]),
            dict(status=EXT, head="conv2d · float compares", lines=[
                "+ int32 DFB bug, max/min reroute",
                "QUASAR_PARITY_GAPS.md §Priorities",
            ]),
        ],
    ),
]

MAP_LABEL = "ttnn.add  →  ttnn::operations::experimental::quasar::binary::add     (isQuasar)"
MAP_SUB = "binary.cpp:45   →   binary.hpp:183      the only seam between the arch-neutral stack and Quasar code"

FOOT = [
    "The gate: nothing in phase 3 can be validated until phase 2 executes — you can compile a Quasar binary for any op, but not tell a correct one from a wrong one.",
    "Later, not shown: perf modelling has no calibrated Quasar numbers, and the optimizer is compiled out (TTMLIR_ENABLE_OPMODEL=OFF) so Quasar's 4 MiB L1 is unexploited.",
]

ANCHOR_ROWS = [("repo grid", [
    ("stamp", "forge/csrc/passes/lower_to_mlir.cpp:194"),
    ("options", "forge/csrc/passes/mlir_config.cpp:167"),
    ("sim test", "forge/test/mlir/test_quasar_sim.py"),
    ("mock desc", "third_party/tt-mlir/lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp:99"),
    ("live desc", "third_party/tt-mlir/runtime/lib/common/system_desc.cpp:181"),
    ("the fork", "third_party/tt-mlir/runtime/lib/ttnn/operations/eltwise/binary/binary.cpp"),
    ("kernel resolve",
     "third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/kernels/kernel.cpp:112"),
    ("quasar add",
     "third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/experimental/quasar/binary/binary.hpp:183"),
])]


def map_banner_geom():
    """The mapping banner spans the dispatch and ops columns, clamped to the canvas."""
    x = cell_x(2)
    w = min(CELL_W * 2 + GAP_X, W - 40 - x)
    return x, w


def cell_x(c):
    return X0 + LANE_W + GAP_X + c * (CELL_W + GAP_X)


def lane_y(r):
    return Y0 + r * (CELL_H + GAP_Y)


def emit_mxfile():
    cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']

    def box(cid, value, style, x, y, w, h):
        cells.append(
            f'<mxCell id={quoteattr(cid)} value={quoteattr(value)} '
            f'style={quoteattr(style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />'
            "</mxCell>")

    txt = "text;html=1;align=left;verticalAlign=middle;"
    box("title",
        f'<b>{escape(TITLE)}</b><br><font style="font-size:11px">{escape(SUB)}</font>',
        txt + "fontSize=17;fontColor=#0f172a;", 40, 34, W - 80, 56)

    # the mapping banner
    fill, stroke, font = BADGE_STYLE[PATCHED]
    bx, bw = map_banner_geom()
    box("mapbanner",
        f'<b>{escape(MAP_LABEL)}</b><br>'
        f'<font style="font-size:9px;color:#5b6472">{escape(MAP_SUB)}</font>',
        f"rounded=1;arcSize=14;whiteSpace=wrap;html=1;fillColor={fill};"
        f"strokeColor={stroke};fontColor={font};align=center;verticalAlign=middle;"
        "fontSize=12;strokeWidth=2;", bx, 104, bw, 52)

    # column headers
    for c, name in enumerate(PHASES):
        box(f"ph{c}", f'<b>{escape(name)}</b>',
            txt + "align=left;fontSize=13;fontColor=#1f2937;",
            cell_x(c), Y0 - HDR_H, CELL_W, HDR_H - 6)

    for r, lane in enumerate(LANES):
        box(f"lane{r}",
            f'<b>{escape(lane["name"])}</b><br>'
            f'<font style="font-size:9px;color:#5b6472">{escape(lane["note"])}</font>',
            txt + "align=left;fontSize=14;fontColor=#0f172a;",
            X0, lane_y(r), LANE_W, CELL_H)

        for c, cell in enumerate(lane["cells"]):
            cid = f"{lane['id']}{c}"
            if cell["status"] == NONE and not cell["lines"]:
                box(cid, '<font style="font-size:13px;color:#8891a3">—</font>',
                    txt + "align=center;fontSize=13;", cell_x(c), lane_y(r),
                    CELL_W, CELL_H)
                continue
            f2, s2, fo = BADGE_STYLE[TONE[cell["status"]]]
            dash = "dashed=1;" if cell["status"] == NONE else ""
            body = [f'<b>{escape(cell["head"])}</b>']
            if TAG[cell["status"]]:
                body.append(f'<font style="font-size:9px"><b>'
                            f'{escape(TAG[cell["status"]])}</b></font>')
            body.append("")
            body += [f'<font style="font-size:9px">{escape(l)}</font>'
                     for l in cell["lines"]]
            box(cid, "<br>".join(body),
                f"rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor={f2};"
                f"strokeColor={s2};fontColor={fo};align=left;verticalAlign=top;"
                f"spacingLeft=10;spacingTop=6;fontSize=11;strokeWidth=1.5;{dash}",
                cell_x(c), lane_y(r), CELL_W, CELL_H)

    # the one cross-lane arrow: tt-mlir dispatch -> ttnn dispatch
    cells.append(
        f'<mxCell id="mapping" value={quoteattr("maps to")} '
        f'style={quoteattr("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
                           "endArrow=block;endFill=1;strokeColor=#b8791f;"
                           "strokeWidth=3.4;fontSize=10;labelBackgroundColor=none;")} '
        f'edge="1" parent="1" source="ttmlir2" target="ttnn2">'
        '<mxGeometry relative="1" as="geometry" /></mxCell>')

    # the gate, between columns 2 and 3
    box("gate", '<b><font style="font-size:10px;color:#b8791f">gates</font></b>',
        txt + "align=center;fontSize=10;",
        cell_x(2) - GAP_X - 6, Y0 - HDR_H, GAP_X + 12, HDR_H - 6)

    box("foot", "<br>".join(f'<font style="font-size:10px">{escape(l)}</font>'
                            for l in FOOT),
        txt + "fontSize=10;fontColor=#3f4854;", 40, H - 92, W - 80, 46)

    body = "".join(cells)
    return ('<mxfile host="app.diagrams.net" agent="gen_quasar_phase_diagram.py" '
            'type="device"><diagram id="quasar-repo-grid" name="who does what">'
            f'<mxGraphModel dx="{W}" dy="{H}" grid="0" gridSize="10" guides="1" '
            'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            f'pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">'
            f"<root>{body}</root></mxGraphModel></diagram></mxfile>")


def emit_svg(mxfile):
    p = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']
    p.append(svg_text(40, 52, TITLE, 18, "bold"))
    p.append(svg_text(40, 72, SUB, 11, "normal", "#5b6472"))

    # mapping banner
    fill, stroke, font = BADGE_STYLE[PATCHED]
    bx, bw = map_banner_geom()
    p.append(f'<rect x="{bx}" y="104" width="{bw}" height="52" rx="12" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    p.append(f'<text x="{bx + bw / 2}" y="126" text-anchor="middle" '
             'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
             f'font-size="12" font-weight="bold" fill="{font}">'
             f'{escape(MAP_LABEL)}</text>')
    p.append(f'<text x="{bx + bw / 2}" y="144" text-anchor="middle" '
             'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
             f'font-size="9" fill="#5b6472">{escape(MAP_SUB)}</text>')

    for c, name in enumerate(PHASES):
        p.append(svg_text(cell_x(c), Y0 - 12, name, 13, "bold", "#1f2937"))
    gx = cell_x(2) - GAP_X / 2
    p.append(f'<text x="{gx}" y="{Y0 - 12}" text-anchor="middle" '
             'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
             'font-size="9.5" font-weight="bold" fill="#b8791f">gates</text>')

    for r, lane in enumerate(LANES):
        y = lane_y(r)
        p.append(svg_text(X0, y + 26, lane["name"], 14, "bold", "#0f172a"))
        p.append(svg_text(X0, y + 44, lane["note"][:44], 9, "normal", "#5b6472"))
        if len(lane["note"]) > 44:
            p.append(svg_text(X0, y + 56, lane["note"][44:], 9, "normal", "#5b6472"))

        for c, cell in enumerate(lane["cells"]):
            x = cell_x(c)
            if cell["status"] == NONE and not cell["lines"]:
                p.append(f'<text x="{x + CELL_W / 2}" y="{y + CELL_H / 2 + 5}" '
                         'text-anchor="middle" font-family="sans-serif" '
                         'font-size="15" fill="#8891a3">&#8212;</text>')
                continue
            f2, s2, fo = BADGE_STYLE[TONE[cell["status"]]]
            dash = ' stroke-dasharray="6,4"' if cell["status"] == NONE else ""
            p.append(f'<rect x="{x}" y="{y}" width="{CELL_W}" height="{CELL_H}" '
                     f'rx="9" fill="{f2}" stroke="{s2}" stroke-width="1.5"{dash}/>')
            ty = y + 24
            p.append(svg_text(x + 11, ty, cell["head"], 11.5, "bold", fo))
            if TAG[cell["status"]]:
                ty += 15
                p.append(svg_text(x + 11, ty, TAG[cell["status"]], 9, "bold", s2))
            ty += 18
            for l in cell["lines"]:
                p.append(svg_text(x + 11, ty, l, 9, "normal", "#3f4854"))
                ty += 12

    # the one cross-lane arrow
    x_mid = cell_x(2) + CELL_W / 2
    y_from = lane_y(1) + CELL_H
    y_to = lane_y(2)
    p.append(f'<path d="M {x_mid} {y_from} L {x_mid} {y_to - 5}" stroke="#b8791f" '
             'stroke-width="3.4" fill="none" marker-end="url(#gm)"/>')
    p.append(f'<rect x="{x_mid + 8}" y="{y_from + 1}" width="60" height="14" '
             'fill="#ffffff" stroke="none"/>')
    p.append(f'<text x="{x_mid + 12}" y="{y_from + 12}" '
             'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
             'font-size="10" font-weight="bold" fill="#b8791f">maps to</text>')

    for i, l in enumerate(FOOT):
        p.append(svg_text(40, H - 78 + i * 15, l, 10, "normal", "#3f4854"))

    defs = ("<defs>"
            '<marker id="gm" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
            'markerHeight="6" orient="auto-start-reverse">'
            '<path d="M 0 1 L 9 5 L 0 9 z" fill="#b8791f"/></marker></defs>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{W}" '
            f'height="{H}" viewBox="0 0 {W} {H}" content={quoteattr(mxfile)}>'
            f"{defs}{''.join(p)}</svg>\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    bad, seen = check_anchors(ANCHOR_ROWS)
    if bad:
        for b in bad:
            print(f"  ANCHOR FAIL {b}", file=sys.stderr)
        raise SystemExit(f"{len(bad)} of {seen} anchors do not resolve")
    print(f"anchors: {seen} checked, all resolve")

    # the cleanliness rule, enforced rather than hoped for
    for lane in LANES:
        if len(lane["cells"]) != len(PHASES):
            raise SystemExit(f"{lane['id']}: {len(lane['cells'])} cells, "
                             f"{len(PHASES)} phases")
        for c, cell in enumerate(lane["cells"]):
            if len(cell["lines"]) > MAX_CELL_LINES:
                raise SystemExit(
                    f"{lane['id']} phase {c + 1}: {len(cell['lines'])} body lines, "
                    f"max is {MAX_CELL_LINES} -- put it in quasar.md instead")
            for l in cell["lines"] + [cell["head"]]:
                if len(l) * 5.3 + 22 > CELL_W:
                    raise SystemExit(
                        f"{lane['id']} phase {c + 1}: line needs "
                        f"{len(l) * 5.3 + 22:.0f}px, cell is {CELL_W}px: {l!r}")

    right = cell_x(len(PHASES) - 1) + CELL_W
    bottom = lane_y(len(LANES) - 1) + CELL_H
    if right > W - 20:
        raise SystemExit(f"grid ends at x={right}, canvas is {W}")
    if bottom > H - 100:
        raise SystemExit(f"grid ends at y={bottom}, footer starts at {H - 92}")
    print(f"layout: {len(LANES)}x{len(PHASES)} grid, ends at ({right},{bottom}), "
          f"max {MAX_CELL_LINES} lines/cell respected")

    mxfile = emit_mxfile()
    svg = emit_svg(mxfile)
    if args.check:
        print("--check: nothing written")
        return
    with open(OUT, "w") as fh:
        fh.write(svg)
    print(f"wrote {os.path.relpath(OUT, REPO)} ({len(svg) // 1024} KiB, {W}x{H})")


if __name__ == "__main__":
    main()
