#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the Quasar bringup phase diagram.

  docs/source/imgs/compiler_arch/quasar-bringup-phases.drawio.svg

An SVG that is also a diagrams.net document: the mxfile XML rides in the `content`
attribute of the <svg> root, so the same file renders in the docs and opens editable
at app.diagrams.net.

This draws the *dependency structure*, not a list of eight phases -- the list is in
docs/source/dev_notes/quasar.md and needs no picture. What needs a picture is that the
critical path runs through execution rather than op dispatch: dispatch work cannot be
validated while nothing runs, so the intuitive descriptor -> ops -> execute ordering
wastes effort.

Styles are imported from gen_pipeline_diagram.py so both diagrams read as one family.

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

W, H = 1880, 830
BW, BH = 322, 156

# tone -> (fill, stroke, font) reused from the pipeline diagram's badge palette.
# SYSDESC = the unblocking spine, PATCHED = ours, BLOCKED = external/Metal,
# NEUTRAL = deferred.
OWNER = {
    SYSDESC: "ours · blocking",
    PATCHED: "ours",
    BLOCKED: "Metal owns",
    NEUTRAL: "deferred",
}

NODES = [
    dict(
        id="p00", x=40, y=150, num="00", tone=SYSDESC,
        title="Stop the work evaporating",
        lines=[
            "third_party/tt-mlir gitlink",
            "tt-mlir/third_party/CMakeLists.txt:3",
            "",
            "no source edits — pins only",
        ],
    ),
    dict(
        id="p01", x=400, y=150, num="01", tone=SYSDESC,
        title="Descriptor truth",
        lines=[
            "TTCoreOpsTypes.cpp:85-99",
            "runtime/lib/common/system_desc.cpp",
            "  :181-198 formats  :238 num_cbs",
            "+ 6 Arch::WormholeB0 defaults",
        ],
    ),
    dict(
        id="p02", x=760, y=150, num="02", tone=SYSDESC,
        title="Get anything to execute",
        lines=[
            "craq-sim unpack→math stall",
            "srcA/srcB valid=1 unpack=1 matrix=0",
            "scripts/quasar_sim_env.sh",
            "",
        ],
    ),
    dict(
        id="p03", x=1180, y=150, num="03", tone=PATCHED,
        title="Finish runtime op dispatch",
        lines=[
            "tt-mlir/runtime/lib/ttnn/operations/**",
            "9 of 121 files routed today",
            "forge/test/mlir/test_quasar_sim.py",
            "demand-driven, not all 131 OpTypes",
        ],
    ),
    dict(
        id="p04", x=760, y=390, num="04", tone=BLOCKED, dashed=True,
        title="The genuine Metal asks",
        lines=[
            "quasar/conv2d/** + program_spec.cpp",
            "float compares (tt_llk_quasar)",
            "int32 DFB · max/min reroute",
            "QUASAR_PARITY_GAPS.md §Priorities",
        ],
    ),
    dict(
        id="p05", x=1180, y=600, num="05", tone=PATCHED,
        title="Perf modelling",
        lines=[
            "TTNNCollectPerfMetrics.cpp",
            "  :809-827 BW/AICLK  :1063-1077 skip",
            "profiler kernel_profiler.hpp:115",
            "",
        ],
    ),
    dict(
        id="p06", x=760, y=600, num="06", tone=PATCHED,
        title="Turn the optimizer on",
        lines=[
            "tt-mlir/CMakeLists.txt:40  (OFF)",
            "SingletonDeviceContext.cpp:266, :83-112",
            "",
            "where arch-neutrality ENDS",
        ],
    ),
    dict(
        id="p07", x=400, y=600, num="07", tone=NEUTRAL, dashed=True,
        title="Scale-out & the D2M path",
        lines=[
            "tt_metal impl/dispatch/topology.cpp:494",
            "D2M/Utils/DMAUtils.cpp:51-60",
            "",
            "hard-rejected upstream today",
        ],
    ),
]

BY_ID = {n["id"]: n for n in NODES}

# (src, dst, label, waypoint route for the SVG, dashed)
EDGES = [
    ("p00", "p01", "", [(362, 228), (400, 228)], False),
    ("p01", "p02", "", [(722, 228), (760, 228)], False),
    (
        "p02", "p03", "gates validation of",
        [(1082, 228), (1180, 228)], False,
    ),
    (
        "p04", "p03", "unblocks specific ops",
        [(1082, 468), (1130, 468), (1130, 280), (1180, 280)], True,
    ),
    (
        "p03", "p05", "only after correctness holds",
        [(1341, 306), (1341, 600)], False,
    ),
    ("p05", "p06", "", [(1180, 678), (1082, 678)], False),
    ("p06", "p07", "", [(760, 678), (722, 678)], True),
]

# Anchors quoted in the drawing, checked before anything is written. check_anchors()
# reads the anchor at index 1 of each row, so the label comes first.
ANCHOR_ROWS = [
    ("phase diagram", [
        ("pin", "third_party/tt-mlir/third_party/CMakeLists.txt:3"),
        ("mock descriptor formats",
         "third_party/tt-mlir/lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp:99"),
        ("live descriptor num_cbs",
         "third_party/tt-mlir/runtime/lib/common/system_desc.cpp:238"),
        ("perf metrics skip",
         "third_party/tt-mlir/lib/Dialect/TTNN/Transforms/TTNNCollectPerfMetrics.cpp:1077"),
        ("opmodel arch default",
         "third_party/tt-mlir/lib/OpModel/TTNN/SingletonDeviceContext.cpp:266"),
        ("d2m quasar reject",
         "third_party/tt-mlir/lib/Dialect/D2M/Utils/DMAUtils.cpp:60"),
        ("opmodel switch", "third_party/tt-mlir/CMakeLists.txt:40"),
        ("sim test", "forge/test/mlir/test_quasar_sim.py"),
        ("sim env", "scripts/quasar_sim_env.sh"),
    ]),
]

NO_CHANGE = [
    "every pass in ttir-to-ttnn-backend-pipeline",
    "  all 57 — measured, 3-line arch diff",
    "the forge frontend and TVM path",
    "the flatbuffer schema (carries system_desc)",
    "forge/forge/tools/*.py  (netlist-era dead code)",
    "device_config.hpp is_wormhole_b0()  (inert)",
]

TITLE = "Adding Quasar device support to tt-forge-onnx — the eight phases and what gates what"
SUB = (
    "The critical path runs through 02, not 03: op dispatch cannot be validated while "
    "nothing executes. Solid arrows are hard dependencies. "
    "generated by scripts/gen_quasar_phase_diagram.py"
)


def node_html(n):
    parts = [
        f'<b>{n["num"]}  {escape(n["title"])}</b>',
        f'<font style="font-size:9.5px;color:#5b6472">{escape(OWNER[n["tone"]])}</font>',
        "",
    ]
    for l in n["lines"]:
        parts.append(
            f'<font style="font-size:9px">{escape(l)}</font>' if l else "&nbsp;"
        )
    return "<br>".join(parts)


def emit_mxfile():
    cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']

    def cell(cid, value, style, x, y, w, h):
        cells.append(
            f'<mxCell id={quoteattr(cid)} value={quoteattr(value)} '
            f'style={quoteattr(style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />'
            "</mxCell>"
        )

    cell(
        "title",
        f'<b>{escape(TITLE)}</b><br>'
        f'<font style="font-size:10px">{escape(SUB)}</font>',
        "text;html=1;align=left;verticalAlign=middle;fontSize=15;fontColor=#0f172a;",
        40, 40, W - 80, 60,
    )

    for n in NODES:
        fill, stroke, font = BADGE_STYLE[n["tone"]]
        style = (
            f"rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor={fill};"
            f"strokeColor={stroke};fontColor={font};align=left;verticalAlign=top;"
            "spacingLeft=10;spacingTop=4;fontSize=11;"
            + ("dashed=1;" if n.get("dashed") else "")
        )
        cell(n["id"], node_html(n), style, n["x"], n["y"], BW, BH)

    for i, (src, dst, label, _route, dashed) in enumerate(EDGES):
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;"
            "strokeColor=#4b5563;strokeWidth=1.8;fontSize=9;labelBackgroundColor=none;"
            + ("dashed=1;" if dashed else "")
        )
        cells.append(
            f'<mxCell id="e{i}" value={quoteattr(label)} style={quoteattr(style)} '
            f'edge="1" parent="1" source={quoteattr(src)} target={quoteattr(dst)}>'
            '<mxGeometry relative="1" as="geometry" /></mxCell>'
        )

    cell(
        "nochange",
        "<b>Needs no change at all</b><br>"
        + "<br>".join(
            f'<font style="font-size:9px">{escape(l)}</font>' for l in NO_CHANGE
        ),
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#f4f5f9;strokeColor=#94a3b8;"
        "fontColor=#0f172a;align=left;verticalAlign=top;spacingLeft=10;spacingTop=4;"
        "fontSize=11;dashed=1;",
        40, 390, BW, 156,
    )

    body = "".join(cells)
    return (
        '<mxfile host="app.diagrams.net" agent="gen_quasar_phase_diagram.py" '
        'type="device"><diagram id="quasar-bringup-phases" name="quasar phases">'
        f'<mxGraphModel dx="{W}" dy="{H}" grid="0" gridSize="10" guides="1" '
        'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        f'pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">'
        f"<root>{body}</root></mxGraphModel></diagram></mxfile>"
    )


def emit_svg(mxfile):
    p = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']
    p.append(svg_text(40, 62, TITLE, 16, "bold"))
    for i, chunk in enumerate(
        [SUB[:104], SUB[104:]] if len(SUB) > 104 else [SUB]
    ):
        p.append(svg_text(40, 82 + i * 14, chunk.strip(), 10.5, "normal", "#5b6472"))

    def box(x, y, w, h, tone, dashed, heading, sub, lines):
        fill, stroke, font = BADGE_STYLE[tone]
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        out = [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.3"{dash}/>'
        ]
        out.append(svg_text(x + 11, y + 22, heading, 12, "bold", font))
        if sub:
            out.append(svg_text(x + 11, y + 37, sub, 9.5, "normal", "#5b6472"))
        ty = y + 58
        for l in lines:
            if l:
                out.append(svg_text(x + 11, ty, l, 9, "normal", "#3f4854"))
            ty += 13
        return out

    for n in NODES:
        p += box(
            n["x"], n["y"], BW, BH, n["tone"], n.get("dashed"),
            f'{n["num"]}  {n["title"]}', OWNER[n["tone"]], n["lines"],
        )

    p += box(40, 390, BW, 156, NEUTRAL, True, "Needs no change at all", "", NO_CHANGE)

    for src, dst, label, route, dashed in EDGES:
        d = " ".join(
            ("M" if i == 0 else "L") + f" {x} {y}" for i, (x, y) in enumerate(route)
        )
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        p.append(
            f'<path d="{d}" stroke="#4b5563" stroke-width="1.8" fill="none"'
            f'{dash} marker-end="url(#pa)"/>'
        )
        if label:
            mx = sum(x for x, _ in route) / len(route)
            my = min(y for _, y in route)
            p.append(
                f'<rect x="{mx - len(label) * 2.6 - 5}" y="{my - 18}" '
                f'width="{len(label) * 5.2 + 10}" height="14" fill="#ffffff" '
                'stroke="none"/>'
            )
            p.append(
                f'<text x="{mx}" y="{my - 7}" text-anchor="middle" '
                'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
                f'font-size="9.5" fill="#4b5563">{escape(label)}</text>'
            )

    lx, ly = 40, H - 22
    p.append(svg_text(lx, ly, "legend:", 10, "bold", "#3f4854"))
    lx += 62
    for tone, text in (
        (SYSDESC, "the unblocking spine"),
        (PATCHED, "ours"),
        (BLOCKED, "Metal owns"),
        (NEUTRAL, "deferred / no change"),
    ):
        fill, stroke, _ = BADGE_STYLE[tone]
        p.append(
            f'<rect x="{lx}" y="{ly - 9}" width="12" height="11" rx="2" '
            f'fill="{fill}" stroke="{stroke}"/>'
        )
        p.append(svg_text(lx + 17, ly, text, 9.5, "normal", "#3f4854"))
        lx += 32 + int(len(text) * 5.9)

    defs = (
        "<defs>"
        '<marker id="pa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 1 L 9 5 L 0 9 z" fill="#4b5563"/></marker>'
        "</defs>"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{W}" '
        f'height="{H}" viewBox="0 0 {W} {H}" content={quoteattr(mxfile)}>'
        f"{defs}{''.join(p)}</svg>\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = ap.parse_args()

    bad, seen = check_anchors(ANCHOR_ROWS)
    if bad:
        for b in bad:
            print(f"  ANCHOR FAIL {b}", file=sys.stderr)
        raise SystemExit(f"{len(bad)} of {seen} anchors do not resolve")
    print(f"anchors: {seen} checked, all resolve")

    ids = {n["id"] for n in NODES}
    dangling = [(s, d) for s, d, _, _, _ in EDGES if s not in ids or d not in ids]
    if dangling:
        raise SystemExit(f"edges reference unknown nodes: {dangling}")
    for n in NODES:
        if n["x"] + BW > W or n["y"] + BH > H:
            raise SystemExit(f"node {n['id']} falls outside the {W}x{H} canvas")
    print(f"graph: {len(NODES)} phases, {len(EDGES)} dependencies, no dangling edges")

    mxfile = emit_mxfile()
    svg = emit_svg(mxfile)
    if args.check:
        print("--check: nothing written")
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(svg)
    print(f"wrote {os.path.relpath(OUT, REPO)} ({len(svg) // 1024} KiB, {W}x{H})")


if __name__ == "__main__":
    main()
