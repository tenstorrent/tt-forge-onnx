#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the Add-on-Quasar dispatch mapping diagram.

  docs/source/imgs/compiler_arch/quasar-add-mapping.drawio.svg

An SVG that is also a diagrams.net document: the mxfile XML rides in the `content`
attribute of the <svg> root, so the file renders in the docs and opens editable at
app.diagrams.net.

What it draws is the *mechanism of the mapping*, not the pipeline again: one ONNX Add
becomes one `ttnn.add` in a flatbuffer, and a single runtime branch -- isQuasar() --
decides which of two implementations executes it. The mainline branch is not merely
slower on Quasar, it is refused at kernel construction, which is the reason the branch
has to exist at all. That fork is the whole point of the picture.

Styles are imported from gen_pipeline_diagram.py so the diagrams read as one family.

    python scripts/gen_add_mapping_diagram.py
    python scripts/gen_add_mapping_diagram.py --check
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
    REPO, "docs", "source", "imgs", "compiler_arch", "quasar-add-mapping.drawio.svg"
)

W, H = 1560, 1120
BW, BH = 400, 118

TITLE = "How one ONNX Add is mapped onto Quasar"
SUB = ("Compilation is arch-neutral: the same ttnn.add reaches the runtime for every "
       "target. A single branch then picks the implementation — and on Quasar the "
       "mainline one is refused, not slow.")

# id, x, y, tone, heading, sub, lines, dashed
NODES = [
    dict(id="onnx", x=580, y=118, tone=NEUTRAL, w=400,
         head="ONNX  Add", sub="your model / test",
         lines=['helper.make_node("Add", ...)', "float32, one node"]),
    dict(id="ttir", x=580, y=272, tone=NEUTRAL, w=400,
         head='"add"  →  ttir::AddOp', sub="forge/csrc/passes/lower_to_mlir.cpp:843",
         lines=["handler map at :297", "arch-neutral: shapes and ops only"]),
    dict(id="ttnn", x=580, y=426, tone=NEUTRAL, w=400,
         head="ttnn.add", sub="ttir-to-ttnn-backend-pipeline",
         lines=["ttnn-layout then convert-ttir-to-ttnn",
                "IDENTICAL for wormhole_b0 and quasar"]),
    dict(id="fb", x=580, y=580, tone=SYSDESC, w=400,
         head="EltwiseBinaryOp", sub="in the .ttnn flatbuffer",
         lines=["one entry in Program.operations",
                "carries its MLIR line as debug_info"]),
    dict(id="switch", x=580, y=734, tone=NEUTRAL, w=400,
         head="switch (op->type_type())", sub="runtime/lib/ttnn/program_executor.cpp:318",
         lines=["case OpType::EltwiseBinaryOp:",
                "  operations::eltwise::binary::run(...)"]),
    dict(id="fork", x=580, y=888, tone=SYSDESC, w=400,
         head="utils::isQuasar()  ←  THE MAPPING", sub="operations/eltwise/binary/binary.cpp",
         lines=["RUN_ELTWISE_BINARY(NAME) selects between two",
                "entry points. One branch. Everything above is shared."]),
    # the two branches
    dict(id="mainline", x=40, y=1010, tone=BLOCKED, w=480, h=96,
         head="false  →  ::ttnn::add", sub="REFUSED on Quasar, not merely slow",
         lines=["the program factory builds DataMovementKernel, whose ctor",
                "TT_FATALs: \"not supported on Quasar. Use QuasarDataMovementKernel\"",
                "tt_metal/impl/kernels/kernel.hpp:417"]),
    dict(id="quasar", x=1040, y=1010, tone=PATCHED, w=480, h=96,
         head="true  →  quasar::binary::add", sub="the mapping we added",
         lines=["ttnn::operations::experimental::quasar::binary::add",
                "TTNN_BINARY_OP_TENSOR_TENSOR(add, ADD)",
                "tt-metal .../experimental/quasar/binary/binary.hpp:183"]),
]

# The two side panels: why the substitution is legal, and what actually happens.
PANELS = [
    dict(id="why", x=40, y=272, w=480, h=250, tone=NEUTRAL,
         head="Why a one-line swap is enough",
         lines=[
             "The Quasar entry point takes the same leading",
             "arguments as the mainline op:",
             "",
             "    (lhs, rhs, output_dtype, memory_config)",
             "",
             "so one forwarded parameter pack binds to both and",
             "the macro needs no per-op shim:",
             "",
             "  return utils::isQuasar()",
             "    ? quasar::binary::NAME(args...)",
             "    : ::ttnn::NAME(args...);",
             "",
             "This holds for all 14 eltwise binary ops, which is",
             "why they were dispatched in one commit. Where the",
             "signatures diverge — conv2d, matmul program",
             "configs — the swap does not work and the op needs",
             "real per-op handling instead.",
         ]),
    dict(id="status", x=1040, y=272, w=480, h=250, tone=NEUTRAL,
         head="Does it run?  Score it three ways",
         lines=[
             "1. does a Quasar op exist?        yes",
             "2. does the runtime dispatch to it? yes",
             "3. does it actually run?            see below",
             "",
             "Only (3) counts, and (1)+(2) being green is why",
             "'add is unmapped' is the wrong diagnosis.",
             "",
             "Driven from tt-metal directly, the Quasar add",
             "PASSES: 4/4 resnet50 residual-add shapes on",
             "craq-sim, bf16, TILE, HEIGHT_SHARDED in L1,",
             "8x4 = 32 cores, fused RELU.",
             "",
             "Driven from forge it wedges: f32, DRAM,",
             "INTERLEAVED, 1x1 grid. srcA/srcB reach",
             "valid=1 unpack=1 matrix=0 — operands delivered,",
             "math unit never consumes them.",
         ]),
]

EDGES = [
    ("onnx", "ttir", "lower_to_mlir", [(780, 236), (780, 268)], False),
    ("ttir", "ttnn", "one pipeline, both arches", [(780, 390), (780, 422)], False),
    ("ttnn", "fb", "ttnnToFlatbuffer", [(780, 544), (780, 576)], False),
    ("fb", "switch", "runtime replays the op stream", [(780, 698), (780, 730)], False),
    ("switch", "fork", "", [(780, 852), (780, 884)], False),
    ("fork", "mainline", "false", [(700, 1006), (700, 1030), (520, 1030)], True),
    ("fork", "quasar", "true", [(860, 1006), (860, 1030), (1040, 1030)], False),
]

ANCHOR_ROWS = [("add mapping", [
    ("ttir emit", "forge/csrc/passes/lower_to_mlir.cpp:843"),
    ("dispatch switch", "third_party/tt-mlir/runtime/lib/ttnn/program_executor.cpp:318"),
    ("the fork", "third_party/tt-mlir/runtime/lib/ttnn/operations/eltwise/binary/binary.cpp"),
    ("isQuasar", "third_party/tt-mlir/runtime/include/tt/runtime/detail/ttnn/operations/utils.h:29"),
    ("the TT_FATAL",
     "third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/kernels/kernel.hpp:417"),
    ("the quasar op",
     "third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/experimental/quasar/binary/binary.hpp:183"),
])]


def _all():
    return NODES + PANELS


def emit_mxfile():
    cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']

    def cell(cid, value, style, x, y, w, h):
        cells.append(
            f'<mxCell id={quoteattr(cid)} value={quoteattr(value)} '
            f'style={quoteattr(style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />'
            "</mxCell>"
        )

    cell("title",
         f'<b>{escape(TITLE)}</b><br><font style="font-size:10px">{escape(SUB)}</font>',
         "text;html=1;align=left;verticalAlign=middle;fontSize=16;fontColor=#0f172a;",
         40, 30, W - 80, 64)

    for n in _all():
        fill, stroke, font = BADGE_STYLE[n["tone"]]
        style = (
            f"rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor={fill};"
            f"strokeColor={stroke};fontColor={font};align=left;verticalAlign=top;"
            "spacingLeft=10;spacingTop=4;fontSize=11;"
            + ("dashed=1;" if n.get("dashed") else "")
        )
        body = [f'<b>{escape(n["head"])}</b>']
        if n.get("sub"):
            body.append(
                f'<font style="font-size:9.5px;color:#5b6472">{escape(n["sub"])}</font>')
        body.append("")
        body += [f'<font style="font-size:9px">{escape(l)}</font>' if l else "&nbsp;"
                 for l in n["lines"]]
        cell(n["id"], "<br>".join(body), style,
             n["x"], n["y"], n.get("w", BW), n.get("h", BH))

    for i, (src, dst, label, _r, dashed) in enumerate(EDGES):
        style = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;"
                 "endFill=1;strokeColor=#4b5563;strokeWidth=1.8;fontSize=9;"
                 "labelBackgroundColor=none;" + ("dashed=1;" if dashed else ""))
        cells.append(
            f'<mxCell id="e{i}" value={quoteattr(label)} style={quoteattr(style)} '
            f'edge="1" parent="1" source={quoteattr(src)} target={quoteattr(dst)}>'
            '<mxGeometry relative="1" as="geometry" /></mxCell>')

    body = "".join(cells)
    return ('<mxfile host="app.diagrams.net" agent="gen_add_mapping_diagram.py" '
            'type="device"><diagram id="quasar-add-mapping" name="add on quasar">'
            f'<mxGraphModel dx="{W}" dy="{H}" grid="0" gridSize="10" guides="1" '
            'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            f'pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">'
            f"<root>{body}</root></mxGraphModel></diagram></mxfile>")


def emit_svg(mxfile):
    p = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']
    p.append(svg_text(40, 52, TITLE, 17, "bold"))
    words, lines, cur = SUB.split(), [], ""
    for w in words:
        if len(f"{cur} {w}".strip()) <= 118:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    for i, l in enumerate(lines):
        p.append(svg_text(40, 72 + i * 14, l, 10.5, "normal", "#5b6472"))

    for n in _all():
        fill, stroke, font = BADGE_STYLE[n["tone"]]
        w, h = n.get("w", BW), n.get("h", BH)
        dash = ' stroke-dasharray="6,4"' if n.get("dashed") else ""
        p.append(f'<rect x="{n["x"]}" y="{n["y"]}" width="{w}" height="{h}" rx="5" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"{dash}/>')
        p.append(svg_text(n["x"] + 12, n["y"] + 22, n["head"], 12.5, "bold", font))
        ty = n["y"] + 22
        if n.get("sub"):
            ty += 15
            p.append(svg_text(n["x"] + 12, ty, n["sub"], 9.5, "normal", "#5b6472"))
        ty += 17
        for l in n["lines"]:
            if l:
                p.append(svg_text(n["x"] + 12, ty, l, 9, "normal", "#3f4854"))
            ty += 12

    for src, dst, label, route, dashed in EDGES:
        d = " ".join(("M" if i == 0 else "L") + f" {x} {y}"
                     for i, (x, y) in enumerate(route))
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        p.append(f'<path d="{d}" stroke="#4b5563" stroke-width="1.8" fill="none"'
                 f'{dash} marker-end="url(#ma)"/>')
        if label:
            x0, y0 = route[0]
            p.append(f'<rect x="{x0 + 8}" y="{y0 + 2}" width="{len(label) * 5.3 + 8}" '
                     'height="13" fill="#ffffff" stroke="none"/>')
            p.append(f'<text x="{x0 + 12}" y="{y0 + 12}" '
                     'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
                     f'font-size="9.5" fill="#4b5563">{escape(label)}</text>')

    lx, ly = 40, H - 18
    p.append(svg_text(lx, ly, "legend:", 10, "bold", "#3f4854"))
    lx += 62
    for tone, text in ((NEUTRAL, "arch-neutral"), (SYSDESC, "the decision point"),
                       (PATCHED, "the Quasar mapping"), (BLOCKED, "refused on Quasar")):
        fill, stroke, _ = BADGE_STYLE[tone]
        p.append(f'<rect x="{lx}" y="{ly - 9}" width="12" height="11" rx="2" '
                 f'fill="{fill}" stroke="{stroke}"/>')
        p.append(svg_text(lx + 17, ly, text, 9.5, "normal", "#3f4854"))
        lx += 32 + int(len(text) * 5.9)

    defs = ("<defs>"
            '<marker id="ma" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            'markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 1 L 9 5 L 0 9 z" fill="#4b5563"/></marker></defs>')
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

    ids = {n["id"] for n in _all()}
    dangling = [(s, d) for s, d, _, _, _ in EDGES if s not in ids or d not in ids]
    if dangling:
        raise SystemExit(f"edges reference unknown nodes: {dangling}")
    boxes = [(n["id"], n["x"], n["y"], n.get("w", BW), n.get("h", BH)) for n in _all()]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            _, ax, ay, aw, ah = boxes[i]
            _, bx, by, bw, bh = boxes[j]
            if ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah:
                raise SystemExit(f"overlap: {boxes[i][0]} / {boxes[j][0]}")
    for bid, x, y, w, h in boxes:
        if x + w > W or y + h > H:
            raise SystemExit(f"{bid} falls outside the {W}x{H} canvas")
    print(f"graph: {len(boxes)} boxes, {len(EDGES)} edges, no overlaps, in bounds")

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
