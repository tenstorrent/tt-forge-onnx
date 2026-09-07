#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the Add-on-Quasar mapping diagram.

  docs/source/imgs/compiler_arch/quasar-add-mapping.drawio.svg

An SVG that is also a diagrams.net document, so it renders in the docs and opens
editable at app.diagrams.net.

One linear chain, ONNX module -> TTIR -> TTNN -> flatbuffer -> dispatch -> Tensix, with
the Quasar mapping branching off the dispatch box. Drawn this way on purpose: the chain
is shared by every architecture and a single branch forks it, so a straight line with
one fork says the thing a box-per-layer picture cannot.

Both sides of the fork are shown, because the comparison is the point -- the mainline
op is not slower on Quasar, it is refused at kernel construction, which is the reason
the branch has to exist.

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

W, H = 1960, 620
BW, BH = 232, 92          # chain box
GAP = 32
CHAIN_Y = 262
X0 = 40

TITLE = "How one ONNX Add is mapped onto Quasar"
SUB = ("The chain is identical for every architecture. One branch forks it -- and on "
       "Quasar the mainline op is refused, not slow.")

# The main chain, left to right.
CHAIN = [
    dict(id="onnx", label="ONNX module", sub="Add node, float32",
         anchor="", tone=NEUTRAL),
    dict(id="graph", label="forge graph", sub="graphlib::Graph, op = \"add\"",
         anchor="lower_to_mlir.cpp:843", tone=NEUTRAL),
    dict(id="ttir", label="TTIR", sub="ttir.add",
         anchor="arch-neutral", tone=NEUTRAL),
    dict(id="ttnn", label="TTNN", sub="ttnn.add",
         anchor="same for wormhole + quasar", tone=NEUTRAL),
    dict(id="fb", label=".ttnn flatbuffer", sub="EltwiseBinaryOp",
         anchor="Program.operations", tone=NEUTRAL),
    dict(id="disp", label="runtime dispatch", sub="isQuasar()  ?  :",
         anchor="binary.cpp:45", tone=SYSDESC),
    dict(id="tensix", label="Tensix", sub="add_tiles on the FPU",
         anchor="32 cores, 4x8", tone=NEUTRAL),
]

# The fork, hung off the dispatch box.
BRANCH_ABOVE = dict(
    id="quasar_add",
    label="ttnn::operations::experimental::quasar::binary::add",
    sub="THE MAPPING — what we wired up",
    lines=[
        "TTNN_BINARY_OP_TENSOR_TENSOR(add, ADD)",
        "tt-metal .../experimental/quasar/binary/binary.hpp:183",
        "",
        "Same leading args as mainline — (lhs, rhs, output_dtype,",
        "memory_config) — so one forwarded pack binds to both and",
        "the macro needs no per-op shim. All 14 eltwise binaries",
        "landed in one commit because of it.",
    ],
    tone=PATCHED,
)

BRANCH_BELOW = dict(
    id="mainline_add",
    label="::ttnn::add   —   REFUSED on Quasar",
    sub="not slower: it cannot build a kernel",
    lines=[
        "The mainline program factory constructs DataMovementKernel,",
        "whose constructor TT_FATALs:",
        '  "not supported on Quasar. Use QuasarDataMovementKernel"',
        "tt_metal/impl/kernels/kernel.hpp:417",
    ],
    tone=BLOCKED,
)

FOOT = [
    "Score every op three ways, and only the third counts:  (1) does a Quasar op exist   (2) does the runtime dispatch to it   (3) does it run.",
    "For Add, (1) and (2) were green long before it ran — which is exactly why \"add is unmapped\" was the wrong diagnosis. It needed bf16 and cwd=$TT_METAL_HOME.",
]

ANCHOR_ROWS = [("add mapping", [
    ("ttir emit", "forge/csrc/passes/lower_to_mlir.cpp:843"),
    ("the fork", "third_party/tt-mlir/runtime/lib/ttnn/operations/eltwise/binary/binary.cpp"),
    ("isQuasar", "third_party/tt-mlir/runtime/include/tt/runtime/detail/ttnn/operations/utils.h:29"),
    ("the TT_FATAL",
     "third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/kernels/kernel.hpp:417"),
    ("the quasar op",
     "third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/experimental/quasar/binary/binary.hpp:183"),
])]


def chain_x(i):
    return X0 + i * (BW + GAP)


def disp_index():
    return next(i for i, b in enumerate(CHAIN) if b["id"] == "disp")


def branch_geom():
    """The fork boxes sit over/under the dispatch box, right-aligned to the canvas."""
    bw = 700
    x = min(chain_x(disp_index()) - 210, W - 40 - bw)
    return x, bw


def main_boxes():
    out = []
    for i, b in enumerate(CHAIN):
        out.append(dict(**b, x=chain_x(i), y=CHAIN_Y, w=BW, h=BH, lines=[]))
    bx, bw = branch_geom()
    out.append(dict(**BRANCH_ABOVE, x=bx, y=80, w=bw, h=140, anchor=""))
    out.append(dict(**BRANCH_BELOW, x=bx, y=410, w=bw, h=98, anchor=""))
    return out


def emit_mxfile():
    cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']

    def cell(cid, value, style, x, y, w, h):
        cells.append(
            f'<mxCell id={quoteattr(cid)} value={quoteattr(value)} '
            f'style={quoteattr(style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />'
            "</mxCell>")

    cell("title",
         f'<b>{escape(TITLE)}</b><br><font style="font-size:11px">{escape(SUB)}</font>',
         "text;html=1;align=left;verticalAlign=middle;fontSize=17;fontColor=#0f172a;",
         40, 20, W - 80, 50)

    for b in main_boxes():
        fill, stroke, font = BADGE_STYLE[b["tone"]]
        style = (f"rounded=1;arcSize=18;whiteSpace=wrap;html=1;fillColor={fill};"
                 f"strokeColor={stroke};fontColor={font};align=left;verticalAlign=top;"
                 "spacingLeft=10;spacingTop=6;fontSize=12;strokeWidth=1.6;")
        body = [f'<b>{escape(b["label"])}</b>']
        if b.get("sub"):
            body.append(f'<font style="font-size:10px">{escape(b["sub"])}</font>')
        if b.get("anchor"):
            body.append(f'<font style="font-size:9px;color:#5b6472">'
                        f'{escape(b["anchor"])}</font>')
        for l in b.get("lines", []):
            body.append(f'<font style="font-size:9px">{escape(l)}</font>' if l
                        else "&nbsp;")
        cell(b["id"], "<br>".join(body), style, b["x"], b["y"], b["w"], b["h"])

    edge_style = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;"
                  "endFill=1;strokeColor=#3f4854;strokeWidth=2.2;fontSize=10;"
                  "labelBackgroundColor=none;")
    for i in range(len(CHAIN) - 1):
        cells.append(
            f'<mxCell id="c{i}" style={quoteattr(edge_style)} edge="1" parent="1" '
            f'source={quoteattr(CHAIN[i]["id"])} '
            f'target={quoteattr(CHAIN[i + 1]["id"])}>'
            '<mxGeometry relative="1" as="geometry" /></mxCell>')

    cells.append(
        f'<mxCell id="fork_up" value={quoteattr("mapped to the experimental add")} '
        f'style={quoteattr(edge_style + "strokeColor=#b8791f;")} edge="1" parent="1" '
        f'source="disp" target="quasar_add">'
        '<mxGeometry relative="1" as="geometry" /></mxCell>')
    cells.append(
        f'<mxCell id="fork_down" value={quoteattr("not taken on Quasar")} '
        f'style={quoteattr(edge_style + "strokeColor=#c2453f;dashed=1;")} '
        f'edge="1" parent="1" source="disp" target="mainline_add">'
        '<mxGeometry relative="1" as="geometry" /></mxCell>')

    cell("foot", "<br>".join(f'<font style="font-size:10px">{escape(l)}</font>'
                             for l in FOOT),
         "text;html=1;align=left;verticalAlign=middle;fontSize=10;fontColor=#3f4854;",
         40, 540, W - 80, 46)

    body = "".join(cells)
    return ('<mxfile host="app.diagrams.net" agent="gen_add_mapping_diagram.py" '
            'type="device"><diagram id="quasar-add-mapping" name="add on quasar">'
            f'<mxGraphModel dx="{W}" dy="{H}" grid="0" gridSize="10" guides="1" '
            'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            f'pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">'
            f"<root>{body}</root></mxGraphModel></diagram></mxfile>")


def emit_svg(mxfile):
    p = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']
    p.append(svg_text(40, 34, TITLE, 18, "bold"))
    p.append(svg_text(40, 54, SUB, 11, "normal", "#5b6472"))

    for b in main_boxes():
        fill, stroke, font = BADGE_STYLE[b["tone"]]
        p.append(f'<rect x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" '
                 f'height="{b["h"]}" rx="16" fill="{fill}" stroke="{stroke}" '
                 'stroke-width="1.6"/>')
        ty = b["y"] + 24
        p.append(svg_text(b["x"] + 12, ty, b["label"], 12.5, "bold", font))
        if b.get("sub"):
            ty += 16
            p.append(svg_text(b["x"] + 12, ty, b["sub"], 10, "normal", "#3f4854"))
        if b.get("anchor"):
            ty += 14
            p.append(svg_text(b["x"] + 12, ty, b["anchor"], 9, "normal", "#5b6472"))
        for l in b.get("lines", []):
            ty += 12
            if l:
                p.append(svg_text(b["x"] + 12, ty, l, 9, "normal", "#3f4854"))

    # chain arrows
    y = CHAIN_Y + BH / 2
    for i in range(len(CHAIN) - 1):
        x0 = chain_x(i) + BW
        p.append(f'<path d="M {x0} {y} L {chain_x(i + 1) - 5} {y}" '
                 'stroke="#3f4854" stroke-width="2.2" fill="none" '
                 'marker-end="url(#ar1)"/>')

    # the fork
    di = disp_index()
    cx = chain_x(di) + BW / 2
    bx, bw = branch_geom()
    tx = bx + bw / 2
    p.append(f'<path d="M {cx} {CHAIN_Y} L {cx} 240 L {tx} 240 L {tx} 225" '
             'stroke="#b8791f" stroke-width="2.4" fill="none" '
             'marker-end="url(#ar2)"/>')
    p.append(f'<rect x="{cx + 10}" y="{CHAIN_Y - 44}" width="212" height="15" '
             'fill="#ffffff" stroke="none"/>')
    p.append(f'<text x="{cx + 14}" y="{CHAIN_Y - 33}" '
             'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
             'font-size="10" font-weight="bold" fill="#b8791f">'
             'mapped to the experimental add</text>')

    p.append(f'<path d="M {cx} {CHAIN_Y + BH} L {cx} 390 L {tx} 390 L {tx} 405" '
             'stroke="#c2453f" stroke-width="2.2" fill="none" stroke-dasharray="6,4" '
             'marker-end="url(#ar3)"/>')
    p.append(f'<rect x="{cx + 10}" y="{CHAIN_Y + BH + 8}" width="150" height="15" '
             'fill="#ffffff" stroke="none"/>')
    p.append(f'<text x="{cx + 14}" y="{CHAIN_Y + BH + 19}" '
             'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
             'font-size="10" fill="#c2453f">not taken on Quasar</text>')

    for i, l in enumerate(FOOT):
        p.append(svg_text(40, 556 + i * 15, l, 10, "normal", "#3f4854"))

    defs = ("<defs>"
            + "".join(
                f'<marker id="ar{n}" viewBox="0 0 10 10" refX="9" refY="5" '
                'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M 0 1 L 9 5 L 0 9 z" fill="{c}"/></marker>'
                for n, c in ((1, "#3f4854"), (2, "#b8791f"), (3, "#c2453f")))
            + "</defs>")
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

    bs = main_boxes()
    for b in bs:
        if b["x"] < 0 or b["x"] + b["w"] > W or b["y"] + b["h"] > H:
            raise SystemExit(f"{b['id']} outside the {W}x{H} canvas")
    for i in range(len(bs)):
        for j in range(i + 1, len(bs)):
            a, c = bs[i], bs[j]
            if (a["x"] < c["x"] + c["w"] and c["x"] < a["x"] + a["w"]
                    and a["y"] < c["y"] + c["h"] and c["y"] < a["y"] + a["h"]):
                raise SystemExit(f"overlap: {a['id']} / {c['id']}")
    print(f"layout: {len(CHAIN)} chain boxes + 2 fork boxes, no overlaps, in bounds")

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
