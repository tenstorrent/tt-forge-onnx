#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the Quasar change map: the compile path, with our changes branching off it.

  docs/source/imgs/compiler_arch/quasar-change-map.drawio.svg

An SVG that is also a diagrams.net document, so it renders in the docs and opens
editable at app.diagrams.net.

A straight spine -- ONNX module, lowering to TTIR, the descriptor stamp, TTNN, the
flatbuffer, runtime dispatch, the kernel -- with each change we made hung off the exact
step it touches. Drawn this way because the spine is entirely stock: it is the same path
a Wormhole compile takes, and every Quasar change is a branch off it rather than a
replacement for part of it. Four branches, four changes.

    python scripts/gen_quasar_change_map.py
    python scripts/gen_quasar_change_map.py --check
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
    REPO, "docs", "source", "imgs", "compiler_arch", "quasar-change-map.drawio.svg"
)

W, H = 2480, 1000
SW, SH = 258, 86            # spine box
SGAP = 46
SX0, SY = 40, 452           # the spine sits mid-canvas, branches above and below

TITLE = "What we changed to run an op on Quasar"
SUB = ("The spine is stock — the identical path a Wormhole compile takes. Every Quasar "
       "change is a branch off one step of it, not a replacement for it.")

SPINE = [
    dict(id="s0", label="ONNX module", sub="Add node, float32"),
    dict(id="s1", label="forge graph", sub='graphlib::Graph, op = "add"'),
    dict(id="s2", label="lower_to_mlir → TTIR", sub="ttir.add · lower_to_mlir.cpp:843"),
    dict(id="s3", label="stamp the descriptor", sub="ttcore-register-device"),
    dict(id="s4", label="TTNN", sub="ttnn.add · arch-neutral"),
    dict(id="s5", label=".ttnn flatbuffer", sub="EltwiseBinaryOp"),
    dict(id="s6", label="runtime dispatch", sub="isQuasar() ? :"),
    dict(id="s7", label="Tensix kernel", sub="add_tiles on the FPU"),
]

# side: -1 above, +1 below.  attach: spine index.
CHANGES = [
    dict(
        id="c1", attach=3, side=-1, tone=SYSDESC, tag="CHANGE 1",
        title="System descriptor corrected for Quasar",
        edge="we modified this",
        lines=[
            "It advertised Wormhole's 13 data formats — including",
            "block-float bf8_b / bf4_b, which no Quasar device runs.",
            "Both the mock and the live builder hardcoded the same",
            "list, so they agreed and the bug stayed invisible.",
            "",
            "NOW: both ask tt-metal.",
            "  tt::is_data_format_supported(format, arch)",
            "  quasar 5 formats · wormhole unchanged at 13",
            "  num_cbs from the HAL, not an array-sizing constant",
            "",
            "TTCoreOpsTypes.cpp:85 · system_desc.cpp:181, :238",
        ],
    ),
    dict(
        id="c2", attach=1, side=+1, tone=PATCHED, tag="CHANGE 2",
        title="bf16, not f32",
        edge="we set this",
        lines=[
            "cfg.default_df_override =",
            "    forge._C.DataFormat.Float16_b",
            "",
            "Not a narrower dtype — a different compute path:",
            "is_binary_sfpu_op is true for ANY f32 op including",
            "add, so f32 routes Quasar's SFPU kernel and bf16",
            "routes the FPU binary_ng kernel. f32 livelocks.",
        ],
    ),
    dict(
        id="c3", attach=6, side=-1, tone=PATCHED, tag="CHANGE 3",
        title="ttnn op → ttnn experimental Quasar op",
        edge="THE MAPPING",
        lines=[
            "RUN_ELTWISE_BINARY(NAME) picks the entry point:",
            "",
            "  utils::isQuasar()",
            "    ? experimental::quasar::binary::NAME(args...)",
            "    : ::ttnn::NAME(args...)",
            "",
            "One forwarded pack binds to both, because the Quasar",
            "op takes the same leading (lhs, rhs, output_dtype,",
            "memory_config) — so all 14 eltwise binaries landed at",
            "once. Mainline is REFUSED on Quasar, not slower:",
            "DataMovementKernel's ctor TT_FATALs.",
            "",
            "binary.cpp:45 → binary.hpp:183 · kernel.hpp:417",
        ],
    ),
    dict(
        id="c4", attach=6, side=+1, tone=PATCHED, tag="CHANGE 4",
        title="Run from $TT_METAL_HOME",
        edge="we fixed this",
        lines=[
            "Quasar's binary_ng factory passes kernel include",
            "paths RELATIVE to the tt-metal root, and tt-metal",
            "resolves them against the process cwd, not",
            "TT_METAL_HOME (kernel.cpp:100-112).",
            "",
            "Identical upstream — a tt-metal convention, not a bug:",
            "their suite always runs from their repo root.",
            "An autouse fixture chdirs for the test.",
        ],
    ),
]

RESULT = dict(
    tone=SYSDESC,
    title="RESULT — Add/Mul/Sub/Div run and verify on craq-sim",
    lines=[
        "Add: PCC 0.999985, ~1.4 s of execution. 4 passed in 11 s. No tt-metal pin bump needed.",
        "None of the four changes touched the op mapping's correctness — it was already wired and already being reached.",
    ],
)

OPEN = dict(
    tone=BLOCKED,
    title="STILL OPEN — the memory config, not the op",
    lines=[
        "At ResNet-50 shapes the same ops FAIL from forge (add pcc 0.756) yet PASS from tt-metal (4 passed).",
        "Difference: forge emits DRAM-interleaved on a 1x1 grid; the model config is height-sharded across 32 cores in L1.",
        "Forge emits that because the optimizer is compiled out (TTMLIR_ENABLE_OPMODEL=OFF) — so enabling it is a CORRECTNESS prerequisite, not a perf task.",
    ],
)

ANCHOR_ROWS = [("change map", [
    ("ttir emit", "forge/csrc/passes/lower_to_mlir.cpp:843"),
    ("mock desc", "third_party/tt-mlir/lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp:99"),
    ("live desc", "third_party/tt-mlir/runtime/lib/common/system_desc.cpp:238"),
    ("register device",
     "third_party/tt-mlir/lib/Dialect/TTCore/Transforms/TTCoreRegisterDevice.cpp:45"),
    ("the mapping",
     "third_party/tt-mlir/runtime/lib/ttnn/operations/eltwise/binary/binary.cpp"),
    ("kernel resolve",
     "third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/kernels/kernel.cpp:112"),
    ("the TT_FATAL",
     "third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/kernels/kernel.hpp:417"),
])]

CW = 560                      # change-card width
LINE_H = 12.5


def spine_x(i):
    return SX0 + i * (SW + SGAP)


def card_geom(ch):
    """A change card sits over/under its spine box, clamped inside the canvas."""
    cx = spine_x(ch["attach"]) + SW / 2
    x = max(20, min(cx - CW / 2, W - 20 - CW))
    h = 56 + int(len(ch["lines"]) * LINE_H)
    y = SY - 60 - h if ch["side"] < 0 else SY + SH + 60
    return x, y, CW, h


def banner_geom(which):
    h = 44 + int(len(which["lines"]) * LINE_H)
    y = 30 if which is RESULT else H - 20 - h
    return 40, y, W - 80, h


def emit_mxfile():
    cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']

    def box(cid, value, style, x, y, w, h):
        cells.append(
            f'<mxCell id={quoteattr(cid)} value={quoteattr(value)} '
            f'style={quoteattr(style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />'
            "</mxCell>")

    box("title",
        f'<b>{escape(TITLE)}</b><br><font style="font-size:11px">{escape(SUB)}</font>',
        "text;html=1;align=left;verticalAlign=middle;fontSize=18;fontColor=#0f172a;",
        40, 132, W - 80, 52)

    for i, s in enumerate(SPINE):
        fill, stroke, font = BADGE_STYLE[NEUTRAL]
        box(s["id"],
            f'<b>{escape(s["label"])}</b><br>'
            f'<font style="font-size:9.5px">{escape(s["sub"])}</font>',
            f"rounded=1;arcSize=20;whiteSpace=wrap;html=1;fillColor={fill};"
            f"strokeColor={stroke};fontColor={font};align=center;"
            "verticalAlign=middle;fontSize=12;strokeWidth=1.8;",
            spine_x(i), SY, SW, SH)

    for i in range(len(SPINE) - 1):
        cells.append(
            f'<mxCell id="sp{i}" style={quoteattr("edgeStyle=orthogonalEdgeStyle;"
                "rounded=1;html=1;endArrow=block;endFill=1;strokeColor=#3f4854;"
                "strokeWidth=2.4;")} edge="1" parent="1" '
            f'source={quoteattr(SPINE[i]["id"])} '
            f'target={quoteattr(SPINE[i + 1]["id"])}>'
            '<mxGeometry relative="1" as="geometry" /></mxCell>')

    for ch in CHANGES:
        x, y, w, h = card_geom(ch)
        fill, stroke, font = BADGE_STYLE[ch["tone"]]
        body = [
            f'<b>{escape(ch["tag"])} — {escape(ch["title"])}</b>',
            "",
        ] + [f'<font style="font-size:9px">{escape(l)}</font>' if l else "&nbsp;"
             for l in ch["lines"]]
        box(ch["id"], "<br>".join(body),
            f"rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor={fill};"
            f"strokeColor={stroke};fontColor={font};align=left;verticalAlign=top;"
            "spacingLeft=11;spacingTop=6;fontSize=11;strokeWidth=2;", x, y, w, h)
        cells.append(
            f'<mxCell id="e_{ch["id"]}" value={quoteattr(ch["edge"])} '
            f'style={quoteattr("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
                f"endArrow=block;endFill=1;strokeColor={stroke};strokeWidth=3;"
                "fontSize=10;labelBackgroundColor=none;")} '
            f'edge="1" parent="1" source={quoteattr(SPINE[ch["attach"]]["id"])} '
            f'target={quoteattr(ch["id"])}>'
            '<mxGeometry relative="1" as="geometry" /></mxCell>')

    for name, which in (("result", RESULT), ("open", OPEN)):
        x, y, w, h = banner_geom(which)
        fill, stroke, font = BADGE_STYLE[which["tone"]]
        body = [f'<b>{escape(which["title"])}</b>', ""] + [
            f'<font style="font-size:9.5px">{escape(l)}</font>' for l in which["lines"]]
        box(name, "<br>".join(body),
            f"rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor={fill};"
            f"strokeColor={stroke};fontColor={font};align=left;verticalAlign=top;"
            "spacingLeft=11;spacingTop=5;fontSize=12;strokeWidth=2;", x, y, w, h)

    body = "".join(cells)
    return ('<mxfile host="app.diagrams.net" agent="gen_quasar_change_map.py" '
            'type="device"><diagram id="quasar-change-map" name="quasar changes">'
            f'<mxGraphModel dx="{W}" dy="{H}" grid="0" gridSize="10" guides="1" '
            'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            f'pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">'
            f"<root>{body}</root></mxGraphModel></diagram></mxfile>")


def emit_svg(mxfile):
    p = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']
    p.append(svg_text(40, 152, TITLE, 19, "bold"))
    p.append(svg_text(40, 172, SUB, 11, "normal", "#5b6472"))

    def card(x, y, w, h, tone, title, lines, title_size=11.5, line_size=9):
        fill, stroke, font = BADGE_STYLE[tone]
        out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
               f'stroke="{stroke}" stroke-width="2"/>']
        out.append(svg_text(x + 12, y + 22, title, title_size, "bold", font))
        ty = y + 40
        for l in lines:
            if l:
                out.append(svg_text(x + 12, ty, l, line_size, "normal", "#3f4854"))
            ty += LINE_H
        return out

    for name, which in (("result", RESULT), ("open", OPEN)):
        x, y, w, h = banner_geom(which)
        p += card(x, y, w, h, which["tone"], which["title"], which["lines"],
                  title_size=12.5, line_size=9.5)

    for ch in CHANGES:
        x, y, w, h = card_geom(ch)
        p += card(x, y, w, h, ch["tone"], f'{ch["tag"]} — {ch["title"]}', ch["lines"])

    for i, s in enumerate(SPINE):
        fill, stroke, font = BADGE_STYLE[NEUTRAL]
        x = spine_x(i)
        p.append(f'<rect x="{x}" y="{SY}" width="{SW}" height="{SH}" rx="18" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
        p.append(f'<text x="{x + SW / 2}" y="{SY + 36}" text-anchor="middle" '
                 'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
                 f'font-size="12.5" font-weight="bold" fill="{font}">'
                 f'{escape(s["label"])}</text>')
        p.append(f'<text x="{x + SW / 2}" y="{SY + 55}" text-anchor="middle" '
                 'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
                 f'font-size="9.5" fill="#3f4854">{escape(s["sub"])}</text>')

    y = SY + SH / 2
    for i in range(len(SPINE) - 1):
        p.append(f'<path d="M {spine_x(i) + SW} {y} L {spine_x(i + 1) - 5} {y}" '
                 'stroke="#3f4854" stroke-width="2.4" fill="none" '
                 'marker-end="url(#cm0)"/>')

    for n, ch in enumerate(CHANGES):
        x, cy, w, h = card_geom(ch)
        _, stroke, _ = BADGE_STYLE[ch["tone"]]
        sx = spine_x(ch["attach"]) + SW / 2
        tx = x + w / 2
        if ch["side"] < 0:
            y0, y1, mid = SY, cy + h, (SY + cy + h) / 2
        else:
            y0, y1, mid = SY + SH, cy, (SY + SH + cy) / 2
        p.append(f'<path d="M {sx} {y0} L {sx} {mid} L {tx} {mid} L {tx} {y1}" '
                 f'stroke="{stroke}" stroke-width="3" fill="none" '
                 f'marker-end="url(#cm{n + 1})"/>')
        lx, ly = sx + 12, mid - 6 if ch["side"] < 0 else mid + 14
        p.append(f'<rect x="{lx - 3}" y="{ly - 11}" '
                 f'width="{len(ch["edge"]) * 5.6 + 8}" height="15" fill="#ffffff" '
                 'stroke="none"/>')
        p.append(f'<text x="{lx}" y="{ly}" '
                 'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
                 f'font-size="10" font-weight="bold" fill="{stroke}">'
                 f'{escape(ch["edge"])}</text>')

    defs = ["<defs>"]
    defs.append('<marker id="cm0" viewBox="0 0 10 10" refX="9" refY="5" '
                'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                '<path d="M 0 1 L 9 5 L 0 9 z" fill="#3f4854"/></marker>')
    for n, ch in enumerate(CHANGES):
        _, stroke, _ = BADGE_STYLE[ch["tone"]]
        defs.append(f'<marker id="cm{n + 1}" viewBox="0 0 10 10" refX="9" refY="5" '
                    'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                    f'<path d="M 0 1 L 9 5 L 0 9 z" fill="{stroke}"/></marker>')
    defs.append("</defs>")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{W}" '
            f'height="{H}" viewBox="0 0 {W} {H}" content={quoteattr(mxfile)}>'
            f"{''.join(defs)}{''.join(p)}</svg>\n")


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

    right = spine_x(len(SPINE) - 1) + SW
    if right > W - 20:
        raise SystemExit(f"spine ends at x={right}, canvas is {W}")

    rects = [("spine", spine_x(i), SY, SW, SH) for i in range(len(SPINE))]
    rects += [(c["id"], *card_geom(c)) for c in CHANGES]
    rects += [(n, *banner_geom(w)) for n, w in (("result", RESULT), ("open", OPEN))]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            (na, ax, ay, aw, ah), (nb, bx, by, bw, bh) = rects[i], rects[j]
            if na == nb == "spine":
                continue
            if (ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah):
                raise SystemExit(f"overlap: {na} / {nb}")
    for n, x, y, w, h in rects:
        if x < 0 or x + w > W or y < 0 or y + h > H:
            raise SystemExit(f"{n} outside the {W}x{H} canvas: {x},{y},{w},{h}")
        for ch in CHANGES:
            if ch["id"] == n:
                for l in ch["lines"]:
                    if len(l) * 5.25 + 24 > w:
                        raise SystemExit(f"{n}: line needs "
                                         f"{len(l) * 5.25 + 24:.0f}px, card is {w}: {l!r}")
    print(f"layout: {len(SPINE)} spine boxes, {len(CHANGES)} change cards, "
          f"2 banners, no overlaps, all text fits")

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
