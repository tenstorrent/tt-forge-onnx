#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the stage-wise Quasar support flow.

  docs/source/imgs/compiler_arch/quasar-bringup-phases.drawio.svg

An SVG that is also a diagrams.net document, so it renders in the docs and opens
editable at app.diagrams.net.

One left-to-right chain of stages, each carrying its status, what it delivers and the
files it touches. Drawn as a flow rather than a dependency graph because the ordering
is the lesson: the critical path runs through stage 2, execution, not stage 3, op
dispatch -- dispatch work cannot be validated while nothing runs, so the intuitive
descriptor-then-ops-then-execute ordering wastes effort.

Status is per stage and current as of the header date. Regenerate after any stage
moves; the generator checks that every source anchor it quotes still resolves.

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

W, H = 2200, 700
BW, BH = 240, 258
GAP = 28
X0 = 40
ROW_Y = 190

TITLE = "Quasar support, stage by stage"
SUB = ("Status as of 2026-09-07. The ordering is the lesson: the critical path runs "
       "through stage 2, not stage 3 — op dispatch cannot be validated while nothing "
       "executes.")

DONE, NOW, NEXT, EXT, LATER = "done", "now", "next", "external", "later"
STATUS_TONE = {DONE: SYSDESC, NOW: PATCHED, NEXT: PATCHED,
               EXT: BLOCKED, LATER: NEUTRAL}
STATUS_LABEL = {DONE: "DONE", NOW: "GREEN — as of today", NEXT: "NEXT — the bulk",
                EXT: "METAL OWNS", LATER: "AFTER CORRECTNESS"}

STAGES = [
    dict(
        id="s0", n="0", title="Make the work survive",
        status=DONE, owner="ours",
        lines=[
            "Quasar support lives on branches that",
            "the build force-checks-out back to its",
            "pins. It had already silently reverted",
            "once.",
            "",
            "DONE: tt-mlir branch pushed, forge pin",
            "bumped to it, so a force-checkout now",
            "lands ON the fix.",
            "",
            "third_party/tt-mlir gitlink",
            "third_party/CMakeLists.txt:3",
        ],
    ),
    dict(
        id="s1", n="1", title="Descriptor truth",
        status=DONE, owner="ours",
        lines=[
            "The compiler is arch-neutral, so the",
            "descriptor's numbers are the ONLY route",
            "an arch reaches compilation.",
            "",
            "DONE: data formats now come from",
            "tt::is_data_format_supported, not a",
            "hardcoded Wormhole list. num_cbs from",
            "the HAL. Quasar 5 formats, WH 13.",
            "",
            "TTCoreOpsTypes.cpp:85",
            "system_desc.cpp:181, :238",
        ],
    ),
    dict(
        id="s2", n="2", title="Get anything to execute",
        status=NOW, owner="ours",
        lines=[
            "THE CRITICAL PATH. Until a graph runs,",
            "every dispatch change in stage 3 is",
            "unverifiable — you can compile it, but",
            "not tell correct from wrong.",
            "",
            "GREEN: Add/Mul/Sub/Div run and verify",
            "in bf16. Add PCC 0.999985, ~1.4 s.",
            "Needed bf16 + cwd=$TT_METAL_HOME.",
            "",
            "OPEN: f32 livelocks — SFPU vs FPU",
            "kernel, a different compute path.",
        ],
    ),
    dict(
        id="s3", n="3", title="Finish op dispatch",
        status=NEXT, owner="ours",
        lines=[
            "Where the arches actually diverge, and",
            "the bulk of our work. Mechanical: an",
            "isQuasar() branch and a namespace swap.",
            "",
            "9 of 121 runtime op files wired today.",
            "28 Quasar op families exist in tt-metal,",
            "10 reached — so 18 are available and",
            "unwired: pad, slice, transpose, typecast,",
            "to_memory_config, tilize/untilize, ...",
            "",
            "Scope from the target model, not the",
            "131 OpTypes.",
        ],
    ),
    dict(
        id="s4", n="4", title="The genuine Metal asks",
        status=EXT, owner="Metal",
        lines=[
            "Not ours. Cite rather than rediscover —",
            "QUASAR_PARITY_GAPS.md §Priorities is",
            "Metal's own owned list.",
            "",
            "conv2d: Gen1/Gen2 compute-config",
            "  mismatch, tt-metal #48552",
            "float compares: unported (Quasar HAS",
            "  compare SFPU, Int32 only)",
            "int32 DFB bug · max/min reroute",
            "a unary family, if relu-as-add is not",
            "  acceptable long term",
        ],
    ),
    dict(
        id="s5", n="5", title="Perf modelling",
        status=LATER, owner="ours",
        lines=[
            "The pass self-disables on Quasar rather",
            "than producing wrong numbers, which is",
            "the right default — but it means no",
            "perf estimate exists at all.",
            "",
            "No calibrated DRAM BW or AICLK.",
            "cyclesPerTileMatmul hardcoded.",
            "Device profiling blocked upstream.",
            "",
            "TTNNCollectPerfMetrics.cpp:809, :1063",
        ],
    ),
    dict(
        id="s6", n="6", title="Turn the optimizer on",
        status=LATER, owner="ours",
        lines=[
            "Where arch-neutrality ENDS — and where",
            "the performance story lives.",
            "",
            "Every pass is arch-neutral today only",
            "because TTMLIR_ENABLE_OPMODEL is OFF, so",
            "nothing shards or picks L1 layouts and",
            "everything lands DRAM-interleaved at",
            "opt level 0.",
            "",
            "Quasar's 4 MiB L1 — 2.8x Wormhole's —",
            "is currently unexploited.",
            "",
            "CMakeLists.txt:40",
        ],
    ),
    dict(
        id="s7", n="7", title="Scale-out & D2M",
        status=LATER, owner="deferred",
        lines=[
            "Hard-rejected upstream today. Listed so",
            "nobody plans around them, not because",
            "they need doing now.",
            "",
            "Multi-chip dispatch TT_THROWs.",
            "D2M -> TTMetal/TTKernel hard-errors, so",
            "only the TTNN path is viable — which is",
            "the path forge takes anyway.",
            "",
            "topology.cpp:494",
            "DMAUtils.cpp:51-60",
        ],
    ),
]

FOOT = [
    "Stages 0-2 are green as of today, which is what unblocked everything downstream — stage 3 was always the bigger pile of work, but it was unmeasurable until stage 2 landed.",
    "What needs NO stage: every pass in ttir-to-ttnn-backend-pipeline (all 57, measured — the two arch dumps differ on three descriptor lines), the forge frontend and TVM path, and the flatbuffer schema.",
]

ANCHOR_ROWS = [("stages", [
    ("pin", "third_party/tt-mlir/third_party/CMakeLists.txt:3"),
    ("mock desc", "third_party/tt-mlir/lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp:99"),
    ("live desc", "third_party/tt-mlir/runtime/lib/common/system_desc.cpp:238"),
    ("sim test", "forge/test/mlir/test_quasar_sim.py"),
    ("perf", "third_party/tt-mlir/lib/Dialect/TTNN/Transforms/TTNNCollectPerfMetrics.cpp:1063"),
    ("opmodel", "third_party/tt-mlir/CMakeLists.txt:40"),
    ("d2m", "third_party/tt-mlir/lib/Dialect/D2M/Utils/DMAUtils.cpp:60"),
])]


def sx(i):
    return X0 + i * (BW + GAP)


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
         40, 30, W - 80, 60)

    for i, s in enumerate(STAGES):
        fill, stroke, font = BADGE_STYLE[STATUS_TONE[s["status"]]]
        style = (f"rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor={fill};"
                 f"strokeColor={stroke};fontColor={font};align=left;verticalAlign=top;"
                 "spacingLeft=10;spacingTop=6;fontSize=12;strokeWidth=1.6;")
        body = [
            f'<b>{s["n"]} · {escape(s["title"])}</b>',
            f'<font style="font-size:9px"><b>{escape(STATUS_LABEL[s["status"]])}</b>'
            f'&nbsp;&nbsp;<font style="color:#5b6472">{escape(s["owner"])}</font></font>',
            "",
        ]
        body += [f'<font style="font-size:9px">{escape(l)}</font>' if l else "&nbsp;"
                 for l in s["lines"]]
        cell(s["id"], "<br>".join(body), style, sx(i), ROW_Y, BW, BH)

    for i in range(len(STAGES) - 1):
        gate = i == 2
        style = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;"
                 "endFill=1;fontSize=9;labelBackgroundColor=none;"
                 + ("strokeColor=#b8791f;strokeWidth=3;" if gate
                    else "strokeColor=#3f4854;strokeWidth=2.2;"))
        cells.append(
            f'<mxCell id="e{i}" value={quoteattr("gates" if gate else "")} '
            f'style={quoteattr(style)} edge="1" parent="1" '
            f'source={quoteattr(STAGES[i]["id"])} '
            f'target={quoteattr(STAGES[i + 1]["id"])}>'
            '<mxGeometry relative="1" as="geometry" /></mxCell>')

    cell("foot", "<br>".join(f'<font style="font-size:10px">{escape(l)}</font>'
                             for l in FOOT),
         "text;html=1;align=left;verticalAlign=middle;fontSize=10;fontColor=#3f4854;",
         40, 480, W - 80, 46)
    body = "".join(cells)
    return ('<mxfile host="app.diagrams.net" agent="gen_quasar_phase_diagram.py" '
            'type="device"><diagram id="quasar-stages" name="quasar stages">'
            f'<mxGraphModel dx="{W}" dy="{H}" grid="0" gridSize="10" guides="1" '
            'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            f'pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">'
            f"<root>{body}</root></mxGraphModel></diagram></mxfile>")


def emit_svg(mxfile):
    p = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']
    p.append(svg_text(40, 44, TITLE, 18, "bold"))
    words, lines, cur = SUB.split(), [], ""
    for w in words:
        if len(f"{cur} {w}".strip()) <= 150:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    for i, l in enumerate(lines):
        p.append(svg_text(40, 64 + i * 14, l, 11, "normal", "#5b6472"))

    for i, s in enumerate(STAGES):
        fill, stroke, font = BADGE_STYLE[STATUS_TONE[s["status"]]]
        x = sx(i)
        p.append(f'<rect x="{x}" y="{ROW_Y}" width="{BW}" height="{BH}" rx="11" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
        p.append(svg_text(x + 11, ROW_Y + 22, f'{s["n"]} · {s["title"]}', 12, "bold",
                          font))
        p.append(svg_text(x + 11, ROW_Y + 37, STATUS_LABEL[s["status"]], 9, "bold",
                          stroke))
        p.append(svg_text(x + 11 + int(len(STATUS_LABEL[s["status"]]) * 5.2) + 10,
                          ROW_Y + 37, s["owner"], 9, "normal", "#5b6472"))
        ty = ROW_Y + 54
        for l in s["lines"]:
            if l:
                p.append(svg_text(x + 11, ty, l, 9, "normal", "#3f4854"))
            ty += 11.5

    y = ROW_Y + BH / 2
    for i in range(len(STAGES) - 1):
        gate = i == 2
        col = "#b8791f" if gate else "#3f4854"
        wid = "3" if gate else "2.2"
        mk = "url(#sg)" if gate else "url(#sa)"
        p.append(f'<path d="M {sx(i) + BW} {y} L {sx(i + 1) - 5} {y}" '
                 f'stroke="{col}" stroke-width="{wid}" fill="none" marker-end="{mk}"/>')
        if gate:
            p.append(f'<rect x="{sx(i) + BW + 1}" y="{y - 18}" width="30" height="13" '
                     'fill="#ffffff" stroke="none"/>')
            p.append(f'<text x="{sx(i) + BW + 3}" y="{y - 8}" '
                     'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
                     f'font-size="9.5" font-weight="bold" fill="{col}">gates</text>')

    for i, l in enumerate(FOOT):
        p.append(svg_text(40, 494 + i * 15, l, 10, "normal", "#3f4854"))

    defs = ("<defs>"
            + "".join(
                f'<marker id="{n}" viewBox="0 0 10 10" refX="9" refY="5" '
                'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M 0 1 L 9 5 L 0 9 z" fill="{c}"/></marker>'
                for n, c in (("sa", "#3f4854"), ("sg", "#b8791f")))
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

    right = sx(len(STAGES) - 1) + BW
    if right > W - 20:
        raise SystemExit(f"chain ends at {right}, past the {W}px canvas")
    if ROW_Y + BH > H - 20:
        raise SystemExit("stages overflow the canvas height")
    longest = max(len(l) for s in STAGES for l in s["lines"])
    if longest * 5.2 + 22 > BW:
        raise SystemExit(f"a body line needs {longest * 5.2 + 22:.0f}px, box is {BW}")
    print(f"layout: {len(STAGES)} stages, chain ends at x={right}, "
          f"longest line {longest} chars, fits")

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
