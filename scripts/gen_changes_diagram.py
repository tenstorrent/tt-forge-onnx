#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the "what we changed" diagram for the Quasar add bring-up.

  docs/source/imgs/compiler_arch/quasar-changes.drawio.svg

An SVG that is also a diagrams.net document, so it renders in the docs and opens
editable at app.diagrams.net.

Built for explaining the work out loud. Three rows, one per change, each read
left-to-right as BEFORE -> WHAT WE CHANGED -> AFTER, with the file and line on the
change box so a listener can follow along in the source. The companion script for the
call is docs/source/dev_notes/quasar_changes_transcript.md.

    python scripts/gen_changes_diagram.py
    python scripts/gen_changes_diagram.py --check
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
    REPO, "docs", "source", "imgs", "compiler_arch", "quasar-changes.drawio.svg"
)

W, H = 1860, 1080
COL = [40, 660, 1280]          # before / change / after
CW = 560                        # column width
ROW_Y = [150, 400, 650]
RH = 210

TITLE = "Running Add on Quasar: what we changed"
SUB = ("Three changes. One is a real code fix in tt-mlir; the other two are how the "
       "compiler has to be driven. None of them touched the op mapping — that was "
       "already correct.")

ROWS = [
    dict(
        n="1",
        headline="The system descriptor lied about Quasar's data formats",
        before_tone=BLOCKED,
        before_head="BEFORE — one list for every arch",
        before=[
            "Quasar advertised Wormhole's 13 formats,",
            "including bfp_bf8 / bfp_bf4 / u16 / u32,",
            "none of which Quasar can run.",
            "",
            "Both the mock AND the live builder had the",
            "same hardcoded list, so they agreed with",
            "each other and the bug was invisible.",
            "The live builder even said so:",
            '  “temporary place-holder value to be',
            '   replaced by API value”',
        ],
        change_tone=PATCHED,
        change_head="CHANGE — ask tt-metal instead",
        change_anchor="tt-mlir: TTCoreOpsTypes.cpp:85  +  system_desc.cpp:181",
        change=[
            "tt::is_data_format_supported(fmt, arch)",
            "  tt_backend_api_types.hpp:72  (public API)",
            "  -> is_supported_quasar()",
            "",
            "Live builder: enumerate all 13 target types,",
            "map each to a tt::DataFormat, keep what that",
            "arch admits. No hardcoded list at all.",
            "",
            "Mock: carry the matching 5 so a device-free",
            "compile agrees with the device.",
        ],
        after_tone=SYSDESC,
        after_head="AFTER — arch-correct, and self-consistent",
        after=[
            "quasar   : f32, f16, bf16, u8, si32     (5)",
            "wormhole : unchanged                    (13)",
            "",
            "num_cbs fixed the same way:",
            "  was NUM_CIRCULAR_BUFFERS, an array-sizing",
            "  constant = 64 on ANY host build",
            "  now hal::get_arch_num_circular_buffers()",
            "  -> live Wormhole 64 -> 32 (was wrong!)",
            "",
            "Verified 4 ways: mock + live, both arches.",
        ],
    ),
    dict(
        n="2",
        headline="Forge emitted f32, which takes a different kernel",
        before_tone=BLOCKED,
        before_head="BEFORE — f32 livelocks",
        before=[
            "An ONNX graph is float32, so forge emitted",
            "f32 tensors and the run hung forever:",
            "",
            "  pending_tensix = 4",
            "  srcA/srcB = valid=1 unpack=1 matrix=0",
            "",
            "Operands delivered; the math unit never",
            "consumed them. The watchdog stays silent by",
            "design — the loop keeps retiring",
            "instructions, so it is a livelock.",
        ],
        change_tone=PATCHED,
        change_head="CHANGE — ask for bf16",
        change_anchor="one line of config — no code change",
        change=[
            "cfg = CompilerConfig()",
            "cfg.default_df_override = \\",
            "    forge._C.DataFormat.Float16_b",
            "",
            "Not just a narrower dtype:",
            "  is_binary_sfpu_op is TRUE for any f32 op",
            "  including add, so f32 routes the SFPU",
            "  kernel, and bf16 routes the FPU",
            "  binary_ng kernel.",
            "A different compute path, not a wider one.",
        ],
        after_tone=SYSDESC,
        after_head="AFTER — bf16 executes",
        after=[
            'IR: "ttnn.add"(%0, %1)',
            "       : (tensor<2x32x32xbf16, ...)",
            "",
            "Executes in ~1.4 s.",
            "PCC 0.999985",
            "",
            "f32 still livelocks and is tracked",
            "separately — tests marked `f32`,",
            "deselectable with -k \"not f32\".",
        ],
    ),
    dict(
        n="3",
        headline="Kernel include paths resolve against the cwd, not TT_METAL_HOME",
        before_tone=BLOCKED,
        before_head="BEFORE — fails from the forge repo",
        before=[
            "With bf16 the hang was replaced by a hard,",
            "fast error:",
            "",
            "  TT_THROW: Compiler include directory",
            "  'ttnn/cpp/.../binary_ng/device/kernels/",
            "   compute' not found relative to current",
            "  working directory '/.../tt-forge-onnx'",
            "",
            "Quasar's binary_ng factory passes include",
            "paths RELATIVE to the tt-metal root.",
        ],
        change_tone=PATCHED,
        change_head="CHANGE — chdir for the test",
        change_anchor="forge: test/mlir/test_quasar_sim.py  (_tt_metal_cwd fixture)",
        change=[
            "@pytest.fixture(autouse=True)",
            "def _tt_metal_cwd():",
            "    prev = os.getcwd()",
            "    os.chdir(os.environ['TT_METAL_HOME'])",
            "    try:    yield",
            "    finally: os.chdir(prev)",
            "",
            "resolve_compiler_include_dir uses",
            "fs::current_path(), NOT TT_METAL_HOME",
            "  kernel.cpp:100-112",
        ],
        after_tone=SYSDESC,
        after_head="AFTER — runs from anywhere",
        after=[
            "Identical upstream, so this is a tt-metal",
            "CONVENTION, not a bug: their suite always",
            "runs from their repo root, so it never bit",
            "them. forge is a different repo.",
            "",
            "Outside pytest: cd \"$TT_METAL_HOME\" first.",
            "",
            "4 passed in 11 s, run from the forge root.",
        ],
    ),
]

RESULT = dict(
    head="RESULT",
    lines=[
        "test_add_bf16 / test_mul_bf16 / test_sub_bf16 / test_div_bf16   ->  4 passed in 11 s,  Add PCC 0.999985",
        "test_add_op.py  ->  16 passed in 14 s, device-free  (5 of them Quasar: lowering, dtype, descriptor contents, Bfp8_b rejection)",
        "No tt-metal pin bump was needed.   Open: f32 still livelocks — different kernel path, tracked separately.",
    ],
)

WHY = dict(
    head="Why none of this was the op mapping",
    lines=[
        "The mapping was already correct and already being reached: binary.cpp's RUN_ELTWISE_BINARY picks",
        "ttnn::operations::experimental::quasar::binary::add when isQuasar(). Score every op three ways —",
        "(1) does a Quasar op exist, (2) does the runtime dispatch to it, (3) does it run. Only (3) counts,",
        "and (1)+(2) being green is exactly why “add is unmapped” was the wrong diagnosis for weeks.",
    ],
)

ANCHOR_ROWS = [("changes", [
    ("mock desc", "third_party/tt-mlir/lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp:99"),
    ("live desc", "third_party/tt-mlir/runtime/lib/common/system_desc.cpp:238"),
    ("kernel resolve",
     "third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/kernels/kernel.cpp:112"),
    ("the fixture", "forge/test/mlir/test_quasar_sim.py"),
    ("dtype tests", "forge/test/mlir/test_add_op.py"),
    ("dispatch", "third_party/tt-mlir/runtime/lib/ttnn/operations/eltwise/binary/binary.cpp"),
])]


def boxes():
    out = []
    for r, row in enumerate(ROWS):
        y = ROW_Y[r]
        out.append(dict(id=f"h{r}", x=COL[0], y=y - 34, w=W - 80, h=26,
                        tone=None, head=f"{row['n']}.  {row['headline']}", lines=[]))
        for c, key in enumerate(("before", "change", "after")):
            out.append(dict(
                id=f"b{r}{c}", x=COL[c], y=y, w=CW, h=RH,
                tone=row[f"{key}_tone"], head=row[f"{key}_head"],
                anchor=row.get("change_anchor") if key == "change" else None,
                lines=row[key],
            ))
    out.append(dict(id="why", x=COL[0], y=880, w=W - 80, h=86, tone=NEUTRAL,
                    head=WHY["head"], lines=WHY["lines"]))
    out.append(dict(id="result", x=COL[0], y=976, w=W - 80, h=72, tone=SYSDESC,
                    head=RESULT["head"], lines=RESULT["lines"]))
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
         40, 34, W - 80, 64)

    for b in boxes():
        if b["tone"] is None:
            cell(b["id"], f'<b>{escape(b["head"])}</b>',
                 "text;html=1;align=left;verticalAlign=middle;fontSize=14;"
                 "fontColor=#0f172a;", b["x"], b["y"], b["w"], b["h"])
            continue
        fill, stroke, font = BADGE_STYLE[b["tone"]]
        style = (f"rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor={fill};"
                 f"strokeColor={stroke};fontColor={font};align=left;verticalAlign=top;"
                 "spacingLeft=10;spacingTop=4;fontSize=11;")
        body = [f'<b>{escape(b["head"])}</b>']
        if b.get("anchor"):
            body.append(f'<font style="font-size:9px;color:#5b6472">'
                        f'{escape(b["anchor"])}</font>')
        body.append("")
        body += [f'<font style="font-size:9px">{escape(l)}</font>' if l else "&nbsp;"
                 for l in b["lines"]]
        cell(b["id"], "<br>".join(body), style, b["x"], b["y"], b["w"], b["h"])

    # before -> change -> after arrows, per row
    for r in range(len(ROWS)):
        for c in (0, 1):
            style = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;"
                     "endFill=1;strokeColor=#4b5563;strokeWidth=2;")
            cells.append(
                f'<mxCell id="e{r}{c}" style={quoteattr(style)} edge="1" parent="1" '
                f'source={quoteattr(f"b{r}{c}")} target={quoteattr(f"b{r}{c+1}")}>'
                '<mxGeometry relative="1" as="geometry" /></mxCell>')

    body = "".join(cells)
    return ('<mxfile host="app.diagrams.net" agent="gen_changes_diagram.py" '
            'type="device"><diagram id="quasar-changes" name="what we changed">'
            f'<mxGraphModel dx="{W}" dy="{H}" grid="0" gridSize="10" guides="1" '
            'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            f'pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">'
            f"<root>{body}</root></mxGraphModel></diagram></mxfile>")


def emit_svg(mxfile):
    p = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']
    p.append(svg_text(40, 58, TITLE, 18, "bold"))
    words, lines, cur = SUB.split(), [], ""
    for w in words:
        if len(f"{cur} {w}".strip()) <= 132:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    for i, l in enumerate(lines):
        p.append(svg_text(40, 80 + i * 15, l, 11, "normal", "#5b6472"))

    for b in boxes():
        if b["tone"] is None:
            p.append(svg_text(b["x"], b["y"] + 18, b["head"], 14, "bold", "#0f172a"))
            continue
        fill, stroke, font = BADGE_STYLE[b["tone"]]
        p.append(f'<rect x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" '
                 f'height="{b["h"]}" rx="5" fill="{fill}" stroke="{stroke}" '
                 'stroke-width="1.3"/>')
        p.append(svg_text(b["x"] + 12, b["y"] + 20, b["head"], 11.5, "bold", font))
        ty = b["y"] + 20
        if b.get("anchor"):
            ty += 14
            p.append(svg_text(b["x"] + 12, ty, b["anchor"], 9, "normal", "#5b6472"))
        ty += 16
        for l in b["lines"]:
            if l:
                p.append(svg_text(b["x"] + 12, ty, l, 9, "normal", "#3f4854"))
            ty += 12

    for r in range(len(ROWS)):
        y = ROW_Y[r] + RH / 2
        for c in (0, 1):
            x0 = COL[c] + CW
            x1 = COL[c + 1] - 4
            p.append(f'<path d="M {x0} {y} L {x1} {y}" stroke="#4b5563" '
                     'stroke-width="2" fill="none" marker-end="url(#ca)"/>')

    defs = ("<defs>"
            '<marker id="ca" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
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

    bs = boxes()
    for b in bs:
        if b["x"] + b["w"] > W or b["y"] + b["h"] > H:
            raise SystemExit(f"{b['id']} outside the {W}x{H} canvas")
    real = [b for b in bs if b["tone"] is not None]
    for i in range(len(real)):
        for j in range(i + 1, len(real)):
            a, c = real[i], real[j]
            if (a["x"] < c["x"] + c["w"] and c["x"] < a["x"] + a["w"]
                    and a["y"] < c["y"] + c["h"] and c["y"] < a["y"] + a["h"]):
                raise SystemExit(f"overlap: {a['id']} / {c['id']}")
    print(f"layout: {len(bs)} boxes ({len(real)} panels), no overlaps, in bounds")

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
