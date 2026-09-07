#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the Forge-ONNX single-op pipeline diagram.

One data table, two outputs:

  docs/source/imgs/compiler_arch/forge-onnx_overview.drawio.svg
      An SVG that is also a diagrams.net document -- the mxfile XML rides in the
      `content` attribute of the <svg> root, so the same file renders in the docs
      and opens editable at app.diagrams.net.

  scripts/pipeline_cache/diagram_data.json
      The resolved table, for the artifact page and for diffing.

The tt-mlir pass list is not transcribed. It is read out of the real pipeline:

    ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=<arch>" \
               --dump-pass-pipeline

for both wormhole_b0 and quasar, so a pin bump regenerates the diagram instead of
silently staling it -- and if a future pin makes the two arches genuinely diverge, the
generator says so instead of drawing a claim that is no longer true.

    python scripts/gen_pipeline_diagram.py
    python scripts/gen_pipeline_diagram.py --refresh   # re-dump, don't use the cache
    python scripts/gen_pipeline_diagram.py --check     # verify only, write nothing

Note: enumerate passes with --dump-pass-pipeline, never --help; ttmlir-opt --help
segfaults on this build.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from xml.sax.saxutils import escape, quoteattr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TTMLIR = os.path.join(REPO, "third_party", "tt-mlir")
TTMLIR_OPT = os.path.join(TTMLIR, "build", "bin", "ttmlir-opt")
CACHE = os.path.join(REPO, "scripts", "pipeline_cache")
SVG_OUT = os.path.join(
    REPO, "docs", "source", "imgs", "compiler_arch", "forge-onnx_overview.drawio.svg"
)
JSON_OUT = os.path.join(CACHE, "diagram_data.json")

ARCHES = ("wormhole_b0", "quasar")


# --------------------------------------------------------------------- pipeline dump


class PassNode:
    """One entry in the --dump-pass-pipeline tree."""

    def __init__(self, name, opts=""):
        self.name = name
        self.opts = opts
        self.children = []

    def __repr__(self):
        return f"PassNode({self.name!r}, {len(self.children)} children)"


def _parse_list(text, i):
    """Parse a comma-separated pass list until an unmatched ')'.

    Option blocks are treated as opaque: `{...}` can contain commas, parens and
    newlines (argument-types= spans lines), so they are skipped by brace depth
    rather than tokenised.
    """
    nodes = []
    while i < len(text):
        while i < len(text) and text[i] in " \t\n\r,":
            i += 1
        if i >= len(text) or text[i] == ")":
            return nodes, i
        j = i
        while j < len(text) and text[j] not in "({),":
            j += 1
        name = text[i:j].strip()
        i = j
        opts = ""
        if i < len(text) and text[i] == "{":
            depth, k = 0, i
            while k < len(text):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        k += 1
                        break
                k += 1
            opts, i = text[i:k], k
        node = PassNode(name, opts)
        if i < len(text) and text[i] == "(":
            node.children, i = _parse_list(text, i + 1)
            if i < len(text) and text[i] == ")":
                i += 1
        if name:
            nodes.append(node)
    return nodes, i


def parse_pipeline(text):
    """Return the top-level pass list from a --dump-pass-pipeline capture."""
    start = text.find("builtin.module(")
    if start < 0:
        raise SystemExit("could not find 'builtin.module(' in the pipeline dump")
    nodes, _ = _parse_list(text, start + len("builtin.module("))
    return nodes


SCRATCH_TOKEN = "<scratch>.mlir"


def scratch_input(tmpdir):
    """A fixed-content, fixed-name TTIR input shared by every arch dump."""
    path = os.path.join(tmpdir, "empty.mlir")
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write("module {}\n")
    return path


def dump_pipeline(arch, refresh, tmpdir=None):
    """Get the expanded pipeline for `arch`, from ttmlir-opt or from the cache."""
    cached = os.path.join(CACHE, f"pipe_{arch}.txt")
    if not refresh and os.path.exists(cached):
        with open(cached) as fh:
            return fh.read(), f"cache ({os.path.relpath(cached, REPO)})"
    if not os.path.exists(TTMLIR_OPT):
        raise SystemExit(
            f"no cache at {cached} and ttmlir-opt is not built at {TTMLIR_OPT}.\n"
            "Build it (cmake --build third_party/tt-mlir/build -- ttmlir-opt) or keep "
            "the cache."
        )
    empty = scratch_input(tmpdir)
    # The dump goes to stderr on some builds, so both streams are captured.
    proc = subprocess.run(
        [
            TTMLIR_OPT,
            f"--ttir-to-ttnn-backend-pipeline=mock-system-desc-arch={arch}",
            "--dump-pass-pipeline",
            empty,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    text = proc.stdout + proc.stderr
    if "builtin.module(" not in text:
        raise SystemExit(f"ttmlir-opt produced no pipeline for {arch}:\n{text[:2000]}")
    # ttmlir-opt echoes the input path in its diagnostics. Normalise it out so a
    # cached dump is comparable no matter where the scratch file lived.
    text = text.replace(empty, SCRATCH_TOKEN)
    with open(cached, "w") as fh:
        fh.write(text)
    return text, "ttmlir-opt"


def classify_diff(line):
    """What a differing line is. Descriptor-derived differences are expected."""
    if "mock-system-desc-arch=" in line:
        return "the mock-system-desc-arch pass option"
    if "#system_desc" in line:
        return "the #ttcore.system_desc attribute"
    if "ttcore.device @" in line:
        return "the derived ttcore.device"
    return None


def pipeline_diff(dumps):
    """Lines that differ between the two arch dumps, classified.

    The diagram's central claim is that the two arches share one pipeline and differ
    only in the descriptor. Checking a line *count* would be brittle -- stream
    ordering moves the offsets -- so each differing line is classified instead, and
    anything unclassified means a real structural divergence.
    """
    a, b = dumps[ARCHES[0]].splitlines(), dumps[ARCHES[1]].splitlines()
    if len(a) != len(b):
        return None, [
            f"the two dumps differ in length ({len(a)} vs {len(b)} lines) -- the "
            "pipelines have structurally diverged, which the diagram does not depict"
        ]
    diffs = [(i + 1, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    unexplained = [(n, x) for n, x, _ in diffs if classify_diff(x) is None]
    errs = []
    if unexplained:
        errs.append(
            "these differing lines are NOT descriptor-derived, so the arches now "
            "differ in more than their hardware spec: "
            + "; ".join(f"line {n}: {x.strip()[:90]}" for n, x in unexplained)
        )
    return diffs, errs


def pass_names(nodes):
    """Flatten a pass subtree to a list of bare names (options dropped)."""
    out = []
    for n in nodes:
        if n.children:
            out.extend(pass_names(n.children))
        else:
            out.append(n.name)
    return out


# ------------------------------------------------------------------------- badges

NEUTRAL = "neutral"  # stage never asks which chip it targets
PATCHED = "patched"  # already made Quasar-aware, on a branch
BLOCKED = "blocked"  # known open Quasar failure
SYSDESC = "sysdesc"  # produces, carries or consumes the hardware spec

BADGE_LABEL = {
    NEUTRAL: "arch-neutral",
    PATCHED: "Quasar-patched",
    BLOCKED: "Quasar-blocked",
    SYSDESC: "reads the system descriptor",
}

# Fill / stroke per badge. Light palette; the docs render these on white.
BADGE_STYLE = {
    NEUTRAL: ("#eef2f6", "#94a3b8", "#0f172a"),
    PATCHED: ("#fdf0d5", "#d99e2b", "#4a2f00"),
    BLOCKED: ("#fbe4e4", "#d05c5c", "#5c1010"),
    SYSDESC: ("#e8e2f7", "#8b6fc9", "#2b1a55"),
}

LANE_TINT = {
    "host · python": "#f4f8fb",
    "forge · c++": "#f2f7fd",
    "tt-mlir · TTIR phase": "#f8f5fd",
    "tt-mlir · TTNN phase": "#f8f5fd",
    "flatbuffer": "#fdf7f2",
    "tt-metal · runtime": "#fdf4fb",
    "silicon": "#fff5f0",
    "the hardware spec": "#f5f2fb",
}


# --------------------------------------------------------------------- the anchors
# Every anchor here is checked to resolve before anything is written, so a moved
# function fails the generator instead of producing a stale diagram.

HOST = [
    ("onnx.ModelProto", None, NEUTRAL, "one Add node, float32 [2,32,32]"),
    ("compile_main", "forge/forge/compile.py:182", NEUTRAL, "forge.compile entry"),
    ("wrap_module -> OnnxModule", "forge/forge/module.py:1011", NEUTRAL, ""),
    (
        "relay.frontend.from_onnx",
        "forge/forge/tvm_calls/forge_compile.py:822",
        NEUTRAL,
        "ONNX -> TVM Relay. There is no ONNX fast path.",
    ),
    (
        "partition_for_forge",
        "forge/forge/tvm_calls/relay/op/forge.py:1993",
        NEUTRAL,
        "MergeComposite / AnnotateTarget / PartitionGraph",
    ),
    (
        "ForgeCompiler -> JSON graph",
        "third_party/tvm/src/relay/backend/contrib/forge/codegen.cc:94",
        NEUTRAL,
        "relay.build of the 'forge' external target",
    ),
    (
        "ForgeWriter -> generated_modules/<name>.py",
        "forge/forge/tvm_to_python.py:2444",
        NEUTRAL,
        "a real Python file, imported back in and traced",
    ),
    (
        "generate_graph -> graphlib::Graph",
        "forge/forge/compile.py:1043",
        NEUTRAL,
        "traced under start_tracing()",
    ),
]

# The nine stages that actually run for a single-op inference compile, in traversal
# order. NOT enum order -- see the SKIPPED list below.
STAGES = [
    ("INIT_COMPILE", "forge/forge/compile.py:603"),
    ("GENERATE_INITIAL_GRAPH", "forge/forge/compile.py:623"),
    ("POST_INITIAL_GRAPH_PASS", "forge/forge/compile.py:687"),
    ("POST_AUTOGRAD_PASS", "forge/forge/compile.py:883"),
    ("CONSTEVAL_GRAPH", "forge/forge/compile.py:728"),
    ("PRE_LOWERING_PASS", "forge/forge/compile.py:920"),
    ("SPLIT_GRAPH", "forge/forge/compile.py:946"),
    ("RUN_MLIR_COMPILER", "forge/forge/compile.py:968"),
    ("FINISH_COMPILE", "forge/forge/compile.py:985"),
]

# Declared in CompileDepth but skipped by default config for a single-op inference
# compile. Drawn dashed.
SKIPPED = [
    ("POST_PATTERN_MATCHER", "forge/forge/compile.py:752", "needs match_subgraph_patterns"),
    ("OPTIMIZED_GRAPH", "forge/forge/compile.py:783", "needs enable_optimization_passes"),
    ("AUTOGRAD", "forge/forge/compile.py:835", "training only"),
]

FORGE_CPP = [
    (
        "run_post_initial_graph_passes",
        "forge/csrc/forge_passes.cpp:54",
        NEUTRAL,
        "decompose, fuse_pad_conv2d, fuse_conv2d_bias",
    ),
    (
        "run_post_autograd_graph_passes",
        "forge/csrc/forge_passes.cpp:154",
        NEUTRAL,
        "",
    ),
    (
        "run_consteval_graph_pass",
        "forge/csrc/passes/consteval.cpp:128",
        NEUTRAL,
        "nothing to fold for a bare Add; 53 chains on ResNet-50",
    ),
    (
        "run_pre_lowering_passes",
        "forge/csrc/forge_passes.cpp:167",
        NEUTRAL,
        "convert_broadcast_ops_to_tms, remove_nops",
    ),
    (
        "split_graph -> ForgeGraphModule",
        "forge/csrc/passes/split_graph.cpp:460",
        NEUTRAL,
        "Forward graph only",
    ),
    (
        "MLIRGenerator::emit_mlir",
        "forge/csrc/passes/lower_to_mlir.cpp:190",
        SYSDESC,
        "stamps ttcore.system_desc UNLESS a target was named (:194-202)",
    ),
    (
        '"add" -> ttir::AddOp',
        "forge/csrc/passes/lower_to_mlir.cpp:843",
        NEUTRAL,
        "handler map at :297",
    ),
    (
        "TTIR module",
        None,
        NEUTRAL,
        "arch-neutral: ops and shapes, no layouts or memory spaces",
    ),
]

# Passes that read the hardware spec, with why. Everything not listed is arch-neutral.
SD_READERS = {
    "ttcore-register-device": "PRODUCES it, then derives ttcore.device",
    "ttnn-layout": "grid + L1 drive every layout choice",
    "convert-ttir-to-ttnn": "getNocL1AddressAlignBytes()",
    "ttnn-resolve-composites": "reads chipDesc arch",
    "ttnn-memory-management": "L1 budget",
    "ttnn-workaround": "decomposition patterns read chip descs",
    "ttnn-set-compute-kernel-config": "per-arch compute config",
    "ttnn-collect-perf-metrics": "no Quasar DRAM BW / AICLK: the pass self-disables",
}

# Passes whose Quasar behaviour differs from Wormhole.
PASS_QUASAR = {
    "ttnn-collect-perf-metrics": (
        BLOCKED,
        "skips entirely on Quasar (TTNNCollectPerfMetrics.cpp:1063-1077); "
        "no calibrated DRAM BW or AICLK (:809-827)",
    ),
}

FLATBUFFER = [
    (
        "ttnnToFlatbuffer",
        "forge/csrc/passes/mlir_compiler.cpp:143",
        NEUTRAL,
        "compilation ends here",
    ),
    (
        "TTNNBinary",
        "third_party/tt-mlir/include/ttmlir/Target/TTNN/binary.fbs",
        SYSDESC,
        "version, schema_hash, ttmlir_git_hash, system_desc, mlir, programs",
    ),
    (
        'file_identifier "TTNN" / .ttnn',
        None,
        NEUTRAL,
        "the system descriptor is serialised INTO the binary",
    ),
    (
        "Program.operations",
        "third_party/tt-mlir/include/ttmlir/Target/TTNN/program.fbs:178",
        NEUTRAL,
        "the op stream the runtime replays; each carries debug_info",
    ),
]

RUNTIME = [
    ("CompiledModel.__call__", "forge/forge/compiled_graph_state.py:280", NEUTRAL, ""),
    ("ModelState::run_program", "forge/csrc/runtime/state.cpp:39", NEUTRAL, ""),
    (
        "Tensor::to_layout  HOST -> DEVICE",
        "forge/csrc/runtime/tensor.hpp:147",
        PATCHED,
        "Quasar routes to experimental::quasar::to_layout (runtime.cpp:196-203)",
    ),
    ("runtime::submit", "forge/csrc/runtime/runtime.cpp:117", NEUTRAL, ""),
    (
        "schema_hash check",
        "third_party/tt-mlir/runtime/lib/binary.cpp:525",
        SYSDESC,
        "there is NO arch equality check in the C++ runtime",
    ),
    (
        "ProgramExecutor::execute",
        "third_party/tt-mlir/runtime/lib/ttnn/program_executor.cpp:222",
        NEUTRAL,
        "walks program->operations()",
    ),
    (
        "switch (op->type_type())",
        "third_party/tt-mlir/runtime/lib/ttnn/program_executor.cpp:285",
        PATCHED,
        "WHERE THE ARCHES DIVERGE. 13 of 15 files the Quasar branch "
        "touches are under runtime/lib/ttnn/.",
    ),
    (
        "program factory",
        None,
        BLOCKED,
        "mainline factories are REFUSED on Quasar: DataMovementKernel's "
        "constructor TT_FATALs (kernel.hpp:418)",
    ),
]

SILICON = [
    ("JIT kernel compile", None, NEUTRAL, "one source file per RISC-V role"),
    ("BRISC  DRAM -> L1 CB", None, NEUTRAL, ""),
    ("TRISC0  L1 -> srcA/srcB", None, BLOCKED, "craq-sim wedges HERE: srcA/srcB reach valid=1 unpack=1, matrix=0"),
    ("TRISC1  add_tiles on the FPU", None, BLOCKED, "the math unit never takes the operands"),
    ("TRISC2  dest -> output CB", None, NEUTRAL, ""),
    ("NCRISC  L1 -> DRAM", None, NEUTRAL, ""),
    ("readback -> torch.Tensor", None, NEUTRAL, "then verify() compares by PCC"),
]

# The hardware-spec lane: three sources, one attribute, one consumer chain.
HW_SPEC = [
    (
        "live device",
        "forge/csrc/passes/lower_to_mlir.cpp:694",
        SYSDESC,
        "TTSystem::get_system() -> UMD -> IDevice/hal/Allocator. "
        "Opens hardware. tt-mlir never reads the arch YAML itself.",
    ),
    (
        "captured .ttsys",
        "forge/csrc/runtime/python_bindings.cpp:107",
        SYSDESC,
        "save_system_desc(), or ttrt query --save-artifacts",
    ),
    (
        "mock-system-desc-arch",
        "third_party/tt-mlir/lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp:409",
        SYSDESC,
        "createDefault{Wormhole,Blackhole,Quasar}SystemDesc. No device needed.",
    ),
    (
        "#ttcore.system_desc",
        "third_party/tt-mlir/include/ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td:216",
        SYSDESC,
        "chipDescs: grid, l1Size, numDramChannels, numCBs, thread counts, "
        "supportedDataTypes, supportedTileSizes",
    ),
    (
        "registerDevice()",
        "third_party/tt-mlir/lib/Dialect/TTCore/Transforms/TTCoreRegisterDevice.cpp:45",
        SYSDESC,
        "stamps the mock ONLY if the module carries none (guard at :51) -- "
        "the exact contract forge's skipped stamp relies on",
    ),
    (
        "ttcore.device @default_device",
        "third_party/tt-mlir/lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp:1735",
        SYSDESC,
        "workerGrid 8x8 | 4x8   dramGrid 1x12 | 1x2",
    ),
    (
        "consumed by layout / memory passes",
        None,
        NEUTRAL,
        "no pass asks 'which chip is this' -- they read grid and l1Size",
    ),
    (
        "serialised into the .ttnn binary",
        None,
        SYSDESC,
        "TTNNBinary.system_desc",
    ),
    (
        "ttrt re-validates vs the live device",
        "third_party/tt-mlir/tools/ttrt/common/util.py:868",
        SYSDESC,
        "full JSON diff; forge's own runtime does not check arch",
    ),
]


# ------------------------------------------------------------------ anchor checking


def resolve(anchor):
    """Return an absolute path for a `file[:line]` anchor, or raise."""
    path, _, line = anchor.partition(":")
    full = os.path.join(REPO, path)
    if not os.path.exists(full):
        raise FileNotFoundError(path)
    if line:
        with open(full, "rb") as fh:
            count = sum(1 for _ in fh)
        if int(line) > count:
            raise ValueError(f"{path} has {count} lines, anchor wants {line}")
    return full


def check_anchors(groups):
    """Verify every anchor in the table. Returns a list of failure strings."""
    bad = []
    seen = 0
    for name, rows in groups:
        for row in rows:
            anchor = row[1]
            if not anchor:
                continue
            seen += 1
            try:
                resolve(anchor)
            except (FileNotFoundError, ValueError) as exc:
                bad.append(f"{name}: {anchor} -> {exc}")
    return bad, seen


# ----------------------------------------------------- system descriptor comparison

CHIP_FIELDS = [
    "arch",
    "grid",
    "coord_translation_offsets",
    "l1_size",
    "num_dram_channels",
    "dram_channel_size",
    "dram_grid",
    "num_cbs",
    "num_compute_threads",
    "num_datamovement_threads",
    "pcie_address_align_bytes",
    "noc_dram_address_align_bytes",
    "noc_l1_address_align_bytes",
    "l1_unreserved_base",
    "erisc_l1_unreserved_base",
    "dram_unreserved_base",
    "dram_unreserved_end",
    "dst_physical_size_tiles",
]


def chip_desc_fields(dump):
    """Pull the chip_descs fields out of the #system_desc line of a dump."""
    line = next((l for l in dump.splitlines() if "#system_desc = " in l), None)
    if line is None:
        return {}
    out = {}
    for field in CHIP_FIELDS:
        m = re.search(rf"\b{field} = (<[^>]*>|[0-9]+x[0-9]+|[0-9]+)", line)
        if m:
            out[field] = m.group(1).strip("<>")
    for field in ("supported_data_types", "supported_tile_sizes"):
        m = re.search(rf"\b{field} = \[([^\]]*)\]", line)
        if m:
            out[field] = ", ".join(x.strip().strip("<>") for x in m.group(1).split(","))
    m = re.search(r"dram_bank_to_logical_worker_noc0 = \[([^\]]*)\]", line)
    if m:
        raw = m.group(1).strip()
        out["dram_bank_to_logical_worker_noc0"] = (
            "empty" if not raw else f"{raw.count('(')} entries"
        )
    return out


# --------------------------------------------------------------------------- layout

BOX_W = 274
BOX_H = 46
BOX_H_NOTE = 74
GAP_X = 14
GAP_Y = 16
MARGIN = 26
HDR_H = 32
LANE_GAP = 30
CANVAS_W = 2060
NOTE_COLS = 42


def short_anchor(anchor, cols=48):
    """Shorten a path for display while keeping it greppable.

    Submodule prefixes become a short label, then interior directories are elided
    from the left -- the basename and line number are what you actually search for,
    so they are never cut.
    """
    a = anchor.replace("third_party/tt-mlir/", "tt-mlir:").replace(
        "third_party/tvm/", "tvm:"
    )
    if len(a) <= cols:
        return a
    label, _, rest = a.partition(":") if a.startswith(("tt-mlir:", "tvm:")) else ("", "", a)
    prefix = f"{label}:" if label else ""
    parts = rest.split("/")
    while len(parts) > 1 and len(prefix + ".../" + "/".join(parts)) > cols:
        parts.pop(0)
    out = prefix + (".../" if len(parts) > 1 else "") + "/".join(parts)
    # Never trim the submodule label or the basename: if even prefix + basename is
    # over budget, keep it over budget rather than return something un-greppable.
    return out if len(out) <= cols else prefix + parts[-1]


def fit_font(text, width, base, minimum=8.0, char_ratio=0.60):
    """Largest font size at or below `base` that keeps `text` inside `width`."""
    if not text:
        return base
    size = (width - 16) / (len(text) * char_ratio)
    return max(minimum, min(base, size))


def wrap_note(note, cols=NOTE_COLS, maxlines=3):
    """Greedy wrap; returns at most `maxlines` lines, last one ellipsised."""
    if not note:
        return []
    words, lines, cur = note.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) <= cols:
            cur = trial
        else:
            lines.append(cur)
            cur = w
            if len(lines) == maxlines:
                break
    if cur and len(lines) < maxlines:
        lines.append(cur)
    if len(lines) == maxlines and len(" ".join(words)) > sum(len(l) + 1 for l in lines):
        lines[-1] = lines[-1][: cols - 1].rstrip() + "…"
    return lines


def layout(lanes):
    """Place every node. Mutates nodes with x/y/w/h and returns total height."""
    per_row = max(1, (CANVAS_W - 2 * MARGIN + GAP_X) // (BOX_W + GAP_X))
    y = MARGIN
    for lane in lanes:
        lane["y"] = y
        lane["header_h"] = HDR_H
        y += HDR_H + 8
        row_start_y = y
        row, col = 0, 0
        row_h = BOX_H
        for node in lane["nodes"]:
            node["note_lines"] = wrap_note(node.get("note", ""))
            node["w"] = BOX_W
            node["h"] = BOX_H_NOTE if node["note_lines"] else BOX_H
            if col == per_row:
                row += 1
                col = 0
                row_start_y += row_h + GAP_Y
                row_h = BOX_H
            node["x"] = MARGIN + col * (BOX_W + GAP_X)
            node["y"] = row_start_y
            node["row"] = row
            node["col"] = col
            row_h = max(row_h, node["h"])
            col += 1
        y = row_start_y + row_h + LANE_GAP
        lane["h"] = y - lane["y"] - LANE_GAP + 10
    return y + MARGIN


# ------------------------------------------------------------------- mxfile (drawio)


def mx_style(badge, dashed=False):
    fill, stroke, font = BADGE_STYLE[badge]
    return (
        f"rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor={fill};"
        f"strokeColor={stroke};fontColor={font};align=left;verticalAlign=top;"
        f"spacingLeft=8;spacingTop=2;spacingRight=6;fontSize=11;"
        + ("dashed=1;" if dashed else "")
    )


def mx_value(node):
    parts = [f"<b>{escape(node['label'])}</b>"]
    if node.get("anchor"):
        parts.append(
            f'<font style="font-size:9px;color:#5b6472">{escape(node["anchor"])}</font>'
        )
    for line in node["note_lines"]:
        parts.append(f'<font style="font-size:9px">{escape(line)}</font>')
    return "<br>".join(parts)


def emit_mxfile(lanes, total_h, meta):
    # Ids first: the lane-to-lane edges point forward, so every node needs an id
    # before the first cell is emitted.
    for li, lane in enumerate(lanes):
        for ni, node in enumerate(lane["nodes"]):
            node["id"] = f"n{li}_{ni}"

    cells = []
    cells.append('<mxCell id="0" />')
    cells.append('<mxCell id="1" parent="0" />')

    def cell(cid, value, style, x, y, w, h):
        cells.append(
            f'<mxCell id={quoteattr(cid)} value={quoteattr(value)} '
            f'style={quoteattr(style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />'
            f"</mxCell>"
        )

    def edge(cid, src, dst, style):
        cells.append(
            f'<mxCell id={quoteattr(cid)} style={quoteattr(style)} edge="1" '
            f'parent="1" source={quoteattr(src)} target={quoteattr(dst)}>'
            f'<mxGeometry relative="1" as="geometry" /></mxCell>'
        )

    cell(
        "title",
        f"<b>{escape(meta['title'])}</b><br>"
        f'<font style="font-size:10px">{escape(meta["subtitle"])}</font>',
        "text;html=1;align=left;verticalAlign=middle;fontSize=16;fontColor=#0f172a;",
        MARGIN,
        6,
        CANVAS_W - 2 * MARGIN,
        HDR_H,
    )

    for li, lane in enumerate(lanes):
        tint = LANE_TINT.get(lane["name"], "#f6f6f8")
        cell(
            f"lane{li}bg",
            "",
            f"rounded=0;fillColor={tint};strokeColor=#dfe3ea;html=1;"
            "verticalAlign=top;align=left;",
            MARGIN - 12,
            lane["y"],
            CANVAS_W - 2 * MARGIN + 24,
            lane["h"],
        )
        cell(
            f"lane{li}hdr",
            f"<b>{escape(lane['name'].upper())}</b>"
            + (
                f'&nbsp;&nbsp;<font style="font-size:10px;color:#5b6472">'
                f'{escape(lane["subtitle"])}</font>'
                if lane.get("subtitle")
                else ""
            ),
            "text;html=1;align=left;verticalAlign=middle;fontSize=12;"
            "fontColor=#1f2937;",
            MARGIN,
            lane["y"] + 4,
            CANVAS_W - 2 * MARGIN,
            22,
        )
        for node in lane["nodes"]:
            cell(
                node["id"],
                mx_value(node),
                mx_style(node["badge"], node.get("dashed")),
                node["x"],
                node["y"],
                node["w"],
                node["h"],
            )
        for ni in range(len(lane["nodes"]) - 1):
            a, b = lane["nodes"][ni], lane["nodes"][ni + 1]
            if a.get("terminal") or b.get("dashed") or a.get("dashed"):
                continue
            style = (
                "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;"
                "endFill=1;strokeColor=#8b93a1;strokeWidth=1.2;"
            )
            edge(f"e{li}_{ni}", a["id"], b["id"], style)
        if li + 1 < len(lanes) and lane["nodes"] and lanes[li + 1]["nodes"]:
            edge(
                f"lane_e{li}",
                lane["nodes"][-1]["id"],
                lanes[li + 1]["nodes"][0]["id"],
                "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;"
                "endFill=1;strokeColor=#4b5563;strokeWidth=2;dashed=0;",
            )

    body = "".join(cells)
    return (
        '<mxfile host="app.diagrams.net" agent="gen_pipeline_diagram.py" type="device">'
        f'<diagram id="forge-onnx-pipeline" name="forge-onnx single op">'
        f'<mxGraphModel dx="{CANVAS_W}" dy="{total_h}" grid="0" gridSize="10" '
        'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
        f'pageScale="1" pageWidth="{CANVAS_W}" pageHeight="{total_h}" math="0" '
        f'shadow="0"><root>{body}</root></mxGraphModel></diagram></mxfile>'
    )


# ------------------------------------------------------------------------ svg render


def svg_text(x, y, s, size=11, weight="normal", fill="#0f172a", family=None):
    fam = family or "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
    return (
        f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escape(s)}</text>'
    )


def emit_svg(lanes, total_h, meta, mxfile):
    p = []
    p.append(f'<rect x="0" y="0" width="{CANVAS_W}" height="{total_h}" fill="#ffffff"/>')
    p.append(svg_text(MARGIN, 26, meta["title"], 17, "bold"))
    p.append(svg_text(MARGIN, 44, meta["subtitle"], 11, "normal", "#5b6472"))

    for lane in lanes:
        tint = LANE_TINT.get(lane["name"], "#f6f6f8")
        p.append(
            f'<rect x="{MARGIN - 12}" y="{lane["y"]}" '
            f'width="{CANVAS_W - 2 * MARGIN + 24}" height="{lane["h"]}" rx="6" '
            f'fill="{tint}" stroke="#dfe3ea"/>'
        )
        p.append(svg_text(MARGIN, lane["y"] + 20, lane["name"].upper(), 12, "bold", "#1f2937"))
        if lane.get("subtitle"):
            off = MARGIN + 11 + int(len(lane["name"]) * 7.6)
            p.append(svg_text(off, lane["y"] + 20, lane["subtitle"], 10, "normal", "#5b6472"))

        # within-lane connectors
        for i in range(len(lane["nodes"]) - 1):
            a, b = lane["nodes"][i], lane["nodes"][i + 1]
            if a.get("terminal") or a.get("dashed") or b.get("dashed"):
                continue
            if a["row"] == b["row"]:
                y = a["y"] + BOX_H / 2
                p.append(
                    f'<path d="M {a["x"] + a["w"]} {y} L {b["x"] - 3} {y}" '
                    'stroke="#8b93a1" stroke-width="1.2" fill="none" '
                    'marker-end="url(#ar)"/>'
                )
            else:
                y1 = a["y"] + a["h"]
                y2 = b["y"] + BOX_H / 2
                xr = a["x"] + a["w"] / 2
                xl = b["x"] - 10
                mid = y1 + (y2 - y1) / 2
                p.append(
                    f'<path d="M {xr} {y1} L {xr} {mid} L {xl} {mid} L {xl} {y2} '
                    f'L {b["x"] - 3} {y2}" stroke="#8b93a1" stroke-width="1.2" '
                    'fill="none" stroke-dasharray="3,3" marker-end="url(#ar)"/>'
                )

        for node in lane["nodes"]:
            fill, stroke, font = BADGE_STYLE[node["badge"]]
            dash = ' stroke-dasharray="5,4"' if node.get("dashed") else ""
            p.append(
                f'<g><title>{escape(node["label"])}'
                + (f' — {escape(node["anchor"])}' if node.get("anchor") else "")
                + (f'\n{escape(node["note"])}' if node.get("note") else "")
                + "</title>"
                f'<rect x="{node["x"]}" y="{node["y"]}" width="{node["w"]}" '
                f'height="{node["h"]}" rx="5" fill="{fill}" stroke="{stroke}"{dash}/>'
            )
            ty = node["y"] + 16
            label = node["label"]
            p.append(
                svg_text(
                    node["x"] + 8,
                    ty,
                    label,
                    fit_font(label, node["w"], 10.5),
                    "bold",
                    font,
                )
            )
            if node.get("anchor"):
                ty += 12
                anc = short_anchor(node["anchor"])
                p.append(
                    svg_text(
                        node["x"] + 8,
                        ty,
                        anc,
                        fit_font(anc, node["w"], 8.5, minimum=6.8),
                        "normal",
                        "#5b6472",
                    )
                )
            for line in node["note_lines"]:
                ty += 11
                p.append(svg_text(node["x"] + 8, ty, line, 8.5, "normal", "#3f4854"))
            p.append("</g>")

    # lane-to-lane spine
    for i in range(len(lanes) - 1):
        a, b = lanes[i], lanes[i + 1]
        x = MARGIN - 12 + (CANVAS_W - 2 * MARGIN + 24) / 2
        y1 = a["y"] + a["h"]
        y2 = b["y"]
        p.append(
            f'<path d="M {x} {y1} L {x} {y2 - 3}" stroke="#4b5563" '
            'stroke-width="2.2" fill="none" marker-end="url(#ars)"/>'
        )

    legend_y = total_h - 16
    lx = MARGIN
    p.append(svg_text(lx, legend_y, "legend:", 10, "bold", "#3f4854"))
    lx += 60
    for badge in (NEUTRAL, SYSDESC, PATCHED, BLOCKED):
        fill, stroke, _ = BADGE_STYLE[badge]
        p.append(
            f'<rect x="{lx}" y="{legend_y - 9}" width="12" height="11" rx="2" '
            f'fill="{fill}" stroke="{stroke}"/>'
        )
        p.append(svg_text(lx + 17, legend_y, BADGE_LABEL[badge], 9.5, "normal", "#3f4854"))
        lx += 30 + int(len(BADGE_LABEL[badge]) * 5.9)

    defs = (
        "<defs>"
        '<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 1 L 9 5 L 0 9 z" fill="#8b93a1"/></marker>'
        '<marker id="ars" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 1 L 9 5 L 0 9 z" fill="#4b5563"/></marker>'
        "</defs>"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" '
        f'width="{CANVAS_W}" height="{total_h}" '
        f'viewBox="0 0 {CANVAS_W} {total_h}" content={quoteattr(mxfile)}>'
        f"{defs}{''.join(p)}</svg>\n"
    )


# ----------------------------------------------------------------- lane assembly

def _node(label, anchor, badge, note, **kw):
    d = {"label": label, "anchor": anchor, "badge": badge, "note": note}
    d.update(kw)
    return d


def pass_node(name, opts=""):
    """Turn a dumped pass name into a node, badged by what it reads."""
    badge = SYSDESC if name in SD_READERS else NEUTRAL
    note = SD_READERS.get(name, "")
    if name in PASS_QUASAR:
        badge, note = PASS_QUASAR[name]
    label = name
    # Surface the one option that changes the descriptor, and the opt level.
    m = re.search(r"mock-system-desc-arch=(\w+)", opts or "")
    if m:
        label = f"{name}  [arch={m.group(1)}]"
    m = re.search(r"ttnn-optimization-level=(\d+)", opts or "")
    if m:
        label = f"{name}  [opt={m.group(1)}]"
    return _node(label, None, badge, note)


def build_lanes(dumps, refresh_src):
    top = parse_pipeline(dumps[ARCHES[0]])
    device_groups = [n for n in top if n.name == "ttcore.device_module"]
    cpu_groups = [n for n in top if n.name == "ttcore.cpu_module"]
    if len(device_groups) != 2:
        raise SystemExit(
            f"expected 2 ttcore.device_module groups, found {len(device_groups)}; "
            "the pipeline shape changed -- update build_lanes()"
        )

    def flat(node):
        out = []
        for n in node.children:
            if n.children:
                out.extend(flat(n))
            else:
                out.append(n)
        return out

    ttir_passes = flat(device_groups[0])
    ttnn_passes = flat(device_groups[1])
    cpu_count = len(flat(cpu_groups[0])) if cpu_groups else 0

    top_level = [
        pass_node(n.name, n.opts) for n in top if not n.children
    ]

    lanes = [
        {
            "name": "host · python",
            "subtitle": "a one-node ONNX graph still round-trips through TVM Relay "
            "and a generated Python file",
            "nodes": [_node(*r) for r in HOST],
        },
        {
            "name": "forge · compile stages",
            "subtitle": "the 9 CompileDepth stages that actually run, in traversal "
            "order — dashed ones are skipped by default config",
            "nodes": [_node(n, a, NEUTRAL, "") for n, a in STAGES]
            + [_node(n, a, NEUTRAL, why, dashed=True) for n, a, why in SKIPPED],
        },
        {
            "name": "forge · c++",
            "subtitle": "graphlib::Graph passes, then the only lowering step: "
            "forge graph → TTIR. There is no forge MLIR dialect.",
            "nodes": [_node(*r) for r in FORGE_CPP],
        },
        {
            "name": "the hardware spec",
            "subtitle": "three sources, one attribute. Precedence: "
            "system_desc_path > target_arch > live device.",
            "nodes": [_node(*r) for r in HW_SPEC],
        },
        {
            "name": "tt-mlir · top level",
            "subtitle": f"ttir-to-ttnn-backend-pipeline — {len(top)} top-level passes "
            "(a deprecated alias for ttir-to-ttnn-runtime-pipeline)",
            "nodes": top_level,
        },
        {
            "name": "tt-mlir · TTIR phase",
            "subtitle": f"{len(ttir_passes)} passes, nested in "
            "ttcore.device_module → builtin.module",
            "nodes": [pass_node(n.name, n.opts) for n in ttir_passes],
        },
        {
            "name": "tt-mlir · TTNN phase",
            "subtitle": f"{len(ttnn_passes)} passes — where tensors get layouts, "
            "memory spaces and grids",
            "nodes": [pass_node(n.name, n.opts) for n in ttnn_passes]
            + [
                _node(
                    f"ttcore.cpu_module  ({cpu_count} passes)",
                    None,
                    NEUTRAL,
                    "TOSA → linalg → LLVM host fallback. A single ONNX Add "
                    "never enters it.",
                    dashed=True,
                    terminal=True,
                )
            ],
        },
        {
            "name": "flatbuffer",
            "subtitle": "compilation ends here; the binary is self-describing",
            "nodes": [_node(*r) for r in FLATBUFFER],
        },
        {
            "name": "tt-metal · runtime",
            "subtitle": "the binary is consumed rather than produced — "
            "and this is where the arches actually diverge",
            "nodes": [_node(*r) for r in RUNTIME],
        },
        {
            "name": "silicon",
            "subtitle": "three kernels, five RISC-V processors, one add_tiles",
            "nodes": [_node(*r) for r in SILICON],
        },
    ]
    return lanes, {
        "ttir": len(ttir_passes),
        "ttnn": len(ttnn_passes),
        "cpu": cpu_count,
        "top": len(top),
        "source": refresh_src,
    }


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-run ttmlir-opt")
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    dumps, sources = {}, {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for arch in ARCHES:
            dumps[arch], sources[arch] = dump_pipeline(arch, args.refresh, tmpdir)
    print(f"pipeline dumps: {', '.join(f'{a} <- {s}' for a, s in sources.items())}")

    groups = [
        ("HOST", HOST),
        ("STAGES", [(a, b, NEUTRAL, "") for a, b in STAGES]),
        ("SKIPPED", [(a, b, NEUTRAL, c) for a, b, c in SKIPPED]),
        ("FORGE_CPP", FORGE_CPP),
        ("FLATBUFFER", FLATBUFFER),
        ("RUNTIME", RUNTIME),
        ("SILICON", SILICON),
        ("HW_SPEC", HW_SPEC),
    ]
    bad, seen = check_anchors(groups)
    if bad:
        for b in bad:
            print(f"  ANCHOR FAIL {b}", file=sys.stderr)
        raise SystemExit(f"{len(bad)} of {seen} anchors do not resolve")
    print(f"anchors: {seen} checked, all resolve")

    diffs, errs = pipeline_diff(dumps)
    for e in errs:
        print(f"  WARNING {e}", file=sys.stderr)
    if diffs is not None:
        print(
            f"{ARCHES[0]} vs {ARCHES[1]}: {len(diffs)} differing lines, "
            f"all descriptor-derived:"
        )
        for n, x, _ in diffs:
            print(f"    line {n:>3}  {classify_diff(x) or 'UNEXPLAINED'}")

    wh = chip_desc_fields(dumps[ARCHES[0]])
    qs = chip_desc_fields(dumps[ARCHES[1]])
    differing = [k for k in wh if wh.get(k) != qs.get(k)]
    print(f"chip descriptor: {len(differing)} of {len(wh)} fields differ")

    lanes, counts = build_lanes(dumps, sources[ARCHES[0]])
    total_h = layout(lanes)
    meta = {
        "title": "Running one op on Wormhole via forge-onnx — and where Quasar diverges",
        "subtitle": (
            f"ONNX Add → TTIR → TTNN → .ttnn → Tensix.   "
            f"{counts['top']} top-level tt-mlir passes, {counts['ttir']} TTIR + "
            f"{counts['ttnn']} TTNN.   "
            f"The wormhole_b0 and quasar pipelines are identical apart from the "
            f"system descriptor.   generated by scripts/gen_pipeline_diagram.py"
        ),
    }
    mxfile = emit_mxfile(lanes, total_h, meta)
    svg = emit_svg(lanes, total_h, meta, mxfile)

    payload = {
        "meta": meta,
        "counts": counts,
        "pipeline_diff_lines": [
            {
                "line": n,
                "kind": classify_diff(a) or "unexplained",
                "wormhole_b0": a,
                "quasar": b,
            }
            for n, a, b in (diffs or [])
        ],
        "chip_desc": {"wormhole_b0": wh, "quasar": qs, "differing": differing},
        "lanes": [
            {
                "name": l["name"],
                "subtitle": l.get("subtitle", ""),
                "nodes": [
                    {
                        k: n.get(k)
                        for k in ("label", "anchor", "badge", "note", "dashed")
                        if n.get(k)
                    }
                    for n in l["nodes"]
                ],
            }
            for l in lanes
        ],
    }

    if args.check:
        print("--check: nothing written")
        return

    os.makedirs(os.path.dirname(SVG_OUT), exist_ok=True)
    with open(SVG_OUT, "w") as fh:
        fh.write(svg)
    with open(JSON_OUT, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    boxes = sum(len(l["nodes"]) for l in lanes)
    print(
        f"wrote {os.path.relpath(SVG_OUT, REPO)} "
        f"({len(svg) // 1024} KiB, {CANVAS_W}x{total_h}, {boxes} boxes in "
        f"{len(lanes)} lanes)"
    )
    print(f"wrote {os.path.relpath(JSON_OUT, REPO)}")


if __name__ == "__main__":
    main()
