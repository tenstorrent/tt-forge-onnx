#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Interactive walkthrough of a single ONNX Add through tt-forge-onnx.

Press Enter to advance. The left pane is the code path -- which file, which function,
what it does. The right pane is the flatbuffer being assembled: what this stage
contributes to the binary that eventually runs on hardware.

Everything on the right is read out of a real compile performed at startup. Nothing
is mocked. The compile names a target architecture, so it needs no device and runs
anywhere.

    python scripts/add_op_walkthrough.py
    python scripts/add_op_walkthrough.py --arch quasar
    python scripts/add_op_walkthrough.py --no-pager      # print everything and exit

Keys:  Enter next   b back   q quit   <n> jump to stage n
"""

import argparse
import json
import os
import shutil
import sys
import textwrap

# --------------------------------------------------------------------------- ANSI

_TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code):
    return (lambda s: f"\033[{code}m{s}\033[0m") if _TTY else (lambda s: s)


BOLD = _c("1")
DIM = _c("2")
CODE_C = _c("38;5;75")  # left pane accent  - blue
FB_C = _c("38;5;209")  # right pane accent - amber
NUM = _c("38;5;150")
WARN = _c("38;5;203")
RULE = _c("38;5;240")

LAYER_COLORS = {
    "host · python": _c("38;5;110"),
    "forge · c++": _c("38;5;75"),
    "tt-mlir": _c("38;5;140"),
    "tt-metal": _c("38;5;176"),
    "silicon": _c("38;5;209"),
}

SHAPE = [2, 32, 32]


# --------------------------------------------------------------------------- capture


def run_compile(arch_name):
    """Compile a single ONNX Add for `arch_name` and return everything we can observe."""
    import torch
    from onnx import TensorProto, helper

    import forge
    from forge.config import CompilerConfig, MLIRConfig

    arch = {
        "wormhole_b0": forge._C.Arch.WORMHOLE_B0,
        "blackhole": forge._C.Arch.BLACKHOLE,
        "quasar": forge._C.Arch.QUASAR,
    }[arch_name]

    node = helper.make_node("Add", inputs=["input_A", "input_B"], outputs=["output"])
    graph = helper.make_graph(
        nodes=[node],
        name="AddGraph",
        inputs=[
            helper.make_tensor_value_info("input_A", TensorProto.FLOAT, SHAPE),
            helper.make_tensor_value_info("input_B", TensorProto.FLOAT, SHAPE),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)],
    )
    model = helper.make_model(graph, producer_name="AddModel", opset_imports=[helper.make_operatorsetid("", 21)])

    cfg = CompilerConfig(mlir_config=MLIRConfig().set_target_arch(arch))
    compiled = forge.compile(
        model,
        [torch.rand(SHAPE), torch.rand(SHAPE)],
        module_name=f"walkthrough_add_{arch_name}",
        compiler_cfg=cfg,
    )

    binary = compiled.compiled_binary
    blob = json.loads(binary.as_json())

    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"walkthrough_add_{arch_name}.ttnn")
    try:
        binary.store(path)
        size = os.path.getsize(path)
    except Exception:
        path, size = None, None

    return {"arch": arch_name, "onnx": model, "blob": blob, "path": path, "size": size}


# --------------------------------------------------------------------------- helpers


def ttnn_source(cap):
    return cap["blob"].get("mlir", {}).get("source", "") or ""


def program(cap):
    programs = cap["blob"].get("programs") or [{}]
    return programs[0]


def op_lines(cap):
    """One line per op: the SSA expression, with the type signature trimmed.

    debug_info carries the whole MLIR line including a multi-hundred-character
    ttnn_layout; keeping it would bury the op it is describing.
    """
    out = []
    for i, op in enumerate(program(cap).get("operations", [])):
        debug = (op.get("debug_info") or "").split("\n")[0]
        expr = debug.split(" : (")[0].strip()
        out.append(f"{i}  {op.get('type_type','?')}")
        if expr:
            out.append(f"     {expr[:52]}")
    return out or ["(no operations)"]


def grep_source(cap, needle, before=0, after=0, limit=14):
    """Pull the lines around `needle` out of the embedded TTNN module."""
    lines = ttnn_source(cap).splitlines()
    hits = []
    for i, line in enumerate(lines):
        if needle in line:
            lo, hi = max(0, i - before), min(len(lines), i + after + 1)
            hits.extend(lines[lo:hi])
            if len(hits) >= limit:
                break
    return hits[:limit] or [f"(no line containing {needle!r})"]


def hexdump(path, count=96, per_row=8):
    """8 bytes per row so a line fits inside one pane without wrapping."""
    if not path or not os.path.exists(path):
        return ["(binary not written to disk)"]
    with open(path, "rb") as fh:
        data = fh.read(count)
    out = []
    for off in range(0, len(data), per_row):
        chunk = data[off : off + per_row]
        hexpart = " ".join(f"{b:02x}" for b in chunk).ljust(per_row * 3 - 1)
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        out.append(f"{off:04x}  {hexpart}  {text}")
    return out


# --------------------------------------------------------------------------- stages


def build_stages(cap):
    blob = cap["blob"]
    prog = program(cap)
    ops = prog.get("operations", [])
    arch = cap["arch"]

    def sysdesc_excerpt():
        raw = json.dumps(blob.get("system_desc", {}))
        chip = blob.get("system_desc", {}).get("chip_descs", [{}])
        fields = ("arch", "grid", "l1_size", "num_dram_channels", "dram_channel_size")
        if chip and isinstance(chip[0], dict):
            return [f"{k} = {chip[0].get(k)}" for k in fields if k in chip[0]] or [raw[:300]]
        return [raw[:300]]

    return [
        dict(
            layer="host · python",
            title="The ONNX graph",
            where="your test / scripts/add_op_walkthrough.py",
            code=[
                "A single-node ONNX graph is the smallest thing that",
                "exercises the entire stack end to end.",
                "",
                'helper.make_node("Add",',
                '    inputs=["input_A", "input_B"],',
                '    outputs=["output"])',
                "",
                f"Both inputs are float32 {tuple(SHAPE)}.",
            ],
            fb_title="nothing yet",
            fb=[
                "The flatbuffer does not exist.",
                "",
                "Everything up to stage 8 is host-side graph",
                "manipulation. The binary is only serialised at",
                "the very end.",
            ],
        ),
        dict(
            layer="host · python",
            title="forge.compile()",
            where="forge/forge/compile.py:279-290",
            code=[
                "Entry point. Drives a fixed table of twelve",
                "CompileDepth stages, each a function that",
                "transforms the graph and hands it on:",
                "",
                "  INIT_COMPILE",
                "  GENERATE_INITIAL_GRAPH   <- TVM converts ONNX",
                "  POST_INITIAL_GRAPH_PASS",
                "  CONSTEVAL_GRAPH          <- fold constant subgraphs",
                "  POST_PATTERN_MATCHER",
                "  OPTIMIZED_GRAPH",
                "  AUTOGRAD / POST_AUTOGRAD",
                "  PRE_LOWERING_PASS",
                "  SPLIT_GRAPH",
                "  RUN_MLIR_COMPILER        <- the interesting one",
                "  FINISH_COMPILE",
            ],
            fb_title="still nothing",
            fb=[
                "Consteval matters for the binary even though it",
                "produces none of it: anything folded here never",
                "becomes an operation in the final program.",
                "",
                "For a bare Add there is nothing constant to fold.",
                "On ResNet-50 this removes 53 weight-preparation",
                "chains before the flatbuffer is ever written.",
            ],
        ),
        dict(
            layer="forge · c++",
            title="Where the target comes from",
            where="forge/csrc/passes/lower_to_mlir.cpp:201",
            code=[
                "Normally emit_mlir() stamps the module with the",
                "descriptor of the attached device:",
                "",
                "  graphModule_->setAttr(",
                "      SystemDescAttr::name,",
                "      get_system_desc_attr(graphModule_));",
                "",
                "which calls TTSystem::get_system() and opens",
                "hardware.",
                "",
                f"This run set target_arch={arch}, so that stamp is",
                "SKIPPED and tt-mlir builds the descriptor from the",
                "pipeline option instead. That is why no device was",
                "needed to produce what you are looking at.",
            ],
            fb_title="system_desc — first real content",
            fb=sysdesc_excerpt()
            + [
                "",
                "This lands in the binary. The runtime reads it back",
                "to check the program matches the machine it is",
                "about to run on.",
            ],
        ),
        dict(
            layer="forge · c++",
            title="lower_to_mlir → TTIR",
            where="forge/csrc/passes/lower_to_mlir.cpp",
            code=[
                "Each forge graph node is emitted as a TTIR op",
                "through a handler map. The Add becomes:",
                "",
                '  %0 = "ttir.add"(%arg0, %arg1)',
                "",
                "TTIR is the arch-neutral tensor IR. No layouts, no",
                "memory spaces, no grids -- just ops and shapes.",
                "",
                DIM("Note: the TTIR module is not retained in the"),
                DIM("binary. Only the final TTNN module is embedded,"),
                DIM("which is what stage 7 shows."),
            ],
            fb_title="nothing yet",
            fb=[
                "TTIR is purely in-memory. It is the input to the",
                "pipeline, never serialised.",
            ],
        ),
        dict(
            layer="tt-mlir",
            title="ttcore-register-device",
            where="lib/Dialect/TTCore/Transforms/TTCoreRegisterDevice.cpp:74-88",
            code=[
                "First pass of ttir-to-ttnn-backend-pipeline.",
                "",
                "  systemDescPath.empty()",
                "    ? registerDevice(op, mockSystemDescArch, ...)",
                "    : registerDevice(op, systemDescPath, ...)",
                "",
                "It only fabricates a descriptor if the module does",
                "not already carry one -- which is exactly the",
                "handoff stage 3 set up.",
                "",
                "A ttcore.device with the worker grid is attached.",
            ],
            fb_title="grid drives everything downstream",
            fb=grep_source(cap, "ttcore.device", limit=4)
            + [
                "",
                "Every layout and sharding decision from here is",
                "derived from this grid and the L1 size. No pass",
                "asks 'which chip is this'.",
            ],
        ),
        dict(
            layer="tt-mlir",
            title="TTIR-level passes",
            where="lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp:32-97",
            code=[
                "Decomposition, fusing, implicit-broadcast folding,",
                "sliding-window flattening, inverse-op erasure,",
                "inlining, CSE.",
                "",
                "Composite ops are broken into primitives here and",
                "patterns like conv+relu are fused into one op.",
                "",
                "A bare Add passes through nearly untouched -- which",
                "is why it is the right thing to walk through first.",
            ],
            fb_title="op count is decided here",
            fb=[
                "Fusion and decomposition are what determine how",
                "many operations end up in the program.",
                "",
                f"This compile ended with {len(ops)} operations.",
            ],
        ),
        dict(
            layer="tt-mlir",
            title="TTNNLayout → ConvertTTIRToTTNN",
            where="TTNNPipelines.cpp:166-174",
            code=[
                "The step that decides WHERE TENSORS LIVE.",
                "",
                "  pm.addPass(createTTNNLayout());",
                "  pm.addPass(createConvertTTIRToTTNNPass());",
                "",
                "Each tensor gets a ttnn_layout: tile shape, core",
                "grid, memref, DRAM vs L1, interleaved vs sharded.",
                "Then TTIR ops are rewritten as TTNN ops.",
                "",
                "This is where ttir.add becomes ttnn.add, and where",
                "the to_layout conversions around it appear.",
            ],
            fb_title="the layout that will be serialised",
            fb=grep_source(cap, "#ttnn_layout", limit=6)
            + ["", "DRAM + interleaved is what optimization-level 0", "gives you: the most conservative choice."],
        ),
        dict(
            layer="tt-mlir",
            title="The TTNN module",
            where="embedded in the binary as mlir.source",
            code=[
                "The final IR before serialisation. This is the",
                "authoritative description of what will run.",
                "",
                "The right pane is the real @forward function from",
                "the binary you just built -- not a reconstruction.",
                "",
                "Note the deallocates: liveness analysis inserted",
                "them so intermediates are freed as soon as their",
                "last use passes.",
            ],
            fb_title="mlir.source (real, from the binary)",
            fb=grep_source(cap, "ttnn.", before=0, after=0, limit=16),
        ),
        dict(
            layer="tt-mlir",
            title="ttnnToFlatbuffer",
            where="ttmlir/Target/TTNN/TTNNToFlatbuffer.h",
            code=[
                "The TTNN module is serialised. Compilation ends",
                "here -- everything after this is execution.",
                "",
                "  auto binary = ttnnToFlatbuffer(mlir_module.get());",
                "",
                "The binary is self-describing: it carries the ops,",
                "the tensor descriptors, the system descriptor it",
                "was compiled against, and a schema hash so the",
                "runtime can refuse a mismatched build.",
            ],
            fb_title="the binary now exists",
            fb=[
                f"top-level keys : {', '.join(blob.keys())}",
                f"schema_hash    : {str(blob.get('schema_hash'))[:40]}",
                f"ttmlir version : {blob.get('version')}",
                f"program name   : {prog.get('name')}",
                f"operations     : {len(ops)}",
                f"inputs         : {len(prog.get('inputs', []))}",
                f"outputs        : {len(prog.get('outputs', []))}",
                f"size on disk   : {cap['size']} bytes" if cap["size"] else "size on disk   : (not written)",
            ],
        ),
        dict(
            layer="tt-mlir",
            title="What is actually in it",
            where=cap["path"] or "(in memory)",
            code=[
                "The op stream the runtime will replay, in order.",
                "",
                "Two things worth noticing:",
                "",
                "  - the add is one op among several; most of the",
                "    program is layout conversion and memory",
                "    management",
                "",
                "  - each op carries its MLIR line as debug_info,",
                "    so a runtime failure can be traced back to the",
                "    IR that produced it",
            ],
            fb_title="programs[0].operations",
            fb=op_lines(cap),
        ),
        dict(
            layer="tt-mlir",
            title="The bytes",
            where=cap["path"] or "(in memory)",
            code=[
                "A flatbuffer: no parsing step, the runtime reads",
                "fields directly out of the mapped buffer.",
                "",
                "This file is the complete deliverable of the",
                "compiler. Hand it to ttrt or to forge's runtime on",
                "a matching machine and it runs.",
            ],
            fb_title="first 96 bytes",
            fb=hexdump(cap["path"]),
        ),
        dict(
            layer="tt-metal",
            title="Runtime replay",
            where="runtime/lib/ttnn/program_executor.cpp:222-290",
            code=[
                "ProgramExecutor::execute() walks the operation list",
                "and calls runOperation() on each -- a switch on",
                "OpType routing every op to its implementation:",
                "",
                "  case OpType::ToLayoutOp:",
                "      return operations::layout::run(...)",
                "  case OpType::EltwiseBinaryOp:",
                "      return operations::binary::run(...)",
                "",
                "which calls ttnn::add, which selects a device",
                "operation and then a PROGRAM FACTORY.",
            ],
            fb_title="binary → live ops",
            fb=[
                "The flatbuffer is now being consumed rather than",
                "produced.",
                "",
                "Factory selection is where architectures diverge.",
                "Everything up to this point was arch-neutral.",
                "",
                WARN("On Quasar the mainline factories are refused:"),
                WARN("DataMovementKernel's constructor hard-fails."),
            ],
        ),
        dict(
            layer="silicon",
            title="Three kernels, five processors",
            where="binary_ng_utils.cpp:82-115 (get_kernel_file_path)",
            code=[
                "For an un-broadcast interleaved Add the factory",
                "names one source file per RISC-V role:",
                "",
                "  reader_interleaved_no_bcast.cpp   BRISC",
                "  eltwise_binary_no_bcast.cpp       TRISC0/1/2",
                "  writer_interleaved_no_bcast.cpp   NCRISC",
                "",
                "They are JIT-compiled for the arch, then enqueued.",
            ],
            fb_title="the binary is spent",
            fb=[
                "Nothing of the flatbuffer survives past here. It",
                "described what to run; the kernels are what runs.",
                "",
                "  BRISC   DRAM -> L1 circular buffer",
                "  TRISC0  L1 -> srcA / srcB registers",
                "  TRISC1  FPU -> destination registers",
                "  TRISC2  dest -> output circular buffer",
                "  NCRISC  L1 -> DRAM",
            ],
        ),
        dict(
            layer="silicon",
            title="The addition",
            where="kernels/compute/eltwise_binary_no_bcast.cpp:46-57",
            code=[
                "The only arithmetic in the entire journey:",
                "",
                "  tile_regs_acquire();",
                "  add_tiles(cb_a, cb_b, i, i, dst);",
                "  tile_regs_commit();",
                "",
                "  tile_regs_wait();",
                "  pack_tile(i, cb_out);",
                "  tile_regs_release();",
                "",
                "Everything else -- all twelve stages before this --",
                "was logistics to get two tiles next to an FPU.",
            ],
            fb_title="round trip complete",
            fb=[
                "Result tiles go back out to DRAM, are copied to",
                "host memory, and become a torch.Tensor.",
                "",
                "verify() then compares against the framework model",
                "by PCC.",
                "",
                DIM("On Quasar this stage is where execution currently"),
                DIM("wedges: srcA/srcB reach valid=1 unpack=1 and"),
                DIM("matrix=0 -- unpack delivered, math never took it."),
            ],
        ),
    ]


# --------------------------------------------------------------------------- render


def wrap(lines, width):
    out = []
    for line in lines:
        if not line:
            out.append("")
            continue
        # Pre-coloured lines are passed through; ANSI breaks naive wrapping.
        if "\033[" in line:
            out.append(line)
            continue
        out.extend(textwrap.wrap(line, width) or [""])
    return out


def visible_len(s):
    out, i = 0, 0
    while i < len(s):
        if s[i] == "\033":
            while i < len(s) and s[i] != "m":
                i += 1
        else:
            out += 1
        i += 1
    return out


def pad(s, width):
    return s + " " * max(0, width - visible_len(s))


def render(stage, index, total, cols):
    layer = stage["layer"]
    lc = LAYER_COLORS.get(layer, BOLD)

    print("\033[2J\033[H" if _TTY else "")
    head = f" {index + 1}/{total}  {stage['title']} "
    print(BOLD(head))
    print(f" {lc(layer.upper())}   {DIM(stage['where'])}")
    print(RULE("─" * min(cols, 100)))
    print()

    stacked = cols < 96
    if stacked:
        print(CODE_C("── CODE ──"))
        for line in wrap(stage["code"], min(cols - 2, 88)):
            print("  " + line)
        print()
        print(FB_C(f"── FLATBUFFER · {stage['fb_title']} ──"))
        for line in wrap(stage["fb"], min(cols - 2, 88)):
            print("  " + line)
    else:
        gutter = 3
        pane = (min(cols, 150) - gutter) // 2
        left = wrap(stage["code"], pane)
        right = wrap(stage["fb"], pane)
        print(pad(CODE_C("── CODE ──"), pane) + " " * gutter + FB_C(f"── FLATBUFFER · {stage['fb_title']}"))
        print()
        for i in range(max(len(left), len(right))):
            l = left[i] if i < len(left) else ""
            r = right[i] if i < len(right) else ""
            print(pad(l, pane) + " " * gutter + r)

    print()
    print(RULE("─" * min(cols, 100)))
    print(DIM(" Enter next   b back   q quit   <n> jump"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="wormhole_b0", choices=["wormhole_b0", "blackhole", "quasar"])
    ap.add_argument("--no-pager", action="store_true", help="print every stage and exit")
    args = ap.parse_args()

    print(f"Compiling a single ONNX Add for {args.arch} (no device needed)…")
    try:
        cap = run_compile(args.arch)
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        print(f"\n{WARN('Compile failed:')} {exc}\n")
        print("This walkthrough reads a real compile. Check that forge imports:")
        print("    source env/activate")
        return 1

    stages = build_stages(cap)
    total = len(stages)
    print(f"Done — {len(program(cap).get('operations', []))} ops, {cap['size']} bytes.\n")

    if args.no_pager:
        cols = shutil.get_terminal_size((100, 40)).columns
        for i, st in enumerate(stages):
            render(st, i, total, cols)
            print()
        return 0

    i = 0
    while True:
        cols = shutil.get_terminal_size((100, 40)).columns
        render(stages[i], i, total, cols)
        try:
            key = input(" > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if key == "q":
            return 0
        if key == "b":
            i = max(0, i - 1)
        elif key.isdigit() and 1 <= int(key) <= total:
            i = int(key) - 1
        else:
            i += 1
            if i >= total:
                print(BOLD("\n  End of walkthrough.\n"))
                return 0


if __name__ == "__main__":
    sys.exit(main())
