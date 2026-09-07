#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Step an ONNX Add through tt-forge-onnx, file by file, watching the IR change.

Press Enter to travel to the next file. Three things on screen at once:

  travel bar   the chain of files, with where you are now
  left pane    the REAL source, read off disk, at the line that matters
  right pane   the IR at this point, with +/- marking what this hop changed

Nothing is transcribed or paraphrased. Source is read live from the working tree,
so it stays correct as the code moves. The IR snapshots come from running the real
pipeline passes one at a time, and the flatbuffer from a real forge compile that
names a target architecture -- so the whole thing needs no device.

    python scripts/add_op_walkthrough.py
    python scripts/add_op_walkthrough.py --arch quasar
    python scripts/add_op_walkthrough.py --no-pager
    python scripts/add_op_walkthrough.py --ir 6        # print one IR snapshot whole

Keys:  Enter next   b back   q quit   <n> jump   f full source   i full IR
"""

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TTMLIR = os.path.join(REPO, "third_party", "tt-mlir")
TTMETAL = os.path.join(TTMLIR, "third_party", "tt-metal", "src", "tt-metal")
OPT = os.path.join(TTMLIR, "build", "bin", "ttmlir-opt")
SHAPE = [2, 32, 32]

# --------------------------------------------------------------------------- colour

_TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code):
    return (lambda s: f"\033[{code}m{s}\033[0m") if _TTY else (lambda s: s)


BOLD, DIM = _c("1"), _c("2")
BLUE, AMBER = _c("38;5;75"), _c("38;5;209")
GREEN, RED = _c("38;5;114"), _c("38;5;203")
RULE, HERE = _c("38;5;240"), _c("48;5;24")
LAYER = {
    "host": _c("38;5;110"),
    "forge": _c("38;5;75"),
    "tt-mlir": _c("38;5;140"),
    "tt-metal": _c("38;5;176"),
    "silicon": _c("38;5;209"),
}


def vlen(s):
    """Printable width, ignoring ANSI escapes."""
    out, i = 0, 0
    while i < len(s):
        if s[i] == "\033":
            while i < len(s) and s[i] != "m":
                i += 1
        else:
            out += 1
        i += 1
    return out


def pad(s, w):
    return s + " " * max(0, w - vlen(s))


def clip(s, w):
    """Truncate to printable width. Only used on lines known to be ANSI-free."""
    return s if len(s) <= w else s[: max(0, w - 1)] + "…"


# --------------------------------------------------------------------------- source


def read_source(relpath, anchor, before=6, after=10):
    """Real lines around `anchor` in `relpath`, as (lineno, text, is_anchor)."""
    path = relpath if os.path.isabs(relpath) else os.path.join(REPO, relpath)
    if not os.path.exists(path):
        return [(0, f"(missing: {relpath})", False)]
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    idx = next((i for i, l in enumerate(lines) if anchor in l), None)
    if idx is None:
        return [(0, f"(anchor not found: {anchor!r})", False)]
    lo, hi = max(0, idx - before), min(len(lines), idx + after + 1)
    return [(i + 1, lines[i], i == idx) for i in range(lo, hi)]


def render_source(rows, width):
    out = []
    for no, text, is_anchor in rows:
        body = clip(text.rstrip("\n").replace("\t", "    "), width - 7)
        num = f"{no:5d} " if no else "      "
        if is_anchor:
            out.append(HERE(pad(f"{num}{body}", width)))
        else:
            out.append(DIM(num) + body)
    return out


# --------------------------------------------------------------------------- IR chain


def run_opt(passes, src, arch):
    """Run ttmlir-opt with an explicit pass list and return the resulting IR."""
    if not os.path.exists(OPT):
        return None
    cmd = [OPT]
    for p in passes:
        cmd.append(p.format(arch=arch))
    cmd.append(src)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (subprocess.TimeoutExpired, OSError):
        return None
    return res.stdout if res.returncode == 0 and res.stdout.strip() else None


def build_ir_chain(arch, scratch):
    """Real IR after each interesting pass, by invoking the passes individually."""
    ttir = (
        "module {\n"
        f"  func.func @forward(%arg0: tensor<{'x'.join(map(str, SHAPE))}xf32>, "
        f"%arg1: tensor<{'x'.join(map(str, SHAPE))}xf32>) "
        f"-> tensor<{'x'.join(map(str, SHAPE))}xf32> {{\n"
        f'    %0 = "ttir.add"(%arg0, %arg1) : (tensor<{"x".join(map(str, SHAPE))}xf32>, '
        f'tensor<{"x".join(map(str, SHAPE))}xf32>) -> tensor<{"x".join(map(str, SHAPE))}xf32>\n'
        f"    return %0 : tensor<{'x'.join(map(str, SHAPE))}xf32>\n"
        "  }\n"
        "}\n"
    )
    src = os.path.join(scratch, f"add_{arch}.ttir.mlir")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write(ttir)

    reg = ["--ttcore-register-device=mock-system-desc-arch={arch}"]
    chain = {
        "ttir": ttir,
        "after_register_device": run_opt(reg, src, arch),
        "after_ttnn_layout": run_opt(reg + ["--ttnn-layout"], src, arch),
        "after_convert": run_opt(reg + ["--ttnn-layout", "--convert-ttir-to-ttnn"], src, arch),
        "full_pipeline": run_opt(["--ttir-to-ttnn-backend-pipeline=mock-system-desc-arch={arch}"], src, arch),
    }
    return chain, src


def func_body(ir, keep=("ttir.", "ttnn.", "func.func", "return", "ttcore.device")):
    """The interesting lines of a module -- op lines, not the attribute preamble."""
    if not ir:
        return ["(pass did not run — is ttmlir-opt built?)"]
    out = [l.rstrip() for l in ir.splitlines() if any(k in l for k in keep)]
    return out or ["(no matching lines)"]


def ir_diff(prev, cur, width):
    """Unified-ish diff of two IR snapshots, marking what this hop changed."""
    a, b = func_body(prev), func_body(cur)
    out = []
    for line in difflib.unified_diff(a, b, lineterm="", n=1):
        if line.startswith(("---", "+++", "@@")):
            continue
        body = clip(line[1:].strip(), width - 2)
        if line.startswith("+"):
            out.append(GREEN("+ " + body))
        elif line.startswith("-"):
            out.append(RED("- " + body))
        else:
            out.append(DIM("  " + body))
    return out or [DIM("(no change to the op lines at this step)")]


def ir_plain(cur, width):
    return [clip(l.strip(), width) for l in func_body(cur)]


# --------------------------------------------------------------------------- compile


def run_compile(arch_name, scratch):
    import torch
    from onnx import TensorProto, helper

    import forge
    from forge.config import CompilerConfig, MLIRConfig

    arch = {
        "wormhole_b0": forge._C.Arch.WORMHOLE_B0,
        "blackhole": forge._C.Arch.BLACKHOLE,
        "quasar": forge._C.Arch.QUASAR,
    }[arch_name]

    graph = helper.make_graph(
        nodes=[helper.make_node("Add", inputs=["input_A", "input_B"], outputs=["output"])],
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
    path = os.path.join(scratch, f"walkthrough_add_{arch_name}.ttnn")
    try:
        binary.store(path)
        size = os.path.getsize(path)
    except Exception:
        path, size = None, None
    return {"blob": blob, "path": path, "size": size}


def hexdump(path, count=64, per_row=8):
    if not path or not os.path.exists(path):
        return ["(binary not on disk)"]
    with open(path, "rb") as fh:
        data = fh.read(count)
    out = []
    for off in range(0, len(data), per_row):
        chunk = data[off : off + per_row]
        hx = " ".join(f"{b:02x}" for b in chunk).ljust(per_row * 3 - 1)
        tx = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        out.append(f"{off:04x}  {hx}  {tx}")
    return out


# --------------------------------------------------------------------------- stages


def build_stages(cap, chain, arch):
    blob, prog = cap["blob"], (cap["blob"].get("programs") or [{}])[0]
    ops = prog.get("operations", [])

    def op_stream(width):
        out = []
        for i, op in enumerate(ops):
            expr = (op.get("debug_info") or "").split("\n")[0].split(" : (")[0].strip()
            out.append(f"{i}  {op.get('type_type','?')}")
            if expr:
                out.append(clip("     " + expr, width))
        return out or ["(none)"]

    def fb_summary(width):
        return [
            f"keys        {', '.join(blob.keys())}",
            f"program     {prog.get('name')}",
            f"operations  {len(ops)}",
            f"inputs      {len(prog.get('inputs', []))}",
            f"outputs     {len(prog.get('outputs', []))}",
            f"bytes       {cap['size']}",
            f"tt-mlir     {str(blob.get('ttmlir_git_hash'))[:12]}",
        ]

    S = []
    add = S.append

    add(
        dict(
            layer="host",
            short="compile.py",
            title="forge.compile dispatches the stages",
            file="forge/forge/compile.py",
            anchor="CompileDepth.RUN_MLIR_COMPILER: run_mlir_compiler",
            note="Twelve stages in the table, nine actually run, and not in enum order: each returns its successor, so POST_AUTOGRAD_PASS precedes CONSTEVAL_GRAPH. RUN_MLIR_COMPILER is the hop into C++.",
            ir_key=None,
            ir_label="no IR yet",
            ir=lambda w: ["The graph is still forge's own IR.", "", "Nothing MLIR exists until the next file."],
        )
    )
    add(
        dict(
            layer="host",
            short="compile.py",
            title="…and calls into the C++ compiler",
            file="forge/forge/compile.py",
            anchor="context.compiled_binary = forge._C.run_mlir_compiler",
            note="The pybind boundary. compiler_cfg.mlir_config carries target_arch.",
            ir_key=None,
            ir_label="no IR yet",
            ir=lambda w: [
                f"This run passes target_arch={arch},",
                "which is what removes the device",
                "requirement two files from now.",
            ],
        )
    )
    add(
        dict(
            layer="forge",
            short="mlir_compiler.cpp",
            title="Hand the graph to the lowering",
            file="forge/csrc/passes/mlir_compiler.cpp",
            anchor="lower_to_mlir(module, context, mlir_config)",
            note="mlir_config is threaded in so lowering can see the target.",
            ir_key=None,
            ir_label="no IR yet",
            ir=lambda w: ["Next file builds the first MLIR."],
        )
    )
    add(
        dict(
            layer="forge",
            short="lower_to_mlir.cpp",
            title="Emit TTIR — and decide the target",
            file="forge/csrc/passes/lower_to_mlir.cpp",
            anchor="SystemDescAttr::name, get_system_desc_attr",
            note="Stamping the live device's descriptor here is what normally forces hardware. "
            "Skipped when a target is named.",
            ir_key="ttir",
            ir_label="TTIR — first IR in existence",
            ir=None,
        )
    )
    add(
        dict(
            layer="forge",
            short="mlir_passes.cpp",
            title="Look up the pipeline by name",
            file="forge/csrc/passes/mlir_passes.cpp",
            anchor="ttir-to-ttnn-backend-pipeline",
            note="The options string built from MLIRConfig is parsed by MLIR itself.",
            ir_key="ttir",
            ir_label="TTIR — unchanged, about to enter the pipeline",
            ir=None,
        )
    )
    add(
        dict(
            layer="tt-mlir",
            short="TTCoreRegisterDevice.cpp",
            title="First pass: attach the device",
            file="third_party/tt-mlir/lib/Dialect/TTCore/Transforms/TTCoreRegisterDevice.cpp",
            anchor="mockSystemDescArch",
            before=8,
            after=12,
            note="Builds the descriptor from the pipeline option, because the module carries none.",
            ir_key="after_register_device",
            ir_label="+ ttcore.device appears",
            ir=None,
        )
    )
    add(
        dict(
            layer="tt-mlir",
            short="TTNNPipelines.cpp",
            title="Decide where tensors live",
            file="third_party/tt-mlir/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp",
            anchor="createTTNNLayout()",
            before=4,
            after=8,
            note="Grid and L1 size from the descriptor become a layout on every tensor.",
            ir_key="after_ttnn_layout",
            ir_label="+ ttnn_layout on every tensor (ops still ttir)",
            ir=None,
        )
    )
    add(
        dict(
            layer="tt-mlir",
            short="TTNNPipelines.cpp",
            title="Rewrite TTIR ops as TTNN ops",
            file="third_party/tt-mlir/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp",
            anchor="createConvertTTIRToTTNNPass()",
            before=4,
            after=6,
            note="ttir.add becomes ttnn.add. This is the dialect boundary.",
            ir_key="after_convert",
            ir_label="ttir.add → ttnn.add",
            ir=None,
        )
    )
    add(
        dict(
            layer="tt-mlir",
            short="(full pipeline)",
            title="The rest of the pipeline",
            file="third_party/tt-mlir/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp",
            anchor="createTTIRToTTIRDecompositionPass()",
            before=6,
            after=10,
            note="Decomposition, fusing, layout conversions and liveness deallocates.",
            ir_key="full_pipeline",
            ir_label="+ to_layout, + deallocate",
            ir=None,
        )
    )
    add(
        dict(
            layer="tt-mlir",
            short="flatbuffer",
            title="Serialise — compilation ends",
            file="forge/csrc/passes/mlir_compiler.cpp",
            anchor="ttnnToFlatbuffer",
            note="From here the artifact is bytes, not IR. Everything after is execution.",
            ir_key=None,
            ir_label="the binary (real, from this compile)",
            ir=fb_summary,
        )
    )
    add(
        dict(
            layer="tt-mlir",
            short="flatbuffer",
            title="What the runtime will replay",
            file="forge/csrc/passes/mlir_compiler.cpp",
            anchor="ttnnToFlatbuffer",
            note="The op stream, in order, read back out of the binary just built.",
            ir_key=None,
            ir_label="programs[0].operations",
            ir=op_stream,
        )
    )
    add(
        dict(
            layer="forge",
            short="runtime.cpp",
            title="Open a device, submit the binary",
            file="forge/csrc/runtime/runtime.cpp",
            anchor="runtime::submit(device, binary, program_idx, rt_inputs)",
            note="Now hardware is genuinely required — this is execution, not compilation.",
            ir_key=None,
            ir_label="first 64 bytes on disk",
            ir=lambda w: hexdump(cap["path"]),
        )
    )
    add(
        dict(
            layer="tt-metal",
            short="program_executor.cpp",
            title="Replay op by op",
            file="third_party/tt-mlir/runtime/lib/ttnn/program_executor.cpp",
            anchor="void ProgramExecutor::runOperation",
            before=2,
            after=16,
            note="A switch on OpType routes each flatbuffer op to a TTNN call.",
            ir_key=None,
            ir_label="binary being consumed",
            ir=lambda w: [
                "EltwiseBinaryOp → ttnn::add",
                "",
                "which selects a device operation,",
                "then a program factory.",
            ],
        )
    )
    add(
        dict(
            layer="tt-metal",
            short="binary_ng_utils.cpp",
            title="Pick the kernels",
            file=os.path.join(TTMETAL, "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp"),
            anchor="get_kernel_file_path",
            before=2,
            after=14,
            note="One source file per RISC-V role. This is where architectures diverge.",
            ir_key=None,
            ir_label="the Quasar boundary",
            ir=lambda w: [
                "Everything so far was arch-neutral.",
                "",
                RED("On Quasar the mainline factories are"),
                RED("refused outright — next file shows why."),
            ],
        )
    )
    add(
        dict(
            layer="tt-metal",
            short="kernel.hpp",
            title="Why Quasar needs its own factories",
            file=os.path.join(TTMETAL, "tt_metal/impl/kernels/kernel.hpp"),
            anchor="DataMovementKernel is not supported on Quasar",
            before=6,
            after=4,
            note="A hard fail in the constructor, so an unported op compiles then dies at execution.",
            ir_key=None,
            ir_label="compile vs run",
            ir=lambda w: [
                "This is the asymmetry that makes Quasar",
                "compile-clean and run-broken.",
                "",
                f"This walkthrough compiled for {arch}",
                "without ever reaching this line.",
            ],
        )
    )
    add(
        dict(
            layer="silicon",
            short="eltwise_binary_no_bcast.cpp",
            title="The actual addition",
            file=os.path.join(
                TTMETAL,
                "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp",
            ),
            anchor="tile_regs_acquire",
            before=8,
            after=12,
            note="The only arithmetic in the whole journey. Everything else was logistics.",
            ir_key=None,
            ir_label="round trip",
            ir=lambda w: [
                "dest regs → output CB → DRAM → host",
                "→ torch.Tensor → verify() by PCC.",
                "",
                DIM("On Quasar execution wedges just before"),
                DIM("this: srcA/srcB valid=1, matrix=0."),
            ],
        )
    )
    return S


# --------------------------------------------------------------------------- render


def travel_bar(stages, i, width):
    """The chain of files, with the current hop highlighted."""
    names = []
    for n, st in enumerate(stages):
        s = st["short"]
        names.append(HERE(f" {s} ") if n == i else (DIM(s) if abs(n - i) > 2 else s))
    bar, out = "", []
    for n, chunk in enumerate(names):
        sep = DIM(" → ") if n else ""
        if vlen(bar) + vlen(sep) + vlen(chunk) > width:
            out.append(bar)
            bar = chunk
        else:
            bar += sep + chunk
    out.append(bar)
    return out


def render(stages, i, chain, cols):
    st = stages[i]
    print("\033[2J\033[H" if _TTY else "")
    lc = LAYER.get(st["layer"], BOLD)
    print(BOLD(f" {i + 1}/{len(stages)}  {st['title']}"))
    print(f" {lc(st['layer'].upper())}")
    for line in travel_bar(stages, i, min(cols, 150) - 2):
        print(" " + line)
    print(RULE("─" * min(cols, 150)))

    shown = st["file"]
    if shown.startswith(REPO):
        shown = os.path.relpath(shown, REPO)
    print(f" {BLUE(shown)}")
    for line in textwrap.wrap(st["note"], min(cols, 150) - 2):
        print(" " + DIM(line))
    print()

    stacked = cols < 110
    pane = (min(cols, 170) - 3) // 2 if not stacked else min(cols, 110) - 2

    rows = read_source(st["file"], st["anchor"], st.get("before", 6), st.get("after", 10))
    left = render_source(rows, pane)

    if st["ir_key"]:
        keys = list(chain.keys())
        pos = keys.index(st["ir_key"])
        prev = chain[keys[pos - 1]] if pos > 0 else None
        right = ir_diff(prev, chain[st["ir_key"]], pane) if prev else ir_plain(chain[st["ir_key"]], pane)
    else:
        right = [clip(x, pane) if "\033" not in x else x for x in st["ir"](pane)]

    if stacked:
        print(BLUE("── SOURCE ──"))
        for l in left:
            print(" " + l)
        print()
        print(AMBER(f"── IR · {st['ir_label']} ──"))
        for r in right:
            print(" " + r)
    else:
        print(pad(BLUE("── SOURCE ──"), pane) + "   " + AMBER(f"── IR · {st['ir_label']}"))
        print()
        for n in range(max(len(left), len(right))):
            l = left[n] if n < len(left) else ""
            r = right[n] if n < len(right) else ""
            print(pad(l, pane) + "   " + r)

    print()
    print(RULE("─" * min(cols, 150)))
    print(DIM(" Enter next   b back   q quit   <n> jump   f full source   i full IR"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="wormhole_b0", choices=["wormhole_b0", "blackhole", "quasar"])
    ap.add_argument("--no-pager", action="store_true", help="print every stage and exit")
    ap.add_argument("--ir", type=int, metavar="N", help="print stage N's IR snapshot whole and exit")
    args = ap.parse_args()

    scratch = os.environ.get("TMPDIR", "/tmp")
    print(f"Building IR snapshots for {args.arch} …")
    chain, ttir_path = build_ir_chain(args.arch, scratch)
    if chain["full_pipeline"] is None:
        print(f"  {RED('warning')}: ttmlir-opt produced nothing — is {OPT} built?")

    print("Compiling a single ONNX Add (no device needed) …")
    try:
        cap = run_compile(args.arch, scratch)
    except Exception as exc:  # noqa: BLE001
        print(f"\n{RED('Compile failed:')} {exc}\n  try:  source env/activate")
        return 1

    stages = build_stages(cap, chain, args.arch)
    nops = len((cap["blob"].get("programs") or [{}])[0].get("operations", []))
    print(f"Ready — {nops} ops, {cap['size']} bytes, {len(stages)} hops.\n")

    if args.ir is not None:
        st = stages[max(0, min(args.ir - 1, len(stages) - 1))]
        print(chain.get(st["ir_key"]) or "(this stage has no IR snapshot)")
        return 0

    cols = shutil.get_terminal_size((120, 40)).columns
    if args.no_pager:
        for i in range(len(stages)):
            render(stages, i, chain, cols)
            print()
        return 0

    i = 0
    while True:
        cols = shutil.get_terminal_size((120, 40)).columns
        render(stages, i, chain, cols)
        try:
            key = input(" > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if key == "q":
            return 0
        if key == "f":
            rows = read_source(stages[i]["file"], stages[i]["anchor"], 30, 40)
            print("\n".join(render_source(rows, cols - 2)))
            input(DIM(" (enter) "))
        elif key == "i":
            print(chain.get(stages[i]["ir_key"]) or "(no IR snapshot at this stage)")
            input(DIM(" (enter) "))
        elif key == "b":
            i = max(0, i - 1)
        elif key.isdigit() and 1 <= int(key) <= len(stages):
            i = int(key) - 1
        else:
            i += 1
            if i >= len(stages):
                print(BOLD("\n  Journey complete.\n"))
                return 0


if __name__ == "__main__":
    sys.exit(main())
