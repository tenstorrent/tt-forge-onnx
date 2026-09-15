"""Execute ONE single-op ONNX model on the attached device. Argv: op name.

One op per process on purpose: the simulator-vs-hardware choice is process-wide,
and a hang in one op must not take the rest of the probe with it.
Prints a single RESULT line the driver greps.
"""
import sys, json
import numpy as np
import torch
from onnx import TensorProto, helper, numpy_helper
import forge
from forge.config import CompilerConfig
from forge._C.runtime.experimental import TTSystem

OP = sys.argv[1]

def vi(n, s, t=TensorProto.FLOAT):
    return helper.make_tensor_value_info(n, t, s)

def model(nodes, ins, outs, inits=()):
    return helper.make_model(helper.make_graph(nodes, "G", ins, outs, list(inits)),
                             producer_name="P",
                             opset_imports=[helper.make_operatorsetid("", 21)])

torch.manual_seed(0)

def case(op):
    if op == "reshape":
        s = numpy_helper.from_array(np.array([1, 2048], dtype=np.int64), name="s")
        m = model([helper.make_node("Reshape", ["a", "s"], ["o"])],
                  [vi("a", [1, 64, 32])], [vi("o", [1, 2048])], [s])
        x = [torch.rand(1, 64, 32)]
        return m, x, lambda a: a.reshape(1, 2048)
    if op == "permute":
        m = model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 2, 1])],
                  [vi("a", [1, 64, 32])], [vi("o", [1, 32, 64])])
        x = [torch.rand(1, 64, 32)]
        return m, x, lambda a: a.permute(0, 2, 1)
    if op == "permute_generic":
        # NCHW->NHWC style: a generic 4-D permute, not a last-two-dim swap. This is the
        # program conv2d's internal permute uses (reader_permute_interleaved_tiled_generic).
        m = model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 2, 3, 1])],
                  [vi("a", [1, 2, 32, 64])], [vi("o", [1, 32, 64, 2])])
        x = [torch.rand(1, 2, 32, 64)]
        return m, x, lambda a: a.permute(0, 2, 3, 1)
    if op == "transpose_hw":
        # Swap the last two dims of a 4-D tensor -> the swap_hw tiled-permute program,
        # which is a different program from the generic permute already verified.
        m = model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 1, 3, 2])],
                  [vi("a", [1, 1, 32, 64])], [vi("o", [1, 1, 64, 32])])
        x = [torch.rand(1, 1, 32, 64)]
        return m, x, lambda a: a.permute(0, 1, 3, 2)
    if op == "matmul":
        m = model([helper.make_node("MatMul", ["a", "b"], ["o"])],
                  [vi("a", [1, 32, 64]), vi("b", [1, 64, 32])], [vi("o", [1, 32, 32])])
        x = [torch.rand(1, 32, 64), torch.rand(1, 64, 32)]
        return m, x, lambda a, b: a @ b
    if op == "conv2d":
        w = numpy_helper.from_array(
            np.random.RandomState(0).rand(8, 3, 3, 3).astype("float32"), name="w")
        m = model([helper.make_node("Conv", ["a", "w"], ["o"], kernel_shape=[3, 3],
                                    pads=[1, 1, 1, 1], strides=[1, 1])],
                  [vi("a", [1, 3, 32, 32])], [vi("o", [1, 8, 32, 32])], [w])
        x = [torch.rand(1, 3, 32, 32)]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.conv2d(a, wt, padding=1)
    if op == "conv2d_32":
        # Channel-aligned conv: conv2d factories derive DFB entry_size from shard page
        # sizes, and a channel count that is not a multiple of 32 yields page sizes the
        # tile-indexed TDMA path cannot divide (e.g. 2080 vs a 2048-byte tile).
        w = numpy_helper.from_array(
            np.random.RandomState(0).rand(32, 32, 3, 3).astype("float32") * 0.1, name="w")
        m = model([helper.make_node("Conv", ["a", "w"], ["o"], kernel_shape=[3, 3],
                                    pads=[1, 1, 1, 1], strides=[1, 1])],
                  [vi("a", [1, 32, 32, 32])], [vi("o", [1, 32, 32, 32])], [w])
        x = [torch.rand(1, 32, 32, 32)]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.conv2d(a, wt, padding=1)
    if op == "conv2d_3x3_sp4":
        # ResNet-50's layer2 3x3 convolutions at the reduced proxy resolution: 4x4
        # spatial. Signed weights, so relu and sign errors are visible.
        w = numpy_helper.from_array(
            (np.random.RandomState(0).randn(128, 128, 3, 3) * 0.05).astype("float32"),
            name="w")
        m = model([helper.make_node("Conv", ["a", "w"], ["o"], kernel_shape=[3, 3],
                                    pads=[1, 1, 1, 1], strides=[1, 1])],
                  [vi("a", [1, 128, 4, 4])], [vi("o", [1, 128, 4, 4])], [w])
        x = [torch.rand(1, 128, 4, 4) - 0.5]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.conv2d(a, wt, padding=1)
    if op == "reshape_split_conv":
        # The reshape conv2d_1x1_sp2's graph performs on the convolution result:
        # [1,1,4,1024] -> [1,2,2,1024]. On a TILE tensor the 4 logical rows are
        # padded to 32; splitting them into two groups of 2, each padded to 32,
        # is a physical repack, not a relabel.
        sh = numpy_helper.from_array(np.array([1, 2, 2, 1024], dtype=np.int64), name="sh")
        m = model([helper.make_node("Reshape", ["a", "sh"], ["o"])],
                  [vi("a", [1, 1, 4, 1024])], [vi("o", [1, 2, 2, 1024])], [sh])
        x = [torch.rand(1, 1, 4, 1024) - 0.5]
        return m, x, lambda a: a.reshape(1, 2, 2, 1024)
    if op == "reshape_split_pool":
        # The same shape of reshape in max_pool2d_resnet's graph:
        # [1,1,64,64] -> [1,8,8,64].
        sh = numpy_helper.from_array(np.array([1, 8, 8, 64], dtype=np.int64), name="sh")
        m = model([helper.make_node("Reshape", ["a", "sh"], ["o"])],
                  [vi("a", [1, 1, 64, 64])], [vi("o", [1, 8, 8, 64])], [sh])
        x = [torch.rand(1, 1, 64, 64) - 0.5]
        return m, x, lambda a: a.reshape(1, 8, 8, 64)
    if op == "max_pool2d_small":
        # ResNet-50's pool geometry (3x3, stride 2, pad 1) at a quarter of the
        # rows, to tell "too slow on a one-worker emulator grid" apart from
        # "does not complete": same kernel, 64 rows instead of 256.
        m = model([helper.make_node("MaxPool", ["a"], ["o"], kernel_shape=[3, 3],
                                    pads=[1, 1, 1, 1], strides=[2, 2])],
                  [vi("a", [1, 32, 8, 8])], [vi("o", [1, 32, 4, 4])])
        x = [torch.rand(1, 32, 8, 8)]
        return m, x, lambda a: torch.nn.functional.max_pool2d(a, 3, 2, padding=1)
    if op in ("conv2d_1x1_bias_sp2", "conv2d_3x3_bias_sp2"):
        # Every convolution in a real ResNet block carries a bias: BatchNorm is
        # folded into it, and the graph passes it as conv2d's third operand. None
        # of the other conv probes here have one, which is exactly how the ops can
        # all pass while the model does not. Shapes are layer3's.
        k = 1 if "1x1" in op else 3
        cin, cout = (256, 1024) if k == 1 else (256, 256)
        w = numpy_helper.from_array(
            (np.random.RandomState(0).randn(cout, cin, k, k) * 0.05).astype("float32"),
            name="w")
        b = numpy_helper.from_array(
            (np.random.RandomState(1).randn(cout) * 0.5).astype("float32"), name="b")
        m = model([helper.make_node("Conv", ["a", "w", "b"], ["o"], kernel_shape=[k, k],
                                    pads=[k // 2] * 4, strides=[1, 1])],
                  [vi("a", [1, cin, 2, 2])], [vi("o", [1, cout, 2, 2])], [w, b])
        x = [torch.rand(1, cin, 2, 2) - 0.5]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        bt = torch.from_numpy(numpy_helper.to_array(b))
        return m, x, lambda a: torch.nn.functional.conv2d(a, wt, bt, padding=k // 2)
    if op.startswith("conv1x1_down"):
        # l3chain1 exactly: a 1x1 convolution 1024->256 on 2x2 spatial, with bias,
        # optionally followed by relu (which the compiler fuses into Conv2dConfig).
        # The passing probe was 256->1024 with bias and no relu, so this separates
        # the channel direction from the fused activation.
        wide = "_relu" in op
        w = numpy_helper.from_array(
            (np.random.RandomState(0).randn(256, 1024, 1, 1) * 0.05).astype("float32"),
            name="w")
        b = numpy_helper.from_array(
            (np.random.RandomState(1).randn(256) * 0.5).astype("float32"), name="b")
        nodes = [helper.make_node("Conv", ["a", "w", "b"], ["c"], kernel_shape=[1, 1],
                                  pads=[0, 0, 0, 0], strides=[1, 1])]
        if wide:
            nodes.append(helper.make_node("Relu", ["c"], ["o"]))
        else:
            nodes[0].output[0] = "o"
        m = model(nodes, [vi("a", [1, 1024, 2, 2])], [vi("o", [1, 256, 2, 2])], [w, b])
        x = [torch.rand(1, 1024, 2, 2) - 0.5]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        bt = torch.from_numpy(numpy_helper.to_array(b))
        f = lambda a: torch.nn.functional.conv2d(a, wt, bt)
        return m, x, ((lambda a: torch.relu(f(a))) if wide else f)
    if op.startswith("matmul_m4_k"):
        # The matmul the 1x1 convolution fast path performs, on its own: M=4 rows
        # (2x2 spatial, sub-tile) and a contraction dim K taken from the op name.
        # 1024->256 fails as a convolution and 256->1024 passes, and the only
        # structural difference is K, so vary K alone with no conv machinery.
        K = int(op[len("matmul_m4_k"):])
        N = 256
        m = model([helper.make_node("MatMul", ["a", "b"], ["o"])],
                  [vi("a", [1, 1, 4, K]), vi("b", [1, 1, K, N])], [vi("o", [1, 1, 4, N])])
        x = [torch.rand(1, 1, 4, K) - 0.5, torch.rand(1, 1, K, N) - 0.5]
        return m, x, lambda a, b: a @ b
    if op == "permute_in_1024":
        # conv1x1_down's input permute: [1,1024,2,2] NCHW->NHWC. permute_conv_in
        # covers the same direction at 256 channels, not 1024.
        m = model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 2, 3, 1])],
                  [vi("a", [1, 1024, 2, 2])], [vi("o", [1, 2, 2, 1024])])
        x = [torch.rand(1, 1024, 2, 2) - 0.5]
        return m, x, lambda a: a.permute(0, 2, 3, 1)
    if op == "reshape_merge_1024":
        # And its input reshape: [1,2,2,1024] -> [1,1,4,1024], the MERGE direction
        # (two tile rows into one) that was broken at 256 channels.
        sh = numpy_helper.from_array(np.array([1, 1, 4, 1024], dtype=np.int64), name="sh")
        m = model([helper.make_node("Reshape", ["a", "sh"], ["o"])],
                  [vi("a", [1, 2, 2, 1024])], [vi("o", [1, 1, 4, 1024])], [sh])
        x = [torch.rand(1, 2, 2, 1024) - 0.5]
        return m, x, lambda a: a.reshape(1, 1, 4, 1024)
    if op == "permute_reshape_1024":
        # conv1x1_down's whole input path as one graph: [1,1024,2,2] -> NHWC ->
        # [1,1,4,1024]. Each half passes on its own; conv2d's own act-vs-input
        # check cannot see corruption here, because it compares against the
        # convolution's input, which is already this pair's output.
        sh = numpy_helper.from_array(np.array([1, 1, 4, 1024], dtype=np.int64), name="sh")
        m = model([helper.make_node("Transpose", ["a"], ["t"], perm=[0, 2, 3, 1]),
                   helper.make_node("Reshape", ["t", "sh"], ["o"])],
                  [vi("a", [1, 1024, 2, 2])], [vi("o", [1, 1, 4, 1024])], [sh])
        x = [torch.rand(1, 1024, 2, 2) - 0.5]
        return m, x, lambda a: a.permute(0, 2, 3, 1).reshape(1, 1, 4, 1024)
    if op == "permute_reshape_256":
        sh = numpy_helper.from_array(np.array([1, 1, 4, 256], dtype=np.int64), name="sh")
        m = model([helper.make_node("Transpose", ["a"], ["t"], perm=[0, 2, 3, 1]),
                   helper.make_node("Reshape", ["t", "sh"], ["o"])],
                  [vi("a", [1, 256, 2, 2])], [vi("o", [1, 1, 4, 256])], [sh])
        x = [torch.rand(1, 256, 2, 2) - 0.5]
        return m, x, lambda a: a.permute(0, 2, 3, 1).reshape(1, 1, 4, 256)
    if op.startswith("relu_sp2_c"):
        # Readback of an NCHW tensor with 2x2 spatial at C channels. conv1x1_down
        # returns [1,256,2,2] and fails; conv2d_1x1_bias_sp2 returns [1,1024,2,2]
        # and passes, and by this point every op between them is verified exact,
        # so the difference has to be the shape that comes back off the device.
        C = int(op[len("relu_sp2_c"):])
        m = model([helper.make_node("Relu", ["a"], ["o"])],
                  [vi("a", [1, C, 2, 2])], [vi("o", [1, C, 2, 2])])
        x = [torch.rand(1, C, 2, 2) - 0.5]
        return m, x, lambda a: torch.relu(a)
    if op == "add_sp2_1024":
        # The residual add at layer3's dimensions: two [1,1024,2,2] tensors.
        # 2x2 spatial flattens to 4 rows -- sub-tile -- at 1024 channels.
        m = model([helper.make_node("Add", ["a", "b"], ["o"])],
                  [vi("a", [1, 1024, 2, 2]), vi("b", [1, 1024, 2, 2])],
                  [vi("o", [1, 1024, 2, 2])])
        x = [torch.rand(1, 1024, 2, 2) - 0.5, torch.rand(1, 1024, 2, 2) - 0.5]
        return m, x, lambda a, b: a + b
    if op == "conv_add_sp2_1024":
        # A 1x1 convolution whose output is added back to the convolution's own
        # input -- the residual shape of an identity block, where the skip tensor
        # has to stay live across the convolution.
        w = numpy_helper.from_array(
            (np.random.RandomState(0).randn(1024, 1024, 1, 1) * 0.03).astype("float32"),
            name="w")
        m = model([helper.make_node("Conv", ["a", "w"], ["c"], kernel_shape=[1, 1],
                                    pads=[0, 0, 0, 0], strides=[1, 1]),
                   helper.make_node("Add", ["c", "a"], ["o"])],
                  [vi("a", [1, 1024, 2, 2])], [vi("o", [1, 1024, 2, 2])], [w])
        x = [torch.rand(1, 1024, 2, 2) - 0.5]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.conv2d(a, wt) + a
    if op == "permute_conv_in":
        # conv2d_1x1_sp2's input permute, exactly as compiled: [1,256,2,2]
        # NCHW->NHWC, trailing dims 2 and 2.
        m = model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 2, 3, 1])],
                  [vi("a", [1, 256, 2, 2])], [vi("o", [1, 2, 2, 256])])
        x = [torch.rand(1, 256, 2, 2) - 0.5]
        return m, x, lambda a: a.permute(0, 2, 3, 1)
    if op == "permute_conv_out":
        # Its output permute: [1,2,2,1024] NHWC->NCHW.
        m = model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 3, 1, 2])],
                  [vi("a", [1, 2, 2, 1024])], [vi("o", [1, 1024, 2, 2])])
        x = [torch.rand(1, 2, 2, 1024) - 0.5]
        return m, x, lambda a: a.permute(0, 3, 1, 2)
    if op == "permute_pool_in":
        # The first permute in max_pool2d_resnet's graph, exactly as compiled:
        # [1,64,16,16] NCHW->NHWC. Both trailing dims are sub-tile.
        m = model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 2, 3, 1])],
                  [vi("a", [1, 64, 16, 16])], [vi("o", [1, 16, 16, 64])])
        x = [torch.rand(1, 64, 16, 16) - 0.5]
        return m, x, lambda a: a.permute(0, 2, 3, 1)
    if op == "permute_pool_out":
        # The second permute in that graph: [1,8,8,64] NHWC->NCHW.
        m = model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 3, 1, 2])],
                  [vi("a", [1, 8, 8, 64])], [vi("o", [1, 64, 8, 8])])
        x = [torch.rand(1, 8, 8, 64) - 0.5]
        return m, x, lambda a: a.permute(0, 3, 1, 2)
    if op.startswith("slice_tap"):
        # The slice the conv tap decomposition performs, on its own. conv2d_3x3_sp2
        # pads [1,256,2,2] to [1,4,4,256] in NHWC and takes one 2x2 window per kernel
        # tap: tap (0,0) starts at 0 on both spatial axes, the other eight taps start
        # at 1 or 2. A tap path that is right only where the start is 0 scores about
        # 1/9 of scale, which is what the emulator reports.
        off = int(op[len("slice_tap"):][0])
        # Forge rejects a strided slice on more than one axis, so cut one spatial
        # axis at a time; the offset is what is under test, not the rank.
        st = numpy_helper.from_array(np.array([off], dtype=np.int64), name="st")
        en = numpy_helper.from_array(np.array([off + 2], dtype=np.int64), name="en")
        ax = numpy_helper.from_array(np.array([1], dtype=np.int64), name="ax")
        sp = numpy_helper.from_array(np.array([1], dtype=np.int64), name="sp")
        m = model([helper.make_node("Slice", ["a", "st", "en", "ax", "sp"], ["o"])],
                  [vi("a", [1, 4, 4, 256])], [vi("o", [1, 2, 4, 256])], [st, en, ax, sp])
        x = [torch.rand(1, 4, 4, 256) - 0.5]
        return m, x, lambda a: a[:, off:off + 2, :, :]
    if op == "conv2d_3x3_sp2":
        # layer3's 3x3 convolutions: 2x2 spatial, where the 3x3 kernel with pad 1 is
        # larger than the feature map.
        w = numpy_helper.from_array(
            (np.random.RandomState(0).randn(256, 256, 3, 3) * 0.05).astype("float32"),
            name="w")
        m = model([helper.make_node("Conv", ["a", "w"], ["o"], kernel_shape=[3, 3],
                                    pads=[1, 1, 1, 1], strides=[1, 1])],
                  [vi("a", [1, 256, 2, 2])], [vi("o", [1, 256, 2, 2])], [w])
        x = [torch.rand(1, 256, 2, 2) - 0.5]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.conv2d(a, wt, padding=1)
    if op == "conv2d_1x1_expand":
        # Exactly the identity block's conv3: 1x1, C_in=64 -> C_out=256 at 8x8, no
        # activation, no bias, signed weights. In the block this op is 83% wrong
        # while conv1 and conv2 beside it are correct; run it alone to find out
        # whether the shape or the surrounding graph is responsible.
        w = numpy_helper.from_array(
            (np.random.RandomState(0).randn(256, 64, 1, 1) * 0.05).astype("float32"),
            name="w")
        m = model([helper.make_node("Conv", ["a", "w"], ["o"], kernel_shape=[1, 1],
                                    pads=[0, 0, 0, 0], strides=[1, 1])],
                  [vi("a", [1, 64, 8, 8])], [vi("o", [1, 256, 8, 8])], [w])
        x = [torch.rand(1, 64, 8, 8) - 0.5]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.conv2d(a, wt)
    if op == "conv2d_1x1_reduce":
        # The identity block's first convolution: 1x1 with C_in=256 -> C_out=64, a
        # channel REDUCTION (matmul K=256, N=64). Every 1x1 probe so far expanded
        # channels or kept them equal, so this contraction width is untested.
        w = numpy_helper.from_array(
            (np.random.RandomState(0).randn(64, 256, 1, 1) * 0.05).astype("float32"),
            name="w")
        m = model([helper.make_node("Conv", ["a", "w"], ["o"], kernel_shape=[1, 1],
                                    pads=[0, 0, 0, 0], strides=[1, 1])],
                  [vi("a", [1, 256, 8, 8])], [vi("o", [1, 64, 8, 8])], [w])
        x = [torch.rand(1, 256, 8, 8) - 0.5]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.conv2d(a, wt)
    if op == "conv2d_1x1_reduce_relu":
        # Same, with the relu the real block fuses into it.
        w = numpy_helper.from_array(
            (np.random.RandomState(0).randn(64, 256, 1, 1) * 0.05).astype("float32"),
            name="w")
        nodes = [helper.make_node("Conv", ["a", "w"], ["c"], kernel_shape=[1, 1],
                                  pads=[0, 0, 0, 0], strides=[1, 1]),
                 helper.make_node("Relu", ["c"], ["o"])]
        m = model(nodes, [vi("a", [1, 256, 8, 8])], [vi("o", [1, 64, 8, 8])], [w])
        x = [torch.rand(1, 256, 8, 8) - 0.5]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.relu(
            torch.nn.functional.conv2d(a, wt))
    if op == "conv2d_1x1_sp2":
        # The 1x1 expansion at 2x2 spatial, through the fast path.
        w = numpy_helper.from_array(
            (np.random.RandomState(0).randn(1024, 256, 1, 1) * 0.05).astype("float32"),
            name="w")
        m = model([helper.make_node("Conv", ["a", "w"], ["o"], kernel_shape=[1, 1],
                                    pads=[0, 0, 0, 0], strides=[1, 1])],
                  [vi("a", [1, 256, 2, 2])], [vi("o", [1, 1024, 2, 2])], [w])
        x = [torch.rand(1, 256, 2, 2) - 0.5]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.conv2d(a, wt)
    if op == "conv2d_3x3_s1_16":
        # Same shape as conv2d_3x3_s2 but stride 1 in the graph. conv2d_32 (3x3 stride 1)
        # passes at 32x32 spatial; if this hangs at 16x16 then the hang is the shape, not
        # the stride, and running strided convs at stride 1 cannot help them.
        w = numpy_helper.from_array(
            np.random.RandomState(0).rand(32, 32, 3, 3).astype("float32") * 0.1, name="w")
        m = model([helper.make_node("Conv", ["a", "w"], ["o"], kernel_shape=[3, 3],
                                    pads=[1, 1, 1, 1], strides=[1, 1])],
                  [vi("a", [1, 32, 16, 16])], [vi("o", [1, 32, 16, 16])], [w])
        x = [torch.rand(1, 32, 16, 16)]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.conv2d(a, wt, padding=1)
    if op == "conv2d_fused_relu":
        # Conv immediately followed by Relu: the frontend fuses the relu INTO the
        # conv (Conv2dConfig::activation), which is why ResNet-50's stem graph has a
        # conv2d and no ttnn.relu. A bare Conv node has nothing to fuse, so no other
        # probe here exercises the fused-activation path.
        # SIGNED weights and inputs on purpose. Every other probe in this file uses
        # rand()*0.1 weights with rand() inputs -- all strictly positive, so every
        # conv output is positive and relu is an identity. That made the fused
        # activation untestable: resnet_block had 2 of its 3 relus fused into convs
        # and still scored 0.999934 while the runtime was dropping them entirely.
        # Real ResNet-50 uses Kaiming init, so about half the outputs are negative.
        w = numpy_helper.from_array(
            (np.random.RandomState(0).randn(64, 32, 3, 3) * 0.1).astype("float32"), name="w")
        nodes = [helper.make_node("Conv", ["a", "w"], ["c"], kernel_shape=[3, 3],
                                  pads=[1, 1, 1, 1], strides=[1, 1]),
                 helper.make_node("Relu", ["c"], ["o"])]
        m = model(nodes, [vi("a", [1, 32, 16, 16])], [vi("o", [1, 64, 16, 16])], [w])
        x = [torch.rand(1, 32, 16, 16) - 0.5]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.relu(
            torch.nn.functional.conv2d(a, wt, padding=1))
    if op == "conv2d_1x1_fused_relu":
        # Same, through the 1x1 fast path rather than the tap decomposition.
        # Signed, for the same reason as conv2d_fused_relu above.
        w = numpy_helper.from_array(
            (np.random.RandomState(0).randn(64, 32, 1, 1) * 0.1).astype("float32"), name="w")
        nodes = [helper.make_node("Conv", ["a", "w"], ["c"], kernel_shape=[1, 1],
                                  pads=[0, 0, 0, 0], strides=[1, 1]),
                 helper.make_node("Relu", ["c"], ["o"])]
        m = model(nodes, [vi("a", [1, 32, 16, 16])], [vi("o", [1, 64, 16, 16])], [w])
        x = [torch.rand(1, 32, 16, 16) - 0.5]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.relu(
            torch.nn.functional.conv2d(a, wt))
    if op == "conv2d_1x1":
        # 1x1, no padding: no halo gather, and the activation needs no spatial
        # rearrangement, so the misaligned staged-read branch should not be taken.
        # Isolates "is the conv pipeline correct" from "is the misaligned path correct".
        w = numpy_helper.from_array(
            np.random.RandomState(0).rand(32, 32, 1, 1).astype("float32") * 0.1, name="w")
        m = model([helper.make_node("Conv", ["a", "w"], ["o"], kernel_shape=[1, 1],
                                    pads=[0, 0, 0, 0], strides=[1, 1])],
                  [vi("a", [1, 32, 32, 32])], [vi("o", [1, 32, 32, 32])], [w])
        x = [torch.rand(1, 32, 32, 32)]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.conv2d(a, wt)
    if op == "conv2d_1x1_asym":
        # Deliberately asymmetric: C_in=32, C_out=64, H=16, W=8 all differ, so an axis
        # permutation is unambiguous (unlike C==H==W==32, where many coincide).
        w = numpy_helper.from_array(
            np.random.RandomState(0).rand(64, 32, 1, 1).astype("float32") * 0.1, name="w")
        m = model([helper.make_node("Conv", ["a", "w"], ["o"], kernel_shape=[1, 1],
                                    pads=[0, 0, 0, 0], strides=[1, 1])],
                  [vi("a", [1, 32, 16, 8])], [vi("o", [1, 64, 16, 8])], [w])
        x = [torch.rand(1, 32, 16, 8)]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.conv2d(a, wt)
    if op == "permute_256_inv_reshaped":
        # The same failing permute, but with the result reshaped to a tile-aligned
        # shape before it leaves the graph. If this passes while permute_256_inv
        # fails, the permute is correct (as its per-swap CPU check already says) and
        # the fault is in un-padding a [1,256,8,8] output -- sub-tile trailing dims
        # with more than 64 tiles.
        sh = numpy_helper.from_array(np.array([1, 1, 256, 64], dtype=np.int64), name="sh")
        nodes = [helper.make_node("Transpose", ["a"], ["t"], perm=[0, 3, 1, 2]),
                 helper.make_node("Reshape", ["t", "sh"], ["o"])]
        m = model(nodes, [vi("a", [1, 8, 8, 256])], [vi("o", [1, 1, 256, 64])], [sh])
        x = [torch.rand(1, 8, 8, 256) - 0.5]
        return m, x, lambda a: a.permute(0, 3, 1, 2).reshape(1, 1, 256, 64)
    if op == "relu_nchw_96":
        # Locate the threshold between 64 (passes) and 128 (fails).
        m = model([helper.make_node("Relu", ["a"], ["o"])],
                  [vi("a", [1, 96, 8, 8])], [vi("o", [1, 96, 8, 8])])
        x = [torch.rand(1, 96, 8, 8) - 0.5]
        return m, x, lambda a: torch.relu(a)
    if op == "permute_256_fwd":
        # The NCHW->NHWC permute around a 256-channel ResNet tensor: [1,256,8,8] ->
        # [1,8,8,256]. Both spatial extents are 8 (sub-tile) and the channel count is
        # 256 (8 tiles). The verified subtile probes used [1,32,16,8] and [1,16,8,64].
        m = model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 2, 3, 1])],
                  [vi("a", [1, 256, 8, 8])], [vi("o", [1, 8, 8, 256])])
        x = [torch.rand(1, 256, 8, 8) - 0.5]
        return m, x, lambda a: a.permute(0, 2, 3, 1)
    if op == "permute_256_inv":
        # And the inverse, [1,8,8,256] -> [1,256,8,8].
        m = model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 3, 1, 2])],
                  [vi("a", [1, 8, 8, 256])], [vi("o", [1, 256, 8, 8])])
        x = [torch.rand(1, 8, 8, 256) - 0.5]
        return m, x, lambda a: a.permute(0, 3, 1, 2)
    if op == "reshape_256_merge":
        # The reshape that pairs with them: [1,8,8,256] -> [1,1,64,256].
        sh = numpy_helper.from_array(np.array([1, 1, 64, 256], dtype=np.int64), name="sh")
        m = model([helper.make_node("Reshape", ["a", "sh"], ["o"])],
                  [vi("a", [1, 8, 8, 256])], [vi("o", [1, 1, 64, 256])], [sh])
        x = [torch.rand(1, 8, 8, 256) - 0.5]
        return m, x, lambda a: a.reshape(1, 1, 64, 256)
    if op == "permute_via_reshape_wh":
        # The NCHW->NHWC permute expressed WITHOUT an HC (axes 1,2) swap: merge the
        # spatial axes, swap the last two, split again. Identity:
        #   [N,C,H,W] -reshape-> [N,1,C,H*W] -WH-> [N,1,H*W,C] -reshape-> [N,H,W,C]
        # Both reshapes are pure row-major regroupings of adjacent axes. WH is the one
        # transpose verified correct on the emulator (transpose_2d 0.999996), while the
        # HC swap the normal decomposition uses is not.
        s1 = numpy_helper.from_array(np.array([1, 1, 32, 128], dtype=np.int64), name="s1")
        s2 = numpy_helper.from_array(np.array([1, 16, 8, 32], dtype=np.int64), name="s2")
        nodes = [helper.make_node("Reshape", ["a", "s1"], ["m"]),
                 helper.make_node("Transpose", ["m"], ["t"], perm=[0, 1, 3, 2]),
                 helper.make_node("Reshape", ["t", "s2"], ["o"])]
        m = model(nodes, [vi("a", [1, 32, 16, 8])], [vi("o", [1, 16, 8, 32])], [s1, s2])
        x = [torch.rand(1, 32, 16, 8) - 0.5]
        return m, x, lambda a: a.permute(0, 2, 3, 1)
    if op == "permute_nchw2nhwc_subtile":
        # The permute the conv2d lowering inserts ahead of the conv for
        # conv2d_1x1_asym: NCHW [1,32,16,8] -> NHWC [1,16,8,32]. Both moved dims are
        # SUB-TILE (16 and 8 < 32), unlike permute_generic's [1,2,32,64], so the
        # padded-tile handling is exercised for the first time.
        m = model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 2, 3, 1])],
                  [vi("a", [1, 32, 16, 8])], [vi("o", [1, 16, 8, 32])])
        x = [torch.rand(1, 32, 16, 8)]
        return m, x, lambda a: a.permute(0, 2, 3, 1)
    if op == "permute_nhwc2nchw_subtile":
        # The inverse permute the lowering inserts after the conv:
        # NHWC [1,16,8,64] -> NCHW [1,64,16,8].
        m = model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 3, 1, 2])],
                  [vi("a", [1, 16, 8, 64])], [vi("o", [1, 64, 16, 8])])
        x = [torch.rand(1, 16, 8, 64)]
        return m, x, lambda a: a.permute(0, 3, 1, 2)
    if op == "reshape_subtile_merge":
        # The reshape the conv2d lowering inserts BEFORE the conv: the channel-last
        # activation [1,16,8,32] flattened to [1,1,128,32]. Dim 8 is sub-tile, so in
        # TILE layout the source pads 8 rows out to 32 and the merge has to re-pack.
        # This is the only step left that changes how the matmul groups elements
        # without changing the values themselves.
        s = numpy_helper.from_array(np.array([1, 1, 128, 32], dtype=np.int64), name="s")
        m = model([helper.make_node("Reshape", ["a", "s"], ["o"])],
                  [vi("a", [1, 16, 8, 32])], [vi("o", [1, 1, 128, 32])], [s])
        x = [torch.rand(1, 16, 8, 32)]
        return m, x, lambda a: a.reshape(1, 1, 128, 32)
    if op == "reshape_subtile_split":
        # The reshape the lowering inserts around the conv: the flattened
        # channel-last activation [1,1,128,32] split back to [1,16,8,32].
        s = numpy_helper.from_array(np.array([1, 16, 8, 32], dtype=np.int64), name="s")
        m = model([helper.make_node("Reshape", ["a", "s"], ["o"])],
                  [vi("a", [1, 1, 128, 32])], [vi("o", [1, 16, 8, 32])], [s])
        x = [torch.rand(1, 1, 128, 32)]
        return m, x, lambda a: a.reshape(1, 16, 8, 32)
    if op == "transpose_2d_asym":
        # NON-SQUARE tiled transpose [1,1,64,32] -> [1,1,32,64]: 2 tile-rows by 1
        # tile-col becoming 1 by 2. Exactly what the 1x1 conv decomposition does to the
        # weight, and a different code path from the square 32x32 case already verified.
        m = model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 1, 3, 2])],
                  [vi("a", [1, 1, 64, 32])], [vi("o", [1, 1, 32, 64])])
        x = [torch.rand(1, 1, 64, 32)]
        return m, x, lambda a: a.permute(0, 1, 3, 2)
    if op == "matmul_composed":
        # Mirrors the C++ 1x1 decomposition exactly, but expressed in ONNX so it runs
        # through the ordinary (already verified) runtime op path: weight [64,32],
        # Transpose -> [32,64], MatMul with act [1,128,32] -> [1,128,64].
        # If this passes, the ops compose correctly and the fault is in the C++ path.
        wv = np.random.RandomState(0).rand(64, 32).astype("float32") * 0.1
        w = numpy_helper.from_array(wv, name="w")
        nodes = [helper.make_node("Transpose", ["w"], ["wt"], perm=[1, 0]),
                 helper.make_node("MatMul", ["a", "wt"], ["o"])]
        m = model(nodes, [vi("a", [1, 128, 32])], [vi("o", [1, 128, 64])], [w])
        x = [torch.rand(1, 128, 32)]
        W = torch.from_numpy(wv)
        return m, x, lambda a: a @ W.t()
    if op == "resnet_block_identity":
        # ResNet-50's blocks 2 and 3 of every stage: the skip is the block INPUT
        # itself, with no projection convolution. That is the one structure the
        # projection-skip probe does not cover, and it means the input tensor is
        # live across the whole block -- so it is where an aliasing or
        # premature-deallocate bug in a reimplemented conv would show up.
        rs = np.random.RandomState(0)
        ws = [numpy_helper.from_array((rs.randn(*shp) * 0.05).astype("float32"), name=n)
              for n, shp in [("w1", (64, 256, 1, 1)), ("w2", (64, 64, 3, 3)),
                             ("w3", (256, 64, 1, 1))]]
        nodes = [
            helper.make_node("Conv", ["a", "w1"], ["c1"], kernel_shape=[1, 1],
                             pads=[0, 0, 0, 0], strides=[1, 1]),
            helper.make_node("Relu", ["c1"], ["r1"]),
            helper.make_node("Conv", ["r1", "w2"], ["c2"], kernel_shape=[3, 3],
                             pads=[1, 1, 1, 1], strides=[1, 1]),
            helper.make_node("Relu", ["c2"], ["r2"]),
            helper.make_node("Conv", ["r2", "w3"], ["c3"], kernel_shape=[1, 1],
                             pads=[0, 0, 0, 0], strides=[1, 1]),
            helper.make_node("Add", ["c3", "a"], ["sum"]),
            helper.make_node("Relu", ["sum"], ["o"]),
        ]
        m = model(nodes, [vi("a", [1, 256, 8, 8])], [vi("o", [1, 256, 8, 8])], ws)
        x = [torch.rand(1, 256, 8, 8) - 0.5]
        W = [torch.from_numpy(numpy_helper.to_array(w)) for w in ws]
        def ref(a):
            F = torch.nn.functional
            y = F.relu(F.conv2d(a, W[0]))
            y = F.relu(F.conv2d(y, W[1], padding=1))
            y = F.conv2d(y, W[2])
            return F.relu(y + a)
        return m, x, ref
    if op == "resnet_block":
        # A real ResNet-50 bottleneck: 1x1 -> relu -> 3x3 pad 1 -> relu -> 1x1, plus a
        # 1x1 projection on the skip, then add -> relu. Every op kind the residual
        # stages use, in the order the real graph uses them. Single ops passing does
        # not prove they compose -- the conv2d fault fixed yesterday lived in a
        # neighbouring permute, not in conv2d.
        # SIGNED weights and ResNet-50's real layer1 channel counts
        # (64 -> 64 -> 64 -> 256, projection 64 -> 256). The earlier version used
        # all-positive weights and C_out=64, which made the fused relus no-ops and
        # the block trivially correct: it scored 0.999934 while the runtime was
        # dropping 2 of its 3 relus.
        rs = np.random.RandomState(0)
        ws = [numpy_helper.from_array((rs.randn(*shp) * 0.05).astype("float32"), name=n)
              for n, shp in [("w1", (64, 64, 1, 1)), ("w2", (64, 64, 3, 3)),
                             ("w3", (256, 64, 1, 1)), ("wd", (256, 64, 1, 1))]]
        nodes = [
            helper.make_node("Conv", ["a", "w1"], ["c1"], kernel_shape=[1, 1],
                             pads=[0, 0, 0, 0], strides=[1, 1]),
            helper.make_node("Relu", ["c1"], ["r1"]),
            helper.make_node("Conv", ["r1", "w2"], ["c2"], kernel_shape=[3, 3],
                             pads=[1, 1, 1, 1], strides=[1, 1]),
            helper.make_node("Relu", ["c2"], ["r2"]),
            helper.make_node("Conv", ["r2", "w3"], ["c3"], kernel_shape=[1, 1],
                             pads=[0, 0, 0, 0], strides=[1, 1]),
            helper.make_node("Conv", ["a", "wd"], ["d"], kernel_shape=[1, 1],
                             pads=[0, 0, 0, 0], strides=[1, 1]),
            helper.make_node("Add", ["c3", "d"], ["sum"]),
            helper.make_node("Relu", ["sum"], ["o"]),
        ]
        m = model(nodes, [vi("a", [1, 64, 8, 8])], [vi("o", [1, 256, 8, 8])], ws)
        x = [torch.rand(1, 64, 8, 8) - 0.5]
        W = [torch.from_numpy(numpy_helper.to_array(w)) for w in ws]
        def ref(a):
            F = torch.nn.functional
            y = F.relu(F.conv2d(a, W[0]))
            y = F.relu(F.conv2d(y, W[1], padding=1))
            y = F.conv2d(y, W[2])
            return F.relu(y + F.conv2d(a, W[3]))
        return m, x, ref
    if op == "resnet_head":
        # ResNet-50's head: global average pool over the spatial axes then the fc layer.
        # Exercises mean and linear together in the order the real graph uses them.
        rs = np.random.RandomState(0)
        wfc = numpy_helper.from_array(rs.rand(64, 16).astype("float32") * 0.1, name="wfc")
        # A leading 1x1 conv on purpose: in the real graph the pool follows a conv, so
        # its input is already channel-last and the mean lowers to dim -2 (the H
        # reduce). Pooling straight off the model input instead makes the frontend
        # emit [1,1,C,H*W] with the reduction on the last axis -- the W reduce, whose
        # Quasar program factory is unported, and which ResNet-50 never uses.
        wc = numpy_helper.from_array(
            rs.rand(64, 64, 1, 1).astype("float32") * 0.1, name="wc")
        sh = numpy_helper.from_array(np.array([1, 64], dtype=np.int64), name="sh")
        nodes = [helper.make_node("Conv", ["a", "wc"], ["c"], kernel_shape=[1, 1],
                                  pads=[0, 0, 0, 0], strides=[1, 1]),
                 helper.make_node("GlobalAveragePool", ["c"], ["p"]),
                 helper.make_node("Reshape", ["p", "sh"], ["f"]),
                 helper.make_node("MatMul", ["f", "wfc"], ["o"])]
        m = model(nodes, [vi("a", [1, 64, 7, 7])], [vi("o", [1, 16])], [wfc, wc, sh])
        x = [torch.rand(1, 64, 7, 7)]
        Wfc = torch.from_numpy(numpy_helper.to_array(wfc))
        Wc = torch.from_numpy(numpy_helper.to_array(wc))
        def ref(a):
            c = torch.nn.functional.conv2d(a, Wc)
            p = torch.nn.functional.adaptive_avg_pool2d(c, 1).reshape(1, 64)
            return p @ Wfc
        return m, x, ref
    if op == "mean_h_axis":
        # ResNet-50's mean exactly: a rank-4 channel-last-flattened tensor [1,1,H*W,C]
        # reduced over dim -2 (the spatial axis), keepdims. That selects the H reduce,
        # whose Quasar program factory IS ported -- unlike the W reduce, which my
        # GlobalAveragePool probes hit because the frontend put C before H*W there.
        # ResNet's real op is [1,1,49,2048]; 64 channels here to keep it quick.
        ax = numpy_helper.from_array(np.array([-2], dtype=np.int64), name="ax")
        m = model([helper.make_node("ReduceMean", ["a", "ax"], ["o"], keepdims=1)],
                  [vi("a", [1, 1, 49, 64])], [vi("o", [1, 1, 1, 64])], [ax])
        x = [torch.rand(1, 1, 49, 64)]
        return m, x, lambda a: a.mean(dim=-2, keepdim=True)
    if op == "mean_h_axis_2048":
        # Same reduction at ResNet's real channel count.
        ax = numpy_helper.from_array(np.array([-2], dtype=np.int64), name="ax")
        m = model([helper.make_node("ReduceMean", ["a", "ax"], ["o"], keepdims=1)],
                  [vi("a", [1, 1, 49, 2048])], [vi("o", [1, 1, 1, 2048])], [ax])
        x = [torch.rand(1, 1, 49, 2048)]
        return m, x, lambda a: a.mean(dim=-2, keepdim=True)
    if op == "mean_global":
        # ResNet-50's global average pool: reduce H and W, keep dims. Lowers to ttnn.mean.
        m = model([helper.make_node("GlobalAveragePool", ["a"], ["o"])],
                  [vi("a", [1, 64, 8, 8])], [vi("o", [1, 64, 1, 1])])
        x = [torch.rand(1, 64, 8, 8)]
        return m, x, lambda a: torch.nn.functional.adaptive_avg_pool2d(a, 1)
    if op == "mean_hw_7x7":
        # Same reduction at ResNet's real final spatial size, where H=W=7 is sub-tile.
        m = model([helper.make_node("GlobalAveragePool", ["a"], ["o"])],
                  [vi("a", [1, 64, 7, 7])], [vi("o", [1, 64, 1, 1])])
        x = [torch.rand(1, 64, 7, 7)]
        return m, x, lambda a: torch.nn.functional.adaptive_avg_pool2d(a, 1)
    if op == "typecast":
        # ttnn.typecast appears twice in the ResNet-50 graph.
        nodes = [helper.make_node("Cast", ["a"], ["c"], to=TensorProto.DOUBLE),
                 helper.make_node("Cast", ["c"], ["o"], to=TensorProto.FLOAT)]
        m = model(nodes, [vi("a", [1, 1, 32, 64])], [vi("o", [1, 1, 32, 64])])
        x = [torch.rand(1, 1, 32, 64)]
        return m, x, lambda a: a
    if op == "conv2d_1x1_s2":
        # ResNet-50 downsample convs: 1x1 with stride 2. The C++ 1x1 fast path gates on
        # stride == 1, so these go through Quasar's own conv2d.
        w = numpy_helper.from_array(
            np.random.RandomState(0).rand(64, 32, 1, 1).astype("float32") * 0.1, name="w")
        m = model([helper.make_node("Conv", ["a", "w"], ["o"], kernel_shape=[1, 1],
                                    pads=[0, 0, 0, 0], strides=[2, 2])],
                  [vi("a", [1, 32, 16, 16])], [vi("o", [1, 64, 8, 8])], [w])
        x = [torch.rand(1, 32, 16, 16)]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.conv2d(a, wt, stride=2)
    if op == "conv2d_3x3_s2":
        # The stride-2 3x3 at the start of each ResNet stage.
        w = numpy_helper.from_array(
            np.random.RandomState(0).rand(32, 32, 3, 3).astype("float32") * 0.1, name="w")
        m = model([helper.make_node("Conv", ["a", "w"], ["o"], kernel_shape=[3, 3],
                                    pads=[1, 1, 1, 1], strides=[2, 2])],
                  [vi("a", [1, 32, 16, 16])], [vi("o", [1, 32, 8, 8])], [w])
        x = [torch.rand(1, 32, 16, 16)]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.conv2d(a, wt, stride=2, padding=1)
    if op == "conv2d_stem":
        # ResNet-50's first conv: 7x7 stride 2 pad 3 with C_in=3 (NOT a multiple of 32).
        # Small spatial so it finishes in the simulator; the real one is 224x224.
        w = numpy_helper.from_array(
            np.random.RandomState(0).rand(64, 3, 7, 7).astype("float32") * 0.1, name="w")
        m = model([helper.make_node("Conv", ["a", "w"], ["o"], kernel_shape=[7, 7],
                                    pads=[3, 3, 3, 3], strides=[2, 2])],
                  [vi("a", [1, 3, 32, 32])], [vi("o", [1, 64, 16, 16])], [w])
        x = [torch.rand(1, 3, 32, 32)]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.conv2d(a, wt, stride=2, padding=3)
    if op == "conv2d_1x1_7x7spatial":
        # ResNet's last stage runs at 7x7 spatial -- sub-tile in both H and W, the shape
        # class that produced the padding-repack permute bug.
        w = numpy_helper.from_array(
            np.random.RandomState(0).rand(64, 32, 1, 1).astype("float32") * 0.1, name="w")
        m = model([helper.make_node("Conv", ["a", "w"], ["o"], kernel_shape=[1, 1],
                                    pads=[0, 0, 0, 0], strides=[1, 1])],
                  [vi("a", [1, 32, 7, 7])], [vi("o", [1, 64, 7, 7])], [w])
        x = [torch.rand(1, 32, 7, 7)]
        wt = torch.from_numpy(numpy_helper.to_array(w))
        return m, x, lambda a: torch.nn.functional.conv2d(a, wt)
    if op == "max_pool2d_resnet":
        # ResNet-50's actual maxpool: 3x3, stride 2, padding 1, C=64 (channel-aligned).
        # Real spatial is 112x112; 16x16 here so it finishes in the simulator.
        m = model([helper.make_node("MaxPool", ["a"], ["o"], kernel_shape=[3, 3],
                                    pads=[1, 1, 1, 1], strides=[2, 2])],
                  [vi("a", [1, 64, 16, 16])], [vi("o", [1, 64, 8, 8])])
        x = [torch.rand(1, 64, 16, 16)]
        return m, x, lambda a: torch.nn.functional.max_pool2d(a, 3, 2, padding=1)
    if op == "max_pool2d_32":
        # The original max_pool2d probe used C=3, which is not a multiple of 32 -- the
        # same unaligned-channel condition that breaks conv. Same geometry, C=32, to
        # separate "pool is broken" from "unaligned channels are broken".
        m = model([helper.make_node("MaxPool", ["a"], ["o"], kernel_shape=[2, 2],
                                    strides=[2, 2])],
                  [vi("a", [1, 32, 32, 32])], [vi("o", [1, 32, 16, 16])])
        x = [torch.rand(1, 32, 32, 32)]
        return m, x, lambda a: torch.nn.functional.max_pool2d(a, 2, 2)
    if op == "max_pool2d":
        m = model([helper.make_node("MaxPool", ["a"], ["o"], kernel_shape=[2, 2],
                                    strides=[2, 2])],
                  [vi("a", [1, 3, 32, 32])], [vi("o", [1, 3, 16, 16])])
        x = [torch.rand(1, 3, 32, 32)]
        return m, x, lambda a: torch.nn.functional.max_pool2d(a, 2, 2)
    if op == "linear":
        rs = np.random.RandomState(0)
        wb = numpy_helper.from_array(rs.rand(64, 32).astype("float32"), name="w")
        cb = numpy_helper.from_array(rs.rand(32).astype("float32"), name="c")
        m = model([helper.make_node("Gemm", ["a", "w", "c"], ["o"])],
                  [vi("a", [1, 64])], [vi("o", [1, 32])], [wb, cb])
        x = [torch.rand(1, 64)]
        W = torch.from_numpy(numpy_helper.to_array(wb))
        C = torch.from_numpy(numpy_helper.to_array(cb))
        return m, x, lambda a: a @ W + C
    if op == "matmul_1tile":
        # Single tile per operand: in0_block_w == 1, one subblock. Isolates the matmul
        # block/subblock loop logic from the underlying matmul primitives.
        m = model([helper.make_node("MatMul", ["a", "b"], ["o"])],
                  [vi("a", [1, 32, 32]), vi("b", [1, 32, 32])], [vi("o", [1, 32, 32])])
        x = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]
        return m, x, lambda a, b: a @ b
    if op == "transpose_wh_32x128":
        # The exact WH transpose the rotation route performs for [0,2,3,1]:
        # merged [1,32,128] -> [1,128,32]. transpose_2d only covers ONE tile
        # (32x32), so it says nothing about this.
        m = model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 1, 3, 2])],
                  [vi("a", [1, 1, 32, 128])], [vi("o", [1, 1, 128, 32])])
        x = [torch.rand(1, 1, 32, 128) - 0.5]
        return m, x, lambda a: a.permute(0, 1, 3, 2)
    if op == "transpose_wh_128x64":
        # The WH transpose the rotation route performs for [0,3,1,2], which passes
        # on the emulator. Same op, different extents.
        m = model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 1, 3, 2])],
                  [vi("a", [1, 1, 128, 64])], [vi("o", [1, 1, 64, 128])])
        x = [torch.rand(1, 1, 128, 64) - 0.5]
        return m, x, lambda a: a.permute(0, 1, 3, 2)
    if op == "transpose_2d":
        # Swap the last two dims of a tile-shaped 4-D tensor -- the exact operation the
        # conv 1x1 decomposition needs for its weight, and the suspected scrambler.
        m = model([helper.make_node("Transpose", ["a"], ["o"], perm=[0, 1, 3, 2])],
                  [vi("a", [1, 1, 32, 32])], [vi("o", [1, 1, 32, 32])])
        x = [torch.rand(1, 1, 32, 32)]
        return m, x, lambda a: a.permute(0, 1, 3, 2)
    if op == "relu_tiles_64":
        # Rank-4 with leading 1s, so the frontend inserts no permute (see
        # relu_4d_256). [1,1,256,256] is 8x8 = 64 tiles.
        m = model([helper.make_node("Relu", ["a"], ["o"])],
                  [vi("a", [1, 1, 256, 256])], [vi("o", [1, 1, 256, 256])])
        x = [torch.rand(1, 1, 256, 256) - 0.5]
        return m, x, lambda a: torch.relu(a)
    if op == "relu_tiles_128":
        # [1,1,512,256] is 16x8 = 128 tiles: same op, same shape class, twice the
        # tiles. If 64 passes and 128 fails with no permute anywhere, the fault is a
        # size limit in the op or the read-back, not a layout bug.
        m = model([helper.make_node("Relu", ["a"], ["o"])],
                  [vi("a", [1, 1, 512, 256])], [vi("o", [1, 1, 512, 256])])
        x = [torch.rand(1, 1, 512, 256) - 0.5]
        return m, x, lambda a: torch.relu(a)
    if op == "relu_nchw_64":
        # Same op and same 8x8 spatial as relu_nchw_256, fewer channels. If this
        # passes and the 256-channel one fails, the fault tracks the channel count
        # (i.e. the number of padded tiles), not the sub-tile spatial extent.
        m = model([helper.make_node("Relu", ["a"], ["o"])],
                  [vi("a", [1, 64, 8, 8])], [vi("o", [1, 64, 8, 8])])
        x = [torch.rand(1, 64, 8, 8) - 0.5]
        return m, x, lambda a: torch.relu(a)
    if op == "relu_nchw_128":
        m = model([helper.make_node("Relu", ["a"], ["o"])],
                  [vi("a", [1, 128, 8, 8])], [vi("o", [1, 128, 8, 8])])
        x = [torch.rand(1, 128, 8, 8) - 0.5]
        return m, x, lambda a: torch.relu(a)
    if op == "relu_nchw_256_16sp":
        # 256 channels but 16x16 spatial, to separate channel count from spatial extent.
        m = model([helper.make_node("Relu", ["a"], ["o"])],
                  [vi("a", [1, 256, 16, 16])], [vi("o", [1, 256, 16, 16])])
        x = [torch.rand(1, 256, 16, 16) - 0.5]
        return m, x, lambda a: torch.relu(a)
    if op == "relu_4d_256":
        # The relu that hangs in a real ResNet-50 bottleneck is the post-residual one,
        # on the block output: rank 4, 256 channels. The passing `relu` probe is rank 3
        # [1,64,32], which is a different shape class entirely.
        m = model([helper.make_node("Relu", ["a"], ["o"])],
                  [vi("a", [1, 1, 64, 256])], [vi("o", [1, 1, 64, 256])])
        x = [torch.rand(1, 1, 64, 256) - 0.5]
        return m, x, lambda a: torch.relu(a)
    if op == "relu_nchw_256":
        # Same tensor in NCHW, so the frontend inserts its permute/reshape around it --
        # the form the real graph actually has.
        m = model([helper.make_node("Relu", ["a"], ["o"])],
                  [vi("a", [1, 256, 8, 8])], [vi("o", [1, 256, 8, 8])])
        x = [torch.rand(1, 256, 8, 8) - 0.5]
        return m, x, lambda a: torch.relu(a)
    if op == "add_relu_256":
        # Residual add followed by relu at the real shape: the exact tail of a
        # bottleneck block, where l1b0 hangs.
        nodes = [helper.make_node("Add", ["a", "b"], ["s"]),
                 helper.make_node("Relu", ["s"], ["o"])]
        m = model(nodes, [vi("a", [1, 256, 8, 8]), vi("b", [1, 256, 8, 8])],
                  [vi("o", [1, 256, 8, 8])])
        x = [torch.rand(1, 256, 8, 8) - 0.5, torch.rand(1, 256, 8, 8) - 0.5]
        return m, x, lambda a, b: torch.relu(a + b)
    if op == "relu":
        m = model([helper.make_node("Relu", ["a"], ["o"])],
                  [vi("a", [1, 64, 32])], [vi("o", [1, 64, 32])])
        x = [torch.rand(1, 64, 32) - 0.5]
        return m, x, lambda a: torch.relu(a)
    if op == "multiply":
        m = model([helper.make_node("Mul", ["a", "b"], ["o"])],
                  [vi("a", [1, 64, 32]), vi("b", [1, 64, 32])], [vi("o", [1, 64, 32])])
        x = [torch.rand(1, 64, 32), torch.rand(1, 64, 32)]
        return m, x, lambda a, b: a * b
    raise SystemExit(f"unknown op {op}")

m, inputs, ref = case(OP)

d = TTSystem.get_system().devices
print(f"[{OP}] arch: {d[0].arch if d else 'no device'}", flush=True)

# PROBE_OPT=1 selects the BENCHMARK-optimised pipeline -- the same config as
# dump_resnet50_optimized.py. It matters because that pipeline emits ops the default
# one does not: prepare_conv2d_weights / prepare_conv2d_bias and to_memory_config.
# Testing them needs the config that produces them, not a hand-written single-op graph.
import os as _os
if _os.environ.get("PROBE_OPT") == "1":
    from forge.config import MLIRConfig
    _mc = (
        MLIRConfig()
        .set_enable_consteval(True)
        .set_optimization_level(2)
        .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi2)
        .set_enable_remove_dead_values(True)
        .set_max_legal_layouts(8)
    )
    cfg = CompilerConfig(mlir_config=_mc)
    cfg.enable_optimization_passes = True
else:
    cfg = CompilerConfig()
cfg.default_df_override = forge._C.DataFormat.Float16_b

try:
    c = forge.compile(m, inputs, module_name=f"probe_{OP}", compiler_cfg=cfg)
except Exception as e:
    print(f"[{OP}] RESULT: COMPILE_FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)
    raise SystemExit(0)

src = json.loads(c.compiled_binary.as_json()).get("mlir", {}).get("source", "")
ops = sorted({t for l in src.splitlines() for t in l.split('"') if t.startswith("ttnn.")})
print(f"[{OP}] ttnn ops: {', '.join(ops)}", flush=True)
if __import__("os").environ.get("PROBE_DUMP_IR"):
    print(f"[{OP}] ---- TTNN IR ----\n{src}\n[{OP}] ---- END IR ----", flush=True)
for _l in src.splitlines():
    if "func.func" in _l or "ttnn.conv2d" in _l:
        print(f"[{OP}] IR: {_l.strip()[:240]}", flush=True)
# Full module to a file: the op order and the layout each op runs under is the only
# way to tell which surrounding op a numeric failure belongs to.
import os as _os
_os.makedirs("/proj_sw/user_dev/ctr-lelanchelian/tt-forge-onnx/add_rs/ir", exist_ok=True)
with open(f"/proj_sw/user_dev/ctr-lelanchelian/tt-forge-onnx/add_rs/ir/{OP}{'_opt' if _os.environ.get('PROBE_OPT') == '1' else ''}.mlir", "w") as _f:
    _f.write(src)

try:
    out = c(*inputs)
except Exception as e:
    kind = "QUASAR_REFUSED" if "not supported on Quasar" in str(e) else "RUN_FAIL"
    print(f"[{OP}] RESULT: {kind} {type(e).__name__}", flush=True)
    print(f"[{OP}] FULL EXCEPTION >>>", flush=True)
    print(str(e), flush=True)
    print(f"[{OP}] <<< END EXCEPTION", flush=True)
    raise SystemExit(0)

got = (out[0] if isinstance(out, (list, tuple)) else out).to(torch.float32).cpu().flatten()
g = ref(*inputs).to(torch.float32).flatten()
if got.numel() != g.numel():
    print(f"[{OP}] RESULT: SHAPE_MISMATCH got {got.numel()} want {g.numel()}", flush=True)
    raise SystemExit(0)
# PCC alone is scale- and offset-blind: got = 2*g would still score 1.0. Report
# absolute and relative error alongside it so a systematic gain or bias shows up,
# and gate PASS on all three.
# Layout-aware cross-check: this test uses C == H == W == 32, so an NCHW/NHWC
# mismatch is invisible to a shape or element-count check yet scrambles every
# position -- which looks exactly like "right values, wrong places".
_gt = ref(*inputs).to(torch.float32)
if _gt.dim() == 4:
    import itertools
    _best = []
    for _perm in itertools.permutations(range(4)):
        _alt = _gt.permute(*_perm).contiguous().flatten()
        if _alt.numel() != got.numel():
            continue
        _p = torch.corrcoef(torch.stack([_alt, got]))[0, 1].item()
        _best.append((_p, _perm))
    _best.sort(reverse=True)
    for _p, _perm in _best[:3]:
        print(f"[{OP}] PERM-SEARCH {_perm}: pcc={_p:.6f}", flush=True)
pcc = torch.corrcoef(torch.stack([g, got]))[0, 1].item()
max_abs = (g - got).abs().max().item()
denom = g.abs().clamp_min(1e-3)
max_rel = ((g - got).abs() / denom).max().item()
# bf16 has ~8 mantissa bits -> ~0.4% relative; allow a little slack for accumulation.
# Per-element relative error is meaningless once an op can output exact zeros (any
# relu, fused or not): the tiny absolute error divides by the 1e-3 floor and looks
# enormous. conv2d_fused_relu measured max_rel_err 2.93 while being accurate to
# 0.0148 out of a ~1.0 tensor. Gate on error normalised by the tensor's own scale,
# and keep printing the per-element figure for continuity with earlier records.
norm_err = (g - got).abs().max().item() / max(g.abs().max().item(), 1e-6)
ok = (pcc > 0.999) and (norm_err < 0.02)
print(f"[{OP}] RESULT: {'PASS' if ok else 'NUMERIC_FAIL'} "
      f"pcc={pcc:.6f} max_abs_err={max_abs:.5f} max_rel_err={max_rel:.5f} err/scale={norm_err:.5f}", flush=True)

_slope = (torch.dot(g - g.mean(), got - got.mean()) /
          torch.clamp(torch.dot(g - g.mean(), g - g.mean()), min=1e-12)).item()
print(f"[{OP}] SLOPE {_slope:.6f}  (1.0 = no systematic gain error)", flush=True)
if not ok:
    # Layout-vs-arithmetic discriminator: if the device produced the right values in the
    # wrong order, the sorted multisets still agree. That points at operand layout /
    # permutation rather than wrong maths.
    gs, _ = torch.sort(g)
    ds, _ = torch.sort(got)
    sorted_pcc = torch.corrcoef(torch.stack([gs, ds]))[0, 1].item()
    sorted_max_abs = (gs - ds).abs().max().item()
    print(f"[{OP}] SORTED-MULTISET pcc={sorted_pcc:.6f} max_abs={sorted_max_abs:.5f} "
          f"-> {'SAME VALUES, WRONG ORDER (layout bug)' if sorted_max_abs < 0.05 else 'values genuinely differ (arithmetic)'}",
          flush=True)
    _a = (torch.dot(g - g.mean(), got - got.mean()) /
          torch.clamp(torch.dot(g - g.mean(), g - g.mean()), min=1e-12)).item()
    print(f"[{OP}] FIT slope={_a:.6f}", flush=True)
    nz = (got.abs() > 1e-6).sum().item()
    print(f"[{OP}] nonzero {nz}/{got.numel()}  device[0:6]={[round(v,4) for v in got[:6].tolist()]}", flush=True)
    print(f"[{OP}] golden[0:6]={[round(v,4) for v in g[:6].tolist()]}", flush=True)
    # Where are the wrong elements? A failure that is uniform across the tensor
    # means arithmetic; one confined to particular positions of the output's own
    # index space names the axis the layout got wrong.
    import os as _os
    if _os.environ.get("PROBE_ERRMAP"):
        import numpy as _np
        shp = tuple(ref(*inputs).shape)
        d = (g - got).abs().numpy()
        scale = max(float(g.abs().max()), 1e-6)
        bad = d > 0.05 * scale
        print(f"[{OP}] ERRMAP bad={int(bad.sum())}/{bad.size} shape={shp}", flush=True)
        if shp and len(shp) == 4:
            b = bad.reshape(shp)
            per_c = b.any(axis=(0, 2, 3)).sum()
            print(f"[{OP}] ERRMAP channels_with_error={int(per_c)}/{shp[1]}", flush=True)
            for h in range(shp[2]):
                row = [int(b[:, :, h, w].sum()) for w in range(shp[3])]
                print(f"[{OP}] ERRMAP h={h} bad_per_w={row}", flush=True)
