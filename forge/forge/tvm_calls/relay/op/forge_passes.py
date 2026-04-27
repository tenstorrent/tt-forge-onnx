# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tvm

from tvm.relay import transform
from tvm.relay.dataflow_pattern import wildcard, is_op
from tvm._ffi.base import TVMError

import numpy as np
import math
import numpy as np
from tvm.relay.dataflow_pattern import *
from loguru import logger
from .utils import *
from tvm.relay.op import _make

# NOTE: TVM crashes when groups != 1 or groups != input_channels
class ConvertLayout(DFPatternCallback):
    def __init__(
        self,
    ):
        super().__init__()

        self.input = wildcard()
        self.conv2d = is_op("nn.conv2d")(self.input, wildcard())
        self.max_pool2d = is_op("nn.max_pool2d")(self.input)
        self.avg_pool2d = is_op("nn.avg_pool2d")(self.input)
        self.conv2d_tran = is_op("nn.conv2d_transpose")(self.input, wildcard())
        self.globalmax_pool2d = is_op("nn.global_max_pool2d")(self.input)
        self.globalavg_pool2d = is_op("nn.global_avg_pool2d")(self.input)
        self.adaptivemax_pool2d = is_op("nn.adaptive_max_pool2d")(self.input)
        self.adaptiveavg_pool2d = is_op("nn.adaptive_avg_pool2d")(self.input)
        self.imageresize2d = is_op("image.resize2d")(self.input)

        self.pattern = (
            self.conv2d
            | self.max_pool2d
            | self.avg_pool2d
            | self.conv2d_tran
            | self.globalmax_pool2d
            | self.globalavg_pool2d
            | self.adaptivemax_pool2d
            | self.adaptiveavg_pool2d
            | self.imageresize2d
        )

    def callback(self, pre, post, node_map):
        act = node_map[self.input][0]

        if (
            node_map[self.pattern][0].op.name == "nn.conv2d"
            and node_map[self.conv2d][0].attrs.data_layout == "NHWC"
            and node_map[self.conv2d][0].attrs.kernel_layout in ["HWIO", "HWOI"]
        ):

            weight = node_map[self.conv2d][0].args[1]
            if node_map[self.conv2d][0].attrs.kernel_layout == "HWOI":
                channel_first_weight = tvm.relay.transpose(weight, axes=[2, 3, 0, 1])
            else:
                channel_first_weight = tvm.relay.transpose(weight, axes=[3, 2, 0, 1])

            new_conv2d = tvm.relay.op.nn.conv2d(
                act,
                channel_first_weight,
                strides=post.attrs.strides,
                padding=post.attrs.padding,
                groups=post.attrs.groups,
                channels=post.attrs.channels,
                kernel_size=post.attrs.kernel_size,
                data_layout="NHWC",
                kernel_layout="OIHW",
            )
            return new_conv2d
        elif (
            node_map[self.pattern][0].op.name == "nn.global_max_pool2d"
            and node_map[self.globalmax_pool2d][0].attrs.layout == "NHWC"
        ):
            raise NotImplementedError
        elif (
            node_map[self.pattern][0].op.name == "nn.global_avg_pool2d"
            and node_map[self.globalavg_pool2d][0].attrs.layout == "NHWC"
        ):
            raise NotImplementedError
        elif (
            node_map[self.pattern][0].op.name == "nn.adaptive_max_pool2d"
            and node_map[self.adaptivemax_pool2d][0].attrs.layout == "NHWC"
        ):
            raise NotImplementedError
        elif (
            node_map[self.pattern][0].op.name == "nn.adaptive_avg_pool2d"
            and node_map[self.adaptiveavg_pool2d][0].attrs.layout == "NHWC"
        ):
            raise NotImplementedError
        else:
            return post


class ResolveConvChannels(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.act = wildcard()
        self.weight = wildcard()

        t1 = is_op("transpose")(self.act)
        t2 = is_op("transpose")(t1)
        self.conv = (
            is_op("nn.conv2d")(t2, self.weight)
            | is_op("nn.conv2d_transpose")(t2, self.weight)
            | is_op("nn.max_pool2d")(t2)
            | is_op("nn.avg_pool2d")(t2)
        )
        t3 = is_op("transpose")(self.conv)
        self.pattern = is_op("transpose")(t3)

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        weight = node_map[self.weight][0] if self.weight in node_map else None
        conv = node_map[self.conv][0]

        pool = False
        if conv.op.name == "nn.conv2d":
            op = tvm.relay.nn.conv2d
        elif conv.op.name == "nn.conv2d_transpose":
            op = tvm.relay.nn.conv2d_transpose
        elif conv.op.name == "nn.max_pool2d":
            op = tvm.relay.nn.max_pool2d
            pool = True
        else:
            op = tvm.relay.nn.avg_pool2d
            pool = True

        conv_layout = conv.attrs.data_layout if not pool else conv.attrs.layout

        data_layout = "NHWC" if conv_layout == "NCHW" else "NCHW"

        if not pool:
            return op(
                act,
                weight,
                strides=conv.attrs.strides,
                padding=conv.attrs.padding,
                groups=conv.attrs.groups,
                channels=conv.attrs.channels,
                kernel_size=conv.attrs.kernel_size,
                data_layout=data_layout,
                kernel_layout=conv.attrs.kernel_layout,
            )
        else:
            return op(
                act,
                pool_size=conv.attrs.pool_size,
                strides=conv.attrs.strides,
                padding=conv.attrs.padding,
                layout=data_layout,
                ceil_mode=conv.attrs.ceil_mode,
            )


class FuseConvAndPoolPadding(DFPatternCallback):
    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        self.act = wildcard()
        self.weight = wildcard()
        self.pad = is_op("nn.pad")(self.act, wildcard())
        self.pattern = is_op("nn.conv2d")(self.pad, self.weight) | is_op("nn.max_pool2d")(self.pad)

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        conv_pool = node_map[self.pattern][0]
        pad = node_map[self.pad][0]

        if (
            not all(conv_pool.attrs.padding[i] == 0 for i in range(len(conv_pool.attrs.padding)))
            or not isinstance(pad.args[1], tvm.relay.Constant)
            or not pad.args[1].data.shape == ()
            or not int(pad.args[1].data.numpy()) == 0
        ):
            return

        pad_width = pad.attrs.pad_width
        top_pad, bottom_pad = pad_width[-2]
        left_pad, right_pad = pad_width[-1]

        pad_mode = pad.attrs.pad_mode

        # Fuse Pad Only if the mode is constant
        # Fusion is skipped if the padding is asymmetric for max-pooling or if the padding mode is not "constant".
        # 'edge' == 'replicate' in forge
        if ((top_pad != bottom_pad or left_pad != right_pad) and (conv_pool.op.name == "nn.max_pool2d")) or (
            pad_mode == "edge" or pad_mode == "reflect"
        ):
            act = tvm.relay.op.nn.pad(act, pad_width, pad_mode=pad_mode)

            padding = [0, 0, 0, 0]
        else:
            padding = [top_pad, left_pad, bottom_pad, right_pad]

        # update conv padding by adding the padding from the pad op
        new_padding = [
            padding[0] + conv_pool.attrs.padding[0],
            padding[1] + conv_pool.attrs.padding[1],
            padding[2] + conv_pool.attrs.padding[2],
            padding[3] + conv_pool.attrs.padding[3],
        ]

        op_attrs = {**conv_pool.attrs}
        op_attrs["padding"] = new_padding

        if conv_pool.op.name == "nn.conv2d":
            weight = node_map[self.weight][0]
            return tvm.relay.op.nn.conv2d(act, weight, **op_attrs)
        else:
            return tvm.relay.op.nn.max_pool2d(act, **op_attrs)


class DecomposeRoll(DFPatternCallback):
    def __init__(self):
        super().__init__(require_type=True)
        self.indices = is_constant()
        self.input = wildcard()
        self.gather = is_op("gather")(self.input, self.indices)
        self.pattern = self.gather

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        gather_axis = int(post.attrs.axis)
        indices = node_map[self.indices][0].data.numpy()
        input_shape = pre_node_map[self.input][0].checked_type.shape
        stop = input_shape[gather_axis]
        postion_arr = np.arange(int(stop))
        slicing_index = np.zeros((len(input_shape),), dtype=int).tolist()
        slicing_index[int(gather_axis)] = postion_arr.tolist()
        if len(slicing_index) == 1:
            indices_arr = indices[slicing_index[0]]
        elif len(slicing_index) == 2:
            indices_arr = indices[slicing_index[0], slicing_index[1]]
        elif len(slicing_index) == 3:
            indices_arr = indices[slicing_index[0], slicing_index[1], slicing_index[2]]
        elif len(slicing_index) == 4:
            indices_arr = indices[slicing_index[0], slicing_index[1], slicing_index[2], slicing_index[3]]
        else:
            raise NotImplementedError
        diff_arr = postion_arr - indices_arr
        if len(set(diff_arr)) == 2:
            shift_size = indices_arr[0]
            slice1_start = slice2_start = 0
            slice1_end = slice2_end = int(input_shape[int(gather_axis)])
            if shift_size < 0:
                slice1_start = int(shift_size) * -1
                slice2_end = int(shift_size) * -1
            else:
                slice1_start = int(shift_size)
                slice2_end = int(shift_size)
            act = post.args[0]
            slice1 = tvm.relay.strided_slice(
                act,
                begin=[
                    slice1_start,
                ],
                end=[
                    slice1_end,
                ],
                strides=[
                    1,
                ],
                axes=[
                    int(gather_axis),
                ],
            )
            slice2 = tvm.relay.strided_slice(
                act,
                begin=[
                    slice2_start,
                ],
                end=[
                    slice2_end,
                ],
                strides=[
                    1,
                ],
                axes=[
                    int(gather_axis),
                ],
            )
            slice_concatenate = tvm.relay.concatenate([slice1, slice2], axis=int(gather_axis))
            return slice_concatenate
        return post


class DecomposeReverse(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=False, require_type=True)
        self.input = wildcard()
        self.pattern = is_op("reverse")(self.input)

    def callback(self, pre, post, node_map):
        input_shape = node_map[self.input][0].checked_type.shape
        axis = int(post.attrs.axis)
        if int(axis) < 0:
            axis = abs(int(axis) + int(len(input_shape)))
        act = post.args[0]
        start = int(input_shape[axis]) - 1
        stop = -1
        step = -1
        indices = tvm.relay.Constant(tvm.nd.array(np.arange(start, stop, step).astype(int)))
        if int(axis) == 0:
            adv_index_out = tvm.relay.adv_index([act, indices])
            return adv_index_out
        else:
            transpose_1_axes = [int(axis)]
            intermediate = list(set(np.arange(int(len(input_shape))).tolist()).difference(set(transpose_1_axes)))
            intermediate.sort()
            transpose_1_axes.extend(intermediate)
            transpose_2_axes = []
            transpose_2_axes.extend(np.arange(1, int(len(input_shape))).tolist())
            transpose_2_axes.insert(int(axis), 0)
            transpose_1 = tvm.relay.transpose(act, axes=transpose_1_axes)
            adv_index_1 = tvm.relay.adv_index([transpose_1, indices])
            transpose_2 = tvm.relay.transpose(adv_index_1, axes=transpose_2_axes)
            return transpose_2


class DecomposeDynamicResize2d(DFPatternCallback):
    def __init__(self):
        super().__init__(require_type=True)
        self.input = wildcard()
        self.input2 = wildcard()
        self.input3 = wildcard()
        self.resize = is_op("dyn.image.resize2d")(self.input, self.input2, self.input3)
        self.pattern = (
            wildcard()(self.resize) | wildcard()(wildcard(), self.resize) | wildcard()(self.resize, wildcard())
        )

    def callback(self, pre, post, node_map):
        act = node_map[self.input][0]
        roi = tuple(node_map[self.input3][0].data.numpy())
        resize = node_map[self.resize][0]
        consumer = node_map[self.pattern][0]

        supported_ops = [
            "add",
            "subtract",
            "multiply",
        ]

        assert consumer.op.name in supported_ops, f"Do not support shape finding for op {consumer.op.name}"

        new_shape = list(consumer.checked_type.shape)
        new_resize = tvm.relay.image.resize2d(
            act,
            size=new_shape[-2:],
            roi=roi,
            layout="NCHW",
            method=resize.attrs.method,
            coordinate_transformation_mode=resize.attrs.coordinate_transformation_mode,
            rounding_method=resize.attrs.rounding_method,
            cubic_alpha=resize.attrs.cubic_alpha,
            cubic_exclude=resize.attrs.cubic_exclude,
            extrapolation_value=resize.attrs.extrapolation_value,
            out_dtype="float32",
        )

        args = list(consumer.args)
        resize_idx = args.index(resize)
        args[resize_idx] = new_resize

        return tvm.relay.expr.Call(consumer.op, args)


class RemoveCast(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.pattern = is_op("cast")(self.act)

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        return act


class DecomposeStack(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)

        # This occurs when compiling tf.stack
        self.act1 = wildcard()
        self.act2 = wildcard()
        self.exp_dims1 = is_op("expand_dims")(self.act1)
        self.exp_dims2 = is_op("expand_dims")(self.act2)

        self.tup = is_tuple([self.exp_dims1, self.exp_dims2])
        self.concatenate = is_op("concatenate")(self.tup)
        self.r4 = is_op("reshape")(self.concatenate)
        self.pattern = self.r4

    def callback(self, pre: Expr, post: Expr, node_map: tvm.ir.container.Map) -> Expr:

        # Get original concatenate axis and target shape
        concat_node = node_map[self.concatenate][0]
        target_axis = concat_node.attrs.axis
        target_shape = post.attrs.newshape

        # replacing concat op with stack, as is done in the pytorch decomposition
        act1 = node_map[self.act1][0]
        act2 = node_map[self.act2][0]
        tup = tvm.relay.Tuple([act1, act2])
        # Use original axis for stack
        stacked = tvm.relay.stack(tup, axis=target_axis)
        # Use original target shape
        output = tvm.relay.reshape(stacked, newshape=target_shape)

        return output


class DecomposeMultiAxisMax(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.max = is_op("max")(self.act)

        self.pattern = self.max

    def callback(self, pre, post, node_map):
        if post.attrs.axis == None:
            reduce_axes = [x for x in range(len(list(post.args[0].checked_type.shape)))]
        else:
            reduce_axes = list(post.attrs.axis)
        if len(reduce_axes) == 1:
            return post

        acts = node_map[self.act][0]
        keepdims = bool(post.attrs.keepdims)
        output_shape = list(pre.checked_type.shape)

        for axis in reduce_axes:
            acts = tvm.relay.max(acts, axis=int(axis), keepdims=True)

        if keepdims == False:
            acts = tvm.relay.reshape(acts, newshape=output_shape)
        return acts


class DecomposeMultiAxisTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.transpose = is_op("transpose")(self.act)
        self.pattern = self.transpose

    def callback(self, pre, post, node_map):

        transpose_axes = pre.attrs.axes
        act_shape = pre.args[0].checked_type.shape
        acts = node_map[self.act][0]

        if transpose_axes == None:
            return post

        if len(transpose_axes) == 2:
            return post

        ndim = len(act_shape)
        total_permute = list(transpose_axes)
        total_permute = [int(x) for x in total_permute]
        no_permute = list(range(ndim))
        output = acts

        for i in range(ndim):
            if total_permute[i] != i:
                new_order = total_permute.copy()
                new_order[i + 1 :] = no_permute[i + 1 :]
                new_order[new_order[i]] = i

                total_permute = [new_order[i] for i in total_permute]
                output = tvm.relay.transpose(output, axes=new_order)

        return output


class DecomposeMultiAxisMean(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.mean = is_op("mean")(self.act)
        self.pattern = self.mean

    def callback(self, pre, post, node_map):
        if post.attrs.axis == None:
            reduce_axes = [x for x in range(len(list(post.args[0].checked_type.shape)))]
        else:
            reduce_axes = list(post.attrs.axis)
        if len(reduce_axes) == 1:
            return post

        acts = node_map[self.act][0]

        keepdims = bool(post.attrs.keepdims)
        output_shape = list(pre.checked_type.shape)

        for axis in reduce_axes:
            acts = tvm.relay.mean(acts, axis=int(axis), keepdims=True)

        if keepdims == False:
            # Need to squeeze in order from rightmost dim to leftmost dim
            # Reverse sort since axes are positive
            for axis in sorted(reduce_axes, reverse=True):
                acts = tvm.relay.squeeze(acts, axis=[axis])
        return acts


class DecomposeMultiAxisSum(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.sum = is_op("sum")(self.act)
        self.pattern = self.sum

    def callback(self, pre, post, node_map):
        reduce_axes = list(post.attrs.axis)
        if len(reduce_axes) == 1:
            return post

        acts = node_map[self.act][0]

        keepdims = bool(post.attrs.keepdims)
        output_shape = list(pre.checked_type.shape)

        for axis in reduce_axes:
            acts = tvm.relay.sum(acts, axis=int(axis), keepdims=True)

        if keepdims == False:
            acts = tvm.relay.reshape(acts, newshape=output_shape)
        return acts


class DecomposeMultiAxisBroadcast(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.broadcast_to = is_op("broadcast_to")(self.act)
        self.pattern = self.broadcast_to

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        acts = node_map[self.act][0]
        inp_shape = list(pre_node_map[self.act][0].checked_type.shape)
        target_shape = list(post.attrs.shape)

        while len(inp_shape) < len(target_shape):
            inp_shape = [1] + inp_shape
            acts = tvm.relay.reshape(acts, newshape=inp_shape)

        for i, (inp_dim, target_dim) in enumerate(zip(inp_shape, target_shape)):
            if inp_dim == target_dim:
                continue

            one_axis_target = target_shape[: i + 1] + inp_shape[i + 1 :]
            acts = tvm.relay.broadcast_to(acts, one_axis_target)

        return acts


class DecomposeConv1DToConv2D(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.act = wildcard()
        self.weight = wildcard()
        self.conv_pattern = is_op("nn.conv1d")(self.act, self.weight)
        self.pattern = self.conv_pattern

    def callback(self, pre, post, node_map):
        if post.attrs.data_layout == "NWC" and post.attrs.kernel_layout == "WIO":
            assert False, "Conv1d from TF is not supported yet"
            # TODO: converting TF conv1d - channel-last to channel-first Conv2d
        else:
            # reshape activation and reshape weight
            expected_output_shape = pre.checked_type.shape

            acts = node_map[self.act][0]
            acts_shape = list(pre.args[0].checked_type.shape)
            weights = node_map[self.weight][0]
            weights_shape = list(pre.args[1].checked_type.shape)

            new_acts_shape = [acts_shape[0], acts_shape[1], acts_shape[2], 1]
            new_weights_shape = [weights_shape[0], weights_shape[1], weights_shape[2], 1]
            reshaped_acts = tvm.relay.reshape(acts, newshape=new_acts_shape)
            reshaped_weights = tvm.relay.reshape(weights, newshape=new_weights_shape)

            new_conv2d = tvm.relay.op.nn.conv2d(
                reshaped_acts,
                reshaped_weights,
                strides=[post.attrs.strides[0], 1],
                padding=[post.attrs.padding[0], 0, post.attrs.padding[1], 0],
                # (TODO) Since weight kernel is 1 on unsqueezed dim, dilation shouldnt matter. This is needed because we dont support different
                # dilation for each dim in forge conv2d.
                dilation=[post.attrs.dilation[0], post.attrs.dilation[0]],
                groups=post.attrs.groups,
                channels=post.attrs.channels,
                kernel_size=[post.attrs.kernel_size[0], 1],
                data_layout="NCHW",
                kernel_layout="OIHW",
            )

            # reshape back result
            reshaped_back_result = tvm.relay.reshape(new_conv2d, newshape=expected_output_shape)
            return reshaped_back_result


class ReformatTFConv2d(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.weight = wildcard()
        conv = is_op("nn.conv2d")(self.act, self.weight)
        self.pattern = conv

    def callback(self, pre, post, node_map):
        if post.attrs.data_layout == "NHWC" and post.attrs.kernel_layout == "HWIO":
            # convert TF channel-last to channel-first
            act = node_map[self.act][0]
            weight = node_map[self.weight][0]

            channel_first_act = tvm.relay.transpose(act, axes=[0, 3, 1, 2])
            channel_first_weight = tvm.relay.transpose(weight, axes=[3, 2, 0, 1])

            new_conv2d = tvm.relay.op.nn.conv2d(
                channel_first_act,
                channel_first_weight,
                strides=post.attrs.strides,
                padding=post.attrs.padding,
                groups=post.attrs.groups,
                channels=post.attrs.channels,
                kernel_size=post.attrs.kernel_size,
                data_layout="NCHW",
                kernel_layout="OIHW",
            )
            out_reshape = tvm.relay.transpose(new_conv2d, axes=[0, 2, 3, 1])
            return out_reshape
        else:
            return post


class ReformatTFMaxpool(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        maxpool = is_op("nn.max_pool2d")(
            self.act,
        )
        self.pattern = maxpool

    def callback(self, pre, post, node_map):
        if post.attrs.layout == "NHWC":
            # convert TF channel-last to channel-first
            act = node_map[self.act][0]

            channel_first_act = tvm.relay.transpose(act, axes=[0, 3, 1, 2])

            new_pool = tvm.relay.op.nn.max_pool2d(
                channel_first_act,
                pool_size=post.attrs.pool_size,
                strides=post.attrs.strides,
                padding=post.attrs.padding,
                layout="NCHW",
                ceil_mode=post.attrs.ceil_mode,
            )
            out_reshape = tvm.relay.transpose(new_pool, axes=[0, 2, 3, 1])
            return out_reshape
        else:
            return post


class DecomposePower(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.exponent = is_constant()
        power = is_op("power")(self.act, self.exponent)

        self.pattern = power

    def callback(self, pre, post, node_map):

        act = node_map[self.act][0]
        exponent = node_map[self.exponent][0].data.numpy().item()

        op = act
        if exponent < 0:
            exponent *= -1
            ADD_DIVIDE = True
        else:
            ADD_DIVIDE = False

        if not exponent.is_integer():
            dec = exponent - int(exponent)
            if dec == 0.5:
                op = tvm.relay.sqrt(op)
                exponent -= dec
            elif dec == 0.75:
                op = tvm.relay.sqrt(op)
                op = tvm.relay.multiply(op, act)
                op = tvm.relay.sqrt(op)
                exponent -= dec
            else:
                assert False, f"Exponent value {dec} is not yet supported."

        if exponent.is_integer():
            original_op = op
            while exponent > 1:
                op = tvm.relay.multiply(op, original_op)
                exponent -= 1

        if ADD_DIVIDE:
            # add an divide
            op = tvm.relay.reciprocal(op)

        assert exponent == 0 or exponent == 1, "Exponent has not been decomposed"

        return op


class DenseWeightTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.weight = wildcard()
        act = wildcard()
        self.pattern = is_op("nn.dense")(act, self.weight)
        self.transpose_pattern = is_op("transpose")(wildcard())

    def callback(self, pre, post, node_map):
        # If there's already a transpose, we don't need another one to
        # fuse into forge.matmul
        if self.transpose_pattern.match(post.args[1]):
            return post

        # Doesn't have transpose, add two, one to fuse, the other to undo
        act = post.args[0]
        weight = post.args[1]
        wt1 = tvm.relay.transpose(weight)
        wt2 = tvm.relay.transpose(wt1)
        dtype = post.checked_type.dtype
        res = tvm.relay.nn.dense(act, wt2, out_dtype=dtype)
        return res


class LiftLinearSplit(DFPatternCallback):
    """
    Lifting head split of the KQV matmul above linear/dense layer.

    This is done by hoisting the split above the linear/dense layer, and then
    doing the split on weights and bias of the linear/dense layer.
    """

    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        # Linear layer
        self.act = wildcard()
        self.weight = wildcard()
        self.bias = wildcard()
        self.dense = is_op("nn.dense")(self.act, self.weight)

        # I) Bias with reshape first variant
        self.reshape1 = is_op("reshape")(self.dense)
        self.add1 = is_op("add")(self.reshape1, self.bias)
        self.split1 = is_op("split")(self.add1)
        self.pattern1 = self.split1

        # II) Bias with add first variant
        self.add2 = is_op("add")(self.dense, self.bias)
        self.reshape2 = is_op("reshape")(self.add2)
        self.split2 = is_op("split")(self.reshape2)
        self.pattern2 = self.split2

        # III) No bias variant
        self.reshape3 = is_op("reshape")(self.dense)
        self.split3 = is_op("split")(self.reshape3)
        self.pattern3 = self.split3

        # Recognize any of the three patterns
        self.pattern = self.pattern1 | self.pattern2 | self.pattern3

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        assert pre.op.name == "split", "Split should be last op in pattern matcher"

        # Determine if bias is present or not based on pattern
        if self.pattern1.match(post):
            has_bias = True
            reshape_node = pre_node_map[self.reshape1][0]
        elif self.pattern2.match(post):
            has_bias = True
            reshape_node = pre_node_map[self.reshape2][0]
        elif self.pattern3.match(post):
            has_bias = False
            reshape_node = pre_node_map[self.reshape3][0]
        else:
            assert False, "Invalid pattern match case, shouldn't happen"

        # Linear/Dense attributes
        weight_shape = pre_node_map[self.weight][0].checked_type.shape

        # Split attributes
        split_op_axis = pre.attrs.axis
        split_op_indices_or_sections = pre.attrs.indices_or_sections

        # Shape of Reshape producer
        pre_reshape_shape = list(reshape_node.args[0].checked_type.shape)
        pre_reshape_shape = [int(x) for x in pre_reshape_shape]
        # Shape of split producer
        pre_split_shape = list(pre.args[0].checked_type.shape)
        pre_split_shape = [int(x) for x in pre_split_shape]

        # If split is defined by number of sections, compute the indices
        if isinstance(split_op_indices_or_sections, tvm.tir.expr.IntImm):
            total_len = int(pre_split_shape[split_op_axis])
            section_len = total_len // int(split_op_indices_or_sections)
            split_op_indices_or_sections = list(range(section_len, total_len, section_len))
        else:
            split_op_indices_or_sections = [int(ios) for ios in split_op_indices_or_sections]
        split_op_indices_or_sections_num = len(split_op_indices_or_sections) + 1

        # Define new reshape shape for fractured dense paths
        newshape = list(pre_split_shape)
        newshape[split_op_axis] = -1

        # Weight should be transposed in dense/linear layer. Therefore, if split
        # is done along the last axis (C dim), we need to do the split over the
        # first one (R dim) instead
        if split_op_axis == len(pre_split_shape) - 1 or split_op_axis == -1:
            head_split_reshape_before_split_is_valid = bool(pre_split_shape[split_op_axis] == weight_shape[0])
            head_split_reshape_after_split_is_valid = bool(pre_split_shape[-1] * pre_split_shape[-2] == weight_shape[0])
            split_op_axis = 0

            assert (
                head_split_reshape_before_split_is_valid or head_split_reshape_after_split_is_valid
            ), "Invalid (not covered) split case for linear split uplift"

            # Use split op selection instead of indices for proper weights cut
            if head_split_reshape_after_split_is_valid:
                split_op_indices_or_sections = split_op_indices_or_sections_num

        # Split weights and biases
        weight = node_map[self.weight][0]
        bias = node_map[self.bias][0] if has_bias else None

        split_weights = []
        split_biases = []
        if pre_reshape_shape[-1] == np.prod(pre_split_shape[-2:]) and np.prod(pre_reshape_shape[:-1]) == np.prod(
            pre_split_shape[:-2]
        ):
            # Weight dim is sliced, use strided slice to get the correct weight
            size_per_y_dim = pre_split_shape[-1] // split_op_indices_or_sections_num

            for i in range(split_op_indices_or_sections_num):
                split_weights_i = []
                split_biases_i = []
                for j in range(pre_split_shape[-2]):
                    split_weights_i.append(
                        tvm.relay.strided_slice(
                            weight,
                            begin=[
                                j * pre_split_shape[-1] + i * size_per_y_dim,
                            ],
                            end=[
                                j * pre_split_shape[-1] + (i + 1) * size_per_y_dim,
                            ],
                            strides=[
                                1,
                            ],
                            axes=[
                                split_op_axis,
                            ],
                        )
                    )

                if len(split_weights_i) > 1:
                    split_weights.append(tvm.relay.concatenate(split_weights_i, axis=split_op_axis))
                else:
                    split_weights.append(split_weights_i[0])

                if has_bias:
                    for j in range(pre_split_shape[-2]):
                        split_biases_i.append(
                            tvm.relay.strided_slice(
                                bias,
                                begin=[
                                    j * pre_split_shape[-1] + i * size_per_y_dim,
                                ],
                                end=[
                                    j * pre_split_shape[-1] + (i + 1) * size_per_y_dim,
                                ],
                                strides=[
                                    1,
                                ],
                                axes=[
                                    split_op_axis,
                                ],
                            )
                        )

                    if len(split_biases_i) > 1:
                        split_biases.append(tvm.relay.concatenate(split_biases_i, axis=split_op_axis))
                    else:
                        split_biases.append(split_biases_i[0])

            split_weights = tvm.relay.expr.Tuple(split_weights)
            split_biases = tvm.relay.expr.Tuple(split_biases) if has_bias else None
        else:
            split_weights = tvm.relay.split(weight, split_op_indices_or_sections, split_op_axis)
            split_biases = tvm.relay.split(bias, split_op_indices_or_sections, -1) if has_bias else None

        # Defined by number of splits, create fractured dense paths with splitted
        # weights and biases (if applied). For example, this means that instead of
        # having one dense layer, if there is a split of 3, we will have 3 new dense
        # layers with 1/3 of the weights and biases (if applied) each. At the end,
        # the outputs of these 3 dense layers will be concatenated to provide valid
        # KQV split results
        outputs = []
        act = node_map[self.act][0]
        for i in range(split_op_indices_or_sections_num):
            single_path_output = tvm.relay.nn.dense(act, split_weights[i])
            single_path_output = tvm.relay.add(single_path_output, split_biases[i]) if has_bias else single_path_output
            single_path_output = tvm.relay.reshape(single_path_output, newshape=newshape)
            outputs.append(single_path_output)
        return tvm.relay.expr.Tuple(outputs)


class LowerSplitToStridedSlice(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.split = is_op("split")(self.act)

        self.pattern = is_tuple_get_item(wildcard())

    def callback(self, pre, post, node_map):
        split = post.tuple_value().op

        if not self.split.match(split):
            return post

        act = split.args[0]
        act_shape = pre.tuple_value().op.args[0].checked_type.shape
        axis = split.attrs.axis
        if axis < 0:
            axis += len(act_shape)

        if isinstance(split.attrs.indices_or_sections, tvm.tir.expr.IntImm):
            sections = int(split.attrs.indices_or_sections)
            total_length = int(act_shape[axis])
            ios = list(range(total_length // sections, total_length, total_length // sections))
        else:
            ios = [int(dim) for dim in split.attrs.indices_or_sections]
        ios.append(act_shape[axis])

        begin = 0 if post.index == 0 else ios[post.index - 1]
        end = ios[post.index]

        # Check if this strided slice does nothing. If so just return act
        if end - begin == act_shape[axis]:
            return act

        sliced_act = tvm.relay.strided_slice(act, (begin,), (end,), axes=(axis,))
        return sliced_act


class RemoveRedundantTake(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__(rewrite_once=rewrite_once, require_type=require_type)
        self.input_tensor = wildcard()
        self.indices = is_constant()
        self.pattern = is_op("take")(self.input_tensor, self.indices)

    def callback(self, pre, post, node_map):
        if node_map[self.indices][0].data.numpy().size == 1:
            indices = node_map[self.indices][0].data.numpy().item()
        else:
            return post

        act = node_map[self.input_tensor][0]
        act_shape = list(pre.args[0].checked_type.shape)
        axis = pre.attrs.axis

        # Skip removal of takes which contain dynamic shapes
        if any([isinstance(dim, tvm.tir.expr.Any) for dim in act_shape]):
            return post

        if act_shape[int(axis)] == 1 and indices == 0:
            newshape = act_shape
            del newshape[int(axis)]

            out = tvm.relay.reshape(act, newshape=newshape)
            return out

        return post


class PopulateTransposeAxes(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__()
        self.input_tensor = wildcard()

        self.pattern = is_op("transpose")(wildcard())

    def callback(self, pre, post, node_map):
        if pre.attrs.axes is not None:
            return post

        transpose_dims = len(pre.checked_type.shape)
        last_dim = -transpose_dims - 1
        taxes = list(range(-1, last_dim, -1))

        return tvm.relay.transpose(post.args[0], axes=taxes)


class PopulateReduceAxes(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__()
        self.pattern = is_op("sum")(wildcard())

    def callback(self, pre, post, node_map):
        if pre.attrs.axis is not None:
            return post

        ndims = len(pre.args[0].checked_type.shape)
        raxes = list(range(0, ndims, 1))

        return tvm.relay.sum(post.args[0], axis=raxes, keepdims=pre.attrs.keepdims)


class RemoveRedundantReshape(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__(rewrite_once=rewrite_once, require_type=require_type)
        self.input_tensor = wildcard()
        self.reshape = is_op("reshape")(self.input_tensor)
        self.pattern = self.reshape

    def callback(self, pre, post, node_map):
        act = node_map[self.input_tensor][0]
        reshape_op = node_map[self.reshape][0]
        input_shape = list(act.checked_type.shape)
        new_shape = list(reshape_op.attrs.newshape)
        # Remove the reshape if the input and target shapes are identical
        if input_shape == new_shape:
            return act
        else:
            return post


class LowerCopyToNOP(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__(rewrite_once=rewrite_once, require_type=require_type)
        self.pattern = is_op("copy")(wildcard())

    def callback(self, pre, post, node_map):
        result = tvm.relay.identity(post.args[0])
        return post.args[0]


class ArgmaxAndMaxReconstruct(DFPatternCallback):
    def __init__(self, rewrite_once=True):
        super().__init__(rewrite_once=rewrite_once)
        self.input_tensor = wildcard()
        self.softmax = is_op("nn.softmax")(self.input_tensor)

        self.argmax = is_op("argmax")(self.softmax)
        self.copy1 = is_op("copy")(self.argmax)
        self.copy2 = is_op("copy")(self.copy1)
        self.indices = is_constant()
        self.add = is_op("add")(self.indices, self.argmax)

        self.reshape = is_op("reshape")(self.softmax)
        self.take = is_op("take")(self.reshape, self.add)
        self.copy3 = is_op("copy")(self.take)
        self.copy4 = is_op("copy")(self.copy3)

        self.pattern = is_tuple([wildcard(), self.copy2, self.copy4])

    def callback(self, pre, post, node_map):
        act = node_map[self.input_tensor][0]
        softmax = tvm.relay.nn.softmax(act, node_map[self.softmax][0].attrs.axis)
        argmax = tvm.relay.argmax(
            softmax,
            axis=node_map[self.argmax][0].attrs.axis,
            keepdims=True,
            exclude=node_map[self.argmax][0].attrs.exclude,
            select_last_index=node_map[self.argmax][0].attrs.select_last_index,
        )
        argmax = tvm.relay.squeeze(argmax, axis=node_map[self.argmax][0].attrs.axis)
        maximum = tvm.relay.max(
            softmax,
            axis=node_map[self.argmax][0].attrs.axis,
            keepdims=True,
            exclude=node_map[self.argmax][0].attrs.exclude,
        )
        maximum = tvm.relay.squeeze(maximum, axis=node_map[self.argmax][0].attrs.axis)
        return tvm.relay.Tuple([post.fields[0], argmax, maximum], span=pre.span)


class LowerTakeToStridedSlice(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__(rewrite_once=rewrite_once)
        self.input_tensor = wildcard()
        self.indices = is_constant()
        self.pattern = is_op("take")(self.input_tensor, self.indices)

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        # Skip removal of takes which contain dynamic shapes
        if any([isinstance(dim, tvm.tir.expr.Any) for dim in pre.checked_type.shape]):
            return post

        act = node_map[self.input_tensor][0]
        try:
            act_shape = list(pre_node_map[self.input_tensor][0].checked_type.shape)
        except ValueError as e:
            act_shape = list(pre_node_map[self.input_tensor][0].attrs.newshape)

        # If shape is not fully known, return
        for dim in act_shape:
            if isinstance(dim, tvm.tir.Any):
                return post

        try:
            indices = node_map[self.indices][0].data.numpy().flatten()
            start_value = indices[0]
            # Make sure indices are continuously ascending with stride of 1
            for idx, val in enumerate(indices):
                if start_value + idx != val:
                    raise ValueError
        except ValueError as v:
            return post

        axis = pre.attrs.axis
        if len(indices) == 1 and indices[0] == -1:
            index = act_shape[int(axis)] - 1
            strided_slice = tvm.relay.strided_slice(act, begin=(index,), end=(index + 1,), strides=(1,), axes=(axis,))
        else:
            strided_slice = tvm.relay.strided_slice(
                act, begin=(indices[0],), end=(indices[-1] + 1,), strides=(1,), axes=(axis,)
            )

        reshape = tvm.relay.reshape(strided_slice, newshape=pre.checked_type.shape)
        return reshape


class AddSqueezeForArgmax(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__(rewrite_once=rewrite_once, require_type=require_type)
        self.input_tensor = wildcard()
        self.pattern = is_op("argmax")(self.input_tensor)

    def callback(self, pre, post, node_map):
        if post.attrs.keepdims:
            return post

        inp = node_map[self.input_tensor][0]
        axis = post.attrs.axis
        new_argmax = tvm.relay.argmax(
            inp,
            axis=axis,
            keepdims=True,
        )
        result = tvm.relay.squeeze(new_argmax, axis=axis)
        return result


class ConvertArgmaxTakeToReduceMax(DFPatternCallback):
    def __init__(self, rewrite_once=True):
        super().__init__(rewrite_once=rewrite_once)
        self.input_tensor = wildcard()
        self.argmax = wildcard()
        self.reshape = is_op("reshape")(self.input_tensor)
        self.const = is_constant()
        self.add = is_op("add")(self.const, self.argmax)
        self.take = is_op("take")(self.reshape, self.add)
        self.pattern = self.take

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        act = node_map[self.input_tensor][0]
        input_shape = list(pre_node_map[self.input_tensor][0].checked_type.shape)

        argmax_op = node_map[self.argmax][0]
        if argmax_op.op.name != "argmax":
            return post

        argmax_axis = int(argmax_op.attrs.axis[0])
        if argmax_axis != -1 and int(input_shape[0]) == 1:
            return post

        reshape_op = node_map[self.reshape][0]
        new_shape = list(reshape_op.attrs.newshape)
        if not (len(new_shape) == 1 and (int(new_shape[0]) == -1 or int(new_shape[0]) == 144512)):
            return post

        const_data = node_map[self.const][0].data.numpy()
        arange_max = int(input_shape[-2]) * int(input_shape[-1])
        test_np = np.expand_dims(
            np.arange(start=0, stop=arange_max, step=int(input_shape[-1]), dtype=const_data.dtype), axis=0
        )
        match = np.array_equal(const_data, test_np, equal_nan=False)
        if not match:
            return post

        result = tvm.relay.max(act, axis=-1, keepdims=True)
        result = tvm.relay.squeeze(result, axis=[-1])
        return result


class DecomposeMultiRangeTake(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__(rewrite_once=rewrite_once)
        self.input_tensor = wildcard()
        self.indices = is_constant()
        self.pattern = is_op("take")(self.input_tensor, self.indices)

    def callback(self, pre, post, node_map):
        """
        The goal here is to decompose an original take op to multiple take ops where
        the new ops, each represent a single continouos range.
        tvm.take can have a list of arbitrary indices.
        """
        act = node_map[self.input_tensor][0]
        try:
            indices = node_map[self.indices][0].data.numpy().item()
        except ValueError as v:
            return post

        return post


class EstimateWhereInCausalMask(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act1 = wildcard()
        self.act2 = wildcard()
        self.matmul = is_op("nn.batch_matmul")(self.act1, self.act2)
        self.reshape = is_op("reshape")(self.matmul)
        self.reciprocal = is_op("reciprocal")(is_constant())
        self.strided_slice = is_op("strided_slice")(wildcard())
        self.multiply = is_op("multiply")(self.reshape, self.reciprocal)

        self.pattern = is_op("where")(
            self.strided_slice | is_constant(), self.reshape | self.multiply | wildcard(), wildcard()
        )

    def callback(self, pre, post, node_map):
        # by assuming the masked value is >> activation, this allows
        # so simulate causal masking with eltwise ops, i.e. simply add
        # the masked value
        causal_mask = post.args[0]
        zero = tvm.relay.const(0.0, dtype=pre.checked_type.dtype)
        one = tvm.relay.const(1.0, dtype=pre.checked_type.dtype)
        inverse_causal_mask = tvm.relay.where(causal_mask, zero, one)

        value = post.args[2]
        mask = tvm.relay.multiply(inverse_causal_mask, value)

        act = post.args[1]
        return tvm.relay.add(act, mask)


class CastWhereConditionToBool(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)

        self.pattern = is_op("where")(wildcard(), wildcard(), wildcard())

    def callback(self, pre, post, node_map):
        cond = tvm.relay.cast(post.args[0], "bool")
        return tvm.relay.where(cond, post.args[1], post.args[2])


class DecomposeNegative(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.in_a = wildcard()

        self.pattern = is_op("negative")(self.in_a)

    def callback(self, pre, post, node_map):
        negative_one = tvm.relay.const(-1.0, dtype=pre.checked_type.dtype)
        mul = tvm.relay.multiply(post.args[0], negative_one)
        return mul


class DecomposeEinsum(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = is_tuple(None)

        self.pattern = is_op("einsum")(self.act)

    def callback(self, pre, post, node_map):
        equation = str(post.attrs.equation)
        if match_einsum_pattern("bct,bcs->bts", equation):
            assert len(node_map[self.act][0]) == 2
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            result = tvm.relay.nn.batch_matmul(srcA, srcB, transpose_a=True, transpose_b=False)
            return result
        elif match_einsum_pattern("bts,bcs->bct", equation):
            assert len(node_map[self.act][0]) == 2
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            result = tvm.relay.nn.batch_matmul(srcA, srcB, transpose_a=False, transpose_b=True)
            return tvm.relay.transpose(result, axes=[0, 2, 1])

        elif match_einsum_pattern("bnqd,bnkd->bnqk", equation):
            assert len(node_map[self.act][0]) == 2
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            transpose_srcB = tvm.relay.transpose(
                srcB,
                axes=[
                    0,
                    1,
                    3,
                    2,
                ],
            )
            new_shape_srcA = [int(srcA_shape[0]) * int(srcA_shape[1]), int(srcA_shape[2]), int(srcA_shape[3])]
            new_shape_srcB = [int(srcB_shape[0]) * int(srcB_shape[1]), int(srcB_shape[3]), int(srcB_shape[2])]

            reshape_srcA = tvm.relay.reshape(srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)

            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)
            return tvm.relay.reshape(result, newshape=[srcA_shape[0], srcA_shape[1], srcA_shape[2], srcB_shape[2]])

        elif match_einsum_pattern("abc,cde->abde", equation):
            assert len(node_map[self.act][0]) == 2
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]
            srcA_shape = srcA.checked_type.shape
            srcB_shape = srcB.checked_type.shape

            new_shape_srcA = [int(srcA_shape[0]) * int(srcA_shape[1]), int(srcA_shape[2])]  # a*b c
            new_shape_srcB = [int(srcB_shape[0]), int(srcB_shape[1]) * int(srcB_shape[2])]  # c d*e

            reshape_srcA = tvm.relay.reshape(srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(srcB, newshape=new_shape_srcB)

            # MatMul
            result = tvm.relay.nn.matmul(reshape_srcA, reshape_srcB)  # a*b d*e

            # Reshape
            reshape_result = tvm.relay.reshape(
                result, newshape=[srcA_shape[0], srcA_shape[1], srcB_shape[1], srcB_shape[2]]
            )
            return reshape_result

        elif match_einsum_pattern("ibnd,jbnd->bnij", equation):
            srcA_shape = list(pre.args[0][0].checked_type.shape)
            srcB_shape = list(pre.args[0][1].checked_type.shape)
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            if srcA_shape[1] != srcB_shape[1] and srcA_shape[1] == 1:
                # broadcast
                srcA_shape[1] = srcA_shape[1] * srcB_shape[1]
                srcA = tvm.relay.broadcast_to(srcA, srcA_shape)
            elif srcA_shape[1] != srcB_shape[1]:
                # Dont know how to handle
                assert False, f"TVM einsum decomposition does not support {equation} with srcA shape {srcA_shape}."

            transpose_srcA = tvm.relay.transpose(srcA, axes=[1, 2, 0, 3])  # bnid
            transpose_srcB = tvm.relay.transpose(srcB, axes=[1, 2, 3, 0])  # bndj
            new_shape_srcA = [int(srcA_shape[1]) * int(srcA_shape[2]), int(srcA_shape[0]), int(srcA_shape[3])]
            new_shape_srcB = [int(srcB_shape[1]) * int(srcB_shape[2]), int(srcB_shape[3]), int(srcB_shape[0])]

            reshape_srcA = tvm.relay.reshape(transpose_srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)

            # Batch MatMul
            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)

            # Reshape
            reshape_result = tvm.relay.reshape(
                result, newshape=[srcA_shape[1], srcA_shape[2], srcA_shape[0], srcB_shape[0]]
            )
            return reshape_result

        elif match_einsum_pattern("ibnd,snd->ibns", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            transpose_srcA = tvm.relay.transpose(srcA, axes=[2, 0, 1, 3])  # nibd
            transpose_srcB = tvm.relay.transpose(srcB, axes=[1, 2, 0])  # nds
            new_shape_srcA = [
                int(srcA_shape[2]),
                int(srcA_shape[0]) * int(srcA_shape[1]),
                int(srcA_shape[3]),
            ]  # n i*b d
            reshape_srcA = tvm.relay.reshape(transpose_srcA, newshape=new_shape_srcA)
            # Batch MatMul
            result = tvm.relay.nn.batch_matmul(
                reshape_srcA, transpose_srcB, transpose_a=False, transpose_b=False
            )  # n i*b s

            # Reshape
            reshape_result = tvm.relay.reshape(
                result, newshape=[srcA_shape[2], srcA_shape[0], srcA_shape[1], srcB_shape[0]]
            )
            reshape_result = tvm.relay.transpose(reshape_result, axes=[1, 2, 0, 3])
            return reshape_result

        elif match_einsum_pattern("ijbs,ibns->bnij", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            transpose_srcA = tvm.relay.transpose(
                srcA,
                axes=[
                    0,
                    2,
                    1,
                    3,
                ],
            )  # ibjs
            transpose_srcB = tvm.relay.transpose(srcB, axes=[0, 1, 3, 2])  # ibsn
            new_shape_srcA = [
                int(srcA_shape[0]) * int(srcA_shape[2]),
                int(srcA_shape[1]),
                int(srcA_shape[3]),
            ]  # i*b j s
            new_shape_srcB = [
                int(srcB_shape[0]) * int(srcB_shape[1]),
                int(srcB_shape[3]),
                int(srcB_shape[2]),
            ]  # i*b s n
            reshape_srcA = tvm.relay.reshape(transpose_srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)
            # MatMul
            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)  # ibjn

            # Reshape
            reshape_result = tvm.relay.reshape(
                result, newshape=[srcA_shape[0], srcA_shape[2], srcA_shape[1], srcB_shape[2]]
            )
            reshape_result = tvm.relay.transpose(reshape_result, axes=[1, 3, 0, 2])
            return reshape_result

        elif match_einsum_pattern("ijbn->bnij", equation):
            srcA = node_map[self.act][0][0]
            return tvm.relay.transpose(srcA, axes=[2, 3, 0, 1])
        elif match_einsum_pattern("bnij,jbnd->ibnd", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            transpose_srcB = tvm.relay.transpose(srcB, axes=[1, 2, 0, 3])  # bnjd
            new_shape_srcA = [
                int(srcA_shape[0]) * int(srcA_shape[1]),
                int(srcA_shape[2]),
                int(srcA_shape[3]),
            ]  # b*n i j
            new_shape_srcB = [
                int(srcB_shape[1]) * int(srcB_shape[2]),
                int(srcB_shape[0]),
                int(srcB_shape[3]),
            ]  # b*n j d
            reshape_srcA = tvm.relay.reshape(srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)

            if srcA_shape[-1] != srcB_shape[0] and srcB_shape[0] == 1:
                # broadcast
                new_shape_srcB[-2] = new_shape_srcB[-2] * srcA_shape[-1]
                reshape_srcB = tvm.relay.broadcast_to(reshape_srcB, new_shape_srcB)
            elif srcA_shape[-1] != srcB_shape[0]:
                # Dont know how to handle
                assert False, f"TVM einsum decomposition does not support {equation} with srcB shape {srcB_shape}."

            # MatMul
            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)

            # Reshape
            reshape_result = tvm.relay.reshape(
                result, newshape=[srcA_shape[0], srcA_shape[1], srcA_shape[2], srcB_shape[3]]
            )
            reshape_result = tvm.relay.transpose(reshape_result, axes=[2, 0, 1, 3])
            return reshape_result

        elif match_einsum_pattern("ibnd,hnd->ibh", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            transpose_srcB = tvm.relay.transpose(srcB, axes=[1, 2, 0])  # ibsn
            new_shape_srcA = [int(srcA_shape[0]) * int(srcA_shape[1]), int(srcA_shape[2]) * int(srcA_shape[3])]
            new_shape_srcB = [int(srcB_shape[1]) * int(srcB_shape[2]), int(srcB_shape[0])]
            reshape_srcA = tvm.relay.reshape(srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)

            # MatMul
            result = tvm.relay.nn.matmul(reshape_srcA, reshape_srcB)  # ibjn

            # Reshape
            reshape_result = tvm.relay.reshape(result, newshape=[srcA_shape[0], srcA_shape[1], srcB_shape[0]])
            return reshape_result

        elif match_einsum_pattern("bhd,bmd->bhmd", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            assert (
                srcA_shape[0] == srcB_shape[0]
            ), f"Only support {equation} with srcA/srcB shape being the same on dim b and d"
            assert (
                srcA_shape[-1] == srcB_shape[-1]
            ), f"Only support {equation} with srcA/srcB shape being the same on dim b and d"

            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            transpose_srcA = tvm.relay.transpose(
                srcA,
                axes=[
                    0,
                    2,
                    1,
                ],
            )  # dbh
            transpose_srcB = tvm.relay.transpose(
                srcB,
                axes=[
                    0,
                    2,
                    1,
                ],
            )  # bdm
            new_shape_srcA = [
                int(srcA_shape[0]) * int(srcA_shape[2]),
                int(srcA_shape[1]),
                1,
            ]
            new_shape_srcB = [
                int(srcB_shape[0]) * int(srcB_shape[2]),
                1,
                int(srcB_shape[1]),
            ]
            reshape_srcA = tvm.relay.reshape(transpose_srcA, newshape=new_shape_srcA)  # bxd h 1
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)  # bxd 1 m

            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)

            # Reshape
            reshape_result = tvm.relay.reshape(
                result, newshape=[srcA_shape[0], srcA_shape[2], srcA_shape[1], srcB_shape[1]]
            )
            reshape_result = tvm.relay.transpose(reshape_result, axes=[0, 2, 3, 1])
            return reshape_result

        elif match_einsum_pattern("ijk, qr -> ijr", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            assert len(srcA_shape) == 3 and len(srcB_shape) == 2, "input tensors have incorrect number of dimensions"

            A = tvm.relay.sum(A, axis=[2], keepdims=True)  # i j 1
            B = tvm.relay.sum(B, axis=[0], keepdims=True)  # 1 r

            B = tvm.relay.reshape(B, newshape=[1, srcB_shape[1], 1])  # 1 r 1

            result = tvm.relay.nn.batch_matmul(A, B)  #
            return tvm.relay.reshape(result, newshape=[srcA_shape[0], srcA_shape[1], srcB_shape[1]])

        elif match_einsum_pattern("ijk, kr -> ijr", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            assert len(srcA_shape) == 3 and len(srcB_shape) == 2, "input tensors have incorrect number of dimensions"

            A = tvm.relay.reshape(A, newshape=[srcA_shape[0] * srcA_shape[1], srcA_shape[2]])  # i*j k
            result = tvm.relay.nn.matmul(A, B)  # i*j r
            return tvm.relay.reshape(result, newshape=[srcA_shape[0], srcA_shape[1], srcB_shape[1]])  # i j r

        elif match_einsum_pattern("ij, qr -> i", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == len(srcB_shape) == 2, "input tensors have incorrect number of dimensions"

            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            # have to sum over each axis one by one for forge
            B_sum = tvm.relay.sum(srcB, axis=[0])
            B_sum = tvm.relay.sum(B_sum, axis=[0])

            A_transpose = tvm.relay.transpose(srcA, axes=[1, 0])
            A_sum = tvm.relay.sum(A_transpose, axis=[0])
            out = tvm.relay.multiply(B_sum, A_sum)
            out = tvm.relay.reshape(out, newshape=pre.checked_type.shape)

            return out

        elif match_einsum_pattern("ij, jk -> ik", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == len(srcB_shape) == 2, "input tensors have incorrect number of dimensions"

            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            return tvm.relay.nn.matmul(srcA, srcB)

        elif match_einsum_pattern("ij, kr -> ir", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == len(srcB_shape) == 2, "input tensors have incorrect number of dimensions"

            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            A_sum = tvm.relay.sum(srcA, axis=[1], keepdims=True)
            B_sum = tvm.relay.sum(srcB, axis=[0], keepdims=True)
            return tvm.relay.nn.matmul(A_sum, B_sum)

        elif match_einsum_pattern("abc, bde -> acde", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == len(srcB_shape) == 3, "input tensors have incorrect number of dimensions"

            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            A = tvm.relay.transpose(A, axes=[0, 2, 1])

            new_shape_A = [int(srcA_shape[0]) * int(srcA_shape[2]), int(srcA_shape[1])]  # a*c b
            new_shape_B = [int(srcB_shape[0]), int(srcB_shape[1]) * int(srcB_shape[2])]  # b d*e

            reshape_A = tvm.relay.reshape(A, newshape=new_shape_A)
            reshape_B = tvm.relay.reshape(B, newshape=new_shape_B)

            # MatMul
            result = tvm.relay.nn.matmul(reshape_A, reshape_B)  # a*c d*e

            # Reshape
            reshape_result = tvm.relay.reshape(
                result, newshape=[srcA_shape[0], srcA_shape[2], srcB_shape[1], srcB_shape[2]]
            )
            return reshape_result

        elif match_einsum_pattern("abcd, dcbe -> ae", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == len(srcB_shape) == 4, "input tensors have incorrect number of dimensions"

            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            B = tvm.relay.transpose(B, axes=[2, 1, 0, 3])

            new_shape_A = [int(srcA_shape[0]), int(srcA_shape[1]) * int(srcA_shape[2]) * int(srcA_shape[3])]  # a b*c*d
            new_shape_B = [int(srcB_shape[0]) * int(srcB_shape[1]) * int(srcB_shape[2]), int(srcB_shape[3])]  # b*c*d e

            A = tvm.relay.reshape(A, newshape=new_shape_A)
            B = tvm.relay.reshape(B, newshape=new_shape_B)

            # MatMul
            result = tvm.relay.nn.matmul(A, B)  # a e
            return result

        elif match_einsum_pattern("abc, bcd -> ad", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == len(srcB_shape) == 3, "input tensors have incorrect number of dimensions"

            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            new_shape_A = [int(srcA_shape[0]), int(srcA_shape[1]) * int(srcA_shape[2])]  # a b*c
            new_shape_B = [int(srcB_shape[0]) * int(srcB_shape[1]), int(srcB_shape[2])]  # b*c d

            reshape_A = tvm.relay.reshape(A, newshape=new_shape_A)
            reshape_B = tvm.relay.reshape(B, newshape=new_shape_B)

            # MatMul
            result = tvm.relay.nn.matmul(reshape_A, reshape_B)  # a d
            return result

        elif match_einsum_pattern("abc, bd -> acd", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == 3 and len(srcB_shape) == 2, "input tensors have incorrect number of dimensions"

            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            A = tvm.relay.transpose(A, axes=[0, 2, 1])  # a c b

            new_shape_A = [int(srcA_shape[0]) * int(srcA_shape[2]), int(srcA_shape[1])]  # a*c b

            A = tvm.relay.reshape(A, newshape=new_shape_A)  # a*c b

            # MatMul
            result = tvm.relay.nn.matmul(A, B)  # a*c b X b d = a*c d
            return tvm.relay.reshape(result, newshape=[srcA_shape[0], srcA_shape[2], srcB_shape[1]])

        elif match_einsum_pattern("ibnd,jbnd->ijbn", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            assert len(srcA_shape) == 4 and len(srcB_shape) == 4
            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            transpose_srcA = tvm.relay.transpose(A, axes=[1, 2, 0, 3])  # bnid
            transpose_srcB = tvm.relay.transpose(B, axes=[1, 2, 3, 0])  # bdm
            new_shape_srcA = [
                int(srcA_shape[1]) * int(srcA_shape[2]),
                int(srcA_shape[0]),
                int(srcA_shape[3]),
            ]  # b*n id
            new_shape_srcB = [
                int(srcB_shape[1]) * int(srcB_shape[2]),
                int(srcB_shape[3]),
                int(srcB_shape[0]),
            ]  # b*n dj
            reshape_srcA = tvm.relay.reshape(transpose_srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)

            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)
            result = tvm.relay.reshape(
                result,
                newshape=[
                    int(srcA_shape[1]),
                    int(srcA_shape[2]),
                    int(srcA_shape[0]),
                    int(srcB_shape[0]),
                ],
            )  # bnij

            return tvm.relay.transpose(result, axes=[2, 3, 0, 1])  # ijbn

        elif match_einsum_pattern("ijbn,jbnd->ibnd", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            assert len(srcA_shape) == 4 and len(srcB_shape) == 4
            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            transpose_srcA = tvm.relay.transpose(A, axes=[2, 3, 0, 1])  # bnij
            transpose_srcB = tvm.relay.transpose(B, axes=[1, 2, 0, 3])  # bnjd
            new_shape_srcA = [
                int(srcA_shape[2]) * int(srcA_shape[3]),
                int(srcA_shape[0]),
                int(srcA_shape[1]),
            ]  # b*n ij
            new_shape_srcB = [
                int(srcB_shape[1]) * int(srcB_shape[2]),
                int(srcB_shape[0]),
                int(srcB_shape[3]),
            ]  # b*n jd
            reshape_srcA = tvm.relay.reshape(transpose_srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)

            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)
            result = tvm.relay.reshape(
                result,
                newshape=[
                    int(srcA_shape[2]),
                    int(srcA_shape[3]),
                    int(srcA_shape[0]),
                    int(srcB_shape[3]),
                ],
            )  # bnid

            return tvm.relay.transpose(result, axes=[2, 0, 1, 3])  # ijbn

        elif match_einsum_pattern("jbki,jfki->jkbf", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            assert len(srcA_shape) == 4 and len(srcB_shape) == 4
            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            transpose_srcA = tvm.relay.transpose(A, axes=[0, 2, 1, 3])  # jkbi
            transpose_srcB = tvm.relay.transpose(B, axes=[0, 2, 3, 1])  # jkif
            new_shape_srcA = [
                int(srcA_shape[0]) * int(srcA_shape[2]),
                int(srcA_shape[1]),
                int(srcA_shape[3]),
            ]  # j*k bi
            new_shape_srcB = [
                int(srcB_shape[0]) * int(srcB_shape[2]),
                int(srcB_shape[3]),
                int(srcB_shape[1]),
            ]  # j*k if
            reshape_srcA = tvm.relay.reshape(transpose_srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)

            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)
            result = tvm.relay.reshape(
                result,
                newshape=[
                    int(srcA_shape[0]),
                    int(srcA_shape[2]),
                    int(srcA_shape[1]),
                    int(srcB_shape[1]),
                ],
            )  # jkbf

            return result

        elif match_einsum_pattern("f,bfde->fbde", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            assert len(srcA_shape) == 1 and len(srcB_shape) == 4
            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            reshape_srcA = tvm.relay.reshape(A, newshape=[1, int(srcA_shape[0]), 1, 1])  # 1 f 1 1
            result = tvm.relay.multiply(reshape_srcA, B)  # b f d e

            return tvm.relay.transpose(result, axes=[1, 0, 2, 3])  # fbde

        elif match_einsum_pattern("jikd,jkgi->jkdg", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape
            assert len(srcA_shape) == 4 and len(srcB_shape) == 4
            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            transpose_srcA = tvm.relay.transpose(A, axes=[0, 2, 3, 1])  # jkdi
            transpose_srcB = tvm.relay.transpose(B, axes=[0, 1, 3, 2])  # jkig
            new_shape_srcA = [
                int(srcA_shape[0]) * int(srcA_shape[2]),
                int(srcA_shape[3]),
                int(srcA_shape[1]),
            ]  # j*k bi
            new_shape_srcB = [
                int(srcB_shape[0]) * int(srcB_shape[1]),
                int(srcB_shape[3]),
                int(srcB_shape[2]),
            ]  # j*k if
            reshape_srcA = tvm.relay.reshape(transpose_srcA, newshape=new_shape_srcA)
            reshape_srcB = tvm.relay.reshape(transpose_srcB, newshape=new_shape_srcB)

            result = tvm.relay.nn.batch_matmul(reshape_srcA, reshape_srcB, transpose_a=False, transpose_b=False)
            result = tvm.relay.reshape(
                result,
                newshape=[
                    int(srcA_shape[0]),
                    int(srcA_shape[2]),
                    int(srcA_shape[3]),
                    int(srcB_shape[2]),
                ],
            )  # jkbf

            return result

        elif match_einsum_pattern("b i d, b j d -> b i j", equation):
            assert len(node_map[self.act][0]) == 2
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]

            result = tvm.relay.nn.batch_matmul(srcA, srcB, transpose_a=False, transpose_b=True)
            return result
        elif match_einsum_pattern("b i j, b j d -> b i d", equation):
            assert len(node_map[self.act][0]) == 2
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]
            result = tvm.relay.nn.batch_matmul(srcA, srcB, transpose_a=False, transpose_b=False)
            return result
        elif match_einsum_pattern("abcg,gf->abcf", equation):
            srcA = node_map[self.act][0][0]  # abcg
            srcB = node_map[self.act][0][1]  # gf
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == 4 and len(srcB_shape) == 2
            assert srcA_shape[-1] == srcB_shape[-2]

            srcA = tvm.relay.reshape(
                srcA, newshape=(srcA_shape[-4] * srcA_shape[-3], srcA_shape[-2], srcA_shape[-1])
            )  # a*b cg
            srcB = tvm.relay.reshape(srcB, newshape=(1, srcB_shape[-2], srcB_shape[-1]))  # 1gf
            res = tvm.relay.nn.batch_matmul(srcA, srcB, transpose_a=False, transpose_b=False)  # a*b cf
            res = tvm.relay.reshape(
                res, newshape=(srcA_shape[-4], srcA_shape[-3], srcA_shape[-2], srcB_shape[-1])
            )  # abcf

            # import torch
            # a = torch.rand((1, 1, 32, 32))
            # b = torch.rand((32, 64))
            # torch_res = torch.einsum("abcg,gf->abcf", a, b)

            # from tvm.relay.frontend.common import infer_shape
            # from tvm.relay.frontend.common import infer_value
            # from tvm.relay.frontend.common import analysis
            # tvm_res = infer_value(res, {'args_0': tvm.nd.array(a), 'Const': tvm.nd.array(b)})

            # assert np.allclose(torch_res.numpy(), tvm_res.numpy())

            return res
        elif match_einsum_pattern("af,cfe->ace", equation):
            srcA = node_map[self.act][0][0]  # af
            srcB = node_map[self.act][0][1]  # cfe
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == 2 and len(srcB_shape) == 3
            assert srcA_shape[-1] == srcB_shape[-2]

            srcA = tvm.relay.reshape(srcA, newshape=(1, srcA_shape[-2], srcA_shape[-1]))  # 1af
            bmm = tvm.relay.nn.batch_matmul(srcA, srcB, transpose_a=False, transpose_b=False)  # cae
            res = tvm.relay.transpose(bmm, axes=[1, 0, 2])  # ace

            # import torch
            # a = torch.rand((32, 64))
            # b = torch.rand((1, 64, 32))
            # torch_res = torch.einsum("af,cfe->ace", a, b)

            # from tvm.relay.frontend.common import infer_shape
            # from tvm.relay.frontend.common import infer_value
            # from tvm.relay.frontend.common import analysis

            # a = a.unsqueeze(0)
            # b = b.transpose(1, 2)
            # tvm_res = infer_value(res, {'args_0': tvm.nd.array(a), 'args_1': tvm.nd.array(b)})

            # assert np.allclose(torch_res.numpy(), tvm_res.numpy())

            return res
        elif match_einsum_pattern("abcd,->abcd", equation):
            srcA = node_map[self.act][0][0]  # abcd
            srcB = node_map[self.act][0][1]  #

            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            res = tvm.relay.multiply(srcA, srcB)

            # import torch
            # from tvm.relay.frontend.common import infer_value
            # a = torch.rand((1, 1, 768))
            # b = torch.rand((1, 1))
            # infer_value(res, {'args_0': tvm.nd.array(a), 'args_1': tvm.nd.array(b)})

            # a_val = infer_value(pre.args[0][0], {'args_0': tvm.nd.array(a), 'args_1': tvm.nd.array(b)})
            # b_val = infer_value(pre.args[0][1], {'args_2': tvm.nd.array(b)})

            # a_valt = torch.ones((1, 12, 1, 1))
            # b_valt = b.squeeze()
            # torch.einsum("abcd,->abcd", a_valt, b_valt)

            return res
        elif match_einsum_pattern("bqnc,bnchw->bqnhw", equation):
            # Ensure the einsum pattern is matched, and there are two input nodes
            assert len(node_map[self.act][0]) == 2

            srcA = node_map[self.act][0][0]  #  (b, q, n, c)
            srcB = node_map[self.act][0][1]  #  (b, n, c, h, w)

            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            # Expand dimensions of srcA to align with the einsum pattern
            srcA_expanded = tvm.relay.expand_dims(srcA, axis=-1)  # Shape: (b, q, n, c, 1)

            # Reshape srcB to collapse the last two spatial dimensions
            srcB_reduced = tvm.relay.reshape(
                srcB, newshape=(srcB_shape[0], srcB_shape[1], srcB_shape[2], srcB_shape[3] * srcB_shape[4])
            )
            # New shape: (b, n, c, h*w)

            # Expand dimensions of srcB to prepare for broadcasting
            srcB_expanded = tvm.relay.expand_dims(srcB_reduced, axis=1)  # Shape: (b, 1, n, c, h*w)

            # Perform element-wise multiplication between expanded tensors
            multiplied = tvm.relay.multiply(srcA_expanded, srcB_expanded)  # Shape: (b, q, n, c, h*w)

            # Sum along the c-axis to match einsum reduction
            result = tvm.relay.sum(multiplied, axis=3)  # Shape: (b, q, n, h*w)

            # Reshape the result tensor back to (b, q, n, h, w)
            result = tvm.relay.reshape(
                result, newshape=(srcA_shape[0], srcA_shape[1], srcA_shape[2], srcB_shape[3], srcB_shape[4])
            )

            return result

        elif match_einsum_pattern("nhwpqc->nchpwq", equation):
            # Ensure the einsum pattern is matched, and there should be one input node
            assert len(node_map[self.act][0]) == 1

            src = node_map[self.act][0][0]  # (n, h, w, p, q, c)
            result = tvm.relay.transpose(src, axes=[0, 5, 1, 3, 2, 4])  # (n, c, h, p, w, q)

            return result

        elif match_einsum_pattern("...si,...id->...sd", equation):
            from tvm.relay.frontend.common import infer_shape

            # Ensure the einsum pattern is matched, and there are two input nodes
            assert len(node_map[self.act][0]) == 2
            srcA = node_map[self.act][0][0]
            srcB = node_map[self.act][0][1]
            assert infer_shape(srcA)[-1] == infer_shape(srcB)[-2]
            return tvm.relay.nn.batch_matmul(srcA, srcB, transpose_a=False, transpose_b=False)

        elif match_einsum_pattern("bhwc,hkc->bhwk", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == 4 and len(srcB_shape) == 3

            assert srcA_shape[-1] == srcB_shape[-1]

            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            # Extract the batch dimension
            batch_dim = int(srcA_shape[0])

            # Case 1: Batched input
            if batch_dim > 1:

                # In the batched case, we need to process each batch independently
                # since einsum does not automatically broadcast batch dimensions.
                # We split the input tensor A along the batch dimension.
                a_split = tvm.relay.split(A, indices_or_sections=batch_dim, axis=0)

                results = []

                for batch_idx in range(batch_dim):
                    # Extract the tensor for the current batch: shape [1, H, W, C]
                    a_batch = a_split[batch_idx]

                    # Expand A: shape becomes [1, H, W, 1, C]
                    A_expanded = tvm.relay.expand_dims(a_batch, axis=3)

                    # Expand B: shape becomes [1, H, 1, K, C]
                    B_expanded = tvm.relay.expand_dims(B, axis=0)
                    B_expanded = tvm.relay.expand_dims(B_expanded, axis=2)

                    # Element-wise multiplication: shape [1, H, W, K, C]
                    multiplied = tvm.relay.multiply(A_expanded, B_expanded)

                    # Sum over the channel dimension (C): result shape [1, H, W, K]
                    result = tvm.relay.sum(multiplied, axis=[-1])

                    results.append(result)

                # Concatenate the batch results to form final output
                final_result = tvm.relay.concatenate(results, axis=0)

            # Case 2: Single-batch input (batch_dim == 1)
            else:

                # Expand A: shape becomes [1, H, W, 1, C]
                A_expanded = tvm.relay.expand_dims(A, axis=3)

                # Expand B: shape becomes [1, H, 1, K, C]
                B_expanded = tvm.relay.expand_dims(B, axis=0)
                B_expanded = tvm.relay.expand_dims(B_expanded, axis=2)

                # Element-wise multiplication: shape [1, H, W, K, C]
                multiplied = tvm.relay.multiply(A_expanded, B_expanded)

                # Sum over the channel dimension (C): result shape [1, H, W, K]
                final_result = tvm.relay.sum(multiplied, axis=[-1])

            return final_result

        elif match_einsum_pattern("bhwc,wkc->bhwk", equation):
            srcA_shape = pre.args[0][0].checked_type.shape
            srcB_shape = pre.args[0][1].checked_type.shape

            assert len(srcA_shape) == 4 and len(srcB_shape) == 3
            assert srcA_shape[-1] == srcB_shape[-1]

            A = node_map[self.act][0][0]
            B = node_map[self.act][0][1]

            b, h, w, c = [int(dim) for dim in srcA_shape]
            _, k, _ = [int(dim) for dim in srcB_shape]

            results = []

            # Iterate over the width dimension (W) of tensor A.
            for wi in range(w):
                # Slice A along width at index wi: shape becomes [B, H, 1, C]
                A_slice = tvm.relay.strided_slice(A, begin=[0, 0, wi, 0], end=[b, h, wi + 1, c])

                # Remove the singleton width dimension
                A_slice = tvm.relay.squeeze(A_slice, axis=[2])

                # Slice B at position wi: shape becomes [1, K, C] → squeeze to [K, C]
                B_slice = tvm.relay.strided_slice(B, begin=[wi, 0, 0], end=[wi + 1, k, c])
                B_slice = tvm.relay.squeeze(B_slice, axis=[0])

                # Rearrange the axes of B_slice and add a new leading dimension
                B_transposed = tvm.relay.transpose(B_slice, axes=[1, 0])
                B_expanded = tvm.relay.expand_dims(B_transposed, axis=0)

                # Perform batched matrix multiplication
                out_slice = tvm.relay.nn.batch_matmul(A_slice, B_expanded, transpose_b=False)

                # Add back the width dimension at axis=2
                out_slice = tvm.relay.expand_dims(out_slice, axis=2)

                results.append(out_slice)

            # Concatenate all the width slices to reconstruct full output tensor: [B, H, W, K]
            final_result = tvm.relay.concatenate(results, axis=2)

            return final_result

        elif match_einsum_pattern("bmchw,bnmc->bmhwn", equation):
            assert len(node_map[self.act][0]) == 2

            srcA = node_map[self.act][0][0]  # (b, m, c, h, w)
            srcB = node_map[self.act][0][1]  # (b, n, m, c)

            # Transpose B to match A's dims: (b, n, m, c) -> (b, m, c, n)
            B_perm = tvm.relay.transpose(srcB, axes=(0, 2, 3, 1))

            # Expand A: (b, m, c, h, w) → (b, m, c, h, w, 1)
            A_exp = tvm.relay.expand_dims(srcA, axis=-1)

            # Expand B: (b, m, c, n) → (b, m, c, 1, 1, n)
            B_exp = tvm.relay.expand_dims(B_perm, axis=3)  # (b, m, c, 1, n)
            B_exp = tvm.relay.expand_dims(B_exp, axis=3)  # (b, m, c, 1, 1, n)

            # Multiply and sum over `c`
            multiplied = tvm.relay.multiply(A_exp, B_exp)  # (b, m, c, h, w, n)
            result = tvm.relay.sum(multiplied, axis=2)  # sum over c → (b, m, h, w, n)

            return result

        elif match_einsum_pattern("bchw,bkc->bkhw", equation):
            assert len(node_map[self.act][0]) == 2
            A = node_map[self.act][0][0]  # (b, c, h, w)
            B = node_map[self.act][0][1]  # (b, k, c)

            A_shape = pre.args[0][0].checked_type.shape  # (b, c, h, w)
            B_shape = pre.args[0][1].checked_type.shape  # (b, k, c)

            b, c, h, w = A_shape
            _, k, _ = B_shape

            # Transpose A: (b, c, h, w) → (b, h, w, c)
            A_perm = tvm.relay.transpose(A, axes=(0, 2, 3, 1))  # (b, h, w, c)
            A_exp = tvm.relay.expand_dims(A_perm, axis=1)  # (b, 1, h, w, c)

            # Expand B to match (b, k, 1, 1, c)
            B_exp = tvm.relay.expand_dims(B, axis=2)  # (b, k, 1, c)
            B_exp = tvm.relay.expand_dims(B_exp, axis=2)  # (b, k, 1, 1, c)

            # Multiply and reduce over c
            multiplied = tvm.relay.multiply(A_exp, B_exp)
            result = tvm.relay.sum(multiplied, axis=4)  # (b, k, h, w)

            return result

        elif match_einsum_pattern("bmnk,bkmc->bnmc", equation):
            assert len(node_map[self.act][0]) == 2

            srcA = node_map[self.act][0][0]  # (b, m, n, k)
            srcB = node_map[self.act][0][1]  # (b, k, m, c)

            # Transpose B → (b, m, k, c) to align with A
            B_perm = tvm.relay.transpose(srcB, axes=(0, 2, 1, 3))

            # Expand A → (b, m, n, k, 1)
            A_exp = tvm.relay.expand_dims(srcA, axis=-1)

            # Expand B → (b, m, 1, k, c)
            B_exp = tvm.relay.expand_dims(B_perm, axis=2)

            # Multiply and reduce over k → (b, m, n, c)
            multiplied = tvm.relay.multiply(A_exp, B_exp)
            reduced = tvm.relay.sum(multiplied, axis=3)
            result = tvm.relay.transpose(reduced, axes=(0, 2, 1, 3))
            return result

        else:
            assert False, f"TVM einsum decomposition does not support {equation} yet."


class DecomposeRsqrt(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.in_a = wildcard()

        self.pattern = is_op("rsqrt")(self.in_a)

    def callback(self, pre, post, node_map):
        sqrt = tvm.relay.sqrt(post.args[0])
        rsqrt = tvm.relay.reciprocal(sqrt)
        return rsqrt


class InvertDivide(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.in_a = wildcard()
        self.in_b = wildcard()

        self.pattern = is_op("divide")(self.in_a, self.in_b)

    def callback(self, pre, post, node_map):
        rep = tvm.relay.reciprocal(post.args[1])
        return tvm.relay.multiply(post.args[0], rep)


class DecomposeLayoutTransform(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.input_tensor = wildcard()

        self.pattern = is_op("layout_transform")(self.input_tensor)

    def callback(self, pre, post, node_map):
        if post.attrs.src_layout == "NHWC" and post.attrs.dst_layout == "NCHW":
            act = node_map[self.input_tensor][0]

            axes = [0, 3, 1, 2]
            return tvm.relay.transpose(act, axes=axes)
        elif post.attrs.src_layout == "NCHW" and post.attrs.dst_layout == "NHWC":
            act = node_map[self.input_tensor][0]

            axes = [0, 2, 3, 1]
            return tvm.relay.transpose(act, axes=axes)
        else:
            return post


class ExplicateTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        self.input_tensor = wildcard()

        self.pattern = is_op("nn.batch_matmul")(wildcard(), wildcard())

    def callback(self, pre, post, node_map):
        transpose_a = post.attrs.transpose_a
        transpose_b = post.attrs.transpose_b

        if not (transpose_a or transpose_b):
            return post

        a = post.args[0]
        ndim = len(post.args[0].checked_type.shape)
        axes = list(range(ndim))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        if transpose_a:
            a = tvm.relay.transpose(a, axes=axes)
        b = post.args[1]
        if transpose_b:
            b = tvm.relay.transpose(b, axes=axes)

        return tvm.relay.nn.batch_matmul(a, b, transpose_a=False, transpose_b=False, out_dtype=post.checked_type.dtype)


class LowerAdaptiveAvgPool(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.input_tensor = wildcard()
        self.pattern = is_op("nn.adaptive_avg_pool2d")(self.input_tensor) | is_op("nn.adaptive_avg_pool1d")(
            self.input_tensor
        )

    def callback(self, pre, post, node_map):
        if node_map[self.pattern][0].op.name == "nn.adaptive_avg_pool2d":
            input_shape = [int(dim) for dim in post.args[0].checked_type.shape]
            output_shape = [int(dim) for dim in pre.checked_type.shape]

            assert post.attrs.layout == "NCHW"

            stride = [in_size // out_size for in_size, out_size in zip(input_shape[-2:], output_shape[-2:])]
            kernel = [
                in_size - (out_size - 1) * stride
                for in_size, out_size, stride in zip(input_shape[-2:], output_shape[-2:], stride)
            ]
            padding = 0

            return tvm.relay.nn.avg_pool2d(
                post.args[0],
                pool_size=kernel,
                strides=stride,
                padding=padding,
                count_include_pad=True,
            )
        elif node_map[self.pattern][0].op.name == "nn.adaptive_avg_pool1d":
            pre_node_map = construct_pre_node_map(self.pattern, pre)
            input_shape = [int(dim) for dim in pre_node_map[self.input_tensor][0].checked_type.shape]
            output_shape = [int(dim) for dim in pre_node_map[self.pattern][0].checked_type.shape]

            assert post.attrs.layout == "NCW"

            stride = [in_size // out_size for in_size, out_size in zip(input_shape[-1:], output_shape[-1:])]
            kernel = [
                in_size - (out_size - 1) * stride
                for in_size, out_size, stride in zip(input_shape[-1:], output_shape[-1:], stride)
            ]
            padding = 0

            return tvm.relay.nn.avg_pool1d(
                post.args[0],
                pool_size=kernel,
                strides=stride,
                padding=padding,
                count_include_pad=True,
            )


class LowerAdaptiveMaxPool(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.input_tensor = wildcard()
        self.pattern = is_op("nn.adaptive_max_pool2d")(wildcard())

    def callback(self, pre, post, node_map):
        input_expr = post.args[0]
        input_shape = [int(dim) for dim in post.args[0].checked_type.shape]
        output_shape = [int(dim) for dim in pre.checked_type.shape]

        assert post.attrs.layout == "NCHW"

        b, c, h, w = input_shape
        out_h, out_w = output_shape[2], output_shape[3]

        # Ensure output size is not larger than input size
        assert out_h <= h and out_w <= w, "AdaptiveMaxPool2d output size must be <= input size"

        # Use decomposition for non-square or asymmetric shapes
        if (h != w) or (out_h != out_w):

            pooled_rows = []

            # Loop over output height index
            for oh in range(out_h):
                # Input height slice range
                h_start = math.floor(oh * h / out_h)
                h_end = math.ceil((oh + 1) * h / out_h)

                row = []

                # Loop over output width index
                for ow in range(out_w):
                    # Input width slice range
                    w_start = math.floor(ow * w / out_w)
                    w_end = math.ceil((ow + 1) * w / out_w)

                    # Extract slice window from input
                    window = tvm.relay.strided_slice(
                        input_expr, begin=[0, 0, h_start, w_start], end=[b, c, h_end, w_end]
                    )

                    # Reduce over H and W using two-stage max
                    pooled_h = tvm.relay.max(window, axis=2, keepdims=True)  # (B, C, 1, W')
                    pooled = tvm.relay.max(pooled_h, axis=3, keepdims=True)  # (B, C, 1, 1)
                    row.append(pooled)

                # Concatenate width-wise
                row_cat = tvm.relay.concatenate(row, axis=3)
                pooled_rows.append(row_cat)

            # Concatenate height-wise
            result = tvm.relay.concatenate(pooled_rows, axis=2)
            return result

        input_size = input_shape[-1]
        output_size = output_shape[-1]

        stride = input_size // output_size
        kernel = input_size - (output_size - 1) * stride
        padding = 0

        return tvm.relay.nn.max_pool2d(
            post.args[0],
            pool_size=kernel,
            strides=stride,
            padding=padding,
        )


class DecomposeMultiDimSqueeze(DFPatternCallback):
    def __init__(self):
        super().__init__(require_type=True)
        self.act = wildcard()
        self.pattern = is_op("squeeze")(self.act)

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        axis = post.attrs.axis
        # Skip removal of squeeze which contain dynamic shapes
        if any([isinstance(dim, tvm.tir.expr.Any) for dim in pre.checked_type.shape]):
            return post
        input_shape = [int(dim) for dim in pre.args[0].checked_type.shape]
        if axis is None:
            # PyTorch's `squeeze_()` removes all dimensions of size 1.
            # Adjust all size-1 dimensions into negative axis indices.
            adjusted_axes = [i - len(input_shape) for i, dim in enumerate(input_shape) if dim == 1]
        else:
            adjusted_axes = [(ax - len(input_shape)) if ax >= 0 else ax for ax in axis]
        assert all(ax < 0 for ax in adjusted_axes), "Invalid squeeze dimension: all axes must be negative."
        for ax in sorted(adjusted_axes):
            act = tvm.relay.squeeze(act, axis=[ax])
        return act


class LowerSqueezeToReshape(DFPatternCallback):
    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        self.input_tensor = wildcard()
        self.pattern = is_op("squeeze")(wildcard())

    def callback(self, pre, post, node_map):
        # Skip removal of squeeze which contain dynamic shapes
        if any([isinstance(dim, tvm.tir.expr.Any) for dim in pre.checked_type.shape]):
            return post

        return tvm.relay.reshape(post.args[0], newshape=pre.checked_type.shape)


class TransposePad(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.pattern = is_op("nn.pad")(wildcard(), is_constant())

    def callback(self, pre, post, node_map):
        pad_width = [[int(pad) for pad in dim] for dim in post.attrs.pad_width]
        non_zero_dims = [idx for idx, pad in enumerate(pad_width) if pad != [0, 0]]
        assert len(non_zero_dims) <= 2

        pad_mode = post.attrs.pad_mode
        arg = post.args[0]
        arg = tvm.relay.nn.pad(arg, pad_width=pad_width, pad_value=post.args[1], pad_mode=pad_mode)
        return arg


class PopulateStridedSliceAxes(DFPatternCallback):
    def __init__(self, rewrite_once=True, require_type=True):
        super().__init__(rewrite_once=rewrite_once, require_type=require_type)
        self.input_tensor = wildcard()

        self.pattern = is_op("strided_slice")(wildcard())

    def callback(self, pre, post, node_map):
        if pre.attrs.axes is not None:
            return post

        input_shape = [int(dim) for dim in pre.args[0].checked_type.shape]
        output_shape = [int(dim) for dim in pre.checked_type.shape]

        begin = [int(dim) for dim in pre.attrs.begin]
        end = [int(dim) for dim in pre.attrs.end]
        stride = [int(dim) for dim in pre.attrs.strides]

        act = post.args[0]
        for dim, (in_dim, out_dim) in enumerate(zip(input_shape, output_shape)):
            if in_dim != out_dim:
                if pre.attrs.slice_mode == "size":
                    final_stride = None
                    final_end = (begin[dim] + end[dim],)
                else:
                    if len(stride) == len(input_shape):
                        final_stride = (stride[dim],)
                    else:
                        assert len(stride) == 1
                        final_stride = (stride[0],)  # Stride can be a length 1 list
                    final_end = (end[dim],)
                act = tvm.relay.strided_slice(
                    act, begin=(begin[dim],), end=final_end, strides=final_stride, axes=(dim,), slice_mode="end"
                )

        return act


class ConvertAddToBiasAddAfterConv2d(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.input = wildcard()
        self.weight = wildcard()
        self.bias = wildcard()
        self.act = is_op("nn.conv2d")(self.input, self.weight) | is_op("nn.conv2d_transpose")(self.input, self.weight)
        self.pattern = is_op("add")(self.act, self.bias)

    def callback(self, pre, post, node_map):
        bias = node_map[self.bias][0]
        act = node_map[self.act][0]
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        # Check if it's either a bias in form of var or a constant, or evan a reshape
        # which follows bias.
        if not (
            isinstance(bias, (tvm.relay.expr.Var, tvm.relay.expr.Constant))
            or bias.op.name == "reshape"
            and isinstance(bias.args[0], (tvm.relay.expr.Var, tvm.relay.expr.Constant))
        ):
            return post

        # Skip reshape if follows bias
        if (
            not isinstance(bias, (tvm.relay.expr.Var, tvm.relay.expr.Constant))
            and isinstance(bias.args[0], (tvm.relay.expr.Var, tvm.relay.expr.Constant))
            and bias.op.name == "reshape"
        ):
            bias = bias.args[0]
            bias_shape = list(pre_node_map[self.bias][0].args[0].checked_type.shape)
        else:
            bias_shape = list(pre_node_map[self.bias][0].checked_type.shape)

        if act.attrs.data_layout == "NHWC":
            single_dim = True
            for i in bias_shape[:-1]:
                single_dim = single_dim and i == 1
            if single_dim:
                bias = tvm.relay.reshape(bias, [bias_shape[-1]])
            return tvm.relay.nn.bias_add(act, bias, axis=-1)
        elif act.attrs.data_layout == "NCHW":
            single_dim = True
            for i in bias_shape[:-3] + bias_shape[-2:]:
                single_dim = single_dim and i == 1
            if single_dim and len(bias_shape) >= 3:
                bias = tvm.relay.reshape(bias, [bias_shape[-3]])
            return tvm.relay.nn.bias_add(act, bias)
        else:
            raise NotImplementedError(f"Unhandled data layout: {act.attrs.data_layout}")


class ConvertAddToBiasAddAfterConv2dTFWithChannelFirst(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.bias = wildcard()

        self.act = wildcard()
        self.t_act1 = is_op("transpose")(self.act)
        self.t_act2 = is_op("transpose")(self.t_act1)

        self.weight = wildcard()
        self.t_weight1 = is_op("transpose")(self.weight)
        self.t_weight2 = is_op("transpose")(self.t_weight1)
        self.t_weight3 = is_op("transpose")(self.t_weight2)

        self.conv = is_op("nn.conv2d")(self.t_act2, self.t_weight3)
        self.t_conv1 = is_op("transpose")(self.conv)
        self.t_conv2 = is_op("transpose")(self.t_conv1)

        self.add = is_op("add")(self.t_conv2, self.bias)

        self.relu = is_op("nn.relu")(self.add)
        self.t_relu1 = is_op("transpose")(self.relu)
        self.t_relu2 = is_op("transpose")(self.t_relu1)

        self.pattern = self.t_relu2

    def callback(self, pre, post, node_map):
        bias = node_map[self.bias][0]
        conv_act = node_map[self.conv][0]

        bias_add = tvm.relay.nn.bias_add(conv_act, bias)
        relu = tvm.relay.nn.relu(bias_add)

        return relu


class RemoveRedundantTranposesBetwenAvgPoolAndFlatteningReshape(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.avg_pool = is_op("nn.avg_pool2d")(self.act)
        self.t_ap1 = is_op("transpose")(self.avg_pool)
        self.t_ap2 = is_op("transpose")(self.t_ap1)
        self.reshape = is_op("reshape")(self.t_ap2)

        self.pattern = self.reshape

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]

        avg_pool = node_map[self.avg_pool][0]
        t1 = node_map[self.t_ap1][0]
        t2 = node_map[self.t_ap2][0]
        flatten_reshape = node_map[self.reshape][0]

        avg_pool_shape = list(avg_pool.checked_type.shape)
        t1_shape = list(t1.checked_type.shape)
        t2_shape = list(t2.checked_type.shape)
        flatten_reshape_shape = list(flatten_reshape.checked_type.shape)

        if not (
            avg_pool_shape == t1_shape
            and avg_pool_shape[-1] == t2_shape[-3]
            and flatten_reshape_shape == [1, avg_pool_shape[-1] * avg_pool_shape[-2] * avg_pool_shape[-3]]
        ):
            return post

        new_avg_pool = tvm.relay.nn.avg_pool2d(
            act,
            pool_size=avg_pool.attrs["pool_size"],
            strides=avg_pool.attrs["strides"],
            dilation=avg_pool.attrs["dilation"],
            padding=avg_pool.attrs["padding"],
            layout=avg_pool.attrs["layout"],
            out_layout=avg_pool.attrs["out_layout"],
            ceil_mode=avg_pool.attrs["ceil_mode"],
            count_include_pad=avg_pool.attrs["count_include_pad"],
        )
        flatten = tvm.relay.reshape(new_avg_pool, newshape=flatten_reshape_shape)

        return flatten


class EnsureKeepdims(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)

        self.pattern = is_op("sum")(wildcard()) | is_op("mean")(wildcard())

    def callback(self, pre, post, node_map):
        if post.attrs.keepdims == 0:
            act = post.args[0]
            if node_map[self.pattern][0].op.name == "sum":
                reduce_ = tvm.relay.sum(
                    act,
                    axis=post.attrs.axis,
                    keepdims=True,
                )
            else:
                reduce_ = tvm.relay.mean(
                    act,
                    axis=post.attrs.axis,
                    keepdims=True,
                )
            result = tvm.relay.squeeze(reduce_, axis=post.attrs.axis)
            return result
        else:
            return post


class DecomposeBatchFlatten(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.pattern = is_op("nn.batch_flatten")(self.act)

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        input_shape = list(pre.args[0].checked_type.shape)
        target_shape = [input_shape[0]] + [math.prod(input_shape[1:])]

        return tvm.relay.reshape(act, newshape=target_shape)


class ConvertExpandDimsToReshape(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.pattern = is_op("expand_dims")(self.act)

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        target_shape = list(pre.checked_type.shape)

        return tvm.relay.reshape(act, newshape=target_shape)


class ConvertGlobalAvgPool2dtoAvgPool2d(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.pattern = is_op("nn.global_avg_pool2d")(wildcard())

    def callback(self, pre, post, node_map):
        strides = (1, 1)
        pool_size = pre.args[0].checked_type.shape[2:]
        layout = post.attrs.layout
        act = post.args[0]

        avg_pool2d = tvm.relay.op.nn.avg_pool2d(
            act,
            pool_size=pool_size,
            strides=strides,
            layout=layout,
        )
        return avg_pool2d


class ConvertUpsampleToResize2d(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.pattern = is_op("nn.upsampling")(wildcard())

    def callback(self, pre, post, node_map):
        method = pre.attrs.method
        align_corners = pre.attrs.align_corners
        assert pre.attrs.layout == "NCHW", "Only support NCHW layout for upsample2d"

        target_shape = pre.checked_type.shape[-2:]

        if method == "nearest_neighbor":
            coord_trans = "asymmetric"
        elif align_corners:
            coord_trans = "align_corners"
        else:
            coord_trans = "half_pixel"

        return tvm.relay.image.resize2d(
            post.args[0],
            size=target_shape,
            layout="NCHW",
            method=method,
            coordinate_transformation_mode=coord_trans,
            cubic_alpha=-0.75,
        )


class ReconstructOnnxResize2d(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.reshape1 = is_op("reshape")(self.act)
        self.bcast_to = is_op("broadcast_to")(self.reshape1)
        self.pattern = is_op("reshape")(self.bcast_to)

    def callback(self, pre, post, node_map):

        reshape1 = node_map[self.reshape1][0]
        act = reshape1.args[0]
        bcast_to = node_map[self.bcast_to][0]
        reshape2 = node_map[self.pattern][0]

        act_shape = [int(dim) for dim in act.checked_type.shape]
        reshape1_shape = [int(dim) for dim in reshape1.attrs.newshape]
        bcast_shape = [int(dim) for dim in bcast_to.attrs.shape]
        reshape2_shape = [int(dim) for dim in reshape2.attrs.newshape]

        if len(act_shape) != 4 or len(reshape1_shape) != 6 or len(bcast_shape) != 6 or len(reshape2_shape) != 4:
            return post

        if (
            reshape2_shape[-1] != bcast_shape[-1] * act_shape[-1]
            or reshape2_shape[-2] != bcast_shape[-3] * act_shape[-2]
        ):
            return post

        if reshape2_shape[-3] != act_shape[-3] or reshape2_shape[-4] != act_shape[-4]:
            return post

        new_resize = tvm.relay.image.resize2d(
            act,
            size=reshape2_shape[-2:],
            layout="NCHW",
            method="nearest_neighbor",
        )
        return new_resize


class DecomposeMultiIndexAdvIndex(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.pattern = is_op("adv_index")(wildcard())

    def callback(self, pre, post, node_map):
        if len(pre.args[0].fields) == 2:
            return pre

        data = pre.args[0].fields[0]
        dims_to_drop = len(pre.args[0].fields) - 2
        assert all(
            [int(dim) == 1 for dim in data.checked_type.shape[:dims_to_drop]]
        ), "Dim to drop needs to be singleton"
        squeeze = tvm.relay.op.reshape(data, data.checked_type.shape[dims_to_drop:])
        index = tvm.relay.op.adv_index((squeeze, pre.args[0].fields[-1]))
        unsqueeze = tvm.relay.op.reshape(index, pre.checked_type.shape)

        return unsqueeze


class DecomposeErf(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.pattern = is_op("erf")(self.act)

    def callback(self, pre, post, node_map):
        act = post.args[0]
        act_sign = tvm.relay.op.sign(act)
        act_abs = tvm.relay.op.abs(act)

        # constants
        a1 = tvm.relay.expr.const(0.254829592)
        a2 = tvm.relay.expr.const(-0.284496736)
        a3 = tvm.relay.expr.const(1.421413741)
        a4 = tvm.relay.expr.const(-1.453152027)
        a5 = tvm.relay.expr.const(1.061405429)
        p = tvm.relay.expr.const(0.3275911)
        one = tvm.relay.expr.const(1.0)
        minus_one = tvm.relay.expr.const(-1.0)

        t = one / (one + p * act_abs)
        y = one - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * tvm.relay.op.exp(
            (minus_one * act_abs) * act_abs
        )
        res = act_sign * y  # erf(-x) = -erf(x)

        return res


class ReconstructTFGelu(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.sqrt_half = wildcard()
        self.half = is_constant()
        self.one = is_constant()
        times_root_two = is_op("multiply")(self.act, self.sqrt_half)
        erf = is_op("erf")(times_root_two)
        times_half = is_op("multiply")(self.act, self.half)
        add = is_op("add")(self.one, erf)
        gelu = is_op("multiply")(times_half, add)

        self.pattern = gelu

    def callback(self, pre, post, node_map):
        one = node_map[self.one][0].data.numpy()
        half = node_map[self.half][0].data.numpy()
        sqrt_half = node_map[self.sqrt_half][0]

        # Relay graph may use sqrt(1/2) outright, or take the recipricoral of sqrt(2)
        if isinstance(sqrt_half, tvm.relay.expr.Constant):
            sqrt_half = sqrt_half.data.numpy()
            root_two_multiplied = math.isclose(sqrt_half, 0.70710677, rel_tol=1e-6, abs_tol=1e-6)
        else:
            sqrt_half = sqrt_half.args[0].data.numpy()
            root_two_multiplied = math.isclose(sqrt_half, 1.4142135, rel_tol=1e-6, abs_tol=1e-6)

        one_added = math.isclose(one, 1.0, rel_tol=1e-6, abs_tol=1e-6)
        half_multiplied = math.isclose(half, 0.5, rel_tol=1e-6, abs_tol=1e-6)

        if not (one_added and half_multiplied and root_two_multiplied):
            return post

        return tvm.relay.gelu(node_map[self.act][0])


class ReconstructOnnxGelu(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.one_over_root_two = wildcard()
        self.one = is_constant()
        self.half_added = is_constant()

        times_root_two = is_op("multiply")(self.act, self.one_over_root_two)
        erf = is_op("erf")(times_root_two)
        add = is_op("add")(erf, self.one)
        mult2 = is_op("multiply")(self.act, add)
        gelu = is_op("multiply")(mult2, self.half_added)

        self.pattern = gelu

    def callback(self, pre, post, node_map):
        half_added = math.isclose(node_map[self.half_added][0].data.numpy(), 0.5, rel_tol=1e-6, abs_tol=1e-6)
        one_added = math.isclose(node_map[self.one][0].data.numpy(), 1.0, rel_tol=1e-6, abs_tol=1e-6)

        sqrt_half = node_map[self.one_over_root_two][0]
        # Relay graph may use sqrt(1/2) outright, or take the recipricoral of sqrt(2)
        if isinstance(sqrt_half, tvm.relay.expr.Constant):
            sqrt_half = sqrt_half.data.numpy()
            root_two_multiplied = math.isclose(sqrt_half, 0.70710677, rel_tol=1e-6, abs_tol=1e-6)
        else:
            sqrt_half = sqrt_half.args[0].data.numpy()
            root_two_multiplied = math.isclose(sqrt_half, 1.4142135, rel_tol=1e-6, abs_tol=1e-6)

        if not (half_added and one_added and root_two_multiplied):
            return post

        return tvm.relay.gelu(node_map[self.act][0])


class ReconstructOnnxQuantizedGelu(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.one_over_root_two = wildcard()
        self.one = wildcard()
        self.half_added = wildcard()

        self.act_0 = is_op("qnn.dequantize")(
            self.act,
            wildcard(),
            wildcard(),
        )
        self.act_1 = is_op("qnn.dequantize")(
            self.act,
            wildcard(),
            wildcard(),
        )

        times_root_two = is_op("multiply")(self.act_0, self.one_over_root_two)
        erf = is_op("erf")(times_root_two)

        # CHECK IF WE NEED IT
        # quantize_erf = is_op("qnn.quantize")(erf, wildcard(), wildcard(),)
        # dequantize_erf = is_op("qnn.dequantize")(quantize_erf, wildcard(), wildcard(),)

        add = is_op("add")(erf, self.one)

        # CHECK
        # quantize_add = is_op("qnn.quantize")(add, wildcard(), wildcard(),)
        # dequantize_add = is_op("qnn.dequantize")(quantize_add, wildcard(), wildcard(),)

        mult2 = is_op("multiply")(self.act_1, add)

        # Check
        # quantize_mult2 = is_op("qnn.quantize")(mult2, wildcard(), wildcard(),)
        # dequantize_mult2 = is_op("qnn.dequantize")(quantize_mult2, wildcard(), wildcard(),)

        gelu = is_op("multiply")(mult2, self.half_added)

        self.pattern = gelu

    def callback(self, pre, post, node_map):
        if isinstance(node_map[self.half_added][0], tvm.relay.expr.Constant):
            half_added = math.isclose(node_map[self.half_added][0].data.numpy(), 0.5, rel_tol=1e-6, abs_tol=1e-6)
        elif isinstance(node_map[self.half_added][0].args[0], tvm.relay.expr.Constant):
            # Compute dequant
            op = node_map[self.half_added][0]
            assert op.op.name == "qnn.dequantize"
            input_int = op.args[0].data.numpy()
            input_scale = op.args[1].data.numpy()
            input_zp = op.args[2].data.numpy()
            float_ = (float)(input_int - input_zp) * input_scale
            half_added = math.isclose(float_, 0.5, rel_tol=1e-6, abs_tol=1e-6)
        else:
            return post

        if isinstance(node_map[self.one][0], tvm.relay.expr.Constant):
            one_added = math.isclose(node_map[self.one][0].data.numpy(), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        elif isinstance(node_map[self.one][0].args[0], tvm.relay.expr.Constant):
            # Compute dequant
            op = node_map[self.one][0]
            assert op.op.name == "qnn.dequantize"
            input_int = op.args[0].data.numpy()
            input_scale = op.args[1].data.numpy()
            input_zp = op.args[2].data.numpy()
            float_ = (float)(input_int - input_zp) * input_scale
            one_added = math.isclose(float_, 1.0, rel_tol=1e-6, abs_tol=1e-6)
        else:
            return post

        sqrt_half = node_map[self.one_over_root_two][0]
        # Relay graph may use sqrt(1/2) outright, or take the recipricoral of sqrt(2)
        if isinstance(sqrt_half, tvm.relay.expr.Constant):
            sqrt_half = sqrt_half.data.numpy()
            root_two_multiplied = math.isclose(sqrt_half, 0.70710677, rel_tol=1e-6, abs_tol=1e-6)
        else:
            sqrt_half = sqrt_half.args[0].data.numpy()
            root_two_multiplied = math.isclose(sqrt_half, 1.4142135, rel_tol=1e-6, abs_tol=1e-6)

        if not (half_added and one_added and root_two_multiplied):
            return post

        quantize_act = node_map[self.act][0]
        original_dequant = node_map[self.act_0][0]
        dequant_act = tvm.relay.qnn.op.dequantize(quantize_act, original_dequant.args[1], original_dequant.args[2])
        return tvm.relay.gelu(dequant_act)


class DecomposeQnnConcat(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.pattern = is_op("qnn.concatenate")(
            wildcard(),
            wildcard(),
            wildcard(),
            wildcard(),
            wildcard(),
        )

    def callback(self, pre, post, node_map):
        data = post.args[0]
        input_scales = post.args[1]
        input_zps = post.args[2]
        output_scale = post.args[3]
        output_zp = post.args[4]

        assert len(input_scales) == len(input_zps) == len(data)
        new_concat_inputs = []
        for i in range(len(data)):
            if (
                input_scales[i].data.numpy() == output_scale.data.numpy()
                and input_zps[i].data.numpy() == output_zp.data.numpy()
            ):
                new_concat_inputs.append(data[i])
            else:
                # Insert requant
                inp = tvm.relay.qnn.op.requantize(
                    data[i],
                    input_scale=input_scales[i],
                    input_zero_point=input_zps[i],
                    output_scale=output_scale,
                    output_zero_point=output_zp,
                    out_dtype=post.checked_type.dtype,
                )
                new_concat_inputs.append(inp)

        return tvm.relay.concatenate(new_concat_inputs, axis=post.attrs.axis)


class ReconstructPyTorchGeluNew(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        square = is_op("multiply")(self.act, self.act)
        pow_3 = is_op("multiply")(square, self.act)
        self.c_0044715 = is_constant()
        times_const = is_op("multiply")(pow_3, self.c_0044715)
        addition = is_op("add")(self.act, times_const)
        self.root_two_over_pie = is_constant()
        times_root_2_over_pie = is_op("multiply")(addition, self.root_two_over_pie)
        tanh = is_op("tanh")(times_root_2_over_pie)
        self.one = is_constant()
        tanh_plus_one = is_op("add")(tanh, self.one)
        self.one_half = is_constant()
        divide_by_two = is_op("multiply")(self.act, self.one_half)
        new_gelu = is_op("multiply")(divide_by_two, tanh_plus_one)

        self.pattern = new_gelu

    def callback(self, pre, post, node_map):
        constnats_correct = True
        constnats_correct = constnats_correct and math.isclose(
            node_map[self.c_0044715][0].data.numpy(), 0.044715, rel_tol=1e-6, abs_tol=1e-6
        )
        constnats_correct = constnats_correct and math.isclose(
            node_map[self.root_two_over_pie][0].data.numpy(), math.sqrt(2.0 / math.pi), rel_tol=1e-6, abs_tol=1e-6
        )
        constnats_correct = constnats_correct and math.isclose(
            node_map[self.one][0].data.numpy(), 1, rel_tol=1e-6, abs_tol=1e-6
        )
        constnats_correct = constnats_correct and math.isclose(
            node_map[self.one_half][0].data.numpy(), 0.5, rel_tol=1e-6, abs_tol=1e-6
        )

        if not constnats_correct:
            return post

        return tvm.relay.gelu(node_map[self.act][0], approximate="tanh")


class ReconstructPyTorchGelu(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.one_over_root_two = is_constant()
        self.half_multiplied = is_constant()
        self.half_added = is_constant()

        times_root_two = is_op("multiply")(self.act, self.one_over_root_two)
        erf = is_op("erf")(times_root_two)
        times_half = is_op("multiply")(erf, self.half_multiplied)
        add = is_op("add")(self.half_added, times_half)
        gelu = is_op("multiply")(self.act, add)

        self.pattern = gelu

    def callback(self, pre, post, node_map):
        half_added = math.isclose(node_map[self.half_added][0].data.numpy(), 0.5, rel_tol=1e-6, abs_tol=1e-6)
        half_multiplied = math.isclose(node_map[self.half_multiplied][0].data.numpy(), 0.5, rel_tol=1e-6, abs_tol=1e-6)
        root_two_multiplied = math.isclose(
            node_map[self.one_over_root_two][0].data.numpy(), 0.70710677, rel_tol=1e-6, abs_tol=1e-6
        )

        if not (half_added and half_multiplied and root_two_multiplied):
            return post

        return tvm.relay.gelu(node_map[self.act][0])


class ReconstructJaxGelu(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.act = wildcard()
        self.sqrt_root = wildcard()
        self.one = is_constant()
        self.two = is_constant()

        reciprocal_root = is_op("reciprocal")(self.sqrt_root)
        times_root_two = is_op("multiply")(self.act, reciprocal_root)
        erf = is_op("erf")(times_root_two)
        add_one = is_op("add")(erf, self.one)
        times_act = is_op("multiply")(self.act, add_one)
        reciprocal_two = is_op("reciprocal")(self.two)
        times_half = is_op("multiply")(times_act, reciprocal_two)

        self.pattern = times_half

    def callback(self, pre, post, node_map):
        reciprocal_sqrt_root = math.isclose(
            node_map[self.sqrt_root][0].data.numpy(), 1.41421356, rel_tol=1e-6, abs_tol=1e-6
        )
        one_added = math.isclose(node_map[self.one][0].data.numpy(), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        reciprocal_two = math.isclose(node_map[self.two][0].data.numpy(), 2.0, rel_tol=1e-6, abs_tol=1e-6)

        if not (reciprocal_sqrt_root and one_added and reciprocal_two):
            return post

        return tvm.relay.gelu(node_map[self.act][0])


class SimplifyGroupNorm(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.reshape0 = is_op("reshape")(self.act)
        self.mean = is_op("mean")(self.reshape0)
        self.sub = is_op("subtract")(self.reshape0, self.mean)
        self.var = is_op("variance")(self.reshape0, self.mean)
        self.add = is_op("add")(self.var, wildcard())
        self.sqrt = is_op("sqrt")(self.add)
        self.div = is_op("divide")(self.sub, self.sqrt)
        self.reshape1 = is_op("reshape")(self.div)
        self.pattern = self.reshape1

    def callback(self, pre, post, node_map):
        if len(node_map[self.reshape0][0].attrs.newshape) <= 4:
            # Dont need decomposition
            return post

        if len(node_map[self.reshape1][0].attrs.newshape) > 4:
            # Cannot decompose
            return post

        old_mean = node_map[self.mean][0]
        old_var = node_map[self.var][0]
        # Handle 5D/6D between the 2 reshapes
        arg = node_map[self.act][0]
        new_shape = node_map[self.reshape0][0].attrs.newshape
        new_shape = new_shape[:3] + [np.prod(new_shape[3:])]
        new_reshape0 = tvm.relay.reshape(arg, newshape=new_shape)
        new_mean = tvm.relay.mean(
            new_reshape0, axis=[-2, -1], keepdims=old_mean.attrs.keepdims, exclude=old_mean.attrs.exclude
        )
        new_sub = tvm.relay.subtract(new_reshape0, new_mean)
        new_var = _make._variance(
            new_reshape0, new_mean, [-2, -1], old_var.attrs.keepdims, old_var.attrs.exclude, old_var.attrs.unbiased
        )
        new_add = tvm.relay.add(new_var, node_map[self.add][0].args[1])
        new_sqrt = tvm.relay.sqrt(new_add)
        new_div = tvm.relay.divide(new_sub, new_sqrt)
        new_reshape1 = tvm.relay.reshape(new_div, node_map[self.reshape1][0].attrs.newshape)

        return new_reshape1


class ReconstructPyTorchLayerNorm(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.gamma = wildcard()
        self.beta = wildcard()
        self.eps = is_constant()

        self.mean_act = is_op("mean")(self.act)
        sub_0 = is_op("subtract")(self.act, self.mean_act)
        mul_0 = is_op("multiply")(sub_0, sub_0)
        var = is_op("mean")(mul_0)

        sum_denom = var.optional(lambda x: is_op("add")(x, self.eps))
        sub = is_op("subtract")(self.act, self.mean_act)
        denom = is_op("sqrt")(sum_denom)
        recp = is_op("reciprocal")(denom)
        coef = is_op("multiply")(sub, recp)
        mul = is_op("multiply")(coef, self.gamma)
        layernorm = is_op("add")(mul, self.beta)

        self.pattern = layernorm

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        gamma = node_map[self.gamma][0]
        beta = node_map[self.beta][0]

        try:
            eps = node_map[self.eps][0].data.asnumpy().item()
        except TVMError:  # Does not have epsilon addition
            eps = 0

        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act_shape = pre_node_map[self.act][0].checked_type.shape
        gamma_shape = list(pre_node_map[self.gamma][0].checked_type.shape)

        axis = pre_node_map[self.mean_act][0].attrs.axis
        assert len(axis) == 1, "TVM Layernorm only supports single dim"
        if axis[0] >= 0:
            layernorm_axis = int(axis[0] - len(act_shape))
        else:
            layernorm_axis = int(axis[0])

        if layernorm_axis != -1:
            return post

        if np.prod(gamma_shape) != act_shape[layernorm_axis]:
            return post

        if np.prod(pre_node_map[self.beta][0].checked_type.shape) != act_shape[layernorm_axis]:
            return post

        return tvm.relay.layernorm(act, gamma, beta, eps, layernorm_axis)


class ReconstructTFLayerNorm(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.gamma = wildcard()
        self.beta = wildcard()
        self.eps = is_constant()

        self.mean_act = is_op("mean")(self.act)
        sub_0 = is_op("subtract")(self.act, self.mean_act)
        mul_0 = is_op("multiply")(sub_0, sub_0)
        var = is_op("mean")(mul_0)

        sum_denom = var.optional(lambda x: is_op("add")(x, self.eps))
        denom = is_op("sqrt")(sum_denom)
        recp = is_op("reciprocal")(denom)

        weight = is_op("multiply")(self.gamma, recp)
        mean_part = is_op("multiply")(self.mean_act, weight)
        act_part = is_op("multiply")(weight, self.act)
        sub_1 = is_op("subtract")(self.beta, mean_part)
        layernorm = is_op("add")(sub_1, act_part)
        self.pattern = layernorm

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        gamma = node_map[self.gamma][0]
        beta = node_map[self.beta][0]

        try:
            eps = node_map[self.eps][0].data.numpy().item()
        except TVMError:  # Does not have epsilon addition
            eps = 0

        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act_shape = pre_node_map[self.act][0].checked_type.shape
        gamma_shape = pre_node_map[self.gamma][0].checked_type.shape
        beta_shape = pre_node_map[self.beta][0].checked_type.shape

        if len(gamma_shape) > 1 and sum([1 if int(x) != 1 else 0 for x in list(gamma_shape)]) == 1:
            # Count the number of dims thats not 1
            gamma_shape = (np.prod([int(x) for x in gamma_shape]),)
            gamma = tvm.relay.reshape(gamma, newshape=gamma_shape)
        else:
            assert len(gamma_shape) == 1, "TVM Layernorm only supports single dim"

        if len(beta_shape) > 1 and sum([1 if int(x) != 1 else 0 for x in list(beta_shape)]) == 1:
            # Count the number of dims thats not 1
            beta_shape = (np.prod([int(x) for x in beta_shape]),)
            beta = tvm.relay.reshape(beta, newshape=gamma_shape)
        else:
            assert len(beta_shape) == 1, "TVM Layernorm only supports single dim"

        axis = pre_node_map[self.mean_act][0].attrs.axis
        assert len(axis) == 1, "TVM Layernorm only supports single dim"
        if axis[0] >= 0:
            layernorm_axis = int(axis[0] - len(act_shape))
        else:
            layernorm_axis = int(axis[0])

        if layernorm_axis != -1:
            return post

        if np.prod(gamma_shape) != act_shape[layernorm_axis]:
            return post

        if np.prod(pre_node_map[self.beta][0].checked_type.shape) != act_shape[layernorm_axis]:
            return post

        return tvm.relay.layernorm(act, gamma, beta, eps, layernorm_axis)


class ReconstructJaxLayerNorm(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()

        # Calculate mean and var
        ## Mean2
        mean2_act_mult = is_op("multiply")(self.act, self.act)
        mean2_act_sum = is_op("sum")(mean2_act_mult)
        mean2_act_reshape = is_op("reshape")(mean2_act_sum)
        mean2_dim_shape_const = is_constant()  # self.axis_val
        mean2_div = is_op("reciprocal")(mean2_dim_shape_const)
        mean2 = is_op("multiply")(mean2_act_reshape, mean2_div)

        ## Mean
        mean_act_sum = is_op("sum")(self.act)
        mean_act_reshape = is_op("reshape")(mean_act_sum)
        self.axis_val = is_constant()
        mean_div = is_op("reciprocal")(self.axis_val)
        self.mean = is_op("multiply")(mean_act_reshape, mean_div)
        abs_mean = is_op("multiply")(self.mean, self.mean)

        ## Var
        mean2_sub_abs_mean = is_op("subtract")(mean2, abs_mean)
        zero_const = is_constant()
        self.var = is_op("maximum")(zero_const, mean2_sub_abs_mean)

        # Normalize
        ## Activation subtract mean
        mean_prep_transpose = is_op("transpose")(self.mean)
        mean_prep_reshape = is_op("reshape")(mean_prep_transpose)
        y_act_sub_mean = is_op("subtract")(self.act, mean_prep_reshape)

        ## Activation scale
        var_prep_transpose = is_op("transpose")(self.var)
        var_prep_reshape = is_op("reshape")(var_prep_transpose)
        self.eps = is_constant()
        var_add_epsilon = is_op("add")(var_prep_reshape, self.eps)
        mul_sqrt = is_op("sqrt")(var_add_epsilon)
        mul_reciprocal = is_op("reciprocal")(mul_sqrt)
        self.gamma = is_constant()
        scale_mul_mul = is_op("multiply")(self.gamma, mul_reciprocal)
        y_scale_mul = is_op("multiply")(scale_mul_mul, y_act_sub_mean)

        ## Bias add
        self.beta = is_constant()
        y_bias_add = is_op("add")(y_scale_mul, self.beta)

        # Set pattern
        self.pattern = y_bias_add

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        gamma = node_map[self.gamma][0]
        beta = node_map[self.beta][0]

        try:
            eps = node_map[self.eps][0].data.numpy().item()
        except TVMError:  # Does not have epsilon addition
            eps = 0

        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act_shape = pre_node_map[self.act][0].checked_type.shape
        gamma_shape = pre_node_map[self.gamma][0].checked_type.shape
        beta_shape = pre_node_map[self.beta][0].checked_type.shape

        if len(gamma_shape) > 1 and sum([1 if int(x) != 1 else 0 for x in list(gamma_shape)]) == 1:
            # Count the number of dims thats not 1
            gamma_shape = (np.prod([int(x) for x in gamma_shape]),)
            gamma = tvm.relay.reshape(gamma, newshape=gamma_shape)
        else:
            assert len(gamma_shape) == 1, "TVM Layernorm only supports single dim"

        if len(beta_shape) > 1 and sum([1 if int(x) != 1 else 0 for x in list(beta_shape)]) == 1:
            # Count the number of dims thats not 1
            beta_shape = (np.prod([int(x) for x in beta_shape]),)
            beta = tvm.relay.reshape(beta, newshape=gamma_shape)
        else:
            assert len(beta_shape) == 1, "TVM Layernorm only supports single dim"

        axis = -1
        axis_val = int(pre_node_map[self.axis_val][0].data.numpy())
        for i, v in reversed(list(enumerate(act_shape))):
            if v == axis_val:
                axis = i - len(act_shape)
                break

        if axis >= 0:
            layernorm_axis = int(axis - len(act_shape))
        else:
            layernorm_axis = int(axis)

        if layernorm_axis != -1:
            return post

        if np.prod(gamma_shape) != act_shape[layernorm_axis]:
            return post

        if np.prod(pre_node_map[self.beta][0].checked_type.shape) != act_shape[layernorm_axis]:
            return post

        return tvm.relay.layernorm(act, gamma, beta, eps, layernorm_axis)


class RepositionQNormScalarMultiplier(DFPatternCallback):
    """
    Reposition Q norm which follows QK matmul for self-attention in order to
    avoid unsupported reshape and transpose ops (initially spot in Bert implemented
    in Jax).
    """

    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.q_bias = is_constant()
        self.q_bias_add = is_op("add")(wildcard(), self.q_bias)
        self.split_head_reshape = is_op("reshape")(self.q_bias_add)

        self.q_norm_last_dim_shape = is_constant()
        self.q_norm_reciprocal = is_op("reciprocal")(self.q_norm_last_dim_shape)

        self.scalar_multiplier = is_op("multiply")(self.split_head_reshape, self.q_norm_reciprocal)
        self.multiplier_transpose = is_op("transpose")(self.scalar_multiplier)
        self.multiplier_reshape = is_op("reshape")(self.multiplier_transpose)
        self.k_hidden_states = wildcard()
        self.batch_matmul = is_op("nn.batch_matmul")(self.multiplier_reshape, self.k_hidden_states)

        self.pattern = self.batch_matmul

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        q_bias_add = node_map[self.q_bias_add][0]
        q_norm_reciprocal = node_map[self.q_norm_reciprocal][0]
        k_hidden_states = node_map[self.k_hidden_states][0]

        split_head_reshape_attribute = pre_node_map[self.split_head_reshape][0].attrs.newshape
        transpose_reshape_multiplier_attribute = pre_node_map[self.multiplier_transpose][0].attrs.axes
        reshape_transposed_multiplier_attribute = pre_node_map[self.multiplier_reshape][0].attrs.newshape
        repositioned_batch_matmul_transpose_a = pre_node_map[self.batch_matmul][0].attrs.transpose_a
        repositioned_batch_matmul_transpose_b = pre_node_map[self.batch_matmul][0].attrs.transpose_b

        bias_add_norm_const_multiply = tvm.relay.multiply(q_bias_add, q_norm_reciprocal)
        reshape_multiplier = tvm.relay.reshape(bias_add_norm_const_multiply, newshape=split_head_reshape_attribute)
        transpose_multiplier = tvm.relay.transpose(reshape_multiplier, axes=transpose_reshape_multiplier_attribute)
        reshape_transposed_multiplier = tvm.relay.reshape(
            transpose_multiplier, newshape=reshape_transposed_multiplier_attribute
        )
        batch_matmul_with_reposition = tvm.relay.nn.batch_matmul(
            reshape_transposed_multiplier,
            k_hidden_states,
            transpose_a=repositioned_batch_matmul_transpose_a,
            transpose_b=repositioned_batch_matmul_transpose_b,
        )

        return batch_matmul_with_reposition


class CombineReshapes(DFPatternCallback):
    def __init__(self):
        super().__init__(require_type=True)

        self.act = wildcard()
        self.rs1 = is_op("reshape")(self.act)
        self.pattern = is_op("reshape")(self.rs1)

    def callback(self, pre, post, node_map):

        act = node_map[self.act][0]
        final_shape = pre.checked_type.shape

        return tvm.relay.reshape(act, final_shape)


class DecompEinsumWithWTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act1 = wildcard()
        self.act2 = wildcard()
        self.tup = is_tuple([self.act1, self.act2])
        self.einsum = is_op("einsum")(self.tup).has_attr({"equation": "abfd,f->fabd"})
        self.transpose = is_op("transpose")(self.einsum).has_attr({"axes": [1, 0, 3, 2]})

        self.pattern = self.transpose

    def callback(self, pre, post, node_map):
        act1 = node_map[self.act1][0]
        act2 = node_map[self.act2][0]
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act2_shape = list(pre_node_map[self.act2][0].checked_type.shape)
        new_shape = (
            [
                1,
                1,
            ]
            + act2_shape
            + [
                1,
            ]
        )
        act2_reshape = tvm.relay.reshape(act2, newshape=new_shape)
        mul = tvm.relay.multiply(act1, act2_reshape)
        transpose_yz = tvm.relay.transpose(mul, axes=[0, 2, 1, 3])
        transpose_xy = tvm.relay.transpose(transpose_yz, axes=[0, 1, 3, 2])

        return transpose_xy


class DecompWTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act1 = wildcard()
        self.act2 = wildcard()
        self.transpose_before = is_op("transpose")(self.act1).has_attr({"axes": [2, 0, 1, 3]})
        self.mul = is_op("multiply")(self.transpose_before, self.act2)
        self.final_transpose = is_op("transpose")(self.mul)
        self.pattern = self.final_transpose

    def callback(self, pre, post, node_map):
        act1 = node_map[self.act1][0]
        act2 = node_map[self.act2][0]

        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act2_shape = list(pre_node_map[self.act2][0].checked_type.shape)
        new_shape = (
            [
                1,
                1,
            ]
            + [act2_shape[0]]
            + [
                1,
            ]
        )
        act2_reshape = tvm.relay.reshape(act2, newshape=new_shape)
        mul = tvm.relay.multiply(act1, act2_reshape)
        transpose_yz = tvm.relay.transpose(mul, axes=[0, 2, 1, 3])
        final_transpose_axes = list(node_map[self.final_transpose][0].attrs.axes)
        if final_transpose_axes[2:] == [2, 3]:
            return transpose_yz
        elif final_transpose_axes[2:] == [3, 2]:
            transpose_xy = tvm.relay.transpose(transpose_yz, axes=[0, 1, 3, 2])
        else:
            return post

        return transpose_xy


class RemoveRedundantReshapeTransposeReshape(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.reshape_1 = is_op("reshape")(self.act)
        self.transpose = is_op("transpose")(
            self.reshape_1,
        )
        self.reshape_2 = is_op("reshape")(self.transpose)
        self.pattern = self.reshape_2

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        input_shape = pre_node_map[self.act][0].checked_type.shape
        reshape2 = node_map[self.reshape_2][0]
        final_shape = reshape2.attrs.newshape

        reshape_1_new_shape = list(pre_node_map[self.reshape_1][0].checked_type.shape)
        transpose_axis_list = list(pre_node_map[self.transpose][0].attrs.axes)

        # Extract transpose dimensions(i.e dim0 and dim1) from transpose op axes parameter
        # which is used for verification of data movement in tranpose op for reduction
        # into single reshape op.
        # Eg: If the input shape for tranpose op is (1,4,1,1,4,9) and tranpose axes is
        #     (0,2,1,3,4,5), then the tranpose dimemsions dim0 = 1 and dim1 = 2 are
        #     extracted by taking subtraction of transpose axes (0,2,1,3,4,5) from
        #     list (0,1,2,3,4,5) which is of length of original tranpose op input minus one
        #     and then tranpose dimensions are filtered out by taking non-zero index values
        #     from subraction array (0,-1,1,0,0,0).
        dim0, dim1 = np.nonzero(np.subtract(np.arange(len(reshape_1_new_shape)), np.array(transpose_axis_list)))[
            0
        ].tolist()
        is_reshapeable = False

        # If both the transpose dims values are 1, it will be reshapable
        # Eg: newshape = (1,4,1,1,4,9), dim0 = 0, dim1 = 2
        if reshape_1_new_shape[dim0] == 1 and reshape_1_new_shape[dim1] == 1:
            is_reshapeable = True

        # If the dim0 value is 1 and the dim0 is ahead or behind the dim1, it will be reshapeable
        # Eg: newshape = (1,4,1,1,4,9), dim0 = 0, dim1 = 1
        elif reshape_1_new_shape[dim0] == 1 and (dim0 - 1 == dim1 or dim0 + 1 == dim1):
            is_reshapeable = True

        # If the dim1 value is 1 and the dim1 value is ahead or behind the dim0, it will be reshapeable
        # Eg: newshape = (1,4,1,1,4,9), dim0 = 2, dim1 = 1
        elif reshape_1_new_shape[dim1] == 1 and (dim1 - 1 == dim0 or dim1 + 1 == dim0):
            is_reshapeable = True

        # If the dim0 or dim1 value is 1 and the intermediate tranpose dims values are 1, it will reshapeable
        # Eg: newshape = (1,4,1,1,4,9), dim0 = 2, dim1 = 4
        elif (reshape_1_new_shape[dim0] == 1 or reshape_1_new_shape[dim1] == 1) and np.all(
            np.array(reshape_1_new_shape[(min(dim0, dim1) + 1) : max(dim0, dim1)]) == 1
        ):
            is_reshapeable = True

        else:
            is_reshapeable = False

        if is_reshapeable and (
            list(final_shape) == list(input_shape)[-2:]
            or list(final_shape) == [1] + list(input_shape)
            or list(final_shape) == list(input_shape)
        ):
            return tvm.relay.reshape(node_map[self.act][0], newshape=final_shape)

        return post


class AttemptRemoveStackWDim(DFPatternCallback):
    def __init__(self):
        super().__init__(require_type=True)
        self.acts = wildcard()
        self.stack = is_op("stack")(self.acts)
        self.t1 = is_op("transpose")(
            self.stack,
        )
        self.reshape = is_op("reshape")(self.t1)
        self.pattern = is_op("transpose")(self.reshape)

    def callback(self, pre: Expr, post: Expr, node_map: tvm.ir.container.Map) -> Expr:
        acts = node_map[self.acts][0]
        stack = node_map[self.stack][0]
        t1 = node_map[self.t1][0]
        reshape = node_map[self.reshape][0]
        t2 = node_map[self.pattern][0]

        if (
            all(len(stack.args[0].fields[i].checked_type.shape) == 3 for i in range(len(stack.args[0].fields)))
            and len(stack.checked_type.shape) > 3
            and stack.checked_type.shape[-4] > 1
            and t2.checked_type.shape[-2] == stack.checked_type.shape[-4]
        ):

            reshaped_inps = []
            for arg in stack.args[0].fields:
                newdim = 1
                for d in arg.checked_type.shape:
                    newdim *= d

                if arg.checked_type.shape[-2] == 1:
                    arg = tvm.relay.transpose(arg, axes=[1, 0, 2])

                flattened = tvm.relay.reshape(arg, newshape=[1, 1, newdim])
                reshaped_inps.append(flattened)

            return tvm.relay.concatenate(reshaped_inps, -2)

        return super().callback(pre, post, node_map)


class SimplifyReshape(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.reshape_0 = is_op("reshape")(self.act)
        self.transpose_0 = is_op("transpose")(self.reshape_0).has_attr({"axes": [0, 3, 2, 1]})
        self.transpose_1 = is_op("transpose")(self.transpose_0).has_attr({"axes": [0, 1, 3, 2]})
        self.reshape_1 = is_op("reshape")(self.transpose_1)
        self.pattern = self.reshape_1

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        input_shape = list(pre_node_map[self.act][0].checked_type.shape)
        reshape_1 = node_map[self.reshape_1][0]
        final_shape = list(reshape_1.attrs.newshape)

        if input_shape == final_shape and len(input_shape) >= 3:
            final_transpose_axes = (
                np.arange(int(len(input_shape) - 2)).tolist()
                + np.flip(np.arange(int(len(input_shape) - 2), len(input_shape))).tolist()
            )
            final_transpose = tvm.relay.transpose(node_map[self.act][0], axes=final_transpose_axes)
            return final_transpose

        return post


class ReplicateForgeReshapeTranspose(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.transpose_1 = is_op("transpose")(self.act)
        self.transpose_2 = is_op("transpose")(
            self.transpose_1,
        )
        self.reshape = is_op("reshape")(self.transpose_2)
        self.pattern = self.reshape

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act_shape = pre_node_map[self.act][0].checked_type.shape
        t1 = node_map[self.transpose_1][0]
        t1_axes = [int(dim) for dim in t1.attrs.axes]
        t2 = node_map[self.transpose_2][0]
        t2_axes = [int(dim) for dim in t2.attrs.axes]

        reshape = node_map[self.reshape][0]
        newshape = [int(dim) for dim in reshape.attrs.newshape]
        input_shape = [int(dim) for dim in t2.checked_type.shape]
        if len(input_shape) < 4:
            return post
        eos = [input_shape[0], input_shape[1] * input_shape[2], input_shape[3]]
        if t1_axes == [0, 2, 1, 3] and t2_axes == [0, 1, 3, 2] and newshape == eos:
            newshape = [1, int(act_shape[0]), eos[2], eos[1]]
            r1 = tvm.relay.reshape(node_map[self.act][0], newshape)
            t1 = tvm.relay.transpose(r1, axes=[0, 1, 3, 2])
            squeeze = tvm.relay.reshape(t1, newshape=[int(act_shape[0]), eos[1], eos[2]])
            return squeeze
        else:
            return post


class CommuteIndexPastReshape(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.index = is_op("strided_slice")(self.act)
        self.bias = wildcard()
        self.reshape_bias = is_op("reshape")(self.bias)
        self.add = is_op("add")(self.index, self.reshape_bias)
        self.reshape0 = is_op("reshape")(self.add)
        self.transpose0 = is_op("transpose")(self.reshape0).has_attr({"axes": [0, 1, 3, 2]})
        self.reshape1 = is_op("reshape")(self.transpose0)
        self.pattern = self.reshape1

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        index_attrs = node_map[self.index][0].attrs
        if len(index_attrs.begin) != 1 or len(index_attrs.end) != 1:
            return post
        bias = node_map[self.bias][0]

        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act_shape = list(pre_node_map[self.act][0].checked_type.shape)
        if len(act_shape) != 4:
            return post

        bias_shape = list(pre_node_map[self.bias][0].checked_type.shape)
        if len(bias_shape) != 1:
            return post

        reshape0_target = list(node_map[self.reshape0][0].attrs.newshape)
        if reshape0_target != [1, 1, index_attrs.end[0] - index_attrs.begin[0], act_shape[-2] * act_shape[-1]]:
            return post

        reshape1_target = list(node_map[self.reshape1][0].attrs.newshape)
        if reshape1_target != [1, act_shape[-2] * act_shape[-1], index_attrs.end[0] - index_attrs.begin[0]]:
            return post

        new_reshape0 = tvm.relay.reshape(act, newshape=[1, 1, act_shape[-3], act_shape[-2] * act_shape[-1]])
        new_transpose = tvm.relay.transpose(new_reshape0, axes=[0, 1, 3, 2])
        new_reshape1 = tvm.relay.reshape(new_transpose, newshape=[1, act_shape[-2] * act_shape[-1], act_shape[-3]])
        new_index = tvm.relay.strided_slice(
            new_reshape1, begin=index_attrs.begin, end=index_attrs.end, strides=index_attrs.strides, axes=[2]
        )

        new_bias_reshape = tvm.relay.reshape(bias, newshape=[1, 1, bias_shape[0]])
        new_add = tvm.relay.add(new_index, new_bias_reshape)
        return new_add


class DecomposeScatterND(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        # ScatterND op with all inputs
        self.data = wildcard()
        self.indices = wildcard()
        self.updates = wildcard()
        self.mode = wildcard()
        self.scatter_nd = is_op("scatter_nd")(self.data, self.indices, self.updates)

        self.pattern = self.scatter_nd

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        # Arguments & Op
        data = node_map[self.data][0]
        indices = node_map[self.indices][0]
        updates = node_map[self.updates][0]

        # Handle only update mode (replacement instead of accumulation)
        if not pre_node_map[self.scatter_nd][0].attrs.mode == "update":
            return post

        # Handle only 2D tensors
        if (
            len(pre_node_map[self.data][0].checked_type.shape) != 2
            or len(pre_node_map[self.indices][0].checked_type.shape) != 2
            or len(pre_node_map[self.updates][0].checked_type.shape) != 2
        ):
            return post

        # Debug values
        # import torch
        # from tvm.relay.frontend.common import analysis
        # from tvm.relay.frontend.common import infer_shape
        # from tvm.relay.frontend.common import infer_value

        # analysis.free_vars(data)
        # data_shape = infer_shape(data)
        # a = torch.arange(0, np.prod(data_shape), dtype=torch.float32).view(data_shape) / 18
        # a = tvm.nd.array(a)
        # analysis.free_vars(indices)
        # indices_shape = infer_shape(indices)
        # b = torch.rand(indices_shape, dtype=torch.float32).view(indices_shape)
        # b = tvm.nd.array(b)

        # Scatter ND using where op
        out = tvm.relay.where(indices, updates, data)

        # infer_shape(out)
        # infer_value(out, {'input_act': a, 'threshold_1': b})

        return out


class ConvertIsNaN(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        # IsNaN op with all inputs
        self.data = wildcard()
        self.isnan = is_op("isnan")(self.data)

        self.pattern = self.isnan

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        data = pre_node_map[self.data][0]

        # NaN (Not a Number) is the only value in floating-point arithmetic that is not equal to itself.
        # So, comparing data with itself will return True if data is NaN, and False otherwise.
        # This condition is used to identify NaN values in the data tensor.

        cond = tvm.relay.not_equal(data, data)

        return cond


class RemoveRedundantBinaryStacks(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.act = wildcard()
        self.tuple = is_tuple([self.act])
        self.stack = is_op("stack")(self.tuple)
        self.reshape = is_op("reshape")(self.stack)

        self.pattern = self.reshape

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        act = node_map[self.act][0]
        stack = pre_node_map[self.stack][0]
        reshape = pre_node_map[self.reshape][0]

        if len(stack.args) != 1 and len(reshape.args) != 1:
            return post

        stack_field_input_shape = [int(i) for i in stack.args[0].checked_type.fields[0].shape]
        reshape_shape = [int(i) for i in reshape.attrs.newshape]

        if stack_field_input_shape != reshape_shape:
            return post

        # Creates duplicates
        # from tvm.relay.frontend.common import infer_type
        # return infer_type(act.fields[0])

        # Also, creates duplicates
        # mod = tvm.ir.IRModule.from_expr(act.fields[0])
        # mod = transform.InferType()(mod)
        # return mod['main'].body

        return act


class RemoveStopFusionAnnotationNodes(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.act = wildcard()
        self.stop_fusion_node = is_op("annotation.stop_fusion")(self.act)

        self.pattern = self.stop_fusion_node

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]

        return act


class Enforce1DOutputForArgwhereOp(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.act = wildcard()
        self.argwhere = is_op("argwhere")(self.act)
        self.transpose = is_op("transpose")(self.argwhere)

        self.pattern = self.transpose

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        act = node_map[self.act][0]
        act_shape = pre_node_map[self.act][0].checked_type.shape
        transpose_shape = pre_node_map[self.transpose][0].checked_type.shape

        if (
            len(act_shape) == 2
            and int(act_shape[0]) == 1
            and len(transpose_shape) == 2
            and int(transpose_shape[0]) == 2
        ):
            out = tvm.relay.squeeze(act, axis=[0])
            out = tvm.relay.argwhere(out)
            out = tvm.relay.transpose(out, axes=[1, 0])
            out = tvm.relay.squeeze(out, axis=[0])
            return out

        return post


class BroadcastScatterValuesToMatchIndices(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.act = wildcard()
        self.indices = wildcard()
        self.updates = wildcard()
        self.scatter = is_op("scatter")(self.act, self.indices, self.updates)

        self.pattern = self.scatter

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        act = node_map[self.act][0]
        indices = node_map[self.indices][0]
        updates = node_map[self.updates][0]
        updates_shape = pre_node_map[self.updates][0].checked_type.shape
        scatter = node_map[self.scatter][0]

        if len(updates_shape) == 1 and isinstance(updates_shape[0], tvm.tir.expr.IntImm) and int(updates_shape[0]) == 1:
            # Match dtype for bcast
            updates = tvm.relay.cast(updates, indices.checked_type.dtype)
            updates = tvm.relay.broadcast_to_like(updates, indices)
            # Match dtype for scatter
            updates = tvm.relay.cast(updates, act.checked_type.dtype)
            out = tvm.relay.scatter(act, indices, updates, int(scatter.attrs.axis))
            return out

        return post


class InverseMaskGen(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.zero = wildcard()
        self.mask = wildcard()
        self.mask_gen = is_op("equal")(self.mask, self.zero)

        self.pattern = self.mask_gen

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        if (
            not isinstance(pre_node_map[self.zero][0], tvm.relay.Constant)
            or pre_node_map[self.zero][0].data.numpy() != 0
        ):
            return post

        casted = tvm.relay.cast(pre_node_map[self.mask][0], "bool")
        casted = tvm.relay.cast(casted, "int32")
        subtract = tvm.relay.subtract(tvm.relay.const(1, "int32"), casted)
        booled = tvm.relay.cast(subtract, "bool")
        return booled


class DecomposeVariance(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.variance = is_op("variance")(wildcard(), wildcard())

        self.pattern = self.variance

    def callback(self, pre, post, node_map):
        mean = post.args[1]
        sub = tvm.relay.subtract(post.args[0], mean)
        mul = tvm.relay.multiply(sub, sub)
        var = tvm.relay.mean(mul, axis=post.attrs.axis, keepdims=post.attrs.keepdims, exclude=post.attrs.exclude)

        return var


class DecomposePRelu(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.prelu = is_op("nn.prelu")(wildcard(), wildcard())

        self.pattern = self.prelu

    def callback(self, pre, post, node_map):
        relu_ = tvm.relay.nn.relu(post.args[0])
        neg_one = tvm.relay.Constant(
            tvm.nd.array(
                np.array(
                    [
                        -1,
                    ],
                    dtype="float32",
                )
            )
        )
        neg_ = tvm.relay.multiply(post.args[0], neg_one)
        neg_relu_ = tvm.relay.nn.relu(neg_)
        mul_ = tvm.relay.multiply(post.args[1], neg_relu_)
        mul_2 = tvm.relay.multiply(mul_, neg_one)
        out = tvm.relay.add(relu_, mul_2)
        return out


class SimplifyTransposeReshape(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.act = wildcard()
        self.transpose = is_op("transpose")(self.act)
        self.reshape = is_op("reshape")(self.transpose)

        self.pattern = self.reshape

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        act_shape = list(pre_node_map[self.act][0].checked_type.shape)
        target_shape = list(node_map[self.reshape][0].attrs.newshape)
        transpose_axes = node_map[self.transpose][0].attrs.axes

        dims = np.arange(len(act_shape))
        # At this point, we should only have single axis transpose
        count = 0
        for (index, val) in enumerate(transpose_axes):
            if index != val:
                count += 1

        assert count == 2, "Multi-axis transpose should be decomposed into single-axis transpose at this point"
        is_transpose_yz = (
            len(transpose_axes) >= 3
            and len(dims) >= 3
            and (transpose_axes[-2] == dims[-3] and transpose_axes[-3] == dims[-2])
        )

        if (
            int(act_shape[0]) == 1
            and len(act_shape) - 1 == len(target_shape)
            and int(target_shape[0]) == -1
            and is_transpose_yz
            and act_shape[-2] == 1
        ):
            # this is equivalent to a squeeze
            return tvm.relay.squeeze(node_map[self.act][0], axis=[2])

        else:
            return post


class DecomposeNonZeroPadtoConcat(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)
        self.act = wildcard()
        self.pad = is_op("nn.pad")(self.act, wildcard())

        self.pattern = self.pad

    def callback(self, pre, post, node_map):
        act = post.args[0]
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        pad_width = post.attrs.pad_width
        pad_value = post.args[1].data.numpy()
        if pad_value == 0:
            return post

        pad_shape = list(pre_node_map[self.act][0].checked_type.shape)
        pad_shape = [int(x) for x in pad_shape]
        for i, item in enumerate(pad_width):
            before = int(item[0])
            after = int(item[1])
            if before == 0 and after == 0:
                continue

            if before != 0:
                current_pad_shape = pad_shape.copy()
                current_pad_shape[i] = before
                const = tvm.relay.const(
                    np.ones(current_pad_shape) * pad_value, dtype=pre_node_map[self.act][0].checked_type.dtype
                )
                act = tvm.relay.concatenate([const, act], axis=i)
                current_pad_shape[i] += pad_shape[i]
                pad_shape = current_pad_shape

            if after != 0:
                current_pad_shape = pad_shape.copy()
                current_pad_shape[i] = after
                const = tvm.relay.const(
                    np.ones(current_pad_shape) * pad_value, dtype=pre_node_map[self.act][0].checked_type.dtype
                )
                act = tvm.relay.concatenate([act, const], axis=i)
                current_pad_shape[i] += pad_shape[i]
                pad_shape = current_pad_shape
        return act


class ReplaceYolov5Perf(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.mul1_1_const = wildcard()
        self.mul1_3_const = wildcard()
        self.mul2_1_const = wildcard()
        self.add2_2_const = wildcard()
        self.mul2_3_const = wildcard()

        self.act = wildcard()
        self.reshape = is_op("reshape")(self.act)
        transpose1 = is_op("transpose")(self.reshape)
        transpose2 = is_op("transpose")(transpose1)
        sigmoid = is_op("sigmoid")(transpose2)

        self.slice1 = is_op("strided_slice")(sigmoid)
        mul1_1 = is_op("multiply")(self.slice1, self.mul1_1_const)
        mul1_2 = is_op("multiply")(mul1_1, mul1_1)
        mul1_3 = is_op("multiply")(mul1_2, self.mul1_3_const)

        self.slice2 = is_op("strided_slice")(sigmoid)
        mul2_1 = is_op("multiply")(self.slice2, self.mul2_1_const)
        add2_2 = is_op("add")(mul2_1, self.add2_2_const)
        mul2_3 = is_op("multiply")(add2_2, self.mul2_3_const)

        self.slice3 = is_op("strided_slice")(sigmoid)

        tup = is_tuple([mul2_3, mul1_3, self.slice3])
        self.pattern = is_op("concatenate")(tup)

    def callback(self, pre, post, node_map):
        reshaped_shape = list(node_map[self.reshape][0].checked_type.shape)
        reshaped_shape = [int(x) for x in reshaped_shape]
        shape1 = [1, 1, reshaped_shape[1] * reshaped_shape[2], reshaped_shape[3] * reshaped_shape[4]]
        shape2 = [1, reshaped_shape[1], reshaped_shape[2], reshaped_shape[3] * reshaped_shape[4]]

        def separate_slice_attr(slice_attrs):
            return int(slice_attrs.begin[0]), int(slice_attrs.end[0]), int(slice_attrs.axes[0])

        slice1_begin, slice1_end, slice1_axis = separate_slice_attr(node_map[self.slice1][0].attrs)
        slice2_begin, slice2_end, slice2_axis = separate_slice_attr(node_map[self.slice2][0].attrs)
        slice3_begin, slice3_end, slice3_axis = separate_slice_attr(node_map[self.slice3][0].attrs)
        if slice1_axis != 4 or slice2_axis != 4 or slice3_axis != 4 or len(reshaped_shape) != 5:
            return post

        act = node_map[self.act][0]
        mul1_1_const = node_map[self.mul1_1_const][0]
        mul1_3_const = node_map[self.mul1_3_const][0]
        mul2_1_const = node_map[self.mul2_1_const][0]
        add2_2_const = node_map[self.add2_2_const][0]
        mul2_3_const = node_map[self.mul2_3_const][0]

        # pad add2_2
        slice2_length = slice2_end - slice2_begin
        pad2_shape = reshaped_shape.copy()
        pad2_shape[2] -= slice2_length
        pad2 = tvm.relay.Constant(tvm.nd.array(np.zeros(pad2_shape, dtype=act.checked_type.dtype)))
        add2_2_const = tvm.relay.transpose(add2_2_const, axes=[0, 1, 4, 2, 3])
        add2_2_const = tvm.relay.concatenate([add2_2_const, pad2], axis=2)
        add2_2_const = tvm.relay.reshape(
            add2_2_const, newshape=[1, reshaped_shape[1] * reshaped_shape[2], reshaped_shape[3], reshaped_shape[4]]
        )

        # pad mul1_3
        slice1_length = slice1_end - slice1_begin
        pad1_1_shape = reshaped_shape.copy()
        pad1_1_shape[2] = slice2_length
        pad1_1 = tvm.relay.Constant(tvm.nd.array(np.zeros(pad1_1_shape, dtype=act.checked_type.dtype)))
        pad1_3_shape = reshaped_shape.copy()
        pad1_3_shape[2] -= slice2_length + slice1_length
        pad1_3 = tvm.relay.Constant(tvm.nd.array(np.zeros(pad1_3_shape, dtype=act.checked_type.dtype)))
        mul1_3_const = tvm.relay.transpose(mul1_3_const, axes=[0, 1, 4, 2, 3])
        mul1_3_const = tvm.relay.concatenate([pad1_1, mul1_3_const, pad1_3], axis=2)
        mul1_3_const = tvm.relay.reshape(
            mul1_3_const, newshape=[1, reshaped_shape[1] * reshaped_shape[2], reshaped_shape[3], reshaped_shape[4]]
        )

        # generate masks
        def generate_slice_mask(begin, end):
            slice_mask_index = np.zeros(reshaped_shape.copy(), dtype=act.checked_type.dtype)
            slice_mask_index[:, :, begin:end, :, :] = 1.0
            slice_mask = tvm.relay.Constant(tvm.nd.array(slice_mask_index))
            slice_mask = tvm.relay.reshape(
                slice_mask, newshape=[1, reshaped_shape[1] * reshaped_shape[2], reshaped_shape[3], reshaped_shape[4]]
            )
            return slice_mask

        slice1_mask = generate_slice_mask(slice1_begin, slice1_end)
        slice2_mask = generate_slice_mask(slice2_begin, slice2_end)
        slice3_mask = generate_slice_mask(slice3_begin, slice3_end)

        # re-connect
        sigmoid = tvm.relay.sigmoid(act)

        mul1_1 = tvm.relay.multiply(sigmoid, mul1_1_const)
        mul1_2 = tvm.relay.multiply(mul1_1, mul1_1)
        mul1_3 = tvm.relay.multiply(mul1_2, mul1_3_const)
        mul1_3_masked = tvm.relay.multiply(mul1_3, slice1_mask)

        mul2_1 = tvm.relay.multiply(sigmoid, mul2_1_const)
        add2_2 = tvm.relay.add(mul2_1, add2_2_const)
        mul2_3 = tvm.relay.multiply(add2_2, mul2_3_const)
        mul2_3_masked = tvm.relay.multiply(mul2_3, slice2_mask)

        slice3_masked = tvm.relay.multiply(sigmoid, slice3_mask)

        partial_sum = tvm.relay.add(mul1_3_masked, mul2_3_masked)
        _sum = tvm.relay.add(partial_sum, slice3_masked)

        reshape = tvm.relay.reshape(_sum, newshape=shape1)
        reshape = tvm.relay.reshape(reshape, newshape=shape2)
        transpose = tvm.relay.transpose(reshape, axes=[0, 1, 3, 2])
        return transpose


class TransformDenseIntoBatchMM(DFPatternCallback):
    """
    This pass will transform dense into batch_matmul and therefore
    remove redundant squeeze and unsqueeze ops.
    """

    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.act = wildcard()
        self.weight = wildcard()

        self.reshape1 = is_op("reshape")(self.act)
        self.transpose = is_op("transpose")(self.weight)

        self.lm_head = is_op("nn.dense")(self.reshape1, self.transpose)
        self.reshape2 = is_op("reshape")(self.lm_head)

        self.pattern = self.reshape2

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)

        activations = node_map[self.act][0]
        weight = node_map[self.weight][0]
        squeeze = node_map[self.reshape1][0]
        weight_transpose = node_map[self.transpose][0]
        lm_head = node_map[self.lm_head][0]
        unsqueeze = node_map[self.reshape2][0]

        if not (len(weight.args) == 1 and isinstance(weight.args[0], tvm.relay.Var)):
            return post
        if len(list(activations.checked_type.shape)) != 3:
            return post
        if len(list(weight_transpose.checked_type.shape)) != 2:
            return post
        if len(list(lm_head.checked_type.shape)) != 2:
            return post
        if len(list(unsqueeze.checked_type.shape)) != 3:
            return post

        new_weight_shape = [1] + list(weight.checked_type.shape)
        weight_reshape = tvm.relay.reshape(weight, newshape=new_weight_shape)
        lm_head = tvm.relay.nn.batch_matmul(activations, weight_reshape, transpose_a=False, transpose_b=False)

        return lm_head


class GQABroadcastReshape(DFPatternCallback):
    """
    Callback for Grouped Query Attention Pattern. When parsing a standard GQA,
    A subpattern appears that is in the form:

    (bs, n_kv_heads, seq_len, head_dim) ->[reshape0]-> (bs, n_kv_heads, 1, seq_len, head_dim) ->[bc0]->
    (bs, n_kv_heads, 1, seq_len, head_dim) ->[bc1]-> (bs, n_kv_heads, n_kv_blocks, seq_len, head_dim) ->[reshape1]->
    (n_query_heads, bs*seq_len, head_dim) ->[transpose]-> (n_query_heads, head_dim, bs*seq_len)

    Where n_query_heads == n_kv_heads * n_kv_blocks. The problem with this subpattern is this broadcast that is
    performed (bc1) which generates a 5D tensor with 4 dimensions that are not equal to 1.
    That bc output is then input to reshape1 and forge compiler has no way to decompose a reshape that is performed
    on such input tensor. That is why we change this pattern so that this doesn't occur.

    Modification:

    (bs, n_kv_heads, seq_len, head_dim) ->[transpose] -> (bs, seq_len, n_kv_heads, head_dim)
    ->[reshape]-> (bs, n_kv_heads*seq_len, 1, head_dim) ->[bc]-> (bs, seq_len*n_kv_heads, brcst_val, head_dim)
    ->[reshape]-> (bs*seqlen, n_kv_heads*brcst_val, head_dim) ->[transpose]-> (n_kv_heads*brcst_val, bs*seqlen, head_dim)
    ->[transpose]-> (n_kv_heads*brcst_val, head_dim, bs*seqlen)

    """

    def __init__(self, require_type=False, rewrite_once=False):
        super().__init__(require_type, rewrite_once)
        self.act = wildcard()
        self.reshape0 = is_op("reshape")(self.act)
        self.bc0 = is_op("broadcast_to")(self.reshape0)
        self.bc1 = is_op("broadcast_to")(self.bc0)
        self.reshape1 = is_op("reshape")(self.bc1)
        self.pattern = is_op("transpose")(self.reshape1)

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]  # [bs, n_kv_heads, seq_len, head_dim]
        orig_shape = act.checked_type.shape

        # idea is to catch only reshapes [bs, n_kv_heads, seq_len, head_dim] -> [bs, n_kv_heads, 1, seq_len, head_dim]
        if len(orig_shape) != 4:
            return post

        if len(node_map[self.reshape0][0].attrs.newshape) != 5:
            return post

        transpose0 = tvm.relay.transpose(act, axes=[0, 2, 1, 3])  # [bs, seq_len, n_kv_heads, head_dim]
        prev_shape = (orig_shape[-4], orig_shape[-2], orig_shape[-3], orig_shape[-1])  # a.k.a. transpose0 shape

        new_shape = (prev_shape[-4], int(prev_shape[-3] * prev_shape[-2]), 1, prev_shape[-1])

        reshape0 = tvm.relay.reshape(transpose0, newshape=new_shape)  # (bs, seq_len*n_kv_heads, 1, head_dim)

        if new_shape[-3] != prev_shape[-3] * prev_shape[-2]:
            return post

        bc1 = node_map[self.bc1][0]
        pre_broadcast_shape = list(bc1.type_args[0].shape)
        post_broadcast_shape = list(bc1.attrs.shape)

        # get the value of dimension that is different after applying broadcast
        broadcasted_value = [el for idx, el in enumerate(post_broadcast_shape) if el != pre_broadcast_shape[idx]][0]

        new_broadcast_shape = [el for el in new_shape]
        new_broadcast_shape[-2] = broadcasted_value

        bc = tvm.relay.broadcast_to(
            reshape0, new_broadcast_shape
        )  # (bs, seq_len*n_kv_heads, 1, head_dim) -> (bs, seq_len*n_kv_heads, brcst_val, head_dim)

        new_shape = [
            prev_shape[-4] * prev_shape[-3],
            new_broadcast_shape[-3] * new_broadcast_shape[-2] // prev_shape[-3],
            prev_shape[-1],
        ]  # (bs*seqlen, n_kv_heads*brcst_val, head_dim)
        reshape1 = tvm.relay.reshape(bc, new_shape)

        transpose1 = tvm.relay.transpose(reshape1, axes=[1, 0, 2])  # (n_kv_heads*brcst_val, bs*seqlen, head_dim)
        transpose2 = tvm.relay.transpose(transpose1, axes=[0, 2, 1])  # (n_kv_heads*brcst_val, head_dim, bs*seqlen)
        return transpose2


class RemoveDenseInputSqueeze(DFPatternCallback):
    """
    TVM adds squeeze ops around nn.dense activations to ensure that both the
    lhs and rhs are the same rank. This is unnecessarry since TTNN can handle
    differently ranked input tensors for matmul, just like pytorch and other
    frameworks can. This pattern callback removes those squeezes. The reason
    that reshapes are being pattern matched here is because the squeezes get
    converted to reshapes by this point.
    """

    def __init__(self, require_type=False, rewrite_once=False):
        super().__init__(require_type, rewrite_once)
        self.act = wildcard()
        self.reshape1 = is_op("reshape")(self.act)
        self.dense = is_op("nn.dense")(self.reshape1, wildcard())
        self.pattern = is_op("reshape")(self.dense)

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        reshape1 = node_map[self.reshape1][0]
        reshape2 = node_map[self.pattern][0]
        dense = node_map[self.dense][0]

        # check if reshape is squeeze
        act_shape = act.checked_type.shape
        reshape1_shape = reshape1.checked_type.shape
        reshape2_shape = reshape2.checked_type.shape
        dense_shape = dense.checked_type.shape

        # Check that reshape did not affect matmul dims
        if reshape1_shape[-2:] != act_shape[-2:]:
            return post

        # check that second reshape is the same rank as activation
        if len(reshape2_shape) != len(act_shape):
            return post

        # check that the first reshape is a squeeze
        act_dims = set(act_shape[:-2])
        reshape1_dims = set(reshape1_shape[:-2])
        if (act_dims - reshape1_dims) != set([1]):
            return post

        # check that the second reshape is an unsqueeze
        dense_dims = set(dense_shape[:-2])
        reshape2_dims = set(reshape2_shape[:-2])
        if (reshape2_dims - dense_dims) != set([1]):
            return post

        return tvm.relay.nn.dense(act, dense.args[1])


class ExpandMultipleDims(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.act = wildcard()
        self.pattern = is_op("expand_dims")(self.act)

    def callback(self, pre, post, node_map):
        act = node_map[self.act][0]
        num_newaxis = int(post.attrs.num_newaxis)
        axis = int(post.attrs.axis)
        if num_newaxis > 1:
            assert axis >= 0, f"Error: Axis is negative. axis: {axis}"
            while num_newaxis:
                act = tvm.relay.expand_dims(act, axis=axis)
                num_newaxis -= 1
            return act
        return post


class DecomposeGridSample(DFPatternCallback):
    """Decomposes ``image.grid_sample(data, grid)`` into primitive relay ops

    Overview
    --------
    ``grid_sample`` samples a spatial feature map *data* at floating-point
    coordinates given by *grid*.  Each grid value is a normalised (x, y) pair
    in [-1, 1]; the op maps those coordinates to pixel space, looks up one or
    four neighbouring pixels, and blends them (bilinear) or snaps to the
    nearest one.

    Signatures
    ----------
    data  : (N, C, H, W)   float32  — feature map to sample from
    grid  : (N, Hg, Wg, 2) float32  — grid[..., 0] = x (col), grid[..., 1] = y (row)
    output: (N, C, Hg, Wg) float32  — sampled values

    Attributes
    ----------
    mode          : "bilinear" | "nearest"
    padding_mode  : "zeros"  (only mode currently supported)
    align_corners : True  → grid ±1 maps to the *centre* of the corner pixels
                            unnorm_x = (x + 1) / 2 * (W - 1)
                  : False → grid ±1 maps to the *edge* of the corner pixels
                            unnorm_x = (x + 1) * W / 2 - 0.5
                    The False convention is resolution-agnostic and common in
                    recent BEV / camera-projection networks.

    Decomposition steps
    -------------------
    The callback rewrites a single ``image.grid_sample`` node into a sequence
    of relay primitives.  The steps below describe one batch element (N is
    handled by splitting on axis 0 and concatenating results at the end).

    Step 1 — Batch split
        ``relay.split(data, N, axis=0)`` and ``relay.split(grid, N, axis=0)``
        produce per-sample slices (1, C, H, W) and (1, Hg, Wg, 2).
        Static shape specialisation requires N to be known at compile time.

    Step 2 — Coordinate unnormalisation
        ``strided_slice`` extracts the x and y coordinate planes from the
        (1, Hg, Wg, 2) grid tensor; ``squeeze`` removes the trailing dim.
        The normalised values are then mapped to pixel space:
          align_corners=True:  unnorm = (coord + 1) / 2 * (dim - 1)
          align_corners=False: unnorm = (coord + 1) * dim / 2 - 0.5

    Step 3 — Zero-padding
        ``relay.nn.pad`` adds a 1-pixel border of zeros around the spatial
        dims, producing a (1, C, H+2, W+2) padded image.  Out-of-bounds
        grid coordinates are later clamped into this border, so they
        automatically sample zero — implementing padding_mode="zeros".

    Step 4 — Integer pixel indices  [hardware workaround A]
        The padded pixel index is:
            x0_p = cast(floor(unnorm_x + 1), int32)
        The "+1" pre-shift combines two effects:
          (a) It is the pad offset, so no separate addition is needed later.
          (b) It keeps the floor argument ≥ 0 for all valid grid coords.
              - align_corners=True:  unnorm ∈ [0, W-1]     → shifted ∈ [1, W]
              - align_corners=False: unnorm ∈ [-0.5, W-0.5] → shifted ∈ [0.5, W+0.5]
        This matters because ``DecomposeFloor`` rewrites ``floor`` as a
        Less/Where chain; the TT hardware evaluates ``Less(x, 0)`` as always
        False, so negative non-integers are truncated toward zero instead of
        being floored.  The pre-shift ensures the negative branch never fires.
        floor(x + 1) == floor(x) + 1 for all x, so the resulting index is
        mathematically identical to the standard padded index.
        A final ``relay.clip`` clamps fully out-of-range coords to the padded
        boundary (indices [0, padded_dim - 1]).

    Step 5 — Bilinear weights  [hardware workaround B]
        Standard formula: dx0 = unnorm_x - floor(unnorm_x)  (fractional part)
        On TT hardware, Cast(int32_from_floor_chain, float32) silently
        returns zero, so dx0 computed that way would corrupt the interpolation.
        Fix — exploit the identity dx0 + dx1 = 1:
            dx1 = cast(x1_p, float32) - x_shifted   ← x1_p is x0_p+1, never
            dx0 = 1.0 - dx1                            derived from a floor cast
        The four bilinear corner weights (squeezed to (Hg, Wg) — see below):
            wa = dx1 * dy1   corner (x1, y1)
            wb = dx1 * dy0   corner (x1, y0)
            wc = dx0 * dy1   corner (x0, y1)
            wd = dx0 * dy0   corner (x0, y0)
        Weights are squeezed from (1, Hg, Wg) to (Hg, Wg) because a
        (C, 1, H, W) * (1, 1, H, W) broadcast multiply silently produces
        zeros on TT hardware.  [hardware workaround C]

    Step 6 — Flattened 2-D → 1-D indexing  [hardware workaround D]
        The padded image is reshaped to (1, C, H_p * W_p).  Pixel addresses
        are computed as:
            flat = cast_f32(x) + cast_f32(y) * padded_w
        The intermediate arithmetic is kept in float32 because int32 * int32
        multiply silently returns zeros on TT hardware.  The final result is
        cast back to int32 for the gather step.

    Step 7 — Per-channel gather  [hardware workaround E]
        The flat image is split along the channel axis (C slices of shape
        (1, H_p * W_p)).  Each channel slice is transposed to (H_p*W_p, 1)
        so that ``relay.adv_index`` treats it as an embedding table of
        H_p*W_p rows with 1 element each — the standard Embedding layout.
        For bilinear mode, the four corner index vectors are concatenated into
        a single (4*Hg*Wg,) tensor and passed to ONE ``adv_index`` call per
        channel.  The result (4*Hg*Wg, 1) is split into four equal parts,
        transposed, and reshaped to (Hg, Wg) before applying the weights.
        A single adv_index is required because multiple adv_index ops that
        read the same embedding table return zeros on TT hardware.

    Step 8 — Output reconstruction
        Per-channel results (Hg, Wg) are reshaped to (1, Hg, Wg), concatenated
        along axis 0 to form (C, Hg, Wg), then reshaped to (1, C, Hg, Wg).
        Batch results are concatenated along axis 0 to give (N, C, Hg, Wg).
    """

    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.data = wildcard()
        self.grid = wildcard()
        self.grid_sample = is_op("image.grid_sample")(self.data, self.grid)
        self.pattern = self.grid_sample

    def callback(self, pre, post, node_map):
        from tvm.relay.frontend.common import infer_shape

        data_node = node_map[self.data][0]
        grid_node = node_map[self.grid][0].args[0]
        mode = post.attrs["method"]
        align_corners = post.attrs["align_corners"]

        # Step 1: split along batch axis so each element has a known static shape
        batch_size = infer_shape(data_node)[0]
        split_im = tvm.relay.split(data_node, indices_or_sections=batch_size, axis=0)
        split_grid = tvm.relay.split(grid_node, indices_or_sections=batch_size, axis=0)
        results = []

        for batch_idx in range(batch_size):
            im_b = split_im[batch_idx]
            grid_b = split_grid[batch_idx]
            n, c, h, w = infer_shape(im_b)
            gn, gh, gw, _ = infer_shape(grid_b)

            # Step 2: extract x/y planes and unnormalise to pixel coordinates
            x = tvm.relay.squeeze(
                tvm.relay.strided_slice(grid_b, [0, 0, 0, 0], [gn, gh, gw, 1], [1, 1, 1, 1]),
                axis=[3],
            )
            y = tvm.relay.squeeze(
                tvm.relay.strided_slice(grid_b, [0, 0, 0, 1], [gn, gh, gw, 2], [1, 1, 1, 1]),
                axis=[3],
            )
            if align_corners:
                # ±1 → pixel centres: unnorm = (coord + 1) / 2 * (dim - 1)
                x = ((x + tvm.relay.const(1.0)) / tvm.relay.const(2.0)) * tvm.relay.const(float(w - 1))
                y = ((y + tvm.relay.const(1.0)) / tvm.relay.const(2.0)) * tvm.relay.const(float(h - 1))
            else:
                # ±1 → pixel edges: unnorm = (coord + 1) * dim / 2 - 0.5
                x = ((x + tvm.relay.const(1.0)) * tvm.relay.const(float(w)) - tvm.relay.const(1.0)) / tvm.relay.const(
                    2.0
                )
                y = ((y + tvm.relay.const(1.0)) * tvm.relay.const(float(h)) - tvm.relay.const(1.0)) / tvm.relay.const(
                    2.0
                )

            # Step 3: 1-pixel zero-pad; OOB coords clip into the zero border
            im_padded = tvm.relay.nn.pad(im_b, pad_width=((0, 0), (0, 0), (1, 1), (1, 1)))
            padded_h = h + 2
            padded_w = w + 2
            temp_w = padded_w - 1  # max valid padded x index
            temp_h = padded_h - 1  # max valid padded y index

            # Step 4: compute padded integer indices
            # x_shifted = unnorm_x + 1  combines the pad offset with the
            # non-negativity guarantee required by the hardware (workaround A).
            x_shifted = x + tvm.relay.const(1.0, "float32")
            y_shifted = y + tvm.relay.const(1.0, "float32")

            if mode == "bilinear":
                # floor(x_shifted) gives x0_p, the lower padded column index
                x0_p = tvm.relay.cast(tvm.relay.floor(x_shifted), "int32")
                y0_p = tvm.relay.cast(tvm.relay.floor(y_shifted), "int32")
                x1_p = x0_p + tvm.relay.const(1, dtype="int32")  # upper column
                y1_p = y0_p + tvm.relay.const(1, dtype="int32")  # upper row

                # Step 5: bilinear weights (workarounds B and C)
                # dx1 = distance from x_shifted to the *upper* integer boundary
                # dx0 = 1 - dx1  (distance to the *lower* integer boundary)
                dx1 = tvm.relay.cast(x1_p, "float32") - x_shifted
                dy1 = tvm.relay.cast(y1_p, "float32") - y_shifted
                dx0 = tvm.relay.const(1.0) - dx1
                dy0 = tvm.relay.const(1.0) - dy1
                # Squeeze from (1, Hg, Wg) → (Hg, Wg) to avoid broadcast-multiply bug
                wa_2d = tvm.relay.squeeze(dx1 * dy1, axis=[0])  # weight for (x1, y1)
                wb_2d = tvm.relay.squeeze(dx1 * dy0, axis=[0])  # weight for (x1, y0)
                wc_2d = tvm.relay.squeeze(dx0 * dy1, axis=[0])  # weight for (x0, y1)
                wd_2d = tvm.relay.squeeze(dx0 * dy0, axis=[0])  # weight for (x0, y0)

                # Clamp indices to padded image bounds
                x0 = tvm.relay.clip(x0_p, a_min=0, a_max=temp_w)
                y0 = tvm.relay.clip(y0_p, a_min=0, a_max=temp_h)
                x1 = tvm.relay.clip(x1_p, a_min=0, a_max=temp_w)
                y1 = tvm.relay.clip(y1_p, a_min=0, a_max=temp_h)

            elif mode == "nearest":
                # Round to nearest: floor(x + 0.5); combined with pre-shift → floor(x_shifted + 0.5)
                x0 = tvm.relay.clip(
                    tvm.relay.cast(tvm.relay.floor(x_shifted + tvm.relay.const(0.5, "float32")), "int32"),
                    a_min=0,
                    a_max=temp_w,
                )
                y0 = tvm.relay.clip(
                    tvm.relay.cast(tvm.relay.floor(y_shifted + tvm.relay.const(0.5, "float32")), "int32"),
                    a_min=0,
                    a_max=temp_h,
                )
            else:
                assert False, f"Unsupported mode: {mode}. Supported modes: ‘bilinear’, ‘nearest’."

            # Step 6: flatten padded image to (n, c, H_p*W_p) for 1-D index gather
            # flat index = x + y * padded_w  (row-major, computed in float32 — workaround D)
            padded_w_f32 = tvm.relay.const(float(padded_w), dtype="float32")
            im_flat = tvm.relay.reshape(im_padded, (n, c, -1))

            def flat_idx(xi, yi):
                """2-D padded (xi, yi) → 1-D flat index, float32 arithmetic."""
                return tvm.relay.reshape(
                    tvm.relay.cast(
                        tvm.relay.cast(xi, "float32") + tvm.relay.cast(yi, "float32") * padded_w_f32,
                        "int32",
                    ),
                    newshape=[-1],
                )

            idx_x0y0 = flat_idx(x0, y0)
            if mode == "bilinear":
                idx_x0y1 = flat_idx(x0, y1)
                idx_x1y0 = flat_idx(x1, y0)
                idx_x1y1 = flat_idx(x1, y1)

            # Step 7: per-channel gather
            # Split into C channel slices of shape (1, H_p*W_p), transpose each
            # to (H_p*W_p, 1) to match the adv_index Embedding table layout.
            split_channels = tvm.relay.split(im_flat, indices_or_sections=c, axis=1)
            output_parts = []

            for i in range(c):
                ch = tvm.relay.transpose(tvm.relay.squeeze(split_channels[i], axis=[0]), [1, 0])

                if mode == "bilinear":
                    # Concatenate all 4 corner index vectors into one (4*Hg*Wg,) tensor
                    # and issue a SINGLE adv_index — workaround E prevents silent zeros
                    # from multiple adv_index calls on the same channel table.
                    all_idx = tvm.relay.concatenate([idx_x0y0, idx_x0y1, idx_x1y0, idx_x1y1], axis=0)
                    # adv_index((H_p*W_p, 1), (4*Hg*Wg,)) → (4*Hg*Wg, 1)
                    corners = tvm.relay.split(tvm.relay.adv_index([ch, all_idx]), indices_or_sections=4, axis=0)

                    def corner(k, _c=corners):
                        # (Hg*Wg, 1) → transpose → (1, Hg*Wg) → reshape → (Hg, Wg)
                        return tvm.relay.reshape(tvm.relay.transpose(_c[k], [1, 0]), newshape=[gh, gw])

                    val = corner(0) * wa_2d + corner(1) * wb_2d + corner(2) * wc_2d + corner(3) * wd_2d
                else:
                    # nearest: single gather, reshape to (Hg, Wg)
                    sampled = tvm.relay.adv_index([ch, idx_x0y0])  # (Hg*Wg, 1)
                    val = tvm.relay.reshape(tvm.relay.transpose(sampled, [1, 0]), newshape=[gh, gw])

                output_parts.append(tvm.relay.reshape(val, [1, gh, gw]))

            # Step 8: reconstruct (n, c, gh, gw) from per-channel (1, gh, gw) slices
            output = tvm.relay.reshape(
                tvm.relay.concatenate(output_parts, axis=0),  # (c, gh, gw)
                newshape=[n, c, gh, gw],
            )
            results.append(output)

        # Concatenate batch results → (N, C, Hg, Wg)
        return tvm.relay.concatenate(results, axis=0)


class DecomposeFloor(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True, require_type=True)

        self.data = wildcard()
        self.floor = is_op("floor")(self.data)

        self.pattern = self.floor

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        data = node_map[self.data][0]

        # Decompose floor operation using the formula:
        # floor(x) = int(x) - 1 if x < 0 and x != int(x)
        int_part = tvm.relay.cast(data, "int32")
        int_part = tvm.relay.cast(int_part, "float32")
        negative_mask = tvm.relay.less(data, tvm.relay.const(0.0, "float32"))
        not_equal_mask = tvm.relay.not_equal(data, int_part)

        # Compute the adjustment: subtract 1 for negative non-integer values
        adjustment = tvm.relay.where(
            negative_mask, tvm.relay.cast(not_equal_mask, "float32"), tvm.relay.const(0.0, "float32")
        )
        floor_result = tvm.relay.subtract(int_part, adjustment)  # Subtract adjustment to get the floor value
        return floor_result


class DecomposeOnesOrZerosToFull(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.zeros = is_op("zeros")()
        self.ones = is_op("ones")()
        self.pattern = self.zeros | self.ones

    def callback(self, pre, post, node_map):
        shape = pre.attrs.shape
        dtype = pre.attrs.dtype
        op_name = node_map[self.pattern][0].op.name
        value = None
        if op_name == "zeros":
            value = 0
        elif op_name == "ones":
            value = 1
        assert value is not None, "Either zeros or ones should be matched"
        return tvm.relay.full(tvm.relay.const(0, dtype=dtype), shape=shape, dtype=dtype)


class DecomposeMeshgrid(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.inputs = wildcard()
        self.pattern = is_op("meshgrid")(self.inputs)

    def callback(self, pre, post, node_map):

        assert pre.attrs.indexing == "ij", f"Indexing type {pre.attrs.indexing} is not supported yet"

        from tvm.relay.frontend.common import infer_shape

        inputs = node_map[self.inputs][0]

        # Determine the number of input tensors
        num_dims = len(inputs)

        # Get the shape (length) of each input tensor
        lengths = [infer_shape(t)[0] for t in inputs]
        all_ones = all(length == 1 for length in lengths)
        assert not all_ones, "Meshgrid requires all input tensors to be 1D"

        # Create expanded versions of each input tensor for proper broadcasting
        expanded_inputs = []
        for i, tensor in enumerate(inputs):
            expanded = tensor  # Start with the original tensor

            # Expand dimensions along all axes except the current one
            for axis in range(num_dims):
                if axis != i:
                    expanded = tvm.relay.expand_dims(expanded, axis=axis)

            expanded_inputs.append(expanded)  # Store the expanded tensor

        # Broadcast each expanded tensor to match the full meshgrid shape
        broadcasted_grids = [tvm.relay.broadcast_to(expanded_inputs[i], shape=tuple(lengths)) for i in range(num_dims)]

        return tvm.relay.Tuple(broadcasted_grids)


class DecomposeDepthToSpace(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.input = wildcard()
        self.pattern = is_op("nn.depth_to_space")(self.input)

    def callback(self, pre, post, node_map):
        from tvm.relay.frontend.common import infer_shape

        data = node_map[self.input][0]
        attrs = pre.attrs
        blocksize = int(attrs.block_size)

        assert attrs.mode == "CRD", f"Only CRD mode is supported for now, got {attrs.mode}"

        # Get input shape
        shape = infer_shape(data)
        b, c, h, w = shape

        r = blocksize
        out_c = c // (r * r)

        # Reshape to [b, out_c, r, r, h, w]
        reshaped = tvm.relay.reshape(data, [b, out_c, r, r, h, w])

        # Transpose to [b, out_c, h, r, w, r]
        transposed = tvm.relay.transpose(reshaped, axes=[0, 1, 4, 2, 5, 3])

        # Reshape to [b, out_c, h * r, w * r]
        out = tvm.relay.reshape(transposed, [b, out_c, h * r, w * r])

        return out


class RemoveEmptyConcat(DFPatternCallback):
    """
    Relay pass to eliminate unnecessary `concatenate` ops involving empty tensors.

    In some models (e.g., Phi-3), rotary embedding logic performs slicing on the last
    dimension of the query tensor to split it into two parts. If the slicing boundaries
    are incorrectly defined, this may create an empty tensor (e.g., shape with size 0
    along the concatenation axis).

    This pass identifies `concatenate` operations between two tensors where one has
    dimension size 0 along the concatenation axis, and removes the redundant concat
    by returning the non-empty operand directly.

    This prevents downstream errors like:
        `AssertionError: start < operandA.shape[dim]`

    which occur due to operations on invalid tensor slices.

    Example:
        q_rot  = query[..., 0:96]
        q_pass = query[..., 96:96]  # shape: (1, 32, 256, 0)
        concat(q_rot, q_pass, axis=-1)  # → Rewritten to just q_rot
    """

    def __init__(self):
        super().__init__(rewrite_once=False, require_type=True)
        self.act1 = wildcard()
        self.act2 = wildcard()
        self.pattern = is_op("concatenate")(is_tuple([self.act1, self.act2]))

    def callback(self, pre, post, node_map):
        concat_axis = int(node_map[self.pattern][0].attrs.axis)
        act1 = node_map[self.act1][0]
        act2 = node_map[self.act2][0]
        act1_shape = list(act1.checked_type.shape)
        act2_shape = list(act2.checked_type.shape)
        if act1_shape[concat_axis] == 0:
            return act2
        elif act2_shape[concat_axis] == 0:
            return act1
        else:
            return post


def _get_callback_name(callback):
    if isinstance(callback, DFPatternCallback):
        return type(callback).__name__
    elif isinstance(callback, tvm.transform.Pass):
        return callback.info.name
    else:
        raise NotImplementedError(f"Type of callback ({(callback)}) not implemented")


def _run_pattern_callback(relay_module, callback, callback_name):
    if isinstance(callback, DFPatternCallback):
        relay_module["main"] = rewrite(callback, relay_module["main"])
    elif isinstance(callback, tvm.transform.Pass):
        relay_module = tvm.transform.Sequential([callback])(relay_module)
    else:
        raise NotImplementedError(f"Type of callback ({type(callback)}) not implemented")

    logger.trace(f"After {callback_name}")
    logger.trace(relay_module.functions)

    return relay_module


def run_pattern_callbacks(
    relay_module, callbacks, params=None, inputs=None, target=None, framework_outputs=None, verify_cfg=None
):

    run_verify = verify_cfg and params and inputs and target and framework_outputs and verify_cfg.verify_each_forge_pass
    if verify_cfg and verify_cfg.verify_each_forge_pass and not run_verify:
        logger.warning(
            f"Cannot verify relay module after forge passes because one of (params, inputs, target, golden_outputs, veirfy_cfg) is None"
        )

    for callback in callbacks:
        callback_name = _get_callback_name(callback)
        try:
            relay_module = _run_pattern_callback(relay_module, callback, callback_name)
        except Exception as ex:
            logger.error(f'Failed on "{callback_name}" TVM callback')
            raise ex
        if run_verify:
            logger.trace(f"Verifying {callback_name}")
            verify_tvm_compile(relay_module, params, inputs, target, framework_outputs, callback_name, verify_cfg)

    return relay_module


def run_forge_compile_passes(
    relay_module, params=None, inputs=None, target=None, framework_outputs=None, verify_cfg=None
):
    return run_pattern_callbacks(
        relay_module,
        [
            DecomposeDepthToSpace(),
            DecomposeOnesOrZerosToFull(),
            DecomposeMeshgrid(),
            DecomposeGridSample(),
            DecomposeFloor(),
            ExpandMultipleDims(),
            DecomposeReverse(),
            ConvertLayout(),
            ResolveConvChannels(),
            FuseConvAndPoolPadding(),
            DecomposeDynamicResize2d(),
            DecomposePRelu(),
            DecomposeRoll(),
            DecomposeStack(),
            SimplifyGroupNorm(),
            DecomposeVariance(),
            ArgmaxAndMaxReconstruct(),
            ConvertArgmaxTakeToReduceMax(),
            DecompEinsumWithWTranspose(),
            DecompWTranspose(),
            DecomposeEinsum(),
            DecomposeLayoutTransform(),
            LiftLinearSplit(),
            LowerSplitToStridedSlice(),
            DenseWeightTranspose(),
            DecomposePower(),
            DecomposeNegative(),
            DecomposeRsqrt(),
            ExplicateTranspose(),
            DecomposeConv1DToConv2D(),
            PopulateReduceAxes(),
            DecomposeMultiAxisMax(),
            DecomposeMultiAxisTranspose(),
            EstimateWhereInCausalMask(),
            CastWhereConditionToBool(),
            LowerAdaptiveAvgPool(),
            LowerAdaptiveMaxPool(),
            SimplifyTransposeReshape(),
            DecomposeMultiDimSqueeze(),
            PopulateTransposeAxes(),
            PopulateStridedSliceAxes(),
            DecomposeMultiAxisMean(),
            DecomposeMultiAxisSum(),
            ReconstructOnnxResize2d(),
            DecomposeMultiAxisBroadcast(),
            EnsureKeepdims(),
            RemoveRedundantTake(),
            RemoveRedundantReshape(),
            LowerCopyToNOP(),
            TransposePad(),
            DecomposeMultiRangeTake(),
            LowerTakeToStridedSlice(),
            ConvertAddToBiasAddAfterConv2d(),
            DecomposeBatchFlatten(),
            ConvertGlobalAvgPool2dtoAvgPool2d(),
            ConvertUpsampleToResize2d(),
            DecomposeMultiIndexAdvIndex(),
            ReconstructOnnxQuantizedGelu(),
            DecomposeQnnConcat(),
            ReconstructTFGelu(),
            ReconstructOnnxGelu(),
            ReconstructPyTorchGeluNew(),
            ReconstructPyTorchGelu(),
            ReconstructJaxGelu(),
            ReconstructTFLayerNorm(),
            RepositionQNormScalarMultiplier(),
            CombineReshapes(),
            ReconstructJaxLayerNorm(),
            RemoveRedundantTranposesBetwenAvgPoolAndFlatteningReshape(),
            RemoveRedundantReshapeTransposeReshape(),
            SimplifyReshape(),
            ReplicateForgeReshapeTranspose(),
            CommuteIndexPastReshape(),
            AttemptRemoveStackWDim(),
            DecomposeScatterND(),
            ConvertIsNaN(),
            RemoveStopFusionAnnotationNodes(),
            Enforce1DOutputForArgwhereOp(),
            BroadcastScatterValuesToMatchIndices(),
            InverseMaskGen(),
            GQABroadcastReshape(),
            RemoveDenseInputSqueeze(),
            RemoveEmptyConcat(),
        ],
        params=params,
        inputs=inputs,
        target=target,
        framework_outputs=framework_outputs,
        verify_cfg=verify_cfg,
    )
