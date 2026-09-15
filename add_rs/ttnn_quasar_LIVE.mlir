#dram = #ttnn.buffer_type<dram>
#loc = loc("add_rs_live":0:0)
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <quasar>, grid = 4x8, coord_translation_offsets = 2x2, l1_size = 4194304, num_dram_channels = 2, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 64, noc_dram_address_align_bytes = 64, l1_unreserved_base = 313088, erisc_l1_unreserved_base = 88576, dram_unreserved_base = 1048704, dram_unreserved_end = 1068732416, supported_data_types = [<f32>, <f16>, <bf16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 4, num_datamovement_threads = 6, dram_grid = 1x2, dram_bank_to_logical_worker_noc0 = [], dram_bank_to_logical_worker_noc1 = []}], [0], [1 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<64x32xf32, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<2x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module @add_rs_live attributes {ttcore.system_desc = #system_desc} {
  ttcore.device_module {
    builtin.module @add_rs_live attributes {ttcore.system_desc = #system_desc, ttnn.l1_const_eval_usage = 1024 : ui64} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<4x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x2>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 2, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 2) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]> loc(#loc)
      func.func @forward(%arg0: tensor<2x32x32xf32, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input_A"} loc("add_rs_live":0:0), %arg1: tensor<2x32x32xf32, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input_B"} loc("add_rs_live":0:0)) -> (tensor<2x32x32xf32, #ttnn_layout1> {ttir.name = "add_rs_live.output_Add_0"}) attributes {tt.function_type = "forward_device"} {
        %0 = "ttnn.to_layout"(%arg0) : (tensor<2x32x32xf32, #ttnn_layout>) -> tensor<2x32x32xf32, #ttnn_layout1> loc(#loc)
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<2x32x32xf32, #ttnn_layout>) -> () loc(#loc)
        %1 = "ttnn.to_layout"(%arg1) : (tensor<2x32x32xf32, #ttnn_layout>) -> tensor<2x32x32xf32, #ttnn_layout1> loc(#loc)
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<2x32x32xf32, #ttnn_layout>) -> () loc(#loc)
        %2 = "ttnn.add"(%0, %1) : (tensor<2x32x32xf32, #ttnn_layout1>, tensor<2x32x32xf32, #ttnn_layout1>) -> tensor<2x32x32xf32, #ttnn_layout1> loc(#loc3)
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<2x32x32xf32, #ttnn_layout1>) -> () loc(#loc3)
        "ttnn.deallocate"(%0) <{force = false}> : (tensor<2x32x32xf32, #ttnn_layout1>) -> () loc(#loc3)
        return %2 : tensor<2x32x32xf32, #ttnn_layout1> loc(#loc2)
      } loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("Add_0")
#loc2 = loc(unknown)
#loc3 = loc("Add_0"(#loc1))
