#loc = loc("add_rs_quasar":0:0)
module @add_rs_quasar {
  func.func @forward(%arg0: tensor<2x32x32xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input_A"} loc("add_rs_quasar":0:0), %arg1: tensor<2x32x32xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input_B"} loc("add_rs_quasar":0:0)) -> (tensor<2x32x32xf32> {ttir.name = "add_rs_quasar_1.output_Add_1"}) {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<2x32x32xf32>, tensor<2x32x32xf32>) -> tensor<2x32x32xf32> loc(#loc3)
    return %0 : tensor<2x32x32xf32> loc(#loc2)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("Add_0")
#loc2 = loc(unknown)
#loc3 = loc("Add_1"(#loc1))
