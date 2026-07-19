# How to run standalone MLIR, based on generated Forge-ONNX MLIR graphs


1. Change Directory to tt-mlir repo in tt-forge-onnx third parties
    ```bash
    $ cd tt-forge-onnx/third_party/tt-mlir
    ```

2. Build TTRT (once) - (Inside tt-mlir repo)
    ```bash
    $ uv pip install patchelf
    $ cmake --build build -- ttrt
    ```

3. Save system descriptor artifacts file. For more info, refer [ttrt docs](https://docs.tenstorrent.com/tt-mlir/ttrt.html#generate-a-flatbuffer-file-from-compiler)
    ```bash
    $ ttrt query --save-artifacts
    ```

4. Convert TTIR MLIR to TTNN MLIR
    - Save ttir mlir from logs in <some_name>_ttir.mlir . Ex: softmax_check_ttir.mlir
    - The first line of TTIR MLIR should be like below.
        ```mlir
        module attributes {} {
        ```

        Ex. softmax_check_ttir.mlir
        ```mlir
        module attributes {} {
            func.func @forward(%arg0: tensor<13x89x3xf32> {ttir.name = "x"}, %arg1: tensor<13x89x3xf32> {ttir.name = "y"}, %arg2: tensor<1x89x3xf32> {ttir.name = "input_0_multiply_1"}, %arg3: tensor<1x89x3xf32> {ttir.name = "input_0_reciprocal_0"}) -> (tensor<13x89x3xf32> {ttir.name = "ModelConstEvalPass.output_add_3"}) {
                %0 = tensor.empty() : tensor<1x89x3xf32>
                %1 = "ttir.reciprocal"(%arg3, %0) : (tensor<1x89x3xf32>, tensor<1x89x3xf32>) -> tensor<1x89x3xf32>
                %2 = tensor.empty() : tensor<1x89x3xf32>
                %3 = "ttir.multiply"(%arg2, %1, %2) : (tensor<1x89x3xf32>, tensor<1x89x3xf32>, tensor<1x89x3xf32>) -> tensor<1x89x3xf32>
                %4 = tensor.empty() : tensor<13x89x3xf32>
                %5 = "ttir.add"(%arg0, %arg1, %4) : (tensor<13x89x3xf32>, tensor<13x89x3xf32>, tensor<13x89x3xf32>) -> tensor<13x89x3xf32>
                %6 = tensor.empty() : tensor<13x89x3xf32>
                %7 = "ttir.add"(%3, %5, %6) : (tensor<1x89x3xf32>, tensor<13x89x3xf32>, tensor<13x89x3xf32>) -> tensor<13x89x3xf32>
                return %7 : tensor<13x89x3xf32>
            }
        }
        ```
    - Generate TTNN MLIR from TTIR MLIR
        - Replace path to `system_desc.ttsys` to your corresponding path.
        ```bash
        $ ./build/bin/ttmlir-opt --ttir-load-system-desc="path=/proj_sw/user_dev/akannan/forge/tt-forge-onnx/third_party/tt-mlir/ttrt-artifacts/system_desc.ttsys" --ttir-to-ttnn-backend-pipeline softmax_check_ttir.mlir -o softmax_check_ttnn.mlir
        ```

5. Create Flatbuffers Serialized Binary
    - Generate flatbuffer binary from TTNN MLIR
        ```bash
        $ ./build/bin/ttmlir-translate --ttnn-to-flatbuffer softmax_check_ttnn.mlir -o softmax_check.ttnn
        ```

6. Run TTNN Binary
    ```bash
    $ ttrt run softmax_check.ttnn
    ```
