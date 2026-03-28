# qnn_native_linear

原生 QNN 小实验，不经过 MNN backend。

当前实现：

- `qnn_linear_front_probe`
- 只验证 `LinearAttention` 前半段：
  - prepend zero conv state
  - `Concat`
  - `Transpose`
  - `Reshape`
  - `DepthWiseConv2d`
  - `Sigmoid`
  - `ElementWiseMultiply` (`SiLU`)
  - `Split(q/k/v)`

用途：

- 单独验证 `QNNLinearAttention` 前半段的卷积权重打包和张量接线是否正确
- 避开 MNN `Module::load` / direct QNN state 接线问题

## Host build

```bash
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.42.0.251225
cmake -S tools/qnn_native_linear -B build_qnn_native_linear
cmake --build build_qnn_native_linear -j8
```

## Host run

```bash
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.42.0.251225
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
build_qnn_native_linear/qnn_linear_front_probe \
  --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnCpu.so
```

## Android build

```bash
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.42.0.251225
cmake -S tools/qnn_native_linear -B build_qnn_native_linear_android \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-31 \
  -DCMAKE_TOOLCHAIN_FILE=/opt/ndk/build/cmake/android.toolchain.cmake
cmake --build build_qnn_native_linear_android -j8
```
