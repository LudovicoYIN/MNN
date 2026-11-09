#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "core/TensorUtils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace MNN;

static bool run(int precision, int batch, int dim, int maskMode) {
    BackendConfig config;
    config.precision = static_cast<BackendConfig::PrecisionMode>(precision);
    CUDA::CUDARuntimeWrapper runtime(config.precision, config.power, config.memory);
    std::unique_ptr<Backend> backend(runtime.onCreate(&config, nullptr));
    if (!backend) return false;
    OpT op;
    op.type = OpType_Attention;
    op.main.type = OpParameter_AttentionParam;
    op.main.value = new AttentionParamT;
    op.main.AsAttentionParam()->kv_cache = true;
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Op::Pack(builder, &op));
    std::unique_ptr<Execution> execution;
    const int heads = 16, kvheads = 2;
    int past = 0;
    float worst = 0;
    for (int added : {127, 1, 1, 127, 1}) {
        const int total = past + added;
        // Every batch has the same mask: batch-specific mask semantics are outside this fixture.
        const int masklen = maskMode == 0 && total % 2 ? 1 : total;
        const std::vector<std::vector<int>> shapes = {
            {batch, 1, heads, dim}, {batch, added, kvheads, dim},
            {batch, added, kvheads, dim}, {batch, 1, 1, masklen}, {batch, 1, heads, dim}};
        std::vector<std::unique_ptr<Tensor>> device, host;
        for (auto& shape : shapes) {
            device.emplace_back(Tensor::createDevice<float>(shape, Tensor::TENSORFLOW));
            host.emplace_back(Tensor::create<float>(shape, nullptr, Tensor::TENSORFLOW));
            if (!backend->onAcquireBuffer(device.back().get(), Backend::STATIC)) return false;
            std::fill(host.back()->host<float>(), host.back()->host<float>() + host.back()->elementSize(), 0.f);
        }
        for (int b = 0; b < batch; ++b) for (int t = 0; t < added; ++t)
            for (int h = 0; h < kvheads; ++h) for (int d = 0; d < dim; ++d)
                host[2]->host<float>()[((b * added + t) * kvheads + h) * dim + d] =
                    float((past + t) % 7) / 8.f + b / 4.f + h / 8.f + (d % 3) / 16.f;
        for (int b = 0; b < batch; ++b) for (int t = 0; t < masklen; ++t) {
            const bool hidden = maskMode == 0 ? t == masklen - 1 : (maskMode == 1 ? t != 0 : true);
            host[3]->host<float>()[b * masklen + t] = hidden ? -INFINITY : 0.f;
        }
        for (int i = 0; i < 4; ++i) backend->onCopyBuffer(host[i].get(), device[i].get());
        std::vector<Tensor*> inputs, outputs{device[4].get()};
        for (int i = 0; i < 4; ++i) inputs.push_back(device[i].get());
        if (!execution) execution.reset(backend->onCreate(inputs, outputs, flatbuffers::GetRoot<Op>(builder.GetBufferPointer())));
        if (!execution || execution->onResize(inputs, outputs) != NO_ERROR || execution->onExecute(inputs, outputs) != NO_ERROR) return false;
        backend->onCopyBuffer(device[4].get(), host[4].get());
        double tokenMean = 0;
        for (int t = 0; t < total - 1; ++t) tokenMean += float(t % 7) / 8.f;
        tokenMean /= total - 1;
        for (int b = 0; b < batch; ++b) for (int h = 0; h < heads; ++h) for (int d = 0; d < dim; ++d) {
            const float offset = b / 4.f + (h / (heads / kvheads)) / 8.f + (d % 3) / 16.f;
            const float expected = maskMode == 2 ? 0.f : offset + (maskMode == 1 ? 0.f : tokenMean);
            const float actual = host[4]->host<float>()[(b * heads + h) * dim + d];
            if (!std::isfinite(actual)) return false;
            worst = std::max(worst, std::abs(actual - expected));
        }
        if (worst > (precision == 2 ? 0.002f : 0.00002f)) return false;
        for (auto& tensor : device) backend->onReleaseBuffer(tensor.get(), Backend::STATIC);
        std::printf("PASS precision=%d batch=%d dim=%d total_KV=%d mask_length=%d mask_mode=%d max_abs_error=%g\n", precision, batch, dim, total, masklen, maskMode, worst);
        past = total;
    }
    return true;
}

int main() {
    for (int p : {0, 2}) for (int b : {1, 2}) for (int d : {64, 80, 128, 256}) for (int mask : {0, 1, 2})
        if (!run(p, b, d, mask)) { std::fprintf(stderr, "FAIL p=%d B=%d D=%d mask=%d\n", p, b, d, mask); return 1; }
    return 0;
}
