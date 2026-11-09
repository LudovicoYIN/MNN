#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "core/TensorUtils.hpp"
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace MNN;

static bool run(int precision, bool packedQ, bool packedK, int seq, int dim, int rotary, bool norm) {
    BackendConfig config;
    config.precision = static_cast<BackendConfig::PrecisionMode>(precision);
    CUDA::CUDARuntimeWrapper runtime(config.precision, config.power, config.memory);
    std::unique_ptr<Backend> backend(runtime.onCreate(&config, nullptr));
    if (!backend) return false;
    const int qHeads = 3, kHeads = 1;
    std::vector<std::unique_ptr<Tensor>> devices, hosts;
    const std::vector<std::vector<int>> shapes = {
        {seq, qHeads * dim, 1, 1}, {seq, kHeads * dim, 1, 1}, {seq, rotary}, {seq, rotary},
        {1, seq, qHeads, dim}, {1, seq, kHeads, dim}};
    for (int i = 0; i < 6; ++i) {
        auto format = i >= 4 ? Tensor::TENSORFLOW : Tensor::CAFFE;
        devices.emplace_back(Tensor::createDevice<float>(shapes[i], format));
        hosts.emplace_back(Tensor::create<float>(shapes[i], nullptr, format));
        if ((i == 0 && packedQ) || (i == 1 && packedK))
            TensorUtils::getDescribe(devices.back().get())->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        if (!backend->onAcquireBuffer(devices.back().get(), Backend::STATIC)) return false;
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < hosts[i]->elementSize(); ++j)
            hosts[i]->host<float>()[j] = ((j % 17) - 8) * 0.13f;
    for (int t = 0; t < seq; ++t)
        for (int d = 0; d < rotary; ++d) {
            const float angle = 0.013f * (t + 1) * (d + 1);
            hosts[2]->host<float>()[t * rotary + d] = std::cos(angle);
            hosts[3]->host<float>()[t * rotary + d] = std::sin(angle);
        }
    for (int i = 0; i < 4; ++i) backend->onCopyBuffer(hosts[i].get(), devices[i].get());
    OpT op;
    op.type = OpType_RoPE;
    op.main.type = OpParameter_RoPEParam;
    op.main.value = new RoPEParamT;
    auto param = op.main.AsRoPEParam();
    param->num_head = qHeads;
    param->kv_num_head = kHeads;
    param->head_dim = dim;
    param->rope_cut_head_dim = rotary;
    if (norm) {
        param->q_norm.reset(new LayerNormT);
        param->k_norm.reset(new LayerNormT);
        for (auto p : {param->q_norm.get(), param->k_norm.get()}) {
            p->useRMSNorm = true;
            p->epsilon = 1e-6f;
            for (int d = 0; d < dim; ++d) p->gamma.push_back(0.7f + 0.003f * d);
        }
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Op::Pack(builder, &op));
    std::vector<Tensor*> inputs, outputs;
    for (int i = 0; i < 4; ++i) inputs.push_back(devices[i].get());
    for (int i = 4; i < 6; ++i) outputs.push_back(devices[i].get());
    std::unique_ptr<Execution> execution(backend->onCreate(inputs, outputs, flatbuffers::GetRoot<Op>(builder.GetBufferPointer())));
    if (!execution || execution->onResize(inputs, outputs) != NO_ERROR ||
        execution->onExecute(inputs, outputs) != NO_ERROR) return false;
    float maxError = 0;
    for (int i = 0; i < 2; ++i) {
        backend->onCopyBuffer(devices[i + 4].get(), hosts[i + 4].get());
        const int heads = i == 0 ? qHeads : kHeads;
        for (int t = 0; t < seq; ++t) for (int h = 0; h < heads; ++h) {
            std::vector<float> value(dim);
            float sum = 0;
            for (int d = 0; d < dim; ++d) {
                value[d] = hosts[i]->host<float>()[(t * heads + h) * dim + d];
                sum += value[d] * value[d];
            }
            if (norm) for (int d = 0; d < dim; ++d)
                value[d] *= (0.7f + 0.003f * d) / std::sqrt(sum / dim + 1e-6f);
            for (int d = 0; d < dim; ++d) {
                float expected = value[d];
                if (d < rotary) {
                    const int half = rotary / 2;
                    expected = value[d] * hosts[2]->host<float>()[t * rotary + d] +
                        (d < half ? -value[d + half] : value[d - half]) * hosts[3]->host<float>()[t * rotary + d];
                }
                const float actual = hosts[i + 4]->host<float>()[(t * heads + h) * dim + d];
                if (!std::isfinite(actual)) return false;
                maxError = std::max(maxError, std::abs(actual - expected));
            }
        }
    }
    std::printf("precision=%d qPacked=%d kPacked=%d seq=%d dim=%d rotary=%d norm=%d maxAbs=%g\n",
                precision, packedQ, packedK, seq, dim, rotary, norm, maxError);
    return maxError < (precision == 2 ? 0.01f : 2e-5f);
}

int main() {
    int passed = 0;
    for (int precision : {0, 2}) for (int layout = 0; layout < 4; ++layout)
        for (int shape = 0; shape < 3; ++shape) for (bool norm : {false, true}) {
            const int dim = shape == 0 ? 6 : (shape == 1 ? 128 : 256);
            const int rotary = shape == 0 ? 4 : (shape == 1 ? 128 : 64);
            if (!run(precision, layout & 1, layout & 2, shape == 1 ? 1 : 3, dim, rotary, norm)) return 1;
            ++passed;
        }
    std::printf("PASS %d direct-backend cases\n", passed);
}
