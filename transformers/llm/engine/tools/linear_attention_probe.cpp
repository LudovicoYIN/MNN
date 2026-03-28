#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "MNN_generated.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sys/stat.h>
#include <sys/types.h>
#include <sstream>
#include <string>
#include <vector>

using namespace MNN;
using namespace MNN::Express;

namespace {

struct FloatCompareResult {
    size_t size = 0;
    double maxAbs = 0.0;
    double meanAbs = 0.0;
    double rmse = 0.0;
    double cosine = 0.0;
    size_t nanCount = 0;
    size_t infCount = 0;
};

struct ProbeConfig {
    std::string backend = "both";
    int batch = 1;
    int numKHeads = 16;
    int numVHeads = 16;
    int headKDim = 128;
    int headVDim = 128;
    int kernelSize = 4;
    int prefillSeqLen = 1;
    int decodeSteps = 2;
    bool useL2Norm = true;
    float qkvScale = 0.01f;
    float gateBase = -0.2f;
    float betaBase = 0.2f;
    float decodeOffset = 0.005f;
    std::string dumpPrefix;
    std::string inputDir;
};

static void printUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [options]\n";
    std::cerr << "Options:\n";
    std::cerr << "  --backend cpu|npu|both      Backend to run. Default: both\n";
    std::cerr << "  --batch N                   Batch size. Default: 1\n";
    std::cerr << "  --num-k-heads N            Number of key heads. Default: 16\n";
    std::cerr << "  --num-v-heads N            Number of value heads. Default: 16\n";
    std::cerr << "  --head-k-dim N             Key head dim. Default: 128\n";
    std::cerr << "  --head-v-dim N             Value head dim. Default: 128\n";
    std::cerr << "  --kernel-size N            Conv kernel size. Default: 4\n";
    std::cerr << "  --prefill-seq N            Prefill sequence length. Default: 1\n";
    std::cerr << "  --decode-steps N           Number of decode steps. Default: 2\n";
    std::cerr << "  --no-l2norm                Disable q/k l2 norm\n";
    std::cerr << "  --qkv-scale F              QKV input scale. Default: 0.01\n";
    std::cerr << "  --gate-base F              Base negative gate value. Default: -0.2\n";
    std::cerr << "  --beta-base F              Base beta value. Default: 0.2\n";
    std::cerr << "  --decode-offset F          Per-step qkv offset. Default: 0.005\n";
    std::cerr << "  --dump-prefix PATH         Dump CPU outputs to PATH_*.bin/.json\n";
    std::cerr << "  --input-dir PATH           Load qkv/gate/beta/conv_weight/output_ref from PATH\n";
}

static bool parseIntArg(const char* text, int& value) {
    char* end = nullptr;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

static bool parseFloatArg(const char* text, float& value) {
    char* end = nullptr;
    float parsed = strtof(text, &end);
    if (end == text || *end != '\0') {
        return false;
    }
    value = parsed;
    return true;
}

static bool parseArgs(int argc, char** argv, ProbeConfig& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto needValue = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--backend") {
            auto value = needValue("--backend");
            if (!value) {
                return false;
            }
            cfg.backend = value;
        } else if (arg == "--batch") {
            auto value = needValue("--batch");
            if (!value || !parseIntArg(value, cfg.batch)) {
                return false;
            }
        } else if (arg == "--num-k-heads") {
            auto value = needValue("--num-k-heads");
            if (!value || !parseIntArg(value, cfg.numKHeads)) {
                return false;
            }
        } else if (arg == "--num-v-heads") {
            auto value = needValue("--num-v-heads");
            if (!value || !parseIntArg(value, cfg.numVHeads)) {
                return false;
            }
        } else if (arg == "--head-k-dim") {
            auto value = needValue("--head-k-dim");
            if (!value || !parseIntArg(value, cfg.headKDim)) {
                return false;
            }
        } else if (arg == "--head-v-dim") {
            auto value = needValue("--head-v-dim");
            if (!value || !parseIntArg(value, cfg.headVDim)) {
                return false;
            }
        } else if (arg == "--kernel-size") {
            auto value = needValue("--kernel-size");
            if (!value || !parseIntArg(value, cfg.kernelSize)) {
                return false;
            }
        } else if (arg == "--prefill-seq") {
            auto value = needValue("--prefill-seq");
            if (!value || !parseIntArg(value, cfg.prefillSeqLen)) {
                return false;
            }
        } else if (arg == "--decode-steps") {
            auto value = needValue("--decode-steps");
            if (!value || !parseIntArg(value, cfg.decodeSteps)) {
                return false;
            }
        } else if (arg == "--qkv-scale") {
            auto value = needValue("--qkv-scale");
            if (!value || !parseFloatArg(value, cfg.qkvScale)) {
                return false;
            }
        } else if (arg == "--gate-base") {
            auto value = needValue("--gate-base");
            if (!value || !parseFloatArg(value, cfg.gateBase)) {
                return false;
            }
        } else if (arg == "--beta-base") {
            auto value = needValue("--beta-base");
            if (!value || !parseFloatArg(value, cfg.betaBase)) {
                return false;
            }
        } else if (arg == "--decode-offset") {
            auto value = needValue("--decode-offset");
            if (!value || !parseFloatArg(value, cfg.decodeOffset)) {
                return false;
            }
        } else if (arg == "--dump-prefix") {
            auto value = needValue("--dump-prefix");
            if (!value) {
                return false;
            }
            cfg.dumpPrefix = value;
        } else if (arg == "--input-dir") {
            auto value = needValue("--input-dir");
            if (!value) {
                return false;
            }
            cfg.inputDir = value;
        } else if (arg == "--no-l2norm") {
            cfg.useL2Norm = false;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return false;
        }
    }
    return true;
}

static MNNForwardType parseBackendType(const std::string& backend) {
    if (backend == "cpu") {
        return MNN_FORWARD_CPU;
    }
    if (backend == "npu") {
        return MNN_FORWARD_NN;
    }
    return MNN_FORWARD_AUTO;
}

static std::string shapeString(VARP var) {
    if (var == nullptr || var->getInfo() == nullptr) {
        return "<null>";
    }
    std::ostringstream os;
    os << "[";
    for (int i = 0; i < var->getInfo()->dim.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << var->getInfo()->dim[i];
    }
    os << "]";
    return os.str();
}

static FloatCompareResult compareFloatBuffer(const float* ref, const float* cand, size_t size) {
    FloatCompareResult result;
    result.size = size;
    if (size == 0) {
        return result;
    }
    double sumAbs = 0.0;
    double sumSq = 0.0;
    double dot = 0.0;
    double normRef = 0.0;
    double normCand = 0.0;
    for (size_t i = 0; i < size; ++i) {
        const double a = ref[i];
        const double b = cand[i];
        if (std::isnan(a) || std::isnan(b)) {
            result.nanCount++;
        }
        if (std::isinf(a) || std::isinf(b)) {
            result.infCount++;
        }
        const double diff = std::abs(a - b);
        result.maxAbs = std::max(result.maxAbs, diff);
        sumAbs += diff;
        sumSq += diff * diff;
        dot += a * b;
        normRef += a * a;
        normCand += b * b;
    }
    result.meanAbs = sumAbs / static_cast<double>(size);
    result.rmse = std::sqrt(sumSq / static_cast<double>(size));
    const double denom = std::sqrt(normRef) * std::sqrt(normCand);
    result.cosine = denom > 0.0 ? dot / denom : 0.0;
    return result;
}

static void printCompare(const std::string& tag, const FloatCompareResult& result) {
    std::cout << "[" << tag << "]"
              << " size=" << result.size
              << " maxAbs=" << std::fixed << std::setprecision(6) << result.maxAbs
              << " meanAbs=" << result.meanAbs
              << " rmse=" << result.rmse
              << " cosine=" << result.cosine
              << " nan=" << result.nanCount
              << " inf=" << result.infCount
              << "\n";
}

static std::shared_ptr<Module> makeLinearAttentionModule(MNNForwardType backendType,
                                                         int batch,
                                                         int seqLen,
                                                         int kernelSize,
                                                         int convDim,
                                                         int numKHeads,
                                                         int numVHeads,
                                                         int headKDim,
                                                         int headVDim,
                                                         bool useL2Norm) {
    auto qkv = _Input({batch, convDim, seqLen}, NCHW, halide_type_of<float>());
    auto gate = _Input({batch, seqLen, numVHeads}, NCHW, halide_type_of<float>());
    auto beta = _Input({batch, seqLen, numVHeads}, NCHW, halide_type_of<float>());
    auto convW = _Input({convDim, 1, kernelSize}, NCHW, halide_type_of<float>());

    std::shared_ptr<MNN::OpT> op(new MNN::OpT);
    op->type = MNN::OpType_LinearAttention;
    op->main.type = MNN::OpParameter_LinearAttentionParam;
    op->main.value = new MNN::LinearAttentionParamT;
    auto* param = op->main.AsLinearAttentionParam();
    param->attn_type = "gated_delta_rule";
    param->num_k_heads = numKHeads;
    param->num_v_heads = numVHeads;
    param->head_k_dim = headKDim;
    param->head_v_dim = headVDim;
    param->use_qk_l2norm = useL2Norm;

    auto output = Variable::create(Expr::create(op.get(), {qkv, gate, beta, convW}));
    auto buffer = Variable::save({output});
    if (const char* dumpPath = ::getenv("MNN_LINEAR_ATTENTION_DUMP_MODEL")) {
        std::ofstream ofs(dumpPath, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    }

    ScheduleConfig config;
    BackendConfig backendConfig;
    config.type = backendType;
    config.numThread = 1;
    backendConfig.memory = BackendConfig::Memory_Low;
    backendConfig.precision = BackendConfig::Precision_Low;
    backendConfig.power = BackendConfig::Power_Low;
    config.backendConfig = &backendConfig;

    auto runtimeManager = std::shared_ptr<Executor::RuntimeManager>(Executor::RuntimeManager::createRuntimeManager(config));
    if (runtimeManager == nullptr) {
        return nullptr;
    }
    if (backendType == MNN_FORWARD_NN) {
        runtimeManager->setHint(MNN::Interpreter::INIT_THREAD_NUMBER, 1);
        runtimeManager->setCache(".");
    }
    return std::shared_ptr<Module>(Module::load({"Input1", "Input2", "Input3", "Input4"}, {},
                                                reinterpret_cast<uint8_t*>(buffer.data()), buffer.size(), runtimeManager));
}

static void fillDeterministic(std::vector<float>& data, float scale, float offset) {
    for (int i = 0; i < data.size(); ++i) {
        data[i] = ((i % 23) - 11) * scale + offset;
    }
}

static void fillGate(std::vector<float>& data, float base) {
    for (int i = 0; i < data.size(); ++i) {
        data[i] = base - 0.03f * static_cast<float>(i % 7);
    }
}

static void fillBeta(std::vector<float>& data, float base) {
    for (int i = 0; i < data.size(); ++i) {
        float value = base + 0.05f * static_cast<float>(i % 5);
        data[i] = std::max(0.01f, std::min(0.99f, value));
    }
}

static void fillConvWeight(std::vector<float>& data) {
    for (int i = 0; i < data.size(); ++i) {
        data[i] = ((i % 9) - 4) * 0.02f;
    }
}

static VARP makeInputVar(const std::vector<int>& dims, const std::vector<float>& values) {
    auto var = _Input(dims, NCHW, halide_type_of<float>());
    ::memcpy(var->writeMap<float>(), values.data(), values.size() * sizeof(float));
    return var;
}

static void ensureParentDir(const std::string& path) {
    auto pos = path.rfind('/');
    if (pos == std::string::npos) {
        return;
    }
    std::string dir = path.substr(0, pos);
    if (dir.empty()) {
        return;
    }
    std::string current;
    for (size_t i = 0; i < dir.size(); ++i) {
        current.push_back(dir[i]);
        if (dir[i] == '/' || i + 1 == dir.size()) {
            if (!current.empty() && current != "/") {
                ::mkdir(current.c_str(), 0755);
            }
        }
    }
}

static void dumpTensor(const std::string& prefix, const std::vector<int>& shape, const std::vector<float>& values) {
    if (prefix.empty()) {
        return;
    }
    ensureParentDir(prefix);
    std::ofstream bin(prefix + ".bin", std::ios::binary);
    bin.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    std::ofstream json(prefix + ".json");
    json << "{\n";
    json << "  \"dtype\": \"float32\",\n";
    json << "  \"shape\": [";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) json << ", ";
        json << shape[i];
    }
    json << "],\n";
    json << "  \"elements\": " << values.size() << ",\n";
    json << "  \"file\": \"" << prefix.substr(prefix.find_last_of('/') == std::string::npos ? 0 : prefix.find_last_of('/') + 1) << ".bin\"\n";
    json << "}\n";
}

static bool readWholeFile(const std::string& path, std::vector<char>& data) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to open " << path << "\n";
        return false;
    }
    ifs.seekg(0, std::ios::end);
    auto size = ifs.tellg();
    if (size < 0) {
        std::cerr << "Failed to stat " << path << "\n";
        return false;
    }
    ifs.seekg(0, std::ios::beg);
    data.resize(static_cast<size_t>(size));
    if (!data.empty()) {
        ifs.read(data.data(), size);
        if (!ifs) {
            std::cerr << "Failed to read " << path << "\n";
            return false;
        }
    }
    return true;
}

static bool parseShapeFromJson(const std::string& text, std::vector<int>& shape) {
    auto pos = text.find("\"shape\"");
    if (pos == std::string::npos) {
        return false;
    }
    pos = text.find('[', pos);
    if (pos == std::string::npos) {
        return false;
    }
    auto end = text.find(']', pos);
    if (end == std::string::npos) {
        return false;
    }
    std::stringstream ss(text.substr(pos + 1, end - pos - 1));
    std::string item;
    shape.clear();
    while (std::getline(ss, item, ',')) {
        std::stringstream value(item);
        int v = 0;
        value >> v;
        shape.push_back(v);
    }
    return !shape.empty();
}

static bool loadTensorDirEntry(const std::string& dir,
                               const std::string& name,
                               std::vector<int>& shape,
                               std::vector<float>& values) {
    std::vector<char> jsonData;
    if (!readWholeFile(dir + "/" + name + ".json", jsonData)) {
        return false;
    }
    std::string jsonText(jsonData.begin(), jsonData.end());
    if (!parseShapeFromJson(jsonText, shape)) {
        std::cerr << "Failed to parse shape from " << dir << "/" << name << ".json\n";
        return false;
    }
    std::vector<char> binData;
    if (!readWholeFile(dir + "/" + name + ".bin", binData)) {
        return false;
    }
    if (binData.size() % sizeof(float) != 0) {
        std::cerr << "Unexpected byte size for " << dir << "/" << name << ".bin\n";
        return false;
    }
    values.resize(binData.size() / sizeof(float));
    if (!values.empty()) {
        ::memcpy(values.data(), binData.data(), binData.size());
    }
    return true;
}

static bool runModuleForward(const std::shared_ptr<Module>& module,
                             const std::vector<float>& qkv,
                             const std::vector<float>& gate,
                             const std::vector<float>& beta,
                             const std::vector<float>& convWeight,
                             int batch,
                             int convDim,
                             int seqLen,
                             int numHeads,
                             int kernelSize,
                             std::vector<float>& output) {
    auto qkvVar = makeInputVar({batch, convDim, seqLen}, qkv);
    auto gateVar = makeInputVar({batch, seqLen, numHeads}, gate);
    auto betaVar = makeInputVar({batch, seqLen, numHeads}, beta);
    auto convVar = makeInputVar({convDim, 1, kernelSize}, convWeight);
    auto outputs = module->onForward({qkvVar, gateVar, betaVar, convVar});
    if (outputs.empty() || outputs[0] == nullptr || outputs[0]->getInfo() == nullptr) {
        return false;
    }
    const auto outSize = outputs[0]->getInfo()->size;
    output.resize(outSize);
    const float* ptr = outputs[0]->readMap<float>();
    if (ptr == nullptr) {
        return false;
    }
    ::memcpy(output.data(), ptr, outSize * sizeof(float));
    return true;
}

static bool runLoadedInputCompare(const ProbeConfig& cfg) {
    std::vector<int> qkvShape;
    std::vector<int> gateShape;
    std::vector<int> betaShape;
    std::vector<int> convShape;
    std::vector<float> qkv;
    std::vector<float> gate;
    std::vector<float> beta;
    std::vector<float> convWeight;
    std::vector<float> refOutput;
    std::vector<int> refShape;
    if (!loadTensorDirEntry(cfg.inputDir, "qkv", qkvShape, qkv) ||
        !loadTensorDirEntry(cfg.inputDir, "gate", gateShape, gate) ||
        !loadTensorDirEntry(cfg.inputDir, "beta", betaShape, beta) ||
        !loadTensorDirEntry(cfg.inputDir, "conv_weight", convShape, convWeight)) {
        return false;
    }
    loadTensorDirEntry(cfg.inputDir, "output_ref", refShape, refOutput);
    if (qkvShape.size() != 3 || gateShape.size() != 3 || betaShape.size() != 3 || convShape.size() != 3) {
        std::cerr << "Unexpected input rank in " << cfg.inputDir << "\n";
        return false;
    }
    const int batch = qkvShape[0];
    const int convDim = qkvShape[1];
    const int seqLen = qkvShape[2];
    const int kernelSize = convShape[2];
    auto runOne = [&](const std::string& name, MNNForwardType type, std::vector<float>& output) -> bool {
        auto module = makeLinearAttentionModule(type, batch, seqLen, kernelSize, convDim,
                                                cfg.numKHeads, cfg.numVHeads, cfg.headKDim, cfg.headVDim, cfg.useL2Norm);
        if (module == nullptr) {
            std::cerr << "Failed to create " << name << " module.\n";
            return false;
        }
        if (!runModuleForward(module, qkv, gate, beta, convWeight, batch, convDim, seqLen, cfg.numVHeads, kernelSize, output)) {
            std::cerr << name << " forward failed.\n";
            return false;
        }
        const std::vector<int> outputShape = {batch, seqLen, cfg.numVHeads, cfg.headVDim};
        std::cout << "[" << name << "] output_shape=[";
        for (int i = 0; i < outputShape.size(); ++i) {
            if (i > 0) {
                std::cout << ",";
            }
            std::cout << outputShape[i];
        }
        std::cout << "]\n";
        if (!cfg.dumpPrefix.empty()) {
            dumpTensor(cfg.dumpPrefix + "_" + name, outputShape, output);
        }
        if (!refOutput.empty()) {
            if (refOutput.size() != output.size()) {
                std::cerr << "Reference output size mismatch for " << name << ": ref=" << refOutput.size()
                          << " got=" << output.size() << "\n";
                return false;
            }
            printCompare(name + "_vs_ref", compareFloatBuffer(refOutput.data(), output.data(), output.size()));
        }
        return true;
    };

    std::vector<float> cpuOutput;
    std::vector<float> qnnOutput;
    if (cfg.backend == "cpu") {
        return runOne("cpu", MNN_FORWARD_CPU, cpuOutput);
    }
    if (cfg.backend == "npu") {
        return runOne("qnn", MNN_FORWARD_NN, qnnOutput);
    }
    if (!runOne("cpu", MNN_FORWARD_CPU, cpuOutput) || !runOne("qnn", MNN_FORWARD_NN, qnnOutput)) {
        return false;
    }
    if (cpuOutput.size() != qnnOutput.size()) {
        std::cerr << "CPU/QNN output size mismatch.\n";
        return false;
    }
    printCompare("cpu_vs_qnn", compareFloatBuffer(cpuOutput.data(), qnnOutput.data(), cpuOutput.size()));
    return true;
}

static bool runPrefillCompare(const ProbeConfig& cfg) {
    const int keyDim = cfg.numKHeads * cfg.headKDim;
    const int valueDim = cfg.numVHeads * cfg.headVDim;
    const int convDim = 2 * keyDim + valueDim;

    auto cpuModule = makeLinearAttentionModule(MNN_FORWARD_CPU, cfg.batch, cfg.prefillSeqLen, cfg.kernelSize, convDim,
                                               cfg.numKHeads, cfg.numVHeads, cfg.headKDim, cfg.headVDim, cfg.useL2Norm);
    auto qnnModule = makeLinearAttentionModule(MNN_FORWARD_NN, cfg.batch, cfg.prefillSeqLen, cfg.kernelSize, convDim,
                                               cfg.numKHeads, cfg.numVHeads, cfg.headKDim, cfg.headVDim, cfg.useL2Norm);
    if (cpuModule == nullptr || qnnModule == nullptr) {
        std::cerr << "Failed to create CPU/QNN LinearAttention modules.\n";
        return false;
    }

    std::vector<float> qkv(cfg.batch * convDim * cfg.prefillSeqLen);
    std::vector<float> gate(cfg.batch * cfg.prefillSeqLen * cfg.numVHeads);
    std::vector<float> beta(cfg.batch * cfg.prefillSeqLen * cfg.numVHeads);
    std::vector<float> convWeight(convDim * cfg.kernelSize);
    fillDeterministic(qkv, cfg.qkvScale, 0.0f);
    fillGate(gate, cfg.gateBase);
    fillBeta(beta, cfg.betaBase);
    fillConvWeight(convWeight);

    std::vector<float> cpuOutput;
    std::vector<float> qnnOutput;
    if (!runModuleForward(cpuModule, qkv, gate, beta, convWeight, cfg.batch, convDim, cfg.prefillSeqLen,
                          cfg.numVHeads, cfg.kernelSize, cpuOutput)) {
        std::cerr << "CPU prefill forward failed.\n";
        return false;
    }
    if (!runModuleForward(qnnModule, qkv, gate, beta, convWeight, cfg.batch, convDim, cfg.prefillSeqLen,
                          cfg.numVHeads, cfg.kernelSize, qnnOutput)) {
        std::cerr << "QNN prefill forward failed.\n";
        return false;
    }
    std::cout << "[prefill] cpu_size=" << cpuOutput.size() << " qnn_size=" << qnnOutput.size() << "\n";
    if (cpuOutput.size() != qnnOutput.size()) {
        std::cerr << "Prefill output size mismatch.\n";
        return false;
    }
    printCompare("prefill_cpu_vs_qnn", compareFloatBuffer(cpuOutput.data(), qnnOutput.data(), cpuOutput.size()));
    if (!cfg.dumpPrefix.empty()) {
        dumpTensor(cfg.dumpPrefix + "_prefill_cpu", {cfg.batch, cfg.prefillSeqLen, cfg.numVHeads, cfg.headVDim}, cpuOutput);
        dumpTensor(cfg.dumpPrefix + "_prefill_qnn", {cfg.batch, cfg.prefillSeqLen, cfg.numVHeads, cfg.headVDim}, qnnOutput);
    }
    return true;
}

static bool runDecodeCompare(const ProbeConfig& cfg) {
    const int keyDim = cfg.numKHeads * cfg.headKDim;
    const int valueDim = cfg.numVHeads * cfg.headVDim;
    const int convDim = 2 * keyDim + valueDim;

    auto cpuModule = makeLinearAttentionModule(MNN_FORWARD_CPU, cfg.batch, 1, cfg.kernelSize, convDim,
                                               cfg.numKHeads, cfg.numVHeads, cfg.headKDim, cfg.headVDim, cfg.useL2Norm);
    auto qnnModule = makeLinearAttentionModule(MNN_FORWARD_NN, cfg.batch, 1, cfg.kernelSize, convDim,
                                               cfg.numKHeads, cfg.numVHeads, cfg.headKDim, cfg.headVDim, cfg.useL2Norm);
    if (cpuModule == nullptr || qnnModule == nullptr) {
        std::cerr << "Failed to create CPU/QNN LinearAttention modules for decode.\n";
        return false;
    }

    std::vector<float> convWeight(convDim * cfg.kernelSize);
    fillConvWeight(convWeight);

    for (int step = 0; step < cfg.decodeSteps; ++step) {
        std::vector<float> qkv(cfg.batch * convDim);
        std::vector<float> gate(cfg.batch * cfg.numVHeads);
        std::vector<float> beta(cfg.batch * cfg.numVHeads);
        fillDeterministic(qkv, cfg.qkvScale, cfg.decodeOffset * static_cast<float>(step + 1));
        fillGate(gate, cfg.gateBase - 0.01f * static_cast<float>(step));
        fillBeta(beta, cfg.betaBase + 0.01f * static_cast<float>(step));

        std::vector<float> cpuOutput;
        std::vector<float> qnnOutput;
        if (!runModuleForward(cpuModule, qkv, gate, beta, convWeight, cfg.batch, convDim, 1,
                              cfg.numVHeads, cfg.kernelSize, cpuOutput)) {
            std::cerr << "CPU decode forward failed at step " << step << ".\n";
            return false;
        }
        if (!runModuleForward(qnnModule, qkv, gate, beta, convWeight, cfg.batch, convDim, 1,
                              cfg.numVHeads, cfg.kernelSize, qnnOutput)) {
            std::cerr << "QNN decode forward failed at step " << step << ".\n";
            return false;
        }
        if (cpuOutput.size() != qnnOutput.size()) {
            std::cerr << "Decode output size mismatch at step " << step << ".\n";
            return false;
        }
        printCompare("decode_step_" + std::to_string(step), compareFloatBuffer(cpuOutput.data(), qnnOutput.data(), cpuOutput.size()));
        if (!cfg.dumpPrefix.empty()) {
            dumpTensor(cfg.dumpPrefix + "_decode" + std::to_string(step) + "_cpu", {cfg.batch, 1, cfg.numVHeads, cfg.headVDim}, cpuOutput);
            dumpTensor(cfg.dumpPrefix + "_decode" + std::to_string(step) + "_qnn", {cfg.batch, 1, cfg.numVHeads, cfg.headVDim}, qnnOutput);
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    ProbeConfig cfg;
    if (!parseArgs(argc, argv, cfg)) {
        return argc > 1 ? 1 : 0;
    }
    std::cout << "Compare LinearAttention with"
              << " B=" << cfg.batch
              << " kvHeads=" << cfg.numKHeads
              << " vHeads=" << cfg.numVHeads
              << " headKDim=" << cfg.headKDim
              << " headVDim=" << cfg.headVDim
              << " kernel=" << cfg.kernelSize
              << " prefillSeq=" << cfg.prefillSeqLen
              << " decodeSteps=" << cfg.decodeSteps
              << " useL2Norm=" << (cfg.useL2Norm ? 1 : 0)
              << "\n";

    if (!cfg.inputDir.empty()) {
        return runLoadedInputCompare(cfg) ? 0 : 4;
    }

    if (!runPrefillCompare(cfg)) {
        return 2;
    }
    if (!runDecodeCompare(cfg)) {
        return 3;
    }
    return 0;
}
