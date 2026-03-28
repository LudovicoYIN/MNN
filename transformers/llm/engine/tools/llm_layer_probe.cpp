#include "llm/llm.hpp"
#include "llmconfig.hpp"
#include "kvmeta.hpp"

#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Transformer;

namespace {

struct FloatCompareResult {
    size_t size = 0;
    double maxAbs = 0.0;
    double meanAbs = 0.0;
    double rmse = 0.0;
    double cosine = 0.0;
    size_t nanCount = 0;
    size_t infCount = 0;
    size_t zeroCount = 0;
};

using LlmPtr = std::unique_ptr<Llm, void (*)(Llm*)>;

static MNNForwardType backendTypeConvert(const std::string& typeStr) {
    if (typeStr == "cpu") {
        return MNN_FORWARD_CPU;
    }
    if (typeStr == "metal") {
        return MNN_FORWARD_METAL;
    }
    if (typeStr == "cuda") {
        return MNN_FORWARD_CUDA;
    }
    if (typeStr == "opencl") {
        return MNN_FORWARD_OPENCL;
    }
    if (typeStr == "opengl") {
        return MNN_FORWARD_OPENGL;
    }
    if (typeStr == "vulkan") {
        return MNN_FORWARD_VULKAN;
    }
    if (typeStr == "npu") {
        return MNN_FORWARD_NN;
    }
    return MNN_FORWARD_AUTO;
}

static void setRuntimeHint(const std::shared_ptr<LlmConfig>& cfg,
                           const std::shared_ptr<KVMeta>& meta,
                           std::shared_ptr<Executor::RuntimeManager>& rtg) {
    rtg->setHint(MNN::Interpreter::INIT_THREAD_NUMBER, 4);
    rtg->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);

    int legacyAttentionMode = cfg->config_.value("quant_qkv", 8);
    int attentionMode = cfg->config_.value("attention_mode", legacyAttentionMode);
    rtg->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode);
    if (cfg->reuse_kv() && attentionMode == 10) {
        rtg->setHint(MNN::Interpreter::ATTENTION_OPTION, 9);
    }
    if (cfg->use_cached_mmap()) {
        rtg->setHint(MNN::Interpreter::USE_CACHED_MMAP, 1);
    }
    std::string tmpPath = cfg->tmp_path();
    if (cfg->kvcache_mmap()) {
        rtg->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_PATH_KVCACHE_DIR);
    }
    rtg->setExternalPath(cfg->prefix_cache_path(), MNN::Interpreter::EXTERNAL_PATH_PREFIXCACHE_DIR);
    if (cfg->use_mmap()) {
        rtg->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
    }
    rtg->setExternalPath(cfg->npu_model_dir(), MNN::Interpreter::EXTERNAL_NPU_FILE_DIR);
    rtg->setHint(MNN::Interpreter::DYNAMIC_QUANT_OPTIONS, cfg->dynamic_option());
    rtg->setHintPtr(MNN::Interpreter::KVCACHE_INFO, meta.get());
    if (backendTypeConvert(cfg->backend_type()) != MNN_FORWARD_CPU) {
        std::string cacheFilePath = tmpPath.empty() ? "." : tmpPath;
        rtg->setCache(cacheFilePath + "/mnn_cachefile.bin");
    }
    rtg->setHint(MNN::Interpreter::CPU_SME2_NEON_DIVISION_RATIO, cfg->config_.value("cpu_sme2_neon_division_ratio", 41));
    rtg->setHint(MNN::Interpreter::CPU_SME_CORES, cfg->config_.value("cpu_sme_core_num", 2));
}

static std::shared_ptr<Executor::RuntimeManager> createRuntime(const std::shared_ptr<LlmConfig>& cfg,
                                                               const std::shared_ptr<KVMeta>& meta) {
    ScheduleConfig scheduleConfig;
    BackendConfig backendConfig;
    scheduleConfig.type = backendTypeConvert(cfg->backend_type());
    scheduleConfig.numThread = cfg->thread_num();
    if (scheduleConfig.type == MNN_FORWARD_OPENCL) {
        scheduleConfig.numThread |= 64;
    }
    if (cfg->power() == "high") {
        backendConfig.power = BackendConfig::Power_High;
    } else if (cfg->power() == "low") {
        backendConfig.power = BackendConfig::Power_Low;
    }
    if (cfg->memory() == "high") {
        backendConfig.memory = BackendConfig::Memory_High;
    } else if (cfg->memory() == "low") {
        backendConfig.memory = BackendConfig::Memory_Low;
    }
    if (cfg->precision() == "high") {
        backendConfig.precision = BackendConfig::Precision_High;
    } else if (cfg->precision() == "low") {
        backendConfig.precision = BackendConfig::Precision_Low;
    }
    scheduleConfig.backendConfig = &backendConfig;

    auto rtg = std::shared_ptr<Executor::RuntimeManager>(Executor::RuntimeManager::createRuntimeManager(scheduleConfig));
    setRuntimeHint(cfg, meta, rtg);
    return rtg;
}

static LlmPtr makeLlm(const std::string& configPath) {
    return LlmPtr(Llm::createLLM(configPath), Llm::destroy);
}

static std::string shapeString(VARP var) {
    if (var == nullptr || var->getInfo() == nullptr) {
        return "<null>";
    }
    std::ostringstream os;
    os << "[";
    const auto& dims = var->getInfo()->dim;
    for (int i = 0; i < dims.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << dims[i];
    }
    os << "]";
    return os.str();
}

static std::string escapePiece(const std::string& piece) {
    std::ostringstream os;
    for (unsigned char ch : piece) {
        if (ch == '\n') {
            os << "\\n";
        } else if (ch == '\r') {
            os << "\\r";
        } else if (ch == '\t') {
            os << "\\t";
        } else if (std::iscntrl(ch)) {
            os << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(ch) << std::dec;
        } else {
            os << static_cast<char>(ch);
        }
    }
    return os.str();
}

static std::vector<std::string> parseCsv(const std::string& spec) {
    std::vector<std::string> result;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

static std::string sanitizeName(const std::string& name) {
    std::string out;
    out.reserve(name.size());
    for (char ch : name) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') ||
            ch == '_' || ch == '-' || ch == '.') {
            out.push_back(ch);
        } else {
            out.push_back('_');
        }
    }
    return out;
}

static std::string dataTypeString(halide_type_t type) {
    if (type == halide_type_of<float>()) {
        return "float32";
    }
    if (type == halide_type_of<int>()) {
        return "int32";
    }
    if (type == halide_type_of<uint8_t>()) {
        return "uint8";
    }
    std::ostringstream os;
    os << "code" << static_cast<int>(type.code)
       << "_bits" << static_cast<int>(type.bits)
       << "_lanes" << static_cast<int>(type.lanes);
    return os.str();
}

static bool saveTensorFile(const std::string& rootDir,
                           const std::string& prefix,
                           const std::string& name,
                           VARP var) {
    if (rootDir.empty() || var == nullptr || var->getInfo() == nullptr) {
        return false;
    }
    auto* info = var->getInfo();
    std::string current;
    for (size_t i = 0; i < rootDir.size(); ++i) {
        current.push_back(rootDir[i]);
        if (rootDir[i] == '/' || i + 1 == rootDir.size()) {
            if (!current.empty() && current != "/" && ::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                std::cerr << "Failed to create dir " << current << "\n";
                return false;
            }
        }
    }
    const std::string baseName = sanitizeName(prefix + "__" + name);
    const std::string binPath = rootDir + "/" + baseName + ".bin";
    const std::string jsonPath = rootDir + "/" + baseName + ".json";

    std::ofstream bin(binPath, std::ios::binary);
    if (!bin) {
        std::cerr << "Failed to open " << binPath << " for write\n";
        return false;
    }
    size_t bytes = 0;
    if (info->type == halide_type_of<float>()) {
        bytes = static_cast<size_t>(info->size) * sizeof(float);
        bin.write(reinterpret_cast<const char*>(var->readMap<float>()), bytes);
    } else if (info->type == halide_type_of<int>()) {
        bytes = static_cast<size_t>(info->size) * sizeof(int);
        bin.write(reinterpret_cast<const char*>(var->readMap<int>()), bytes);
    } else {
        std::cerr << "Skip dump for unsupported dtype " << dataTypeString(info->type)
                  << " tensor=" << name << "\n";
        return false;
    }
    bin.close();

    std::ofstream json(jsonPath);
    if (!json) {
        std::cerr << "Failed to open " << jsonPath << " for write\n";
        return false;
    }
    json << "{\n";
    json << "  \"name\": \"" << name << "\",\n";
    json << "  \"file\": \"" << baseName << ".bin\",\n";
    json << "  \"dtype\": \"" << dataTypeString(info->type) << "\",\n";
    json << "  \"shape\": [";
    for (int i = 0; i < info->dim.size(); ++i) {
        if (i > 0) {
            json << ", ";
        }
        json << info->dim[i];
    }
    json << "],\n";
    json << "  \"elements\": " << info->size << ",\n";
    json << "  \"bytes\": " << bytes << "\n";
    json << "}\n";
    return true;
}

template <typename T>
static VARP cloneInputVar(VARP src) {
    auto* info = src->getInfo();
    auto dst = _Input(info->dim, NCHW, halide_type_of<T>());
    ::memcpy(dst->template writeMap<T>(), src->template readMap<T>(), info->size * sizeof(T));
    return dst;
}

static VARP cloneFloatVar(VARP src) {
    return cloneInputVar<float>(src);
}

static VARP cloneIntVar(VARP src) {
    return cloneInputVar<int>(src);
}

static FloatCompareResult compareFloatData(const float* ref, const float* cand, size_t size) {
    FloatCompareResult result;
    result.size = size;
    if (ref == nullptr || cand == nullptr || size == 0) {
        return result;
    }
    double sumAbs = 0.0;
    double sumSq = 0.0;
    double dot = 0.0;
    double refNorm = 0.0;
    double candNorm = 0.0;
    for (size_t i = 0; i < size; ++i) {
        const double rv = ref[i];
        const double cv = cand[i];
        if (std::isnan(cv)) {
            ++result.nanCount;
        }
        if (std::isinf(cv)) {
            ++result.infCount;
        }
        if (cv == 0.0f) {
            ++result.zeroCount;
        }
        const double diff = rv - cv;
        const double absDiff = std::fabs(diff);
        result.maxAbs = std::max(result.maxAbs, absDiff);
        sumAbs += absDiff;
        sumSq += diff * diff;
        dot += rv * cv;
        refNorm += rv * rv;
        candNorm += cv * cv;
    }
    result.meanAbs = sumAbs / static_cast<double>(size);
    result.rmse = std::sqrt(sumSq / static_cast<double>(size));
    if (refNorm > 0.0 && candNorm > 0.0) {
        result.cosine = dot / std::sqrt(refNorm * candNorm);
    }
    return result;
}

static void printFloatCompare(const std::string& label, VARP refVar, VARP candVar) {
    if (refVar == nullptr || candVar == nullptr) {
        std::cout << "[" << label << "] null tensor ref=" << (refVar.get() != nullptr)
                  << " cand=" << (candVar.get() != nullptr) << "\n";
        return;
    }
    auto* refInfo = refVar->getInfo();
    auto* candInfo = candVar->getInfo();
    if (refInfo == nullptr || candInfo == nullptr) {
        std::cout << "[" << label << "] missing tensor info\n";
        return;
    }
    if (refInfo->size != candInfo->size) {
        std::cout << "[" << label << "] shape mismatch ref=" << shapeString(refVar)
                  << " cand=" << shapeString(candVar)
                  << " size_ref=" << refInfo->size
                  << " size_cand=" << candInfo->size << "\n";
        return;
    }
    auto result = compareFloatData(refVar->readMap<float>(), candVar->readMap<float>(), refInfo->size);
    std::cout << "[" << label << "] shape=" << shapeString(refVar)
              << " size=" << result.size
              << " maxAbs=" << std::fixed << std::setprecision(6) << result.maxAbs
              << " meanAbs=" << result.meanAbs
              << " rmse=" << result.rmse
              << " cosine=" << result.cosine
              << " cand_nan=" << result.nanCount
              << " cand_inf=" << result.infCount
              << " cand_zero=" << result.zeroCount << "\n";
}

static std::vector<int> topIndices(VARP var, int k) {
    std::vector<int> indices;
    if (var == nullptr || var->getInfo() == nullptr || var->getInfo()->size <= 0) {
        return indices;
    }
    const float* ptr = var->readMap<float>();
    const int size = var->getInfo()->size;
    std::vector<std::pair<float, int>> values;
    values.reserve(size);
    for (int i = 0; i < size; ++i) {
        values.push_back(std::make_pair(ptr[i], i));
    }
    std::partial_sort(values.begin(), values.begin() + std::min(k, size), values.end(),
                      [](const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
                          return lhs.first > rhs.first;
                      });
    for (int i = 0; i < std::min(k, size); ++i) {
        indices.push_back(values[i].second);
    }
    return indices;
}

static void printProbeSummary(const std::string& label, VARP var) {
    if (var == nullptr || var->getInfo() == nullptr) {
        std::cout << "[" << label << "] unavailable\n";
        return;
    }
    auto* info = var->getInfo();
    if (info->size <= 0) {
        std::cout << "[" << label << "] shape=" << shapeString(var) << " size=0\n";
        return;
    }
    const float* ptr = var->readMap<float>();
    double minValue = std::numeric_limits<double>::max();
    double maxValue = std::numeric_limits<double>::lowest();
    double sum = 0.0;
    for (int i = 0; i < info->size; ++i) {
        double v = ptr[i];
        minValue = std::min(minValue, v);
        maxValue = std::max(maxValue, v);
        sum += v;
    }
    std::cout << "[" << label << "] shape=" << shapeString(var)
              << " size=" << info->size
              << " min=" << std::fixed << std::setprecision(6) << minValue
              << " max=" << maxValue
              << " mean=" << (sum / static_cast<double>(info->size)) << "\n";
}

struct ProbeModule {
    std::shared_ptr<LlmConfig> config;
    std::shared_ptr<KVMeta> meta;
    std::shared_ptr<Executor::RuntimeManager> runtime;
    std::shared_ptr<Module> module;
};

static bool loadProbeModule(const std::string& configPath,
                            const std::vector<std::string>& outputNames,
                            ProbeModule& probe) {
    probe.config.reset(new LlmConfig(configPath));
    probe.meta.reset(new KVMeta);
    probe.runtime = createRuntime(probe.config, probe.meta);

    Module::Config moduleConfig;
    if (probe.config->backend_type() == "opencl" || probe.config->backend_type() == "vulkan" || probe.config->backend_type() == "npu") {
        moduleConfig.shapeMutable = false;
    } else {
        moduleConfig.shapeMutable = true;
    }
    moduleConfig.rearrange = true;

    std::vector<std::string> inputNames {"input_ids", "attention_mask", "position_ids", "logits_index"};
    if (probe.config->has_deepstack()) {
        inputNames.emplace_back("deepstack_embeds");
    }

    probe.runtime->setExternalFile(probe.config->llm_weight());
    probe.module.reset(Module::load(inputNames, outputNames, probe.config->llm_model().c_str(), probe.runtime, &moduleConfig), Module::destroy);
    probe.runtime->setExternalFile("");
    if (probe.module == nullptr) {
        std::cerr << "Failed to load probe module from " << probe.config->llm_model() << "\n";
        return false;
    }
    return true;
}

static std::vector<VARP> buildSharedInputs(Llm* llm, const std::vector<int>& ids, bool addDeepstack) {
    std::vector<VARP> inputs;
    auto embeds = llm->embedding(ids);
    auto mask = llm->gen_attention_mask(static_cast<int>(ids.size()));
    auto pos = llm->gen_position_ids(static_cast<int>(ids.size()));
    int logitsIndexValue = -1;
    auto logitsIndex = _Const((const void*)&logitsIndexValue, {1}, NHWC, halide_type_of<int>());
    inputs.push_back(cloneFloatVar(embeds));
    inputs.push_back(cloneFloatVar(mask));
    inputs.push_back(cloneIntVar(pos));
    inputs.push_back(logitsIndex);
    if (addDeepstack) {
        const std::vector<int> deepstackDims{3, 1, 1};
        auto shape = _Const((const void*)deepstackDims.data(), {3}, NHWC, halide_type_of<int>());
        inputs.push_back(_Fill(shape, _Scalar<float>(0.0f)));
    }
    return inputs;
}

static void printUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " <cpu_config.json> <qnn_config.json> <prompt> [probe_outputs_csv] [dump_dir]\n";
    std::cerr << "Default probe outputs:\n";
    std::cerr << "  /blocks.0/self_attn/fused_attn/FusedLinearAttention_output_0,\n";
    std::cerr << "  /blocks.0/self_attn/Reshape_2_output_0,\n";
    std::cerr << "  /blocks.0/self_attn/out_proj/FakeLinear_output_0,\n";
    std::cerr << "  /blocks.0/Add_output_0\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }

    std::vector<std::string> probeOutputs{
        "/blocks.0/self_attn/fused_attn/FusedLinearAttention_output_0",
        "/blocks.0/self_attn/Reshape_2_output_0",
        "/blocks.0/self_attn/out_proj/FakeLinear_output_0",
        "/blocks.0/Add_output_0"
    };
    if (argc >= 5) {
        probeOutputs = parseCsv(argv[4]);
    }
    const std::string dumpDir = argc >= 6 ? argv[5] : "";

    const std::string cpuConfig = argv[1];
    const std::string qnnConfig = argv[2];
    const std::string prompt = argv[3];

    LlmPtr cpuLlm = makeLlm(cpuConfig);
    LlmPtr qnnLlm = makeLlm(qnnConfig);
    if (!cpuLlm || !qnnLlm) {
        std::cerr << "Failed to create LLMs.\n";
        return 2;
    }
    if (!cpuLlm->load() || !qnnLlm->load()) {
        std::cerr << "Failed to load CPU/QNN LLM.\n";
        return 3;
    }

    auto cpuIds = cpuLlm->tokenizer_encode(prompt);
    auto qnnIds = qnnLlm->tokenizer_encode(prompt);
    std::cout << "cpu ids size=" << cpuIds.size() << "\n";
    std::cout << "qnn ids size=" << qnnIds.size() << "\n";
    if (cpuIds != qnnIds) {
        std::cout << "tokenizer mismatch, use CPU ids as canonical probe input.\n";
    }
    if (cpuIds.empty()) {
        std::cerr << "Prompt encoded to zero tokens.\n";
        return 4;
    }

    auto cpuEmb = cpuLlm->embedding(cpuIds);
    auto qnnEmb = qnnLlm->embedding(cpuIds);
    printFloatCompare("shared_input_embeds_check", cpuEmb, qnnEmb);

    auto cpuMask = cpuLlm->gen_attention_mask(static_cast<int>(cpuIds.size()));
    auto qnnMask = qnnLlm->gen_attention_mask(static_cast<int>(cpuIds.size()));
    printFloatCompare("shared_attention_mask_check", cpuMask, qnnMask);

    ProbeModule cpuProbe;
    ProbeModule qnnProbe;
    if (!loadProbeModule(cpuConfig, probeOutputs, cpuProbe)) {
        return 5;
    }
    if (!loadProbeModule(qnnConfig, probeOutputs, qnnProbe)) {
        return 6;
    }

    auto cpuInputs = buildSharedInputs(cpuLlm.get(), cpuIds, cpuProbe.config->has_deepstack());
    auto qnnInputs = buildSharedInputs(cpuLlm.get(), cpuIds, qnnProbe.config->has_deepstack());

    auto cpuOutputs = cpuProbe.module->onForward(cpuInputs);
    auto qnnOutputs = qnnProbe.module->onForward(qnnInputs);
    if (cpuOutputs.size() != probeOutputs.size()) {
        std::cerr << "CPU probe output size mismatch: expected " << probeOutputs.size()
                  << " got " << cpuOutputs.size() << "\n";
        return 7;
    }
    if (qnnOutputs.size() != probeOutputs.size()) {
        std::cerr << "QNN probe output size mismatch: expected " << probeOutputs.size()
                  << " got " << qnnOutputs.size() << "\n";
        return 8;
    }

    for (size_t i = 0; i < probeOutputs.size(); ++i) {
        std::cout << "\n=== probe " << i << " ===\n";
        std::cout << probeOutputs[i] << "\n";
        printProbeSummary("cpu", cpuOutputs[i]);
        printProbeSummary("qnn", qnnOutputs[i]);
        printFloatCompare(probeOutputs[i], cpuOutputs[i], qnnOutputs[i]);
        if (!dumpDir.empty()) {
            saveTensorFile(dumpDir, "cpu", probeOutputs[i], cpuOutputs[i]);
            saveTensorFile(dumpDir, "qnn", probeOutputs[i], qnnOutputs[i]);
        }
    }

    return 0;
}
