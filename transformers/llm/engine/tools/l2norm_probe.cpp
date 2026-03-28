#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/Module.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

using namespace MNN;
using namespace MNN::Express;

namespace {

struct Config {
    std::string backend = "both";
    std::string inputFile;
    std::string dumpPrefix;
    std::vector<int> shape{1, 1, 16, 128};
    float eps = 1e-6f;
};

struct CompareResult {
    double maxAbs = 0.0;
    double meanAbs = 0.0;
    double rmse = 0.0;
    double cosine = 0.0;
    size_t nanCount = 0;
    size_t infCount = 0;
};

static bool parseInt(const char* text, int& value) {
    char* end = nullptr;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

static bool parseFloat(const char* text, float& value) {
    char* end = nullptr;
    float parsed = strtof(text, &end);
    if (end == text || *end != '\0') {
        return false;
    }
    value = parsed;
    return true;
}

static void printUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --input FILE [options]\n";
    std::cerr << "Options:\n";
    std::cerr << "  --backend cpu|npu|both\n";
    std::cerr << "  --shape N0 N1 N2 N3\n";
    std::cerr << "  --eps FLOAT\n";
    std::cerr << "  --dump-prefix PATH\n";
}

static bool parseArgs(int argc, char** argv, Config& cfg) {
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
            if (!value) return false;
            cfg.backend = value;
        } else if (arg == "--input") {
            auto value = needValue("--input");
            if (!value) return false;
            cfg.inputFile = value;
        } else if (arg == "--eps") {
            auto value = needValue("--eps");
            if (!value || !parseFloat(value, cfg.eps)) return false;
        } else if (arg == "--dump-prefix") {
            auto value = needValue("--dump-prefix");
            if (!value) return false;
            cfg.dumpPrefix = value;
        } else if (arg == "--shape") {
            cfg.shape.resize(4);
            for (int k = 0; k < 4; ++k) {
                auto value = needValue("--shape");
                if (!value || !parseInt(value, cfg.shape[k])) return false;
            }
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            return false;
        }
    }
    return !cfg.inputFile.empty();
}

static bool readFile(const std::string& path, std::vector<float>& data) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        std::cerr << "Failed to open " << path << "\n";
        return false;
    }
    auto size = ifs.tellg();
    if (size < 0 || (size % sizeof(float)) != 0) {
        std::cerr << "Unexpected file size for " << path << "\n";
        return false;
    }
    data.resize(static_cast<size_t>(size) / sizeof(float));
    ifs.seekg(0, std::ios::beg);
    if (!data.empty()) {
        ifs.read(reinterpret_cast<char*>(data.data()), size);
    }
    return !!ifs;
}

static void ensureParentDir(const std::string& path) {
    auto pos = path.rfind('/');
    if (pos == std::string::npos) {
        return;
    }
    std::string dir = path.substr(0, pos);
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
    bin.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    std::ofstream json(prefix + ".json");
    json << "{\n  \"dtype\": \"float32\",\n  \"shape\": [";
    for (int i = 0; i < shape.size(); ++i) {
        if (i > 0) json << ", ";
        json << shape[i];
    }
    json << "],\n  \"elements\": " << values.size() << "\n}\n";
}

static std::shared_ptr<Module> makeModule(MNNForwardType backend, const std::vector<int>& shape, float eps) {
    auto x = _Input(shape, NCHW, halide_type_of<float>());
    auto square = _Square(x);
    auto sum = _ReduceSum(square, {3}, true);
    auto plus = _Add(sum, _Scalar<float>(eps));
    auto denom = _Sqrt(plus);
    auto y = _Divide(x, denom);
    auto buffer = Variable::save({y});

    ScheduleConfig config;
    BackendConfig backendConfig;
    config.type = backend;
    config.numThread = 1;
    backendConfig.memory = BackendConfig::Memory_Low;
    backendConfig.power = BackendConfig::Power_Low;
    backendConfig.precision = BackendConfig::Precision_Low;
    config.backendConfig = &backendConfig;

    auto runtime = std::shared_ptr<Executor::RuntimeManager>(Executor::RuntimeManager::createRuntimeManager(config));
    if (runtime == nullptr) {
        return nullptr;
    }
    if (backend == MNN_FORWARD_NN) {
        runtime->setHint(MNN::Interpreter::INIT_THREAD_NUMBER, 1);
        runtime->setCache(".");
    }
    return std::shared_ptr<Module>(Module::load({"Input1"}, {}, reinterpret_cast<uint8_t*>(buffer.data()), buffer.size(), runtime));
}

static bool runModule(const std::shared_ptr<Module>& module,
                      const std::vector<int>& shape,
                      const std::vector<float>& input,
                      std::vector<float>& output) {
    auto x = _Input(shape, NCHW, halide_type_of<float>());
    ::memcpy(x->writeMap<float>(), input.data(), input.size() * sizeof(float));
    auto res = module->onForward({x});
    if (res.empty() || res[0] == nullptr || res[0]->getInfo() == nullptr) {
        return false;
    }
    auto size = res[0]->getInfo()->size;
    output.resize(size);
    ::memcpy(output.data(), res[0]->readMap<float>(), size * sizeof(float));
    return true;
}

static CompareResult compare(const std::vector<float>& a, const std::vector<float>& b) {
    CompareResult r;
    double sumAbs = 0.0, sumSq = 0.0, dot = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double x = a[i], y = b[i];
        if (std::isnan(x) || std::isnan(y)) r.nanCount++;
        if (std::isinf(x) || std::isinf(y)) r.infCount++;
        double d = std::abs(x - y);
        r.maxAbs = std::max(r.maxAbs, d);
        sumAbs += d;
        sumSq += d * d;
        dot += x * y;
        normA += x * x;
        normB += y * y;
    }
    r.meanAbs = sumAbs / a.size();
    r.rmse = std::sqrt(sumSq / a.size());
    double denom = std::sqrt(normA) * std::sqrt(normB);
    r.cosine = denom > 0.0 ? dot / denom : 0.0;
    return r;
}

static void printStats(const char* tag, const std::vector<float>& data) {
    double minV = data[0], maxV = data[0], maxAbs = std::fabs(data[0]), sumAbs = 0.0;
    for (float v : data) {
        minV = std::min(minV, (double)v);
        maxV = std::max(maxV, (double)v);
        maxAbs = std::max(maxAbs, std::fabs((double)v));
        sumAbs += std::fabs((double)v);
    }
    std::cout << "[" << tag << "] min=" << minV << " max=" << maxV
              << " meanAbs=" << (sumAbs / data.size()) << " maxAbs=" << maxAbs << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    if (!parseArgs(argc, argv, cfg)) {
        return argc > 1 ? 1 : 0;
    }
    std::vector<float> input;
    if (!readFile(cfg.inputFile, input)) {
        return 2;
    }
    int elementCount = 1;
    for (int dim : cfg.shape) elementCount *= dim;
    if ((int)input.size() != elementCount) {
        std::cerr << "Input size mismatch, expect " << elementCount << " got " << input.size() << "\n";
        return 3;
    }
    auto cpu = makeModule(MNN_FORWARD_CPU, cfg.shape, cfg.eps);
    auto npu = makeModule(MNN_FORWARD_NN, cfg.shape, cfg.eps);
    if (cpu == nullptr || npu == nullptr) {
        std::cerr << "Failed to create module\n";
        return 4;
    }
    std::vector<float> cpuOut, npuOut;
    if (cfg.backend == "cpu" || cfg.backend == "both") {
        if (!runModule(cpu, cfg.shape, input, cpuOut)) {
            std::cerr << "CPU run failed\n";
            return 5;
        }
        printStats("cpu", cpuOut);
        if (!cfg.dumpPrefix.empty()) dumpTensor(cfg.dumpPrefix + "_cpu", cfg.shape, cpuOut);
    }
    if (cfg.backend == "npu" || cfg.backend == "both") {
        if (!runModule(npu, cfg.shape, input, npuOut)) {
            std::cerr << "NPU run failed\n";
            return 6;
        }
        printStats("npu", npuOut);
        if (!cfg.dumpPrefix.empty()) dumpTensor(cfg.dumpPrefix + "_npu", cfg.shape, npuOut);
    }
    if (cfg.backend == "both") {
        auto r = compare(cpuOut, npuOut);
        std::cout << "[cpu_vs_npu] maxAbs=" << std::fixed << std::setprecision(6) << r.maxAbs
                  << " meanAbs=" << r.meanAbs << " rmse=" << r.rmse << " cosine=" << r.cosine
                  << " nan=" << r.nanCount << " inf=" << r.infCount << "\n";
    }
    return 0;
}
