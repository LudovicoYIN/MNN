#include <QNN/QnnGraph.h>
#include <QNN/QnnInterface.h>
#include <QNN/QnnTensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using QnnInterfaceGetProvidersFn_t =
    Qnn_ErrorHandle_t (*)(const QnnInterface_t*** providerList, uint32_t* numProviders);

struct Config {
    std::string backendPath;
    int batch = 1;
    int numKHeads = 16;
    int numVHeads = 16;
    int headKDim = 128;
    int headVDim = 128;
    int kernelSize = 4;
    int seqLen = 1;
    bool useQkL2Norm = false;
    bool fp16 = false;
    bool recurrentFp16 = false;
    bool verbose = false;
    std::string inputDir;
    std::string outputDir;
};

struct CompareStats {
    double maxAbs = 0.0;
    double meanAbs = 0.0;
    double rmse = 0.0;
    double cosine = 0.0;
    size_t nanCount = 0;
    size_t infCount = 0;
};

static size_t elementCount(const std::vector<uint32_t>& dims);

static void die(const std::string& message) {
    std::cerr << message << "\n";
    std::exit(1);
}

static bool parseInt(const char* text, int& out) {
    char* end = nullptr;
    long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

static Config parseArgs(int argc, char** argv) {
    Config cfg;
    auto qnnRoot = std::getenv("QNN_SDK_ROOT");
    if (qnnRoot) {
        cfg.backendPath = std::string(qnnRoot) + "/lib/x86_64-linux-clang/libQnnCpu.so";
    }
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto needValue = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                die(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--backend") {
            cfg.backendPath = needValue("--backend");
        } else if (arg == "--batch") {
            if (!parseInt(needValue("--batch"), cfg.batch)) die("Invalid --batch");
        } else if (arg == "--num-k-heads") {
            if (!parseInt(needValue("--num-k-heads"), cfg.numKHeads)) die("Invalid --num-k-heads");
        } else if (arg == "--num-v-heads") {
            if (!parseInt(needValue("--num-v-heads"), cfg.numVHeads)) die("Invalid --num-v-heads");
        } else if (arg == "--head-k-dim") {
            if (!parseInt(needValue("--head-k-dim"), cfg.headKDim)) die("Invalid --head-k-dim");
        } else if (arg == "--head-v-dim") {
            if (!parseInt(needValue("--head-v-dim"), cfg.headVDim)) die("Invalid --head-v-dim");
        } else if (arg == "--kernel-size") {
            if (!parseInt(needValue("--kernel-size"), cfg.kernelSize)) die("Invalid --kernel-size");
        } else if (arg == "--seq-len") {
            if (!parseInt(needValue("--seq-len"), cfg.seqLen)) die("Invalid --seq-len");
        } else if (arg == "--use-qk-l2norm") {
            cfg.useQkL2Norm = true;
        } else if (arg == "--fp16") {
            cfg.fp16 = true;
        } else if (arg == "--recurrent-fp16") {
            cfg.recurrentFp16 = true;
        } else if (arg == "--verbose") {
            cfg.verbose = true;
        } else if (arg == "--input-dir") {
            cfg.inputDir = needValue("--input-dir");
        } else if (arg == "--output-dir") {
            cfg.outputDir = needValue("--output-dir");
        } else {
            die("Unknown argument: " + arg);
        }
    }
    if (cfg.backendPath.empty()) {
        die("Missing --backend and QNN_SDK_ROOT is not set.");
    }
    return cfg;
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

static std::string basePath(const std::string& dir, const std::string& name) {
    return dir + "/" + sanitizeName(name);
}

static void writeTensorFiles(const std::string& dir,
                             const std::string& name,
                             const std::vector<uint32_t>& dims,
                             const std::vector<float>& values,
                             const char* dtype) {
    if (dir.empty()) {
        return;
    }
    std::filesystem::create_directories(dir);
    const std::string base = basePath(dir, name);
    std::ofstream bin(base + ".bin", std::ios::binary);
    if (!bin) {
        die("Failed to open output bin for " + name);
    }
    bin.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    bin.close();

    std::ofstream json(base + ".json");
    if (!json) {
        die("Failed to open output json for " + name);
    }
    json << "{\n";
    json << "  \"name\": \"" << name << "\",\n";
    json << "  \"file\": \"" << sanitizeName(name) << ".bin\",\n";
    json << "  \"dtype\": \"" << dtype << "\",\n";
    json << "  \"shape\": [";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) {
            json << ", ";
        }
        json << dims[i];
    }
    json << "],\n";
    json << "  \"elements\": " << values.size() << ",\n";
    json << "  \"bytes\": " << values.size() * sizeof(float) << "\n";
    json << "}\n";
}

static std::string loadTextFile(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        die("Failed to open " + path);
    }
    std::ostringstream os;
    os << input.rdbuf();
    return os.str();
}

static std::vector<uint32_t> parseShapeArray(const std::string& text) {
    auto key = text.find("\"shape\"");
    if (key == std::string::npos) {
        die("Missing shape in tensor metadata");
    }
    auto lb = text.find('[', key);
    auto rb = text.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos || rb < lb) {
        die("Malformed shape in tensor metadata");
    }
    std::vector<uint32_t> dims;
    std::stringstream ss(text.substr(lb + 1, rb - lb - 1));
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        uint32_t value = static_cast<uint32_t>(std::stoul(item));
        dims.push_back(value);
    }
    return dims;
}

static std::string parseDType(const std::string& text) {
    auto key = text.find("\"dtype\"");
    if (key == std::string::npos) {
        return "float32";
    }
    auto first = text.find('"', key + 7);
    if (first == std::string::npos) {
        die("Malformed dtype in tensor metadata");
    }
    auto second = text.find('"', first + 1);
    if (second == std::string::npos) {
        die("Malformed dtype in tensor metadata");
    }
    return text.substr(first + 1, second - first - 1);
}

static std::vector<float> readTensorFile(const std::string& dir,
                                         const std::string& name,
                                         const std::vector<uint32_t>& expectedDims,
                                         bool required) {
    const std::string base = basePath(dir, name);
    const std::string jsonPath = base + ".json";
    const std::string binPath = base + ".bin";
    if (!std::filesystem::exists(jsonPath) || !std::filesystem::exists(binPath)) {
        if (required) {
            die("Missing tensor files for " + name + " under " + dir);
        }
        return {};
    }
    const std::string meta = loadTextFile(jsonPath);
    const auto dims = parseShapeArray(meta);
    if (!expectedDims.empty() && dims != expectedDims) {
        std::ostringstream os;
        os << "Shape mismatch for " << name << ": expected [";
        for (size_t i = 0; i < expectedDims.size(); ++i) {
            if (i > 0) os << ",";
            os << expectedDims[i];
        }
        os << "] got [";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) os << ",";
            os << dims[i];
        }
        os << "]";
        die(os.str());
    }
    const std::string dtype = parseDType(meta);
    if (dtype != "float32") {
        die("Unsupported input dtype for " + name + ": " + dtype);
    }
    const size_t count = elementCount(dims);
    std::vector<float> values(count);
    std::ifstream bin(binPath, std::ios::binary);
    if (!bin) {
        die("Failed to open " + binPath);
    }
    bin.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (static_cast<size_t>(bin.gcount()) != count * sizeof(float)) {
        die("Unexpected byte count in " + binPath);
    }
    return values;
}

static uint16_t floatToHalfBits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000u) >> 13));
}

static float halfBitsToFloat(uint16_t value) {
    uint32_t sign = (static_cast<uint32_t>(value & 0x8000u)) << 16;
    uint32_t exp = (value >> 10) & 0x1fu;
    uint32_t mant = value & 0x3ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static void toBackendData(const std::vector<float>& src, bool fp16, std::vector<uint8_t>& dst) {
    if (fp16) {
        dst.resize(src.size() * sizeof(uint16_t));
        auto* ptr = reinterpret_cast<uint16_t*>(dst.data());
        for (size_t i = 0; i < src.size(); ++i) ptr[i] = floatToHalfBits(src[i]);
    } else {
        dst.resize(src.size() * sizeof(float));
        std::memcpy(dst.data(), src.data(), dst.size());
    }
}

static std::vector<float> fromBackendData(const std::vector<uint8_t>& src, bool fp16) {
    if (fp16) {
        std::vector<float> out(src.size() / sizeof(uint16_t));
        auto* ptr = reinterpret_cast<const uint16_t*>(src.data());
        for (size_t i = 0; i < out.size(); ++i) out[i] = halfBitsToFloat(ptr[i]);
        return out;
    }
    std::vector<float> out(src.size() / sizeof(float));
    std::memcpy(out.data(), src.data(), src.size());
    return out;
}

struct TensorHandle {
    std::string name;
    std::vector<uint32_t> dims;
    std::vector<uint8_t> storage;
    Qnn_Tensor_t tensor = QNN_TENSOR_INIT;

    TensorHandle() = default;
    TensorHandle(const TensorHandle& other) : name(other.name), dims(other.dims), storage(other.storage), tensor(other.tensor) {
        refreshPointers();
    }
    TensorHandle(TensorHandle&& other) noexcept
        : name(std::move(other.name)), dims(std::move(other.dims)), storage(std::move(other.storage)), tensor(other.tensor) {
        refreshPointers();
    }
    TensorHandle& operator=(const TensorHandle& other) {
        if (this != &other) {
            name = other.name;
            dims = other.dims;
            storage = other.storage;
            tensor = other.tensor;
            refreshPointers();
        }
        return *this;
    }
    TensorHandle& operator=(TensorHandle&& other) noexcept {
        if (this != &other) {
            name = std::move(other.name);
            dims = std::move(other.dims);
            storage = std::move(other.storage);
            tensor = other.tensor;
            refreshPointers();
        }
        return *this;
    }
    void refreshPointers() {
        tensor.v1.name = name.c_str();
        tensor.v1.dimensions = dims.data();
        if (!storage.empty()) {
            tensor.v1.clientBuf.data = storage.data();
            tensor.v1.clientBuf.dataSize = storage.size();
        }
    }
};

struct ParamTensorHandle {
    const char* paramName = nullptr;
    std::string tensorName;
    std::vector<uint32_t> dims;
    std::vector<uint8_t> storage;
    Qnn_Param_t param = QNN_PARAM_INIT;

    ParamTensorHandle() = default;
    ParamTensorHandle(const ParamTensorHandle& other)
        : paramName(other.paramName), tensorName(other.tensorName), dims(other.dims), storage(other.storage), param(other.param) {
        refreshPointers();
    }
    ParamTensorHandle(ParamTensorHandle&& other) noexcept
        : paramName(other.paramName), tensorName(std::move(other.tensorName)), dims(std::move(other.dims)),
          storage(std::move(other.storage)), param(other.param) {
        refreshPointers();
    }
    ParamTensorHandle& operator=(const ParamTensorHandle& other) {
        if (this != &other) {
            paramName = other.paramName;
            tensorName = other.tensorName;
            dims = other.dims;
            storage = other.storage;
            param = other.param;
            refreshPointers();
        }
        return *this;
    }
    ParamTensorHandle& operator=(ParamTensorHandle&& other) noexcept {
        if (this != &other) {
            paramName = other.paramName;
            tensorName = std::move(other.tensorName);
            dims = std::move(other.dims);
            storage = std::move(other.storage);
            param = other.param;
            refreshPointers();
        }
        return *this;
    }
    void refreshPointers() {
        param.name = paramName;
        param.tensorParam.v1.name = tensorName.c_str();
        param.tensorParam.v1.dimensions = dims.data();
        if (!storage.empty()) {
            param.tensorParam.v1.clientBuf.data = storage.data();
            param.tensorParam.v1.clientBuf.dataSize = storage.size();
        }
    }
};

static size_t elementCount(const std::vector<uint32_t>& dims) {
    size_t count = 1;
    for (auto d : dims) count *= static_cast<size_t>(d);
    return count;
}

static Qnn_DataType_t dataTypeOf(bool fp16) {
    return fp16 ? QNN_DATATYPE_FLOAT_16 : QNN_DATATYPE_FLOAT_32;
}

static TensorHandle makeAppTensor(const std::string& name,
                                  Qnn_TensorType_t type,
                                  Qnn_DataType_t dataType,
                                  const std::vector<uint32_t>& dims,
                                  const std::vector<float>* values,
                                  bool setRuntimeData) {
    TensorHandle h;
    h.name = name;
    h.dims = dims;
    h.tensor.version = QNN_TENSOR_VERSION_1;
    h.tensor.v1 = QNN_TENSOR_V1_INIT;
    h.tensor.v1.name = h.name.c_str();
    h.tensor.v1.type = type;
    h.tensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    h.tensor.v1.dataType = dataType;
    h.tensor.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
    h.tensor.v1.rank = static_cast<uint32_t>(h.dims.size());
    h.tensor.v1.dimensions = h.dims.data();
    h.tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    if (values && setRuntimeData) {
        toBackendData(*values, dataType == QNN_DATATYPE_FLOAT_16, h.storage);
        if (type == QNN_TENSOR_TYPE_STATIC) {
            h.tensor.v1.clientBuf.data = h.storage.data();
            h.tensor.v1.clientBuf.dataSize = h.storage.size();
        } else {
            h.tensor.v1.clientBuf.data = nullptr;
            h.tensor.v1.clientBuf.dataSize = 0;
        }
    } else {
        h.tensor.v1.clientBuf.data = nullptr;
        h.tensor.v1.clientBuf.dataSize = 0;
    }
    return h;
}

static TensorHandle makeStaticTensor(const std::string& name,
                                     Qnn_DataType_t dataType,
                                     const std::vector<uint32_t>& dims,
                                     const std::vector<float>& values) {
    TensorHandle h = makeAppTensor(name, QNN_TENSOR_TYPE_STATIC, dataType, dims, &values, true);
    return h;
}

static TensorHandle makeStaticInt32Tensor(const std::string& name,
                                          const std::vector<uint32_t>& dims,
                                          const std::vector<int32_t>& values) {
    TensorHandle h;
    h.name = name;
    h.dims = dims;
    h.storage.resize(values.size() * sizeof(int32_t));
    std::memcpy(h.storage.data(), values.data(), h.storage.size());
    h.tensor.version = QNN_TENSOR_VERSION_1;
    h.tensor.v1 = QNN_TENSOR_V1_INIT;
    h.tensor.v1.name = h.name.c_str();
    h.tensor.v1.type = QNN_TENSOR_TYPE_STATIC;
    h.tensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    h.tensor.v1.dataType = QNN_DATATYPE_INT_32;
    h.tensor.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
    h.tensor.v1.rank = static_cast<uint32_t>(h.dims.size());
    h.tensor.v1.dimensions = h.dims.data();
    h.tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    h.tensor.v1.clientBuf.data = h.storage.data();
    h.tensor.v1.clientBuf.dataSize = h.storage.size();
    return h;
}

static TensorHandle makeNativeTensor(const std::string& name,
                                     Qnn_DataType_t dataType,
                                     const std::vector<uint32_t>& dims) {
    return makeAppTensor(name, QNN_TENSOR_TYPE_NATIVE, dataType, dims, nullptr, false);
}

static ParamTensorHandle makeUInt32ParamTensor(const char* tensorName,
                                               const std::vector<uint32_t>& dims,
                                               const std::vector<uint32_t>& values,
                                               const char* paramName) {
    ParamTensorHandle h;
    h.paramName = paramName;
    h.tensorName = tensorName;
    h.dims = dims;
    h.storage.resize(values.size() * sizeof(uint32_t));
    std::memcpy(h.storage.data(), values.data(), h.storage.size());
    h.param = QNN_PARAM_INIT;
    h.param.paramType = QNN_PARAMTYPE_TENSOR;
    h.param.name = h.paramName;
    h.param.tensorParam.version = QNN_TENSOR_VERSION_1;
    h.param.tensorParam.v1 = QNN_TENSOR_V1_INIT;
    h.param.tensorParam.v1.name = h.tensorName.c_str();
    h.param.tensorParam.v1.type = QNN_TENSOR_TYPE_STATIC;
    h.param.tensorParam.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    h.param.tensorParam.v1.dataType = QNN_DATATYPE_UINT_32;
    h.param.tensorParam.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
    h.param.tensorParam.v1.rank = static_cast<uint32_t>(h.dims.size());
    h.param.tensorParam.v1.dimensions = h.dims.data();
    h.param.tensorParam.v1.memType = QNN_TENSORMEMTYPE_RAW;
    h.param.tensorParam.v1.clientBuf.data = h.storage.data();
    h.param.tensorParam.v1.clientBuf.dataSize = h.storage.size();
    return h;
}

static ParamTensorHandle makeInt32ParamTensor(const char* tensorName,
                                              const std::vector<uint32_t>& dims,
                                              const std::vector<int32_t>& values,
                                              const char* paramName) {
    ParamTensorHandle h;
    h.paramName = paramName;
    h.tensorName = tensorName;
    h.dims = dims;
    h.storage.resize(values.size() * sizeof(int32_t));
    std::memcpy(h.storage.data(), values.data(), h.storage.size());
    h.param = QNN_PARAM_INIT;
    h.param.paramType = QNN_PARAMTYPE_TENSOR;
    h.param.name = h.paramName;
    h.param.tensorParam.version = QNN_TENSOR_VERSION_1;
    h.param.tensorParam.v1 = QNN_TENSOR_V1_INIT;
    h.param.tensorParam.v1.name = h.tensorName.c_str();
    h.param.tensorParam.v1.type = QNN_TENSOR_TYPE_STATIC;
    h.param.tensorParam.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    h.param.tensorParam.v1.dataType = QNN_DATATYPE_INT_32;
    h.param.tensorParam.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
    h.param.tensorParam.v1.rank = static_cast<uint32_t>(h.dims.size());
    h.param.tensorParam.v1.dimensions = h.dims.data();
    h.param.tensorParam.v1.memType = QNN_TENSORMEMTYPE_RAW;
    h.param.tensorParam.v1.clientBuf.data = h.storage.data();
    h.param.tensorParam.v1.clientBuf.dataSize = h.storage.size();
    return h;
}

static Qnn_Param_t makeUInt32ScalarParam(const char* name, uint32_t value) {
    Qnn_Param_t out = QNN_PARAM_INIT;
    out.name = name;
    out.paramType = QNN_PARAMTYPE_SCALAR;
    out.scalarParam.dataType = QNN_DATATYPE_UINT_32;
    out.scalarParam.uint32Value = value;
    return out;
}

static Qnn_Param_t makeInt32ScalarParam(const char* name, int32_t value) {
    Qnn_Param_t out = QNN_PARAM_INIT;
    out.name = name;
    out.paramType = QNN_PARAMTYPE_SCALAR;
    out.scalarParam.dataType = QNN_DATATYPE_INT_32;
    out.scalarParam.int32Value = value;
    return out;
}

static Qnn_Param_t makeBoolScalarParam(const char* name, bool value) {
    Qnn_Param_t out = QNN_PARAM_INIT;
    out.name = name;
    out.paramType = QNN_PARAMTYPE_SCALAR;
    out.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
    out.scalarParam.bool8Value = value ? 1 : 0;
    return out;
}

static CompareStats compare(const std::vector<float>& ref, const std::vector<float>& got) {
    CompareStats s;
    if (ref.size() != got.size()) die("compare size mismatch");
    double sumAbs = 0.0, sumSq = 0.0, dot = 0.0, nr = 0.0, ng = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double a = ref[i], b = got[i];
        if (std::isnan(a) || std::isnan(b)) s.nanCount++;
        if (std::isinf(a) || std::isinf(b)) s.infCount++;
        double d = std::abs(a - b);
        s.maxAbs = std::max(s.maxAbs, d);
        sumAbs += d;
        sumSq += d * d;
        dot += a * b;
        nr += a * a;
        ng += b * b;
    }
    s.meanAbs = sumAbs / static_cast<double>(ref.size());
    s.rmse = std::sqrt(sumSq / static_cast<double>(ref.size()));
    double denom = std::sqrt(nr) * std::sqrt(ng);
    s.cosine = denom > 0.0 ? dot / denom : 0.0;
    return s;
}

static void fillDeterministic(std::vector<float>& data, float scale, float offset) {
    for (size_t i = 0; i < data.size(); ++i) data[i] = (static_cast<int>(i % 23) - 11) * scale + offset;
}

static void fillConvWeight(std::vector<float>& data) {
    for (size_t i = 0; i < data.size(); ++i) data[i] = (static_cast<int>(i % 9) - 4) * 0.02f;
}

static void cpuLinearReference(const Config& cfg,
                               const std::vector<float>& qkv,
                               const std::vector<float>& gate,
                               const std::vector<float>& beta,
                               const std::vector<float>& convWeight,
                               const std::vector<float>& convStateIn,
                               const std::vector<float>& recurrentIn,
                               std::vector<float>& output,
                               std::vector<float>& convStateOut,
                               std::vector<float>& recurrentOut) {
    const int keyDim = cfg.numKHeads * cfg.headKDim;
    const int valueDim = cfg.numVHeads * cfg.headVDim;
    const int D = 2 * keyDim + valueDim;
    const int state = cfg.kernelSize - 1;
    const int totalLen = state + cfg.seqLen;
    const int Hv = cfg.numVHeads;
    const int dk = cfg.headKDim;
    const int dv = cfg.headVDim;
    if (cfg.numKHeads != cfg.numVHeads) {
        die("cpuLinearReference currently requires num-k-heads == num-v-heads");
    }
    std::vector<float> rawConcat(cfg.batch * D * totalLen, 0.0f);
    for (int b = 0; b < cfg.batch; ++b) {
        for (int c = 0; c < D; ++c) {
            for (int l = 0; l < state; ++l) {
                rawConcat[(b * D + c) * totalLen + l] = convStateIn[(b * D + c) * state + l];
            }
            for (int l = 0; l < cfg.seqLen; ++l) {
                rawConcat[(b * D + c) * totalLen + state + l] = qkv[(b * D + c) * cfg.seqLen + l];
            }
        }
    }
    std::vector<float> convOut(cfg.batch * D * cfg.seqLen, 0.0f);
    for (int b = 0; b < cfg.batch; ++b) {
        for (int c = 0; c < D; ++c) {
            for (int l = 0; l < cfg.seqLen; ++l) {
                float sum = 0.0f;
                for (int kk = 0; kk < cfg.kernelSize; ++kk) {
                    float x = rawConcat[(b * D + c) * totalLen + l + kk];
                    float w = convWeight[c * cfg.kernelSize + kk];
                    sum += x * w;
                }
                convOut[(b * D + c) * cfg.seqLen + l] = sum;
            }
        }
    }
    for (float& x : convOut) {
        float sig = 1.0f / (1.0f + std::exp(-x));
        x = x * sig;
    }

    convStateOut.assign(cfg.batch * D * state, 0.0f);
    for (int b = 0; b < cfg.batch; ++b) {
        for (int c = 0; c < D; ++c) {
            for (int l = 0; l < state; ++l) {
                convStateOut[(b * D + c) * state + l] = rawConcat[(b * D + c) * totalLen + cfg.seqLen + l];
            }
        }
    }

    auto convIndex = [&](int b, int c, int l) -> size_t {
        return (static_cast<size_t>(b) * D + c) * cfg.seqLen + l;
    };
    std::vector<float> q(cfg.batch * cfg.seqLen * Hv * dk, 0.0f);
    std::vector<float> k(cfg.batch * cfg.seqLen * Hv * dk, 0.0f);
    std::vector<float> v(cfg.batch * cfg.seqLen * Hv * dv, 0.0f);
    auto qIndex = [&](int b, int l, int h, int d) -> size_t {
        return (((static_cast<size_t>(b) * cfg.seqLen + l) * Hv + h) * dk + d);
    };
    auto vIndex = [&](int b, int l, int h, int d) -> size_t {
        return (((static_cast<size_t>(b) * cfg.seqLen + l) * Hv + h) * dv + d);
    };
    for (int b = 0; b < cfg.batch; ++b) {
        for (int l = 0; l < cfg.seqLen; ++l) {
            for (int h = 0; h < Hv; ++h) {
                for (int d = 0; d < dk; ++d) {
                    q[qIndex(b, l, h, d)] = convOut[convIndex(b, h * dk + d, l)];
                    k[qIndex(b, l, h, d)] = convOut[convIndex(b, keyDim + h * dk + d, l)];
                }
                for (int d = 0; d < dv; ++d) {
                    v[vIndex(b, l, h, d)] = convOut[convIndex(b, 2 * keyDim + h * dv + d, l)];
                }
            }
        }
    }

    if (cfg.useQkL2Norm) {
        for (auto* tensor : {&q, &k}) {
            for (int b = 0; b < cfg.batch; ++b) {
                for (int l = 0; l < cfg.seqLen; ++l) {
                    for (int h = 0; h < Hv; ++h) {
                        float sum = 0.0f;
                        for (int d = 0; d < dk; ++d) {
                            float x = (*tensor)[qIndex(b, l, h, d)];
                            sum += x * x;
                        }
                        float norm = std::sqrt(sum + 1e-6f);
                        for (int d = 0; d < dk; ++d) {
                            (*tensor)[qIndex(b, l, h, d)] /= norm;
                        }
                    }
                }
            }
        }
    }

    const float qScale = 1.0f / std::sqrt(static_cast<float>(dk));
    for (float& x : q) {
        x *= qScale;
    }

    std::vector<float> stateCur = recurrentIn;
    if (cfg.seqLen != 1) {
        std::fill(stateCur.begin(), stateCur.end(), 0.0f);
    }
    auto sIndex = [&](int b, int h, int i, int j) -> size_t {
        return (((static_cast<size_t>(b) * Hv + h) * dk + i) * dv + j);
    };
    output.assign(cfg.batch * cfg.seqLen * Hv * dv, 0.0f);
    for (int t = 0; t < cfg.seqLen; ++t) {
        std::vector<float> stateNext(stateCur.size(), 0.0f);
        for (int b = 0; b < cfg.batch; ++b) {
            for (int h = 0; h < Hv; ++h) {
                float decay = std::exp(gate[(static_cast<size_t>(b) * cfg.seqLen + t) * Hv + h]);
                float betaVal = beta[(static_cast<size_t>(b) * cfg.seqLen + t) * Hv + h];
                std::vector<float> vPred(dv, 0.0f);
                for (int j = 0; j < dv; ++j) {
                    float sum = 0.0f;
                    for (int i = 0; i < dk; ++i) {
                        sum += stateCur[sIndex(b, h, i, j)] * decay * k[qIndex(b, t, h, i)];
                    }
                    vPred[j] = sum;
                }
                for (int i = 0; i < dk; ++i) {
                    float kVal = k[qIndex(b, t, h, i)];
                    for (int j = 0; j < dv; ++j) {
                        float stateDecay = stateCur[sIndex(b, h, i, j)] * decay;
                        float delta = betaVal * (v[vIndex(b, t, h, j)] - vPred[j]);
                        stateNext[sIndex(b, h, i, j)] = stateDecay + kVal * delta;
                    }
                }
                for (int j = 0; j < dv; ++j) {
                    float sum = 0.0f;
                    for (int i = 0; i < dk; ++i) {
                        sum += stateNext[sIndex(b, h, i, j)] * q[qIndex(b, t, h, i)];
                    }
                    output[vIndex(b, t, h, j)] = sum;
                }
            }
        }
        stateCur.swap(stateNext);
    }
    recurrentOut = std::move(stateCur);
}

struct Runtime {
    void* handle = nullptr;
    QNN_INTERFACE_VER_TYPE iface = QNN_INTERFACE_VER_TYPE_INIT;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ContextHandle_t context = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
};

static Runtime createRuntime(const std::string& backendPath) {
    Runtime rt;
    rt.handle = dlopen(backendPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!rt.handle) die(std::string("dlopen backend failed: ") + dlerror());
    auto getProviders = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(dlsym(rt.handle, "QnnInterface_getProviders"));
    if (!getProviders) die("dlsym QnnInterface_getProviders failed");
    const QnnInterface_t** providers = nullptr;
    uint32_t numProviders = 0;
    auto err = getProviders(&providers, &numProviders);
    if ((err & 0xffff) != QNN_SUCCESS || providers == nullptr || numProviders == 0) {
        die("QnnInterface_getProviders failed");
    }
    bool found = false;
    for (uint32_t i = 0; i < numProviders; ++i) {
        auto* p = providers[i];
        if (p->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
            p->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
            rt.iface = p->QNN_INTERFACE_VER_NAME;
            found = true;
            break;
        }
    }
    if (!found) {
        rt.iface = providers[0]->QNN_INTERFACE_VER_NAME;
    }
    err = rt.iface.backendCreate(nullptr, nullptr, &rt.backend);
    if ((err & 0xffff) != QNN_SUCCESS) die("backendCreate failed code=" + std::to_string(err & 0xffff));
    if (rt.iface.deviceCreate) {
        err = rt.iface.deviceCreate(nullptr, nullptr, &rt.device);
        if ((err & 0xffff) != QNN_SUCCESS &&
            (err & 0xffff) != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
            die("deviceCreate failed code=" + std::to_string(err & 0xffff));
        }
    }
    err = rt.iface.contextCreate(rt.backend, rt.device, nullptr, &rt.context);
    if ((err & 0xffff) != QNN_SUCCESS) die("contextCreate failed code=" + std::to_string(err & 0xffff));
    err = rt.iface.graphCreate(rt.context, "linear_front", nullptr, &rt.graph);
    if ((err & 0xffff) != QNN_SUCCESS) die("graphCreate failed code=" + std::to_string(err & 0xffff));
    return rt;
}

static void destroyRuntime(Runtime& rt) {
    if (rt.context) rt.iface.contextFree(rt.context, nullptr);
    if (rt.device && rt.iface.deviceFree) rt.iface.deviceFree(rt.device);
    if (rt.backend) rt.iface.backendFree(rt.backend);
    if (rt.handle) dlclose(rt.handle);
    rt = {};
}

static void createGraphTensor(Runtime& rt, TensorHandle& t) {
    auto err = rt.iface.tensorCreateGraphTensor(rt.graph, &t.tensor);
    if ((err & 0xffff) != QNN_SUCCESS) {
        die("tensorCreateGraphTensor failed for " + t.name + " code=" + std::to_string(err & 0xffff));
    }
}

static void createGraphParamTensor(Runtime& rt, ParamTensorHandle& t) {
    auto err = rt.iface.tensorCreateGraphTensor(rt.graph, &t.param.tensorParam);
    if ((err & 0xffff) != QNN_SUCCESS) {
        die("tensorCreateGraphTensor failed for param " + t.tensorName + " code=" + std::to_string(err & 0xffff));
    }
}

static void bindRuntimeBuffer(TensorHandle& t) {
    if (!t.storage.empty()) {
        t.tensor.v1.clientBuf.data = t.storage.data();
        t.tensor.v1.clientBuf.dataSize = t.storage.size();
    }
}

static void addNode(Runtime& rt,
                    const std::string& name,
                    const std::string& type,
                    std::vector<Qnn_Tensor_t>& inputs,
                    std::vector<Qnn_Param_t>& params,
                    std::vector<Qnn_Tensor_t>& outputs) {
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = name.c_str();
    op.v1.packageName = "qti.aisw";
    op.v1.typeName = type.c_str();
    op.v1.numOfInputs = static_cast<uint32_t>(inputs.size());
    op.v1.inputTensors = inputs.data();
    op.v1.numOfOutputs = static_cast<uint32_t>(outputs.size());
    op.v1.outputTensors = outputs.data();
    op.v1.numOfParams = static_cast<uint32_t>(params.size());
    op.v1.params = params.data();
    auto err = rt.iface.backendValidateOpConfig(rt.backend, op);
    if ((err & 0xffff) != QNN_SUCCESS) {
        die("backendValidateOpConfig failed for " + name + " type=" + type + " code=" + std::to_string(err & 0xffff));
    }
    err = rt.iface.graphAddNode(rt.graph, op);
    if ((err & 0xffff) != QNN_SUCCESS) {
        die("graphAddNode failed for " + name + " type=" + type + " code=" + std::to_string(err & 0xffff));
    }
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg = parseArgs(argc, argv);
    if (cfg.numKHeads != cfg.numVHeads) {
        die("Current native probe supports num-k-heads == num-v-heads only.");
    }
    const int keyDim = cfg.numKHeads * cfg.headKDim;
    const int valueDim = cfg.numVHeads * cfg.headVDim;
    const int D = 2 * keyDim + valueDim;
    const int state = cfg.kernelSize - 1;
    const int totalLen = state + cfg.seqLen;
    const Qnn_DataType_t dt = dataTypeOf(cfg.fp16);
    const Qnn_DataType_t recurrenceDt = cfg.recurrentFp16 ? QNN_DATATYPE_FLOAT_16 : QNN_DATATYPE_FLOAT_32;
    const std::vector<uint32_t> qkvDims{(uint32_t)cfg.batch, (uint32_t)D, (uint32_t)cfg.seqLen};
    const std::vector<uint32_t> gateDims{(uint32_t)cfg.batch, (uint32_t)cfg.seqLen, (uint32_t)cfg.numVHeads};
    const std::vector<uint32_t> betaDims{(uint32_t)cfg.batch, (uint32_t)cfg.seqLen, (uint32_t)cfg.numVHeads};
    const std::vector<uint32_t> convStateDims{(uint32_t)cfg.batch, (uint32_t)D, (uint32_t)state};
    const std::vector<uint32_t> recurrentDims{(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim, (uint32_t)cfg.headVDim};
    const std::vector<uint32_t> convWeightDims{(uint32_t)D, 1u, (uint32_t)cfg.kernelSize};
    const std::vector<uint32_t> outputDims{(uint32_t)cfg.batch, (uint32_t)cfg.seqLen, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headVDim};

    std::vector<float> qkv(cfg.batch * D * cfg.seqLen);
    std::vector<float> gate(cfg.batch * cfg.seqLen * cfg.numVHeads);
    std::vector<float> beta(cfg.batch * cfg.seqLen * cfg.numVHeads);
    std::vector<float> convWeightSrc(D * cfg.kernelSize);
    std::vector<float> convStateIn(cfg.batch * D * state);
    std::vector<float> recurrentIn(cfg.batch * cfg.numVHeads * cfg.headKDim * cfg.headVDim);
    fillDeterministic(qkv, 0.01f, 0.0f);
    fillDeterministic(gate, 0.02f, -0.1f);
    fillDeterministic(beta, 0.01f, 0.3f);
    fillConvWeight(convWeightSrc);
    fillDeterministic(convStateIn, 0.005f, 0.0f);
    fillDeterministic(recurrentIn, 0.002f, 0.0f);
    if (!cfg.inputDir.empty()) {
        qkv = readTensorFile(cfg.inputDir, "qkv", qkvDims, true);
        gate = readTensorFile(cfg.inputDir, "gate", gateDims, true);
        beta = readTensorFile(cfg.inputDir, "beta", betaDims, true);
        convStateIn = readTensorFile(cfg.inputDir, "conv_state_in", convStateDims, true);
        recurrentIn = readTensorFile(cfg.inputDir, "recurrent_in", recurrentDims, true);
        auto convWeightLoaded = readTensorFile(cfg.inputDir, "conv_weight", convWeightDims, false);
        if (!convWeightLoaded.empty()) {
            convWeightSrc = std::move(convWeightLoaded);
        }
    }

    std::vector<float> outputRef, convStateOutRef, recurrentOutRef;
    cpuLinearReference(cfg, qkv, gate, beta, convWeightSrc, convStateIn, recurrentIn,
                       outputRef, convStateOutRef, recurrentOutRef);

    std::vector<float> convWeightPacked(D * cfg.kernelSize, 0.0f);
    for (int c = 0; c < D; ++c) {
        for (int k = 0; k < cfg.kernelSize; ++k) {
            convWeightPacked[c + D * k] = convWeightSrc[c * cfg.kernelSize + k];
        }
    }
    std::vector<float> zeroBias(D, 0.0f);
    std::vector<float> qScaleValue = {1.0f / std::sqrt(static_cast<float>(cfg.headKDim))};
    std::vector<float> epsValue = {1e-6f};

    Runtime rt = createRuntime(cfg.backendPath);

    TensorHandle qkvIn = makeAppTensor("qkv", QNN_TENSOR_TYPE_APP_WRITE, dt,
                                       qkvDims, &qkv, true);
    TensorHandle gateIn = makeAppTensor("gate", QNN_TENSOR_TYPE_APP_WRITE, dt,
                                        gateDims, &gate, true);
    TensorHandle betaIn = makeAppTensor("beta", QNN_TENSOR_TYPE_APP_WRITE, dt,
                                        betaDims, &beta, true);
    TensorHandle convStateInT = makeAppTensor("conv_state_in", QNN_TENSOR_TYPE_APP_WRITE, dt,
                                              convStateDims, &convStateIn, true);
    TensorHandle recurrentInT = makeAppTensor("recurrent_in", QNN_TENSOR_TYPE_APP_WRITE, recurrenceDt,
                                              recurrentDims, &recurrentIn, true);
    TensorHandle outputOut = makeAppTensor("output", QNN_TENSOR_TYPE_APP_READ, dt,
                                           outputDims,
                                           nullptr, false);
    TensorHandle convStateOutT = makeAppTensor("conv_state_out", QNN_TENSOR_TYPE_APP_READ, dt,
                                               convStateDims, nullptr, false);
    TensorHandle recurrentOutT = makeAppTensor("recurrent_out", QNN_TENSOR_TYPE_APP_READ, recurrenceDt,
                                               recurrentDims, nullptr, false);
    outputOut.storage.resize(elementCount(outputOut.dims) * (cfg.fp16 ? sizeof(uint16_t) : sizeof(float)));
    convStateOutT.storage.resize(elementCount(convStateOutT.dims) * (cfg.fp16 ? sizeof(uint16_t) : sizeof(float)));
    recurrentOutT.storage.resize(elementCount(recurrentOutT.dims) * (cfg.recurrentFp16 ? sizeof(uint16_t) : sizeof(float)));

    TensorHandle convWeightT = makeStaticTensor("conv_w", dt,
                                                {1u, (uint32_t)cfg.kernelSize, 1u, (uint32_t)D}, convWeightPacked);
    TensorHandle convBiasT = makeStaticTensor("conv_b", dt, {(uint32_t)D}, zeroBias);
    TensorHandle qScaleT = makeStaticTensor("q_scale", dt, {1u, 1u, 1u, 1u}, qScaleValue);
    TensorHandle epsT = makeStaticTensor("eps", dt, {1u, 1u, 1u, 1u}, epsValue);
    TensorHandle zeroStateScalar = makeStaticTensor("zero_state_scalar", recurrenceDt, {1u}, std::vector<float>{0.0f});

    TensorHandle rawConcat = makeNativeTensor("raw_concat", dt,
                                              {(uint32_t)cfg.batch, (uint32_t)D, (uint32_t)totalLen});
    TensorHandle convTrans = makeNativeTensor("conv_trans", dt,
                                              {(uint32_t)cfg.batch, (uint32_t)totalLen, (uint32_t)D});
    TensorHandle convInput4d = makeNativeTensor("conv_in4d", dt,
                                                {(uint32_t)cfg.batch, 1u, (uint32_t)totalLen, (uint32_t)D});
    TensorHandle convOut4d = makeNativeTensor("conv_out4d", dt,
                                              {(uint32_t)cfg.batch, 1u, (uint32_t)cfg.seqLen, (uint32_t)D});
    TensorHandle convOut3dTrans = makeNativeTensor("conv_out3d_trans", dt,
                                                   {(uint32_t)cfg.batch, (uint32_t)cfg.seqLen, (uint32_t)D});
    TensorHandle convOut = makeNativeTensor("conv_out", dt,
                                            {(uint32_t)cfg.batch, (uint32_t)D, (uint32_t)cfg.seqLen});
    TensorHandle convSigmoid = makeNativeTensor("conv_sigmoid", dt,
                                                {(uint32_t)cfg.batch, (uint32_t)D, (uint32_t)cfg.seqLen});
    TensorHandle convSilu = makeNativeTensor("conv_silu", dt,
                                             {(uint32_t)cfg.batch, (uint32_t)D, (uint32_t)cfg.seqLen});
    TensorHandle qFlat = makeNativeTensor("q_flat", dt,
                                          {(uint32_t)cfg.batch, (uint32_t)keyDim, (uint32_t)cfg.seqLen});
    TensorHandle kFlat = makeNativeTensor("k_flat", dt,
                                          {(uint32_t)cfg.batch, (uint32_t)keyDim, (uint32_t)cfg.seqLen});
    TensorHandle vFlat = makeNativeTensor("v_flat", dt,
                                          {(uint32_t)cfg.batch, (uint32_t)valueDim, (uint32_t)cfg.seqLen});
    TensorHandle q4 = makeNativeTensor("q4", dt,
                                       {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim, (uint32_t)cfg.seqLen});
    TensorHandle k4 = makeNativeTensor("k4", dt,
                                       {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim, (uint32_t)cfg.seqLen});
    TensorHandle v4 = makeNativeTensor("v4", dt,
                                       {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headVDim, (uint32_t)cfg.seqLen});
    TensorHandle qSeq = makeNativeTensor("q_seq", dt,
                                         {(uint32_t)cfg.batch, (uint32_t)cfg.seqLen, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim});
    TensorHandle kSeq = makeNativeTensor("k_seq", dt,
                                         {(uint32_t)cfg.batch, (uint32_t)cfg.seqLen, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim});
    TensorHandle vSeq = makeNativeTensor("v_seq", dt,
                                         {(uint32_t)cfg.batch, (uint32_t)cfg.seqLen, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headVDim});
    TensorHandle qScaled = makeNativeTensor("q_scaled", dt,
                                            {(uint32_t)cfg.batch, (uint32_t)cfg.seqLen, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim});
    TensorHandle stateCur = makeNativeTensor("state_cur", recurrenceDt,
                                             {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim, (uint32_t)cfg.headVDim});
    TensorHandle finalOutFp32 = makeNativeTensor("final_out_fp32", recurrenceDt,
                                                 {(uint32_t)cfg.batch, (uint32_t)cfg.seqLen, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headVDim});
    ParamTensorHandle transposePerm0 = makeUInt32ParamTensor("perm_tensor0", {3u}, {0u, 2u, 1u}, "perm");
    ParamTensorHandle transposePerm1 = makeUInt32ParamTensor("perm_tensor1", {3u}, {0u, 2u, 1u}, "perm");
    ParamTensorHandle transposePermQkv = makeUInt32ParamTensor("perm_tensor_qkv", {4u}, {0u, 3u, 1u, 2u}, "perm");
    ParamTensorHandle convStride = makeUInt32ParamTensor("stride_tensor", {2u}, {1u, 1u}, "stride");
    ParamTensorHandle convDilation = makeUInt32ParamTensor("dilation_tensor", {2u}, {1u, 1u}, "dilation");
    ParamTensorHandle convPad = makeUInt32ParamTensor("pad_tensor", {2u, 2u}, {0u, 0u, 0u, 0u}, "pad_amount");
    ParamTensorHandle splitIndex = makeUInt32ParamTensor("split_index_tensor", {2u},
                                                         {(uint32_t)keyDim, (uint32_t)(2 * keyDim)},
                                                         "split_index");
    ParamTensorHandle convStateSlice = makeInt32ParamTensor("conv_state_slice", {3u, 3u},
                                                            {0, (int32_t)cfg.batch, 1,
                                                             0, D, 1,
                                                             cfg.seqLen, totalLen, 1},
                                                            "ranges");

    for (auto* t : {&qkvIn, &gateIn, &betaIn, &convStateInT, &recurrentInT, &outputOut, &convStateOutT, &recurrentOutT,
                    &convWeightT, &convBiasT, &qScaleT, &epsT, &zeroStateScalar, &rawConcat,
                    &convTrans, &convInput4d, &convOut4d, &convOut3dTrans, &convOut, &convSigmoid, &convSilu,
                    &qFlat, &kFlat, &vFlat, &q4, &k4, &v4, &qSeq, &kSeq, &vSeq, &qScaled, &stateCur, &finalOutFp32}) {
        createGraphTensor(rt, *t);
    }
    for (auto* t : {&transposePerm0, &transposePerm1, &transposePermQkv, &convStride, &convDilation, &convPad, &splitIndex, &convStateSlice}) {
        createGraphParamTensor(rt, *t);
    }
    bindRuntimeBuffer(qkvIn);
    bindRuntimeBuffer(gateIn);
    bindRuntimeBuffer(betaIn);
    bindRuntimeBuffer(convStateInT);
    bindRuntimeBuffer(recurrentInT);
    bindRuntimeBuffer(outputOut);
    bindRuntimeBuffer(convStateOutT);
    bindRuntimeBuffer(recurrentOutT);

    {
        std::vector<Qnn_Tensor_t> inputs = {convStateInT.tensor, qkvIn.tensor};
        std::vector<Qnn_Tensor_t> outputs = {rawConcat.tensor};
        auto axis = makeUInt32ScalarParam("axis", 2);
        std::vector<Qnn_Param_t> params = {axis};
        addNode(rt, "raw_concat", "Concat", inputs, params, outputs);
    }
    {
        std::vector<Qnn_Tensor_t> inputs = {rawConcat.tensor};
        std::vector<Qnn_Tensor_t> outputs = {convStateOutT.tensor};
        std::vector<Qnn_Param_t> params = {convStateSlice.param};
        addNode(rt, "conv_state_out", "StridedSlice", inputs, params, outputs);
    }
    {
        std::vector<Qnn_Tensor_t> inputs = {rawConcat.tensor};
        std::vector<Qnn_Tensor_t> outputs = {convTrans.tensor};
        std::vector<Qnn_Param_t> params = {transposePerm0.param};
        addNode(rt, "conv_transpose", "Transpose", inputs, params, outputs);
    }
    {
        std::vector<Qnn_Tensor_t> inputs = {convTrans.tensor};
        std::vector<Qnn_Tensor_t> outputs = {convInput4d.tensor};
        std::vector<Qnn_Param_t> params;
        addNode(rt, "conv_input4d", "Reshape", inputs, params, outputs);
    }
    {
        std::vector<Qnn_Tensor_t> inputs = {convInput4d.tensor, convWeightT.tensor, convBiasT.tensor};
        std::vector<Qnn_Tensor_t> outputs = {convOut4d.tensor};
        std::vector<Qnn_Param_t> params = {convStride.param, convPad.param, convDilation.param};
        addNode(rt, "dwconv", "DepthWiseConv2d", inputs, params, outputs);
    }
    {
        std::vector<Qnn_Tensor_t> inputs = {convOut4d.tensor};
        std::vector<Qnn_Tensor_t> outputs = {convOut3dTrans.tensor};
        std::vector<Qnn_Param_t> params;
        addNode(rt, "conv_out3dtrans", "Reshape", inputs, params, outputs);
    }
    {
        std::vector<Qnn_Tensor_t> inputs = {convOut3dTrans.tensor};
        std::vector<Qnn_Tensor_t> outputs = {convOut.tensor};
        std::vector<Qnn_Param_t> params = {transposePerm1.param};
        addNode(rt, "conv_out", "Transpose", inputs, params, outputs);
    }
    {
        std::vector<Qnn_Tensor_t> inputs = {convOut.tensor};
        std::vector<Qnn_Tensor_t> outputs = {convSigmoid.tensor};
        std::vector<Qnn_Param_t> params;
        addNode(rt, "sigmoid", "Sigmoid", inputs, params, outputs);
    }
    {
        std::vector<Qnn_Tensor_t> inputs = {convOut.tensor, convSigmoid.tensor};
        std::vector<Qnn_Tensor_t> outputs = {convSilu.tensor};
        std::vector<Qnn_Param_t> params;
        addNode(rt, "silu", "ElementWiseMultiply", inputs, params, outputs);
    }
    {
        auto axis = makeUInt32ScalarParam("axis", 1);
        std::vector<Qnn_Tensor_t> inputs = {convSilu.tensor};
        std::vector<Qnn_Tensor_t> outputs = {qFlat.tensor, kFlat.tensor, vFlat.tensor};
        std::vector<Qnn_Param_t> params = {axis, splitIndex.param};
        addNode(rt, "split_qkv", "Split", inputs, params, outputs);
    }
    {
        std::vector<Qnn_Tensor_t> inputs = {qFlat.tensor};
        std::vector<Qnn_Tensor_t> outputs = {q4.tensor};
        std::vector<Qnn_Param_t> params;
        addNode(rt, "reshape_q4", "Reshape", inputs, params, outputs);
    }
    {
        std::vector<Qnn_Tensor_t> inputs = {kFlat.tensor};
        std::vector<Qnn_Tensor_t> outputs = {k4.tensor};
        std::vector<Qnn_Param_t> params;
        addNode(rt, "reshape_k4", "Reshape", inputs, params, outputs);
    }
    {
        std::vector<Qnn_Tensor_t> inputs = {vFlat.tensor};
        std::vector<Qnn_Tensor_t> outputs = {v4.tensor};
        std::vector<Qnn_Param_t> params;
        addNode(rt, "reshape_v4", "Reshape", inputs, params, outputs);
    }
    {
        std::vector<Qnn_Tensor_t> inputs = {q4.tensor};
        std::vector<Qnn_Tensor_t> outputs = {qSeq.tensor};
        std::vector<Qnn_Param_t> params = {transposePermQkv.param};
        addNode(rt, "transpose_q", "Transpose", inputs, params, outputs);
    }
    {
        std::vector<Qnn_Tensor_t> inputs = {k4.tensor};
        std::vector<Qnn_Tensor_t> outputs = {kSeq.tensor};
        std::vector<Qnn_Param_t> params = {transposePermQkv.param};
        addNode(rt, "transpose_k", "Transpose", inputs, params, outputs);
    }
    {
        std::vector<Qnn_Tensor_t> inputs = {v4.tensor};
        std::vector<Qnn_Tensor_t> outputs = {vSeq.tensor};
        std::vector<Qnn_Param_t> params = {transposePermQkv.param};
        addNode(rt, "transpose_v", "Transpose", inputs, params, outputs);
    }
    TensorHandle* qNormInput = &qSeq;
    TensorHandle* kNormInput = &kSeq;
    std::vector<std::unique_ptr<TensorHandle>> ownedNormTensors;
    if (cfg.useQkL2Norm) {
        auto addL2Norm = [&](const std::string& prefix, TensorHandle* input) -> TensorHandle* {
            auto square = std::make_unique<TensorHandle>(makeNativeTensor(prefix + "_square", dt,
                                                                          {(uint32_t)cfg.batch, (uint32_t)cfg.seqLen, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim}));
            auto sum = std::make_unique<TensorHandle>(makeNativeTensor(prefix + "_sum", dt,
                                                                       {(uint32_t)cfg.batch, (uint32_t)cfg.seqLen, (uint32_t)cfg.numVHeads, 1u}));
            auto plusEps = std::make_unique<TensorHandle>(makeNativeTensor(prefix + "_plus_eps", dt,
                                                                           {(uint32_t)cfg.batch, (uint32_t)cfg.seqLen, (uint32_t)cfg.numVHeads, 1u}));
            auto sqrt = std::make_unique<TensorHandle>(makeNativeTensor(prefix + "_sqrt", dt,
                                                                        {(uint32_t)cfg.batch, (uint32_t)cfg.seqLen, (uint32_t)cfg.numVHeads, 1u}));
            auto norm = std::make_unique<TensorHandle>(makeNativeTensor(prefix + "_norm", dt,
                                                                        {(uint32_t)cfg.batch, (uint32_t)cfg.seqLen, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim}));

            for (auto* t : {square.get(), sum.get(), plusEps.get(), sqrt.get(), norm.get()}) {
                createGraphTensor(rt, *t);
            }

            {
                std::vector<Qnn_Tensor_t> inputs = {input->tensor, input->tensor};
                std::vector<Qnn_Tensor_t> outputs = {square->tensor};
                std::vector<Qnn_Param_t> params;
                addNode(rt, prefix + "_square", "ElementWiseMultiply", inputs, params, outputs);
            }
            {
                auto axesName = prefix + "_axes";
                auto axes = makeUInt32ParamTensor(axesName.c_str(), {1u}, {3u}, "axes");
                std::vector<Qnn_Tensor_t> inputs = {square->tensor};
                std::vector<Qnn_Tensor_t> outputs = {sum->tensor};
                std::vector<Qnn_Param_t> params = {axes.param, makeBoolScalarParam("keep_dims", true)};
                addNode(rt, prefix + "_sum", "ReduceSum", inputs, params, outputs);
            }
            {
                std::vector<Qnn_Tensor_t> inputs = {sum->tensor, epsT.tensor};
                std::vector<Qnn_Tensor_t> outputs = {plusEps->tensor};
                std::vector<Qnn_Param_t> params;
                addNode(rt, prefix + "_plus_eps", "ElementWiseAdd", inputs, params, outputs);
            }
            {
                std::vector<Qnn_Tensor_t> inputs = {plusEps->tensor};
                std::vector<Qnn_Tensor_t> outputs = {sqrt->tensor};
                std::vector<Qnn_Param_t> params;
                addNode(rt, prefix + "_sqrt", "ElementWiseSquareRoot", inputs, params, outputs);
            }
            {
                std::vector<Qnn_Tensor_t> inputs = {input->tensor, sqrt->tensor};
                std::vector<Qnn_Tensor_t> outputs = {norm->tensor};
                std::vector<Qnn_Param_t> params;
                addNode(rt, prefix + "_norm", "ElementWiseDivide", inputs, params, outputs);
            }

            ownedNormTensors.emplace_back(std::move(square));
            ownedNormTensors.emplace_back(std::move(sum));
            ownedNormTensors.emplace_back(std::move(plusEps));
            ownedNormTensors.emplace_back(std::move(sqrt));
            ownedNormTensors.emplace_back(std::move(norm));
            return ownedNormTensors.back().get();
        };
        qNormInput = addL2Norm("q_norm", &qSeq);
        kNormInput = addL2Norm("k_norm", &kSeq);
    }
    {
        std::vector<Qnn_Tensor_t> inputs = {qNormInput->tensor, qScaleT.tensor};
        std::vector<Qnn_Tensor_t> outputs = {qScaled.tensor};
        std::vector<Qnn_Param_t> params;
        addNode(rt, "q_scaled", "ElementWiseMultiply", inputs, params, outputs);
    }
    TensorHandle* kGatherSource = kNormInput;
    if (cfg.seqLen == 1) {
        std::vector<Qnn_Tensor_t> inputs = {recurrentInT.tensor};
        std::vector<Qnn_Tensor_t> outputs = {stateCur.tensor};
        std::vector<Qnn_Param_t> params;
        addNode(rt, "state_cur", "Reshape", inputs, params, outputs);
    } else {
        std::vector<Qnn_Tensor_t> inputs = {recurrentInT.tensor, zeroStateScalar.tensor};
        std::vector<Qnn_Tensor_t> outputs = {stateCur.tensor};
        std::vector<Qnn_Param_t> params;
        addNode(rt, "state_cur_zero", "ElementWiseMultiply", inputs, params, outputs);
    }

    std::vector<TensorHandle> stepOutputSteps;
    stepOutputSteps.reserve(cfg.seqLen);
    TensorHandle* stateCurTensor = &stateCur;
    std::vector<std::unique_ptr<TensorHandle>> ownedStates;
    std::vector<std::unique_ptr<TensorHandle>> ownedSteps;
    std::vector<std::unique_ptr<TensorHandle>> ownedIndices;
    for (int t = 0; t < cfg.seqLen; ++t) {
        auto indexTensor = std::make_unique<TensorHandle>(makeStaticInt32Tensor("gather_index_" + std::to_string(t), {1u}, {t}));
        createGraphTensor(rt, *indexTensor);
        ownedIndices.emplace_back(std::move(indexTensor));
        TensorHandle& gatherIndex = *ownedIndices.back();

        auto qStep = std::make_unique<TensorHandle>(makeNativeTensor("q_step_" + std::to_string(t), dt, {(uint32_t)cfg.batch, 1u, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim}));
        auto kStep = std::make_unique<TensorHandle>(makeNativeTensor("k_step_" + std::to_string(t), dt, {(uint32_t)cfg.batch, 1u, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim}));
        auto vStep = std::make_unique<TensorHandle>(makeNativeTensor("v_step_" + std::to_string(t), dt, {(uint32_t)cfg.batch, 1u, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headVDim}));
        auto gStep = std::make_unique<TensorHandle>(makeNativeTensor("g_step_" + std::to_string(t), dt, {(uint32_t)cfg.batch, 1u, (uint32_t)cfg.numVHeads}));
        auto betaStep = std::make_unique<TensorHandle>(makeNativeTensor("beta_step_" + std::to_string(t), dt, {(uint32_t)cfg.batch, 1u, (uint32_t)cfg.numVHeads}));
        auto qStepFp32 = std::make_unique<TensorHandle>(makeNativeTensor("q_step_fp32_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, 1u, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim}));
        auto kStepFp32 = std::make_unique<TensorHandle>(makeNativeTensor("k_step_fp32_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, 1u, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim}));
        auto vStepFp32 = std::make_unique<TensorHandle>(makeNativeTensor("v_step_fp32_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, 1u, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headVDim}));
        auto gStepFp32 = std::make_unique<TensorHandle>(makeNativeTensor("g_step_fp32_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, 1u, (uint32_t)cfg.numVHeads}));
        auto betaStepFp32 = std::make_unique<TensorHandle>(makeNativeTensor("beta_step_fp32_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, 1u, (uint32_t)cfg.numVHeads}));
        auto gState = std::make_unique<TensorHandle>(makeNativeTensor("g_state_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, 1u, 1u}));
        auto betaVec = std::make_unique<TensorHandle>(makeNativeTensor("beta_vec_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, 1u}));
        auto qVec = std::make_unique<TensorHandle>(makeNativeTensor("q_vec_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim, 1u}));
        auto kVec = std::make_unique<TensorHandle>(makeNativeTensor("k_vec_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim, 1u}));
        auto vVec = std::make_unique<TensorHandle>(makeNativeTensor("v_vec_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headVDim}));
        auto decay = std::make_unique<TensorHandle>(makeNativeTensor("decay_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, 1u, 1u}));
        auto stateDecay = std::make_unique<TensorHandle>(makeNativeTensor("state_decay_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim, (uint32_t)cfg.headVDim}));
        auto weightedState = std::make_unique<TensorHandle>(makeNativeTensor("weighted_state_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim, (uint32_t)cfg.headVDim}));
        auto vPred = std::make_unique<TensorHandle>(makeNativeTensor("v_pred_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headVDim}));
        auto diff = std::make_unique<TensorHandle>(makeNativeTensor("diff_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headVDim}));
        auto delta = std::make_unique<TensorHandle>(makeNativeTensor("delta_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headVDim}));
        auto deltaOuter = std::make_unique<TensorHandle>(makeNativeTensor("delta_outer_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, 1u, (uint32_t)cfg.headVDim}));
        auto update = std::make_unique<TensorHandle>(makeNativeTensor("update_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim, (uint32_t)cfg.headVDim}));
        auto weightedQuery = std::make_unique<TensorHandle>(makeNativeTensor("weighted_query_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim, (uint32_t)cfg.headVDim}));
        auto outputVec = std::make_unique<TensorHandle>(makeNativeTensor("output_vec_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headVDim}));
        auto outputStep = std::make_unique<TensorHandle>(makeNativeTensor("output_step_" + std::to_string(t), recurrenceDt, {(uint32_t)cfg.batch, 1u, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headVDim}));

        std::vector<TensorHandle*> stepTensors = {
            qStep.get(), kStep.get(), vStep.get(), gStep.get(), betaStep.get(),
            qStepFp32.get(), kStepFp32.get(), vStepFp32.get(), gStepFp32.get(), betaStepFp32.get(),
            gState.get(), betaVec.get(), qVec.get(), kVec.get(), vVec.get(), decay.get(), stateDecay.get(),
            weightedState.get(), vPred.get(), diff.get(), delta.get(), deltaOuter.get(), update.get(),
            weightedQuery.get(), outputVec.get(), outputStep.get()
        };
        for (auto* tt : stepTensors) {
            createGraphTensor(rt, *tt);
        }

        auto axis1 = makeInt32ScalarParam("axis", 1);
        {
            std::vector<Qnn_Tensor_t> inputs = {qScaled.tensor, gatherIndex.tensor};
            std::vector<Qnn_Tensor_t> outputs = {qStep->tensor};
            std::vector<Qnn_Param_t> params = {axis1};
            addNode(rt, "gather_q_" + std::to_string(t), "Gather", inputs, params, outputs);
        }
        {
            std::vector<Qnn_Tensor_t> inputs = {kGatherSource->tensor, gatherIndex.tensor};
            std::vector<Qnn_Tensor_t> outputs = {kStep->tensor};
            std::vector<Qnn_Param_t> params = {axis1};
            addNode(rt, "gather_k_" + std::to_string(t), "Gather", inputs, params, outputs);
        }
        {
            std::vector<Qnn_Tensor_t> inputs = {vSeq.tensor, gatherIndex.tensor};
            std::vector<Qnn_Tensor_t> outputs = {vStep->tensor};
            std::vector<Qnn_Param_t> params = {axis1};
            addNode(rt, "gather_v_" + std::to_string(t), "Gather", inputs, params, outputs);
        }
        {
            std::vector<Qnn_Tensor_t> inputs = {gateIn.tensor, gatherIndex.tensor};
            std::vector<Qnn_Tensor_t> outputs = {gStep->tensor};
            std::vector<Qnn_Param_t> params = {axis1};
            addNode(rt, "gather_g_" + std::to_string(t), "Gather", inputs, params, outputs);
        }
        {
            std::vector<Qnn_Tensor_t> inputs = {betaIn.tensor, gatherIndex.tensor};
            std::vector<Qnn_Tensor_t> outputs = {betaStep->tensor};
            std::vector<Qnn_Param_t> params = {axis1};
            addNode(rt, "gather_beta_" + std::to_string(t), "Gather", inputs, params, outputs);
        }

        for (auto pair : std::vector<std::pair<TensorHandle*, TensorHandle*>>{
                 {qStep.get(), qStepFp32.get()}, {kStep.get(), kStepFp32.get()}, {vStep.get(), vStepFp32.get()},
                 {gStep.get(), gStepFp32.get()}, {betaStep.get(), betaStepFp32.get()}}) {
            std::vector<Qnn_Tensor_t> inputs = {pair.first->tensor};
            std::vector<Qnn_Tensor_t> outputs = {pair.second->tensor};
            std::vector<Qnn_Param_t> params;
            addNode(rt, "cast_" + pair.first->name, "Cast", inputs, params, outputs);
        }

        for (auto pair : std::vector<std::pair<TensorHandle*, TensorHandle*>>{
                 {gStepFp32.get(), gState.get()}, {betaStepFp32.get(), betaVec.get()}, {qStepFp32.get(), qVec.get()},
                 {kStepFp32.get(), kVec.get()}, {vStepFp32.get(), vVec.get()}}) {
            std::vector<Qnn_Tensor_t> inputs = {pair.first->tensor};
            std::vector<Qnn_Tensor_t> outputs = {pair.second->tensor};
            std::vector<Qnn_Param_t> params;
            addNode(rt, "reshape_" + pair.first->name, "Reshape", inputs, params, outputs);
        }

        {
            std::vector<Qnn_Tensor_t> inputs = {gState->tensor};
            std::vector<Qnn_Tensor_t> outputs = {decay->tensor};
            std::vector<Qnn_Param_t> params;
            addNode(rt, "exp_decay_" + std::to_string(t), "ElementWiseExp", inputs, params, outputs);
        }
        {
            std::vector<Qnn_Tensor_t> inputs = {stateCurTensor->tensor, decay->tensor};
            std::vector<Qnn_Tensor_t> outputs = {stateDecay->tensor};
            std::vector<Qnn_Param_t> params;
            addNode(rt, "state_decay_" + std::to_string(t), "ElementWiseMultiply", inputs, params, outputs);
        }
        {
            std::vector<Qnn_Tensor_t> inputs = {stateDecay->tensor, kVec->tensor};
            std::vector<Qnn_Tensor_t> outputs = {weightedState->tensor};
            std::vector<Qnn_Param_t> params;
            addNode(rt, "weighted_state_" + std::to_string(t), "ElementWiseMultiply", inputs, params, outputs);
        }
        {
            auto axes = makeUInt32ParamTensor(("reduce_axes_vpred_" + std::to_string(t)).c_str(), {1u}, {2u}, "axes");
            createGraphParamTensor(rt, axes);
            auto keep = makeBoolScalarParam("keep_dims", false);
            std::vector<Qnn_Tensor_t> inputs = {weightedState->tensor};
            std::vector<Qnn_Tensor_t> outputs = {vPred->tensor};
            std::vector<Qnn_Param_t> params = {axes.param, keep};
            addNode(rt, "reduce_vpred_" + std::to_string(t), "ReduceSum", inputs, params, outputs);
        }
        {
            std::vector<Qnn_Tensor_t> inputs = {vVec->tensor, vPred->tensor};
            std::vector<Qnn_Tensor_t> outputs = {diff->tensor};
            std::vector<Qnn_Param_t> params;
            addNode(rt, "diff_" + std::to_string(t), "ElementWiseSubtract", inputs, params, outputs);
        }
        {
            std::vector<Qnn_Tensor_t> inputs = {betaVec->tensor, diff->tensor};
            std::vector<Qnn_Tensor_t> outputs = {delta->tensor};
            std::vector<Qnn_Param_t> params;
            addNode(rt, "delta_" + std::to_string(t), "ElementWiseMultiply", inputs, params, outputs);
        }
        {
            std::vector<Qnn_Tensor_t> inputs = {delta->tensor};
            std::vector<Qnn_Tensor_t> outputs = {deltaOuter->tensor};
            std::vector<Qnn_Param_t> params;
            addNode(rt, "delta_outer_" + std::to_string(t), "Reshape", inputs, params, outputs);
        }
        {
            std::vector<Qnn_Tensor_t> inputs = {kVec->tensor, deltaOuter->tensor};
            std::vector<Qnn_Tensor_t> outputs = {update->tensor};
            std::vector<Qnn_Param_t> params;
            addNode(rt, "update_" + std::to_string(t), "ElementWiseMultiply", inputs, params, outputs);
        }

        TensorHandle* stateNextTensor = nullptr;
        if (t == cfg.seqLen - 1) {
            stateNextTensor = &recurrentOutT;
        } else {
            auto stateNext = std::make_unique<TensorHandle>(makeNativeTensor("state_next_" + std::to_string(t), recurrenceDt,
                                                                             {(uint32_t)cfg.batch, (uint32_t)cfg.numVHeads, (uint32_t)cfg.headKDim, (uint32_t)cfg.headVDim}));
            createGraphTensor(rt, *stateNext);
            stateNextTensor = stateNext.get();
            ownedStates.emplace_back(std::move(stateNext));
        }
        {
            std::vector<Qnn_Tensor_t> inputs = {stateDecay->tensor, update->tensor};
            std::vector<Qnn_Tensor_t> outputs = {stateNextTensor->tensor};
            std::vector<Qnn_Param_t> params;
            addNode(rt, "state_next_" + std::to_string(t), "ElementWiseAdd", inputs, params, outputs);
        }
        {
            std::vector<Qnn_Tensor_t> inputs = {stateNextTensor->tensor, qVec->tensor};
            std::vector<Qnn_Tensor_t> outputs = {weightedQuery->tensor};
            std::vector<Qnn_Param_t> params;
            addNode(rt, "weighted_query_" + std::to_string(t), "ElementWiseMultiply", inputs, params, outputs);
        }
        {
            auto axes = makeUInt32ParamTensor(("reduce_axes_out_" + std::to_string(t)).c_str(), {1u}, {2u}, "axes");
            createGraphParamTensor(rt, axes);
            auto keep = makeBoolScalarParam("keep_dims", false);
            std::vector<Qnn_Tensor_t> inputs = {weightedQuery->tensor};
            std::vector<Qnn_Tensor_t> outputs = {outputVec->tensor};
            std::vector<Qnn_Param_t> params = {axes.param, keep};
            addNode(rt, "reduce_output_" + std::to_string(t), "ReduceSum", inputs, params, outputs);
        }
        {
            std::vector<Qnn_Tensor_t> inputs = {outputVec->tensor};
            std::vector<Qnn_Tensor_t> outputs = {outputStep->tensor};
            std::vector<Qnn_Param_t> params;
            addNode(rt, "reshape_output_step_" + std::to_string(t), "Reshape", inputs, params, outputs);
        }

        stepOutputSteps.push_back(*outputStep);
        ownedSteps.emplace_back(std::move(qStep));
        ownedSteps.emplace_back(std::move(kStep));
        ownedSteps.emplace_back(std::move(vStep));
        ownedSteps.emplace_back(std::move(gStep));
        ownedSteps.emplace_back(std::move(betaStep));
        ownedSteps.emplace_back(std::move(qStepFp32));
        ownedSteps.emplace_back(std::move(kStepFp32));
        ownedSteps.emplace_back(std::move(vStepFp32));
        ownedSteps.emplace_back(std::move(gStepFp32));
        ownedSteps.emplace_back(std::move(betaStepFp32));
        ownedSteps.emplace_back(std::move(gState));
        ownedSteps.emplace_back(std::move(betaVec));
        ownedSteps.emplace_back(std::move(qVec));
        ownedSteps.emplace_back(std::move(kVec));
        ownedSteps.emplace_back(std::move(vVec));
        ownedSteps.emplace_back(std::move(decay));
        ownedSteps.emplace_back(std::move(stateDecay));
        ownedSteps.emplace_back(std::move(weightedState));
        ownedSteps.emplace_back(std::move(vPred));
        ownedSteps.emplace_back(std::move(diff));
        ownedSteps.emplace_back(std::move(delta));
        ownedSteps.emplace_back(std::move(deltaOuter));
        ownedSteps.emplace_back(std::move(update));
        ownedSteps.emplace_back(std::move(weightedQuery));
        ownedSteps.emplace_back(std::move(outputVec));
        ownedSteps.emplace_back(std::move(outputStep));
        stateCurTensor = stateNextTensor;
    }
    if (cfg.seqLen == 1) {
        std::vector<Qnn_Tensor_t> inputs = {stepOutputSteps[0].tensor};
        std::vector<Qnn_Tensor_t> outputs = {finalOutFp32.tensor};
        std::vector<Qnn_Param_t> params;
        addNode(rt, "final_out_fp32", "Reshape", inputs, params, outputs);
    } else {
        std::vector<Qnn_Tensor_t> inputs;
        for (auto& step : stepOutputSteps) inputs.push_back(step.tensor);
        std::vector<Qnn_Tensor_t> outputs = {finalOutFp32.tensor};
        auto axis = makeUInt32ScalarParam("axis", 1);
        std::vector<Qnn_Param_t> params = {axis};
        addNode(rt, "final_out_fp32", "Concat", inputs, params, outputs);
    }
    {
        std::vector<Qnn_Tensor_t> inputs = {finalOutFp32.tensor};
        std::vector<Qnn_Tensor_t> outputs = {outputOut.tensor};
        std::vector<Qnn_Param_t> params;
        addNode(rt, "final_cast", "Cast", inputs, params, outputs);
    }

    auto err = rt.iface.graphFinalize(rt.graph, nullptr, nullptr);
    if ((err & 0xffff) != QNN_SUCCESS) {
        die("graphFinalize failed code=" + std::to_string(err & 0xffff));
    }

    std::vector<Qnn_Tensor_t> execInputs = {qkvIn.tensor, gateIn.tensor, betaIn.tensor, convStateInT.tensor, recurrentInT.tensor};
    std::vector<Qnn_Tensor_t> execOutputs = {outputOut.tensor, convStateOutT.tensor, recurrentOutT.tensor};
    err = rt.iface.graphExecute(rt.graph,
                                execInputs.data(),
                                static_cast<uint32_t>(execInputs.size()),
                                execOutputs.data(),
                                static_cast<uint32_t>(execOutputs.size()),
                                nullptr,
                                nullptr);
    if ((err & 0xffff) != QNN_SUCCESS) {
        die("graphExecute failed code=" + std::to_string(err & 0xffff));
    }

    auto outputGot = fromBackendData(outputOut.storage, cfg.fp16);
    auto convStateGot = fromBackendData(convStateOutT.storage, cfg.fp16);
    auto recurrentGot = fromBackendData(recurrentOutT.storage, cfg.recurrentFp16);
    auto outputStats = compare(outputRef, outputGot);
    auto convStateStats = compare(convStateOutRef, convStateGot);
    auto recurrentStats = compare(recurrentOutRef, recurrentGot);

    std::cout << "backend=" << cfg.backendPath << "\n";
    std::cout << "dtype=" << (cfg.fp16 ? "fp16" : "fp32")
              << " B=" << cfg.batch
              << " D=" << D
              << " L=" << cfg.seqLen
              << " kernel=" << cfg.kernelSize
              << " recurrent=" << (cfg.recurrentFp16 ? "fp16" : "fp32")
              << " qk_l2norm=" << (cfg.useQkL2Norm ? 1 : 0)
              << "\n";
    auto print = [](const char* tag, const CompareStats& s) {
        std::cout << tag
                  << " maxAbs=" << s.maxAbs
                  << " meanAbs=" << s.meanAbs
                  << " rmse=" << s.rmse
                  << " cosine=" << s.cosine
                  << " nan=" << s.nanCount
                  << " inf=" << s.infCount
                  << "\n";
    };
    print("output", outputStats);
    print("conv_state", convStateStats);
    print("recurrent_state", recurrentStats);
    if (!cfg.outputDir.empty()) {
        writeTensorFiles(cfg.outputDir, "output", outputDims, outputGot, "float32");
        writeTensorFiles(cfg.outputDir, "conv_state_out", convStateDims, convStateGot, "float32");
        writeTensorFiles(cfg.outputDir, "recurrent_out", recurrentDims, recurrentGot, "float32");
        writeTensorFiles(cfg.outputDir, "output_ref", outputDims, outputRef, "float32");
        writeTensorFiles(cfg.outputDir, "conv_state_out_ref", convStateDims, convStateOutRef, "float32");
        writeTensorFiles(cfg.outputDir, "recurrent_out_ref", recurrentDims, recurrentOutRef, "float32");
    }

    destroyRuntime(rt);
    return 0;
}
