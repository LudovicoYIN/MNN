#include <MNN/AutoTime.hpp>
#include <MNN/MNNDefine.h>
#include <MNN_generated.h>
#include "flatbuffers/flexbuffers.h"

#include <dlfcn.h>
#include <fstream>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "QNN/QnnContext.h"
#include "QNN/QnnInterface.h"
#include "QNN/System/QnnSystemContext.h"
#include "QNN/System/QnnSystemInterface.h"
#include "QnnTypeMacros.hpp"

using namespace MNN;

namespace {

struct BinaryInfo {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

typedef Qnn_ErrorHandle_t (*QnnSystemInterface_getProviders_t)(const QnnSystemInterface_t*** providerList,
                                                               uint32_t* numProviders);

static bool readFile(const std::string& path, std::vector<uint8_t>& buffer) {
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input.good()) {
        MNN_ERROR("Failed to open file: %s\n", path.c_str());
        return false;
    }
    auto size = input.tellg();
    if (size <= 0) {
        MNN_ERROR("Invalid file size: %s\n", path.c_str());
        return false;
    }
    buffer.resize((size_t)size);
    input.seekg(0, std::ios::beg);
    if (!input.read((char*)buffer.data(), size)) {
        MNN_ERROR("Failed to read file: %s\n", path.c_str());
        return false;
    }
    return true;
}

static bool loadQnnSystemInterface(QNN_SYSTEM_INTERFACE_VER_TYPE& systemInterface, void*& systemHandle) {
    auto root = ::getenv("QNN_SDK_ROOT");
    if (root == nullptr) {
        MNN_ERROR("QNN_SDK_ROOT is not set.\n");
        return false;
    }
    std::string systemLib = std::string(root) + "/lib/x86_64-linux-clang/libQnnSystem.so";
    systemHandle = dlopen(systemLib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (systemHandle == nullptr) {
        MNN_ERROR("Failed to dlopen %s, err=%s\n", systemLib.c_str(), dlerror());
        return false;
    }
    auto getProviders = (QnnSystemInterface_getProviders_t)dlsym(systemHandle, "QnnSystemInterface_getProviders");
    if (getProviders == nullptr) {
        MNN_ERROR("Failed to resolve QnnSystemInterface_getProviders, err=%s\n", dlerror());
        return false;
    }
    const QnnSystemInterface_t** providers = nullptr;
    uint32_t providerCount = 0;
    if (QNN_SUCCESS != getProviders(&providers, &providerCount) || providers == nullptr || providerCount == 0) {
        MNN_ERROR("Failed to get QNN system interface providers.\n");
        return false;
    }
    systemInterface = providers[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
    return true;
}

template <typename T>
static void copyGraphNames(const T& info, BinaryInfo& dst) {
    for (uint32_t i = 0; i < info.numGraphInputs; ++i) {
        dst.inputs.emplace_back(QNN_TENSOR_GET_NAME(&info.graphInputs[i]));
    }
    for (uint32_t i = 0; i < info.numGraphOutputs; ++i) {
        dst.outputs.emplace_back(QNN_TENSOR_GET_NAME(&info.graphOutputs[i]));
    }
}

static bool parseBinaryInfo(const std::string& path, BinaryInfo& info) {
    std::vector<uint8_t> buffer;
    if (!readFile(path, buffer)) {
        return false;
    }
    QNN_SYSTEM_INTERFACE_VER_TYPE systemInterface{};
    void* systemHandle = nullptr;
    if (!loadQnnSystemInterface(systemInterface, systemHandle)) {
        return false;
    }
    QnnSystemContext_Handle_t systemContextHandle = nullptr;
    if (QNN_SUCCESS != systemInterface.systemContextCreate(&systemContextHandle)) {
        MNN_ERROR("Failed to create systemContext for %s\n", path.c_str());
        dlclose(systemHandle);
        return false;
    }
    const QnnSystemContext_BinaryInfo_t* binaryInfo = nullptr;
    Qnn_ContextBinarySize_t binarySize = 0;
    auto error = systemInterface.systemContextGetBinaryInfo(systemContextHandle,
                                                            buffer.data(),
                                                            buffer.size(),
                                                            &binaryInfo,
                                                            &binarySize);
    if (QNN_SUCCESS != error || binaryInfo == nullptr) {
        MNN_ERROR("Failed to get binary info for %s, error=%d\n", path.c_str(), (int)(error & 0xFFFF));
        systemInterface.systemContextFree(systemContextHandle);
        dlclose(systemHandle);
        return false;
    }
    if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1 &&
        binaryInfo->contextBinaryInfoV1.graphs != nullptr &&
        binaryInfo->contextBinaryInfoV1.numGraphs > 0) {
        copyGraphNames(binaryInfo->contextBinaryInfoV1.graphs[0].graphInfoV1, info);
    } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2 &&
               binaryInfo->contextBinaryInfoV2.graphs != nullptr &&
               binaryInfo->contextBinaryInfoV2.numGraphs > 0) {
        copyGraphNames(binaryInfo->contextBinaryInfoV2.graphs[0].graphInfoV1, info);
    } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3 &&
               binaryInfo->contextBinaryInfoV3.graphs != nullptr &&
               binaryInfo->contextBinaryInfoV3.numGraphs > 0) {
        copyGraphNames(binaryInfo->contextBinaryInfoV3.graphs[0].graphInfoV3, info);
    } else {
        MNN_ERROR("Unsupported binary info version for %s\n", path.c_str());
        systemInterface.systemContextFree(systemContextHandle);
        dlclose(systemHandle);
        return false;
    }
    systemInterface.systemContextFree(systemContextHandle);
    dlclose(systemHandle);
    return true;
}

static const Attribute* findAttr(const Plugin* plugin, const char* key) {
    if (plugin == nullptr || plugin->attr() == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < plugin->attr()->size(); ++i) {
        auto attr = plugin->attr()->GetAs<Attribute>(i);
        if (attr != nullptr && attr->key() != nullptr && attr->key()->str() == key) {
            return attr;
        }
    }
    return nullptr;
}

static std::vector<std::string> getStringList(const Attribute* attr) {
    std::vector<std::string> result;
    if (attr == nullptr || attr->list() == nullptr || attr->list()->s() == nullptr) {
        return result;
    }
    auto list = attr->list()->s();
    result.reserve(list->size());
    for (int i = 0; i < list->size(); ++i) {
        result.emplace_back(list->GetAsString(i)->str());
    }
    return result;
}

static int getStateEntryCount(const Attribute* attr) {
    if (attr == nullptr || attr->tensor() == nullptr || attr->tensor()->uint8s() == nullptr) {
        return 0;
    }
    auto ref = flexbuffers::GetRoot(attr->tensor()->uint8s()->data(), attr->tensor()->uint8s()->size());
    auto entries = ref.AsMap()["entries"];
    if (!entries.IsVector()) {
        return 0;
    }
    return entries.AsVector().size();
}

static bool matchSuffix(const std::string& name, const char* suffix) {
    auto suffixLen = ::strlen(suffix);
    if (name.size() < suffixLen) {
        return false;
    }
    return name.compare(name.size() - suffixLen, suffixLen, suffix) == 0;
}

static bool isSyntheticAlias(const std::string& alias) {
    return matchSuffix(alias, "/self_attn/Gather_output_0") || matchSuffix(alias, "/self_attn/Gather_1_output_0");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        MNN_PRINT("Usage: %s /path/to/model_dir\n", argv[0]);
        return 1;
    }
    std::string modelDir = argv[1];
    std::string shellPath = modelDir + "/qnn/llm.mnn";
    std::vector<uint8_t> shellBuffer;
    if (!readFile(shellPath, shellBuffer)) {
        return 1;
    }
    auto net = GetNet(shellBuffer.data());
    if (net == nullptr || net->oplists() == nullptr) {
        MNN_ERROR("Invalid shell model: %s\n", shellPath.c_str());
        return 1;
    }
    int pluginCount = 0;
    int mismatchCount = 0;
    for (int i = 0; i < net->oplists()->size(); ++i) {
        auto op = net->oplists()->GetAs<Op>(i);
        if (op == nullptr || op->type() != OpType_Plugin || op->main_type() != OpParameter_Plugin) {
            continue;
        }
        auto plugin = op->main_as_Plugin();
        if (plugin == nullptr) {
            continue;
        }
        auto pathAttr = findAttr(plugin, "path");
        if (pathAttr == nullptr || pathAttr->s() == nullptr) {
            continue;
        }
        auto outputs = getStringList(findAttr(plugin, "outputs"));
        auto outputAliases = getStringList(findAttr(plugin, "output_aliases"));
        auto stateEntryCount = getStateEntryCount(findAttr(plugin, "state"));
        BinaryInfo binaryInfo;
        auto relativePath = pathAttr->s()->str();
        std::string binaryPath;
        if (relativePath.find("qnn/") == 0) {
            binaryPath = modelDir + "/" + relativePath;
        } else {
            binaryPath = modelDir + "/qnn/" + relativePath;
        }
        if (!parseBinaryInfo(binaryPath, binaryInfo)) {
            return 1;
        }
        std::set<std::string> outputSet(binaryInfo.outputs.begin(), binaryInfo.outputs.end());
        int missingRealOutput = 0;
        int missingSyntheticOutput = 0;
        for (int outputIdx = 0; outputIdx < outputs.size(); ++outputIdx) {
            if (outputSet.find(outputs[outputIdx]) != outputSet.end()) {
                continue;
            }
            bool synthetic = outputIdx < outputAliases.size() && isSyntheticAlias(outputAliases[outputIdx]);
            if (synthetic) {
                ++missingSyntheticOutput;
            } else {
                ++missingRealOutput;
            }
        }
        int stateInputs = 0;
        int stateOutputs = 0;
        for (auto& name : binaryInfo.inputs) {
            if (name.find("_mnn_i") == 0) {
                ++stateInputs;
            }
        }
        for (auto& name : binaryInfo.outputs) {
            if (name.find("_mnn_o") == 0) {
                ++stateOutputs;
            }
        }
        bool ok = (missingRealOutput == 0 && stateEntryCount == stateInputs && stateEntryCount == stateOutputs);
        ++pluginCount;
        if (!ok) {
            ++mismatchCount;
        }
        MNN_PRINT("[%s] outputs=%d aliases=%d missing_real=%d missing_synthetic=%d state_entries=%d state_inputs=%d state_outputs=%d %s\n",
                  pathAttr->s()->c_str(),
                  (int)outputs.size(),
                  (int)outputAliases.size(),
                  missingRealOutput,
                  missingSyntheticOutput,
                  stateEntryCount,
                  stateInputs,
                  stateOutputs,
                  ok ? "OK" : "MISMATCH");
    }
    MNN_PRINT("plugin_count=%d mismatch_count=%d\n", pluginCount, mismatchCount);
    return mismatchCount == 0 ? 0 : 2;
}
