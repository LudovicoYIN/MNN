#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "QNN/QnnBackend.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnInterface.h"
#include "QNN/QnnLog.h"
#include "QNN/QnnProperty.h"
#include "QNN/HTP/QnnHtpContext.h"
#include "QNN/System/QnnSystemContext.h"
#include "QNN/System/QnnSystemInterface.h"

#include <dlfcn.h>

using QnnInterfaceGetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
using QnnSystemInterfaceGetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);

struct Runtime {
    void* htpHandle = nullptr;
    void* systemHandle = nullptr;
    QNN_INTERFACE_VER_TYPE interface{};
    QNN_SYSTEM_INTERFACE_VER_TYPE systemInterface{};
    Qnn_LogHandle_t logHandle = nullptr;
    Qnn_BackendHandle_t backendHandle = nullptr;
    Qnn_DeviceHandle_t deviceHandle = nullptr;
};

static bool readFile(const char* path, std::vector<uint8_t>& buffer) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.good()) {
        std::printf("readFile: failed to open %s\n", path);
        return false;
    }
    auto size = input.tellg();
    if (size <= 0) {
        std::printf("readFile: invalid size for %s\n", path);
        return false;
    }
    buffer.resize((size_t)size);
    input.seekg(0, std::ios::beg);
    if (!input.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::printf("readFile: failed to read %s\n", path);
        return false;
    }
    return true;
}

static bool loadRuntime(Runtime& rt) {
    rt.htpHandle = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
    if (rt.htpHandle == nullptr) {
        std::printf("dlopen libQnnHtp.so failed: %s\n", dlerror());
        return false;
    }
    auto getProviders = reinterpret_cast<QnnInterfaceGetProvidersFn>(dlsym(rt.htpHandle, "QnnInterface_getProviders"));
    if (getProviders == nullptr) {
        std::printf("dlsym QnnInterface_getProviders failed: %s\n", dlerror());
        return false;
    }
    const QnnInterface_t** providers = nullptr;
    uint32_t providerCount = 0;
    if (QNN_SUCCESS != getProviders(&providers, &providerCount) || providers == nullptr || providerCount == 0) {
        std::printf("QnnInterface_getProviders failed: count=%u\n", providerCount);
        return false;
    }
    bool found = false;
    for (uint32_t i = 0; i < providerCount; ++i) {
        if (providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
            providers[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
            rt.interface = providers[i]->QNN_INTERFACE_VER_NAME;
            found = true;
            break;
        }
    }
    if (!found) {
        std::printf("No compatible QNN interface provider found\n");
        return false;
    }

    rt.systemHandle = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
    if (rt.systemHandle == nullptr) {
        std::printf("dlopen libQnnSystem.so failed: %s\n", dlerror());
        return false;
    }
    auto getSystemProviders = reinterpret_cast<QnnSystemInterfaceGetProvidersFn>(dlsym(rt.systemHandle, "QnnSystemInterface_getProviders"));
    if (getSystemProviders == nullptr) {
        std::printf("dlsym QnnSystemInterface_getProviders failed: %s\n", dlerror());
        return false;
    }
    const QnnSystemInterface_t** systemProviders = nullptr;
    uint32_t systemProviderCount = 0;
    if (QNN_SUCCESS != getSystemProviders(&systemProviders, &systemProviderCount) ||
        systemProviders == nullptr || systemProviderCount == 0) {
        std::printf("QnnSystemInterface_getProviders failed: count=%u\n", systemProviderCount);
        return false;
    }
    found = false;
    for (uint32_t i = 0; i < systemProviderCount; ++i) {
        if (systemProviders[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
            systemProviders[i]->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
            rt.systemInterface = systemProviders[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
            found = true;
            break;
        }
    }
    if (!found) {
        std::printf("No compatible QNN system interface provider found\n");
        return false;
    }

    if (QNN_SUCCESS != rt.interface.logCreate(nullptr, QNN_LOG_LEVEL_ERROR, &rt.logHandle) || rt.logHandle == nullptr) {
        std::printf("logCreate failed\n");
        return false;
    }
    if (QNN_SUCCESS != rt.interface.backendCreate(rt.logHandle, nullptr, &rt.backendHandle) || rt.backendHandle == nullptr) {
        std::printf("backendCreate failed\n");
        return false;
    }
    if (QNN_SUCCESS != rt.interface.deviceCreate(rt.logHandle, nullptr, &rt.deviceHandle) || rt.deviceHandle == nullptr) {
        std::printf("deviceCreate failed\n");
        return false;
    }
    return true;
}

static void freeRuntime(Runtime& rt) {
    if (rt.deviceHandle != nullptr) {
        rt.interface.deviceFree(rt.deviceHandle);
        rt.deviceHandle = nullptr;
    }
    if (rt.backendHandle != nullptr) {
        rt.interface.backendFree(rt.backendHandle);
        rt.backendHandle = nullptr;
    }
    if (rt.logHandle != nullptr) {
        rt.interface.logFree(rt.logHandle);
        rt.logHandle = nullptr;
    }
    if (rt.systemHandle != nullptr) {
        dlclose(rt.systemHandle);
        rt.systemHandle = nullptr;
    }
    if (rt.htpHandle != nullptr) {
        dlclose(rt.htpHandle);
        rt.htpHandle = nullptr;
    }
}

static void dumpBinaryInfo(Runtime& rt, const std::vector<uint8_t>& buffer) {
    QnnSystemContext_Handle_t systemContext = nullptr;
    if (QNN_SUCCESS != rt.systemInterface.systemContextCreate(&systemContext) || systemContext == nullptr) {
        std::printf("systemContextCreate failed\n");
        return;
    }
    const QnnSystemContext_BinaryInfo_t* info = nullptr;
    Qnn_ContextBinarySize_t infoSize = 0;
    auto err = rt.systemInterface.systemContextGetBinaryInfo(systemContext,
                                                             const_cast<uint8_t*>(buffer.data()),
                                                             buffer.size(),
                                                             &info,
                                                             &infoSize);
    std::printf("systemContextGetBinaryInfo err=%u size=%zu version=%u\n",
                (unsigned)(err & 0xFFFF),
                (size_t)infoSize,
                info == nullptr ? 0u : (unsigned)info->version);
    if (err == QNN_SUCCESS && info != nullptr) {
        if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
            std::printf("binary graphs=%u\n", info->contextBinaryInfoV1.numGraphs);
        } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
            std::printf("binary graphs=%u\n", info->contextBinaryInfoV2.numGraphs);
        } else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
            std::printf("binary graphs=%u\n", info->contextBinaryInfoV3.numGraphs);
        }
    }
    rt.systemInterface.systemContextFree(systemContext);
}

static void tryCreate(Runtime& rt, const std::vector<uint8_t>& buffer, const char* label, const QnnContext_Config_t** config) {
    Qnn_ContextHandle_t context = nullptr;
    auto err = rt.interface.contextCreateFromBinary(rt.backendHandle,
                                                    rt.deviceHandle,
                                                    config,
                                                    const_cast<uint8_t*>(buffer.data()),
                                                    buffer.size(),
                                                    &context,
                                                    nullptr);
    std::printf("contextCreateFromBinary[%s] err=%u ctx=%p\n", label, (unsigned)(err & 0xFFFF), context);
    if (context != nullptr) {
        rt.interface.contextFree(context, nullptr);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s /path/to/graph.bin [default|weight|weight_io|all]\n", argv[0]);
        return 1;
    }
    std::string mode = argc >= 3 ? argv[2] : "all";

    Runtime rt;
    if (!loadRuntime(rt)) {
        freeRuntime(rt);
        return 2;
    }

    std::vector<uint8_t> buffer;
    if (!readFile(argv[1], buffer)) {
        freeRuntime(rt);
        return 3;
    }

    dumpBinaryInfo(rt, buffer);

    QnnHtpContext_CustomConfig_t htpCustom[2]{};
    QnnContext_Config_t configStorage[3]{};
    const QnnContext_Config_t* configListWeight[2]{};
    const QnnContext_Config_t* configListWeightIo[3]{};

    htpCustom[0] = QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;
    htpCustom[0].option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
    htpCustom[0].weightSharingEnabled = true;
    configStorage[0] = QNN_CONTEXT_CONFIG_INIT;
    configStorage[0].option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    configStorage[0].customConfig = &htpCustom[0];
    configListWeight[0] = &configStorage[0];
    configListWeight[1] = nullptr;

    htpCustom[1] = QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;
    htpCustom[1].option = QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION;
    htpCustom[1].ioMemEstimation = true;
    configStorage[1] = QNN_CONTEXT_CONFIG_INIT;
    configStorage[1].option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    configStorage[1].customConfig = &htpCustom[1];
    configListWeightIo[0] = &configStorage[0];
    configListWeightIo[1] = &configStorage[1];
    configListWeightIo[2] = nullptr;
    if (mode == "default" || mode == "all") {
        tryCreate(rt, buffer, "default", nullptr);
    }
    if (mode == "weight" || mode == "all") {
        tryCreate(rt, buffer, "weight_sharing", configListWeight);
    }
    if (mode == "weight_io" || mode == "all") {
        tryCreate(rt, buffer, "weight_sharing+io_mem", configListWeightIo);
    }

    freeRuntime(rt);
    return 0;
}
