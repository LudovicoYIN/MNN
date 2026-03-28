//
//  QNNBackend.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNBackend.hpp"
#include "core/MNNFileUtils.h"
#include "QnnTypeMacros.hpp"
#include "HTP/QnnHtpContext.h"
#include "dsprpc_interface.h"
// #define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "core/FileLoader.hpp"
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <unordered_map>
// #define QNN_PROFILE_OP
// #define QNN_PROFILE_SUMMARIZE
// #define QNN_VERBOSE
#ifdef ENABLE_QNN_CONVERT_MODE
#define QNN_FORWARD_TYPE MNN_CONVERT_QNN
#else
#define QNN_FORWARD_TYPE MNN_FORWARD_NN
#endif

namespace MNN {
static std::string gExtraIoPrefix = "_mnn";
namespace QNN {

class OnlineRPCBuffer {
public:
    ~OnlineRPCBuffer() {
        if (mPtr != nullptr) {
            if (mUseRpcMem) {
                rpcmem_free(mPtr);
            } else {
                ::free(mPtr);
            }
        }
    }

    static std::shared_ptr<OnlineRPCBuffer> alloc(size_t size) {
        void* data = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, size);
        bool useRpcMem = data != nullptr;
        if (!useRpcMem) {
            MNN_PRINT("MNN_QNN: rpcmem_alloc unavailable for extra state buffer, fallback to host buffer, size=%zu\n", size);
            data = ::malloc(size);
            if (data == nullptr) {
                MNN_ERROR("host malloc failed for extra state buffer, size=%zu\n", size);
                return nullptr;
            }
        }
        int fd = -1;
        if (useRpcMem) {
            fd = rpcmem_to_fd(data);
            if (fd == -1) {
                MNN_ERROR("rpcmem_to_fd failed for extra state buffer, size=%zu\n", size);
                rpcmem_free(data);
                return nullptr;
            }
        }
        return std::shared_ptr<OnlineRPCBuffer>(new OnlineRPCBuffer(data, fd, size, useRpcMem));
    }

    bool bind(Qnn_Tensor_t* tensor, QNN_INTERFACE_VER_TYPE* interface, Qnn_ContextHandle_t context) {
        if (!mUseRpcMem) {
            QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_RAW);
            Qnn_ClientBuffer_t clientBuf = {mPtr, static_cast<uint32_t>(mSize)};
            QNN_TENSOR_SET_CLIENT_BUF(tensor, clientBuf);
            return true;
        }
        if (!mRegistered) {
            Qnn_MemDescriptor_t memDescriptor = {
                {QNN_TENSOR_GET_RANK(tensor), QNN_TENSOR_GET_DIMENSIONS(tensor), nullptr},
                QNN_TENSOR_GET_DATA_TYPE(tensor),
                QNN_MEM_TYPE_ION,
                {{-1}}};
            memDescriptor.ionInfo.fd = mFd;
            auto res = interface->memRegister(context, &memDescriptor, 1, &mHandle);
            if (res != QNN_SUCCESS) {
                MNN_ERROR("memRegister failed for extra state tensor %s, error=%llu\n",
                          QNN_TENSOR_GET_NAME(tensor), (unsigned long long)res);
                return false;
            }
            mRegistered = true;
        }
        QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
        QNN_TENSOR_SET_MEM_HANDLE(tensor, mHandle);
        return true;
    }

    void zero() {
        if (mPtr != nullptr) {
            ::memset(mPtr, 0, mSize);
        }
    }

    void* ptr() const {
        return mPtr;
    }

private:
    OnlineRPCBuffer(void* ptr, int fd, size_t size, bool useRpcMem) : mPtr(ptr), mFd(fd), mSize(size), mUseRpcMem(useRpcMem) {
    }

private:
    void* mPtr = nullptr;
    int mFd = -1;
    size_t mSize = 0;
    bool mUseRpcMem = false;
    bool mRegistered = false;
    Qnn_MemHandle_t mHandle = nullptr;
};

struct QnnContext {
    QNN_INTERFACE_VER_TYPE interface{};
    QNN_SYSTEM_INTERFACE_VER_TYPE systemInterface{};
    Qnn_LogHandle_t logHandle = nullptr;
    Qnn_BackendHandle_t backendHandle = nullptr;
    Qnn_DeviceHandle_t deviceHandle = nullptr;
    int soc_id;
    int dsp_arch;
};

static QnnContext gContext;
static std::mutex gQnnContextMutex;

static void createQnnContext(){
    std::lock_guard<std::mutex> lck(gQnnContextMutex);
    QNN_INTERFACE_VER_TYPE qnnInterface{};
#ifndef ENABLE_QNN_CONVERT_MODE
    {
        QnnInterface_t** interfaceProviders = nullptr;
        uint32_t numProviders = 0;
        if (QNN::QnnInterface_getProviders((const QnnInterface_t***)&interfaceProviders, &numProviders) != QNN_SUCCESS) {
            MNN_PRINT("MNN_QNN: Failed to call 'QnnInterface_getProviders'.\n");
            return;
        }
        if (interfaceProviders == nullptr) {
            MNN_PRINT("MNN_QNN: Failed to get interface providers: null interface providers received.\n");
            return;
        }
        if (numProviders == 0) {
            MNN_PRINT("MNN_QNN: Failed to get interface providers: 0 interface providers.\n");
            return;
        }
        bool foundValidInterface = false;
        for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
            if (QNN_API_VERSION_MAJOR == interfaceProviders[pIdx]->apiVersion.coreApiVersion.major &&
                QNN_API_VERSION_MINOR <= interfaceProviders[pIdx]->apiVersion.coreApiVersion.minor) {
                foundValidInterface = true;
                qnnInterface = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
                break;
            }
        }
        if (!foundValidInterface) {
            MNN_PRINT("MNN_QNN: Failed to find a valid interface.\n");
            return;
        }
    }
#else
    qnnInterface = QNN::gQnnConvertorInterface;
#endif

    // Create Log.
    Qnn_LogHandle_t logHandle = nullptr;
    {
        QnnLog_Callback_t logCallback = nullptr;
        if ((QNN_GET_ERROR_CODE(qnnInterface.logCreate(logCallback, QNN_LOG_LEVEL_ERROR, &logHandle)) != QNN_SUCCESS) ||
            (logHandle == nullptr)) {
            MNN_PRINT("MNN_QNN: Failed to initialize logging in the backend.\n");
            return;
        }
    }

    // Create Backend.
    Qnn_BackendHandle_t backendHandle = nullptr;
    {
        const QnnBackend_Config_t** backendConfig = nullptr;
        auto backendStatus = qnnInterface.backendCreate(logHandle, backendConfig, &backendHandle);
        if ((QNN_GET_ERROR_CODE(backendStatus) != QNN_SUCCESS) ||
            (backendHandle == nullptr)) {
            MNN_PRINT("MNN_QNN: Failed to create the backend.\n");
            return;
        }
    }

    // Create Device.
    Qnn_DeviceHandle_t deviceHandle = nullptr;
    QnnHtpDevice_Arch_t dspArch = QNN_HTP_DEVICE_ARCH_NONE;
    uint32_t socId = 0;
    {
        // Check whether the device API is supported.
        bool supportDevice = QNN::checkCapability(qnnInterface, QNN_PROPERTY_GROUP_DEVICE);
        if (supportDevice) {
            const QnnDevice_Config_t ** deviceConfig = nullptr;
            auto qnnStatus = qnnInterface.deviceCreate(logHandle, deviceConfig, &deviceHandle);
            if(qnnStatus != QNN_SUCCESS || (deviceHandle == nullptr)) {
                MNN_PRINT("MNN_QNN: Failed to create the device, error:%lu\n", (unsigned long)qnnStatus);
                return;
            }

            if (qnnInterface.deviceGetPlatformInfo == nullptr) {
                MNN_PRINT("[Warning]: No QnnDevice_getPlatformInfo API");
            } else {
                // QnnDevice_PlatformInfo_t platformInfo = QNN_DEVICE_PLATFORM_INFO_INIT;
                const QnnDevice_PlatformInfo_t* backendPlatformInfoPtr = nullptr;
                qnnStatus = qnnInterface.deviceGetPlatformInfo(logHandle, &backendPlatformInfoPtr);
                if(qnnStatus != QNN_SUCCESS || backendPlatformInfoPtr == nullptr) {
                    MNN_PRINT("[Warning]: deviceGetPlatformInfo Failed to query platform info");
                } else {
                    QnnDevice_HardwareDeviceInfo_t* hwDeviceInfo = backendPlatformInfoPtr->v1.hwDevices;
                    dspArch = hwDeviceInfo->v1.deviceInfoExtension->onChipDevice.arch;
                    socId = hwDeviceInfo->v1.deviceInfoExtension->onChipDevice.socModel;
                }
            }
        } else {
            MNN_PRINT("MNN_QNN: Not supporting device API.\n");
            return;
        }
    }

    // Create System Interface
    QNN_SYSTEM_INTERFACE_VER_TYPE systemInterface{};
#ifndef ENABLE_QNN_CONVERT_MODE
    #ifdef MNN_WITH_PLUGIN
    {
        QnnSystemInterface_t** interfaceProviders = nullptr;
        uint32_t numProviders = 0;
        if (QNN::QnnSystemInterface_getProviders((const QnnSystemInterface_t***)&interfaceProviders, &numProviders) != QNN_SUCCESS) {
            MNN_PRINT("MNN_QNN: Failed to call 'QnnInterface_getProviders'.\n");
            return;
        }
        if (interfaceProviders == nullptr) {
            MNN_PRINT("MNN_QNN: Failed to get interface providers: null interface providers received.\n");
            return;
        }
        if (numProviders == 0) {
            MNN_PRINT("MNN_QNN: Failed to get interface providers: 0 interface providers.\n");
            return;
        }
        bool foundValidSystemInterface{false};
        for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
            if (QNN_SYSTEM_API_VERSION_MAJOR == interfaceProviders[pIdx]->systemApiVersion.major &&
                QNN_SYSTEM_API_VERSION_MINOR <= interfaceProviders[pIdx]->systemApiVersion.minor) {
                foundValidSystemInterface = true;
                systemInterface = interfaceProviders[pIdx]->QNN_SYSTEM_INTERFACE_VER_NAME;
                break;
            }
        }
        if (!foundValidSystemInterface) {
            MNN_PRINT("MNN_QNN: Failed to find a valid interface.\n");
            return;
        }
    }
    #endif
#else
    systemInterface = QNN::gQnnConvertorSystemInterface;
#endif


    QNN::gContext.interface = qnnInterface;
    QNN::gContext.systemInterface = systemInterface;
    QNN::gContext.backendHandle = backendHandle;
    QNN::gContext.deviceHandle = deviceHandle;
    QNN::gContext.logHandle = logHandle;
    QNN::gContext.soc_id = socId;
    QNN::gContext.dsp_arch = dspArch;
}

static bool ensureQnnContextReady() {
    if (QNN::gContext.backendHandle != nullptr && QNN::gContext.deviceHandle != nullptr) {
        return true;
    }
    QNN::createQnnContext();
    return QNN::gContext.backendHandle != nullptr && QNN::gContext.deviceHandle != nullptr;
}

#ifdef QNN_PROFILE_SUMMARIZE
static std::string getOpTypeFromName(const std::string& nodeName) {
    // The pattern is usually "OpType_..."
    size_t pos = nodeName.find('_');
    if (pos != std::string::npos) {
        return nodeName.substr(0, pos);
    }
    // Fallback for names without '_', like "Input OpId_2 (cycles)"
    pos = nodeName.find(' ');
    if (pos != std::string::npos) {
        return nodeName.substr(0, pos);
    }
    // If no delimiter is found, return the whole name as the type
    return nodeName;
}
#endif

static void createProfileHandle(const QNN_INTERFACE_VER_TYPE& interface, const Qnn_BackendHandle_t& backend_handle, Qnn_ProfileHandle_t* profile_handle_ptr) {
    #if defined(QNN_PROFILE_SUMMARIZE) || defined(QNN_PROFILE_OP)
    if (*profile_handle_ptr == nullptr) {
        // set QNN_PROFILE_LEVEL_DETAILED
        QnnProfile_Level_t profileLevel = QNN_PROFILE_LEVEL_DETAILED;
        MNN_PRINT("[QNN Profile] Creating QNN Profile Handle with DETAILED level.\n");
        auto profile_err = interface.profileCreate(backend_handle, profileLevel, profile_handle_ptr);
        if (profile_err != QNN_SUCCESS || *profile_handle_ptr == nullptr) {
            MNN_ERROR("[QNN Profile] Failed to create QNN Profile Handle, error: %d\n", (int)profile_err);
            *profile_handle_ptr = nullptr;
        }
    }
    #endif
}

static void doProfile(const QNN_INTERFACE_VER_TYPE& interface, const Qnn_ProfileHandle_t& profile_handle) {
#ifdef QNN_PROFILE_OP
    if (profile_handle) {
        uint32_t numTopLevelEvents = 0;
        const QnnProfile_EventId_t* topLevelEvents = nullptr;

        auto get_err = interface.profileGetEvents(profile_handle, &topLevelEvents, &numTopLevelEvents);
        if (get_err != QNN_SUCCESS) {
            MNN_PRINT("[QNN Profile] Failed to get top-level events. Error: %d\n", (int)get_err);
            return;
        }

        MNN_PRINT("\n--- QNN Node-level Performance Report ---\n");
        bool foundNodeData = false;

        for (uint32_t i = 0; i < numTopLevelEvents; ++i) {
            QnnProfile_EventData_t eventData = QNN_PROFILE_EVENT_DATA_INIT;
            interface.profileGetEventData(topLevelEvents[i], &eventData);

            if (eventData.type) {
                MNN_PRINT("Found EXECUTE event. Total time: %llu us. Querying sub-events...\n", (unsigned long long)eventData.value);

                uint32_t numSubEvents = 0;
                const QnnProfile_EventId_t* subEvents = nullptr;

                // 3. GetSubEvents
                auto get_sub_err = interface.profileGetSubEvents(topLevelEvents[i], &subEvents, &numSubEvents);
                if (get_sub_err != QNN_SUCCESS) {
                    MNN_PRINT("[QNN Profile] Failed to get sub-events for EXECUTE event. Error: %d\n", (int)get_sub_err);
                    continue;
                }

                for (uint32_t j = 0; j < numSubEvents; ++j) {
                    QnnProfile_EventData_t subEventData = QNN_PROFILE_EVENT_DATA_INIT;
                    interface.profileGetEventData(subEvents[j], &subEventData);

                    if (subEventData.type == QNN_PROFILE_EVENTTYPE_NODE) {
                        foundNodeData = true;
                        const char* nodeName = subEventData.identifier;
                        uint64_t value = subEventData.value;

                        switch (subEventData.unit) {
                            case QNN_PROFILE_EVENTUNIT_MICROSEC:
                                MNN_PRINT("Node: %-45s | Time: %10llu us (%.3f ms)\n",
                                        nodeName, (unsigned long long)value, (double)value / 1000.0);
                                break;
                            case QNN_PROFILE_EVENTUNIT_CYCLES:
                                MNN_PRINT("Node: %-45s | Cycles: %.2f*10^6\n", nodeName, (double)value / 1000000.0);
                                break;
                            // ... other dealing ...
                            default:
                                MNN_PRINT("Node: %-45s | Value: %10llu (Unit: %u - Unknown)\n",
                                        nodeName, (unsigned long long)value, subEventData.unit);
                                break;
                        }
                    }
                }
            }
        }

        if (!foundNodeData) {
            MNN_PRINT("No node-specific performance data found. Please ensure you have set:\n");
            MNN_PRINT("1. Profile level to QNN_PROFILE_LEVEL_DETAILED.\n");
            MNN_PRINT("2. HTP graph config with QNN_HTP_GRAPH_CONFIG_OPTION_PERF_PROFILE (if available).\n");
        }
        MNN_PRINT("-----------------------------------------\n");
    }
#endif

#ifdef QNN_PROFILE_SUMMARIZE
    if (profile_handle) {
        std::map<std::string, uint64_t> opCycleStats;
        uint64_t totalNodeCycles = 0;

        uint32_t numTopLevelEvents = 0;
        const QnnProfile_EventId_t* topLevelEvents = nullptr;

        auto get_err = interface.profileGetEvents(profile_handle, &topLevelEvents, &numTopLevelEvents);
        if (get_err != QNN_SUCCESS) {
            MNN_PRINT("[QNN Profile] Failed to get top-level events. Error: %d\n", (int)get_err);
            return;
        }

        for (uint32_t i = 0; i < numTopLevelEvents; ++i) {
            QnnProfile_EventData_t eventData = QNN_PROFILE_EVENT_DATA_INIT;
            interface.profileGetEventData(topLevelEvents[i], &eventData);

            if (eventData.type) { // == QNN_PROFILE_EVENTTYPE_EXECUTE) {
                uint32_t numSubEvents = 0;
                const QnnProfile_EventId_t* subEvents = nullptr;
                auto get_sub_err = interface.profileGetSubEvents(topLevelEvents[i], &subEvents, &numSubEvents);
                if (get_sub_err != QNN_SUCCESS) continue;

                for (uint32_t j = 0; j < numSubEvents; ++j) {
                    QnnProfile_EventData_t subEventData = QNN_PROFILE_EVENT_DATA_INIT;
                    interface.profileGetEventData(subEvents[j], &subEventData);

                    if (subEventData.type == QNN_PROFILE_EVENTTYPE_NODE) {
                        if (subEventData.identifier) {
                            std::string opType = getOpTypeFromName(subEventData.identifier);
                            opCycleStats[opType] += subEventData.value;
                            totalNodeCycles += subEventData.value;
                        }
                    }
                }
            }
        }

        if (!opCycleStats.empty()) {
            MNN_PRINT("\n--- QNN Operator-wise Performance Summary ---\n");
            MNN_PRINT("%-20s | %15s | %s\n", "Operator Type", "Total Cycles", "Percentage");
            MNN_PRINT("--------------------------------------------------\n");

            std::vector<std::pair<std::string, uint64_t>> sortedStats(opCycleStats.begin(), opCycleStats.end());
            std::sort(sortedStats.begin(), sortedStats.end(), [](const std::pair<std::string, uint64_t>& a, const std::pair<std::string, uint64_t>& b) {
                return a.second > b.second; // sort by large -> small
            });

            for (const auto& pair : sortedStats) {
                double percentage = (totalNodeCycles > 0) ? ((double)pair.second * 100.0 / totalNodeCycles) : 0.0;
                MNN_PRINT("%-20s | %15llu | %.2f%%\n", pair.first.c_str(), pair.second, percentage);
            }
            MNN_PRINT("--------------------------------------------------\n");
            MNN_PRINT("%-20s | %15llu | 100.00%%\n", "Total", totalNodeCycles);
        }
    }
    // =========================================================
#endif
}
}
}

#ifdef MNN_WITH_PLUGIN

#include "MNN/plugin/PluginShapeInference.hpp"
#include "MNN/plugin/PluginContext.hpp"
#include "MNN/plugin/PluginKernel.hpp"
#include "shape/SizeComputer.hpp"
#include "flatbuffers/flexbuffers.h"
#include "core/OpCommonUtils.hpp"
#include "dsprpc_interface.h"

namespace MNN {
namespace plugin {

class RPCBuffer {
public:
    void* mPtr = nullptr;
    size_t mSize;
    int mFd;
    bool mReg = false;
    Qnn_MemHandle_t mHandle;
    ~ RPCBuffer() {
        rpcmem_free(mPtr);
    }
    static RPCBuffer* alloc(size_t size) {
        void * data = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, size);
        if (nullptr == data) {
            FUNC_PRINT(1);
            return nullptr;
        }
        auto fd = rpcmem_to_fd(data);
        if (fd == -1) {
            FUNC_PRINT(1);
            rpcmem_free(data);
            return nullptr;
        }
        return new RPCBuffer(data, fd, size);
    }
    bool setToTensor(Qnn_Tensor_t* tensor, QNN_INTERFACE_VER_TYPE* interface, Qnn_ContextHandle_t context) {
        if (!mReg) {
            Qnn_MemDescriptor_t memDescriptor = {
                {QNN_TENSOR_GET_RANK(tensor), QNN_TENSOR_GET_DIMENSIONS(tensor), nullptr},
                QNN_TENSOR_GET_DATA_TYPE(tensor),
                QNN_MEM_TYPE_ION,
                {{-1}}};
            int curFd = mFd;
            memDescriptor.ionInfo.fd = curFd;
            QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
            QNN_TENSOR_SET_MEM_HANDLE(tensor, nullptr);

            mHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
            auto res = interface->memRegister(context, &memDescriptor, 1, &(mHandle));
            if (res != QNN_SUCCESS) {
                const char* tname = QNN_TENSOR_GET_NAME(tensor);
                MNN_ERROR("memRegister fail %s (ctx=%p fd=%d), error: %llu\n", tname, context, curFd, res);
                return false;
            }
            mReg = true;
        }
        QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
        QNN_TENSOR_SET_MEM_HANDLE(tensor, mHandle);
        return true;
    }
private:
    RPCBuffer(void* ptr, int fd, size_t size) {
        mPtr = ptr;
        mFd = fd;
        mSize = size;
    }
};

namespace shape_inference {
class PluginShapeRaw : public InferShapeKernel {
public:
    bool compute(InferShapeContext* ctx) override;
};
static bool computeIndex(PluginContext* ctx, int & index) {
    const std::vector<Tensor *> & inputs = ctx->inputs();
    auto attrAllShape = ctx->getAttr("allInputShape");
    if (nullptr == attrAllShape || nullptr == attrAllShape->list() || nullptr == attrAllShape->list()->i()) {
        MNN_ERROR("MNN_QNN: Incorrect Plugin Op, can't find 'allInputShape' attr.\n");
        return false;
    }
    bool dumpShape = (::getenv("MNN_QNN_DUMP_PLUGIN_SHAPE") != nullptr);
    int dimSum = 0;
    for (int i = 0; i < inputs.size(); i++) {
        auto inputDim = inputs[i]->dimensions();
        dimSum += inputDim;
        if (dumpShape) {
            std::ostringstream os;
            os << "[";
            for (int j = 0; j < inputDim; ++j) {
                if (j > 0) {
                    os << ",";
                }
                os << inputs[i]->length(j);
            }
            os << "]";
            MNN_ERROR("MNN_QNN_SHAPE input[%d]=%s dims=%d\n", i, os.str().c_str(), inputDim);
        }
    }
    if (0 == dimSum) {
        // Scalar
        index = 0;
        return true;
    }
    auto indexNumber = attrAllShape->list()->i()->size() / dimSum;
    for (int si=0; si<indexNumber; ++si) {
        auto dstSi = attrAllShape->list()->i()->data() + si * dimSum;
        if (dumpShape) {
            std::ostringstream os;
            os << "[";
            for (int j = 0; j < dimSum; ++j) {
                if (j > 0) {
                    os << ",";
                }
                os << dstSi[j];
            }
            os << "]";
            MNN_ERROR("MNN_QNN_SHAPE candidate[%d]=%s\n", si, os.str().c_str());
        }
        bool valid = true;
        for (int i=0; i<inputs.size(); ++i) {
            auto inputDim = inputs[i]->dimensions();
            for (int j = 0; j < inputDim; j++) {
                if (inputs[i]->length(j) != dstSi[j]) {
                    valid = false;
                    break;
                }
            }
            dstSi += inputDim;
            if (!valid) {
                break;
            }
        }
        if (valid) {
            index = si;
            if (dumpShape) {
                MNN_ERROR("MNN_QNN_SHAPE matched=%d\n", si);
            }
            return true;
        }
    }
    if (dumpShape) {
        MNN_ERROR("MNN_QNN_SHAPE no_match dimSum=%d candidates=%d\n", dimSum, indexNumber);
    }
    return false;
}

bool PluginShapeRaw::compute(InferShapeContext* ctx) {
    if (ctx->hasAttr("op")) {
        if (::getenv("MNN_QNN_DUMP_PLUGIN_SHAPE") != nullptr) {
            MNN_ERROR("MNN_QNN_SHAPE using_embedded_op_shape=1\n");
        }
        auto attr = ctx->getAttr("op");
        if (nullptr != attr->tensor() && nullptr != attr->tensor()->int8s()) {
            auto realop = flatbuffers::GetRoot<Op>(attr->tensor()->int8s()->data());
            return SizeComputer::computeOutputSize(realop, ctx->inputs(), ctx->outputs());
        }
    } else {
        int shapeIndex = 0;
        if (!(computeIndex(ctx, shapeIndex))) {
            MNN_ERROR("MNN_QNN: Failed to compute shape for Plugin Op.\n");
            return false;
        }

        std::string prefix = "o_" + std::to_string(shapeIndex) + "_";
        for (int i=0; i<ctx->outputs().size(); ++i) {
            auto dst = ctx->output(i);
            std::string key = prefix + std::to_string(i);
            auto attr = ctx->getAttr(key.c_str());

            if (nullptr == attr || nullptr == attr->tensor()) {
                MNN_ERROR("MNN_QNN: Failed to find raw shape %s.\n", key.c_str());
                return false;
            }
            auto blob = attr->tensor();
            dst->setType(blob->dataType());
            if (nullptr != blob->dims()) {
                dst->buffer().dimensions = blob->dims()->size();
                for (int j=0; j<blob->dims()->size(); ++j) {
                    dst->setLength(j, blob->dims()->data()[j]);
                }
            } else {
                dst->buffer().dimensions = 0;
            }
            TensorUtils::getDescribe(dst)->dimensionFormat = blob->dataFormat();
        }
        return true;
    }
    return false;
}
}

namespace backend {
static bool freeQnnTensor(Qnn_Tensor_t &tensor) {
  // free all pointer allocations in struct
  free((void *)QNN_TENSOR_GET_NAME(tensor));
  free(QNN_TENSOR_GET_DIMENSIONS(tensor));
  free(QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(tensor));

  auto quant    = QNN_TENSOR_GET_QUANT_PARAMS(tensor);
  auto encoding = quant.quantizationEncoding;
  if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    free(quant.axisScaleOffsetEncoding.scaleOffset);
  } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
    free(quant.bwAxisScaleOffsetEncoding.scales);
    if (quant.bwAxisScaleOffsetEncoding.offsets != nullptr) {
      free(quant.bwAxisScaleOffsetEncoding.offsets);
    }
  }
  return true;
}

static bool freeQnnTensors(Qnn_Tensor_t *&tensors, uint32_t numTensors) {
  // free all pointer allocations in struct
  for (size_t i = 0; i < numTensors; i++) {
    freeQnnTensor(tensors[i]);
  }
  free(tensors);

  return true;
}

struct GraphInfo {
  Qnn_GraphHandle_t graph;
  char *graphName;
  Qnn_Tensor_t *inputTensors;
  uint32_t numInputTensors;
  Qnn_Tensor_t *outputTensors;
  uint32_t numOutputTensors;
};

static bool deepCopyQnnTensorInfo(Qnn_Tensor_t *dst, const Qnn_Tensor_t *src) {
  if (nullptr == dst || nullptr == src) {
    return false;
  }
  // set tensor.version before using QNN_TENSOR_SET macros, as they require the version to be set
  // to correctly assign values
  dst->version           = src->version;
  const char *tensorName = QNN_TENSOR_GET_NAME(src);
  if (!tensorName) {
    QNN_TENSOR_SET_NAME(dst, nullptr);
  } else {
    QNN_TENSOR_SET_NAME(dst, ::strdup(tensorName));
  }
  QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
  QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
  QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
  QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));
  dst->v1.memType = QNN_TENSORMEMTYPE_RAW;
  Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
  qParams.encodingDefinition   = QNN_TENSOR_GET_QUANT_PARAMS(src).encodingDefinition;
  qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
    qParams.scaleOffsetEncoding  = QNN_TENSOR_GET_QUANT_PARAMS(src).scaleOffsetEncoding;
  } else if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding ==
             QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
    qParams.axisScaleOffsetEncoding.axis =
        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.axis;
    qParams.axisScaleOffsetEncoding.numScaleOffsets =
        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
    if (QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets > 0) {
      qParams.axisScaleOffsetEncoding.scaleOffset = (Qnn_ScaleOffset_t *)malloc(
          QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets *
          sizeof(Qnn_ScaleOffset_t));
      if (qParams.axisScaleOffsetEncoding.scaleOffset) {
        for (size_t idx = 0;
             idx < QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
             idx++) {
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale =
              QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].scale;
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset =
              QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].offset;
        }
      }
    }
  }
  QNN_TENSOR_SET_QUANT_PARAMS(dst, qParams);
  QNN_TENSOR_SET_RANK(dst, QNN_TENSOR_GET_RANK(src));
  QNN_TENSOR_SET_DIMENSIONS(dst, nullptr);
  if (QNN_TENSOR_GET_RANK(src) > 0) {
    QNN_TENSOR_SET_DIMENSIONS(dst, (uint32_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)));
    if (QNN_TENSOR_GET_DIMENSIONS(dst)) {
      ::memcpy(QNN_TENSOR_GET_DIMENSIONS(dst),
                             QNN_TENSOR_GET_DIMENSIONS(src),
                             QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t));
    }
    if (QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src)) {
      QNN_TENSOR_SET_IS_DYNAMIC_DIMENSIONS(
          dst, (uint8_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t)));
      ::memcpy(QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(dst),
                             QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src),
                             QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t));
    }
  }
  QNN_TENSOR_SET_SPARSE_PARAMS(dst, QNN_TENSOR_GET_SPARSE_PARAMS(src));
  return true;
}

static bool copyTensorsInfo(const Qnn_Tensor_t *tensorsInfoSrc,
                                 Qnn_Tensor_t *&tensorWrappers,
                                 uint32_t tensorsCount) {
  auto returnStatus = true;
  tensorWrappers    = (Qnn_Tensor_t *)calloc(tensorsCount, sizeof(Qnn_Tensor_t));
  if (nullptr == tensorWrappers) {
    MNN_ERROR("Failed to allocate memory for tensorWrappers.");
    return false;
  }
  if (returnStatus) {
    for (size_t tIdx = 0; tIdx < tensorsCount; tIdx++) {
      #ifdef QNN_VERBOSE
      MNN_PRINT("Extracting tensorInfo for tensor Idx: %d.\n", (int) tIdx);
      #endif
      tensorWrappers[tIdx] = QNN_TENSOR_INIT;
      deepCopyQnnTensorInfo(&tensorWrappers[tIdx], &tensorsInfoSrc[tIdx]);
    }
  }
  return returnStatus;
}

template <typename T>
static bool copyGraphsInfoFromSrc(const T *graphInfoSrc, GraphInfo *graphInfoDst) {
  graphInfoDst->graphName = nullptr;
  if (graphInfoSrc->graphName) {
    graphInfoDst->graphName = ::strdup(graphInfoSrc->graphName);
  }
  graphInfoDst->inputTensors    = nullptr;
  graphInfoDst->numInputTensors = 0;
  if (graphInfoSrc->graphInputs) {
    if (!copyTensorsInfo(
            graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) {
      return false;
    }
    graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
  }
  graphInfoDst->outputTensors    = nullptr;
  graphInfoDst->numOutputTensors = 0;
  if (graphInfoSrc->graphOutputs) {
    if (!copyTensorsInfo(graphInfoSrc->graphOutputs,
                         graphInfoDst->outputTensors,
                         graphInfoSrc->numGraphOutputs)) {
      return false;
    }
    graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
  }
  return true;
}

static bool copyGraphsInfo(const QnnSystemContext_GraphInfo_t *graphsInput,
                                const uint32_t numGraphs,
                                GraphInfo **&graphsInfo) {
  if (!graphsInput) {
    MNN_ERROR("Received nullptr for graphsInput.");
    return false;
  }
  auto returnStatus = true;
  graphsInfo =
      (GraphInfo **)calloc(numGraphs, sizeof(GraphInfo *));
  GraphInfo *graphInfoArr =
      (GraphInfo *)calloc(numGraphs, sizeof(GraphInfo));
  if (nullptr == graphsInfo || nullptr == graphInfoArr) {
    MNN_ERROR("Failure to allocate memory for *graphInfo");
    returnStatus = false;
  }
  if (true == returnStatus) {
    for (size_t gIdx = 0; gIdx < numGraphs; gIdx++) {
      #ifdef QNN_VERBOSE
      MNN_PRINT("Extracting graphsInfo for graph Idx: %d", (int) gIdx);
      #endif
      if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        copyGraphsInfoFromSrc(&graphsInput[gIdx].graphInfoV1, &graphInfoArr[gIdx]);
      } else if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        copyGraphsInfoFromSrc(&graphsInput[gIdx].graphInfoV3, &graphInfoArr[gIdx]);
      }
      graphsInfo[gIdx] = graphInfoArr + gIdx;
    }
  }
  if (true != returnStatus) {
    MNN_ERROR("Received an ERROR during extractGraphsInfo. Freeing resources.");
    if (graphsInfo) {
      for (uint32_t gIdx = 0; gIdx < numGraphs; gIdx++) {
        if (graphsInfo[gIdx]) {
          if (nullptr != graphsInfo[gIdx]->graphName) {
            free(graphsInfo[gIdx]->graphName);
            graphsInfo[gIdx]->graphName = nullptr;
          }
          freeQnnTensors(graphsInfo[gIdx]->inputTensors,
                                          graphsInfo[gIdx]->numInputTensors);
          freeQnnTensors(graphsInfo[gIdx]->outputTensors,
                                          graphsInfo[gIdx]->numOutputTensors);
        }
      }
      free(*graphsInfo);
    }
    free(graphsInfo);
    graphsInfo = nullptr;
  }
  return true;
}

template<typename T>
static bool copyGraphsInfoFromBinaryInfo(const T & binaryInfo, GraphInfo **& graphsInfo, uint32_t & graphsCount) {
    if (binaryInfo.graphs) {
        if (!copyGraphsInfo(binaryInfo.graphs, binaryInfo.numGraphs, graphsInfo)) {
            MNN_ERROR("MNN_QNN: Failed while copying graphs Info.\n");
            return false;
        }
        graphsCount = binaryInfo.numGraphs;
        return true;
    }
    return false;
}

static bool copyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t *binaryInfo,
                                          GraphInfo **&graphsInfo,
                                          uint32_t &graphsCount) {
    if (nullptr == binaryInfo) {
        MNN_ERROR("MNN_QNN: binaryInfo is nullptr.\n");
        return false;
    }
    graphsCount = 0;
    switch (binaryInfo->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
            return copyGraphsInfoFromBinaryInfo(binaryInfo->contextBinaryInfoV1, graphsInfo, graphsCount);
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
            return copyGraphsInfoFromBinaryInfo(binaryInfo->contextBinaryInfoV2, graphsInfo, graphsCount);
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
            return copyGraphsInfoFromBinaryInfo(binaryInfo->contextBinaryInfoV3, graphsInfo, graphsCount);
        default:
            MNN_ERROR("MNN_QNN: Unrecognized system context binary info version.\n");
            return false;
    }
}

static bool freeGraphsInfo(GraphInfo ***graphsInfo, uint32_t numGraphs) {
  if (graphsInfo == nullptr || *graphsInfo == nullptr) {
    return false;
  }
  for (uint32_t i = 0; i < numGraphs; i++) {
    free((*graphsInfo)[i]->graphName);
    freeQnnTensors((*graphsInfo)[i]->inputTensors, (*graphsInfo)[i]->numInputTensors);
    freeQnnTensors((*graphsInfo)[i]->outputTensors, (*graphsInfo)[i]->numOutputTensors);
  }
  free(**graphsInfo);
  free(*graphsInfo);
  *graphsInfo = nullptr;
  return true;
}

static void syncTensorShapeFromMNN(Qnn_Tensor_t& dst, const MNN::Tensor* src) {
  if (src == nullptr) {
    return;
  }
  auto dims = src->dimensions();
  auto dstRank = (int)QNN_TENSOR_GET_RANK(&dst);
  auto dstDims = QNN_TENSOR_GET_DIMENSIONS(&dst);
  if (dstRank != dims || dstDims == nullptr) {
    return;
  }
  for (int i = 0; i < dims; ++i) {
    dstDims[i] = src->length(i);
  }
}

class MMapReader {
private:
    void* mAddr = nullptr;
    file_t mFile = INVALID_FILE;
    size_t mSize = 0;
    void _clean() {
        if (nullptr != mAddr) {
            MNNUnmapFile(mAddr, mSize);
            mAddr = nullptr;
        }
        if (mFile != INVALID_FILE) {
            MNNCloseFile(mFile);
            mFile = INVALID_FILE;
        }
        mSize = 0;
    }
public:
    void* addr() const {
        return mAddr;
    }
    size_t size() const {
        return mSize;
    }
    MMapReader() {
        // Do nothing
    }
    ~MMapReader() {
        _clean();
    }
    bool open(const char* filename) {
        _clean();
        mFile = MNNOpenFile(filename, MNN_FILE_READ);
        mSize = MNNGetFileSize(mFile);
        mAddr = MNNMmapFile(mFile, mSize, true);
        return true;
    }
};

class RawExecutorWrapper {
private:
    Qnn_ContextHandle_t mQnnContextHandle = nullptr;
    const QnnContext_Config_t** mQnnContextConfig = nullptr;
    QnnHtpContext_CustomConfig_t mQnnHtpContextCustomConfigs[2]{};
    QnnContext_Config_t mQnnContextConfigStorage[3]{};
    const QnnContext_Config_t* mQnnContextConfigList[4] = {nullptr, nullptr, nullptr, nullptr};
    std::vector<Qnn_GraphHandle_t> mQnnGraphHandleVec = {};
    QnnHtpGraph_CustomConfig_t mQnnHtpGraphCustomConfig{};
    QnnGraph_Config_t mQnnGraphConfig{};
    Qnn_ProfileHandle_t mQnnProfileHandle = nullptr;
    GraphInfo **mGraphsInfo = nullptr;
    uint32_t mGraphCount = 0;
    std::string mPath;
    std::unique_ptr<QNN::QNNPerf> mPerf;
    void _rebuildContextConfigs(bool enableIoMemEstimation) {
        int configCount = 0;
        bool enableWeightSharing = false;
        auto weightSharingEnv = ::getenv("MNN_QNN_ENABLE_WEIGHT_SHARING");
        if (weightSharingEnv != nullptr && weightSharingEnv[0] != '\0' && weightSharingEnv[0] != '0') {
            enableWeightSharing = true;
        }
        if (enableWeightSharing) {
            mQnnHtpContextCustomConfigs[configCount] = QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;
            mQnnHtpContextCustomConfigs[configCount].option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
            mQnnHtpContextCustomConfigs[configCount].weightSharingEnabled = true;
            mQnnContextConfigStorage[configCount] = QNN_CONTEXT_CONFIG_INIT;
            mQnnContextConfigStorage[configCount].option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
            mQnnContextConfigStorage[configCount].customConfig = &mQnnHtpContextCustomConfigs[configCount];
            mQnnContextConfigList[configCount] = &mQnnContextConfigStorage[configCount];
            ++configCount;
        }
        if (enableIoMemEstimation) {
            mQnnHtpContextCustomConfigs[configCount] = QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;
            mQnnHtpContextCustomConfigs[configCount].option = QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION;
            mQnnHtpContextCustomConfigs[configCount].ioMemEstimation = true;
            mQnnContextConfigStorage[configCount] = QNN_CONTEXT_CONFIG_INIT;
            mQnnContextConfigStorage[configCount].option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
            mQnnContextConfigStorage[configCount].customConfig = &mQnnHtpContextCustomConfigs[configCount];
            mQnnContextConfigList[configCount] = &mQnnContextConfigStorage[configCount];
            ++configCount;
        }
        mQnnContextConfigList[configCount] = nullptr;
        mQnnContextConfig = mQnnContextConfigList;
    }
    bool _shouldEnableIoMemEstimation() const {
        auto deviceGetPlatformInfo = QNN::gContext.interface.deviceGetPlatformInfo;
        if (deviceGetPlatformInfo == nullptr) {
            return true;
        }
        const QnnDevice_PlatformInfo_t* platformInfo = nullptr;
        auto qnnStatus = deviceGetPlatformInfo(nullptr, &platformInfo);
        if (QNN_SUCCESS != qnnStatus || platformInfo == nullptr) {
            MNN_PRINT("MNN_QNN: deviceGetPlatformInfo status=%d, keep io_mem_estimation enabled.\n",
                      (int)(qnnStatus & 0xFFFF));
            return true;
        }
        if (platformInfo->version != QNN_DEVICE_PLATFORM_INFO_VERSION_1 ||
            platformInfo->v1.numHwDevices == 0 || platformInfo->v1.hwDevices == nullptr) {
            return true;
        }
        auto& hwDevice = platformInfo->v1.hwDevices[0];
        if (hwDevice.version != QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1) {
            return true;
        }
        // Follow Qualcomm 2.42 Genie path: disable IO mem estimation on multi-core HTP devices.
        return hwDevice.v1.numCores <= 1;
    }

public:
    RawExecutorWrapper() {
        mPerf = QNN::QNNPerf::create(&QNN::gContext.interface);
        mPerf->setPowerConfigBurst();
        mPerf->setRpcLatencyAndPolling();
        _rebuildContextConfigs(true);
    }
    ~ RawExecutorWrapper() {
        if (mQnnProfileHandle) {
            QNN::gContext.interface.profileFree(mQnnProfileHandle);
            mQnnProfileHandle = nullptr;
        }
        if (nullptr != mQnnContextHandle) {
            CALL_QNN(QNN::gContext.interface.contextFree(mQnnContextHandle, nullptr));
        }
        freeGraphsInfo(&mGraphsInfo, mGraphCount);
    }

    bool compileModel(const std::string& path, size_t offset, size_t size, const std::vector<std::string>& allGraphName) {
        mPath = path;
        if (!QNN::ensureQnnContextReady()) {
            MNN_ERROR("MNN_QNN: QNN global context is not ready before loading %s.\n", path.c_str());
            return false;
        }
        bool enableIoMemEstimation = _shouldEnableIoMemEstimation();
        _rebuildContextConfigs(enableIoMemEstimation);
        void* buffer = nullptr;
        std::vector<char> bufferVec(size, 0);
        MMapReader reader;
        if (size > 0) {
            buffer = bufferVec.data();
            std::unique_ptr<FileLoader> binaryFile(new FileLoader(path.c_str()));
            binaryFile->offset((int64_t)offset);
            binaryFile->read((char *)buffer, (int64_t)size);
        } else {
            reader.open(path.c_str());
            buffer = reader.addr();
            size = reader.size();
        }

        // 1. Set mGraphsInfo and mGraphCount from the buffer.
        {
            QnnSystemContext_Handle_t systemContextHandle = nullptr;
            if (QNN_SUCCESS != QNN::gContext.systemInterface.systemContextCreate(&systemContextHandle)) {
                MNN_ERROR("Could not create system context handle.");
                return false;
            }
            Qnn_ContextBinarySize_t binarySize = 0;
            const QnnSystemContext_BinaryInfo_t* binaryInfo = nullptr;
            CALL_QNN(QNN::gContext.systemInterface.systemContextGetBinaryInfo(systemContextHandle, buffer, size, &binaryInfo,&binarySize));
            copyMetadataToGraphsInfo(binaryInfo, mGraphsInfo, mGraphCount);
            if (QNN_SUCCESS != QNN::gContext.systemInterface.systemContextFree(systemContextHandle)) {
                MNN_ERROR("Could not free system context handle.");
                return false;
            }
        }

        // 2. Retrieve graphs.
        {
            Qnn_ErrorHandle_t error = QNN_SUCCESS;

            // Create Graph profile
            MNN::QNN::createProfileHandle(QNN::gContext.interface, QNN::gContext.backendHandle, &mQnnProfileHandle);

            error = QNN::gContext.interface.contextCreateFromBinary(QNN::gContext.backendHandle,
                                                                    QNN::gContext.deviceHandle,
                                                                    mQnnContextConfig,
                                                                    buffer,
                                                                    size,
                                                                    &mQnnContextHandle,
                                                                    mQnnProfileHandle);
            if (QNN_SUCCESS != error || nullptr == mQnnContextHandle) {
                MNN_ERROR("MNN_QNN: contextCreateFromBinary failed for %s, error=%d, ctx=%p. Recreate global context and retry.\n",
                          path.c_str(), (int)(error & 0xFFFF), mQnnContextHandle);
                QNN::createQnnContext();
                if (!QNN::ensureQnnContextReady()) {
                    MNN_ERROR("MNN_QNN: Failed to recreate global QNN context for %s.\n", path.c_str());
                    return false;
                }
                error = QNN::gContext.interface.contextCreateFromBinary(QNN::gContext.backendHandle,
                                                                        QNN::gContext.deviceHandle,
                                                                        mQnnContextConfig,
                                                                        buffer,
                                                                        size,
                                                                        &mQnnContextHandle,
                                                                        mQnnProfileHandle);
                if (QNN_SUCCESS != error || nullptr == mQnnContextHandle) {
                    MNN_ERROR("MNN_QNN: contextCreateFromBinary retry failed for %s, error=%d, ctx=%p.\n",
                              path.c_str(), (int)(error & 0xFFFF), mQnnContextHandle);
                    return false;
                }
            }

            mQnnGraphHandleVec.resize(mGraphCount, nullptr);

            if (allGraphName.size() == mGraphCount) {
                std::vector<GraphInfo*> sortedGraphsInfo(mGraphCount, nullptr);
                std::map<std::string, GraphInfo*> graphInfoMap;
                for (int i = 0; i < mGraphCount; ++i) {
                    graphInfoMap[mGraphsInfo[i]->graphName] = mGraphsInfo[i];
                }
                for (int i = 0; i < mGraphCount; ++i) {
                    auto it = graphInfoMap.find(allGraphName[i]);
                    MNN_ASSERT(it != graphInfoMap.end());
                    sortedGraphsInfo[i] = it->second;
                }
                for (int i = 0; i < mGraphCount; ++i) {
                    mGraphsInfo[i] = sortedGraphsInfo[i];
                }
            }

            for (int i = 0; i < mGraphCount; i++) {
                error = QNN::gContext.interface.graphRetrieve(mQnnContextHandle, mGraphsInfo[i]->graphName, &(mQnnGraphHandleVec[i]));
                if (QNN_SUCCESS != error || nullptr == mQnnGraphHandleVec[i]) {
                    MNN_ERROR("MNN_QNN: graphRetrieve failed for %s graph=%s error=%d handle=%p\n",
                              path.c_str(), mGraphsInfo[i]->graphName, (int)(error & 0xFFFF), mQnnGraphHandleVec[i]);
                    return false;
                }
            }
        }

        return true;
    }
    Qnn_Tensor_t* _findInput(const std::string& name, int index) {
        GraphInfo* graph = mGraphsInfo[index];
        for (int j=0; j<graph->numInputTensors; ++j) {
            auto& dstT = graph->inputTensors[j];
            #ifdef QNN_VERBOSE
            MNN_PRINT("input name: %s %s\n", inputs[i].second.c_str(), dstT.v1.name);
            #endif
            if (name == dstT.v1.name) {
                return &dstT;
            }
        }
        return nullptr;
    }
    Qnn_Tensor_t* _findOutput(const std::string& name, int index) {
        GraphInfo* graph = mGraphsInfo[index];
        for (int j=0; j<graph->numOutputTensors; ++j) {
            auto& dstT = graph->outputTensors[j];
            #ifdef QNN_VERBOSE
            MNN_PRINT("input name: %s %s\n", inputs[i].second.c_str(), dstT.v1.name);
            #endif
            if (name == dstT.v1.name) {
                return &dstT;
            }
        }
        return nullptr;
    }
    void setupAddress(const std::vector<std::pair<const MNN::Tensor *, std::string>>& inputs, std::vector<std::pair<const MNN::Tensor *, std::string>>& outputs, int shapeIndex) {
        GraphInfo* graph = mGraphsInfo[shapeIndex];
        Qnn_GraphHandle_t qnnGraphHandle = mQnnGraphHandleVec[shapeIndex];

        // MNN_PRINT("%s, Input:%d, output:%d\n", mPath.c_str(), inputs.size(), outputs.size());
        for (int i=0; i<inputs.size(); ++i) {
            auto t = inputs[i].first;
            bool find = false;
            for (int j=0; j<graph->numInputTensors; ++j) {
                auto& dstT = graph->inputTensors[j];
                #ifdef QNN_VERBOSE
                MNN_PRINT("input name: %s %s\n", inputs[i].second.c_str(), dstT.v1.name);
                #endif
                if (inputs[i].second == dstT.v1.name) {
                    syncTensorShapeFromMNN(dstT, t);
                    dstT.v1.clientBuf.data = t->host<void>();
                    dstT.v1.clientBuf.dataSize = t->usize();
                    find = true;
                    break;
                }
            }
            if (!find) {
                MNN_ERROR("%s, can't find %d input: %s\n", mPath.c_str(), i, inputs[i].second.c_str());
            }
        }
        for (int i=0; i<outputs.size(); ++i) {
            auto t = outputs[i].first;
            bool find = false;
            for (int j=0; j<graph->numOutputTensors; ++j) {
                auto& dstT = graph->outputTensors[j];
                #ifdef QNN_VERBOSE
                MNN_PRINT("output name: %s %s\n", outputs[i].second.c_str(), dstT.v1.name);
                #endif
                if (outputs[i].second == dstT.v1.name) {
                    syncTensorShapeFromMNN(dstT, t);
                    dstT.v1.clientBuf.data = t->host<void>();
                    dstT.v1.clientBuf.dataSize = t->usize();
                    find = true;
                    break;
                }
            }
            if (!find) {
                MNN_ERROR("%s, can't find %d output: %s\n", mPath.c_str(), i, outputs[i].second.c_str());
                for (int j = 0; j < graph->numOutputTensors; ++j) {
                    auto& dstT = graph->outputTensors[j];
                    MNN_ERROR("  graph output[%d]: name=%s type=%d dtype=%d rank=%u\n",
                              j,
                              dstT.v1.name == nullptr ? "(null)" : dstT.v1.name,
                              (int)dstT.v1.type,
                              (int)dstT.v1.dataType,
                              (unsigned)dstT.v1.rank);
                }
            }
        }
    }
    bool setupState(RPCBuffer* mask, std::vector<RPCBuffer*> statesInputs, std::vector<RPCBuffer*> statesOutput, int index) {
        auto maskTensor = _findInput(gExtraIoPrefix + "_mask", index);
        if (nullptr != maskTensor && nullptr != mask) {
            if (!mask->setToTensor(maskTensor, &QNN::gContext.interface, mQnnContextHandle)) {
                return false;
            }
        }
        for (int i=0; i<statesInputs.size(); ++i) {
            auto t = _findInput(gExtraIoPrefix + "_i" + std::to_string(i), index);
            if (nullptr == t) {
                MNN_ERROR("Can't find %d input tensor of state\n", i);
                continue;
            }
            if (!statesInputs[i]->setToTensor(t, &QNN::gContext.interface, mQnnContextHandle)) {
                return false;
            }
        }
        for (int i=0; i<statesOutput.size(); ++i) {
            auto t = _findOutput(gExtraIoPrefix + "_o" + std::to_string(i), index);
            if (nullptr == t) {
                MNN_ERROR("Can't find %d output tensor of state\n", i);
                continue;
            }
            if (!statesOutput[i]->setToTensor(t, &QNN::gContext.interface, mQnnContextHandle)) {
                return false;
            }
        }
        return true;
    }

    void invokModel(int shapeIndex) {
        GraphInfo* graph = mGraphsInfo[shapeIndex];
        Qnn_GraphHandle_t qnnGraphHandle = mQnnGraphHandleVec[shapeIndex];
        auto execError = QNN::gContext.interface.graphExecute(qnnGraphHandle,
                                                              graph->inputTensors,
                                                              graph->numInputTensors,
                                                              graph->outputTensors,
                                                              graph->numOutputTensors,
                                                              mQnnProfileHandle,
                                                              nullptr);
        if ((execError & 0xFFFF) != QNN_SUCCESS) {
            MNN_ERROR("MNN_QNN: graphExecute failed path=%s graph=%s index=%d error=%d\n",
                      mPath.c_str(),
                      graph->graphName == nullptr ? "(null)" : graph->graphName,
                      shapeIndex,
                      (int)(execError & 0xFFFF));
        }
        CALL_QNN(execError);
        MNN::QNN::doProfile(QNN::gContext.interface, mQnnProfileHandle);
    }
};

class PluginExecuteRaw : public CPUComputeKernel {
private:
    struct SyntheticOutput {
        enum Kind {
            NONE = 0,
            BATCH_FROM_RESHAPE = 1,
            SEQ_FROM_RESHAPE = 2,
        };
        int outputIndex = -1;
        int reshapeIndex = -1;
        Kind kind = NONE;
    };
    std::shared_ptr<RawExecutorWrapper> mRawExecutor;
    std::vector<std::pair<const MNN::Tensor *, std::string>> mInputs;
    std::vector<std::pair<const MNN::Tensor *, std::string>> mOutputs;
    std::vector<std::string> mOutputAliases;
    std::vector<std::shared_ptr<MNN::Tensor>> mRealInputs;
    std::vector<std::shared_ptr<MNN::Tensor>> mRealOutputs;
    std::vector<SyntheticOutput> mSyntheticOutputs;
    int mShapeIndex;
    std::string mBinaryPath;
    size_t mBinaryOffset = 0;
    size_t mBinarySize = 0;
    std::vector<std::string> mAllGraphName;
    static std::mutex& _executorCacheMutex() {
        static std::mutex mutex;
        return mutex;
    }
    static std::unordered_map<std::string, std::weak_ptr<RawExecutorWrapper>>& _executorCache() {
        static std::unordered_map<std::string, std::weak_ptr<RawExecutorWrapper>> cache;
        return cache;
    }
    static std::unordered_map<std::string, std::shared_ptr<RPCBuffer>>& _replaceStateCache() {
        static std::unordered_map<std::string, std::shared_ptr<RPCBuffer>> cache;
        return cache;
    }
    std::string _executorCacheKey() const {
        return mBinaryPath + "#" + std::to_string(mBinaryOffset) + "#" + std::to_string(mBinarySize);
    }
    std::string _stateCacheKey(int index) const {
        return _executorCacheKey() + "#state#" + std::to_string(index);
    }
    static std::string _normalizePath(const std::string& path) {
        std::string normalized;
        normalized.reserve(path.size());
        bool prevSlash = false;
        for (char c : path) {
            if (c == '/') {
                if (prevSlash) {
                    continue;
                }
                prevSlash = true;
            } else {
                prevSlash = false;
            }
            normalized.push_back(c);
        }
        while (normalized.size() > 2 && normalized.compare(0, 2, "./") == 0) {
            normalized.erase(0, 2);
        }
        return normalized;
    }

    struct StateTensor {
        enum Mode {
            APPEND = 0,
            REPLACE = 1,
        };
        std::shared_ptr<RPCBuffer> data;
        std::vector<int> shape;
        int axis = -1;
        int maxSize = 0;
        Mode mode = APPEND;
        int inside;
        int outside;
        Qnn_DataType_t dataType = QNN_DATATYPE_FLOAT_16;
        int elementBytes = sizeof(int16_t);
        std::vector<std::shared_ptr<RPCBuffer>> update;
    };
    std::vector<StateTensor> mStateInput;
    int mStateCurrent = 0;
    int mStateMaxSize = 0;
    std::vector<int> mSeqLen;
    std::shared_ptr<RPCBuffer> mMask;
    int mLastStateMetaDumpShapeIndex = -1;
    const float mMinValue = -32700.0f;
    bool _shouldReleaseExecutorPerRun() const {
        auto enabled = ::getenv("MNN_QNN_PLUGIN_RELEASE_CONTEXT_EACH_RUN");
        return enabled != nullptr && enabled[0] != '\0' && enabled[0] != '0';
    }
    bool _shouldDumpGraphIo(const std::string& graphName) const {
        auto enabled = ::getenv("MNN_QNN_DUMP_PLUGIN_IO");
        if (enabled == nullptr || enabled[0] == '\0' || enabled[0] == '0') {
            return false;
        }
        auto match = ::getenv("MNN_QNN_DUMP_PLUGIN_MATCH");
        if (match == nullptr || match[0] == '\0') {
            return true;
        }
        return graphName.find(match) != std::string::npos;
    }
    bool _shouldDumpGraphState(const std::string& graphName) const {
        auto enabled = ::getenv("MNN_QNN_DUMP_PLUGIN_STATE");
        if (enabled == nullptr || enabled[0] == '\0') {
            return _shouldDumpGraphIo(graphName);
        }
        if (enabled[0] == '0') {
            return false;
        }
        auto match = ::getenv("MNN_QNN_DUMP_PLUGIN_MATCH");
        if (match == nullptr || match[0] == '\0') {
            return true;
        }
        return graphName.find(match) != std::string::npos;
    }
    static int _computeElementCount(const std::vector<int>& shape) {
        if (shape.empty()) {
            return 0;
        }
        int count = 1;
        for (auto dim : shape) {
            count *= dim;
        }
        return count;
    }
    static std::string _shapeString(const std::vector<int>& shape) {
        std::ostringstream os;
        os << "[";
        for (int i = 0; i < shape.size(); ++i) {
            if (i > 0) {
                os << ",";
            }
            os << shape[i];
        }
        os << "]";
        return os.str();
    }
    static const char* _stateModeString(StateTensor::Mode mode) {
        return mode == StateTensor::REPLACE ? "replace" : "append";
    }
    static int _qnnDataTypeBytes(Qnn_DataType_t dataType) {
        switch (dataType) {
            case QNN_DATATYPE_FLOAT_16:
                return sizeof(int16_t);
            case QNN_DATATYPE_FLOAT_32:
                return sizeof(float);
            default:
                return 0;
        }
    }
    static int16_t _floatToHalfBits(float value) {
        int16_t bits = 0;
        FLOAT_TO_HALF(&value, &bits, 1);
        return bits;
    }
    static void _printFloatStats(const char* phase, const std::string& graphName, const std::string& name,
                                 const std::string& alias, const float* ptr, int size) {
        if (ptr == nullptr || size <= 0) {
            MNN_PRINT("MNN_QNN_DUMP %s graph=%s name=%s alias=%s float32 unavailable size=%d\n",
                      phase, graphName.c_str(), name.c_str(), alias.c_str(), size);
            return;
        }
        double minValue = ptr[0];
        double maxValue = ptr[0];
        double maxAbs = std::fabs((double)ptr[0]);
        double sumAbs = 0.0;
        for (int i = 0; i < size; ++i) {
            double value = ptr[i];
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
            maxAbs = std::max(maxAbs, std::fabs(value));
            sumAbs += std::fabs(value);
        }
        MNN_PRINT("MNN_QNN_DUMP %s graph=%s name=%s alias=%s dtype=float32 size=%d min=%f max=%f meanAbs=%f maxAbs=%f first=",
                  phase, graphName.c_str(), name.c_str(), alias.c_str(), size, minValue, maxValue, sumAbs / size, maxAbs);
        int limit = std::min(size, 8);
        for (int i = 0; i < limit; ++i) {
            MNN_PRINT("%s%f", i == 0 ? "" : ",", ptr[i]);
        }
        MNN_PRINT("\n");
    }
    static void _printHalfStats(const char* phase, const std::string& graphName, const std::string& name,
                                const std::string& alias, const int16_t* ptr, int size) {
        if (ptr == nullptr || size <= 0) {
            MNN_PRINT("MNN_QNN_DUMP %s graph=%s name=%s alias=%s float16 unavailable size=%d\n",
                      phase, graphName.c_str(), name.c_str(), alias.c_str(), size);
            return;
        }
        std::vector<float> values(size);
        HALF_TO_FLOAT(ptr, values.data(), size);
        double first = values[0];
        double minValue = first;
        double maxValue = first;
        double maxAbs = std::fabs(first);
        double sumAbs = 0.0;
        for (int i = 0; i < size; ++i) {
            double value = values[i];
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
            maxAbs = std::max(maxAbs, std::fabs(value));
            sumAbs += std::fabs(value);
        }
        MNN_PRINT("MNN_QNN_DUMP %s graph=%s name=%s alias=%s dtype=float16 size=%d min=%f max=%f meanAbs=%f maxAbs=%f first=",
                  phase, graphName.c_str(), name.c_str(), alias.c_str(), size, minValue, maxValue, sumAbs / size, maxAbs);
        int limit = std::min(size, 8);
        for (int i = 0; i < limit; ++i) {
            MNN_PRINT("%s%f", i == 0 ? "" : ",", values[i]);
        }
        MNN_PRINT("\n");
    }
    static void _printIntStats(const char* phase, const std::string& graphName, const std::string& name,
                               const std::string& alias, const int* ptr, int size) {
        if (ptr == nullptr || size <= 0) {
            MNN_PRINT("MNN_QNN_DUMP %s graph=%s name=%s alias=%s int32 unavailable size=%d\n",
                      phase, graphName.c_str(), name.c_str(), alias.c_str(), size);
            return;
        }
        int minValue = ptr[0];
        int maxValue = ptr[0];
        for (int i = 1; i < size; ++i) {
            minValue = std::min(minValue, ptr[i]);
            maxValue = std::max(maxValue, ptr[i]);
        }
        MNN_PRINT("MNN_QNN_DUMP %s graph=%s name=%s alias=%s dtype=int32 size=%d min=%d max=%d first=",
                  phase, graphName.c_str(), name.c_str(), alias.c_str(), size, minValue, maxValue);
        int limit = std::min(size, 8);
        for (int i = 0; i < limit; ++i) {
            MNN_PRINT("%s%d", i == 0 ? "" : ",", ptr[i]);
        }
        MNN_PRINT("\n");
    }
    void _dumpTensor(const char* phase, const std::string& graphName, const std::string& name, const std::string& alias,
                     const MNN::Tensor* tensor) const {
        if (tensor == nullptr) {
            MNN_PRINT("MNN_QNN_DUMP %s graph=%s name=%s alias=%s null_tensor\n",
                      phase, graphName.c_str(), name.c_str(), alias.c_str());
            return;
        }
        auto type = tensor->getType();
        auto size = tensor->elementSize();
        if (type.code == halide_type_float && type.bits == 32) {
            _printFloatStats(phase, graphName, name, alias, tensor->host<float>(), size);
            return;
        }
        if (type.code == halide_type_float && type.bits == 16) {
            _printHalfStats(phase, graphName, name, alias, tensor->host<int16_t>(), size);
            return;
        }
        if (type.code == halide_type_int && type.bits == 32) {
            _printIntStats(phase, graphName, name, alias, tensor->host<int>(), size);
            return;
        }
        MNN_PRINT("MNN_QNN_DUMP %s graph=%s name=%s alias=%s dtype_code=%d bits=%d size=%d unsupported\n",
                  phase, graphName.c_str(), name.c_str(), alias.c_str(), (int)type.code, (int)type.bits, size);
    }
    void _dumpGraphInputs(const std::string& graphName) const {
        for (int i = 0; i < mInputs.size(); ++i) {
            auto tensor = i < mRealInputs.size() ? mRealInputs[i].get() : nullptr;
            _dumpTensor("input", graphName, mInputs[i].second, mInputs[i].second, tensor);
        }
    }
    void _dumpGraphOutputs(const std::string& graphName) const {
        for (int i = 0; i < mOutputs.size(); ++i) {
            auto tensor = i < mRealOutputs.size() ? mRealOutputs[i].get() : nullptr;
            std::string alias = i < mOutputAliases.size() ? mOutputAliases[i] : mOutputs[i].second;
            _dumpTensor("output", graphName, mOutputs[i].second, alias, tensor);
        }
    }
    void _dumpStateMeta(const std::string& graphName, int shapeIndex) const {
        int seqLen = shapeIndex >= 0 && shapeIndex < mSeqLen.size() ? mSeqLen[shapeIndex] : -1;
        MNN_PRINT("MNN_QNN_DUMP state_meta graph=%s shapeIndex=%d seqLen=%d stateCount=%d stateCurrent=%d stateMaxSize=%d\n",
                  graphName.c_str(), shapeIndex, seqLen, (int)mStateInput.size(), mStateCurrent, mStateMaxSize);
        for (int i = 0; i < mStateInput.size(); ++i) {
            const auto& input = mStateInput[i];
            int elementCount = _computeElementCount(input.shape);
            int storageCount = input.mode == StateTensor::APPEND ? input.maxSize * input.inside * input.outside : elementCount;
            MNN_PRINT("MNN_QNN_DUMP state_meta graph=%s index=%d mode=%s shape=%s axis=%d max=%d inside=%d outside=%d elementCount=%d storageCount=%d\n",
                      graphName.c_str(), i, _stateModeString(input.mode), _shapeString(input.shape).c_str(), input.axis,
                      input.maxSize, input.inside, input.outside, elementCount, storageCount);
        }
        if (mMask != nullptr && mStateMaxSize > 0) {
            _printHalfStats("state_mask", graphName, gExtraIoPrefix + "_mask", "state_mask",
                            reinterpret_cast<const int16_t*>(mMask->mPtr), mStateMaxSize);
        }
    }
    void _dumpStateBuffers(const char* phase, const std::string& graphName, int shapeIndex, bool useUpdateBuffer,
                           int appendTokenCount) const {
        for (int i = 0; i < mStateInput.size(); ++i) {
            const auto& input = mStateInput[i];
            const RPCBuffer* buffer = input.data.get();
            if (useUpdateBuffer) {
                if (shapeIndex < 0 || shapeIndex >= input.update.size()) {
                    MNN_PRINT("MNN_QNN_DUMP %s graph=%s name=state[%d] alias=missing_update_buffer shapeIndex=%d\n",
                              phase, graphName.c_str(), i, shapeIndex);
                    continue;
                }
                buffer = input.update[shapeIndex].get();
            }
            int elementCount = _computeElementCount(input.shape);
            int storageCount = input.mode == StateTensor::APPEND ? input.maxSize * input.inside * input.outside : elementCount;
            int validCount = elementCount;
            if (input.mode == StateTensor::APPEND) {
                int tokens = std::max(0, appendTokenCount);
                if (input.maxSize > 0) {
                    tokens = std::min(tokens, input.maxSize);
                }
                validCount = tokens * input.inside * input.outside;
            }
            std::ostringstream alias;
            alias << "mode=" << _stateModeString(input.mode)
                  << ",dtype=" << input.dataType
                  << ",shape=" << _shapeString(input.shape)
                  << ",axis=" << input.axis
                  << ",max=" << input.maxSize
                  << ",inside=" << input.inside
                  << ",outside=" << input.outside
                  << ",valid=" << validCount
                  << ",storage=" << storageCount
                  << ",buffer=" << (useUpdateBuffer ? "update" : "store");
            if (input.dataType == QNN_DATATYPE_FLOAT_32) {
                _printFloatStats(phase, graphName, "state[" + std::to_string(i) + "]", alias.str(),
                                 buffer == nullptr ? nullptr : reinterpret_cast<const float*>(buffer->mPtr), validCount);
            } else {
                _printHalfStats(phase, graphName, "state[" + std::to_string(i) + "]", alias.str(),
                                buffer == nullptr ? nullptr : reinterpret_cast<const int16_t*>(buffer->mPtr), validCount);
            }
        }
    }
    static bool _matchSuffix(const std::string& name, const char* suffix) {
        size_t suffixLen = ::strlen(suffix);
        if (name.size() < suffixLen) {
            return false;
        }
        return 0 == name.compare(name.size() - suffixLen, suffixLen, suffix);
    }
    bool _isSyntheticOutputIndex(int outputIndex) const {
        for (const auto& synthetic : mSyntheticOutputs) {
            if (synthetic.outputIndex == outputIndex) {
                return true;
            }
        }
        return false;
    }
    void _prepareSyntheticOutputs() {
        mSyntheticOutputs.clear();
        int reshapeIndex = -1;
        for (int i = 0; i < mOutputAliases.size(); ++i) {
            if (_matchSuffix(mOutputAliases[i], "/Reshape_output_0")) {
                reshapeIndex = i;
                break;
            }
        }
        if (reshapeIndex < 0) {
            return;
        }
        for (int i = 0; i < mOutputs.size(); ++i) {
            auto tensor = mRealOutputs[i].get();
            if (nullptr == tensor) {
                continue;
            }
            if (tensor->getType().code != halide_type_int || tensor->getType().bits != 32 || tensor->elementSize() != 1) {
                continue;
            }
            SyntheticOutput synthetic;
            synthetic.outputIndex = i;
            synthetic.reshapeIndex = reshapeIndex;
            if (i < mOutputAliases.size() && _matchSuffix(mOutputAliases[i], "/self_attn/Gather_output_0")) {
                synthetic.kind = SyntheticOutput::BATCH_FROM_RESHAPE;
            } else if (i < mOutputAliases.size() && _matchSuffix(mOutputAliases[i], "/self_attn/Gather_1_output_0")) {
                synthetic.kind = SyntheticOutput::SEQ_FROM_RESHAPE;
            } else {
                continue;
            }
            mSyntheticOutputs.emplace_back(synthetic);
        }
    }
    void _fillSyntheticOutputs() {
        for (auto& synthetic : mSyntheticOutputs) {
            if (synthetic.reshapeIndex < 0 || synthetic.reshapeIndex >= mRealOutputs.size()) {
                continue;
            }
            auto shapeTensor = mRealOutputs[synthetic.reshapeIndex].get();
            auto outputTensor = mRealOutputs[synthetic.outputIndex].get();
            if (nullptr == shapeTensor || nullptr == outputTensor) {
                continue;
            }
            int dimensions = shapeTensor->buffer().dimensions;
            if (dimensions <= 0) {
                continue;
            }
            int value = 1;
            switch (synthetic.kind) {
                case SyntheticOutput::BATCH_FROM_RESHAPE:
                    value = shapeTensor->buffer().dim[0].extent;
                    break;
                case SyntheticOutput::SEQ_FROM_RESHAPE:
                    value = dimensions > 1 ? shapeTensor->buffer().dim[1].extent : 1;
                    break;
                default:
                    continue;
            }
            outputTensor->host<int32_t>()[0] = value;
        }
    }
    void _loadState(std::vector<int> seqLen) {
        if (mStateInput.empty()) {
            return;
        }
        if (mStateMaxSize > 0) {
            mMask.reset(RPCBuffer::alloc(mStateMaxSize * sizeof(int16_t)));
            auto maskPtr = reinterpret_cast<int16_t*>(mMask->mPtr);
            auto minValueHalf = _floatToHalfBits(mMinValue);
            for (int i=0; i<mStateMaxSize; ++i) {
                maskPtr[i] = minValueHalf;
            }
        }
        for (int i=0; i<mStateInput.size(); ++i) {
            auto& state = mStateInput[i];
            int bytes = state.elementBytes;
            MNN_ASSERT(bytes > 0);
            int elementCount = 1;
            for (auto dim : state.shape) {
                elementCount *= dim;
            }
            int storageCount = elementCount;
            if (state.mode == StateTensor::APPEND) {
                storageCount = state.maxSize * state.inside * state.outside;
            }
            if (state.mode == StateTensor::REPLACE) {
                auto cacheKey = _stateCacheKey(i);
                {
                    std::lock_guard<std::mutex> lock(_executorCacheMutex());
                    auto& cache = _replaceStateCache();
                    auto it = cache.find(cacheKey);
                    if (it != cache.end()) {
                        state.data = it->second;
                    } else {
                        state.data.reset(RPCBuffer::alloc(storageCount * bytes));
                        ::memset(state.data->mPtr, 0, storageCount * bytes);
                        cache[cacheKey] = state.data;
                    }
                }
            } else {
                state.data.reset(RPCBuffer::alloc(storageCount * bytes));
                ::memset(state.data->mPtr, 0, storageCount * bytes);
            }
            mStateInput[i].update.resize(seqLen.size());
            for (int j=0; j<seqLen.size(); ++j) {
                int updateCount = elementCount;
                if (state.mode == StateTensor::APPEND) {
                    // QNN graph metadata for state tensors uses the full/max KV shape, not the
                    // current per-step seqLen shape. Bind a full-capacity buffer so memRegister
                    // and graphExecute always see a buffer large enough for the declared tensor.
                    updateCount = storageCount;
                }
                state.update[j].reset(RPCBuffer::alloc(updateCount * bytes));
                ::memset(state.update[j]->mPtr, 0, updateCount * bytes);
            }
        }
    }
    bool _prepareStateDType() {
        for (int i = 0; i < mStateInput.size(); ++i) {
            auto inputTensor = mRawExecutor-> _findInput(gExtraIoPrefix + "_i" + std::to_string(i), 0);
            if (inputTensor == nullptr) {
                MNN_ERROR("MNN_QNN: can't find state input tensor %d for dtype init\n", i);
                return false;
            }
            auto bytes = _qnnDataTypeBytes(QNN_TENSOR_GET_DATA_TYPE(inputTensor));
            if (bytes <= 0) {
                MNN_ERROR("MNN_QNN: unsupported state dtype=%d for state input %d\n",
                          (int)QNN_TENSOR_GET_DATA_TYPE(inputTensor), i);
                return false;
            }
            mStateInput[i].dataType = QNN_TENSOR_GET_DATA_TYPE(inputTensor);
            mStateInput[i].elementBytes = bytes;
        }
        return true;
    }
    bool _ensureExecutorLoaded() {
        if (mRawExecutor != nullptr) {
            return true;
        }
        auto key = _executorCacheKey();
        {
            std::lock_guard<std::mutex> lock(_executorCacheMutex());
            auto& cache = _executorCache();
            auto it = cache.find(key);
            if (it != cache.end()) {
                mRawExecutor = it->second.lock();
                if (mRawExecutor == nullptr) {
                    cache.erase(it);
                }
            }
            if (mRawExecutor == nullptr) {
                auto executor = std::make_shared<RawExecutorWrapper>();
                if (!executor->compileModel(mBinaryPath, mBinaryOffset, mBinarySize, mAllGraphName)) {
                    return false;
                }
                cache[key] = executor;
                mRawExecutor = executor;
            }
        }
        if (!mStateInput.empty() && (mStateInput[0].data == nullptr || mStateInput[0].update.empty())) {
            if (!_prepareStateDType()) {
                return false;
            }
            _loadState(mSeqLen);
        }
        return true;
    }
    bool _bindExecutorTensors(int shapeIndex) {
        std::vector<std::pair<const MNN::Tensor*, std::string>> qnnOutputs;
        qnnOutputs.reserve(mOutputs.size());
        for (int i = 0; i < mOutputs.size(); ++i) {
            if (_isSyntheticOutputIndex(i)) {
                continue;
            }
            qnnOutputs.emplace_back(mOutputs[i]);
        }
        mRawExecutor->setupAddress(mInputs, qnnOutputs, shapeIndex);
        if (!mStateInput.empty()) {
            std::vector<RPCBuffer*> states(mStateInput.size());
            for (int i = 0; i < mStateInput.size(); ++i) {
                states[i] = mStateInput[i].data.get();
            }
            std::vector<RPCBuffer*> statesOutput(mStateInput.size());
            for (int i = 0; i < mStateInput.size(); ++i) {
                statesOutput[i] = mStateInput[i].update[shapeIndex].get();
            }
            if (!mRawExecutor->setupState(mMask.get(), states, statesOutput, shapeIndex)) {
                MNN_ERROR("MNN_QNN: Failed to bind plugin state tensors for graph index %d.\n", shapeIndex);
                return false;
            }
        }
        return true;
    }
    void _releaseExecutor() {
        mRawExecutor.reset();
    }

public:
    ~ PluginExecuteRaw() {
        mRealInputs.clear();
        mRealOutputs.clear();
        mRawExecutor.reset();
    }
    bool init(CPUKernelContext* ctx) override {
        if (QNN::QnnInterface_getProviders == nullptr
#ifdef MNN_WITH_PLUGIN
            || QNN::QnnSystemInterface_getProviders == nullptr
#endif
        ) {
            if (!QNN::loadQNNSymbol()) {
                MNN_ERROR("MNN_QNN: Failed to load QNN symbols in plugin runtime init.\n");
                return false;
            }
        }
        if (!QNN::ensureQnnContextReady()) {
            MNN_ERROR("MNN_QNN: Failed to create QNN device/context for plugin runtime.\n");
            return false;
        }
        auto seqLen = ctx->getAttr("seq_len");
        if (nullptr != seqLen && nullptr != seqLen->list()) {
            mSeqLen.resize(seqLen->list()->i()->size());
            ::memcpy(mSeqLen.data(), seqLen->list()->i()->data(), mSeqLen.size() * sizeof(int));
        }
        auto state = ctx->getAttr("state");
        if (nullptr != state) {
            auto ref = flexbuffers::GetRoot(state->tensor()->uint8s()->data(), state->tensor()->uint8s()->size());
            auto refMap = ref.AsMap();
            auto entries = refMap["entries"];
            if (entries.IsVector()) {
                auto entryVec = entries.AsVector();
                mStateInput.resize(entryVec.size());
                for (int i=0; i<entryVec.size(); ++i) {
                    auto stateMap = entryVec[i].AsMap();
                    auto& input = mStateInput[i];
                    auto modeRef = stateMap["mode"];
                    if (modeRef.IsString() && modeRef.AsString().str() == "replace") {
                        input.mode = StateTensor::REPLACE;
                    } else {
                        input.mode = StateTensor::APPEND;
                    }
                    input.axis = stateMap["axis"].IsInt() ? stateMap["axis"].AsInt32() : -1;
                    input.maxSize = stateMap["max_length"].IsInt() ? stateMap["max_length"].AsInt32() : 0;
                    if (input.mode == StateTensor::APPEND) {
                        mStateMaxSize = std::max(mStateMaxSize, input.maxSize);
                    }
                    auto shapeVector = stateMap["shape"].AsVector();
                    input.shape.resize(shapeVector.size());
                    for (int v=0; v<shapeVector.size(); ++v) {
                        input.shape[v] = shapeVector[v].AsInt32();
                    }
                    if (input.mode == StateTensor::APPEND) {
                        input.outside = 1;
                        for (int j=0; j<input.axis; ++j) {
                            input.outside *= input.shape[j];
                        }
                        auto axisLength = input.shape[input.axis];
                        MNN_ASSERT(1 == axisLength);
                        input.inside = 1;
                        for (int j=input.axis+1; j<input.shape.size(); ++j) {
                            input.inside *= input.shape[j];
                        }
                    } else {
                        input.outside = 1;
                        input.inside = 1;
                    }
                }
            } else {
                int stateNumber = 0;
                int axis = 0;
                auto keys = refMap.Keys();
                std::vector<std::vector<int>> stateShape;
                for (int i=0; i<keys.size(); ++i) {
                    auto key = keys[i].AsKey();
                    if (std::string(key) == "number") {
                        stateNumber = refMap.Values()[i].AsInt32();
                        continue;
                    }
                    if (std::string(key) == "max_length") {
                        mStateMaxSize = refMap.Values()[i].AsInt32();
                        continue;
                    }
                    if (std::string(key) == "axis") {
                        axis = refMap.Values()[i].AsInt32();
                        continue;
                    }
                    if (std::string(key) == "shape") {
                        auto shapeVectors = refMap.Values()[i].AsVector();
                        for (int u=0; u<shapeVectors.size(); ++u) {
                            auto shapeV = shapeVectors[u].AsVector();
                            std::vector<int> shapes;
                            for (int v=0; v<shapeV.size(); ++v) {
                                shapes.emplace_back(shapeV[v].AsInt32());
                            }
                            stateShape.emplace_back(shapes);
                        }
                        continue;
                    }
                }
                mStateInput.resize(stateShape.size());
                for (int i=0; i<stateShape.size(); ++i) {
                    auto& shape = stateShape[i];
                    auto& input = mStateInput[i];
                    input.mode = StateTensor::APPEND;
                    input.shape = shape;
                    input.axis = axis;
                    input.maxSize = mStateMaxSize;
                    input.outside = 1;
                    for (int j=0; j<axis; ++j) {
                        input.outside *= shape[j];
                    }
                    auto axisLength = shape[axis];
                    MNN_ASSERT(1 == axisLength);
                    input.inside = 1;
                    for (int j=axis+1; j<shape.size(); ++j) {
                        input.inside *= shape[j];
                    }
                }
            }
        }
        auto path = MNNFilePathConcat(ctx->dir_path(), ctx->getAttr("path")->s()->str());
        mBinaryPath = _normalizePath(path);

        auto allGraphNameAttr = ctx->getAttr("allGraphName");
        if (allGraphNameAttr && allGraphNameAttr->list() && allGraphNameAttr->list()->s()) {
            auto graphNames = allGraphNameAttr->list()->s();
            for (int i = 0; i < graphNames->size(); ++i) {
                mAllGraphName.push_back(graphNames->GetAsString(i)->str());
            }
        } else {
            MNN_ERROR("MNN_QNN: Incorrect Plugin Op, can't find 'allGraphName' attr.\n");
            return false;
        }

        auto offsetAttr = ctx->getAttr("offset");
        if (offsetAttr && offsetAttr->list() && offsetAttr->list()->i()->size() == 2) {
            const int * dataPtr = offsetAttr->list()->i()->data();
            int lowSrc = dataPtr[0];
            int highSrc = dataPtr[1];

            uint32_t lowDst, highDst;
            ::memcpy(&lowDst, &lowSrc, sizeof(uint32_t));
            ::memcpy(&highDst, &highSrc, sizeof(uint32_t));

            mBinaryOffset = (static_cast<size_t>(highDst) << 32) | static_cast<size_t>(lowDst);
        }

        auto sizeAttr = ctx->getAttr("size");
        if (sizeAttr && sizeAttr->list() && sizeAttr->list()->i()->size() == 2) {
            const int * dataPtr = sizeAttr->list()->i()->data();
            int lowSrc = dataPtr[0];
            int highSrc = dataPtr[1];

            uint32_t lowDst, highDst;
            ::memcpy(&lowDst, &lowSrc, sizeof(uint32_t));
            ::memcpy(&highDst, &highSrc, sizeof(uint32_t));

            mBinarySize = (static_cast<size_t>(highDst) << 32) | static_cast<size_t>(lowDst);
        }
        if (!_ensureExecutorLoaded()) {
            return false;
        }
        if (!_prepareStateDType()) {
            return false;
        }
        _loadState(mSeqLen);
        return true;
    }

    bool resize(CPUKernelContext* ctx) override {
        int shapeIndex = 0;
        if (!(shape_inference::computeIndex(ctx, shapeIndex))) {
            MNN_ERROR("MNN_QNN: Failed to execute Plugin Op.\n");
            return false;
        }
        mShapeIndex = shapeIndex;

        auto inputs = ctx->getAttr("inputs")->list();
        auto inputTensor = ctx->inputs();
        MNN_ASSERT(inputs->s()->size() == inputTensor.size());
        mInputs.resize(inputs->s()->size());
        mRealInputs.resize(inputTensor.size());
        for (int i=0; i<inputs->s()->size(); ++i) {
            mRealInputs[i].reset(new Tensor(inputTensor[i], Tensor::CAFFE));
            mInputs[i].second = inputs->s()->GetAsString(i)->str();
            mInputs[i].first = mRealInputs[i].get();
        }
        auto outputs = ctx->getAttr("outputs")->list();
        auto outputTensor = ctx->outputs();
        mOutputs.resize(outputs->s()->size());
        MNN_ASSERT(outputs->s()->size() == outputTensor.size());
        mRealOutputs.resize(outputTensor.size());
        mOutputAliases.clear();
        auto outputAliases = ctx->getAttr("output_aliases");
        if (outputAliases != nullptr && outputAliases->list() != nullptr && outputAliases->list()->s() != nullptr) {
            auto aliases = outputAliases->list()->s();
            mOutputAliases.resize(aliases->size());
            for (int i = 0; i < aliases->size(); ++i) {
                mOutputAliases[i] = aliases->GetAsString(i)->str();
            }
        }
        for (int i=0; i<outputs->s()->size(); ++i) {
            mRealOutputs[i].reset(new Tensor(outputTensor[i], Tensor::CAFFE));
            mOutputs[i].second = outputs->s()->GetAsString(i)->str();
            mOutputs[i].first = mRealOutputs[i].get();
        }
        _prepareSyntheticOutputs();
        if (!_ensureExecutorLoaded()) {
            return false;
        }
        if (!_bindExecutorTensors(mShapeIndex)) {
            return false;
        }
        return true;
    }

    bool compute(CPUKernelContext* ctx) override {
        AUTOTIME;
        int shapeIndex = mShapeIndex;
        std::string graphName = ctx->getAttr("allGraphName")->list()->s()->GetAsString(shapeIndex)->str();
        if (::getenv("MNN_QNN_TRACE_ENTRY") != nullptr) {
            MNN_PRINT("MNN_QNN_TRACE plugin_compute graph=%s shapeIndex=%d inputs=%d outputs=%d states=%d\n",
                      graphName.c_str(), shapeIndex, (int)mInputs.size(), (int)mOutputs.size(), (int)mStateInput.size());
        }
        bool dumpGraphIo = _shouldDumpGraphIo(graphName);
        bool dumpGraphState = _shouldDumpGraphState(graphName);

        #ifdef QNN_VERBOSE
        MNN_PRINT("Graph name:%s, %d\n", graphName.c_str(), shapeIndex);
        #endif
        auto inputTensor = ctx->inputs();
        auto outputTensor = ctx->outputs();
        if (!_ensureExecutorLoaded()) {
            return false;
        }
        if (!_bindExecutorTensors(shapeIndex)) {
            return false;
        }

        for (int i=0; i<mInputs.size(); ++i) {
            ctx->backend()->onCopyBuffer(inputTensor[i], mRealInputs[i].get());
        }
        if (dumpGraphIo) {
            _dumpGraphInputs(graphName);
        }
        // If has remove, remove invalid state
        auto meta = (KVMeta*)(ctx->backend()->getMetaPtr());
        if (dumpGraphState && mLastStateMetaDumpShapeIndex != shapeIndex) {
            _dumpStateMeta(graphName, shapeIndex);
            mLastStateMetaDumpShapeIndex = shapeIndex;
        }
        if (dumpGraphState && meta != nullptr) {
            MNN_PRINT("MNN_QNN_DUMP state_meta graph=%s meta_add=%zu meta_remove=%zu stateCurrent=%d\n",
                      graphName.c_str(), meta->add, meta->remove, mStateCurrent);
        }
        if (nullptr != meta && mMask.get() != nullptr && mStateMaxSize > 0) {
            auto maskPtr = reinterpret_cast<int16_t*>(mMask->mPtr);
            auto minValueHalf = _floatToHalfBits(mMinValue);
            if (meta->remove > 0) {
                if (meta->remove > mStateCurrent) {
                    MNN_ERROR("QNN: Error: Remove %zu larger than current = %d\n", meta->remove, mStateCurrent);
                    return false;
                }
                mStateCurrent-= meta->remove;
                for (int i=0; i<meta->remove; ++i) {
                    maskPtr[i+mStateCurrent] = minValueHalf;
                }
            }
        }
        if (!mStateInput.empty() && mSeqLen[mShapeIndex] > 1) {
            for (int i=0; i<mStateInput.size(); ++i) {
                if (mStateInput[i].mode != StateTensor::REPLACE) {
                    continue;
                }
                int elementCount = 1;
                for (auto dim : mStateInput[i].shape) {
                    elementCount *= dim;
                }
                ::memset(mStateInput[i].data->mPtr, 0, elementCount * mStateInput[i].elementBytes);
            }
        }
        mRawExecutor->invokModel(shapeIndex);
        _fillSyntheticOutputs();
        if (dumpGraphState) {
            int seqLen = shapeIndex >= 0 && shapeIndex < mSeqLen.size() ? mSeqLen[shapeIndex] : 0;
            _dumpStateBuffers("state_update", graphName, shapeIndex, true, seqLen);
        }
        if (dumpGraphIo) {
            _dumpGraphOutputs(graphName);
        }
        for (int i=0; i<mOutputs.size(); ++i) {
            ctx->backend()->onCopyBuffer(mRealOutputs[i].get(), outputTensor[i]);
        }
        // Update State
        if (!mStateInput.empty()) {
            for (int i=0; i<mStateInput.size(); ++i) {
                auto& input = mStateInput[i];
                if (input.mode != StateTensor::REPLACE) {
                    continue;
                }
                int elementCount = 1;
                for (auto dim : input.shape) {
                    elementCount *= dim;
                }
                ::memcpy(input.data->mPtr, input.update[mShapeIndex]->mPtr, elementCount * input.elementBytes);
            }
        }
        if (nullptr != meta && mMask.get() != nullptr && mStateMaxSize > 0) {
            auto maskPtr = reinterpret_cast<int16_t*>(mMask->mPtr);
            if (meta->add + mStateCurrent > mStateMaxSize) {
                MNN_ERROR("QNN: Error: KV length %zu larger than max size = %d\n", meta->add + mStateCurrent, mStateMaxSize);
                return false;
            }
            for (int i=0; i<meta->add; ++i) {
                maskPtr[i+mStateCurrent] = 0.0f;
            }
            // Temply use StateOutputs[0] size to compute seq_len
            int bytes = 2;
            int seqLen = mSeqLen[mShapeIndex];
            for (int i=0; i<mStateInput.size(); ++i) {
                auto& input = mStateInput[i];
                if (input.mode != StateTensor::APPEND) {
                    continue;
                }
                for (int y=0; y<input.outside; ++y) {
                    auto dstOffset = y * input.inside * mStateMaxSize + mStateCurrent * input.inside;
                    auto srcOffset = y * input.inside * seqLen;
                    bytes = input.elementBytes;
                    auto dst = (uint8_t*)input.data->mPtr + dstOffset * bytes;
                    auto src = (uint8_t*)input.update[mShapeIndex]->mPtr + srcOffset * bytes;
                    ::memcpy(dst, src, meta->add * input.inside * bytes);
                }
            }
            mStateCurrent += meta->add;
        }
        if (dumpGraphState) {
            _dumpStateBuffers("state_store", graphName, shapeIndex, false, mStateCurrent);
        }
        if (_shouldReleaseExecutorPerRun()) {
            _releaseExecutor();
        }
        return true;
    }
};

} // namespace backend
}
}

#endif

namespace MNN {
namespace QNN {

#ifdef ENABLE_QNN_ONLINE_FINALIZE
QnnBackend::QnnBackend(const QnnRuntime* runtime) : Backend(QNN_FORWARD_TYPE), mPower(runtime->mPower) {
    mRuntime = runtime;
    mUseFP16 = (runtime->mPrecision != BackendConfig::Precision_High) ? true : false;
    mPerf = QNNPerf::create(&mRuntime->mQnnInterface);
    if (mPower == BackendConfig::Power_High) {
        mPerf->setPowerConfigBurst();
        mPerf->setRpcLatencyAndPolling();
    }

    // Set mQnnGraphConfig.
    mQnnHtpGraphCustomConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
    mQnnHtpGraphCustomConfig.precision = QNN_PRECISION_FLOAT16;
    mQnnGraphConfig.option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    mQnnGraphConfig.customConfig = &mQnnHtpGraphCustomConfig;
}

QnnBackend::~QnnBackend() {
    clean();
    if (mPower == BackendConfig::Power_High) {
        mPerf->setPowerConfigBalanced();
    }
}

static inline std::map<OpType, QnnBackend::Creator*>* getCreatorMap() {
    static std::once_flag of;
    static std::map<OpType, QnnBackend::Creator*>* ret = nullptr;
    std::call_once(of, [&]() { ret = new std::map<OpType, QnnBackend::Creator*>; });
    return ret;
}

Execution* QnnBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {
    MNN_ASSERT(op != nullptr);
    auto map = getCreatorMap();
    auto iter = map->find(op->type());

    // MNN_PRINT("MNN_QNN::onCreate Type %d, Name %s.\n", op->type(), op->name()->c_str());

    if (iter == map->end()) {
        if(op->name() != nullptr){
            MNN_PRINT("MNN_QNN: Not registered type %d, %s.\n", op->type(), op->name()->c_str());
        } else {
            MNN_PRINT("MNN_QNN: Not registered type %d.\n", op->type());
        }
        return nullptr;
    }

    auto exe = iter->second->onCreate(inputs, outputs, op, this);

    if (nullptr == exe) {
        if(op->name() != nullptr){
            MNN_PRINT("MNN_QNN: Don't support type %d, %s.\n", op->type(), op->name()->c_str());
        } else {
            MNN_PRINT("MNN_QNN: Don't support type %d.\n", op->type());
        }
        return nullptr;
    }

    return exe;
}

bool QnnBackend::addCreator(OpType t, Creator* c) {
    auto map = getCreatorMap();
    if (map->find(t) != map->end()) {
        MNN_PRINT("MNN_QNN: %d type has be added.\n", t);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}


void QnnBackend::onExecuteBegin() const {
    if (mPower == BackendConfig::Power_Normal) {
        mPerf->setPowerConfigBurst();
        mPerf->setRpcLatencyAndPolling();
    }
    return;
}

void QnnBackend::startProfile() const{
    MNN::QNN::doProfile(mRuntime->mQnnInterface, mQnnProfileHandle);
}
const Runtime* QnnBackend::getRuntime() {
    return mRuntime;
}

void QnnBackend::onExecuteEnd() const {
    executeGraph();
    if (mPower == BackendConfig::Power_Normal) {
        mPerf->setPowerConfigBalanced();
    }
    startProfile();
    return;
}

void QnnBackend::onResizeBegin() {
    clean();
    createContextAndGraph();
    return;
}

ErrorCode QnnBackend::onResizeEnd() {
    #ifdef QNN_VERBOSE
    MNN_PRINT("start finalize\n");
    #endif
    buildOutputCast();
    buildOutputDequant();
    finalizeGraph();
    for(auto func : mReleaseFunc){
        func();
    }
    mReleaseFunc.clear();
    #ifdef QNN_VERBOSE
    MNN_PRINT("end finalize\n");
    #endif
    return NO_ERROR;
}

Backend::MemObj* QnnBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
    std::string tName = "QnnTensor_" + std::to_string(mTensorCounter);
    if (TensorUtils::getDescribe(tensor)->index >= 0) {
        tName = std::string("t") + std::to_string(TensorUtils::getDescribe(tensor)->index);
    }

    bool isInput = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
    bool isOutput = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
    bool isConst = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    MNN_ASSERT(!isConst);

    Qnn_TensorType_t tType = QNN_TENSOR_TYPE_NATIVE;
    if (isInput) {
        tType = QNN_TENSOR_TYPE_APP_WRITE;
    }
    if (isOutput) {
        tType = QNN_TENSOR_TYPE_APP_READ;
    }

    Qnn_DataType_t tDataType;
    Qnn_QuantizeParams_t tQuantizeParams{};
    tQuantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    tQuantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    Qnn_ScaleOffset_t tScaleOffsetEncoding;
    tScaleOffsetEncoding.scale = 0.0f;
    tScaleOffsetEncoding.offset = 0;
    auto quant = TensorUtils::getDescribe(tensor)->quantAttr.get();
    bool isQuant = quant != nullptr && TensorUtils::getDescribe(tensor)->applyQuant;
    //MNN_ASSERT((tensor->getType().code == halide_type_float) || (tensor->getType().code == halide_type_int && tensor->getType().bits == 32));
    if (mUseFP16 && tensor->getType().code == halide_type_float) {
        tType = QNN_TENSOR_TYPE_NATIVE;
        tDataType = QNN_DATATYPE_FLOAT_16;
    } else if (tensor->getType().code == halide_type_float) {
        tDataType = QNN_DATATYPE_FLOAT_32;
    } else if (tensor->getType().code == halide_type_int && tensor->getType().bits == 32) {
        tDataType = QNN_DATATYPE_INT_32;
    } else {
        MNN_PRINT("MNN_QNN: Not supported data type in <QnnBackend::onAcquire>.\n");
        return nullptr;
    }
    if(isQuant) {
        tType = QNN_TENSOR_TYPE_NATIVE;
        auto quantType = TensorUtils::getDescribe(tensor)->quantAttr->type;
        if(quantType == DataType_DT_INT8){
            tQuantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
            tQuantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            if(quant->zero != 0){
                MNN_PRINT("MNN_QNN: Not supported asymmetric quant in <QnnBackend::onAcquire>.\n");
                return nullptr;
            }
            tScaleOffsetEncoding.scale = quant->scale;
            tScaleOffsetEncoding.offset = 0;
            tDataType = QNN_DATATYPE_SFIXED_POINT_8;
            if (isOutput) {
                tType = QNN_TENSOR_TYPE_NATIVE;
            }
        }else if(quantType == DataType_DT_INT16){
            // uint16
            tQuantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
            tQuantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            tScaleOffsetEncoding.scale = quant->scale;
            tScaleOffsetEncoding.offset = quant->zero;
            tDataType = QNN_DATATYPE_UFIXED_POINT_16;
            if (isOutput) {
                tType = QNN_TENSOR_TYPE_NATIVE;
            }
        }
    }
    tQuantizeParams.scaleOffsetEncoding = tScaleOffsetEncoding;
    Tensor::DimensionType tensorDimType = tensor->getDimensionType();

    std::vector<int> tDims = tensor->shape();
    if(TensorUtils::getDescribe(tensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4){
        tensorDimType = gQnnTensorDimType;
        std::unique_ptr<Tensor> tempTensor(new Tensor(tensor, tensorDimType, false));
        if (!(tempTensor->shape().empty())) {
            tDims = tempTensor->shape();
        } else {
            tDims = {1};
        }
    }

    std::string suffix = "";
    if(isInput && mUseFP16 && tensor->getType().code == halide_type_float){
        suffix = "_cast";
    }
    if(isOutput && isQuant){
        suffix = "_dequant";
    }
    if(isOutput && mUseFP16 && tensor->getType().code == halide_type_float){
        suffix = "_cast";
    }
    std::shared_ptr<QNNTensorWrapper> qnnTensorWrapper = QNNTensorWrapper::create(tName + suffix, tType, tDataType, tDims, tQuantizeParams);

    Qnn_Tensor_t * qnnTensor = qnnTensorWrapper->getNativeTensor();
    CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnnTensor));
    mQNNTensorWrappers.push_back(qnnTensorWrapper);
    mTensorMap.insert({TensorUtils::getDescribe(tensor), mTensorCounter});

    if (isInput) {
        // create stage tensor to cast
        if (mUseFP16 && tensor->getType().code == halide_type_float) {
            mTensorCounter += 1;
            std::shared_ptr<Tensor> stageTensor;
            stageTensor.reset(Tensor::create<float>(tensor->shape(), nullptr, tensorDimType));
            Qnn_QuantizeParams_t tQuantizeParamstmp = QNN_QUANTIZE_PARAMS_INIT;
            std::shared_ptr<QNNTensorWrapper> qnnCastTensorWrapper = QNNTensorWrapper::create(tName, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, tDims, tQuantizeParamstmp);
            CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnnCastTensorWrapper->getNativeTensor()));
            mInputCastTensorMap.insert({TensorUtils::getDescribe(tensor), {tensor, stageTensor}});
            mQNNTensorWrappers.push_back(qnnCastTensorWrapper);
            mTensorMap.insert({TensorUtils::getDescribe(const_cast<const Tensor*>(stageTensor.get())), mTensorCounter});
            mInputTensorIndexes.push_back(mTensorCounter);
            qnnCastTensorWrapper->alloc(tensorDimType);
            buildInputCast(tensor);
        }else{
            mInputTensorIndexes.push_back(mTensorCounter);
            qnnTensorWrapper->alloc(tensorDimType);
        }
    }
    if (isOutput) {
        if(isQuant){
            mTensorCounter += 1;
            std::shared_ptr<Tensor> stageTensor;
            stageTensor.reset(Tensor::create<float>(tensor->shape(), nullptr, tensorDimType));
            if (tensor->getType().code == halide_type_float) {
                tDataType = QNN_DATATYPE_FLOAT_32;
            } else {
                MNN_PRINT("MNN_QNN: Not supported data type in <QnnBackend::onAcquire>.\n");
                return nullptr;
            }
            Qnn_QuantizeParams_t tQuantizeParamstmp = QNN_QUANTIZE_PARAMS_INIT;
            std::shared_ptr<QNNTensorWrapper> qnnOutputTensorWrapper = QNNTensorWrapper::create(tName, QNN_TENSOR_TYPE_APP_READ, tDataType, tDims, tQuantizeParamstmp);
            CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnnOutputTensorWrapper->getNativeTensor()));
            mDeQuantOutputTensorMap.insert({TensorUtils::getDescribe(tensor), {tensor, stageTensor}});
            mQNNTensorWrappers.push_back(qnnOutputTensorWrapper);
            mTensorMap.insert({TensorUtils::getDescribe(const_cast<const Tensor*>(stageTensor.get())), mTensorCounter});
            mOutputTensorIndexes.push_back(mTensorCounter);
            qnnOutputTensorWrapper->alloc(tensorDimType);
        } else{
            if (mUseFP16 && tensor->getType().code == halide_type_float) {
                mTensorCounter += 1;
                std::shared_ptr<Tensor> stageTensor;
                stageTensor.reset(Tensor::create<float>(tensor->shape(), nullptr, tensorDimType));
                Qnn_QuantizeParams_t tQuantizeParamstmp = QNN_QUANTIZE_PARAMS_INIT;
                std::shared_ptr<QNNTensorWrapper> qnnCastTensorWrapper = QNNTensorWrapper::create(tName, QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32, tDims, tQuantizeParamstmp);
                CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnnCastTensorWrapper->getNativeTensor()));
                mOutputCastTensorMap.insert({TensorUtils::getDescribe(tensor), {tensor, stageTensor}});
                mQNNTensorWrappers.push_back(qnnCastTensorWrapper);
                mTensorMap.insert({TensorUtils::getDescribe(const_cast<const Tensor*>(stageTensor.get())), mTensorCounter});
                mOutputTensorIndexes.push_back(mTensorCounter);
                qnnCastTensorWrapper->alloc(tensorDimType);
            }else{
                mOutputTensorIndexes.push_back(mTensorCounter);
                qnnTensorWrapper->alloc(tensorDimType);
            }
        }
    }

    mTensorCounter += 1;
    #ifdef QNN_VERBOSE
    MNN_PRINT("Total qnn tensor count:%d\n", mTensorCounter);
    #endif
    return new Backend::MemObj();
}


bool QnnBackend::onClearBuffer() {
    return true;
}


void QnnBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
    bool isInput = TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
    bool isOutput = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
    bool isConst = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT || TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    // MNN_ASSERT(!isConst);

    if (isConst) {
        MNN_ASSERT(isInput);
    }

    MNN_ASSERT(isInput || isOutput);

    if (isInput) {
        inputIO(srcTensor, dstTensor);
    } else if (isOutput) {
        outputIO(srcTensor, dstTensor);
    } else {
        // Not support.
    }
}

void QnnBackend::inputIO(const Tensor* srcTensor, const Tensor* dstTensor) const {
    auto iter = mInputCastTensorMap.find(TensorUtils::getDescribe(dstTensor));
    int dstIndex = -1;
    if(iter != mInputCastTensorMap.end()){
        dstIndex = getTensorIdx(iter->second.second.get());
    } else{
        dstIndex = getTensorIdx(dstTensor);
    }
    std::shared_ptr<QNNTensorWrapper> dstQnnTensorWrapper = mQNNTensorWrappers[dstIndex];
    std::shared_ptr<Tensor> dstDataContainer = dstQnnTensorWrapper->getDataContainer();

    bool valid0 = srcTensor->getType().code == halide_type_float;
    bool valid1 = srcTensor->getType().code == halide_type_int && srcTensor->getType().bits == 32;

    // Currently, support float and int input only.
    MNN_ASSERT(valid0 || valid1);

    if(TensorUtils::getDescribe(srcTensor)->dimensionFormat == TensorUtils::getDescribe(dstDataContainer.get())->dimensionFormat){
        ::memcpy(dstDataContainer.get()->host<float>(), srcTensor->host<float>(), srcTensor->elementSize() * sizeof(float));
    }else{
        auto code = CPUTensorConverter::convert(srcTensor, dstDataContainer.get());
        if (NO_ERROR != code) {
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer.\n");
        }
    }
}

void QnnBackend::outputIO(const Tensor* srcTensor, const Tensor* dstTensor) const {
    auto iter = mDeQuantOutputTensorMap.find(TensorUtils::getDescribe(srcTensor));
    int srcIndex = -1;
    if(iter != mDeQuantOutputTensorMap.end()){
        srcIndex = getTensorIdx(iter->second.second.get());
    } else{
        if(mUseFP16){
            auto castIter = mOutputCastTensorMap.find(TensorUtils::getDescribe(srcTensor));
            if(castIter != mOutputCastTensorMap.end()){
                srcIndex = getTensorIdx(castIter->second.second.get());
            }else{
                MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer for cast float to half.\n");
                return;
            }
        }else{
            srcIndex = getTensorIdx(srcTensor);
        }
    }
    std::shared_ptr<QNNTensorWrapper> srcQnnTensorWrapper = mQNNTensorWrappers[srcIndex];
    std::shared_ptr<Tensor> srcDataContainer = srcQnnTensorWrapper->getDataContainer();

    // Currently, support float output only.
    bool valid0 = dstTensor->getType().code == halide_type_float;
    bool valid1 = dstTensor->getType().code == halide_type_int && dstTensor->getType().bits == 32;

    // Currently, support float and int input only.
    MNN_ASSERT(valid0 || valid1);

    if(TensorUtils::getDescribe(dstTensor)->dimensionFormat == TensorUtils::getDescribe(srcDataContainer.get())->dimensionFormat){
        ::memcpy(dstTensor->host<float>(), srcDataContainer.get()->host<float>(), srcTensor->elementSize() * sizeof(float));
    }else{
        auto code = CPUTensorConverter::convert(srcDataContainer.get(), dstTensor);
        if (NO_ERROR != code) {
            MNN_ERROR("MNN_QNN: Error in QNNBackend::onCopyBuffer.\n");
        }
    }
}
bool QnnBackend::useCache() const {
    return mRuntime->mUseCache;
}

void QnnBackend::createContextAndGraph() {
    mRuntime->allocContext();
    const QnnGraph_Config_t * pGraphConfig[] = {&mQnnGraphConfig, nullptr};
    if (mRuntime->mUseCache) {
        CALL_QNN(mRuntime->mQnnInterface.graphRetrieve(mRuntime->mQnnContextHandle, mQnnGraphName.c_str(), &mQnnGraphHandle));
    } else {
        CALL_QNN(mRuntime->mQnnInterface.graphCreate(mRuntime->mQnnContextHandle, mQnnGraphName.c_str(), pGraphConfig, &mQnnGraphHandle));
    }
    MNN_ASSERT(mQnnGraphHandle != nullptr);
}

void QnnBackend::finalizeGraph() {
    // [TODO] Fix this. Add the following branch for empty resize.
    if (mTensorCounter == 0) {
        return;
    }
    #ifdef QNN_VERBOSE
    MNN_PRINT("Total qnn tensor count:%d\n", mTensorCounter);
    #endif

    // Create Prefile Handle
    MNN::QNN::createProfileHandle(mRuntime->mQnnInterface, mRuntime->mQnnBackendHandle, &mQnnProfileHandle);

    CALL_QNN(mRuntime->mQnnInterface.graphFinalize(mQnnGraphHandle, mQnnProfileHandle, mQnnSignalHandle));
}

void QnnBackend::executeGraph() const {
    if (::getenv("MNN_QNN_TRACE_ENTRY") != nullptr) {
        MNN_PRINT("MNN_QNN_TRACE direct_execute graphHandle=%p inputs=%zu outputs=%zu extraInputs=%zu extraOutputs=%zu\n",
                  mQnnGraphHandle, mInputTensorIndexes.size(), mOutputTensorIndexes.size(), mExtraInputs.size(), mExtraOutputs.size());
    }
    std::vector<Qnn_Tensor_t> inputs;
    std::vector<Qnn_Tensor_t> outputs;
    auto dataTypeBytes = [](Qnn_DataType_t dataType) -> size_t {
        switch (dataType) {
            case QNN_DATATYPE_FLOAT_16:
                return sizeof(int16_t);
            case QNN_DATATYPE_FLOAT_32:
                return sizeof(float);
            default:
                return 0;
        }
    };
    auto dumpTensor = [&](const char* prefix, const Qnn_Tensor_t& tensor, int index) {
        std::ostringstream os;
        os << prefix << "[" << index << "]"
           << " name=" << (tensor.v1.name ? tensor.v1.name : "<null>")
           << " type=" << tensor.v1.type
           << " dataType=" << tensor.v1.dataType
           << " memType=" << tensor.v1.memType
           << " rank=" << tensor.v1.rank
           << " dims=[";
        for (uint32_t i = 0; i < tensor.v1.rank; ++i) {
            if (i > 0) {
                os << ",";
            }
            os << tensor.v1.dimensions[i];
        }
        os << "]";
        MNN_ERROR("%s\n", os.str().c_str());
    };
    for (int i = 0; i <  mInputTensorIndexes.size(); i++) {
        inputs.push_back(*(mQNNTensorWrappers[mInputTensorIndexes[i]]->getNativeTensor()));
    }
    for (int j = 0 ; j < mOutputTensorIndexes.size(); j++) {
        outputs.push_back(*(mQNNTensorWrappers[mOutputTensorIndexes[j]]->getNativeTensor()));
    }
    for (const auto& extraInput : mExtraInputs) {
        inputs.push_back(*(extraInput->getNativeTensor()));
    }
    for (const auto& extraOutput : mExtraOutputs) {
        outputs.push_back(*(extraOutput->getNativeTensor()));
    }

    auto execCode = mRuntime->mQnnInterface.graphExecute(mQnnGraphHandle, inputs.data(), inputs.size(), outputs.data(), outputs.size(), mQnnProfileHandle, mQnnSignalHandle);
    if ((execCode & 0xFFFF) != QNN_SUCCESS) {
        MNN_ERROR("MNN_QNN: graphExecute failed code=%lu inputs=%zu outputs=%zu extraInputs=%zu extraOutputs=%zu\n",
                  (unsigned long)execCode, inputs.size(), outputs.size(), mExtraInputs.size(), mExtraOutputs.size());
        for (int i = 0; i < inputs.size(); ++i) {
            dumpTensor("input", inputs[i], i);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            dumpTensor("output", outputs[i], i);
        }
        assert((execCode & 0xFFFF) == QNN_SUCCESS);
    }

    // Direct QNN execution has no plugin-side state manager, so persist LinearAttention-style
    // state explicitly by copying each extra output buffer back into its paired extra input.
    for (int i = 0; i < mExtraInputs.size() && i < mExtraOutputs.size(); ++i) {
        auto src = mExtraOutputs[i]->getNativeTensor();
        uint32_t elementSize = 1;
        for (uint32_t axis = 0; axis < src->v1.rank; ++axis) {
            elementSize *= src->v1.dimensions[axis];
        }
        if (elementSize <= 0) {
            continue;
        }
        auto bytes = static_cast<size_t>(elementSize) * dataTypeBytes(src->v1.dataType);
        MNN_ASSERT(bytes > 0);
        ::memcpy(mExtraStateIoBuffers[i]->ptr(), mExtraStateIoBuffers[mExtraInputs.size() + i]->ptr(), bytes);
    }
}

void QnnBackend::freeContextAndGraph() {
    if (mTensorCounter != 0) {
        mQnnGraphHandle = nullptr;
    }
    mRuntime->freeContext();
}

void QnnBackend::addNodeToGraph(Qnn_OpConfigVersion_t version, const char* nodeName, const char* packageName, const char* nodeType, std::vector<Qnn_Param_t> & params, std::vector<Qnn_Tensor_t> & inputs, std::vector<Qnn_Tensor_t> & outputs) {
    MNN_ASSERT(nodeName != nullptr && packageName != nullptr && nodeType != nullptr && !(inputs.empty()) && !(outputs.empty()));

    Qnn_OpConfig_t opConfig = QNN_OPCONFIG_INIT;
    opConfig.version = version;
    opConfig.v1.name = nodeName;
    opConfig.v1.packageName = packageName;
    opConfig.v1.typeName = nodeType;
    opConfig.v1.numOfParams = params.size();
    opConfig.v1.params = params.data();
    opConfig.v1.numOfInputs = inputs.size();
    opConfig.v1.inputTensors = inputs.data();
    opConfig.v1.numOfOutputs = outputs.size();
    opConfig.v1.outputTensors = outputs.data();

    CALL_QNN(mRuntime->mQnnInterface.backendValidateOpConfig(mRuntime->mQnnBackendHandle, opConfig));

    CALL_QNN(mRuntime->mQnnInterface.graphAddNode(mQnnGraphHandle, opConfig));
}

int QnnBackend::getTensorIdx(const Tensor * tensor) const {
    const Tensor::InsideDescribe::NativeInsideDescribe * tensorKey = TensorUtils::getDescribe(tensor);
    auto iter = mTensorMap.find(tensorKey);
    int idx = -1;
    if (iter == mTensorMap.end()) {
        std::string tName = "QnnTensor_" + std::to_string(mTensorCounter);;
        if (TensorUtils::getDescribe(tensor)->usage != Tensor::InsideDescribe::Usage::CONSTANT) {
            MNN_PRINT("Tensor usage is %d.\n", (int) TensorUtils::getDescribe(tensor)->usage);
        }
        #ifdef QNN_VERBOSE
        MNN_PRINT("qnn tenor usage:%d, dimension:%d\n", TensorUtils::getDescribe(tensor)->usage, tensor->dimensions());
        #endif
        MNN_ASSERT(TensorUtils::getDescribe(tensor)->usage == Tensor::InsideDescribe::Usage::CONSTANT);
        // MNN_ASSERT(tensor->dimensions() <= 2);
        std::vector<uint32_t> tDims = getNHWCShape(tensor);
        Qnn_DataType_t tDataType;
        std::shared_ptr<QNNTensorWrapper> qnnTensorWrapper;
        if (tensor->getType().code == halide_type_int && tensor->getType().bits == 32) {
            tDataType = QNN_DATATYPE_INT_32;
            qnnTensorWrapper = QNNTensorWrapper::createStaticTensor(tName, tDataType, tDims, tensor->host<int>());
        } else if (tensor->getType().code == halide_type_float) {
            tDataType = mUseFP16 ? QNN_DATATYPE_FLOAT_16 : QNN_DATATYPE_FLOAT_32;
            qnnTensorWrapper = QNNTensorWrapper::createStaticFloatTensor(tName, tDataType, tDims, tensor->host<float>());
        } else {
            MNN_ASSERT(false);
        }
        Qnn_Tensor_t * qnnTensor = qnnTensorWrapper->getNativeTensor();
        CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnnTensor));
        mQNNTensorWrappers.push_back(qnnTensorWrapper);
        mTensorMap.insert({tensorKey, mTensorCounter});
        idx = mTensorCounter;
        mTensorCounter += 1;
    } else {
        idx = iter->second;
    }
    return idx;
}

void QnnBackend::addTensor(Qnn_Tensor_t * staticTensor) {
    CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, staticTensor));
}

Qnn_Tensor_t * QnnBackend::getNativeTensor(const Tensor * tensor) {
    int idx = getTensorIdx(tensor);
    return mQNNTensorWrappers[idx]->getNativeTensor();
}

std::shared_ptr<QNNTensorWrapper> QnnBackend::getTensorWrapper(const Tensor * tensor) {
    const Tensor::InsideDescribe::NativeInsideDescribe * tensorKey = TensorUtils::getDescribe(tensor);
    auto iter = mTensorMap.find(tensorKey);
    MNN_ASSERT(iter != mTensorMap.end());
    return mQNNTensorWrappers[iter->second];
}
Qnn_Tensor_t* QnnBackend::getMaskTensor(int maxKVSize) {
    if (mMaskTensor.get() == nullptr) {
        std::vector<int> dimensions = {1, 1, 1, maxKVSize};
        std::shared_ptr<QNNTensorWrapper> tensorWrapper = QNNTensorWrapper::create(gExtraIoPrefix + "_mask", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, dimensions);
        mMaskTensor = tensorWrapper;
        CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, mMaskTensor->getNativeTensor()));
    }
    return mMaskTensor->getNativeTensor();
}

static size_t _qnnDataTypeBytes(Qnn_DataType_t dataType) {
    switch (dataType) {
        case QNN_DATATYPE_FLOAT_16:
            return sizeof(int16_t);
        case QNN_DATATYPE_FLOAT_32:
            return sizeof(float);
        default:
            MNN_ASSERT(false);
            return 0;
    }
}

Qnn_Tensor_t* QnnBackend::addExtraInput(Tensor* tensor, Qnn_DataType_t dataType) {
    auto qnntensor = QNNTensorWrapper::create("", QNN_TENSOR_TYPE_APP_WRITE, dataType, tensor->shape());
    qnntensor->setName(gExtraIoPrefix+"_i" + std::to_string(mExtraInputs.size()));
    qnntensor->getNativeTensor()->v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    auto bytes = (size_t)std::max(1, tensor->elementSize()) * _qnnDataTypeBytes(dataType);
    auto buffer = OnlineRPCBuffer::alloc(bytes);
    MNN_ASSERT(buffer != nullptr);
    buffer->zero();
    MNN_ASSERT(buffer->bind(qnntensor->getNativeTensor(), &mRuntime->mQnnInterface, mRuntime->mQnnContextHandle));
    mExtraStateBuffers[tensor] = buffer;
    mExtraStateIoBuffers.emplace_back(buffer);
    mExtraInputs.emplace_back(qnntensor);
    CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnntensor->getNativeTensor()));

    return qnntensor->getNativeTensor();
}
Qnn_Tensor_t* QnnBackend::addExtraOutput(Tensor* tensor, Qnn_DataType_t dataType) {
    auto qnntensor = QNNTensorWrapper::create("", QNN_TENSOR_TYPE_APP_READ, dataType, tensor->shape());
    qnntensor->setName(gExtraIoPrefix+"_o" + std::to_string(mExtraOutputs.size()));
    qnntensor->getNativeTensor()->v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    auto bytes = (size_t)std::max(1, tensor->elementSize()) * _qnnDataTypeBytes(dataType);
    auto buffer = OnlineRPCBuffer::alloc(bytes);
    MNN_ASSERT(buffer != nullptr);
    buffer->zero();
    mExtraStateIoBuffers.emplace_back(buffer);
    MNN_ASSERT(buffer->bind(qnntensor->getNativeTensor(), &mRuntime->mQnnInterface, mRuntime->mQnnContextHandle));
    mExtraOutputs.emplace_back(qnntensor);
    CALL_QNN(mRuntime->mQnnInterface.tensorCreateGraphTensor(mQnnGraphHandle, qnntensor->getNativeTensor()));
    return qnntensor->getNativeTensor();
}

bool QnnBackend::getUseFP16() const {
    return mUseFP16;
}

void QnnBackend::clean() {
    if (mQnnProfileHandle) {
        mRuntime->mQnnInterface.profileFree(mQnnProfileHandle);
        mQnnProfileHandle = nullptr;
    }
    freeContextAndGraph(); // This function must be called first.
    mTensorCounter = 0;
    mQNNTensorWrappers.clear();
    mTensorMap.clear();
    mInputTensorIndexes.clear();
    mOutputTensorIndexes.clear();
    mDeQuantOutputTensorMap.clear();
    mInputCastTensorMap.clear();
    mOutputCastTensorMap.clear();
    mExtraInputs.clear();
    mExtraOutputs.clear();
    mExtraStateBuffers.clear();
    mExtraStateIoBuffers.clear();
}
void QnnBackend::buildOutputDequant(){
    Qnn_OpConfigVersion_t mOpConfigVersion = QNN_OPCONFIG_VERSION_1;
    std::string mNodeName;
    std::string mPackageName = "qti.aisw";
    std::string mNodeType;
    std::vector<Qnn_Param_t> mParams;
    std::vector<Qnn_Tensor_t> mInputs;
    std::vector<Qnn_Tensor_t> mOutputs;
    for(auto iter : mDeQuantOutputTensorMap){
        mNodeType.clear();
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "Dequantize";
        std::string name = "Dequantize_I_" + std::to_string(getTensorIdx(iter.second.first)) + "_O_" + std::to_string(getTensorIdx(iter.second.second.get()));
        mInputs.push_back(*(getNativeTensor(iter.second.first))); // input
        mOutputs.push_back(*(getNativeTensor(iter.second.second.get()))); // output
        addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
}

void QnnBackend::buildOutputCast(){
    Qnn_OpConfigVersion_t mOpConfigVersion = QNN_OPCONFIG_VERSION_1;
    std::string mNodeName;
    std::string mPackageName = "qti.aisw";
    std::string mNodeType;
    std::vector<Qnn_Param_t> mParams;
    std::vector<Qnn_Tensor_t> mInputs;
    std::vector<Qnn_Tensor_t> mOutputs;
    for(auto iter : mOutputCastTensorMap){
        mNodeType.clear();
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "Cast";
        std::string name = "Cast_I_" + std::to_string(getTensorIdx(iter.second.first)) + "_O_" + std::to_string(getTensorIdx(iter.second.second.get()));
        mInputs.push_back(*(getNativeTensor(iter.second.first))); // input
        mOutputs.push_back(*(getNativeTensor(iter.second.second.get()))); // output
        addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
}

void QnnBackend::buildInputCast(const Tensor *tensor){
    Qnn_OpConfigVersion_t mOpConfigVersion = QNN_OPCONFIG_VERSION_1;
    std::string mNodeName;
    std::string mPackageName = "qti.aisw";
    std::string mNodeType;
    std::vector<Qnn_Param_t> mParams;
    std::vector<Qnn_Tensor_t> mInputs;
    std::vector<Qnn_Tensor_t> mOutputs;
    mNodeType.clear();
    mParams.clear();
    mInputs.clear();
    mOutputs.clear();
    mNodeType = "Cast";
    auto iter = mInputCastTensorMap.find(TensorUtils::getDescribe(tensor));
    if(iter != mInputCastTensorMap.end()){
        std::string name = "Cast_I_" + std::to_string(getTensorIdx(iter->second.second.get())) + "_O_" + std::to_string(getTensorIdx(iter->second.first));
        mInputs.push_back(*(getNativeTensor(iter->second.second.get()))); // input
        mOutputs.push_back(*(getNativeTensor(iter->second.first))); // output
        addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }
}

QnnRuntime::QnnRuntime(const Backend::Info& info, QNN_INTERFACE_VER_TYPE qnnInterface, Qnn_LogHandle_t qnnLogHandle, Qnn_BackendHandle_t qnnBackendHandle, Qnn_DeviceHandle_t qnnDeviceHandle) {
    // MNN_PRINT("QnnRuntime is constructing.\n");
    mInfo = info;
    // Default setting
    mPower = BackendConfig::Power_Normal;
    mMemory = BackendConfig::Memory_Normal;
    mPrecision = BackendConfig::Precision_Normal;
    // User setting
    if (info.user != nullptr) {
        mPrecision = info.user->precision;
        mPower = info.user->power;
        mMemory = info.user->memory;
    }
    mQnnInterface = qnnInterface;
    mQnnLogHandle = qnnLogHandle;
    mQnnBackendHandle = qnnBackendHandle;
    mQnnDeviceHandle = qnnDeviceHandle;
}

QnnRuntime::~QnnRuntime() {
    if (nullptr != mQnnContextHandle) {
        CALL_QNN(mQnnInterface.contextFree(mQnnContextHandle, nullptr));
    }
}
bool QnnRuntime::onSetCache(const void* buffer, size_t size) {
    // TODO: Fix bug and complete
    return false;
    if (nullptr == buffer) {
        return false;
    }
    auto error = mQnnInterface.contextValidateBinary(mQnnBackendHandle, mQnnDeviceHandle, mQnnContextConfig, buffer, size);
    if (QNN_SUCCESS != error) {
        MNN_ERROR("QNN: Failed to validate binary: %d\n", (int) error);
        return false;
    }
    freeContext();
    CALL_QNN(mQnnInterface.contextCreateFromBinary(mQnnBackendHandle, mQnnDeviceHandle, mQnnContextConfig, buffer, size, &mQnnContextHandle, nullptr));
    mUseCache = true;
    return true;
}
void QnnRuntime::allocContext() const {
    CALL_QNN(mQnnInterface.contextCreate(mQnnBackendHandle, mQnnDeviceHandle, mQnnContextConfig, &mQnnContextHandle));
    MNN_ASSERT(mQnnContextHandle != nullptr);
}
void QnnRuntime::freeContext() const {
    if (nullptr != mQnnContextHandle) {
        CALL_QNN(mQnnInterface.contextFree(mQnnContextHandle, nullptr));
        mQnnContextHandle = nullptr;
        mBinaryBuffer.clear();
    }
}

std::pair<const void*, size_t> QnnRuntime::onGetCache() {
    return std::make_pair(nullptr, 0);
    if (!mBinaryBuffer.empty()) {
        return std::make_pair(mBinaryBuffer.data(), mBinaryBuffer.size());
    }
    if (nullptr == mQnnContextHandle) {
        return std::make_pair(nullptr, 0);
    }
    Qnn_ContextBinarySize_t size = 0;
    CALL_QNN(mQnnInterface.contextGetBinarySize(mQnnContextHandle, &size));
    FUNC_PRINT(size);
    if (0 == size) {
        return std::make_pair(nullptr, 0);
    }
    mBinaryBuffer.resize(size);
    Qnn_ContextBinarySize_t writesize = 0;
    CALL_QNN(mQnnInterface.contextGetBinary(mQnnContextHandle, mBinaryBuffer.data(), size, &writesize));
    return std::make_pair(mBinaryBuffer.data(), mBinaryBuffer.size());
}

Backend* QnnRuntime::onCreate(const BackendConfig* config, Backend* origin) const {
    return new QnnBackend(this);
}

QnnRuntime* QnnRuntime::create(const Backend::Info& info) {
    if (QNN::gContext.deviceHandle == nullptr){
        QNN::createQnnContext();
    }
    // Create Interface.
    return new QnnRuntime(info, gContext.interface, gContext.logHandle, gContext.backendHandle, gContext.deviceHandle);
}

// Do nothing
void QnnRuntime::onGabageCollect(int level) {}

Runtime::CompilerType QnnRuntime::onGetCompilerType() const {
    return Compiler_Origin;
}

bool QnnRuntime::onSetCachePath(const char* path, int mode) {
#ifdef ENABLE_QNN_CONVERT_MODE
    MNN_ASSERT(path != nullptr);
    QNNConvertor::OutputDir = std::string(path);
    MNNCreateDir(path);
#endif
    return true;
}

bool QnnRuntime::registerCustomOpPackage(QNN_INTERFACE_VER_TYPE qnnInterface, Qnn_BackendHandle_t backendHandle, const std::string & path, const std::string & interfaceProvider, const std::string & target) {
    if (QNN_GET_ERROR_CODE(qnnInterface.backendRegisterOpPackage(backendHandle, path.c_str(), interfaceProvider.c_str(), target.c_str())) != QNN_SUCCESS) {
        MNN_PRINT("MNN_QNN: Failed to register the Op Package: %s.\n", path.c_str());
        return false;
    }
    return true;
}

class QnnRuntimeCreator : public RuntimeCreator {
public:
    virtual Runtime* onCreate(const Backend::Info& info) const override {
        return QnnRuntime::create(info);
    }
    static bool _supportQuant(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
        auto otype = op->type();
        std::set<OpType> judgOneInputOpTypes = { OpType_Slice, OpType_StridedSlice, OpType_GatherV2, OpType_Reshape, OpType_Unsqueeze, OpType_Flatten, OpType_Squeeze};
        if(judgOneInputOpTypes.find(otype) != judgOneInputOpTypes.end()){
            if (TensorUtils::getDescribe(inputs[0])->quantAttr == nullptr) {
                return false;
            }
        }else{
            for (auto t : inputs) {
                auto des = TensorUtils::getDescribe(t);
                if (des->quantAttr == nullptr) {
                    return false;
                }
            }
        }
        auto quantType = TensorUtils::getDescribe(inputs[0])->quantAttr->type;
        switch (otype) {
            case OpType_Convolution:
            case OpType_ConvolutionDepthwise:
                if (inputs.size() > 1) {
                    return false;
                }
                if (TensorUtils::getDescribe(outputs[0])->quantAttr == nullptr) {
                    return false;
                }
                if (op->main_as_Convolution2D() && op->main_as_Convolution2D()->weight() != nullptr) {
                    return false;
                } else {
                    return true;
                }
            case OpType_ReLU:
                if ((op->main_as_Relu() == nullptr) || op->main_as_Relu()->slope() == 0.f) {
                    return true;
                } else {
                    return false;
                }
            case OpType_LayerNorm:
                if(quantType == DataType_DT_INT16){
                    return true;
                }else{
                    //ToDo :support featuremap int8 quant
                    return false;
                }
            case OpType_Scale:
            case OpType_Attention:
                return false;
            default:
                break;
        }
        return true;
    }
    virtual bool onSetQuantInfo(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) const override {
        if (nullptr == op) {
            return true;
        }
        auto res = _supportQuant(op, inputs, outputs);
        for (auto t : outputs) {
            TensorUtils::getDescribe(t)->applyQuant = res;
        }
        return res;
    }
    virtual bool onValid(Backend::Info& info) const override {
        return true;
    }
    virtual bool onGetDeviceInfo(const std::string& deviceKey, std::string& deviceValue) const override {
        if(deviceKey == "soc_id" && gContext.soc_id != 0) {
            deviceValue = std::to_string(gContext.soc_id);
            return true;
        }
        if(deviceKey == "dsp_arch" && gContext.dsp_arch != 0) {
            deviceValue = "v" + std::to_string(gContext.dsp_arch);
            return true;
        }
        return false;
    }
};
#endif
} // end namespace QNN

void registerQNNRuntimeCreator() {
    bool qnnSymbolReady = true;
#ifndef ENABLE_QNN_CONVERT_MODE
    // check whether the qnn lib is available
    if (!QNN::loadQNNSymbol()) {
        qnnSymbolReady = false;
        MNN_PRINT("MNN_QNN: QNN symbols are not ready during backend registration, plugin path will retry at runtime.\n");
    }
#endif

#ifdef ENABLE_QNN_ONLINE_FINALIZE
    if (qnnSymbolReady) {
        QNN::registerQNNOps();
        MNNInsertExtraRuntimeCreator(QNN_FORWARD_TYPE, new QNN::QnnRuntimeCreator, false);
    }
#endif

#ifdef MNN_WITH_PLUGIN
    rpcmem_init();
    plugin::InferShapeKernelRegister::add("QNN", []() { // NOLINT
        return new plugin::shape_inference::PluginShapeRaw;               // NOLINT
    });
    plugin::ComputeKernelRegistry<plugin::backend::PluginExecuteRaw::KernelT>::add("QNN", []() {
        return new plugin::backend::PluginExecuteRaw;
    });
#endif
}

} // end namespace MNN
