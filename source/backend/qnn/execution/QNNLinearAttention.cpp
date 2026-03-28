#include "QNNLinearAttention.hpp"
#include <cmath>

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

namespace {
static std::vector<uint32_t> toU32(const std::vector<int>& dims) {
    std::vector<uint32_t> res(dims.size());
    for (int i = 0; i < dims.size(); ++i) {
        res[i] = (uint32_t)dims[i];
    }
    return res;
}
}

QNNLinearAttention::QNNLinearAttention(Backend* backend, const Op* op) : QNNCommonExecution(backend, op) {
    auto param = op->main_as_LinearAttentionParam();
    if (nullptr != param) {
        if (nullptr != param->attn_type()) {
            mAttentionType = param->attn_type()->str();
        }
        mNumKHeads = param->num_k_heads();
        mNumVHeads = param->num_v_heads();
        mHeadKDim = param->head_k_dim();
        mHeadVDim = param->head_v_dim();
        mUseQKL2Norm = param->use_qk_l2norm();
    }
}

ErrorCode QNNLinearAttention::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (mAttentionType != "gated_delta_rule") {
        MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
    }
    if (inputs.size() != 4 || outputs.size() != 1) {
        return NOT_SUPPORT;
    }
    if (inputs[3]->host<float>() == nullptr) {
        MNN_ERROR("MNN_QNN: LinearAttention expects constant conv weight.\n");
        return NOT_SUPPORT;
    }

    const int B = inputs[0]->length(0);
    const int D = inputs[0]->length(1);
    const int L = inputs[0]->length(2);
    const int Hk = mNumKHeads;
    const int Hv = mNumVHeads;
    const int dk = mHeadKDim;
    const int dv = mHeadVDim;
    const int keyDim = Hk * dk;
    const int valueDim = Hv * dv;
    const int kernelSize = inputs[3]->length(2);
    const int convStateSize = kernelSize - 1;
    const int totalLen = convStateSize + L;
    const int gqaFactor = Hv > Hk ? Hv / Hk : 1;
    const float eps = 1e-6f;
    const float qScaleValue = 1.0f / std::sqrt((float)dk);
    Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
    const Qnn_DataType_t normDataType = QNN_DATATYPE_FLOAT_32;
    const Qnn_DataType_t recurrentDataType = QNN_DATATYPE_FLOAT_32;

    auto addNode = [&](const std::string& suffix, const std::string& type, std::vector<Qnn_Tensor_t> nodeInputs,
                       std::vector<Qnn_Param_t> nodeParams, std::vector<Qnn_Tensor_t> nodeOutputs) {
        std::string name = mNodeName + "_" + suffix;
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), type.c_str(), nodeParams, nodeInputs, nodeOutputs);
    };
    auto makeStage = [&](const std::string& name, const std::vector<int>& dims) {
        return this->createStageTensor(name, dataType, dims);
    };
    auto makeTypedStage = [&](const std::string& name, const std::vector<int>& dims, Qnn_DataType_t stageType) {
        return this->createStageTensor(name, stageType, dims);
    };
    auto makeScalarInt = [&](const std::string& name, int value) -> std::shared_ptr<QNNParamScalarWrapper> {
        return this->createParamScalar(name, value);
    };
    auto makeAxes = [&](const std::string& name, const std::vector<uint32_t>& axes) -> std::shared_ptr<QNNParamTensorWrapper> {
        return this->createParamTensor(name, QNN_DATATYPE_UINT_32, {(uint32_t)axes.size()}, (void*)axes.data(), name);
    };
    auto addReshape = [&](const std::string& suffix, const Qnn_Tensor_t& in, const Qnn_Tensor_t& out) {
        addNode(suffix, "Reshape", {in}, {}, {out});
    };
    auto addTranspose = [&](const std::string& suffix, const Qnn_Tensor_t& in, const Qnn_Tensor_t& out, const std::vector<uint32_t>& perm) {
        auto permTensor = this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t)perm.size()}, (void*)perm.data(), suffix);
        addNode(suffix, "Transpose", {in}, {*(permTensor->getNativeParam())}, {out});
    };
    auto addConcat = [&](const std::string& suffix, const std::vector<Qnn_Tensor_t>& in, const Qnn_Tensor_t& out, uint32_t axis) {
        auto axisScalar = this->createParamScalar("axis", axis);
        addNode(suffix, "Concat", in, {*(axisScalar->getNativeParam())}, {out});
    };
    auto addSplit = [&](const std::string& suffix, const Qnn_Tensor_t& in, const std::vector<Qnn_Tensor_t>& out, uint32_t axis,
                        const std::vector<uint32_t>& splitIndex) {
        auto axisScalar = this->createParamScalar("axis", axis);
        auto splitTensor = this->createParamTensor("split_index", QNN_DATATYPE_UINT_32, {(uint32_t)splitIndex.size()}, (void*)splitIndex.data(), suffix);
        addNode(suffix, "Split", {in}, {*(axisScalar->getNativeParam()), *(splitTensor->getNativeParam())}, out);
    };
    auto addReduceSum = [&](const std::string& suffix, const Qnn_Tensor_t& in, const Qnn_Tensor_t& out, uint32_t axis, bool keepDims) {
        auto axes = this->createParamTensor("axes", QNN_DATATYPE_UINT_32, {1}, (void*)&axis, suffix);
        auto keep = this->createParamScalar("keep_dims", keepDims);
        addNode(suffix, "ReduceSum", {in}, {*(axes->getNativeParam()), *(keep->getNativeParam())}, {out});
    };
    auto addGather = [&](const std::string& suffix, const Qnn_Tensor_t& in, const Qnn_Tensor_t& out, int axis, int index) {
        auto indexTensor = this->createStaticTensor(suffix + "_index", QNN_DATATYPE_INT_32, {1}, &index);
        auto axisScalar = this->createParamScalar("axis", axis);
        addNode(suffix, "Gather", {in, *(indexTensor->getNativeTensor())}, {*(axisScalar->getNativeParam())}, {out});
    };
    auto addBinary = [&](const std::string& suffix, const char* type, const Qnn_Tensor_t& in0, const Qnn_Tensor_t& in1, const Qnn_Tensor_t& out) {
        addNode(suffix, type, {in0, in1}, {}, {out});
    };
    auto addUnary = [&](const std::string& suffix, const char* type, const Qnn_Tensor_t& in, const Qnn_Tensor_t& out) {
        addNode(suffix, type, {in}, {}, {out});
    };
    auto addCast = [&](const std::string& suffix, const Qnn_Tensor_t& in, const Qnn_Tensor_t& out) {
        addNode(suffix, "Cast", {in}, {}, {out});
    };
    auto addSlice = [&](const std::string& suffix, const Qnn_Tensor_t& in, const Qnn_Tensor_t& out, const std::vector<int>& begin,
                        const std::vector<int>& end) {
        std::vector<int> rangeData(begin.size() * 3);
        for (int i = 0; i < begin.size(); ++i) {
            rangeData[3 * i + 0] = begin[i];
            rangeData[3 * i + 1] = end[i];
            rangeData[3 * i + 2] = 1;
        }
        auto ranges = this->createParamTensor("ranges", QNN_DATATYPE_INT_32, {(uint32_t)begin.size(), 3}, (void*)rangeData.data(), suffix);
        addNode(suffix, "StridedSlice", {in}, {*(ranges->getNativeParam())}, {out});
    };

    std::vector<float> convWeightData(kernelSize * D, 0.0f);
    const float* srcWeight = inputs[3]->host<float>();
    for (int c = 0; c < D; ++c) {
        for (int k = 0; k < kernelSize; ++k) {
            // QNN depthwise conv expects weights packed with channel as the innermost dimension.
            convWeightData[c + D * k] = srcWeight[c * kernelSize + k];
        }
    }
    std::vector<float> zeroBias(D, 0.0f);
    auto convWeight = this->createStaticFloatTensor("conv_weight", dataType, {(uint32_t)1, (uint32_t)kernelSize, (uint32_t)1, (uint32_t)D}, convWeightData.data());
    auto convBias = this->createStaticFloatTensor("conv_bias", dataType, {(uint32_t)D}, zeroBias.data());
    auto qScale = this->createStaticFloatTensor("q_scale", normDataType, {1, 1, 1, 1}, &qScaleValue);
    auto epsTensor = this->createStaticFloatTensor("eps", normDataType, {1, 1, 1, 1}, &eps);

    std::shared_ptr<Tensor> convStateWrap(Tensor::createDevice<float>({B, D, convStateSize}));
    std::shared_ptr<Tensor> recurrentWrap(Tensor::createDevice<float>({B, Hv, dk, dv}));
    Qnn_Tensor_t* convStateIn = mBackend->addExtraInput(convStateWrap.get(), dataType);
    Qnn_Tensor_t* recurrentIn = mBackend->addExtraInput(recurrentWrap.get(), recurrentDataType);
    Qnn_Tensor_t* convStateOut = mBackend->addExtraOutput(convStateWrap.get(), dataType);
    Qnn_Tensor_t* recurrentOut = mBackend->addExtraOutput(recurrentWrap.get(), recurrentDataType);

    auto rawConcat = makeStage("RawConcat", std::vector<int>{B, D, totalLen});
    addConcat("RawConcat", {*(convStateIn), *(mBackend->getNativeTensor(inputs[0]))}, *(rawConcat->getNativeTensor()), 2);
    addSlice("ConvStateOut", *(rawConcat->getNativeTensor()), *convStateOut, {0, 0, L}, {B, D, totalLen});

    auto convTranspose = makeStage("ConvTranspose", std::vector<int>{B, totalLen, D});
    addTranspose("ConvTranspose", *(rawConcat->getNativeTensor()), *(convTranspose->getNativeTensor()), {0, 2, 1});
    auto convInput4D = makeStage("ConvInput4D", std::vector<int>{B, 1, totalLen, D});
    addReshape("ConvInput4D", *(convTranspose->getNativeTensor()), *(convInput4D->getNativeTensor()));

    std::vector<uint32_t> strideData = {1, 1};
    std::vector<uint32_t> padData = {0, 0, 0, 0};
    std::vector<uint32_t> dilationData = {1, 1};
    auto stride = this->createParamTensor("stride", QNN_DATATYPE_UINT_32, {2}, (void*)strideData.data(), "conv");
    auto pad = this->createParamTensor("pad_amount", QNN_DATATYPE_UINT_32, {2, 2}, (void*)padData.data(), "conv");
    auto dilation = this->createParamTensor("dilation", QNN_DATATYPE_UINT_32, {2}, (void*)dilationData.data(), "conv");
    auto convOut4D = makeStage("ConvOut4D", std::vector<int>{B, 1, L, D});
    addNode("DepthWiseConv", "DepthWiseConv2d",
            {*(convInput4D->getNativeTensor()), *(convWeight->getNativeTensor()), *(convBias->getNativeTensor())},
            {*(stride->getNativeParam()), *(pad->getNativeParam()), *(dilation->getNativeParam())},
            {*(convOut4D->getNativeTensor())});

    auto convOut3DTrans = makeStage("ConvOut3DTrans", std::vector<int>{B, L, D});
    addReshape("ConvOut3DTrans", *(convOut4D->getNativeTensor()), *(convOut3DTrans->getNativeTensor()));
    auto convOut = makeStage("ConvOut", std::vector<int>{B, D, L});
    addTranspose("ConvOut", *(convOut3DTrans->getNativeTensor()), *(convOut->getNativeTensor()), {0, 2, 1});
    auto convSigmoid = makeStage("ConvSigmoid", std::vector<int>{B, D, L});
    addUnary("ConvSigmoid", "Sigmoid", *(convOut->getNativeTensor()), *(convSigmoid->getNativeTensor()));
    auto convSilu = makeStage("ConvSilu", std::vector<int>{B, D, L});
    addBinary("ConvSilu", "ElementWiseMultiply", *(convOut->getNativeTensor()), *(convSigmoid->getNativeTensor()), *(convSilu->getNativeTensor()));

    std::vector<std::shared_ptr<QNNTensorWrapper>> splitTensors;
    splitTensors.emplace_back(makeStage("SplitQ", std::vector<int>{B, keyDim, L}));
    splitTensors.emplace_back(makeStage("SplitK", std::vector<int>{B, keyDim, L}));
    splitTensors.emplace_back(makeStage("SplitV", std::vector<int>{B, valueDim, L}));
    addSplit("SplitQKV", *(convSilu->getNativeTensor()),
             {*(splitTensors[0]->getNativeTensor()), *(splitTensors[1]->getNativeTensor()), *(splitTensors[2]->getNativeTensor())},
             1, {(uint32_t)keyDim, (uint32_t)(2 * keyDim)});

    auto q4 = makeStage("Q4", std::vector<int>{B, Hk, dk, L});
    auto k4 = makeStage("K4", std::vector<int>{B, Hk, dk, L});
    auto v4 = makeStage("V4", std::vector<int>{B, Hv, dv, L});
    addReshape("Q4", *(splitTensors[0]->getNativeTensor()), *(q4->getNativeTensor()));
    addReshape("K4", *(splitTensors[1]->getNativeTensor()), *(k4->getNativeTensor()));
    addReshape("V4", *(splitTensors[2]->getNativeTensor()), *(v4->getNativeTensor()));

    auto q = makeStage("Q", std::vector<int>{B, L, Hk, dk});
    auto k = makeStage("K", std::vector<int>{B, L, Hk, dk});
    auto v = makeStage("V", std::vector<int>{B, L, Hv, dv});
    addTranspose("Q", *(q4->getNativeTensor()), *(q->getNativeTensor()), {0, 3, 1, 2});
    addTranspose("K", *(k4->getNativeTensor()), *(k->getNativeTensor()), {0, 3, 1, 2});
    addTranspose("V", *(v4->getNativeTensor()), *(v->getNativeTensor()), {0, 3, 1, 2});

    std::shared_ptr<QNNTensorWrapper> qExpanded = q;
    std::shared_ptr<QNNTensorWrapper> kExpanded = k;
    if (gqaFactor > 1) {
        std::vector<std::shared_ptr<QNNTensorWrapper>> qSplit(Hk), kSplit(Hk);
        std::vector<Qnn_Tensor_t> qSplitTensors, kSplitTensors, qRepeatInputs, kRepeatInputs;
        for (int i = 0; i < Hk; ++i) {
            qSplit[i] = makeStage("QHead" + std::to_string(i), std::vector<int>{B, L, 1, dk});
            kSplit[i] = makeStage("KHead" + std::to_string(i), std::vector<int>{B, L, 1, dk});
            qSplitTensors.push_back(*(qSplit[i]->getNativeTensor()));
            kSplitTensors.push_back(*(kSplit[i]->getNativeTensor()));
        }
        std::vector<uint32_t> splitIndex(Hk - 1);
        for (int i = 0; i < Hk - 1; ++i) {
            splitIndex[i] = i + 1;
        }
        addSplit("SplitQHead", *(q->getNativeTensor()), qSplitTensors, 2, splitIndex);
        addSplit("SplitKHead", *(k->getNativeTensor()), kSplitTensors, 2, splitIndex);
        for (int i = 0; i < Hk; ++i) {
            for (int j = 0; j < gqaFactor; ++j) {
                qRepeatInputs.push_back(*(qSplit[i]->getNativeTensor()));
                kRepeatInputs.push_back(*(kSplit[i]->getNativeTensor()));
            }
        }
        qExpanded = makeStage("QExpanded", std::vector<int>{B, L, Hv, dk});
        kExpanded = makeStage("KExpanded", std::vector<int>{B, L, Hv, dk});
        addConcat("ConcatQHead", qRepeatInputs, *(qExpanded->getNativeTensor()), 2);
        addConcat("ConcatKHead", kRepeatInputs, *(kExpanded->getNativeTensor()), 2);
    }

    auto normalizeVec = [&](const std::string& prefix, std::shared_ptr<QNNTensorWrapper> inputTensor) -> std::shared_ptr<QNNTensorWrapper> {
        std::shared_ptr<QNNTensorWrapper> normInput = inputTensor;
        if (inputTensor->getNativeTensor()->v1.dataType != normDataType) {
            normInput = makeTypedStage(prefix + "_cast_in", std::vector<int>{B, L, Hv, dk}, normDataType);
            addCast(prefix + "_cast_in", *(inputTensor->getNativeTensor()), *(normInput->getNativeTensor()));
        }
        auto square = makeTypedStage(prefix + "_square", std::vector<int>{B, L, Hv, dk}, normDataType);
        addBinary(prefix + "_square", "ElementWiseMultiply", *(normInput->getNativeTensor()), *(normInput->getNativeTensor()), *(square->getNativeTensor()));
        auto sum = makeTypedStage(prefix + "_sum", std::vector<int>{B, L, Hv, 1}, normDataType);
        addReduceSum(prefix + "_sum", *(square->getNativeTensor()), *(sum->getNativeTensor()), 3, true);
        auto plusEps = makeTypedStage(prefix + "_plus_eps", std::vector<int>{B, L, Hv, 1}, normDataType);
        addBinary(prefix + "_plus_eps", "ElementWiseAdd", *(sum->getNativeTensor()), *(epsTensor->getNativeTensor()), *(plusEps->getNativeTensor()));
        auto sqrt = makeTypedStage(prefix + "_sqrt", std::vector<int>{B, L, Hv, 1}, normDataType);
        addUnary(prefix + "_sqrt", "ElementWiseSquareRoot", *(plusEps->getNativeTensor()), *(sqrt->getNativeTensor()));
        auto out = makeTypedStage(prefix + "_norm", std::vector<int>{B, L, Hv, dk}, normDataType);
        addBinary(prefix + "_norm", "ElementWiseDivide", *(normInput->getNativeTensor()), *(sqrt->getNativeTensor()), *(out->getNativeTensor()));
        return out;
    };

    if (mUseQKL2Norm) {
        qExpanded = normalizeVec("QNorm", qExpanded);
        kExpanded = normalizeVec("KNorm", kExpanded);
    }
    std::shared_ptr<QNNTensorWrapper> qScaledInput = qExpanded;
    if (qScaledInput->getNativeTensor()->v1.dataType != normDataType) {
        qScaledInput = makeTypedStage("QScaledInput", std::vector<int>{B, L, Hv, dk}, normDataType);
        addCast("QScaledInput", *(qExpanded->getNativeTensor()), *(qScaledInput->getNativeTensor()));
    }
    auto qScaled = makeTypedStage("QScaled", std::vector<int>{B, L, Hv, dk}, normDataType);
    addBinary("QScaled", "ElementWiseMultiply", *(qScaledInput->getNativeTensor()), *(qScale->getNativeTensor()), *(qScaled->getNativeTensor()));

    std::shared_ptr<QNNTensorWrapper> stateCur;
    if (L == 1) {
        stateCur = makeTypedStage("StateCur", std::vector<int>{B, Hv, dk, dv}, recurrentDataType);
        addReshape("StateCur", *recurrentIn, *(stateCur->getNativeTensor()));
    } else {
        stateCur = makeTypedStage("StateCurZero", std::vector<int>{B, Hv, dk, dv}, recurrentDataType);
        auto zeroState = this->createStaticFloatTensor("zero_state", recurrentDataType, {(uint32_t)1}, zeroBias.data());
        addBinary("StateCurZero", "ElementWiseMultiply", *recurrentIn, *(zeroState->getNativeTensor()), *(stateCur->getNativeTensor()));
    }

    std::vector<Qnn_Tensor_t> stepOutputs;
    for (int t = 0; t < L; ++t) {
        auto qStep = makeTypedStage("QStep" + std::to_string(t), std::vector<int>{B, 1, Hv, dk}, recurrentDataType);
        auto kStep = makeTypedStage("KStep" + std::to_string(t), std::vector<int>{B, 1, Hv, dk}, recurrentDataType);
        auto vStep = makeStage("VStep" + std::to_string(t), std::vector<int>{B, 1, Hv, dv});
        auto gStep = makeStage("GStep" + std::to_string(t), std::vector<int>{B, 1, Hv});
        auto betaStep = makeStage("BetaStep" + std::to_string(t), std::vector<int>{B, 1, Hv});
        addGather("GatherQ" + std::to_string(t), *(qScaled->getNativeTensor()), *(qStep->getNativeTensor()), 1, t);
        addGather("GatherK" + std::to_string(t), *(kExpanded->getNativeTensor()), *(kStep->getNativeTensor()), 1, t);
        addGather("GatherV" + std::to_string(t), *(v->getNativeTensor()), *(vStep->getNativeTensor()), 1, t);
        addGather("GatherG" + std::to_string(t), *(mBackend->getNativeTensor(inputs[1])), *(gStep->getNativeTensor()), 1, t);
        addGather("GatherBeta" + std::to_string(t), *(mBackend->getNativeTensor(inputs[2])), *(betaStep->getNativeTensor()), 1, t);

        auto vStepFp32 = makeTypedStage("VStepFp32" + std::to_string(t), std::vector<int>{B, 1, Hv, dv}, recurrentDataType);
        auto gStepFp32 = makeTypedStage("GStepFp32" + std::to_string(t), std::vector<int>{B, 1, Hv}, recurrentDataType);
        auto betaStepFp32 = makeTypedStage("BetaStepFp32" + std::to_string(t), std::vector<int>{B, 1, Hv}, recurrentDataType);
        addCast("VStepFp32" + std::to_string(t), *(vStep->getNativeTensor()), *(vStepFp32->getNativeTensor()));
        addCast("GStepFp32" + std::to_string(t), *(gStep->getNativeTensor()), *(gStepFp32->getNativeTensor()));
        addCast("BetaStepFp32" + std::to_string(t), *(betaStep->getNativeTensor()), *(betaStepFp32->getNativeTensor()));

        auto gState = makeTypedStage("GState" + std::to_string(t), std::vector<int>{B, Hv, 1, 1}, recurrentDataType);
        auto betaVec = makeTypedStage("BetaVec" + std::to_string(t), std::vector<int>{B, Hv, 1}, recurrentDataType);
        auto qVec = makeTypedStage("QVec" + std::to_string(t), std::vector<int>{B, Hv, dk, 1}, recurrentDataType);
        auto kVec = makeTypedStage("KVec" + std::to_string(t), std::vector<int>{B, Hv, dk, 1}, recurrentDataType);
        auto vVec = makeTypedStage("VVec" + std::to_string(t), std::vector<int>{B, Hv, dv}, recurrentDataType);
        addReshape("GState" + std::to_string(t), *(gStepFp32->getNativeTensor()), *(gState->getNativeTensor()));
        addReshape("BetaVec" + std::to_string(t), *(betaStepFp32->getNativeTensor()), *(betaVec->getNativeTensor()));
        addReshape("QVec" + std::to_string(t), *(qStep->getNativeTensor()), *(qVec->getNativeTensor()));
        addReshape("KVec" + std::to_string(t), *(kStep->getNativeTensor()), *(kVec->getNativeTensor()));
        addReshape("VVec" + std::to_string(t), *(vStepFp32->getNativeTensor()), *(vVec->getNativeTensor()));

        auto decay = makeTypedStage("Decay" + std::to_string(t), std::vector<int>{B, Hv, 1, 1}, recurrentDataType);
        addUnary("Decay" + std::to_string(t), "ElementWiseExp", *(gState->getNativeTensor()), *(decay->getNativeTensor()));
        auto stateDecay = makeTypedStage("StateDecay" + std::to_string(t), std::vector<int>{B, Hv, dk, dv}, recurrentDataType);
        addBinary("StateDecay" + std::to_string(t), "ElementWiseMultiply", *(stateCur->getNativeTensor()), *(decay->getNativeTensor()), *(stateDecay->getNativeTensor()));

        auto weightedState = makeTypedStage("WeightedState" + std::to_string(t), std::vector<int>{B, Hv, dk, dv}, recurrentDataType);
        addBinary("WeightedState" + std::to_string(t), "ElementWiseMultiply", *(stateDecay->getNativeTensor()), *(kVec->getNativeTensor()), *(weightedState->getNativeTensor()));
        auto vPred = makeTypedStage("VPred" + std::to_string(t), std::vector<int>{B, Hv, dv}, recurrentDataType);
        addReduceSum("VPred" + std::to_string(t), *(weightedState->getNativeTensor()), *(vPred->getNativeTensor()), 2, false);

        auto diff = makeTypedStage("Diff" + std::to_string(t), std::vector<int>{B, Hv, dv}, recurrentDataType);
        addBinary("Diff" + std::to_string(t), "ElementWiseSubtract", *(vVec->getNativeTensor()), *(vPred->getNativeTensor()), *(diff->getNativeTensor()));
        auto delta = makeTypedStage("Delta" + std::to_string(t), std::vector<int>{B, Hv, dv}, recurrentDataType);
        addBinary("Delta" + std::to_string(t), "ElementWiseMultiply", *(betaVec->getNativeTensor()), *(diff->getNativeTensor()), *(delta->getNativeTensor()));
        auto deltaOuter = makeTypedStage("DeltaOuter" + std::to_string(t), std::vector<int>{B, Hv, 1, dv}, recurrentDataType);
        addReshape("DeltaOuter" + std::to_string(t), *(delta->getNativeTensor()), *(deltaOuter->getNativeTensor()));
        auto update = makeTypedStage("Update" + std::to_string(t), std::vector<int>{B, Hv, dk, dv}, recurrentDataType);
        addBinary("Update" + std::to_string(t), "ElementWiseMultiply", *(kVec->getNativeTensor()), *(deltaOuter->getNativeTensor()), *(update->getNativeTensor()));

        auto stateNext = (t == L - 1) ? std::shared_ptr<QNNTensorWrapper>() : makeTypedStage("StateNext" + std::to_string(t), std::vector<int>{B, Hv, dk, dv}, recurrentDataType);
        Qnn_Tensor_t* stateNextTensor = (t == L - 1) ? recurrentOut : stateNext->getNativeTensor();
        addBinary("StateNext" + std::to_string(t), "ElementWiseAdd", *(stateDecay->getNativeTensor()), *(update->getNativeTensor()), *stateNextTensor);

        auto weightedQuery = makeTypedStage("WeightedQuery" + std::to_string(t), std::vector<int>{B, Hv, dk, dv}, recurrentDataType);
        addBinary("WeightedQuery" + std::to_string(t), "ElementWiseMultiply", *stateNextTensor, *(qVec->getNativeTensor()), *(weightedQuery->getNativeTensor()));
        auto outputVec = makeTypedStage("OutputVec" + std::to_string(t), std::vector<int>{B, Hv, dv}, recurrentDataType);
        addReduceSum("OutputVec" + std::to_string(t), *(weightedQuery->getNativeTensor()), *(outputVec->getNativeTensor()), 2, false);
        auto outputStepFp32 = makeTypedStage("OutputStepFp32" + std::to_string(t), std::vector<int>{B, 1, Hv, dv}, recurrentDataType);
        addReshape("OutputStepFp32" + std::to_string(t), *(outputVec->getNativeTensor()), *(outputStepFp32->getNativeTensor()));
        auto outputStep = makeTypedStage("OutputStep" + std::to_string(t), std::vector<int>{B, 1, Hv, dv}, dataType);
        addCast("OutputStep" + std::to_string(t), *(outputStepFp32->getNativeTensor()), *(outputStep->getNativeTensor()));
        stepOutputs.push_back(*(outputStep->getNativeTensor()));

        if (t != L - 1) {
            stateCur = stateNext;
        }
    }

    if (L == 1) {
        addReshape("FinalOutput", stepOutputs[0], *(mBackend->getNativeTensor(outputs[0])));
    } else {
        addConcat("FinalOutput", stepOutputs, *(mBackend->getNativeTensor(outputs[0])), 1);
    }
    return NO_ERROR;
}

class QNNLinearAttentionCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                         const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_LinearAttentionParam();
        if (nullptr == param || nullptr == param->attn_type()) {
            return nullptr;
        }
        if (param->attn_type()->str() != "gated_delta_rule") {
            return nullptr;
        }
        return new QNNLinearAttention(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNLinearAttentionCreator, OpType_LinearAttention)

#endif
} // namespace QNN
} // namespace MNN
