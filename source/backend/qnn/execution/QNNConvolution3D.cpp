//
//  QNNConvolution.cpp
//  MNN
//
//  Created by MNN on b'2026/01/29'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNConvolution3D.hpp"
#include <cmath>

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNConvolution3D::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto conv3D     = mOp->main_as_Convolution3D();
    auto common     = conv3D->common();
    Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
    int n;
    int ih, iw, ic, id;
    int oh, ow, oc, od;
    int kernelH, kernelW, kernelD;
    int strideH, strideW, strideD;
    int padD0, padD1, padH0, padH1, padW0, padW1;
    int dilationH, dilationW, dilationD;
    int group;
    auto inputShape = inputs[0]->shape();
    auto outputSHape = outputs[0]->shape();

    // compute shape
    {
        n = inputShape[0];
        ic = inputShape[1]; id = inputShape[2]; ih = inputShape[3]; iw = inputShape[4];
        oc = outputSHape[1]; od = outputSHape[2]; oh = outputSHape[3]; ow = outputSHape[4];
        kernelD = common->kernels()->data()[0]; kernelH = common->kernels()->data()[1]; kernelW = common->kernels()->data()[2];
        strideD = common->strides()->data()[0]; strideH = common->strides()->data()[1]; strideW = common->strides()->data()[2];
        dilationD = common->dilates()->data()[0]; dilationH = common->dilates()->data()[1]; dilationW = common->dilates()->data()[2];
        auto pads = ConvolutionCommon::convolution3DPadFull(inputs[0], outputs[0], common);
        padD0 = std::get<0>(pads); padD1 = std::get<1>(pads); padH0 = std::get<2>(pads); padH1 = std::get<3>(pads); padW0 = std::get<4>(pads); padW1 = std::get<5>(pads);
        group = common->group();
    }

    // create all tensors and params
    {
        std::vector<uint32_t> strideData = {(uint32_t)strideD, (uint32_t)strideH, (uint32_t)strideW};
        std::vector<uint32_t> padAmountData = {(uint32_t)padD0, (uint32_t)padD1, (uint32_t)padH0, (uint32_t)padH1, (uint32_t)padW0, (uint32_t)padW1};
        std::vector<uint32_t> dilationData = {(uint32_t)dilationD, (uint32_t)dilationH, (uint32_t)dilationW};
        this->createParamTensor("stride", QNN_DATATYPE_UINT_32, {3}, (void *)strideData.data());
        this->createParamTensor("pad_amount", QNN_DATATYPE_UINT_32, {3  , 2}, (void *)padAmountData.data());
        this->createParamTensor("dilation", QNN_DATATYPE_UINT_32, {3}, (void *)dilationData.data());
        this->createParamScalar("group", (uint32_t)group);
    }

    this->createWeightAndBias(inputs[0], oc, ic, kernelD, kernelH, kernelW, group);
    if (common->relu() || common->relu6()) {
        this->createStageTensor("ReluTensor", dataType, getNHWCShape(outputs[0]), outputs[0]);
    }

    // add nodes
    {
        if (common->relu() || common->relu6()) {
            // Stage one
            {
                mNodeType = "Conv3d";
                std::string name = mNodeName + "_conv3d";
                mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // stride
                mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // pad_amount
                mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // dilation
                mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // group
        
                mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input
                mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // weight
                mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // bias
        
                mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // stage tensor
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }

            // Stage two
            {
                mNodeType.clear();
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                mNodeType = common->relu6() ? "Relu6" : "Relu";
                std::string name = mNodeName + "_relu";
                
                mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // stage tensor
                mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // output
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }

        } else {
            mNodeType = "Conv3d";
            mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // stride
            mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // pad_amount
            mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // dilation
            mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // group

            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // weight
            mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // bias

            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // output
            mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }

    }
    return NO_ERROR;
}

bool QNNConvolution3D::createWeightAndBias(const Tensor *input, int oc, int ic, int kernelD, int kernelH, int kernelW, int group) {
    {
        auto conv3d = mOp->main_as_Convolution3D();
        std::vector<float> weightData;
        const float* source = nullptr;
        int weightElementNum = 0;
        source = conv3d->weight()->data();
        weightElementNum = conv3d->weight()->size();
        // oc ic kd kh kw ---> kd kh kw ic oc
        weightData.resize(weightElementNum);
        convertWeight(source, (float *) weightData.data(), oc, ic/group, kernelH, kernelW, kernelD);
        Qnn_DataType_t floatDatatype = QNN_DATATYPE_FLOAT_32;
        if(mBackend->getUseFP16()){
            floatDatatype = QNN_DATATYPE_FLOAT_16;
        }
        this->createStaticFloatTensor("weight", floatDatatype, {(uint32_t)kernelD, (uint32_t)kernelH, (uint32_t)kernelW, (uint32_t)ic / (uint32_t)group, (uint32_t)oc}, weightData.data());
    }
    {
        std::vector<float> biasData;
        biasData.resize(oc, 0);
        auto bias = mOp->main_as_Convolution3D()->bias();
        if (nullptr != bias) {
            ::memcpy((void *)biasData.data(), (void *)bias->data(), oc * sizeof(float));
        }
        Qnn_DataType_t floatDatatype = QNN_DATATYPE_FLOAT_32;
        if(mBackend->getUseFP16()){
            floatDatatype = QNN_DATATYPE_FLOAT_16;
        }
        this->createStaticFloatTensor("bias", floatDatatype, {(uint32_t)oc}, biasData.data());
    }
    return NO_ERROR;
}

class QNNConvolution3DCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        if (inputs.size() > 1) {
            MNN_ERROR("QNN only support single conv input\n");
            return nullptr;
        }
        return new QNNConvolution3D(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNConvolution3DCreator, OpType_Convolution3D)
#endif
} // end namespace QNN
} // end namespace MNN
