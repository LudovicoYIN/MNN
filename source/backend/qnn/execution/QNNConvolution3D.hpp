//
//  QNNConvolution.hpp
//  MNN
//
//  Created by MNN on b'2026/01/29'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNCONVOLUTION3D_HPP
#define MNN_QNNCONVOLUTION3D_HPP

#include "QNNCommonExecution.hpp"
#include "QnnTypes.h"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNConvolution3D : public QNNCommonExecution {
public:
    QNNConvolution3D(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    template <typename T>
    void convertWeight(const T * src, T * dst, int oc, int ic, int kernelH, int kernelW, int kernelD) {
        for (int o = 0; o < oc; o++) {
            for (int i = 0; i < ic; i++) {
                for(int d = 0; d < kernelD; d++){
                    for (int h = 0; h < kernelH; h++) {
                        for (int w = 0; w < kernelW; w++) {
                            uint32_t srcOffset = w + kernelW * (h + kernelH * (d + kernelD * (i + ic * o)));
                            uint32_t dstOffset = o + oc * (i + ic * (w + kernelW * (h + kernelH * d)));
                            dst[dstOffset] = src[srcOffset];
                        }
                    }
                }
            }
        }
    }
    bool createWeightAndBias(const Tensor *input, int oc, int ic, int kernelD, int kernelH, int kernelW, int group);
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif
