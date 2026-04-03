#ifndef MNN_QNNLINEARATTENTION_HPP
#define MNN_QNNLINEARATTENTION_HPP

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNLinearAttention : public QNNCommonExecution {
public:
    QNNLinearAttention(Backend* backend, const Op* op);
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    std::string mAttentionType;
    int mNumKHeads = 0;
    int mNumVHeads = 0;
    int mHeadKDim = 0;
    int mHeadVDim = 0;
    bool mUseQKL2Norm = false;
};

#endif
} // namespace QNN
} // namespace MNN

#endif
