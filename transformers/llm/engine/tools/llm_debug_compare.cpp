#include "llm/llm.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MNN::Transformer;
using namespace MNN::Express;

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

struct TopKEntry {
    int index = -1;
    float value = 0.0f;
};

using LlmPtr = std::unique_ptr<Llm, void (*)(Llm*)>;

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

static std::string idsToString(const std::vector<int>& ids, size_t maxCount = 64) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < ids.size() && i < maxCount; ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << ids[i];
    }
    if (ids.size() > maxCount) {
        os << ", ...";
    }
    os << "]";
    return os.str();
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
    const float* ref = refVar->readMap<float>();
    const float* cand = candVar->readMap<float>();
    auto result = compareFloatData(ref, cand, refInfo->size);
    std::cout << "[" << label << "] size=" << result.size
              << " maxAbs=" << std::fixed << std::setprecision(6) << result.maxAbs
              << " meanAbs=" << result.meanAbs
              << " rmse=" << result.rmse
              << " cosine=" << result.cosine
              << " cand_nan=" << result.nanCount
              << " cand_inf=" << result.infCount
              << " cand_zero=" << result.zeroCount << "\n";
}

static void printIntCompare(const std::string& label, VARP refVar, VARP candVar) {
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
    const int* ref = refVar->readMap<int>();
    const int* cand = candVar->readMap<int>();
    int mismatch = 0;
    for (int i = 0; i < refInfo->size; ++i) {
        if (ref[i] != cand[i]) {
            ++mismatch;
        }
    }
    std::cout << "[" << label << "] size=" << refInfo->size
              << " mismatch=" << mismatch;
    if (refInfo->size <= 16) {
        std::cout << " ref=" << shapeString(refVar) << " values=[";
        for (int i = 0; i < refInfo->size; ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << ref[i];
        }
        std::cout << "]";
    }
    std::cout << "\n";
}

static std::pair<const float*, size_t> alignCandidate(const float* ref, size_t refSize,
                                                      const float* cand, size_t candSize,
                                                      int& factor, bool& usedLastSlice) {
    factor = 1;
    usedLastSlice = false;
    if (ref == nullptr || cand == nullptr || refSize == 0) {
        return {cand, candSize};
    }
    if (candSize == refSize) {
        return {cand, candSize};
    }
    if (candSize > refSize && candSize % refSize == 0) {
        factor = static_cast<int>(candSize / refSize);
        usedLastSlice = true;
        return {cand + candSize - refSize, refSize};
    }
    return {cand, candSize};
}

static std::vector<TopKEntry> topK(const float* data, size_t size, int k) {
    std::vector<TopKEntry> values;
    values.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        TopKEntry entry;
        entry.index = static_cast<int>(i);
        entry.value = data[i];
        values.push_back(entry);
    }
    if (k < values.size()) {
        std::partial_sort(values.begin(), values.begin() + k, values.end(),
                          [](const TopKEntry& lhs, const TopKEntry& rhs) { return lhs.value > rhs.value; });
        values.resize(k);
    } else {
        std::sort(values.begin(), values.end(),
                  [](const TopKEntry& lhs, const TopKEntry& rhs) { return lhs.value > rhs.value; });
    }
    return values;
}

static void printTopK(const std::string& title, const float* data, size_t size, Llm* llm, int k = 5) {
    if (data == nullptr || size == 0 || llm == nullptr) {
        std::cout << title << ": unavailable\n";
        return;
    }
    auto entries = topK(data, size, k);
    std::cout << title << ":\n";
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& entry = entries[i];
        std::cout << "  [" << i << "] id=" << entry.index
                  << " logit=" << std::fixed << std::setprecision(6) << entry.value
                  << " piece=" << escapePiece(llm->tokenizer_decode(entry.index)) << "\n";
    }
}

static int chooseTop1(const float* data, size_t size) {
    if (data == nullptr || size == 0) {
        return -1;
    }
    int bestIndex = 0;
    float bestValue = data[0];
    for (size_t i = 1; i < size; ++i) {
        if (data[i] > bestValue) {
            bestValue = data[i];
            bestIndex = static_cast<int>(i);
        }
    }
    return bestIndex;
}

static void printLogitCompare(const std::string& label, VARP refVar, VARP candVar, Llm* llm) {
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
    const float* ref = refVar->readMap<float>();
    const float* cand = candVar->readMap<float>();
    int factor = 1;
    bool aligned = false;
    auto alignedCand = alignCandidate(ref, refInfo->size, cand, candInfo->size, factor, aligned);
    if (aligned) {
        std::cout << "[" << label << "] align candidate logits by taking last slice from factor=" << factor << "\n";
    } else if (alignedCand.second != refInfo->size) {
        std::cout << "[" << label << "] incompatible sizes ref=" << refInfo->size
                  << " cand=" << candInfo->size << "\n";
        return;
    }
    auto result = compareFloatData(ref, alignedCand.first, refInfo->size);
    const int refTop1 = chooseTop1(ref, refInfo->size);
    const int candTop1 = chooseTop1(alignedCand.first, alignedCand.second);
    std::cout << "[" << label << "] logits size=" << refInfo->size
              << " top1(ref=" << refTop1 << ",cand=" << candTop1 << ")"
              << " maxAbs=" << std::fixed << std::setprecision(6) << result.maxAbs
              << " meanAbs=" << result.meanAbs
              << " rmse=" << result.rmse
              << " cosine=" << result.cosine
              << " cand_nan=" << result.nanCount
              << " cand_inf=" << result.infCount
              << " cand_zero=" << result.zeroCount << "\n";
    printTopK("  ref top5", ref, refInfo->size, llm);
    printTopK("  cand top5", alignedCand.first, alignedCand.second, llm);
}

static std::vector<int> parseLengths(const std::string& spec) {
    std::vector<int> lengths;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        lengths.push_back(std::atoi(item.c_str()));
    }
    return lengths;
}

static std::vector<int> expandTokens(const std::vector<int>& base, int targetSize) {
    std::vector<int> ids;
    if (base.empty() || targetSize <= 0) {
        return ids;
    }
    ids.reserve(targetSize);
    while (ids.size() < static_cast<size_t>(targetSize)) {
        const size_t remaining = static_cast<size_t>(targetSize) - ids.size();
        const size_t copyCount = std::min(remaining, base.size());
        ids.insert(ids.end(), base.begin(), base.begin() + copyCount);
    }
    return ids;
}

static bool loadPair(LlmPtr& cpu, LlmPtr& qnn,
                     const std::string& cpuConfig,
                     const std::string& qnnConfig) {
    cpu = makeLlm(cpuConfig);
    qnn = makeLlm(qnnConfig);
    if (!cpu || !qnn) {
        std::cerr << "Failed to create LLM instances.\n";
        return false;
    }
    if (!cpu->load()) {
        std::cerr << "Failed to load CPU config: " << cpuConfig << "\n";
        return false;
    }
    if (!qnn->load()) {
        std::cerr << "Failed to load QNN config: " << qnnConfig << "\n";
        return false;
    }
    cpu->reset();
    qnn->reset();
    return true;
}

static void runPromptCompare(Llm* cpu, Llm* qnn, const std::vector<int>& inputIds, int decodeSteps) {
    std::cout << "prompt tokens=" << inputIds.size() << "\n";
    std::cout << "input ids=" << idsToString(inputIds) << "\n";

    auto cpuEmb = cpu->embedding(inputIds);
    auto qnnEmb = qnn->embedding(inputIds);
    printFloatCompare("prefill_input_embeds", cpuEmb, qnnEmb);

    auto cpuMask = cpu->gen_attention_mask(static_cast<int>(inputIds.size()));
    auto qnnMask = qnn->gen_attention_mask(static_cast<int>(inputIds.size()));
    printFloatCompare("prefill_attention_mask", cpuMask, qnnMask);

    auto cpuPos = cpu->gen_position_ids(static_cast<int>(inputIds.size()));
    auto qnnPos = qnn->gen_position_ids(static_cast<int>(inputIds.size()));
    printIntCompare("prefill_position_ids", cpuPos, qnnPos);

    cpu->reset();
    qnn->reset();
    auto cpuPrefill = cpu->forward(inputIds);
    auto qnnPrefill = qnn->forward(inputIds);
    printLogitCompare("prefill", cpuPrefill, qnnPrefill, cpu);

    VARP cpuLogits = cpuPrefill;
    for (int step = 0; step < decodeSteps; ++step) {
        if (cpuLogits == nullptr || cpuLogits->getInfo() == nullptr) {
            std::cout << "[decode] stop because CPU logits are unavailable\n";
            break;
        }
        int factor = 1;
        bool aligned = false;
        const float* cpuPtr = cpuLogits->readMap<float>();
        const auto cpuSize = static_cast<size_t>(cpuLogits->getInfo()->size);
        const int nextToken = chooseTop1(cpuPtr, cpuSize);
        std::cout << "decode step " << step
                  << " token=" << nextToken
                  << " piece=" << escapePiece(cpu->tokenizer_decode(nextToken)) << "\n";

        auto cpuDecode = cpu->forward(std::vector<int>{nextToken});
        auto qnnDecode = qnn->forward(std::vector<int>{nextToken});
        printLogitCompare("decode_" + std::to_string(step), cpuDecode, qnnDecode, cpu);
        cpuLogits = cpuDecode;
    }
}

static void runSweep(const std::string& cpuConfig,
                     const std::string& qnnConfig,
                     const std::vector<int>& baseIds,
                     const std::vector<int>& lengths) {
    if (lengths.empty()) {
        return;
    }
    std::cout << "\n=== sweep ===\n";
    for (int len : lengths) {
        if (len <= 0) {
            continue;
        }
        LlmPtr cpu(nullptr, Llm::destroy);
        LlmPtr qnn(nullptr, Llm::destroy);
        if (!loadPair(cpu, qnn, cpuConfig, qnnConfig)) {
            std::cout << "[sweep len=" << len << "] load failed\n";
            return;
        }
        auto ids = expandTokens(baseIds, len);
        if (ids.empty()) {
            std::cout << "[sweep len=" << len << "] no tokens\n";
            continue;
        }
        auto cpuMask = cpu->gen_attention_mask(len);
        auto qnnMask = qnn->gen_attention_mask(len);
        std::cout << "[sweep len=" << len << "] mask_ref=" << shapeString(cpuMask)
                  << " mask_qnn=" << shapeString(qnnMask) << "\n";

        auto cpuLogits = cpu->forward(ids);
        auto qnnLogits = qnn->forward(ids);
        if (cpuLogits == nullptr || qnnLogits == nullptr) {
            std::cout << "[sweep len=" << len << "] forward failed cpu=" << (cpuLogits.get() != nullptr)
                      << " qnn=" << (qnnLogits.get() != nullptr) << "\n";
            continue;
        }
        const float* cpuPtr = cpuLogits->readMap<float>();
        const float* qnnPtr = qnnLogits->readMap<float>();
        const size_t cpuSize = static_cast<size_t>(cpuLogits->getInfo()->size);
        const size_t qnnSize = static_cast<size_t>(qnnLogits->getInfo()->size);
        int factor = 1;
        bool aligned = false;
        auto alignedQnn = alignCandidate(cpuPtr, cpuSize, qnnPtr, qnnSize, factor, aligned);
        if (alignedQnn.second != cpuSize) {
            std::cout << "[sweep len=" << len << "] logits size mismatch cpu=" << cpuSize
                      << " qnn=" << qnnSize << "\n";
            continue;
        }
        auto result = compareFloatData(cpuPtr, alignedQnn.first, cpuSize);
        const int cpuTop1 = chooseTop1(cpuPtr, cpuSize);
        const int qnnTop1 = chooseTop1(alignedQnn.first, alignedQnn.second);
        std::cout << "[sweep len=" << len << "]"
                  << " factor=" << factor
                  << " top1(cpu=" << cpuTop1 << ",qnn=" << qnnTop1 << ")"
                  << " maxAbs=" << std::fixed << std::setprecision(6) << result.maxAbs
                  << " meanAbs=" << result.meanAbs
                  << " rmse=" << result.rmse
                  << " cosine=" << result.cosine
                  << " qnn_nan=" << result.nanCount
                  << " qnn_inf=" << result.infCount
                  << " qnn_zero=" << result.zeroCount << "\n";
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <cpu_config.json> <qnn_config.json> <prompt> [decode_steps] [sweep_lengths]\n";
        std::cerr << "Example: " << argv[0]
                  << " ./config.json ./config_qnn.json hi 2 1,18,64,128,129\n";
        return 1;
    }

    const std::string cpuConfig = argv[1];
    const std::string qnnConfig = argv[2];
    const std::string prompt = argv[3];
    const int decodeSteps = argc >= 5 ? std::atoi(argv[4]) : 1;
    const std::vector<int> sweepLengths = argc >= 6 ? parseLengths(argv[5]) : std::vector<int>();

    LlmPtr cpu(nullptr, Llm::destroy);
    LlmPtr qnn(nullptr, Llm::destroy);
    if (!loadPair(cpu, qnn, cpuConfig, qnnConfig)) {
        return 2;
    }

    auto cpuIds = cpu->tokenizer_encode(prompt);
    auto qnnIds = qnn->tokenizer_encode(prompt);
    std::cout << "cpu token count=" << cpuIds.size() << " ids=" << idsToString(cpuIds) << "\n";
    std::cout << "qnn token count=" << qnnIds.size() << " ids=" << idsToString(qnnIds) << "\n";
    if (cpuIds != qnnIds) {
        std::cout << "tokenizer mismatch detected; continuing with CPU token ids as shared input.\n";
    }
    if (cpuIds.empty()) {
        std::cerr << "Prompt encoded to zero tokens.\n";
        return 3;
    }

    runPromptCompare(cpu.get(), qnn.get(), cpuIds, decodeSteps);
    runSweep(cpuConfig, qnnConfig, cpuIds, sweepLengths);
    return 0;
}
