# CUDA masked Attention evidence

Baseline: unmodified alibaba/MNN commit b0239771b07b317a258fbc384683bbd540741406. Candidate: exactly that revision plus AttentionExecution.cu/.hpp changes from commit d561e036. Neither includes the separate RoPE layout fix or llm_bench backend-label fix. Upstream JSON incorrectly labels CUDA as CPU; all benchmark invocations explicitly use `-a cuda`.

RTX 3090 / sm86 (82 SMs), NVIDIA 595.99.02, CUDA 13.2. Same exported Qwen3.5-4B and MiniCPM5-2B Q4/HQQ weights, block size 64. Precision 2, threads 8, KV quantization 0, flashAttention 1, mmap 0, dynamic quantization 0. No GPU clock lock. Local CUDA 13.2 build uses CUDA_NVCC_FLAGS=--static-global-template-stub=false.

## Single-process lengths

Input lengths 512/2048/4096 tokens; 32 generated tokens, one discarded warmup and three measured repetitions per configuration. Sequential variant runs. Bars show mean decode throughput in tokens/s, with ±1 standard deviation. Throughput ratios are not whole-request speedups. Prefill results, including negative changes, are retained in raw JSON; no prefill optimization is claimed.

## Independent-process concurrency

Each process runs batch 1 with separate model weights, KV cache and working/cache directory. This is not model batching or continuous batching. Group configurations: one process at 512/2048/4096 input tokens; two at 512/2048; two with 128+4096; four at 512; four with 128+512+1024+2048. Each worker warms up once then runs three repetitions with 32 generated tokens. Each group is launched once per variant and model.

The graph reports complete group duration in seconds, from process launch to completion, including model loading, warmup, prefill, decode, sampling and polling overhead. It is not steady-state throughput, TTFT, P95 latency or a synchronized decode-only window. Different-length workers need not decode simultaneously. Worker throughputs must not simply be summed and labeled service throughput. GPU memory is sampled whole-device use, not precise per-process peak allocation.

## Backend integration regression

Original upstream and Attention-only each passed 240 output checks through the real CUDA backend creator and persistent execution/cache. FP32/FP16, fixed batch 1/2, dimensions 64/80/128/256, KV lengths 127/128/129/256/257. Cases cover short right-aligned mask, first-token-only visibility and all-masked zero-output convention. Q/K are zero and V is nonconstant; this checks indexing, cache growth and masks, not general numerical quality. Independent batch masks and changes to batch within an execution are not covered.

## Known limitations

Broader independent-corpus quality regression of the final full patch is outstanding. Earlier policy experiments found a MiniCPM greedy top-1 divergence at a FP16 tie; generated text is not guaranteed to match bit for bit. Other GPUs, full unquantized models and arbitrary batched masks remain unvalidated. Earlier RoPE-PR and intermediate-policy benchmark results are excluded from these comparisons.

Reproduction scripts, raw benchmark JSON/logs, graph source, integration helper, patch and pre-commit output accompany these plots. No model weights are included. Supporting artifacts are kept separate from the production patch.
