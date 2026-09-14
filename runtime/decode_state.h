// The state a decode step starts from, and the reference it is checked against
// (design §12 L3, exported by `tools/oracle.py --level l3`).
//
// Why this is in runtime/ and not in tests/
// ----------------------------------------
// Two of the inputs a decode step needs are not produced by any kernel that
// exists: the compressed KV and the indexer's top-k list (design §7.4). Until
// the compressor and the indexer are written, ANY caller that wants to run the
// forty layers -- the test, `deepmoe run`, a benchmark -- has to load them, so
// the loader belongs next to the engine that consumes them rather than inside a
// test binary. Everything here is labelled as LOADED in the Engine's status
// line and in docs/p2_decode.md, and it all goes away when §7.4 lands.
//
// What is loaded, and what is not
// -------------------------------
//   window KV ring, per layer, after prefill   LOADED once. Our own wkv kernel
//                                              writes every slot from then on.
//   compressed KV + top-k list, per step       LOADED per step.
//   engram hash constants                      LOADED, but they are a pure
//                                              function of tokenizer.json (see
//                                              runtime/engram.h) -- the row ids
//                                              are computed here.
//   everything else                            computed.
//
// The file format is the L2 container of docs/p2_attention.md §1 with one
// record per STEP instead of per layer: a JSON index naming every tensor's
// dtype, shape and byte offset, and a flat binary after a 12-byte header.
//
// Ownership/threading: a plain value type, loaded once before the token loop.
#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "core/json.h"
#include "core/status.h"
#include "core/types.h"
#include "runtime/kvstore.h"

namespace deepmoe::runtime {

// One tensor out of the export, always widened to fp32 (`bf16` was the dtype
// the reference held; widening is lossless).
struct StateTensor {
    std::string           dtype;
    std::vector<uint64_t> shape;
    std::vector<float>    f;
    std::vector<int32_t>  i;     // populated for i32 tensors, whose values are ids

    uint64_t elements() const {
        uint64_t n = shape.empty() ? 0 : 1;
        for (uint64_t d : shape) n *= d;
        return n;
    }
};

// The reference's logits at one step, top-`k` plus the three whole-vector
// statistics that catch a runtime whose top-k agrees but whose tail does not.
struct RefLogits {
    std::vector<uint32_t> top_ids;
    std::vector<float>    top_logits;
    float    max_logit = 0.0f, logsumexp = 0.0f, min_logit = 0.0f;
    uint32_t argmax    = 0;
    uint32_t in_token  = 0;      // 0 on the prefill record, which has no input token
    bool     has_in_token = false;

    // top1 - top2 of the reference's own distribution: the number design §12
    // L3 asks to be recorded beside any disagreement.
    float margin() const {
        return top_logits.size() > 1 ? top_logits[0] - top_logits[1] : 0.0f;
    }
};

class DecodeState {
public:
    static Result<DecodeState> load(const std::string& dir);

    const std::string& dir() const { return dir_; }
    uint32_t prefill_len() const { return prefill_len_; }
    uint32_t decode_pos()  const { return decode_pos_; }
    uint32_t steps()       const { return steps_; }
    uint32_t layers()      const { return layers_; }
    uint32_t window()      const { return window_; }
    uint32_t head_dim()    const { return head_dim_; }

    const std::vector<uint32_t>& prompt_ids() const { return prompt_ids_; }
    // The reference's greedy trajectory: [0] is the argmax of the prefill
    // logits and the input to step 0, [s+1] the argmax of step s.
    const std::vector<uint32_t>& greedy_tokens() const { return greedy_; }

    // `step` is 0 for the prefill record and 1..steps() for decode step 0..n-1.
    // That numbering matches `record(step)` below and nothing else uses it.
    const RefLogits& logits(uint32_t step) const { return logits_[step]; }
    const StateTensor* tensor(uint32_t step, const std::string& name) const;

    // Seeds every layer's window ring with what the prompt left there, plus --
    // on the four kv_source_layers, and only if the export carries them -- the
    // compressed-KV cache, the indexer's key cache and the compressor's
    // carried group state. Those three are what a runtime with its own §7.4
    // kernels needs to stop loading anything per step.
    Result<void> seed_prefill(KvStore& kv) const;
    // Whether this export carries them. An older one does not, and a caller
    // that has the §7.4 kernels has to fall back to seeding per step and say so.
    bool has_prefill_ced() const;
    // Seeds the compressed KV and the top-k list every layer read at decode
    // step `s` (0-based).
    Result<void> seed_step(KvStore& kv, uint32_t s) const;

    // The widest compressed run and the longest top-k list in the export: what
    // KvStoreConfig has to be sized for.
    uint32_t max_compressed() const { return max_cmp_; }
    uint32_t max_topk()       const { return max_topk_; }

private:
    struct Record {
        std::string name;
        std::map<std::string, StateTensor, std::less<>> t;
    };
    Result<void> read_record(const std::string& path, const JsonValue& rec,
                             Record& out);

    std::string dir_;
    uint32_t prefill_len_ = 0, decode_pos_ = 0, steps_ = 0;
    uint32_t layers_ = 0, window_ = 128, head_dim_ = 512;
    uint32_t max_cmp_ = 0, max_topk_ = 0;
    std::vector<uint32_t> prompt_ids_, greedy_;
    std::vector<Record>    records_;    // [0] prefill, [1 + s] step s
    std::vector<RefLogits> logits_;
};

}  // namespace deepmoe::runtime
