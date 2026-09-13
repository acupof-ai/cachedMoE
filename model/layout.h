// Compile-time constants for the DeepSeek-V4.1-Flash weight layout, taken from
// the safetensors header statistics in design appendix A and the block layout
// of design §5.1. These are the numbers kernels and the slab allocator are
// built around; v41_config.cpp asserts the parsed config.json agrees with them.
//
// Ownership/threading: constants only.
#pragma once

#include <cstdint>

namespace deepmoe::layout {

// --- topology (design §2.1) --------------------------------------------------
inline constexpr uint32_t kHiddenSize        = 5120;
inline constexpr uint32_t kHcMult            = 4;      // residual stream is [4][5120] fp32
inline constexpr uint32_t kHcFnRows          = 24;     // hc_fn is [24, 4*5120]
inline constexpr uint32_t kHcFnCols          = kHcMult * kHiddenSize;   // 20480
inline constexpr uint32_t kSinkhornIters     = 20;
inline constexpr uint32_t kNumLayers         = 40;     // 0..19 encoder, 20..39 decoder (CED)
inline constexpr uint32_t kEncoderLayers     = 20;
inline constexpr uint32_t kMtpBlocks         = 3;      // DSpark draft blocks -> logical layers 40..42
inline constexpr uint32_t kMtpLayerBase      = kNumLayers;
inline constexpr uint32_t kTotalLogicalLayers = kNumLayers + kMtpBlocks;

inline constexpr uint32_t kRoutedExperts     = 384;
inline constexpr uint32_t kSharedExperts     = 1;
inline constexpr uint32_t kExpertsPerTok     = 6;
inline constexpr uint32_t kMoeIntermediate   = 2304;   // 5120 -> 2304 -> 5120 SwiGLU
inline constexpr float    kSwigluLimit       = 10.0f;  // clamp for gate (upper) and up (both)
inline constexpr float    kRoutedScaling     = 1.5f;
inline constexpr float    kRmsNormEps        = 1e-20f;

inline constexpr uint32_t kDsparkExperts     = 128;
inline constexpr uint32_t kDsparkTopK        = 3;
inline constexpr uint32_t kDsparkBlockSize   = 5;      // draft positions produced per cycle
inline constexpr uint32_t kDsparkMarkovRank  = 256;

// attention (MQA over a 512-wide latent, low-rank Q, grouped low-rank O)
inline constexpr uint32_t kAttnHeads         = 64;
inline constexpr uint32_t kKvHeads           = 1;
inline constexpr uint32_t kHeadDim           = 512;
inline constexpr uint32_t kRopeHeadDim       = 64;     // only the last 64 dims are rotated
inline constexpr uint32_t kQLoraRank         = 1280;
inline constexpr uint32_t kOLoraRank         = 1024;
inline constexpr uint32_t kOGroups           = 8;
inline constexpr uint32_t kSlidingWindow     = 128;
inline constexpr uint32_t kIndexHeads        = 32;
inline constexpr uint32_t kIndexHeadDim      = 128;
inline constexpr uint32_t kIndexTopK         = 512;
inline constexpr uint32_t kCandidateBlocks   = 2048;
inline constexpr uint32_t kCandidateBlockSize = 8;

// engram (design §2.1, §7.10)
inline constexpr uint32_t kEngramLayers      = 2;      // layers 1 and 14
inline constexpr uint32_t kEngramHeads       = 8;
inline constexpr uint32_t kEngramHeadDim     = 256;
inline constexpr uint32_t kEngramMaxNgram    = 4;      // 3 orders x 8 heads = 24 rows per token
inline constexpr uint32_t kEngramRowsPerToken = 24;
inline constexpr uint32_t kEngramWkvRows     = 25600;
inline constexpr uint32_t kEngramWkvCols     = 6144;   // 24 rows x 256

inline constexpr uint32_t kVocabSize         = 129280;

// --- quantisation blocking (design §6, appendix A) --------------------------
inline constexpr uint32_t kFp4ScaleBlock     = 32;     // one UE8M0 exponent per 32 elements along K
inline constexpr uint32_t kFp8ScaleBlockK    = 32;     // fp8 scales are per 32x32 tile
inline constexpr uint32_t kFp8ScaleBlockM    = 32;

// --- byte sizes (design §2.3, appendix A) -----------------------------------
// One routed expert: w1/w3/w2 each [.., K/2] packed FP4 plus its E8M0 scales.
inline constexpr uint64_t kExpertWeightBytesPerMat = 5'898'240;   // e.g. [2304, 2560] packed fp4
inline constexpr uint64_t kExpertScaleBytesPerMat  =   368'640;   // e.g. [2304, 160] E8M0
inline constexpr uint64_t kExpertMats              = 3;           // w1, w3, w2
inline constexpr uint64_t kExpertBytes =
    kExpertMats * (kExpertWeightBytesPerMat + kExpertScaleBytesPerMat);   // 18,800,640
static_assert(kExpertBytes == 18'800'640, "design §2.3: one routed expert is 18,800,640 B");
static_assert(kExpertBytes % 4096 == 0, "design §5.1: an expert block is a whole number of 4 KiB sectors");
inline constexpr uint64_t kExpertSectors = kExpertBytes / 4096;   // 4590

// Engram row: 256 B of FP8 values interleaved with 8 B of E8M0 scale, so one
// 4 KiB read picks up whole rows with their scales (design §5.1).
inline constexpr uint64_t kEngramRowBytes = 256 + 8;              // 264
static_assert(kEngramRowBytes == 264, "design §5.1: engram rows are 264 B interleaved");

// Whole-layer stride in experts.bin: 384 x 18,800,640 = 7.22 GB, read
// sequentially by the expert-major prefill path (design §9.7).
inline constexpr uint64_t kExpertsPerLayerBytes = uint64_t(kRoutedExperts) * kExpertBytes;

// Totals used for budgeting and for the bytes-per-token model of design §2.3.
inline constexpr uint64_t kRoutedExpertCount = uint64_t(kNumLayers) * kRoutedExperts;  // 15,360
inline constexpr uint64_t kRoutedExpertTotalBytes = kRoutedExpertCount * kExpertBytes;

// Per-token resident (hot) traffic, design §2.3. Used as the denominator of the
// LPDDR utilisation figure in §1.3 and as a sanity bound in the profiler.
inline constexpr uint64_t kHotBytesPerToken = 8'500'000'000ull;

// --- file names produced by tools/repack.py (design §5.1) -------------------
inline constexpr const char* kHotFile      = "hot.bin";
inline constexpr const char* kExpertsFile  = "experts.bin";
inline constexpr const char* kMtpFile      = "mtp.bin";
inline constexpr const char* kManifestFile = "manifest.json";
inline constexpr const char* kEngramL1File  = "engram.L1.bin";
inline constexpr const char* kEngramL14File = "engram.L14.bin";

// Byte offset of routed expert (layer, expert) inside experts.bin.
constexpr uint64_t expert_file_offset(uint32_t layer, uint32_t expert) noexcept {
    return uint64_t(layer) * kExpertsPerLayerBytes + uint64_t(expert) * kExpertBytes;
}

}  // namespace deepmoe::layout
