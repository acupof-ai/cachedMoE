// Reader for manifest.json, the address book tools/repack.py writes next to
// hot.bin / experts.bin / engram.L*.bin / mtp.bin (design §5.1).
//
// Every tensor entry answers: which file, at what offset, how many bytes, what
// shape, what dtype, and where its block scales live. The runtime never opens a
// safetensors shard -- the manifest plus the four repacked blobs is the whole
// on-disk contract.
//
// Expected schema (version 1):
//   {
//     "version": 1,
//     "model": "DeepSeek-V4.1-Flash",
//     "files": { "hot": {"path":"hot.bin","bytes":...,"sha256":"..."}, ... },
//     "tensors": {
//       "layers.5.attn.wq_a.weight": {
//         "file":"hot", "offset":..., "bytes":..., "dtype":"fp8_e4m3",
//         "shape":[1280,5120],
//         "scale": {"offset":..., "bytes":..., "dtype":"e8m0",
//                   "shape":[40,160], "block":[32,32]}
//       }, ...
//     },
//     "experts": { "file":"experts", "stride":18800640, "layers":40, "per_layer":384 },
//     "engram":  [ {"layer":1,"file":"engramL1","rows":384006168,"row_bytes":264}, ... ]
//   }
//
// Ownership/threading: Manifest owns its tables; built once at load, read-only
// afterwards, so it is shared freely between the planner, I/O and GPU threads.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/status.h"
#include "core/types.h"

namespace deepmoe {

// Logical blob ids; the manifest maps each to a relative path.
struct FileEntry {
    std::string id;      // "hot", "experts", "mtp", "engramL1", "engramL14"
    std::string path;    // relative to the model directory
    uint64_t    bytes = 0;
    std::string sha256;  // optional; empty when repack.py ran without --verify
};

// Where a tensor's block scales live. Absent (bytes == 0) for bf16/fp32 tensors.
struct ScaleEntry {
    uint64_t             offset = 0;
    uint64_t             bytes  = 0;
    QuantType            dtype  = QuantType::Unknown;
    std::vector<uint64_t> shape;
    uint32_t             block_m = 0;   // rows per scale (32 for fp8 tiles, 1 for fp4 rows)
    uint32_t             block_k = 0;   // K elements per scale (32 everywhere)
    bool present() const { return bytes != 0; }
};

struct TensorEntry {
    std::string          name;
    std::string          file;     // FileEntry::id
    uint64_t             offset = 0;
    uint64_t             bytes  = 0;
    QuantType            dtype  = QuantType::Unknown;
    std::vector<uint64_t> shape;
    ScaleEntry           scale;

    uint64_t elements() const;
};

// design §5.1: routed experts are addressed arithmetically, not by name.
struct ExpertsEntry {
    std::string file = "experts";
    uint64_t    stride    = 0;   // 18,800,640
    uint32_t    layers    = 0;   // 40
    uint32_t    per_layer = 0;   // 384
    uint64_t    base      = 0;   // byte offset of (layer 0, expert 0) inside the file
};

struct EngramEntry {
    uint32_t    layer = 0;
    std::string file;
    uint64_t    rows      = 0;
    uint32_t    row_bytes = 0;   // 264
    uint64_t    base      = 0;
};

class Manifest {
public:
    static Result<Manifest> parse(std::string_view json_text);
    static Result<Manifest> load(const std::string& manifest_path);

    uint32_t           version() const { return version_; }
    const std::string& model()   const { return model_; }

    const FileEntry*   file(std::string_view id) const;
    const TensorEntry* tensor(std::string_view name) const;
    Result<const TensorEntry*> require_tensor(std::string_view name) const;

    const ExpertsEntry& experts() const { return experts_; }
    const std::vector<EngramEntry>& engram() const { return engram_; }
    const EngramEntry* engram_for_layer(uint32_t layer) const;

    const std::unordered_map<std::string, TensorEntry>& tensors() const { return tensors_; }
    const std::vector<FileEntry>& files() const { return files_; }

    // Byte offset of a routed expert in its blob, from the arithmetic layout.
    Result<uint64_t> expert_offset(ExpertKey key) const;

    // Checks internal consistency: every tensor's file exists, no entry runs
    // past its file's byte count, the expert stride matches layout.h.
    Result<void> validate() const;

    // Sum of all tensor bytes plus the expert and engram regions.
    uint64_t total_bytes() const;

private:
    uint32_t    version_ = 0;
    std::string model_;
    std::vector<FileEntry> files_;
    std::unordered_map<std::string, TensorEntry> tensors_;
    ExpertsEntry experts_;
    std::vector<EngramEntry> engram_;
};

// dtype strings used in manifest.json, matching core/types.h QuantType.
QuantType   quant_from_string(std::string_view s);
const char* quant_to_string(QuantType q);

}  // namespace deepmoe
