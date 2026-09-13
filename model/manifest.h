// Reader for deepmoe_manifest.json, the address book tools/manifest.py writes
// next to the ORIGINAL safetensors shards (design §5.1, v0.5).
//
// There is no repack. The 510 GB checkpoint stays as the 48 shards ModelScope
// shipped and this file is the only thing deepMoE adds. The manifest answers,
// for every tensor: which shard, at what absolute byte offset, how many bytes,
// what dtype and shape, and where its block scales live.
//
// The wrinkle the whole schema exists for: a shard's data section starts at
// `8 + header_len` and header_len differs per shard, so absolute tensor offsets
// are multiples of 8 but never of 4096. FILE_FLAG_NO_BUFFERING needs 4 KiB
// alignment on the offset, the length and the destination pointer. Rather than
// teach the I/O layer about bounce buffers, the manifest widens every read to
// sector boundaries and records the *skew*:
//
//     aligned_off   = floor(off / 4096) * 4096
//     aligned_bytes = ceil((off + len) / 4096) * 4096 - aligned_off
//     skew          = off - aligned_off
//
// Byte-contiguous tensors are merged into one *run*, so a routed expert -- whose
// three .weight tensors are adjacent and whose three .scale tensors are adjacent
// -- is exactly two runs: 17,698,816 B of weights and 1,110,016 B of scales.
// The ExpertStore fills one slot with all runs of an expert, laid back to back,
// and hands the kernels `slot_base + run.slot_offset + part.skew` per part.
//
// Schema (version 2):
//   {
//     "version": 2, "model": "DeepSeek-V4.1-Flash", "alignment": 4096,
//     "expert_slot_bytes": 18808832,
//     "expert_parts": ["w1.weight","w1.scale","w2.weight","w2.scale","w3.weight","w3.scale"],
//     "files": [ {"path":"model-00001-of-00048.safetensors",
//                 "bytes":970533624, "data_start":4184, "sha256":""}, ... ],
//     "tensors": {
//       "layers.5.attn.wq_a.weight": {
//         "file":7, "offset":510842328, "bytes":6553600, "dtype":"fp8_e4m3",
//         "shape":[1280,5120],
//         "scale": {"file":7, "offset":7968216, "bytes":6400, "dtype":"e8m0",
//                   "shape":[40,160], "block":[32,32]} }, ... },
//     "experts": [ {"layer":0, "name":"layers.0", "experts":[
//         [ {"file":2,"aligned_off":8269824,"aligned_bytes":1110016,"slot_offset":0,
//            "parts":[{"tensor":"w1.scale","skew":3896,"bytes":368640,"slot_offset":3896}, ...]},
//           {...weights run...} ], ... ]}, ... ],
//     "engram": [ {"layer":1, "rows":384006168,
//                  "value":{"file":46,"offset":664,"row_bytes":256,"dtype":"fp8_e4m3"},
//                  "scale":{"file":46,"offset":98305579672,"row_bytes":8,"dtype":"e8m0"}}, ... ]
//   }
//
// Logical layer numbering follows core/types.h ExpertKey: 0..39 are the main
// MoE layers (384 experts each), 40..42 the three DSpark blocks (128 each).
//
// Ownership/threading: Manifest owns its tables; built once at load, read-only
// afterwards, so it is shared freely between the planner, I/O and GPU threads.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/align.h"
#include "core/status.h"
#include "core/types.h"
#include "model/layout.h"

namespace deepmoe {

// One of the six tensors of a routed expert. The order is the manifest's
// "expert_parts" array and the order of ExpertStore's pointer-table stride.
enum class ExpertPart : uint8_t {
    W1Weight = 0, W1Scale = 1,
    W2Weight = 2, W2Scale = 3,
    W3Weight = 4, W3Scale = 5,
};
inline constexpr uint32_t kExpertPartCount = layout::kExpertParts;

const char*                expert_part_name(ExpertPart p);
std::optional<ExpertPart>  expert_part_from_string(std::string_view s);

// An original safetensors shard. `data_start` is 8 + header_len, the byte at
// which tensor offset 0 lives; every offset in this manifest is already
// absolute, so data_start is carried for diagnostics and re-derivation only.
struct FileEntry {
    std::string path;         // relative to the model directory
    uint64_t    bytes      = 0;
    uint64_t    data_start = 0;
    std::string sha256;       // empty unless manifest.py ran with --verify
};

// Where a tensor's block scales live. Absent (bytes == 0) for bf16/fp32.
struct ScaleEntry {
    uint32_t              file   = 0;
    uint64_t              offset = 0;
    uint64_t              bytes  = 0;
    QuantType             dtype  = QuantType::Unknown;
    std::vector<uint64_t> shape;
    uint32_t              block_m = 0;   // rows per scale (32 for fp8 tiles, 1 for fp4 rows)
    uint32_t              block_k = 0;   // K elements per scale (32 everywhere)
    bool present() const { return bytes != 0; }
};

struct TensorEntry {
    std::string           name;
    uint32_t              file   = 0;    // index into Manifest::files()
    uint64_t              offset = 0;    // absolute in the shard
    uint64_t              bytes  = 0;
    QuantType             dtype  = QuantType::Unknown;
    std::vector<uint64_t> shape;         // logical: fp4 records [out, K], not [out, K/2]
    ScaleEntry            scale;

    uint64_t elements() const;
};

// One part inside a run. `skew` is relative to the run's aligned start;
// `slot_offset` is where the part's first byte lands in the slot.
struct RunPart {
    ExpertPart part        = ExpertPart::W1Weight;
    uint32_t   skew        = 0;
    uint64_t   bytes       = 0;
    uint64_t   slot_offset = 0;
};

// One sector-aligned read. This is exactly what becomes a storage::IoRequest.
struct Run {
    uint32_t             file          = 0;
    uint64_t             aligned_off   = 0;
    uint64_t             aligned_bytes = 0;
    uint64_t             slot_offset   = 0;   // where the run lands in the slot
    std::vector<RunPart> parts;
};

// Everything the store needs to fill one expert's slot.
struct ExpertEntry {
    std::vector<Run> runs;
    uint64_t         slot_bytes = 0;                       // sum of run aligned_bytes
    uint64_t         part_offset[kExpertPartCount] = {};   // slot-relative, per ExpertPart
    uint64_t         part_bytes [kExpertPartCount] = {};

    uint64_t offset_of(ExpertPart p) const { return part_offset[static_cast<uint8_t>(p)]; }
    uint64_t bytes_of (ExpertPart p) const { return part_bytes [static_cast<uint8_t>(p)]; }
};

// One plane of an engram table: 384M rows of `row_bytes` each.
struct EngramPlane {
    uint32_t  file      = 0;
    uint64_t  offset    = 0;    // absolute offset of row 0
    uint64_t  bytes     = 0;
    uint32_t  row_bytes = 0;
    QuantType dtype     = QuantType::Unknown;
};

struct EngramEntry {
    uint32_t    layer = 0;
    uint64_t    rows  = 0;
    EngramPlane value;    // [rows, 256] fp8_e4m3
    EngramPlane scale;    // [rows, 8]   e8m0
};

// One sector-aligned read with the skew that locates the payload inside it.
// The engram row planner and any single-tensor loader speak in these.
struct AlignedRead {
    uint32_t file          = 0;
    uint64_t aligned_off   = 0;
    uint64_t aligned_bytes = 0;
    uint32_t skew          = 0;
    uint64_t bytes         = 0;   // the payload length, not the aligned length
};

// The two reads that fetch one engram row (value plane + scale plane).
struct EngramRowPlan {
    AlignedRead value;
    AlignedRead scale;
};

class Manifest {
public:
    static Result<Manifest> parse(std::string_view json_text);
    static Result<Manifest> load(const std::string& manifest_path);

    uint32_t           version()   const { return version_; }
    const std::string& model()     const { return model_; }
    uint32_t           alignment() const { return alignment_; }
    // The slot size the slab pool must use; layout::kExpertSlotBytes is the
    // compile-time copy and validate() insists the two agree.
    uint64_t           expert_slot_bytes() const { return expert_slot_bytes_; }

    const std::vector<FileEntry>& files() const { return files_; }
    const FileEntry*   file(uint32_t index) const;
    // Index of the shard whose path matches, or nullopt. Used by tools, not the
    // hot path -- every runtime lookup already carries the integer index.
    std::optional<uint32_t> file_index(std::string_view path) const;

    const TensorEntry* tensor(std::string_view name) const;
    Result<const TensorEntry*> require_tensor(std::string_view name) const;
    const std::unordered_map<std::string, TensorEntry>& tensors() const { return tensors_; }

    // The aligned read that covers a whole non-expert tensor (or its scales).
    Result<AlignedRead> tensor_read(std::string_view name) const;
    Result<AlignedRead> tensor_scale_read(std::string_view name) const;

    // --- experts ----------------------------------------------------------
    uint32_t           expert_layers() const { return static_cast<uint32_t>(experts_.size()); }
    uint32_t           experts_in_layer(uint32_t layer) const;
    const ExpertEntry* expert(ExpertKey key) const;
    Result<const ExpertEntry*> require_expert(ExpertKey key) const;

    // --- engram -----------------------------------------------------------
    const std::vector<EngramEntry>& engram() const { return engram_; }
    const EngramEntry* engram_for_layer(uint32_t layer) const;
    // The two aligned reads for one hashed row (design §7.10, §9.6 P2).
    Result<EngramRowPlan> engram_row(uint32_t layer, uint64_t row) const;

    // Checks internal consistency: every file index is in range, no entry runs
    // past its shard, every run is sector-aligned, every expert's parts add up
    // to layout::kExpertBytes and fit in layout::kExpertSlotBytes.
    Result<void> validate() const;

    // Sum of every tensor's bytes plus its scales, and of every expert's parts.
    uint64_t total_bytes() const;
    // Total of the `files` array: what the checkpoint occupies on disk.
    uint64_t file_bytes() const;

private:
    uint32_t    version_ = 0;
    uint32_t    alignment_ = static_cast<uint32_t>(kPageSize);
    uint64_t    expert_slot_bytes_ = 0;
    std::string model_;
    std::vector<FileEntry>  files_;
    std::unordered_map<std::string, TensorEntry> tensors_;
    // experts_[logical layer][expert]; a missing layer is an empty row.
    std::vector<std::vector<ExpertEntry>> experts_;
    std::vector<EngramEntry> engram_;
};

// dtype strings used in the manifest, matching core/types.h QuantType.
QuantType   quant_from_string(std::string_view s);
const char* quant_to_string(QuantType q);

// The aligned read covering [off, off + bytes) in `file`.
AlignedRead align_read(uint32_t file, uint64_t off, uint64_t bytes);

}  // namespace deepmoe
