// The decode-time KV storage of design §11.3, allocated in GPU-addressable
// memory and shaped exactly as gpu/shaders/wkv.slang writes it and
// sparse_attn.slang / indexer.slang read it. docs/p4_kv_ux.md has the byte
// accounting this layout follows.
//
// What lives here
// ---------------
//   window KV        A 128-slot ring per layer of E4M3 bytes plus one UE8M0 scale
//                    per 32 dims, written by the wkv kernel every decode step
//                    exactly as `Attention._window_kv` writes it. 2.7 MB for 40
//                    layers whatever the context. NEVER parked or snapshotted:
//                    whatever restores a context rebuilds it by replaying the
//                    last <= 128 tokens (design §11.2; runtime/session.h).
//   compressed KV    One plane per kv_source_layer (2, 8, 14 at ratio 2; 20 at
//                    ratio 1), `ceil(positions / ratio)` rows of 512 bf16. The
//                    other 34 compressed layers READ their source's plane
//                    (model.py's `shared_attn`), so `layer(l)` hands them its
//                    addresses. Track Q's store gave all 40 layers a plane.
//   index keys       One plane per kv source too (model.py `Indexer.owns_k` is
//                    `layer_id in kv_source_layers`; layers 24..36 run their own
//                    indexer against layer 20's keys), 128 bf16 a row.
//   compressor state The [ratio][latent_dim] fp32 tail of an incomplete group
//                    (`kv_state`, `score_state`), per source.
//   top-k list       Per layer, window + min(index_topk, context) int32.
//
// Why bf16 and not the packed formats. The compressor writes FP4 E2M1 block-16
// + E4M3 and the indexer FP4 block-32 + E8M0 (design §11.3), but
// `sparse_attn.slang` reads the compressed half and `indexer.slang` scores the
// keys as bf16 rows of the dequantised values, and the top-k list that picks
// the rows is computed on the GPU inside the same command buffer -- so there is
// no host point at which a bounded working set could be gathered. The live
// store therefore stays bf16 (3.6x the packed size: 2,560 + 640 B per token
// against 720 + 170), sized per source and by the context actually reached;
// every copy that is NOT being attended -- a parked session, a rollback's saved
// rows -- is packed (`pack` / `unpack`), bit-exactly: the bf16 values are on
// the FP4 grid already, so the pair (nibble, scale) is recovered, not re-quantised.
//
// Growth. `create` allocates `initial_context` positions and the store doubles
// (copying what is live) whenever a step or a seed needs more, up to
// `max_context`. Addresses change on growth, which is safe because every
// caller takes `layer(l)` afresh per step.
//
// Slabs. The store is not one allocation: its regions are laid out in a flat
// space cut into slabs of at most `KvStoreConfig::slab_bytes` (2 GiB, the
// `maxMemoryAllocationSize` of design §1.1), each region wholly inside one
// slab so a plane's address plus a row offset is still one address. That is
// what lifts design §11.3's "KvStore is ONE allocation, so max_context <~ 41.7K":
// with per-source planes 64K needs 213 MB in one slab anyway, and the cut only
// starts to matter past ~600K positions.
//
// Ownership/threading: KvStore owns its allocation through the MemoryAllocator
// it was created with. Written by the GPU, seeded and grown by the host
// between steps, on the engine thread.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/status.h"
#include "core/types.h"
#include "gpu/vulkan/memory.h"
#include "model/v41_config.h"

namespace deepmoe::runtime {

struct KvStoreConfig {
    static constexpr uint32_t kNoPlane = 0xFFFFFFFFu;

    uint32_t layers      = 40;
    uint32_t window      = 128;
    uint32_t latent_dim  = 512;    // head_dim; MQA over one KV head
    uint32_t index_dim   = 128;
    uint32_t max_context = 1024;   // positions the store may grow to
    uint32_t scale_block = 32;
    // Slots in a compressor's carried group state. compressor.slang's
    // `kMaxRatio`; the checkpoint's largest compress_ratio is 2.
    uint32_t max_ratio   = 8;
    uint32_t index_topk  = 512;
    // Positions `create` allocates; 0 = max_context. The store grows past it.
    uint32_t initial_context = 0;
    // The largest single allocation the store will ask for. The store is cut
    // into as many slabs of at most this as its regions need (design §5.3: one
    // allocation may not exceed `maxMemoryAllocationSize`, 2 GiB on this APU --
    // §11.3's "KvStore is ONE allocation, so max_context <~ 41.7K"). Every
    // region lies wholly inside one slab, so an address plus a row offset is
    // still one address. `create` halves it and retries when the driver refuses
    // the size, so a device with a smaller cap needs no configuration.
    uint64_t slab_bytes = 1ull << 31;
    // model.py's `shared_attn`: whose compressed / index-key / state planes
    // layer l uses (kNoPlane for a window-only layer) and the ratio its rows
    // are counted at. Both empty = every layer owns planes at ratio 1, which is
    // what a rig that seeds one arbitrary layer at a time wants.
    std::vector<uint32_t> plane_of;
    std::vector<uint32_t> ratio;

    // The plan config.json implies: planes on kv_source_layer_ids only.
    static KvStoreConfig for_model(const TextConfig& c, uint32_t max_context,
                                   uint32_t initial_context = 0);

    bool     per_source() const { return !plane_of.empty(); }
    uint32_t owner(uint32_t l) const;          // kNoPlane for a window-only layer
    uint32_t plane_ratio(uint32_t owner) const;   // >= 1
    std::vector<uint32_t> owners() const;
    uint32_t rows_for(uint32_t owner, uint32_t positions) const {
        const uint32_t r = plane_ratio(owner);
        return (positions + r - 1) / r;
    }
    uint32_t topk_rows() const;
    uint32_t window_scales() const { return latent_dim / scale_block; }

    // What the store allocates for `positions` (bf16 planes, as the kernels
    // read them), and what the same state costs in the model's packed formats.
    uint64_t window_bytes() const {
        return uint64_t(layers) * window * (latent_dim + window_scales());
    }
    uint64_t compressed_bytes(uint32_t positions) const;
    uint64_t index_key_bytes(uint32_t positions) const;
    uint64_t cmp_state_bytes() const;
    uint64_t topk_bytes() const { return uint64_t(layers) * topk_rows() * sizeof(uint32_t); }
    uint64_t total_bytes(uint32_t positions) const {
        return window_bytes() + compressed_bytes(positions) + index_key_bytes(positions) +
               cmp_state_bytes() + topk_bytes();
    }
    // The bf16 working set a live store of `positions` holds, and how many
    // slabs `slab_bytes` cuts it into (the real figure, padding included, is
    // KvStore::bytes() / KvStore::slabs()).
    uint32_t min_slabs(uint32_t positions) const {
        if (!slab_bytes) return 1;
        return static_cast<uint32_t>((total_bytes(positions) + slab_bytes - 1) / slab_bytes);
    }
    // FP4 E2M1 block-16 + E4M3 compressed rows, FP4 block-32 + E8M0 index keys,
    // the carried state of the ratio > 1 sources: the non-SWA state a parked
    // context keeps (before its token ids).
    uint64_t packed_bytes(uint32_t positions) const;
};

// Where one layer's KV lives, as the kernels address it.
struct KvLayerView {
    DeviceAddress win_val   = kNoDeviceAddress;   // [window][latent_dim] E4M3
    DeviceAddress win_scale = kNoDeviceAddress;   // [window][latent_dim/32] UE8M0
    DeviceAddress cmp_kv    = kNoDeviceAddress;   // [n_cmp][latent_dim] bf16
    DeviceAddress top_idx   = kNoDeviceAddress;   // [n_kv] int32, into the concatenation
    DeviceAddress idx_key   = kNoDeviceAddress;   // [n_cmp][index_dim] bf16
    DeviceAddress cmp_state_kv    = kNoDeviceAddress;  // [max_ratio][latent_dim] f32
    DeviceAddress cmp_state_score = kNoDeviceAddress;
    uint8_t*      win_val_host   = nullptr;
    uint8_t*      win_scale_host = nullptr;
    uint16_t*     cmp_kv_host    = nullptr;
    uint32_t*     top_idx_host   = nullptr;
    uint16_t*     idx_key_host   = nullptr;
    float*        cmp_state_kv_host    = nullptr;
    float*        cmp_state_score_host = nullptr;
    uint32_t      n_cmp = 0;
    uint32_t      n_kv  = 0;
    uint32_t      plane_owner = KvStoreConfig::kNoPlane;   // whose planes these are
};

// --- the packed forms -----------------------------------------------------------

// `n` bf16 values that some FP4 quantiser put on its grid -> E2M1 nibbles (low
// nibble = even element) and one scale per `block`: an E4M3 byte (`e8m0` false,
// the compressor's block-16) or a UE8M0 byte (the indexer's block-32). Returns
// false, leaving the outputs unspecified, when a block is not exactly
// representable -- the caller keeps that row raw. Exact means the bf16 bits
// come back identical from `unpack_fp4_row`, including -0.
bool pack_fp4_row(const uint16_t* bf16, uint32_t n, uint32_t block, bool e8m0,
                  uint8_t* nibbles, uint8_t* scales);
void unpack_fp4_row(const uint8_t* nibbles, const uint8_t* scales, uint32_t n,
                    uint32_t block, bool e8m0, uint16_t* bf16);

struct KvPackedPlane {
    uint32_t layer = 0, ratio = 1, rows = 0;
    std::vector<uint8_t>  cmp_fp4, cmp_scale;    // [rows][latent/2], [rows][latent/16] E4M3
    std::vector<uint8_t>  key_fp4, key_scale;    // [rows][index/2],  [rows][index/32]  E8M0
    std::vector<uint32_t> raw_rows;              // rows kept as bf16 (not on the grid)
    std::vector<uint16_t> raw_cmp, raw_key;
    std::vector<float>    carry_kv, carry_score; // [ratio][latent], ratio > 1 only
};

// The non-SWA KV state of a context of `positions` tokens.
struct KvPacked {
    uint32_t positions = 0;
    std::vector<KvPackedPlane> planes;
    uint64_t bytes() const;
    uint32_t raw_rows() const;
};

// bf16 copies of the rows some range of positions completes, and the carried
// state, taken before a replay overwrites them (runtime/session.h).
struct KvRowBackup {
    uint32_t first_pos = 0, end_pos = 0;
    struct Plane {
        uint32_t layer = 0, ratio = 1, first_row = 0, rows = 0;
        std::vector<uint16_t> cmp, key;
        std::vector<float>    carry_kv, carry_score;
    };
    std::vector<Plane> planes;
    uint64_t bytes() const;
};

class KvStore {
public:
    KvStore() = default;
    ~KvStore() { destroy(); }

    KvStore(const KvStore&) = delete;
    KvStore& operator=(const KvStore&) = delete;

    Result<void> create(gpu::MemoryAllocator& alloc, const KvStoreConfig& cfg);
    void         destroy();
    // Back to the state a fresh `create` leaves: everything zero, every
    // `score_state` slot -inf, every count 0, the window floor 0. What a
    // prefill from position 0 starts from. Keeps the capacity.
    void         clear();

    const KvStoreConfig& config() const { return cfg_; }
    // Summed over the slabs, padding included.
    uint64_t bytes() const;
    uint32_t slabs() const { return static_cast<uint32_t>(bufs_.size()); }
    // Bytes of the slab holding each region, largest first -- what a driver
    // with a 2 GiB cap has to satisfy.
    uint64_t largest_slab() const;
    // Positions currently allocated (<= config().max_context).
    uint32_t capacity() const { return cap_; }
    // Grows to hold `positions`, copying the live rows. A no-op when it
    // already does; ResourceExhausted past max_context.
    Result<void> reserve(uint32_t positions);

    // The addresses layer `l`'s dispatches need. `n_cmp` and `n_kv` come from
    // whatever seeded the layer.
    Result<KvLayerView> layer(uint32_t l) const;

    // --- seeding ------------------------------------------------------------

    // `values` is [rows][latent_dim] in fp32, the post-quantisation window KV
    // the oracle exported. Re-encoded to the E4M3 byte + UE8M0 scale the
    // kernels read; because the values are already on that grid the round trip
    // is exact, which `tests/test_gpu_attn.cpp` asserts.
    Result<void> seed_window(uint32_t layer, const float* values, uint32_t rows);
    // `values` is [n_cmp][latent_dim] fp32, stored as bf16. On a layer that
    // reads another layer's plane only the count is recorded: the rows are the
    // owner's.
    Result<void> seed_compressed(uint32_t layer, const float* values, uint32_t n_cmp);
    // The concatenated index list `sparse_attn` walks: window ring slots below
    // `window`, compressed rows offset by it, -1 for a slot holding nothing.
    Result<void> seed_topk(uint32_t layer, const int32_t* idx, uint32_t n_kv);
    // `values` is [rows][index_dim] fp32, stored as bf16: the indexer's key
    // cache, which only a kv_source_layer owns (ignored on any other layer).
    Result<void> seed_index_k(uint32_t layer, const float* values, uint32_t rows);
    // The compressor's carried group state. `score` is -inf in the slots the
    // reference has never written, which is what makes them score zero.
    Result<void> seed_cmp_state(uint32_t layer, const float* kv, const float* score,
                               uint32_t ratio);

    // --- per-step bookkeeping (design §7.4, produced not loaded) ------------

    // model.py's `get_window_topk_idxs` for one decode query: the ring slots
    // oldest first, with a slot the sequence has not reached yet -- or one
    // below the window floor -- marked -1. Writes the window half of the top-k
    // list and sets n_cmp / n_kv, leaving the compressed half for
    // `indexer.slang` stage 5 to fill at offset `window`. `n_sel` is how many
    // compressed picks the list carries -- min(index_topk, n_cmp) -- so
    // n_kv = window + n_sel. Grows the store to `position + 1`.
    Result<void> set_decode_topk(uint32_t layer, uint32_t position, uint32_t n_cmp,
                                 uint32_t n_sel);
    // n_cmp alone, for a layer whose compressed plane belongs to a source.
    Result<void> set_counts(uint32_t layer, uint32_t n_cmp, uint32_t n_kv);

    // --- restoring a context without its window (runtime/session.h) --------

    // The lowest position whose window KV the ring holds. A ring slot whose
    // position (as the step at `position` implies it) is below this is listed
    // -1, exactly like a slot the sequence has not reached. 0 in every normal
    // run; a replay that rebuilds the ring from position f sets f.
    void     set_window_floor(uint32_t position) { win_lo_ = position; }
    uint32_t window_floor() const { return win_lo_; }
    // Which position each ring slot holds, as far as the store knows: the step
    // that listed it (`set_decode_topk`) records it, `clear` empties it, and a
    // seeded ring (`seed_window`) is "unknown" until `resolve_ring(n)` says the
    // seed came from a context of `n` tokens -- every slot then holds the last
    // position below n in its residue class. A slot is only ever overwritten by
    // a later position, so this is exact.
    void resolve_ring(uint32_t n);
    // Whether slot `position % window` holds `position` and the floor admits it.
    bool ring_holds(uint32_t position) const;

    // The non-SWA state of the first `positions` tokens, packed: rows
    // floor(positions / ratio) of every plane and the carried state as it is
    // now (which is the state after `positions` only if that is where the
    // store is).
    Result<KvPacked> pack(uint32_t positions) const;
    // Writes a packed state back (rows, carried state, every layer's n_cmp).
    // The window ring and the top-k lists are not touched.
    Result<void> unpack(const KvPacked& p);

    // bf16 copies of the rows positions [first_pos, end_pos) complete -- rows
    // [first_pos / ratio, end_pos / ratio) of each plane -- and the carry.
    Result<KvRowBackup> backup_rows(uint32_t first_pos, uint32_t end_pos) const;
    // Writes back the rows that the step at `position` completes, if the backup
    // holds them.
    Result<void> restore_rows(const KvRowBackup& b, uint32_t position);
    Result<void> restore_carry(const KvRowBackup& b);

    // --- speculation: the window ring's rollback (docs/p3_dspark.md §3.5) ---
    //
    // A verify batch writes its positions' ring KV before it knows how many are
    // accepted. The compressed rows and the carried group state do not need
    // undoing -- a group's row is rewritten in place when the group completes
    // again and a half-filled group is not visible -- but the RING does: once
    // the ring has wrapped, position p' hands its slot to p' - window, which is
    // still inside an earlier query's window, so a rejected position leaves a
    // wrong row where a live query reads.
    //
    // A snapshot is `<= k` slots of `[window]` E4M3 rows plus their UE8M0
    // scales, per layer: 40 layers x 5 slots x (512 + 16) B = 105 KB at k = 5,
    // which is why this is a memcpy and not a scheme. `slots` are the ring slot
    // indices to save (position % window); saving a slot that is not written
    // costs a copy and nothing else.
    struct RingSnapshot {
        std::vector<uint32_t> layer;     // one plane a layer, in `slots` order
        std::vector<uint32_t> slot;
        std::vector<uint8_t>  val;       // [layer][slot][latent_dim]
        std::vector<uint8_t>  scale;     // [layer][slot][latent_dim / 32]
        uint32_t latent_dim = 0;
        uint64_t bytes() const { return val.size() + scale.size(); }
        size_t index(uint32_t li, uint32_t si) const {
            return (size_t(li) * slot.size() + si);
        }
    };
    Result<RingSnapshot> snapshot_ring(std::span<const uint32_t> slots,
                                       std::span<const uint32_t> layers) const;
    Result<void> restore_ring(const RingSnapshot& s);

private:
    // Offsets are in one flat space that the slabs tile in order; every region
    // lies wholly inside one slab, so `at` / `dev` map an offset (and any row
    // inside that region) by finding its slab and subtracting its base.
    struct Layout {
        uint32_t cap = 0;
        uint64_t off_win_val = 0, off_win_scale = 0, off_top = 0, off_state = 0, total = 0;
        std::vector<uint64_t> off_cmp, off_idx;   // per owner
        std::vector<uint64_t> slab_at, slab_size; // flat base and size of each slab
        uint32_t slab_of(uint64_t off) const;
    };
    Layout layout_for(uint32_t cap) const;
    uint32_t owner_index(uint32_t l) const;   // index into owners_, or kNoPlane
    // Host pointer / device address of a flat offset.
    std::byte*    at(uint64_t off) const;
    DeviceAddress dev(uint64_t off) const;
    bool          valid() const { return !bufs_.empty(); }
    Result<std::vector<gpu::GpuBuffer>> allocate_slabs(const Layout& l) const;
    static void free_slabs(gpu::MemoryAllocator* a, std::vector<gpu::GpuBuffer>& v);

    gpu::MemoryAllocator* alloc_ = nullptr;
    std::vector<gpu::GpuBuffer> bufs_{};
    KvStoreConfig         cfg_{};
    Layout                lay_{};
    uint32_t              cap_ = 0;
    std::vector<uint32_t> owners_;       // owner layer ids, ascending
    std::vector<uint32_t> owner_of_;     // [layers] -> index into owners_ or kNoPlane
    std::vector<uint32_t> rows_hw_;      // per owner: rows that may hold data
    std::vector<uint32_t> n_cmp_, n_kv_;
    // The -inf score_state fill, built once in ordinary host memory so clear()
    // can memcpy it (§7.1 rule 10).
    std::vector<float>    ninf_;
    uint32_t              win_lo_ = 0;
    static constexpr int64_t kSlotEmpty = -1, kSlotUnknown = -2;
    std::vector<int64_t>  slot_pos_;     // [window]
};

}  // namespace deepmoe::runtime
