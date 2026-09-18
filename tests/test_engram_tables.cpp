// The engram hash constants derived at startup (runtime/engram_tables.h)
// against the ones tools/oracle.py exported into tests/data/l3 from the
// reference's own `NgramHashState` (Track R2, docs/p4_kv_ux.md §5).
//
// `engram_tables.rng` needs nothing; `engram_tables.derived_equals_l3_export`
// needs tokenizer.json from DEEPMOE_MODEL_DIR and config.json beside it.
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "model/v41_config.h"
#include "runtime/engram_tables.h"
#include "tests/test_framework.h"

#ifndef DEEPMOE_TEST_DATA_DIR
#define DEEPMOE_TEST_DATA_DIR "tests/data"
#endif

using namespace deepmoe;

DEEPMOE_TEST(engram_tables, numpy_rng_and_normaliser) {
    // numpy.random.default_rng(10007 * layer).integers(0, bound, 4, np.int64),
    // bound = (2**63 - 1) // 99092 // 2, printed by numpy 2.x.
    const int64_t bound = (INT64_MAX / 99092) / 2;
    const std::vector<int64_t> l1 = runtime::numpy_rng_integers(10007, bound, 4);
    const std::vector<int64_t> l14 = runtime::numpy_rng_integers(10007 * 14, bound, 4);
    CHECK_EQ(l1[0], 38316048023122LL);
    CHECK_EQ(l1[1], 2419938046656LL);
    CHECK_EQ(l1[2], 17979836159674LL);
    CHECK_EQ(l1[3], 36993668729195LL);
    CHECK_EQ(l14[0], 33858405369630LL);
    CHECK_EQ(l14[3], 41309613242795LL);

    // HF normalizers.Sequence([NFKC, NFD, StripAccents, Lowercase, ...]) on a few
    // strings whose answers engram.py relies on.
    CHECK_EQ(runtime::engram_normalize(" The"), std::string("the"));
    CHECK_EQ(runtime::engram_normalize("THE"), std::string("the"));
    CHECK_EQ(runtime::engram_normalize(" "), std::string(" "));             // a lone space survives
    CHECK_EQ(runtime::engram_normalize("\n\n"), std::string(" "));
    CHECK_EQ(runtime::engram_normalize("\xC3\xA9t\xC3\xA9"), std::string("ete"));   // été
    CHECK_EQ(runtime::engram_normalize("\xEF\xBC\xA1"), std::string("a"));          // fullwidth A
    CHECK_EQ(runtime::engram_normalize("\xE2\x80\x83x "), std::string("x"));        // em space stripped
    CHECK_EQ(runtime::engram_normalize("\xEF\xAC\x81"), std::string("fi"));         // ligature
}

DEEPMOE_TEST(engram_tables, derived_equals_l3_export) {
    const char* dir = std::getenv("DEEPMOE_MODEL_DIR");
    if (!dir) { std::printf("      SKIP engram_tables: set DEEPMOE_MODEL_DIR\n"); return; }
    auto cfg = V41Config::load(std::string(dir) + "/config.json");
    REQUIRE_OK(cfg);
    auto derived = runtime::derive_engram_tables(std::string(dir), cfg->text);
    REQUIRE_OK(derived);
    auto exported = runtime::EngramTables::load(std::string(DEEPMOE_TEST_DATA_DIR) + "/l3");
    if (!exported) {
        std::printf("      SKIP engram_tables: no L3 export (%s)\n", exported.error().str().c_str());
        return;
    }
    CHECK_EQ(derived->compressed_vocab_size, exported->compressed_vocab_size);
    CHECK_EQ(derived->pad_id, exported->pad_id);
    REQUIRE(derived->token_map.size() == exported->token_map.size());
    size_t diff = 0;
    for (size_t i = 0; i < derived->token_map.size(); ++i)
        diff += derived->token_map[i] != exported->token_map[i];
    std::printf("    token map: %zu entries, %zu differ; compressed vocab %u\n",
                derived->token_map.size(), diff, derived->compressed_vocab_size);
    CHECK_EQ(diff, size_t(0));
    REQUIRE(derived->layers.size() == exported->layers.size());
    for (size_t l = 0; l < derived->layers.size(); ++l) {
        const auto& a = derived->layers[l];
        const auto& b = exported->layers[l];
        CHECK_EQ(a.layer, b.layer);
        CHECK_EQ(a.num_embeddings, b.num_embeddings);
        CHECK(a.multipliers == b.multipliers);
        CHECK(a.primes == b.primes);
        CHECK(a.offsets == b.offsets);
        std::printf("    layer %u: multipliers %s, primes %s, offsets %s\n", a.layer,
                    a.multipliers == b.multipliers ? "equal" : "DIFFER",
                    a.primes == b.primes ? "equal" : "DIFFER",
                    a.offsets == b.offsets ? "equal" : "DIFFER");
    }
    // The row ids a real trajectory hashes to, both ways.
    std::vector<uint32_t> hist = {0, 1337, 42, 128799, 99, 5, 7};
    for (uint32_t layer : {1u, 14u})
        for (uint64_t p = 0; p < hist.size(); ++p) {
            uint64_t ra[24], rb[24];
            REQUIRE_OK(derived->hash_rows(layer, hist, p, ra));
            REQUIRE_OK(exported->hash_rows(layer, hist, p, rb));
            CHECK(std::equal(ra, ra + 24, rb));
        }
}
