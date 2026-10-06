#pragma once

#include "core/namespace.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace cachedmoe {
// Copy a GPU-written coherent, write-combining mapping into cached host RAM.
// Ordinary scalar reads issue an uncached transaction for every element;
// streaming loads gather whole WC cache lines. Source must be GPU-idle.
inline void wc_readback(void* dst,const void* src,size_t bytes) {
#if defined(__AVX2__)
    if(bytes>=64) {
        auto* out=static_cast<std::byte*>(dst);
        const auto* in=static_cast<const std::byte*>(src);
        const size_t head=(32u-(reinterpret_cast<uintptr_t>(in)&31u))&31u;
        if(head) { std::memcpy(out,in,head);out+=head;in+=head;bytes-=head; }
        const auto* s=reinterpret_cast<const __m256i*>(in);
        auto* d=reinterpret_cast<__m256i*>(out);
        const size_t n=bytes/32;
        for(size_t i=0;i<n;++i)_mm256_storeu_si256(d+i,_mm256_stream_load_si256(s+i));
        _mm_mfence();
        if(bytes%32)std::memcpy(out+n*32,in+n*32,bytes%32);
        return;
    }
#endif
    std::memcpy(dst,src,bytes);
}
}
