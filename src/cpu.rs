#[cfg(all(asm_x86_64, not(feature = "static_simd")))]
cpufeatures::new!(
    cpuid_avx512icl,
    "avx512f",
    "avx512cd",
    "avx512bw",
    "avx512dq",
    "avx512vl",
    "avx512ifma"
);
#[cfg(all(asm_x86_64, not(feature = "static_simd")))]
cpufeatures::new!(cpuid_avx2, "avx2");
#[cfg(all(asm_x86_64, not(feature = "static_simd")))]
cpufeatures::new!(cpuid_ssse3, "ssse3");
#[cfg(all(asm_x86_64, not(feature = "static_simd")))]
cpufeatures::new!(cpuid_sse4, "sse4.1");

#[cfg(all(asm_x86_64, not(feature = "static_simd")))]
pub use cpuid_avx2::get as has_avx2;
#[cfg(all(asm_x86_64, not(feature = "static_simd")))]
pub use cpuid_avx512icl::get as has_avx512icl;
#[cfg(all(asm_x86_64, not(feature = "static_simd")))]
pub use cpuid_sse4::get as has_sse4;
#[cfg(all(asm_x86_64, not(feature = "static_simd")))]
pub use cpuid_ssse3::get as has_ssse3;

#[cfg(all(asm_x86_64, feature = "static_simd"))]
pub const fn has_avx512icl() -> bool {
    cfg_if::cfg_if! {
        if #[cfg(all(
            target_feature = "avx512f",
            target_feature = "avx512cd",
            target_feature = "avx512bw",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512ifma"
        ))] {
            true
        } else {
            false
        }
    }
}

#[cfg(all(asm_x86_64, feature = "static_simd"))]
pub const fn has_avx2() -> bool {
    cfg_if::cfg_if! {
        if #[cfg(target_feature = "avx2")] {
            true
        } else {
            false
        }
    }
}

#[cfg(all(asm_x86_64, feature = "static_simd"))]
pub const fn has_sse4() -> bool {
    cfg_if::cfg_if! {
        if #[cfg(target_feature = "sse4.1")] {
            true
        } else {
            false
        }
    }
}

#[cfg(all(asm_x86_64, feature = "static_simd"))]
pub const fn has_ssse3() -> bool {
    cfg_if::cfg_if! {
        if #[cfg(target_feature = "ssse3")] {
            true
        } else {
            false
        }
    }
}
