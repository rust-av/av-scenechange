#[cfg(asm_x86_64)]
cpufeatures::new!(
    cpuid_avx512icl,
    "avx512f",
    "avx512cd",
    "avx512bw",
    "avx512dq",
    "avx512vl",
    "avx512ifma"
);
#[cfg(asm_x86_64)]
cpufeatures::new!(cpuid_avx2, "avx2");
#[cfg(asm_x86_64)]
cpufeatures::new!(cpuid_ssse3, "ssse3");
#[cfg(asm_x86_64)]
cpufeatures::new!(cpuid_sse4, "sse4.1");

#[cfg(asm_x86_64)]
pub use cpuid_avx2::get as has_avx2;
#[cfg(asm_x86_64)]
pub use cpuid_avx512icl::get as has_avx512icl;
#[cfg(asm_x86_64)]
pub use cpuid_sse4::get as has_sse4;
#[cfg(asm_x86_64)]
pub use cpuid_ssse3::get as has_ssse3;
