//! Build script for av-scenechange

#![allow(clippy::unwrap_used, reason = "build script")]

use std::env;
#[cfg(feature = "asm")]
use std::path::Path;

#[cfg(feature = "asm")]
fn build_nasm_files() {
    let mut config = "
%pragma preproc sane_empty_expansion true
%define private_prefix avsc
%define ARCH_X86_32 0
%define ARCH_X86_64 1
%define PIC 1
%define STACK_ALIGNMENT 16
%define HAVE_AVX512ICL 1
"
    .to_owned();

    if env::var("CARGO_CFG_TARGET_VENDOR").unwrap() == "apple" {
        config += "%define PREFIX 1\n";
    }

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("config.asm");
    std::fs::write(&dest_path, config).expect("can write config.asm");

    let asm_files = &[
        // "src/asm/x86/cdef_avx2.asm",
        // "src/asm/x86/cdef_avx512.asm",
        // "src/asm/x86/cdef_dist.asm",
        // "src/asm/x86/cdef_rav1e.asm",
        // "src/asm/x86/cdef_sse.asm",
        // "src/asm/x86/cdef16_avx2.asm",
        // "src/asm/x86/cdef16_avx512.asm",
        // "src/asm/x86/cdef16_sse.asm",
        "src/asm/x86/ipred_avx2.asm",
        "src/asm/x86/ipred_avx512.asm",
        "src/asm/x86/ipred_sse.asm",
        "src/asm/x86/ipred16_avx2.asm",
        "src/asm/x86/ipred16_avx512.asm",
        "src/asm/x86/ipred16_sse.asm",
        // "src/asm/x86/itx_avx2.asm",
        // "src/asm/x86/itx_avx512.asm",
        // "src/asm/x86/itx_sse.asm",
        // "src/asm/x86/itx16_avx2.asm",
        // "src/asm/x86/itx16_avx512.asm",
        // "src/asm/x86/itx16_sse.asm",
        // "src/asm/x86/looprestoration_avx2.asm",
        // "src/asm/x86/looprestoration_avx512.asm",
        // "src/asm/x86/looprestoration_sse.asm",
        // "src/asm/x86/looprestoration16_avx2.asm",
        // "src/asm/x86/looprestoration16_avx512.asm",
        // "src/asm/x86/looprestoration16_sse.asm",
        // "src/asm/x86/me.asm",
        "src/asm/x86/sad_plane.asm",
        "src/asm/x86/satd.asm",
        "src/asm/x86/satd16_avx2.asm",
        // "src/asm/x86/sse.asm",
        "src/asm/x86/tables.asm",
    ];

    println!("cargo:rerun-if-changed=src/asm/x86");

    let obj = nasm_rs::Build::new()
        .min_version(2, 15, 0)
        .include(&out_dir)
        .include("src")
        .files(asm_files)
        .compile_objects()
        .unwrap_or_else(|e| {
            panic!("NASM build failed. Make sure you have nasm installed or disable the \"asm\" feature.\n\
                    You can get NASM from https://nasm.us or your system's package manager.\n\
                    \n\
                    error: {e}");
        });

    let mut cc = cc::Build::new();
    for o in obj {
        cc.object(o);
    }
    cc.compile("avscasm");
}

#[cfg(feature = "asm")]
fn build_neon_asm_files() {
    let mut config = "
#define PRIVATE_PREFIX avsc_
#define ARCH_AARCH64 1
#define ARCH_ARM 0
#define CONFIG_LOG 1
#define HAVE_ASM 1
"
    .to_owned();

    if env::var("CARGO_CFG_TARGET_VENDOR").unwrap() == "apple" {
        config += "#define PREFIX 1\n";
    }
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("config.h");
    std::fs::write(&dest_path, config).expect("can write config.h");

    let asm_files = &[
        // "src/asm/arm/64/cdef.S",
        // "src/asm/arm/64/cdef16.S",
        // "src/asm/arm/64/cdef_dist.S",
        // "src/asm/arm/64/itx.S",
        // "src/asm/arm/64/itx16.S",
        "src/asm/arm/64/ipred.S",
        "src/asm/arm/64/ipred16.S",
        // "src/asm/arm/64/sad.S",
        "src/asm/arm/64/satd.S",
        // "src/asm/arm/64/sse.S",
        "src/asm/arm/tables.S",
    ];

    println!("cargo:rerun-if-changed=src/asm/arm");

    cc::Build::new()
        .files(asm_files)
        .include(".")
        .include(&out_dir)
        .compile("avsc-aarch64");
}

#[allow(unused_variables)]
fn main() {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    // let env = env::var("CARGO_CFG_TARGET_ENV").unwrap();

    #[cfg(feature = "asm")]
    {
        if arch == "x86_64" {
            println!("cargo:rustc-cfg=asm_x86_64");
            build_nasm_files();
        }
        if arch == "aarch64" {
            println!("cargo:rustc-cfg=asm_neon");
            build_neon_asm_files();
        }
    }

    println!("cargo:rustc-env=PROFILE={}", env::var("PROFILE").unwrap());
    if let Ok(value) = env::var("CARGO_CFG_TARGET_FEATURE") {
        println!("cargo:rustc-env=CARGO_CFG_TARGET_FEATURE={value}");
    }
    println!(
        "cargo:rustc-env=CARGO_ENCODED_RUSTFLAGS={}",
        env::var("CARGO_ENCODED_RUSTFLAGS").unwrap()
    );
}
