// Copyright (c) 2017-2019, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::env;
use std::fs;
use std::path::Path;

fn rerun_dir<P: AsRef<Path>>(dir: P) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        println!("cargo:rerun-if-changed={}", path.to_string_lossy());

        if path.is_dir() {
            rerun_dir(path);
        }
    }
}

fn build_nasm_files() {
    use std::fs::File;
    use std::io::Write;
    let out_dir = env::var("OUT_DIR").unwrap();
    {
        let dest_path = Path::new(&out_dir).join("config.asm");
        let mut config_file = File::create(dest_path).unwrap();
        config_file
            .write_all(b"	%define private_prefix scenechangeasm\n")
            .unwrap();
        config_file.write_all(b"	%define ARCH_X86_32 0\n").unwrap();
        config_file.write_all(b" %define ARCH_X86_64 1\n").unwrap();
        config_file.write_all(b"	%define PIC 1\n").unwrap();
        config_file
            .write_all(b" %define STACK_ALIGNMENT 16\n")
            .unwrap();
        if cfg!(target_os = "macos") {
            config_file.write_all(b" %define PREFIX 1\n").unwrap();
        }
    }
    let mut config_include_arg = String::from("-I");
    config_include_arg.push_str(&out_dir);
    config_include_arg.push('/');
    nasm_rs::compile_library_args(
        "scenechangeasm",
        &[
            "src/asm/x86/ipred.asm",
            "src/asm/x86/ipred_ssse3.asm",
            "src/asm/x86/mc.asm",
            "src/asm/x86/mc_ssse3.asm",
            "src/asm/x86/me.asm",
            "src/asm/x86/sad_avx.asm",
            "src/asm/x86/sad_sse2.asm",
            "src/asm/x86/satd.asm",
            "src/asm/x86/tables.asm",
        ],
        &[&config_include_arg, "-Isrc/"],
    );
    println!("cargo:rustc-link-lib=static=scenechangeasm");
    rerun_dir("src/asm/x86");
    rerun_dir("src/ext/x86");
}

#[allow(unused_variables)]
fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        println!("cargo:rustc-cfg=nasm_x86_64");
        build_nasm_files()
    }
}
