coverage:
    cargo llvm-cov --ignore-filename-regex tests\.rs

lcov:
    cargo llvm-cov --lcov --output-path=lcov.info --ignore-filename-regex tests\.rs

codecov:
    cargo llvm-cov --codecov --output-path codecov.json --ignore-filename-regex tests\.rs
    
codecov-upload:
    just codecov && codecov --token "$AVSC_CODECOV_TOKEN" --file codecov.json --required

precommit:
    cargo fmt && cargo clippy && just lcov
