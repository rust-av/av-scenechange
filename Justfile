lcov:
    cargo llvm-cov --lcov --output-path=lcov.info --ignore-filename-regex tests\.rs
    genhtml lcov.info --dark-mode --flat --missed --output-directory target/coverage_html

codecov-upload:
    just codecov
    codecov --token "$AVSC_CODECOV_TOKEN" --file lcov.info --required

precommit:
    cargo fmt --all
    cargo clippy -- -D warnings
    cargo clippy --features ffmpeg,vapoursynth,ffms2 -- -D warnings
    cargo test
    cargo test --features ffmpeg,vapoursynth,ffms2
