lcov:
    cargo llvm-cov --lcov --output-path=lcov.info --ignore-filename-regex tests\.rs
    genhtml lcov.info --dark-mode --flat --missed --output-directory target/coverage_html

codecov-upload:
    just codecov
    codecov --token "$AVSC_CODECOV_TOKEN" --file lcov.info --required

precommit:
    cargo +nightly fmt
    cargo clippy -- -D warnings
    cargo clippy --features ffmpeg -- -D warnings
    cargo clippy --features vapoursynth -- -D warnings
    cargo clippy --features ffms2 -- -D warnings
    just lcov
