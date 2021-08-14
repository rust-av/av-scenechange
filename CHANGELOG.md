## Version 0.6.0

- Bump rav1e dependency to 0.5. This should bring significant performance improvements,
  but may cause breaking changes.

## Version 0.5.0

- Bump rav1e dependency to 0.4
- Expose `new_detector` and `detect_scene_changes` since these
  may be useful in some situations to use directly

## Version 0.4.2

- Fix compilation on non-x86 targets
- Bump various dependencies

## Version 0.4.1

- Improve performance and memory usage

## Version 0.4.0

- [Breaking, New Feature] `detect_scene_changes` returns a `DetectionOptions` struct,
  which includes the list of scenecut frames, and the total count
  of frames in the video. The CLI output will reflect this as well.
- [Breaking] Replace the default algorithm with an 8x8-block cost-based algorithm.
  This is more accurate in many cases.
- [Breaking] As a result of the above change, now requires nasm for compilation.
  No action is needed if you use a prebuilt binary.
- [Breaking] Replace the `use_chroma` option with a `fast_analysis` option.
  The new name is more accurate, as the updated algorithm will always analyze
  only the luma plane.
- [Breaking] Move the `progress_callback` parameter from `DetectionOptions`
  to `detect_scene_changes`, since it only applies to that interface.
- [New Feature] Expose the `SceneChangeDetector` struct, which allows
  going frame-by-frame to analyze a clip. Needed for some use cases.
  `detect_scene_changes` is the simpler, preferred interface.
- The library for inputting frame data has been replaced
  with one that matches rav1e.
- Simplify/optimize some internal code.

## Version 0.3.0

- [Breaking, New Feature] Add the ability to pass a `progress_callback` function
  to the `DetectionOptions`.

## Version 0.2.0

- [Breaking] Update `y4m` dependency to 0.5

## Version 0.1.0

- Initial release
