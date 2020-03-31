## Version 0.4.0 (unreleased)
- [Breaking, New Feature] Now returns a `DetectionOptions` struct,
  which includes the list of scenecut frames, and the total count
  of frames in the video. The CLI output will reflect this as well.
- [Breaking] Replace the default algorithm with an 8x8-block cost-based algorithm.
  This is more accurate in many cases.
- [Breaking] The interface for inputting frame data has been replaced
  with one that matches rav1e.
- [Breaking] Replace the `use_chroma` option with a `fast_analysis` option.
  The new name is more accurate, as the updated algorithm will always analyze
  only the luma plane.
- Simplify/optimize some internal code.

## Version 0.3.0
- [Breaking, New Feature] Add the ability to pass a `progress_callback` function
  to the `DetectionOptions`.

## Version 0.2.0
- [Breaking] Update `y4m` dependency to 0.5

## Version 0.1.0
- Initial release
