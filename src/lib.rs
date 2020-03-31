#![allow(clippy::too_many_arguments)]

mod cost;
mod frame;
mod mc;
mod me;
mod pred;
mod refs;
mod util;
mod y4m;

use crate::cost::{estimate_inter_costs, estimate_intra_costs};
use ::y4m::Decoder;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use v_frame::frame::Frame;
use v_frame::pixel::{CastFromPrimitive, ChromaSampling, Pixel};
use v_frame::plane::Plane;

pub use v_frame;

/// Options determining how to run scene change detection.
pub struct DetectionOptions {
    /// The normal algorithm uses 8x8-block-level cost estimates
    /// to choose scenecuts.
    /// This is often more accurate, but slower.
    ///
    /// Enabling this option switches to a fast algorithm,
    /// which uses a pixel-by-pixel sum-of-absolute-differences
    /// to determine scenecuts.
    pub fast_analysis: bool,
    /// Enabling this will utilize heuristics to avoid scenecuts
    /// that are too close to each other.
    /// This is generally useful if you want scenecut detection
    /// for use in an encoder.
    /// If you want a raw list of scene changes, you should disable this.
    pub ignore_flashes: bool,
    /// The minimum distane between two scene changes.
    pub min_scenecut_distance: Option<usize>,
    /// The maximum distance between two scene changes.
    pub max_scenecut_distance: Option<usize>,
    /// The distance to look ahead in the video
    /// for scene flash detection.
    ///
    /// Not used if `ignore_flashes` is `true`.
    pub lookahead_distance: usize,
}

impl Default for DetectionOptions {
    fn default() -> Self {
        DetectionOptions {
            fast_analysis: false,
            ignore_flashes: false,
            lookahead_distance: 5,
            min_scenecut_distance: None,
            max_scenecut_distance: None,
        }
    }
}

/// Results from a scene change detection pass.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
pub struct DetectionResults {
    /// The 0-indexed frame numbers where scene changes were detected.
    pub scene_changes: Vec<usize>,
    /// The total number of frames read.
    pub frame_count: usize,
}

/// An optional callback that will fire after each frame is analyzed.
/// Arguments passed in will be, in order,
/// the number of frames analyzed, and the number of keyframes detected.
///
/// This is generally useful for displaying progress, etc.
pub type ProgressCallback = Box<dyn Fn(usize, usize)>;

/// Runs through a y4m video clip,
/// detecting where scene changes occur.
/// This is adjustable based on the `opts` parameters.
///
/// This is the preferred, simplified interface
/// for analyzing a whole clip for scene changes.
pub fn detect_scene_changes<R: Read, T: Pixel>(
    dec: &mut Decoder<R>,
    opts: DetectionOptions,
    progress_callback: Option<ProgressCallback>,
) -> DetectionResults {
    assert!(opts.lookahead_distance >= 1);

    let video_details = y4m::get_video_details(dec);
    let mut detector = SceneChangeDetector::new(
        video_details.bit_depth,
        video_details.chroma_sampling,
        &opts,
    );
    let mut frame_queue = BTreeMap::new();
    let mut keyframes = BTreeSet::new();
    let mut frameno = 0;
    loop {
        let mut next_input_frameno = frame_queue
            .keys()
            .last()
            .copied()
            .map(|key| key + 1)
            .unwrap_or(0);
        while next_input_frameno < frameno + opts.lookahead_distance {
            let frame = y4m::read_video_frame::<R, T>(dec, &video_details);
            if let Ok(frame) = frame {
                frame_queue.insert(next_input_frameno, frame);
                next_input_frameno += 1;
            } else {
                // End of input
                break;
            }
        }

        // The frame_queue should start at whatever the previous frame was
        let frame_set = frame_queue
            .values()
            .take(opts.lookahead_distance + 1)
            .collect::<Vec<_>>();
        if frame_set.len() < 2 {
            // End of video
            break;
        }
        detector.analyze_next_frame(&frame_set, frameno, &mut keyframes);

        if frameno > 0 {
            frame_queue.remove(&(frameno - 1));
        }

        frameno += 1;
        if let Some(ref progress_fn) = progress_callback {
            progress_fn(frameno, keyframes.len());
        }
    }
    DetectionResults {
        scene_changes: keyframes.into_iter().collect(),
        frame_count: frameno,
    }
}

/// Runs keyframe detection on frames from the lookahead queue.
///
/// This is a lower-level interface which allows going frame-by-frame.
/// It is recommended to use `detect_scene_changes` instead.
/// This interface is a fallback if analyzing the whole video in one pass
/// does not meet your use case.
pub struct SceneChangeDetector<'a> {
    /// Minimum average difference between YUV deltas that will trigger a scene change.
    threshold: usize,
    opts: &'a DetectionOptions,
    /// Frames that cannot be marked as keyframes due to the algorithm excluding them.
    /// Storing the frame numbers allows us to avoid looking back more than one frame.
    excluded_frames: BTreeSet<usize>,
    chroma_sampling: ChromaSampling,
    /// The bit depth of the video.
    bit_depth: usize,
}

impl<'a> SceneChangeDetector<'a> {
    pub fn new(
        bit_depth: usize,
        chroma_sampling: ChromaSampling,
        opts: &'a DetectionOptions,
    ) -> Self {
        // This implementation is based on a Python implementation at
        // https://pyscenedetect.readthedocs.io/en/latest/reference/detection-methods/.
        // The Python implementation uses HSV values and a threshold of 30. Comparing the
        // YUV values was sufficient in most cases, and avoided a more costly YUV->RGB->HSV
        // conversion, but the deltas needed to be scaled down. The deltas for keyframes
        // in YUV were about 1/3 to 1/2 of what they were in HSV, but non-keyframes were
        // very unlikely to have a delta greater than 3 in YUV, whereas they may reach into
        // the double digits in HSV. Therefore, 12 was chosen as a reasonable default threshold.
        // This may be adjusted later.
        const BASE_THRESHOLD: usize = 12;
        Self {
            threshold: BASE_THRESHOLD * bit_depth / 8,
            opts,
            excluded_frames: BTreeSet::new(),
            chroma_sampling,
            bit_depth,
        }
    }
    /// Runs keyframe detection on the next frame in the lookahead queue.
    ///
    /// This function requires that a subset of input frames
    /// is passed to it in order, and that `keyframes` is only
    /// updated from this method. `input_frameno` should correspond
    /// to the second frame in `frame_set`.
    ///
    /// This will gracefully handle the first frame in the video as well.
    pub fn analyze_next_frame<T: Pixel>(
        &mut self,
        frame_set: &[&Frame<T>],
        input_frameno: usize,
        keyframes: &mut BTreeSet<usize>,
    ) {
        if input_frameno == 0 {
            keyframes.insert(input_frameno);
            return;
        }

        // Find the distance to the previous keyframe.
        let previous_keyframe = *keyframes.iter().last().unwrap();
        let distance = input_frameno - previous_keyframe;

        // Handle minimum and maximum key frame intervals.
        if distance < self.opts.min_scenecut_distance.unwrap_or(0) {
            return;
        }
        if distance
            >= self
                .opts
                .max_scenecut_distance
                .unwrap_or(usize::max_value())
        {
            keyframes.insert(input_frameno);
            return;
        }

        self.exclude_scene_flashes(&frame_set, input_frameno, previous_keyframe);

        if self.is_key_frame(frame_set[0], frame_set[1], input_frameno, previous_keyframe) {
            keyframes.insert(input_frameno);
        }
    }

    /// Determines if `current_frame` should be a keyframe.
    fn is_key_frame<T: Pixel>(
        &self,
        previous_frame: &Frame<T>,
        current_frame: &Frame<T>,
        current_frameno: usize,
        previous_keyframe: usize,
    ) -> bool {
        if self.excluded_frames.contains(&current_frameno) {
            return false;
        }

        self.has_scenecut(
            previous_frame,
            current_frame,
            current_frameno,
            previous_keyframe,
        )
    }
    /// Uses lookahead to avoid coding short flashes as scenecuts.
    /// Saves excluded frame numbers in `self.excluded_frames`.
    fn exclude_scene_flashes<T: Pixel>(
        &mut self,
        frame_subset: &[&Frame<T>],
        frameno: usize,
        previous_keyframe: usize,
    ) {
        let lookahead_distance = self.opts.lookahead_distance;

        if frame_subset.len() - 1 < lookahead_distance {
            // Don't add a keyframe in the last frame pyramid.
            // It's effectively the same as a scene flash,
            // and really wasteful for compression.
            for frame in frameno..=(frameno + self.opts.lookahead_distance) {
                self.excluded_frames.insert(frame);
            }
            return;
        }

        // Where A and B are scenes: AAAAAABBBAAAAAA
        // If BBB is shorter than lookahead_distance, it is detected as a flash
        // and not considered a scenecut.
        //
        // Search starting with the furthest frame,
        // to enable early loop exit if we find a scene flash.
        for j in (1..=lookahead_distance).rev() {
            if !self.has_scenecut(
                frame_subset[0],
                frame_subset[j],
                frameno - 1 + j,
                previous_keyframe,
            ) {
                // Any frame in between `0` and `j` cannot be a real scenecut.
                for i in 0..=j {
                    let frameno = frameno + i - 1;
                    self.excluded_frames.insert(frameno);
                }
                // Because all frames in this gap are already excluded,
                // exit the loop early as an optimization.
                break;
            }
        }

        // Where A-F are scenes: AAAAABBCCDDEEFFFFFF
        // If each of BB ... EE are shorter than `lookahead_distance`, they are
        // detected as flashes and not considered scenecuts.
        // Instead, the first F frame becomes a scenecut.
        // If the video ends before F, no frame becomes a scenecut.
        for i in 1..lookahead_distance {
            if self.has_scenecut(
                frame_subset[i],
                frame_subset[lookahead_distance],
                frameno - 1 + lookahead_distance,
                previous_keyframe,
            ) {
                // If the current frame is the frame before a scenecut, it cannot also be the frame of a scenecut.
                let frameno = frameno + i - 1;
                self.excluded_frames.insert(frameno);
            }
        }
    }
    /// Run a comparison between two frames to determine if they qualify for a scenecut.
    ///
    /// The current algorithm detects fast cuts using changes in colour and intensity between frames.
    /// Since the difference between frames is used, only fast cuts are detected
    /// with this method. This is intended to change via https://github.com/xiph/rav1e/issues/794.
    fn has_scenecut<T: Pixel>(
        &self,
        frame1: &Frame<T>,
        frame2: &Frame<T>,
        frameno: usize,
        previous_keyframe: usize,
    ) -> bool {
        if self.opts.fast_analysis {
            let len = frame2.planes[0].cfg.width * frame2.planes[0].cfg.height;
            let delta = self.delta_in_planes(&frame1.planes[0], &frame2.planes[0]);
            delta >= self.threshold as u64 * len as u64
        } else {
            let intra_costs = estimate_intra_costs(frame2, self.bit_depth);
            let intra_cost = intra_costs.iter().map(|&cost| cost as u64).sum::<u64>() as f64
                / intra_costs.len() as f64;

            let inter_costs =
                estimate_inter_costs(frame2, frame1, self.bit_depth, self.chroma_sampling);
            let inter_cost = inter_costs.iter().map(|&cost| cost as u64).sum::<u64>() as f64
                / inter_costs.len() as f64;

            // Sliding scale, more likely to choose a keyframe
            // as we get farther from the last keyframe.
            // Based on x264 scenecut code.
            //
            // `THRESH_MAX` determines how likely we are
            // to choose a keyframe, between 0.0-1.0.
            // Higher values mean we are more likely to choose a keyframe.
            // `0.4` was chosen based on trials of the `scenecut-720p` set in AWCY,
            // as it appeared to provide the best average compression.
            // This also matches the default scenecut threshold in x264.
            const THRESH_MAX: f64 = 0.4;
            const THRESH_MIN: f64 = THRESH_MAX * 0.25;
            let distance_from_keyframe = frameno - previous_keyframe;
            let min_keyint = self.opts.min_scenecut_distance.unwrap_or(1);
            let max_keyint = self.opts.max_scenecut_distance;
            let bias = match max_keyint {
                Some(max_keyint) => {
                    if distance_from_keyframe <= min_keyint / 4 {
                        THRESH_MIN / 4.0
                    } else if distance_from_keyframe <= min_keyint {
                        THRESH_MIN * distance_from_keyframe as f64 / min_keyint as f64
                    } else {
                        THRESH_MIN
                            + (THRESH_MAX - THRESH_MIN)
                                * (distance_from_keyframe - min_keyint) as f64
                                / (max_keyint - min_keyint) as f64
                    }
                }
                None => THRESH_MAX,
            };
            let threshold = intra_cost * (1.0 - bias);
            inter_cost > threshold
        }
    }

    fn delta_in_planes<T: Pixel>(&self, plane1: &Plane<T>, plane2: &Plane<T>) -> u64 {
        let mut delta = 0;
        let lines = plane1.rows_iter().zip(plane2.rows_iter());

        for (l1, l2) in lines {
            let delta_line = l1
                .iter()
                .zip(l2.iter())
                .map(|(&p1, &p2)| (i16::cast_from(p1) - i16::cast_from(p2)).abs() as u64)
                .sum::<u64>();
            delta += delta_line;
        }
        delta
    }
}
