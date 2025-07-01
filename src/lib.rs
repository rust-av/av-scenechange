//! Scenechange detection tool based on rav1e's scene detection code.
//! It is focused around detecting scenechange points that will be optimal
//! for an encoder to place keyframes. It may not be the best tool
//! if your use case is to generate scene changes as a human would
//! interpret them--for that there are other tools such as `SCXvid` and `WWXD`.

mod analyze;
#[macro_use]
mod cpu;
mod data;

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};

pub use av_decoders::{self, Decoder};
pub use num_rational::Rational32;
use v_frame::pixel::Pixel;

pub use crate::analyze::{SceneChangeDetector, ScenecutResult};

/// Options determining how to run scene change detection.
#[derive(Debug, Clone, Copy)]
pub struct DetectionOptions {
    /// The speed of detection algorithm to use.
    /// Slower algorithms are more accurate/better for use in encoders.
    pub analysis_speed: SceneDetectionSpeed,
    /// Enabling this will utilize heuristics to avoid scenecuts
    /// that are too close to each other.
    /// This is generally useful if you want scenecut detection
    /// for use in an encoder.
    /// If you want a raw list of scene changes, you should disable this.
    pub detect_flashes: bool,
    /// The minimum distance between two scene changes.
    pub min_scenecut_distance: Option<usize>,
    /// The maximum distance between two scene changes.
    pub max_scenecut_distance: Option<usize>,
    /// The distance to look ahead in the video
    /// for scene flash detection.
    ///
    /// Not used if `detect_flashes` is `false`.
    pub lookahead_distance: usize,
}

impl Default for DetectionOptions {
    #[inline]
    fn default() -> Self {
        DetectionOptions {
            analysis_speed: SceneDetectionSpeed::Standard,
            detect_flashes: true,
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
    /// A map of scores for each frame. Some frames may not have a score.
    pub scores: BTreeMap<usize, ScenecutResult>,
    /// The total number of frames read.
    pub frame_count: usize,
    /// Average speed (FPS)
    pub speed: f64,
}

/// # Errors
///
/// - If using a Vapoursynth script that contains an unsupported video format.
#[inline]
pub fn new_detector<T: Pixel>(
    dec: &mut Decoder,
    opts: DetectionOptions,
) -> anyhow::Result<SceneChangeDetector<T>> {
    let video_details = dec.get_video_details();

    Ok(SceneChangeDetector::new(
        (video_details.width, video_details.height),
        video_details.bit_depth,
        video_details.frame_rate.recip(),
        video_details.chroma_sampling,
        if opts.detect_flashes {
            opts.lookahead_distance
        } else {
            1
        },
        opts.analysis_speed,
        opts.min_scenecut_distance.map_or(0, |val| val),
        opts.max_scenecut_distance
            .map_or_else(|| u32::MAX as usize, |val| val),
    ))
}

/// Runs through a y4m video clip,
/// detecting where scene changes occur.
/// This is adjustable based on the `opts` parameters.
///
/// This is the preferred, simplified interface
/// for analyzing a whole clip for scene changes.
///
/// # Arguments
///
/// - `progress_callback`: An optional callback that will fire after each frame
///   is analyzed. Arguments passed in will be, in order, the number of frames
///   analyzed, and the number of keyframes detected. This is generally useful
///   for displaying progress, etc.
///
/// # Errors
///
/// - If using a Vapoursynth script that contains an unsupported video format.
///
/// # Panics
///
/// - If `opts.lookahead_distance` is 0.
#[inline]
pub fn detect_scene_changes<T: Pixel>(
    dec: &mut Decoder,
    opts: DetectionOptions,
    frame_limit: Option<usize>,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> anyhow::Result<DetectionResults> {
    assert!(opts.lookahead_distance >= 1);

    let mut detector = new_detector::<T>(dec, opts)?;
    let mut frame_queue = BTreeMap::new();
    let mut keyframes = BTreeSet::new();
    keyframes.insert(0);
    let mut scores = BTreeMap::new();

    let start_time = Instant::now();
    let mut frameno = 0;
    loop {
        let mut next_input_frameno = frame_queue.keys().last().copied().map_or(0, |key| key + 1);
        while next_input_frameno
            < (frameno + opts.lookahead_distance + 1).min(frame_limit.unwrap_or(usize::MAX))
        {
            let frame = dec.read_video_frame();
            if let Ok(frame) = frame {
                frame_queue.insert(next_input_frameno, Arc::new(frame));
                next_input_frameno += 1;
            } else {
                // End of input
                break;
            }
        }

        // The frame_queue should start at whatever the previous frame was
        let frame_set = frame_queue
            .values()
            .take(opts.lookahead_distance + 2)
            .collect::<Vec<_>>();
        if frame_set.len() < 2 {
            // End of video
            break;
        }
        if frameno == 0 {
            keyframes.insert(frameno);
        } else {
            let (cut, score) = detector.analyze_next_frame(
                &frame_set,
                frameno,
                *keyframes
                    .iter()
                    .last()
                    .expect("at least 1 keyframe should exist"),
            );
            if let Some(score) = score {
                scores.insert(frameno, score);
            }
            if cut {
                keyframes.insert(frameno);
            }
        };

        if frameno > 0 {
            frame_queue.remove(&(frameno - 1));
        }

        frameno += 1;
        if let Some(progress_fn) = progress_callback {
            progress_fn(frameno, keyframes.len());
        }
        if let Some(frame_limit) = frame_limit {
            if frameno == frame_limit {
                break;
            }
        }
    }
    Ok(DetectionResults {
        scene_changes: keyframes.into_iter().collect(),
        frame_count: frameno,
        speed: frameno as f64 / start_time.elapsed().as_secs_f64(),
        scores,
    })
}

/// Specifies the scene detection algorithm to use
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Eq)]
pub enum SceneDetectionSpeed {
    /// Fastest scene detection using pixel-wise comparison
    Fast,
    /// Scene detection using frame costs and motion vectors
    Standard,
    /// Do not perform scenecut detection, only place keyframes at fixed
    /// intervals
    None,
}
