#![allow(clippy::too_many_arguments)]

mod y4m;

pub use rav1e::scenechange::SceneChangeDetector;

use ::y4m::Decoder;
use rav1e::config::{CpuFeatureLevel, EncoderConfig};
use rav1e::prelude::{Pixel, Sequence};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::sync::Arc;
use std::time::Instant;

/// Options determining how to run scene change detection.
#[derive(Debug, Clone, Copy)]
pub struct DetectionOptions {
    /// The normal algorithm uses 8x8-block-level cost estimates
    /// to choose scenecuts.
    /// This is often more accurate, but slower.
    ///
    /// Enabling this option switches to a fast algorithm,
    /// which uses a pixel-by-pixel sum-of-absolute-differences
    /// to determine scenecuts.
    pub fast_analysis: SceneDetectionSpeed,
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
    /// Not used if `ignore_flashes` is `false`.
    pub lookahead_distance: usize,
}

impl Default for DetectionOptions {
    fn default() -> Self {
        DetectionOptions {
            fast_analysis: SceneDetectionSpeed::Slow,
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
    /// Average speed (FPS)
    pub speed: f64,
}

/// An optional callback that will fire after each frame is analyzed.
/// Arguments passed in will be, in order,
/// the number of frames analyzed, and the number of keyframes detected.
///
/// This is generally useful for displaying progress, etc.
pub type ProgressCallback = Box<dyn Fn(usize, usize)>;

pub fn new_detector<R: Read, T: Pixel>(
    dec: &mut Decoder<R>,
    opts: DetectionOptions,
) -> SceneChangeDetector<T> {
    let video_details = y4m::get_video_details(dec);
    let mut config =
        EncoderConfig::with_speed_preset(if opts.fast_analysis == SceneDetectionSpeed::Fast {
            10
        } else if opts.fast_analysis == SceneDetectionSpeed::Medium {
            8
        } else {
            6
        });

    config.min_key_frame_interval = opts
        .min_scenecut_distance
        .map(|val| val as u64)
        .unwrap_or(0);
    config.max_key_frame_interval = opts
        .max_scenecut_distance
        .map(|val| val as u64)
        .unwrap_or(u32::max_value() as u64);
    config.width = video_details.width;
    config.height = video_details.height;
    config.bit_depth = video_details.bit_depth;
    config.time_base = video_details.time_base;
    config.chroma_sampling = video_details.chroma_sampling;
    config.chroma_sample_position = video_details.chroma_sample_position;

    let sequence = Arc::new(Sequence::new(&config));
    SceneChangeDetector::new(
        config,
        CpuFeatureLevel::default(),
        opts.lookahead_distance,
        sequence,
    )
}

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

    let mut detector = new_detector(dec, opts);
    let video_details = y4m::get_video_details(dec);
    let mut frame_queue = BTreeMap::new();
    let mut keyframes = BTreeSet::new();
    keyframes.insert(0);

    let start_time = Instant::now();
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
            .cloned()
            .take(opts.lookahead_distance + 1)
            .collect::<Vec<_>>();
        if frame_set.len() < 2 {
            // End of video
            break;
        }
        if detector.analyze_next_frame(
            &frame_set,
            frameno as u64,
            *keyframes.iter().last().unwrap(),
        ) {
            keyframes.insert(frameno as u64);
        };

        if frameno > 0 {
            frame_queue.remove(&(frameno - 1));
        }

        frameno += 1;
        if let Some(ref progress_fn) = progress_callback {
            progress_fn(frameno, keyframes.len());
        }
    }
    DetectionResults {
        scene_changes: keyframes.into_iter().map(|val| val as usize).collect(),
        frame_count: frameno,
        speed: frameno as f64 / start_time.elapsed().as_secs() as f64,
    }
}

#[derive(Clone, Copy, Debug, PartialOrd, PartialEq)]
pub enum SceneDetectionSpeed {
    /// Fastest scene detection using pixel-wise comparison
    Fast,
    /// Scene detection using motion vectors
    Medium,
    /// Scene detection using histogram block-based comparison
    Slow,
}
