#![deny(clippy::all)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::inconsistent_struct_constructor)]
#![allow(clippy::inline_always)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::similar_names)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::use_self)]
#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::create_dir)]
#![warn(clippy::dbg_macro)]
#![warn(clippy::default_numeric_fallback)]
#![warn(clippy::exit)]
#![warn(clippy::filetype_is_file)]
#![warn(clippy::float_cmp_const)]
#![warn(clippy::if_then_some_else_none)]
#![warn(clippy::lossy_float_literal)]
#![warn(clippy::map_err_ignore)]
#![warn(clippy::mem_forget)]
#![warn(clippy::mod_module_files)]
#![warn(clippy::multiple_inherent_impl)]
#![warn(clippy::pattern_type_mismatch)]
#![warn(clippy::rc_buffer)]
#![warn(clippy::rc_mutex)]
#![warn(clippy::rest_pat_in_fully_bound_structs)]
#![warn(clippy::same_name_method)]
#![warn(clippy::str_to_string)]
#![warn(clippy::string_to_string)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::unnecessary_self_imports)]
#![warn(clippy::unneeded_field_pattern)]
#![warn(clippy::use_debug)]
#![warn(clippy::verbose_file_reads)]

mod y4m;

use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    sync::Arc,
    time::Instant,
};

use ::y4m::Decoder;
pub use rav1e::scenechange::SceneChangeDetector;
use rav1e::{
    config::{CpuFeatureLevel, EncoderConfig},
    prelude::{Pixel, Sequence},
};

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
    /// The minimum distane between two scene changes.
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
    /// The total number of frames read.
    pub frame_count: usize,
    /// Average speed (FPS)
    pub speed: f64,
}

pub fn new_detector<R: Read, T: Pixel>(
    dec: &mut Decoder<R>,
    opts: DetectionOptions,
) -> SceneChangeDetector<T> {
    let video_details = y4m::get_video_details(dec);
    let mut config =
        EncoderConfig::with_speed_preset(if opts.analysis_speed == SceneDetectionSpeed::Fast {
            10
        } else {
            8
        });

    config.min_key_frame_interval = opts.min_scenecut_distance.map_or(0, |val| val as u64);
    config.max_key_frame_interval = opts
        .max_scenecut_distance
        .map_or_else(|| u32::MAX.into(), |val| val as u64);
    config.width = video_details.width;
    config.height = video_details.height;
    config.bit_depth = video_details.bit_depth;
    config.time_base = video_details.time_base;
    config.chroma_sampling = video_details.chroma_sampling;
    config.chroma_sample_position = video_details.chroma_sample_position;
    // force disable temporal RDO to disable intra cost caching
    config.speed_settings.transform.tx_domain_distortion = true;

    let sequence = Arc::new(Sequence::new(&config));
    SceneChangeDetector::new(
        config,
        CpuFeatureLevel::default(),
        if opts.detect_flashes {
            opts.lookahead_distance
        } else {
            1
        },
        sequence,
    )
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
/// # Panics
///
/// - If `opts.lookahead_distance` is 0.
#[allow(clippy::needless_pass_by_value)]
pub fn detect_scene_changes<R: Read, T: Pixel>(
    dec: &mut Decoder<R>,
    opts: DetectionOptions,
    frame_limit: Option<usize>,
    progress_callback: Option<&dyn Fn(usize, usize)>,
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
        let mut next_input_frameno = frame_queue.keys().last().copied().map_or(0, |key| key + 1);
        while next_input_frameno
            < (frameno + opts.lookahead_distance + 1).min(frame_limit.unwrap_or(usize::MAX))
        {
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
            .take(opts.lookahead_distance + 2)
            .collect::<Vec<_>>();
        if frame_set.len() < 2 {
            // End of video
            break;
        }
        if frameno == 0
            || detector.analyze_next_frame(
                &frame_set,
                frameno as u64,
                *keyframes.iter().last().unwrap(),
            )
        {
            keyframes.insert(frameno as u64);
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
    DetectionResults {
        scene_changes: keyframes.into_iter().map(|val| val as usize).collect(),
        frame_count: frameno,
        speed: frameno as f64 / start_time.elapsed().as_secs_f64(),
    }
}

#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Eq)]
pub enum SceneDetectionSpeed {
    /// Fastest scene detection using pixel-wise comparison
    Fast,
    /// Scene detection using motion vectors
    Standard,
}
