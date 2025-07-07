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
    sync::{Arc, Condvar, Mutex},
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

/// Runs through a seekable video clip,
/// detecting where scene changes occur.
/// This is adjustable based on the `opts` parameters.
///
/// This is the preferred interface for analyzing a whole clip
/// for scene changes while maximizing decode performance.
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
/// - If `dec` does not support seeking.
#[inline]
pub fn detect_scene_changes_recursive<T: Pixel>(
    dec: &mut Decoder,
    opts: DetectionOptions,
    frame_limit: Option<usize>,
    progress_callback: Option<&dyn Fn(usize, usize)>,
    maximum_concurrency: usize,
) -> anyhow::Result<DetectionResults> {
    assert!(opts.lookahead_distance >= 1);
    assert!(
        dec.seek_video_frame::<T>(0).is_ok(),
        "Recursive detection is not supported for this Decoder"
    );

    let video_details = dec.get_video_details();
    let total_frames = video_details.num_frames.unwrap();
    // Build a binary tree of keyframes
    let mut root = Node::new::<T>(dec, opts, frame_limit, 0, total_frames, true);
    // Traverse the tree depth-first and detect scene changes chronologically
    // for each node and at most `maximum_concurrency` nodes at a time
    let result = root.detect_scene_changes::<T>(
        dec,
        opts,
        frame_limit,
        progress_callback,
        &NodeSemaphore::new(maximum_concurrency),
    )?;

    Ok(result)
}

struct Node {
    start: usize,
    end: usize,
    children: Option<(Box<Node>, Box<Node>)>,
    start_is_keyframe: bool,
    keyframes: BTreeSet<usize>,
    scores: BTreeMap<usize, ScenecutResult>,
    elapsed_seconds: f64,
}

impl Node {
    fn new<T: Pixel>(
        dec: &mut Decoder,
        opts: DetectionOptions,
        frame_limit: Option<usize>,
        start: usize,
        end: usize,
        start_is_keyframe: bool,
    ) -> Self {
        let mut this = Node {
            start,
            end,
            children: None,
            start_is_keyframe,
            keyframes: BTreeSet::new(),
            scores: BTreeMap::new(),
            elapsed_seconds: 0.0,
        };
        // A tree must contain at least 10 seconds
        let minimum_frame_count = (10 * dec.get_video_details().frame_rate.to_integer()) as usize;

        if (end - start) <= minimum_frame_count {
            // Tree contains fewer than 10 seconds, do not split into subtrees
            return this;
        }

        println!("Node[{}-{}]: Initializing", start, end);

        let mut detector = new_detector::<T>(dec, opts).unwrap();
        let mut next_keyframe = None;
        let middle_frame_index = start + (end - start) / 2;
        let mut frame_queue = BTreeMap::new();
        let mut current_frame = middle_frame_index;
        loop {
            let mut next_input_frameno = frame_queue
                .keys()
                .last()
                .copied()
                .map_or(current_frame, |key| key + 1);

            // Don't search more than 10 seconds ahead
            while next_input_frameno
                < (current_frame + opts.lookahead_distance + 1)
                    .min(frame_limit.unwrap_or(end.min(start + minimum_frame_count)))
            {
                let frame = dec.seek_video_frame(next_input_frameno);
                if let Ok(frame) = frame {
                    frame_queue.insert(next_input_frameno, Arc::new(frame));
                    next_input_frameno += 1;
                } else {
                    // End of input
                    break;
                }
            }

            // frame_queue should start whatever the previous frame was
            let frame_set = frame_queue
                .values()
                .take(opts.lookahead_distance + 2)
                .collect::<Vec<_>>();
            if frame_set.len() < 2 {
                // End of video
                break;
            }

            // Only the start can be assumed as the previous keyframe since the middle frame
            // could be too close to the next scene cut. This also makes the maximum scene
            // length limit incompatible with this method.
            let (cut, _) = detector.analyze_next_frame(&frame_set, current_frame, start);
            if cut {
                // Next keyframe found
                next_keyframe = Some(current_frame);
            }

            if current_frame > 0 {
                frame_queue.remove(&(current_frame - 1));
            }

            current_frame += 1;
            // if let Some(progress_callback) = progress_callback {
            //     progress_callback(frameno, self.keyframes.len());
            // }
            if let Some(frame_limit) = frame_limit {
                if current_frame == frame_limit {
                    break;
                }
            }

            if next_keyframe.is_some() {
                // First keyframe found, stop seeking
                break;
            }
        }

        let children = if let Some(keyframe) = next_keyframe {
            // first keyframe found within first 10 seconds
            // Split into 2 nodes, left with start to keyframe, right with keyframe to end
            (
                Box::new(Node::new::<T>(
                    dec,
                    opts,
                    frame_limit,
                    start,
                    keyframe,
                    start_is_keyframe,
                )),
                Box::new(Node::new::<T>(dec, opts, frame_limit, keyframe, end, true)),
            )
        } else {
            // No keyframe found within limits
            // Split into 2 nodes, left with start to middle, right with middle to end
            (
                Box::new(Node::new::<T>(
                    dec,
                    opts,
                    frame_limit,
                    start,
                    middle_frame_index,
                    start_is_keyframe,
                )),
                Box::new(Node::new::<T>(
                    dec,
                    opts,
                    frame_limit,
                    middle_frame_index,
                    end,
                    false,
                )),
            )
        };
        this.children = Some(children);
        println!("Node[{}-{}]: Done initializing", start, end);

        this
    }

    fn detect_scene_changes<T: Pixel>(
        &mut self,
        dec: &mut Decoder,
        opts: DetectionOptions,
        frame_limit: Option<usize>,
        progress_callback: Option<&dyn Fn(usize, usize)>,
        semaphore: &NodeSemaphore,
    ) -> anyhow::Result<DetectionResults> {
        if let Some((left, right)) = &mut self.children {
            // Parent Node - Merge the results of the children
            let left_result = left.detect_scene_changes::<T>(
                dec,
                opts,
                frame_limit,
                progress_callback,
                semaphore,
            )?;
            let right_result = right.detect_scene_changes::<T>(
                dec,
                opts,
                frame_limit,
                progress_callback,
                semaphore,
            )?;

            self.keyframes.extend(left_result.scene_changes.iter());
            self.scores.extend(left_result.scores.iter());
            self.keyframes.extend(right_result.scene_changes.iter());
            self.scores.extend(right_result.scores.iter());

            self.elapsed_seconds = left.elapsed_seconds + right.elapsed_seconds;
        } else {
            // Leaf Node - Perform Scene Detection
            // Prevent all leaf nodes from running at the same time
            semaphore.acquire();
            println!("Node[{}-{}]: Starting detection", self.start, self.end);

            let start_time = Instant::now();
            let mut detector = new_detector::<T>(dec, opts).unwrap();
            let mut frame_queue = BTreeMap::new();
            let mut current_frame = self.start;
            loop {
                let mut next_input_frameno = frame_queue
                    .keys()
                    .last()
                    .copied()
                    .map_or(current_frame, |key| key + 1);

                while next_input_frameno
                    < (current_frame + opts.lookahead_distance + 1)
                        .min(frame_limit.unwrap_or(self.end))
                {
                    let frame = dec.seek_video_frame(next_input_frameno);
                    if let Ok(frame) = frame {
                        frame_queue.insert(next_input_frameno, Arc::new(frame));
                        next_input_frameno += 1;
                    } else {
                        // End of input
                        break;
                    }
                }

                // frame_queue should start whatever the previous frame was
                let frame_set = frame_queue
                    .values()
                    .take(opts.lookahead_distance + 2)
                    .collect::<Vec<_>>();
                if frame_set.len() < 2 {
                    // End of video
                    break;
                }
                if current_frame == self.start && self.start_is_keyframe {
                    // No need to recalculate the first keyframe
                    // if already confirmed during initialization
                    self.keyframes.insert(self.start);
                } else {
                    let (cut, score) = detector.analyze_next_frame(
                        &frame_set,
                        current_frame,
                        *self.keyframes.iter().last().unwrap_or(&self.start),
                    );
                    if let Some(score) = score {
                        self.scores.insert(current_frame, score);
                    }
                    if cut {
                        self.keyframes.insert(current_frame);
                    }
                }

                if current_frame > 0 {
                    frame_queue.remove(&(current_frame - 1));
                }

                current_frame += 1;

                if let Some(progress_callback) = progress_callback {
                    progress_callback(current_frame, self.keyframes.len());
                }
                if let Some(frame_limit) = frame_limit {
                    if current_frame == frame_limit {
                        break;
                    }
                }
            }

            self.elapsed_seconds = start_time.elapsed().as_secs_f64();
            println!("Node[{}-{}]: Finished detection", self.start, self.end);
            semaphore.release();
        }

        Ok(DetectionResults {
            scene_changes: self.keyframes.clone().into_iter().collect(),
            frame_count: self.end - self.start,
            speed: (self.end - self.start) as f64 / self.elapsed_seconds,
            scores: self.scores.clone(),
        })
    }
}

struct NodeSemaphore {
    count: Mutex<usize>,
    maximum_threads: usize,
    condvar: Condvar,
}

impl NodeSemaphore {
    fn new(maximum_threads: usize) -> Self {
        Self {
            count: Mutex::new(0),
            maximum_threads,
            condvar: Condvar::new(),
        }
    }

    fn acquire(&self) {
        let mut count = self.count.lock().unwrap();
        while *count >= self.maximum_threads {
            count = self.condvar.wait(count).unwrap();
        }
        *count += 1;
    }

    fn release(&self) {
        let mut count = self.count.lock().unwrap();
        *count -= 1;
        self.condvar.notify_one();
    }
}
