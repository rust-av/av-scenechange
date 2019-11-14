mod pixel;
mod y4m;

use self::pixel::*;
use ::y4m::Decoder;
use std::cmp;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

/// Options determining how to run scene change detection.
#[derive(Debug, Clone, Copy)]
pub struct DetectionOptions {
    /// Whether or not to analyze the chroma planes.
    /// Enabling this is slower, but may be more accurate.
    use_chroma: bool,
    /// Enabling this will utilize heuristics to avoid scenecuts
    /// that are too close to each other.
    /// This is generally useful if you want scenecut detection
    /// for use in an encoder.
    /// If you want a raw list of scene changes, you should disable this.
    ignore_flashes: bool,
    /// The minimum distane between two scene changes.
    min_scenecut_distance: Option<usize>,
    /// The maximum distance between two scene changes.
    max_scenecut_distance: Option<usize>,
    /// The distance to look ahead in the video
    /// for scene flash detection.
    ///
    /// Not used if `ignore_flashes` is `true`.
    lookahead_distance: usize,
}

impl Default for DetectionOptions {
    fn default() -> Self {
        DetectionOptions {
            use_chroma: true,
            ignore_flashes: false,
            lookahead_distance: 5,
            min_scenecut_distance: None,
            max_scenecut_distance: None,
        }
    }
}

/// Runs through a y4m video clip,
/// detecting where scene changes occur.
/// This is adjustable based on the `opts` parameters.
///
/// Returns a `Vec` containing the frame numbers where the scene changes occur.
pub fn detect_scene_changes<R: Read, T: Pixel>(
    dec: &mut Decoder<R>,
    opts: DetectionOptions,
) -> Vec<usize> {
    assert!(opts.lookahead_distance >= 1);

    let bit_depth = dec.get_bit_depth() as u8;
    let mut detector = SceneChangeDetector::new(bit_depth, opts);
    let mut frame_queue = BTreeMap::new();
    let mut keyframes = BTreeSet::new();
    let mut frameno = 0;
    loop {
        let mut next_input_frameno = frame_queue.keys().last().copied().unwrap_or(0);
        while next_input_frameno < frameno + opts.lookahead_distance {
            let frame = y4m::read_video_frame::<R, T>(dec);
            if let Ok(frame) = frame {
                frame_queue.insert(next_input_frameno, frame);
                next_input_frameno += 1;
            } else {
                // End of input
                break;
            }
        }

        let frame_set = frame_queue
            .values()
            .skip(frameno)
            .take(opts.lookahead_distance)
            .collect::<Vec<_>>();
        if frame_set.is_empty() {
            // End of video
            break;
        }
        detector.analyze_next_frame(
            if frameno == 0 {
                None
            } else {
                frame_queue.get(&(frameno - 1))
            },
            &frame_set,
            frameno,
            &mut keyframes,
        );
        frameno += 1;
    }
    keyframes.into_iter().collect()
}

type PlaneData<T> = [Vec<T>; 3];

/// Runs keyframe detection on frames from the lookahead queue.
struct SceneChangeDetector {
    /// Minimum average difference between YUV deltas that will trigger a scene change.
    threshold: u8,
    opts: DetectionOptions,
    /// Frames that cannot be marked as keyframes due to the algorithm excluding them.
    /// Storing the frame numbers allows us to avoid looking back more than one frame.
    excluded_frames: BTreeSet<usize>,
}

impl SceneChangeDetector {
    pub fn new(bit_depth: u8, opts: DetectionOptions) -> Self {
        // This implementation is based on a Python implementation at
        // https://pyscenedetect.readthedocs.io/en/latest/reference/detection-methods/.
        // The Python implementation uses HSV values and a threshold of 30. Comparing the
        // YUV values was sufficient in most cases, and avoided a more costly YUV->RGB->HSV
        // conversion, but the deltas needed to be scaled down. The deltas for keyframes
        // in YUV were about 1/3 to 1/2 of what they were in HSV, but non-keyframes were
        // very unlikely to have a delta greater than 3 in YUV, whereas they may reach into
        // the double digits in HSV. Therefore, 12 was chosen as a reasonable default threshold.
        // This may be adjusted later.
        const BASE_THRESHOLD: u8 = 12;
        Self {
            threshold: BASE_THRESHOLD * bit_depth / 8,
            opts,
            excluded_frames: BTreeSet::new(),
        }
    }
    /// Runs keyframe detection on the next frame in the lookahead queue.
    ///
    /// This function requires that a subset of input frames
    /// is passed to it in order, and that `keyframes` is only
    /// updated from this method. `input_frameno` should correspond
    /// to the first frame in `frame_set`.
    ///
    /// This will gracefully handle the first frame in the video as well.
    pub fn analyze_next_frame<T: Pixel>(
        &mut self,
        previous_frame: Option<&PlaneData<T>>,
        frame_set: &[&PlaneData<T>],
        input_frameno: usize,
        keyframes: &mut BTreeSet<usize>,
    ) {
        let frame_set = match previous_frame {
            Some(frame) => [frame]
                .iter()
                .chain(frame_set.iter())
                .cloned()
                .collect::<Vec<_>>(),
            None => {
                // The first frame is always a keyframe.
                keyframes.insert(0);
                return;
            }
        };

        self.exclude_scene_flashes(&frame_set, input_frameno);

        if self.is_key_frame(&frame_set[0], &frame_set[1], input_frameno, keyframes) {
            keyframes.insert(input_frameno);
        }
    }
    /// Determines if `current_frame` should be a keyframe.
    fn is_key_frame<T: Pixel>(
        &self,
        previous_frame: &PlaneData<T>,
        current_frame: &PlaneData<T>,
        current_frameno: usize,
        keyframes: &mut BTreeSet<usize>,
    ) -> bool {
        // Find the distance to the previous keyframe.
        let previous_keyframe = keyframes.iter().last().unwrap();
        let distance = current_frameno - previous_keyframe;

        // Handle minimum and maximum key frame intervals.
        if distance < self.opts.min_scenecut_distance.unwrap_or(0) {
            return false;
        }
        if distance
            >= self
                .opts
                .max_scenecut_distance
                .unwrap_or(usize::max_value())
        {
            return true;
        }

        if self.excluded_frames.contains(&current_frameno) {
            return false;
        }

        self.has_scenecut(previous_frame, current_frame)
    }
    /// Uses lookahead to avoid coding short flashes as scenecuts.
    /// Saves excluded frame numbers in `self.excluded_frames`.
    fn exclude_scene_flashes<T: Pixel>(&mut self, frame_subset: &[&PlaneData<T>], frameno: usize) {
        let lookahead_distance = cmp::min(self.opts.lookahead_distance, frame_subset.len() - 1);

        // Where A and B are scenes: AAAAAABBBAAAAAA
        // If BBB is shorter than lookahead_distance, it is detected as a flash
        // and not considered a scenecut.
        if lookahead_distance > 1 {
            for j in 1..=lookahead_distance {
                if !self.has_scenecut(&frame_subset[0], &frame_subset[j]) {
                    // Any frame in between `0` and `j` cannot be a real scenecut.
                    for i in 0..=j {
                        let frameno = frameno + i - 1;
                        self.excluded_frames.insert(frameno);
                    }
                }
            }
        }

        // Where A-F are scenes: AAAAABBCCDDEEFFFFFF
        // If each of BB ... EE are shorter than `lookahead_distance`, they are
        // detected as flashes and not considered scenecuts.
        // Instead, the first F frame becomes a scenecut.
        // If the video ends before F, no frame becomes a scenecut.
        for i in 1..=lookahead_distance {
            if i < lookahead_distance
                && self.has_scenecut(&frame_subset[i], &frame_subset[lookahead_distance])
            {
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
    fn has_scenecut<T: Pixel>(&self, frame1: &PlaneData<T>, frame2: &PlaneData<T>) -> bool {
        let mut delta = Self::get_plane_sad(&frame1[0], &frame2[0]);

        if self.opts.use_chroma {
            delta += Self::get_plane_sad(&frame1[1], &frame2[1]);
            delta += Self::get_plane_sad(&frame1[2], &frame2[2]);
        }

        delta >= self.threshold as u64 * frame1[0].len() as u64
    }

    #[inline(always)]
    fn get_plane_sad<T: Pixel>(plane1: &[T], plane2: &[T]) -> u64 {
        assert_eq!(plane1.len(), plane2.len());
        plane1
            .iter()
            .zip(plane2.iter())
            .map(|(&p1, &p2)| (i16::cast_from(p1) - i16::cast_from(p2)).abs() as u64)
            .sum::<u64>()
    }
}
