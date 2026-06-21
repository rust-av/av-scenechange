#[derive(Copy, Clone, Debug)]
#[expect(clippy::upper_case_acronyms)]
pub enum PredictionVariant {
    NONE,
    LEFT,
    TOP,
    BOTH,
}

impl PredictionVariant {
    pub const fn new(x: usize, y: usize) -> Self {
        match (x, y) {
            (0, 0) => PredictionVariant::NONE,
            (_, 0) => PredictionVariant::LEFT,
            (0, _) => PredictionVariant::TOP,
            _ => PredictionVariant::BOTH,
        }
    }
}
