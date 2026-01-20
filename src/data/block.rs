use std::{
    fmt::{self, Display},
    num::NonZeroUsize,
};

use thiserror::Error;

use crate::data::{
    plane::PlaneOffset,
    superblock::{MI_SIZE_LOG2, SB_SIZE_LOG2},
};

pub const MAX_TX_SIZE: usize = 64;
pub const BLOCK_TO_PLANE_SHIFT: usize = MI_SIZE_LOG2;
pub const MIB_SIZE_LOG2: usize = SB_SIZE_LOG2 - MI_SIZE_LOG2;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(test, derive(Default))]
#[expect(non_camel_case_types)]
pub enum BlockSize {
    BLOCK_4X4,
    BLOCK_4X8,
    BLOCK_8X4,
    BLOCK_8X8,
    BLOCK_8X16,
    BLOCK_16X8,
    BLOCK_16X16,
    BLOCK_16X32,
    BLOCK_32X16,
    BLOCK_32X32,
    BLOCK_32X64,
    BLOCK_64X32,
    #[cfg_attr(test, default)]
    BLOCK_64X64,
    BLOCK_64X128,
    BLOCK_128X64,
    BLOCK_128X128,
    BLOCK_4X16,
    BLOCK_16X4,
    BLOCK_8X32,
    BLOCK_32X8,
    BLOCK_16X64,
    BLOCK_64X16,
}

impl BlockSize {
    /// # Errors
    ///
    /// - Returns `InvalidBlockSize` if the given `w` and `h` do not produce a
    ///   valid block size.
    pub fn from_width_and_height_opt(w: usize, h: usize) -> Result<BlockSize, InvalidBlockSize> {
        use crate::data::block::BlockSize::{
            BLOCK_4X4,
            BLOCK_4X8,
            BLOCK_4X16,
            BLOCK_8X4,
            BLOCK_8X8,
            BLOCK_8X16,
            BLOCK_8X32,
            BLOCK_16X4,
            BLOCK_16X8,
            BLOCK_16X16,
            BLOCK_16X32,
            BLOCK_16X64,
            BLOCK_32X8,
            BLOCK_32X16,
            BLOCK_32X32,
            BLOCK_32X64,
            BLOCK_64X16,
            BLOCK_64X32,
            BLOCK_64X64,
            BLOCK_64X128,
            BLOCK_128X64,
            BLOCK_128X128,
        };

        match (w, h) {
            (4, 4) => Ok(BLOCK_4X4),
            (4, 8) => Ok(BLOCK_4X8),
            (4, 16) => Ok(BLOCK_4X16),
            (8, 4) => Ok(BLOCK_8X4),
            (8, 8) => Ok(BLOCK_8X8),
            (8, 16) => Ok(BLOCK_8X16),
            (8, 32) => Ok(BLOCK_8X32),
            (16, 4) => Ok(BLOCK_16X4),
            (16, 8) => Ok(BLOCK_16X8),
            (16, 16) => Ok(BLOCK_16X16),
            (16, 32) => Ok(BLOCK_16X32),
            (16, 64) => Ok(BLOCK_16X64),
            (32, 8) => Ok(BLOCK_32X8),
            (32, 16) => Ok(BLOCK_32X16),
            (32, 32) => Ok(BLOCK_32X32),
            (32, 64) => Ok(BLOCK_32X64),
            (64, 16) => Ok(BLOCK_64X16),
            (64, 32) => Ok(BLOCK_64X32),
            (64, 64) => Ok(BLOCK_64X64),
            (64, 128) => Ok(BLOCK_64X128),
            (128, 64) => Ok(BLOCK_128X64),
            (128, 128) => Ok(BLOCK_128X128),
            _ => Err(InvalidBlockSize),
        }
    }

    /// # Panics
    ///
    /// - If the given `w` and `h` do not produce a valid block size.
    #[expect(clippy::unwrap_used)]
    pub fn from_width_and_height(w: usize, h: usize) -> BlockSize {
        Self::from_width_and_height_opt(w, h).unwrap()
    }

    pub const fn width(self) -> NonZeroUsize {
        NonZeroUsize::new(1 << self.width_log2().get()).expect("cannot be zero")
    }

    pub const fn width_log2(self) -> NonZeroUsize {
        use crate::data::block::BlockSize::{
            BLOCK_4X4,
            BLOCK_4X8,
            BLOCK_4X16,
            BLOCK_8X4,
            BLOCK_8X8,
            BLOCK_8X16,
            BLOCK_8X32,
            BLOCK_16X4,
            BLOCK_16X8,
            BLOCK_16X16,
            BLOCK_16X32,
            BLOCK_16X64,
            BLOCK_32X8,
            BLOCK_32X16,
            BLOCK_32X32,
            BLOCK_32X64,
            BLOCK_64X16,
            BLOCK_64X32,
            BLOCK_64X64,
            BLOCK_64X128,
            BLOCK_128X64,
            BLOCK_128X128,
        };

        match self {
            BLOCK_4X4 | BLOCK_4X8 | BLOCK_4X16 => NonZeroUsize::new(2).expect("non-zero const"),
            BLOCK_8X4 | BLOCK_8X8 | BLOCK_8X16 | BLOCK_8X32 => {
                NonZeroUsize::new(3).expect("non-zero const")
            }
            BLOCK_16X4 | BLOCK_16X8 | BLOCK_16X16 | BLOCK_16X32 | BLOCK_16X64 => {
                NonZeroUsize::new(4).expect("non-zero const")
            }
            BLOCK_32X8 | BLOCK_32X16 | BLOCK_32X32 | BLOCK_32X64 => {
                NonZeroUsize::new(5).expect("non-zero const")
            }
            BLOCK_64X16 | BLOCK_64X32 | BLOCK_64X64 | BLOCK_64X128 => {
                NonZeroUsize::new(6).expect("non-zero const")
            }
            BLOCK_128X64 | BLOCK_128X128 => NonZeroUsize::new(7).expect("non-zero const"),
        }
    }

    pub const fn height(self) -> NonZeroUsize {
        NonZeroUsize::new(1 << self.height_log2().get()).expect("cannot be zero")
    }

    pub const fn height_log2(self) -> NonZeroUsize {
        use crate::data::block::BlockSize::{
            BLOCK_4X4,
            BLOCK_4X8,
            BLOCK_4X16,
            BLOCK_8X4,
            BLOCK_8X8,
            BLOCK_8X16,
            BLOCK_8X32,
            BLOCK_16X4,
            BLOCK_16X8,
            BLOCK_16X16,
            BLOCK_16X32,
            BLOCK_16X64,
            BLOCK_32X8,
            BLOCK_32X16,
            BLOCK_32X32,
            BLOCK_32X64,
            BLOCK_64X16,
            BLOCK_64X32,
            BLOCK_64X64,
            BLOCK_64X128,
            BLOCK_128X64,
            BLOCK_128X128,
        };

        match self {
            BLOCK_4X4 | BLOCK_8X4 | BLOCK_16X4 => NonZeroUsize::new(2).expect("non-zero const"),
            BLOCK_4X8 | BLOCK_8X8 | BLOCK_16X8 | BLOCK_32X8 => {
                NonZeroUsize::new(3).expect("non-zero const")
            }
            BLOCK_4X16 | BLOCK_8X16 | BLOCK_16X16 | BLOCK_32X16 | BLOCK_64X16 => {
                NonZeroUsize::new(4).expect("non-zero const")
            }
            BLOCK_8X32 | BLOCK_16X32 | BLOCK_32X32 | BLOCK_64X32 => {
                NonZeroUsize::new(5).expect("non-zero const")
            }
            BLOCK_16X64 | BLOCK_32X64 | BLOCK_64X64 | BLOCK_128X64 => {
                NonZeroUsize::new(6).expect("non-zero const")
            }
            BLOCK_64X128 | BLOCK_128X128 => NonZeroUsize::new(7).expect("non-zero const"),
        }
    }

    pub const fn tx_size(self) -> TxSize {
        use crate::data::block::{
            BlockSize::{
                BLOCK_4X4,
                BLOCK_4X8,
                BLOCK_4X16,
                BLOCK_8X4,
                BLOCK_8X8,
                BLOCK_8X16,
                BLOCK_8X32,
                BLOCK_16X4,
                BLOCK_16X8,
                BLOCK_16X16,
                BLOCK_16X32,
                BLOCK_16X64,
                BLOCK_32X8,
                BLOCK_32X16,
                BLOCK_32X32,
                BLOCK_32X64,
                BLOCK_64X16,
                BLOCK_64X32,
            },
            TxSize::{
                TX_4X4,
                TX_4X8,
                TX_4X16,
                TX_8X4,
                TX_8X8,
                TX_8X16,
                TX_8X32,
                TX_16X4,
                TX_16X8,
                TX_16X16,
                TX_16X32,
                TX_16X64,
                TX_32X8,
                TX_32X16,
                TX_32X32,
                TX_32X64,
                TX_64X16,
                TX_64X32,
                TX_64X64,
            },
        };

        match self {
            BLOCK_4X4 => TX_4X4,
            BLOCK_4X8 => TX_4X8,
            BLOCK_8X4 => TX_8X4,
            BLOCK_8X8 => TX_8X8,
            BLOCK_8X16 => TX_8X16,
            BLOCK_16X8 => TX_16X8,
            BLOCK_16X16 => TX_16X16,
            BLOCK_16X32 => TX_16X32,
            BLOCK_32X16 => TX_32X16,
            BLOCK_32X32 => TX_32X32,
            BLOCK_32X64 => TX_32X64,
            BLOCK_64X32 => TX_64X32,
            BLOCK_4X16 => TX_4X16,
            BLOCK_16X4 => TX_16X4,
            BLOCK_8X32 => TX_8X32,
            BLOCK_32X8 => TX_32X8,
            BLOCK_16X64 => TX_16X64,
            BLOCK_64X16 => TX_64X16,
            _ => TX_64X64,
        }
    }
}

#[derive(Debug, Copy, Clone, Error, Eq, PartialEq)]
pub struct InvalidBlockSize;

impl Display for InvalidBlockSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid block size")
    }
}

/// Transform Size
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
#[expect(non_camel_case_types)]
pub enum TxSize {
    TX_4X4,
    TX_8X8,
    TX_16X16,
    TX_32X32,
    TX_64X64,

    TX_4X8,
    TX_8X4,
    TX_8X16,
    TX_16X8,
    TX_16X32,
    TX_32X16,
    TX_32X64,
    TX_64X32,

    TX_4X16,
    TX_16X4,
    TX_8X32,
    TX_32X8,
    TX_16X64,
    TX_64X16,
}

impl TxSize {
    pub const fn width(self) -> NonZeroUsize {
        NonZeroUsize::new(1 << self.width_log2().get()).expect("cannot be zero")
    }

    pub const fn width_log2(self) -> NonZeroUsize {
        use crate::data::block::TxSize::{
            TX_4X4,
            TX_4X8,
            TX_4X16,
            TX_8X4,
            TX_8X8,
            TX_8X16,
            TX_8X32,
            TX_16X4,
            TX_16X8,
            TX_16X16,
            TX_16X32,
            TX_16X64,
            TX_32X8,
            TX_32X16,
            TX_32X32,
            TX_32X64,
            TX_64X16,
            TX_64X32,
            TX_64X64,
        };
        match self {
            TX_4X4 | TX_4X8 | TX_4X16 => NonZeroUsize::new(2).expect("non-zero const"),
            TX_8X8 | TX_8X4 | TX_8X16 | TX_8X32 => NonZeroUsize::new(3).expect("non-zero const"),
            TX_16X16 | TX_16X8 | TX_16X32 | TX_16X4 | TX_16X64 => {
                NonZeroUsize::new(4).expect("non-zero const")
            }
            TX_32X32 | TX_32X16 | TX_32X64 | TX_32X8 => {
                NonZeroUsize::new(5).expect("non-zero const")
            }
            TX_64X64 | TX_64X32 | TX_64X16 => NonZeroUsize::new(6).expect("non-zero const"),
        }
    }

    pub const fn height(self) -> NonZeroUsize {
        NonZeroUsize::new(1 << self.height_log2().get()).expect("cannot be zero")
    }

    pub const fn height_log2(self) -> NonZeroUsize {
        use crate::data::block::TxSize::{
            TX_4X4,
            TX_4X8,
            TX_4X16,
            TX_8X4,
            TX_8X8,
            TX_8X16,
            TX_8X32,
            TX_16X4,
            TX_16X8,
            TX_16X16,
            TX_16X32,
            TX_16X64,
            TX_32X8,
            TX_32X16,
            TX_32X32,
            TX_32X64,
            TX_64X16,
            TX_64X32,
            TX_64X64,
        };
        match self {
            TX_4X4 | TX_8X4 | TX_16X4 => NonZeroUsize::new(2).expect("non-zero const"),
            TX_8X8 | TX_4X8 | TX_16X8 | TX_32X8 => NonZeroUsize::new(3).expect("non-zero const"),
            TX_16X16 | TX_8X16 | TX_32X16 | TX_4X16 | TX_64X16 => {
                NonZeroUsize::new(4).expect("non-zero const")
            }
            TX_32X32 | TX_16X32 | TX_64X32 | TX_8X32 => {
                NonZeroUsize::new(5).expect("non-zero const")
            }
            TX_64X64 | TX_32X64 | TX_16X64 => NonZeroUsize::new(6).expect("non-zero const"),
        }
    }
}

/// Absolute offset in blocks, where a block is defined
/// to be an `N*N` square where `N == (1 << BLOCK_TO_PLANE_SHIFT)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockOffset {
    pub x: usize,
    pub y: usize,
}

impl BlockOffset {
    /// Convert to plane offset without decimation.
    pub const fn to_luma_plane_offset(self) -> PlaneOffset {
        PlaneOffset {
            x: (self.x as isize) << BLOCK_TO_PLANE_SHIFT,
            y: (self.y as isize) << BLOCK_TO_PLANE_SHIFT,
        }
    }

    pub fn with_offset(self, col_offset: isize, row_offset: isize) -> BlockOffset {
        let x = self.x as isize + col_offset;
        let y = self.y as isize + row_offset;
        debug_assert!(x >= 0);
        debug_assert!(y >= 0);

        BlockOffset {
            x: x as usize,
            y: y as usize,
        }
    }
}
