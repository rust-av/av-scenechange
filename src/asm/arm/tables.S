/*
 * Copyright (c) 2019-2022, The rav1e contributors. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

/*
 * Copyright © 2018, VideoLAN and dav1d authors
 * Copyright © 2018, Two Orioles, LLC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "src/asm/arm/asm.S"

const filter_intra_taps, export=1,  align=4
.byte  -6,  10,  -5,   2,  -3,   1,  -3,   1 /* 0 */
.byte  -4,   6,  -3,   2,  -3,   2,  -3,   1
.byte   0,   0,  10,   0,   1,  10,   1,   2
.byte   0,   0,   6,   0,   2,   6,   2,   2
.byte   0,  12,   0,   9,   0,   7,  10,   5
.byte   0,   2,   0,   2,   0,   2,   6,   3
.byte   0,   0,   0,   0,   0,   0,   0,   0
.byte  12,   0,   9,   0,   7,   0,   5,   0
.byte -10,  16,  -6,   0,  -4,   0,  -2,   0 /* 1 */
.byte -10,  16,  -6,   0,  -4,   0,  -2,   0
.byte   0,   0,  16,   0,   0,  16,   0,   0
.byte   0,   0,  16,   0,   0,  16,   0,   0
.byte   0,  10,   0,   6,   0,   4,  16,   2
.byte   0,   0,   0,   0,   0,   0,  16,   0
.byte   0,   0,   0,   0,   0,   0,   0,   0
.byte  10,   0,   6,   0,   4,   0,   2,   0
.byte  -8,   8,  -8,   0,  -8,   0,  -8,   0 /* 2 */
.byte  -4,   4,  -4,   0,  -4,   0,  -4,   0
.byte   0,   0,   8,   0,   0,   8,   0,   0
.byte   0,   0,   4,   0,   0,   4,   0,   0
.byte   0,  16,   0,  16,   0,  16,   8,  16
.byte   0,   0,   0,   0,   0,   0,   4,   0
.byte   0,   0,   0,   0,   0,   0,   0,   0
.byte  16,   0,  16,   0,  16,   0,  16,   0
.byte  -2,   8,  -1,   3,  -1,   2,   0,   1 /* 3 */
.byte  -1,   4,  -1,   3,  -1,   2,  -1,   2
.byte   0,   0,   8,   0,   3,   8,   2,   3
.byte   0,   0,   4,   0,   3,   4,   2,   3
.byte   0,  10,   0,   6,   0,   4,   8,   2
.byte   0,   3,   0,   4,   0,   4,   4,   3
.byte   0,   0,   0,   0,   0,   0,   0,   0
.byte  10,   0,   6,   0,   4,   0,   3,   0
.byte -12,  14, -10,   0,  -9,   0,  -8,   0 /* 4 */
.byte -10,  12,  -9,   1,  -8,   0,  -7,   0
.byte   0,   0,  14,   0,   0,  14,   0,   0
.byte   0,   0,  12,   0,   0,  12,   0,   1
.byte   0,  14,   0,  12,   0,  11,  14,  10
.byte   0,   0,   0,   0,   0,   1,  12,   1
.byte   0,   0,   0,   0,   0,   0,   0,   0
.byte  14,   0,  12,   0,  11,   0,   9,   0
endconst

const sgr_x_by_x, export=1, align=4
.byte 255, 128,  85,  64,  51,  43,  37,  32,  28,  26,  23,  21,  20,  18,  17
.byte  16,  15,  14,  13,  13,  12,  12,  11,  11,  10,  10,   9,   9,   9,   9
.byte   8,   8,   8,   8,   7,   7,   7,   7,   7,   6,   6,   6,   6,   6,   6
.byte   6,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   4,   4,   4,   4
.byte   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   3,   3
.byte   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3
.byte   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   2,   2,   2
.byte   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2
.byte   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2
.byte   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2
.byte   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2
.byte   2,   2,   2,   2,   2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1
.byte   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1
.byte   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1
.byte   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1
.byte   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1
.byte   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1
.byte   0
endconst

const sm_weights, export=1
.byte   0,   0 /* Unused, because we always offset by bs, which is at least 2. */
.byte 255, 128 /* bs = 2 */
.byte 255, 149,  85,  64 /* bs = 4 */
.byte 255, 197, 146, 105,  73,  50,  37,  32 /* bs = 8 */
.byte 255, 225, 196, 170, 145, 123, 102,  84 /* bs = 16 */
.byte  68,  54,  43,  33,  26,  20,  17,  16
.byte 255, 240, 225, 210, 196, 182, 169, 157 /* bs =32 */
.byte 145, 133, 122, 111, 101,  92,  83,  74
.byte  66,  59,  52,  45,  39,  34,  29,  25
.byte  21,  17,  14,  12,  10,   9,   8,   8
.byte 255, 248, 240, 233, 225, 218, 210, 203 /* bs = 64 */
.byte 196, 189, 182, 176, 169, 163, 156, 150
.byte 144, 138, 133, 127, 121, 116, 111, 106
.byte 101,  96,  91,  86,  82,  77,  73,  69
.byte  65,  61,  57,  54,  50,  47,  44,  41
.byte  38,  35,  32,  29,  27,  25,  22,  20
.byte  18,  16,  15,  13,  12,  10,   9,   8
.byte   7,   6,   6,   5,   5,   4,   4,   4
endconst
