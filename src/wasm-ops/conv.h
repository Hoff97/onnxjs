// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <stdint.h>
#include <vector>

extern "C" {
void conv_f32(void *);

void conv2D_f32_imp(float *, int32_t *, float *, int32_t *, float *, int32_t *,
                    float *, int32_t *, int32_t, int32_t *, int32_t *, int32_t);

// Helper functions
bool is_a_ge_zero_and_a_lt_b(int32_t a, int32_t b) {
  return static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
}
}
