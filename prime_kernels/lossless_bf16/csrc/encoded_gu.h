#pragma once

#include <cstdint>

struct EncodedGULayer {
  const uint8_t* plane;
  const uint32_t* headers;
  const uint16_t* raw;
  const uint32_t* row_offsets;
  const uint32_t* enabled;
};

static_assert(sizeof(EncodedGULayer) == 5 * sizeof(void*));
