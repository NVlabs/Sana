#pragma once

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      __VA_ARGS__                               \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      __VA_ARGS__                               \
    }                                           
//K/128
#define BLOCK_K_SWITCH(COSNT_NAME, ...)             \
    if (K == 2048) {                          \
      constexpr static int COSNT_NAME = 2048;   \
      __VA_ARGS__                             \
    }                                         \
    else if (K == 4096) {                     \
      constexpr static int COSNT_NAME = 4096;    \
      __VA_ARGS__                               \
    } else if (K == 8192) {                      \
      constexpr static int COSNT_NAME = 8192;   \
      __VA_ARGS__                               \
    } else if (K == 16384) {                      \
      constexpr static int COSNT_NAME = 16384;    \
      __VA_ARGS__                               \
    } else {                                    \
      TORCH_CHECK(false, "Unsupported K value: ", K);   \
    }

#define M_SWITCH(...) \
      constexpr static int BM = 64;    \
      constexpr static int BN = 128;    \
      constexpr static int WARP_ROW = 2;    \
      constexpr static int WARP_COL = 4;    \
    __VA_ARGS__                                                          