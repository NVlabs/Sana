#pragma once

#define DIM_SWITCH(VAR_NAME, CONST_NAME, ...)    \
    if (VAR_NAME == 4096) {                         \
      constexpr static int CONST_NAME = 4096;       \
      __VA_ARGS__                                 \
    } else if (VAR_NAME == 2048){                  \
        constexpr static int CONST_NAME = 2048;    \
        __VA_ARGS__                                 \
    } else if (VAR_NAME == 8192){                   \
        constexpr static int CONST_NAME = 8192;    \
        __VA_ARGS__                                \
    } else if(VAR_NAME == 16384) {                  \
        constexpr static int CONST_NAME = 16384;     \
        __VA_ARGS__                                  \
    } else {                                         \
        TORCH_CHECK(false, "Unsupported DIM_SWITCH value: ", VAR_NAME); \
    }

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      __VA_ARGS__                               \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      __VA_ARGS__                               \
    }                                           \
//K/128
#define BLOCK_K_SWITCH(COSNT_NAME, ...)             \
    if (K == 2048) {                          \
      constexpr static int COSNT_NAME = 16;   \
      __VA_ARGS__                             \
    }                                         \
    else if (K == 4096) {                     \
      constexpr static int COSNT_NAME = 32;    \
      __VA_ARGS__                               \
    } else if (K == 8192) {                      \
      constexpr static int COSNT_NAME = 64;   \
      __VA_ARGS__                               \
    } else if (K == 16384) {                      \
      constexpr static int COSNT_NAME = 128;    \
      __VA_ARGS__                               \
    } else {                                    \
      TORCH_CHECK(false, "Unsupported K value: ", K);   \
    }

#define M_SWITCH(...) \
    if (M <= 1024) {                             \
        constexpr static int BM = 128;    \
        constexpr static int BN = 128;    \
        constexpr static int WARP_ROW = 2;    \
        constexpr static int WARP_COL = 2;    \
      __VA_ARGS__                               \
    } else {                                    \
        constexpr static int BM = 128;    \
        constexpr static int BN = 256;    \
        constexpr static int WARP_ROW = 2;    \
        constexpr static int WARP_COL = 4;    \
        __VA_ARGS__                           \
    }                                          