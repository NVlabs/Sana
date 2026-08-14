// Inspired by https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      __VA_ARGS__                               \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      __VA_ARGS__                               \
    }                                           \
