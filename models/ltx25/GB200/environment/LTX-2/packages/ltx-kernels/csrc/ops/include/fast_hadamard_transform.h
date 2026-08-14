/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct HadamardParamsBase {
    using index_t = int64_t;

    int batch, dim, log_N;

    index_t x_batch_stride;
    index_t out_batch_stride;

    float scale;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ out_ptr;
};

struct UnifiedHadamardParamsBase{
    using index_t = int64_t;

    int batch, dim, log_N;
    int batch_fma_change;

    index_t x_batch_stride;
    index_t out_batch_stride;
    index_t fma_batch_stride;
    index_t cos_freq_batch_stride;
    index_t sin_freq_batch_stride;
    
    float scale;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ out_scales_ptr;

    void *__restrict__ y_scale_ptr;
    void *__restrict__ z_shift_ptr;

    void *__restrict__ weights_ptr;

    void *__restrict__ cos_freq_ptr;
    void *__restrict__ sin_freq_ptr;

};


struct DequantHadamardParamsBase {
    using index_t = int64_t;

    int batch, dim, log_N;

    index_t x_batch_stride;
    index_t out_batch_stride;

    float scale;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ scales_ptr;
    void *__restrict__ out_ptr;
};


struct QuantHadamardParamsBase {
    using index_t = int64_t;

    int batch, dim, log_N;

    index_t x_batch_stride;
    index_t out_batch_stride;

    float scale;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ out_scales_ptr;
};

struct NormFMAHadamardParamsBase {
    using index_t = int64_t;

    int batch, dim, log_N;
    int seqlen;

    index_t x_batch_stride;
    index_t out_batch_stride;
    index_t fma_batch_stride;

    float scale;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ y_scale_ptr;
    void *__restrict__ z_shift_ptr;
    void *__restrict__ weights_ptr;  
};


struct NormRopeHadamardParamsBase {
    using index_t = int64_t;

    int batch, dim, log_N;
    
    index_t x_batch_stride;
    index_t out_batch_stride;
    index_t cos_freq_batch_stride;
    index_t sin_freq_batch_stride;
    
    float scale;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ cos_freq_ptr;
    void *__restrict__ sin_freq_ptr;
    void *__restrict__ weights_ptr;  
};


struct NormHadamardParamsBase {
    using index_t = int64_t;

    int batch, dim, log_N;
    
    index_t x_batch_stride;
    index_t out_batch_stride;
    
    float scale;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ weights_ptr;  
};


struct RopeHadamardParamsBase {
    using index_t = int64_t;

    int batch, dim, log_N;
    
    index_t x_batch_stride;
    index_t out_batch_stride;
    index_t cos_freq_batch_stride;
    index_t sin_freq_batch_stride;
    
    float scale;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ cos_freq_ptr;
    void *__restrict__ sin_freq_ptr;  
};
