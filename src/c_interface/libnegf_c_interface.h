#ifndef MY_WRAPPER_H
#define MY_WRAPPER_H

#include <stdint.h>
#include <complex.h> // Required for double complex

#ifdef __cplusplus
extern "C" {
#endif

// 1. Memory mirror of the Julia CCscMatrix struct
typedef struct {
    int32_t m;
    int32_t n;
    int32_t nnz;
    int32_t* colptr;
    int32_t* rowval;
    double complex* nzval; // Changed to double complex
} CCscMatrix;

// 2. The function signature
CCscMatrix run_bndiag_wrapper(
    int32_t m, 
    int32_t n, 
    int32_t nnz_in,
    int32_t* colptr_in, 
    int32_t* rowval_in, 
    double complex* nzval_in, // Changed to double complex
    
    int32_t n_blocks, 
    int32_t* block_sizes_in,
    
    int32_t n_dict, 
    const char** dict_keys_in, 
    int32_t* dict_vals_in,
    
    int32_t is_hermitian
);

#ifdef __cplusplus
}
#endif

#endif // MY_WRAPPER_H