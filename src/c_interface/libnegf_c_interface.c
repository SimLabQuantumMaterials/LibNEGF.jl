#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <complex.h>
#include "libnegf_c_interface.h"

// -------------------------------------------------------------------
// // to compile this code linking to the shared library
// gcc main.c -o my_app -L. -lLibNEGFCInterface -Wl,-rpath,.
// -------------------------------------------------------------------

// Helper function to generate a random double between 0.0 and 1.0
double rand_double() {
    return (double)rand() / (double)RAND_MAX;
}

int main() {
    // Seed the random number generator
    srand(time(NULL));

    printf("--- Generating Block-Tridiagonal Complex CSC Matrix ---\n");

    // 1. Matrix parameters
    int32_t n_blocks = 10;
    int32_t block_size = 64;
    int32_t m = n_blocks * block_size; // 640
    int32_t n = n_blocks * block_size; // 640
    
    // Calculate exact number of non-zeros (nnz)
    // A block tridiagonal matrix has:
    // - 1 main diagonal of blocks (N blocks)
    // - 1 super-diagonal of blocks (N-1 blocks)
    // - 1 sub-diagonal of blocks (N-1 blocks)
    // Total blocks = 3*N - 2. Elements per block = b^2.
    int32_t total_blocks = 3 * n_blocks - 2;
    int32_t nnz = total_blocks * (block_size * block_size); // 114,688

    printf("Matrix Dimensions: %d x %d\n", m, n);
    printf("Total Non-Zero Elements: %d\n", nnz);

    // 2. Allocate memory for CSC arrays
    int32_t* colptr_in = (int32_t*)malloc((n + 1) * sizeof(int32_t));
    int32_t* rowval_in = (int32_t*)malloc(nnz * sizeof(int32_t));
    double complex* nzval_in = (double complex*)malloc(nnz * sizeof(double complex));

    if (!colptr_in || !rowval_in || !nzval_in) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // 3. Populate the CSC arrays
    int32_t nz_count = 0;
    
    for (int32_t j = 0; j < n; j++) {
        colptr_in[j] = nz_count;
        
        int32_t block_col = j / block_size; // Which block column are we in?

        // Determine which block rows contain data for this block column
        int32_t block_row_start = (block_col > 0) ? block_col - 1 : 0;
        int32_t block_row_end = (block_col < n_blocks - 1) ? block_col + 1 : n_blocks - 1;

        // Iterate through the valid block rows
        for (int32_t K = block_row_start; K <= block_row_end; K++) {
            // Iterate through the actual rows within this block
            for (int32_t r = 0; r < block_size; r++) {
                int32_t i = K * block_size + r; // Actual row index (0-based)
                
                rowval_in[nz_count] = i;
                
                // Assign a random complex value (a + b*I)
                nzval_in[nz_count] = rand_double() + rand_double() * I;
                
                nz_count++;
            }
        }
    }
    colptr_in[n] = nz_count; // Cap off the colptr array

    // 4. Construct Block Sizes Vector
    int32_t* block_sizes_in = (int32_t*)malloc(n_blocks * sizeof(int32_t));
    for (int i = 0; i < n_blocks; i++) {
        block_sizes_in[i] = block_size; // Populate [64, 64, ..., 64]
    }

    // 5. Construct the Dictionary inputs
    int32_t n_dict = 2;
    const char* dict_keys_in[] = {"max_iter", "tolerance"};
    int32_t dict_vals_in[] = {1000, 1};

    // 6. Boolean 
    int32_t is_hermitian = 1;

    printf("\n--- Calling Julia Library ---\n\n");

    // 7. Call the Julia function
    CCscMatrix result = run_bndiag_wrapper(
        m, n, nnz, colptr_in, rowval_in, nzval_in,
        n_blocks, block_sizes_in,
        n_dict, dict_keys_in, dict_vals_in,
        is_hermitian
    );

    printf("\n--- Back in C: Reading Results ---\n");
    printf("Returned Matrix Size: %d x %d\n", result.m, result.n);
    printf("Returned Non-zeros: %d\n", result.nnz);

    // Print just the first few values to verify
    if (result.nnz > 0) {
        printf("First 3 Complex Values Returned:\n");
        for (int i = 0; i < 3 && i < result.nnz; i++) {
            printf("  [%d]: %f + %fi\n", i, creal(result.nzval[i]), cimag(result.nzval[i]));
        }
    }

    // 8. CRITICAL: Cleanup Memory
    // Free the arrays we dynamically allocated for the inputs
    free(colptr_in);
    free(rowval_in);
    free(nzval_in);
    free(block_sizes_in);

    // Free the arrays Julia allocated and passed back to us
    free(result.colptr);
    free(result.rowval);
    free(result.nzval);

    printf("\nMemory freed successfully. Exiting.\n");
    return 0;
}