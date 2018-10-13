#include <iostream>
#include <cstdlib>

#include <curand.h>
#include <curand_kernel.h>

using namespace std;

#define BLOCK_SIZE 128

__global__ void add(int N, float* A, float* B, float* R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        R[i] = A[i] + B[i];
    }
}

__global__ void entrywise_multiply(int N, float* A, float* B, float* R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        R[i] = A[i] * B[i];
    }
}

// TODO(knielsen): Make this more efficient
__global__ void multiply(int A_rows, int A_cols, float* A,
                         int B_rows, int B_cols, float* B,
                         int N, float* R) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < A_rows && col < B_cols) {
        int scalar = 0;
        for (int k = 0; k < A_cols; k++) {
            scalar += A[row * A_cols + k] * B[k * B_cols + col];
        }
        R[row * B_cols + col] = scalar;
    }
}

// TODO(knielsen): Make this more efficient
__global__ void transpose(int rows, int columns, float* input, float* R) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < columns) {
        R[col * rows + row] = input[row * columns + col];
    }
}

__global__ void add_constant_row(float padding, int N, int columns, float* A, float* R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        R[i] = i >= columns ? A[i - columns] : padding;
    }
}

__global__ void setup_random_state(curandState *state, int N, size_t seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        curand_init(seed, index, 0, &state[index]);
    }
}

__global__ void dropout_elements(curandState *state, float rate, int N, float* A, float* R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState local_state = state[index];
    for (int i = index; i < N; i += stride) {
        if (curand_uniform(&local_state) < rate) {
            R[i] = 0;
        } else {
            R[i] = A[i];
        }
    }
    state[index] = local_state;
}

__global__ void dropout_rows(float rate, int columns, float* A, float* R, size_t seed) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, row, 0, &state);
    float discriminator = curand_uniform(&state);

    for (int i = 0; i < columns; i++) {
        R[row * columns + i] = (discriminator < rate) ? 0 : A[row * columns + i];
    }
}

extern "C" {

    struct MatrixHandle {
        size_t rows;
        size_t columns;
        float* elements;
    };

    int matrix_alloc(size_t rows, size_t columns, float* elements, struct MatrixHandle* handle) {
        const size_t N = rows * columns;

        handle->rows = rows;
        handle->columns = columns;
        auto alloc_res = cudaMallocManaged(&handle->elements, N * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        for (size_t i = 0; i < N; i++) {
            handle->elements[i] = elements[i];
        }

        return 0;
    }

    void matrix_free(MatrixHandle* handle) {
        cudaFree(handle->elements);
    }

    int matrix_add(MatrixHandle* A, MatrixHandle* B, MatrixHandle* result_handle) {
        if (A->rows != B->rows || A->columns != B->columns) {
            return 20;
        }

        const auto N = A->rows * A->columns;

        result_handle->rows = A->rows;
        result_handle->columns = A->columns;
        auto alloc_res = cudaMallocManaged(&result_handle->elements, N * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        int blockSize = 128;
        int numBlocks = (N + blockSize - 1) / blockSize;
        add<<<numBlocks, blockSize>>>(N, A->elements, B->elements, result_handle->elements);

        cudaDeviceSynchronize();

        return 0;
    }

    int matrix_entrywise_multiply(MatrixHandle* A, MatrixHandle* B, MatrixHandle* result_handle) {
        if (A->rows != B->rows || A->columns != B->columns) {
            return 20;
        }

        const auto N = A->rows * A->columns;

        result_handle->rows = A->rows;
        result_handle->columns = A->columns;
        auto alloc_res = cudaMallocManaged(&result_handle->elements, N * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        int blockSize = 128;
        int numBlocks = (N + blockSize - 1) / blockSize;
        entrywise_multiply<<<numBlocks, blockSize>>>(N, A->elements, B->elements, result_handle->elements);

        cudaDeviceSynchronize();

        return 0;
    }

    int matrix_multiply(MatrixHandle* A, MatrixHandle* B, MatrixHandle* result_handle) {
        if (A->columns != B->rows) {
            return 30;
        }

        const auto N_result = A->rows * B->columns;

        result_handle->rows = A->rows;
        result_handle->columns = B->columns;
        auto alloc_res = cudaMallocManaged(&result_handle->elements, N_result * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        const int block_size = 16;
        dim3 threads_per_block(block_size, block_size);
        dim3 num_blocks((A->rows + block_size - 1) / block_size,
                        (B->columns + block_size - 1) / block_size);
        multiply<<<num_blocks, threads_per_block>>>(A->rows, A->columns, A->elements,
                                                    B->rows, B->columns, B->elements,
                                                    N_result, result_handle->elements);

        cudaDeviceSynchronize();

        return 0;
    }

    int matrix_transpose(MatrixHandle* A, MatrixHandle* result) {
        const auto N = A->rows * A->columns;

        result->rows = A->columns;
        result->columns = A->rows;
        auto alloc_res = cudaMallocManaged(&result->elements, N * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        const int block_size = 16;
        dim3 threads_per_block(block_size, block_size);
        dim3 num_blocks((A->rows + block_size - 1) / block_size,
                        (A->columns + block_size - 1) / block_size);
        transpose<<<num_blocks, threads_per_block>>>(A->rows, A->columns, A->elements, result->elements);

        cudaDeviceSynchronize();

        return 0;
    }

    int matrix_add_constant_row(float padding, MatrixHandle* A, MatrixHandle* result) {
        result->rows = A->rows + 1;
        result->columns = A->columns;
        const auto N = result->rows * result->columns;
        auto alloc_res = cudaMallocManaged(&result->elements, N * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        int blockSize = 128;
        int numBlocks = (N + blockSize - 1) / blockSize;
        add_constant_row<<<numBlocks, blockSize>>>(padding, N, A->columns, A->elements, result->elements);

        cudaDeviceSynchronize();

        return 0;
    }

    int matrix_dropout_elements(float rate, MatrixHandle* A, MatrixHandle* result) {
        result->rows = A->rows;
        result->columns = A->columns;
        const auto N = result->rows * result->columns;
        auto alloc_res = cudaMallocManaged(&result->elements, N * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        curandState* random_state;
        alloc_res = cudaMalloc(&random_state, N * sizeof(curandState));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        srand(time(NULL));

        int blockSize = BLOCK_SIZE;
        int numBlocks = (N + blockSize - 1) / blockSize;
        setup_random_state<<<numBlocks, blockSize>>>(random_state, N, rand());
        dropout_elements<<<numBlocks, blockSize>>>(random_state, rate, N, A->elements, result->elements);

        cudaDeviceSynchronize();
        cudaFree(random_state);

        return 0;
    }

    int matrix_dropout_rows(float rate, MatrixHandle* A, MatrixHandle* result) {
        result->rows = A->rows;
        result->columns = A->columns;
        const auto N = result->rows * result->columns;
        auto alloc_res = cudaMallocManaged(&result->elements, N * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        srand(time(NULL));
        size_t seed = rand();
        dim3 threads_per_block(1, A->columns);
        dim3 num_blocks(A->rows, 1);
        dropout_rows<<<num_blocks, threads_per_block>>>(rate, A->columns, A->elements, result->elements, seed);

        cudaDeviceSynchronize();

        return 0;
    }
    
}
