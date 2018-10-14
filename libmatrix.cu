#include <iostream>
#include <cstdlib>

#include <curand.h>
#include <curand_kernel.h>

using namespace std;

#define BLOCK_SIZE 128

#ifdef DEBUG
#define DEBUG_SYNCHRONIZE() cudaDeviceSynchronize()
#else
#define DEBUG_SYNCHRONIZE()
#endif

__global__ void add(int N, float* A, float* B, float* R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        R[i] = A[i] + B[i];
    }
}

__global__ void add_assign(int N, float* A, float* B) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        A[i] += B[i];
    }
}

__global__ void sub(int N, float* A, float* B, float* R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        R[i] = A[i] - B[i];
    }
}

__global__ void entrywise_multiply(int N, float* A, float* B, float* R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        R[i] = A[i] * B[i];
    }
}

__global__ void scalar_multiply(int N, float scalar, float* A, float* R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        R[i] = scalar * A[i];
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

__global__ void copy(int N, float* A, float* R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        R[i] = A[i];
    }
}

__global__ void apply_sigmoid(int N, float* A, float* R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        float exponential = exp(A[i]);
        R[i] = exponential / (exponential + 1);
    }
}

__global__ void apply_sigmoid_derivative(int N, float* A, float* R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        float exponential = exp(A[i]);
        R[i] = exponential / (1 + exponential * (exponential + 2));
    }
}

__global__ void apply_relu(int N, float* A, float* R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        R[i] = A[i] < 0 ? 0 : A[i];
    }
}

__global__ void apply_relu_derivative(int N, float* A, float* R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        R[i] = A[i] < 0 ? 0 : 1;
    }
}

__global__ void apply_twoplayerscore(int N, float* A, float* R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        float exponential = exp(A[i]);
        R[i] = (exponential - 1) / (exponential + 1);
    }
}

__global__ void apply_twoplayerscore_derivative(int N, float* A, float* R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        float exponential = exp(A[i]);
        R[i] = (2 * exponential) / (1 + exponential * (exponential + 2));
    }
}

extern "C" {

    struct MatrixHandle {
        size_t rows;
        size_t columns;
        float* elements;
    };

    void matrix_synchronize(bool only_current_thread) {
        if (only_current_thread) {
            cudaStreamSynchronize(cudaStreamPerThread);
        } else {
            cudaDeviceSynchronize();
        }
    }

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

    int matrix_add(const MatrixHandle* A, const MatrixHandle* B, MatrixHandle* result_handle) {
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

        DEBUG_SYNCHRONIZE();

        return 0;
    }

    int matrix_add_assign(MatrixHandle* A, const MatrixHandle* B) {
        if (A->rows != B->rows || A->columns != B->columns) {
            return 20;
        }

        const auto N = A->rows * A->columns;
        int blockSize = 128;
        int numBlocks = (N + blockSize - 1) / blockSize;
        add_assign<<<numBlocks, blockSize>>>(N, A->elements, B->elements);

        DEBUG_SYNCHRONIZE();

        return 0;
    }

    int matrix_sub(const MatrixHandle* A, const MatrixHandle* B, MatrixHandle* result_handle) {
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
        sub<<<numBlocks, blockSize>>>(N, A->elements, B->elements, result_handle->elements);

        DEBUG_SYNCHRONIZE();

        return 0;
    }

    int matrix_entrywise_multiply(const MatrixHandle* A, const MatrixHandle* B, MatrixHandle* result_handle) {
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

        DEBUG_SYNCHRONIZE();

        return 0;
    }

    int matrix_scalar_multiply(const MatrixHandle* A, float scalar, MatrixHandle* result_handle) {
        const auto N = A->rows * A->columns;
        result_handle->rows = A->rows;
        result_handle->columns = A->columns;
        auto alloc_res = cudaMallocManaged(&result_handle->elements, N * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        int blockSize = 128;
        int numBlocks = (N + blockSize - 1) / blockSize;
        scalar_multiply<<<numBlocks, blockSize>>>(N, scalar, A->elements, result_handle->elements);

        DEBUG_SYNCHRONIZE();

        return 0;
    }

    int matrix_multiply(const MatrixHandle* A, const MatrixHandle* B, MatrixHandle* result_handle) {
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

        DEBUG_SYNCHRONIZE();

        return 0;
    }

    int matrix_transpose(const MatrixHandle* A, MatrixHandle* result) {
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

        DEBUG_SYNCHRONIZE();

        return 0;
    }

    int matrix_add_constant_row(float padding, const MatrixHandle* A, MatrixHandle* result) {
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

        DEBUG_SYNCHRONIZE();

        return 0;
    }

    int matrix_dropout_elements(float rate, const MatrixHandle* A, MatrixHandle* result) {
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

        DEBUG_SYNCHRONIZE();
        cudaFree(random_state);

        return 0;
    }

    int matrix_dropout_rows(float rate, const MatrixHandle* A, MatrixHandle* result) {
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

        DEBUG_SYNCHRONIZE();

        return 0;
    }
    
    int matrix_copy(const MatrixHandle* A, MatrixHandle* result) {
        result->rows = A->rows;
        result->columns = A->columns;
        const auto N = result->rows * result->columns;
        auto alloc_res = cudaMallocManaged(&result->elements, N * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        copy<<<num_blocks, BLOCK_SIZE>>>(N, A->elements, result->elements);

        DEBUG_SYNCHRONIZE();

        return 0;
    }

    int matrix_apply_sigmoid(const MatrixHandle* A, MatrixHandle* result) {
        result->rows = A->rows;
        result->columns = A->columns;
        const auto N = result->rows * result->columns;
        auto alloc_res = cudaMallocManaged(&result->elements, N * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        apply_sigmoid<<<num_blocks, BLOCK_SIZE>>>(N, A->elements, result->elements);

        DEBUG_SYNCHRONIZE();

        return 0;
    }

    int matrix_apply_sigmoid_derivative(const MatrixHandle* A, MatrixHandle* result) {
        result->rows = A->rows;
        result->columns = A->columns;
        const auto N = result->rows * result->columns;
        auto alloc_res = cudaMallocManaged(&result->elements, N * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        apply_sigmoid_derivative<<<num_blocks, BLOCK_SIZE>>>(N, A->elements, result->elements);

        DEBUG_SYNCHRONIZE();

        return 0;
    }

    int matrix_apply_relu(const MatrixHandle* A, MatrixHandle* result) {
        result->rows = A->rows;
        result->columns = A->columns;
        const auto N = result->rows * result->columns;
        auto alloc_res = cudaMallocManaged(&result->elements, N * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        apply_relu<<<num_blocks, BLOCK_SIZE>>>(N, A->elements, result->elements);

        DEBUG_SYNCHRONIZE();

        return 0;
    }

    int matrix_apply_relu_derivative(const MatrixHandle* A, MatrixHandle* result) {
        result->rows = A->rows;
        result->columns = A->columns;
        const auto N = result->rows * result->columns;
        auto alloc_res = cudaMallocManaged(&result->elements, N * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        apply_relu_derivative<<<num_blocks, BLOCK_SIZE>>>(N, A->elements, result->elements);

        DEBUG_SYNCHRONIZE();

        return 0;
    }

    int matrix_apply_twoplayerscore(const MatrixHandle* A, MatrixHandle* result) {
        result->rows = A->rows;
        result->columns = A->columns;
        const auto N = result->rows * result->columns;
        auto alloc_res = cudaMallocManaged(&result->elements, N * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        apply_twoplayerscore<<<num_blocks, BLOCK_SIZE>>>(N, A->elements, result->elements);

        DEBUG_SYNCHRONIZE();

        return 0;
    }

    int matrix_apply_twoplayerscore_derivative(const MatrixHandle* A, MatrixHandle* result) {
        result->rows = A->rows;
        result->columns = A->columns;
        const auto N = result->rows * result->columns;
        auto alloc_res = cudaMallocManaged(&result->elements, N * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        apply_twoplayerscore_derivative<<<num_blocks, BLOCK_SIZE>>>(N, A->elements, result->elements);

        DEBUG_SYNCHRONIZE();

        return 0;
    }
}
