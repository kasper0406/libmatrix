#include <iostream>

using namespace std;

__global__ void add(int N, float* A, float* B, float* R) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        R[i] = A[i] + B[i];
    }
}

extern "C" {

    struct MatrixHandle {
        size_t rows;
        size_t columns;
        float* elements;
    };

    int matrix_alloc(size_t rows, size_t columns, float* elements, struct MatrixHandle* handle) {
        cerr << "Allocating matrix " << rows << "x" << columns << endl;
        cerr << "Address of matrix handle: " << handle << endl;

        const size_t N = rows * columns;

        handle->rows = rows;
        handle->columns = columns;

        cerr << "Allocating space on GPU" << endl;

        auto alloc_res = cudaMallocManaged(&handle->elements, N * sizeof(float));
        if (alloc_res != cudaSuccess) {
            return 10;
        }

        cerr << "Writing elements to GPU" << endl;

        for (size_t i = 0; i < N; i++) {
            handle->elements[i] = elements[i];
        }

        cerr << "Finished writing elements" << endl;

        return 0;
    }

    void matrix_free(MatrixHandle* handle) {
        cerr << "Freeing matrix" << endl;

        cudaFree(handle->elements);
    }

    int matrix_add(MatrixHandle* A, MatrixHandle* B, MatrixHandle* result_handle) {
        cerr << "Adding matrices" << endl;

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
}
