#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define MAX_CLASSES 10

// CUDA Kernel to compute squared Euclidean distances
__global__ void compute_distances_kernel(float *d_X_test, float *d_X_train, float *d_distances, 
                                        int n_test, int n_train, int dim) {
    int test_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int train_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (test_idx < n_test && train_idx < n_train) {
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            float diff = d_X_test[test_idx * dim + i] - d_X_train[train_idx * dim + i];
            sum += diff * diff;
        }
        d_distances[test_idx * n_train + train_idx] = sqrtf(sum);
    }
}

extern "C" {
    void knn_cuda_predict(float *h_X_test, int n_test, 
                         float *h_X_train, int n_train, 
                         int *h_y_train, int dim, int k, 
                         int *h_predictions) {
        
        float *d_X_test, *d_X_train, *d_distances;
        int *d_y_train;

        // Allocate Device Memory
        cudaMalloc(&d_X_test, n_test * dim * sizeof(float));
        cudaMalloc(&d_X_train, n_train * dim * sizeof(float));
        cudaMalloc(&d_y_train, n_train * sizeof(int));
        cudaMalloc(&d_distances, n_test * n_train * sizeof(float));

        // Copy data to Device
        cudaMemcpy(d_X_test, h_X_test, n_test * dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_X_train, h_X_train, n_train * dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_train, h_y_train, n_train * sizeof(int), cudaMemcpyHostToDevice);

        // Define Grid and Block dimensions
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((n_test + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (n_train + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Launch Kernel
        compute_distances_kernel<<<numBlocks, threadsPerBlock>>>(d_X_test, d_X_train, d_distances, n_test, n_train, dim);
        cudaDeviceSynchronize();

        // The voting part is easier to do on CPU for simplicity, 
        // or we can sort on GPU (more complex). 
        // For this bonus, we'll pull distances back and vote.
        float *h_distances = (float *)malloc(n_test * n_train * sizeof(float));
        cudaMemcpy(h_distances, d_distances, n_test * n_train * sizeof(float), cudaMemcpyDeviceToHost);

        // Simple CPU voting based on GPU-calculated distances
        for (int i = 0; i < n_test; i++) {
            int counts[MAX_CLASSES] = {0};
            
            // This is a naive O(k * n_train) search for small k.
            // In a production kernel, we would use a priority queue or bitonic sort on GPU.
            for (int j = 0; j < k; j++) {
                float min_dist = 1e10f;
                int min_idx = -1;
                for (int m = 0; m < n_train; m++) {
                    float d = h_distances[i * n_train + m];
                    if (d < min_dist) {
                        min_dist = d;
                        min_idx = m;
                    }
                }
                if (min_idx != -1) {
                    counts[h_y_train[min_idx]]++;
                    h_distances[i * n_train + min_idx] = 1e11f; // Mark as used
                }
            }

            int max_vote = 0;
            int pred = 0;
            for (int c = 0; c < MAX_CLASSES; c++) {
                if (counts[c] > max_vote) {
                    max_vote = counts[c];
                    pred = c;
                }
            }
            h_predictions[i] = pred;
        }

        // Cleanup
        free(h_distances);
        cudaFree(d_X_test);
        cudaFree(d_X_train);
        cudaFree(d_y_train);
        cudaFree(d_distances);
    }
}
