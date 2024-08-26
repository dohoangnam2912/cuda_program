#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 16
#define EPOCHS 5
#define LEARNING_RATE 0.005

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

typedef struct {
    float *weights1;
    float *weights2;
    float *bias1;
    float *bias2;
} NeuralNetwork;

void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

__global__ void initialize_weights(float *weights, int size, float scale, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        weights[idx] = curand_uniform(&state) * scale - (scale / 2.0f);
    }
}

__global__ void initialize_bias(float *bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        bias[idx] = 0.0f;
    }
}

__device__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

__global__ void forward_pass(float *input, float *weights1, float *bias1, float *hidden,
                             float *weights2, float *bias2, float *output,
                             int input_size, int hidden_size, int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (idx < hidden_size && batch_idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += weights1[idx * input_size + i] * input[batch_idx * input_size + i];
        }
        hidden[batch_idx * hidden_size + idx] = relu(sum + bias1[idx]);
    }

    __syncthreads();

    if (idx < output_size && batch_idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            sum += weights2[idx * hidden_size + i] * hidden[batch_idx * hidden_size + i];
        }
        output[batch_idx * output_size + idx] = sum + bias2[idx];
    }
}

__global__ void softmax(float *output, int output_size, int batch_size) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        float max_val = output[batch_idx * output_size];
        for (int i = 1; i < output_size; i++) {
            max_val = fmaxf(max_val, output[batch_idx * output_size + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < output_size; i++) {
            output[batch_idx * output_size + i] = expf(output[batch_idx * output_size + i] - max_val);
            sum += output[batch_idx * output_size + i];
        }

        for (int i = 0; i < output_size; i++) {
            output[batch_idx * output_size + i] /= sum;
        }
    }
}

__global__ void backward_pass(float *input, float *hidden, float *output, int *labels,
                              float *weights1, float *weights2,
                              float *grad_weights1, float *grad_weights2,
                              float *grad_bias1, float *grad_bias2,
                              int input_size, int hidden_size, int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    __shared__ float grad_output[OUTPUT_SIZE];

    if (idx < output_size && batch_idx < batch_size) {
        grad_output[idx] = output[batch_idx * output_size + idx];
        if (idx == labels[batch_idx]) {
            grad_output[idx] -= 1.0f;
        }
    }

    __syncthreads();

    if (idx < hidden_size && batch_idx < batch_size) {
        float grad_hidden = 0.0f;
        for (int i = 0; i < output_size; i++) {
            grad_hidden += grad_output[i] * weights2[i * hidden_size + idx];
        }
        grad_hidden *= relu_derivative(hidden[batch_idx * hidden_size + idx]);

        for (int i = 0; i < input_size; i++) {
            atomicAdd(&grad_weights1[idx * input_size + i], grad_hidden * input[batch_idx * input_size + i]);
        }
        atomicAdd(&grad_bias1[idx], grad_hidden);
    }

    if (idx < output_size * hidden_size && batch_idx < batch_size) {
        int i = idx / hidden_size;
        int j = idx % hidden_size;
        atomicAdd(&grad_weights2[idx], grad_output[i] * hidden[batch_idx * hidden_size + j]);
    }

    if (idx < output_size && batch_idx < batch_size) {
        atomicAdd(&grad_bias2[idx], grad_output[idx]);
    }
}

__global__ void update_weights(float *weights, float *grad_weights, float *bias, float *grad_bias,
                               int size, float learning_rate, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * grad_weights[idx] / batch_size;
        grad_weights[idx] = 0.0f;
    }
    if (idx < size / 10) {  // Assuming bias size is 1/10 of weights size
        bias[idx] -= learning_rate * grad_bias[idx] / batch_size;
        grad_bias[idx] = 0.0f;
    }
}

void train(NeuralNetwork *nn, float *X_train, int *y_train) {
    float *d_weights1, *d_weights2, *d_bias1, *d_bias2;
    float *d_input, *d_hidden, *d_output;
    float *d_grad_weights1, *d_grad_weights2, *d_grad_bias1, *d_grad_bias2;
    int *d_labels;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, BATCH_SIZE * sizeof(int)));

    // Copy weights and biases to device
    CUDA_CHECK(cudaMemcpy(d_weights1, nn->weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights2, nn->weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias1, nn->bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias2, nn->bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    int block_size = 256;
    dim3 grid_size((max(HIDDEN_SIZE, OUTPUT_SIZE) + block_size - 1) / block_size, BATCH_SIZE);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;

        for (int batch = 0; batch < TRAIN_SIZE / BATCH_SIZE; batch++) {
            // Copy input and labels to device
            CUDA_CHECK(cudaMemcpy(d_input, &X_train[batch * BATCH_SIZE * INPUT_SIZE], BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_labels, &y_train[batch * BATCH_SIZE], BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));

            // Forward pass
            forward_pass<<<grid_size, block_size>>>(d_input, d_weights1, d_bias1, d_hidden, d_weights2, d_bias2, d_output, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);
            softmax<<<BATCH_SIZE, 1>>>(d_output, OUTPUT_SIZE, BATCH_SIZE);

            // Backward pass
            backward_pass<<<grid_size, block_size>>>(d_input, d_hidden, d_output, d_labels, d_weights1, d_weights2, d_grad_weights1, d_grad_weights2, d_grad_bias1, d_grad_bias2, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);

            // Update weights
            update_weights<<<(HIDDEN_SIZE * INPUT_SIZE + block_size - 1) / block_size, block_size>>>(d_weights1, d_grad_weights1, d_bias1, d_grad_bias1, HIDDEN_SIZE * INPUT_SIZE, LEARNING_RATE, BATCH_SIZE);
            update_weights<<<(OUTPUT_SIZE * HIDDEN_SIZE + block_size - 1) / block_size, block_size>>>(d_weights2, d_grad_weights2, d_bias2, d_grad_bias2, OUTPUT_SIZE * HIDDEN_SIZE, LEARNING_RATE, BATCH_SIZE);

            // Compute loss and accuracy (on CPU for simplicity)
            float output[BATCH_SIZE * OUTPUT_SIZE];
            CUDA_CHECK(cudaMemcpy(output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            for (int i = 0; i < BATCH_SIZE; i++) {
                float loss = -logf(fmaxf(output[i * OUTPUT_SIZE + y_train[batch * BATCH_SIZE + i]], 1e-7f));
                total_loss += loss;

                int predicted = 0;
                for (int j = 1; j < OUTPUT_SIZE; j++) {
                    if (output[i * OUTPUT_SIZE + j] > output[i * OUTPUT_SIZE + predicted]) {
                        predicted = j;
                    }
                }
                if (predicted == y_train[batch * BATCH_SIZE + i]) {
                    correct++;
                }
            }
        }

        printf("Epoch %d/%d, Loss: %.4f, Accuracy: %.2f%%\n", 
               epoch + 1, EPOCHS, total_loss / TRAIN_SIZE, 100.0f * correct / TRAIN_SIZE);
    }

    // Copy weights and biases back to host
    CUDA_CHECK(cudaMemcpy(nn->weights1, d_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nn->weights2, d_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nn->bias1, d_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nn->bias2, d_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_weights1);
    cudaFree(d_weights2);
    cudaFree(d_bias1);
    cudaFree(d_bias2);
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_grad_weights1);
    cudaFree(d_grad_weights2);
    cudaFree(d_grad_bias1);
    cudaFree(d_grad_bias2);
    cudaFree(d_labels);
}

int main() {
    srand(time(NULL));

    NeuralNetwork nn;
    nn.weights1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    nn.weights2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn.bias1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    nn.bias2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    float *d_weights1, *d_weights2, *d_bias1, *d_bias2;
    CUDA_CHECK(cudaMalloc(&d_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias2, OUTPUT_SIZE * sizeof(float)));

    int block_size = 256;
    int grid_size;

    unsigned long long seed = time(NULL);
    
    grid_size = (HIDDEN_SIZE * INPUT_SIZE + block_size - 1) / block_size;
    initialize_weights<<<grid_size, block_size>>>(d_weights1, HIDDEN_SIZE * INPUT_SIZE, sqrtf(2.0f / INPUT_SIZE), seed);
    
    grid_size = (OUTPUT_SIZE * HIDDEN_SIZE + block_size - 1) / block_size;
    initialize_weights<<<grid_size, block_size>>>(d_weights2, OUTPUT_SIZE * HIDDEN_SIZE, sqrtf(2.0f / HIDDEN_SIZE), seed);

    grid_size = (HIDDEN_SIZE + block_size - 1) / block_size;
    initialize_bias<<<grid_size, block_size>>>(d_bias1, HIDDEN_SIZE);
    
    grid_size = (OUTPUT_SIZE + block_size - 1) / block_size;
    initialize_bias<<<grid_size, block_size>>>(d_bias2, OUTPUT_SIZE);

    CUDA_CHECK(cudaMemcpy(nn.weights1, d_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nn.weights2, d_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nn.bias1, d_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nn.bias2, d_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_weights1);
    cudaFree(d_weights2);
    cudaFree(d_bias1);
    cudaFree(d_bias2);

    float *X_train = (float*)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = (int*)malloc(TRAIN_SIZE * sizeof(int));

    load_data("../../mnist_data/X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE);
    load_labels("../../mnist_data/y_train.bin", y_train, TRAIN_SIZE);

    train(&nn, X_train, y_train);

    free(nn.weights1);
    free(nn.weights2);
    free(nn.bias1);
    free(nn.bias2);
    free(X_train);
    free(y_train);

    return 0;
}