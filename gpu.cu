#include "ppm.h"

__global__ void kernel(int* input, int* output, int width, int height, int size) {

    // We retrieve the index in the vector thanks to the block number
    // and the thread number in this block
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // In case we have launched more threads than needed
    if(index < size) {

      int j = index % 3;
      int x = (index/3) % width;
      int y = (index/3) / width;

      int sum = input[index], nb = 1;

      for(int y2 = y - BLUR_RADIUS; y2 <= y + BLUR_RADIUS; y2++) {
        for (int x2 = x - BLUR_RADIUS; x2 <= x + BLUR_RADIUS; x2++) {

          int index2 = (y2*width + x2)*3 + j;
          if(index2 >= 0 && index2 < size){
            sum += input[index2];
            nb++;
          }

        }
      }

      output[index] = sum / nb;

    }

}

int main() {

  // The timer to get duration info
  clock_t start;
  double duration;

  // Pointer to the vector of data in the GPU memory
  int* data_in;
  int* data_out;

  // Image data
  int width, height;
  ppm_read_metadata(INPUT_FILE, &width, &height);

  int size = 3*width*height;
  int* input = (int*) malloc(size*sizeof(int));
  int* output = (int*) malloc(size*sizeof(int));

  // Read input file and put image data into the array
  ppm_read(INPUT_FILE, input);

  // Start clock
  start = clock();

  cudaMalloc(&data_in, size*sizeof(int)); // Allocate memory in the GPU
  cudaMalloc(&data_out, size*sizeof(int)); // Allocate memory in the GPU
  cudaMemcpy(data_in, input, size*sizeof(int), cudaMemcpyHostToDevice); // Copy original image data to GPU memory

  int block_size = 512;
  int num_blocks = (size + block_size - 1) / block_size;
  kernel<<<num_blocks, block_size>>>(data_in, data_out, width, height, size);

  cudaMemcpy(output, data_out, size*sizeof(int), cudaMemcpyDeviceToHost); // Copy result image data to our vector

  cudaFree(data_in); // Cleaning the GPU memory
  cudaFree(data_out); // Cleaning the GPU memory
  cudaDeviceSynchronize(); // Wait for GPU to finish before accessing on host

  duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
  cout << "Time to generate on GPU : " << duration << "s" << endl;

  /** WRITE FILES */

  cout << "Writing file..." << endl;
  ppm_write(OUTPUT_GPU, output, width, height);
  cout << "GPU generated file written!" << endl;

  return 0;

}
