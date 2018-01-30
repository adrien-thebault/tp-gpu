#include "ppm.h"

__global__ void kernel(int* vector, int width, int height) {

    // We retrieve the index in the vector thanks to the block number
    // and the thread number in this block
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // In case we have launched more threads than needed
    if(i < 3*width*height) {

      int base = vector[i];
      vector[i] = 255;

      while(vector[i] > 255-base) vector[i]--;

    }

}

int main() {

  // The timer to get duration info
  clock_t start;
  double duration;

  // Pointer to the vector of data in the GPU memory
  int* data_pt;

  // Image data
  int width, height;
  ppm_read_metadata(INPUT_FILE, &width, &height);

  int* pixels = (int*) malloc(3*width*height*sizeof(int));

  // Read input file and put image data into the array
  ppm_read(INPUT_FILE, pixels);

  // Start clock
  start = clock();

  cudaMalloc(&data_pt, 3*width*height*sizeof(int)); // Allocate memory in the GPU
  cudaMemcpy(data_pt, pixels, 3*width*height*sizeof(int), cudaMemcpyHostToDevice); // Copy original image data to GPU memory

  int block_size = 512;
  int num_blocks = (3*width*height + block_size - 1) / block_size;
  kernel<<<num_blocks, block_size>>>(data_pt, width, height);

  cudaMemcpy(pixels, data_pt, 3*width*height*sizeof(int), cudaMemcpyDeviceToHost); // Copy result image data to our vector

  cudaFree(data_pt); // Cleaning the GPU memory
  cudaDeviceSynchronize(); // Wait for GPU to finish before accessing on host

  duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
  cout << "Time to generate on GPU : " << duration << "s" << endl;

  /** WRITE FILES */

  cout << "Writing file..." << endl;
  ppm_write(OUTPUT_GPU, pixels, width, height);
  cout << "GPU generated file written!" << endl;

  return 0;

}
