#include "ppm.h"

int main() {

  // The timer to get duration info
  clock_t start;
  double duration;

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
  for(int y = 0; y < height; y++) {
    for(int x = 0; x < width; x++) {
      for(int j = 0; j < 3; j++) {

        int index = (y*width + x)*3 + j;
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
  }

  duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
  cout << "Time to generate on CPU : " << duration << "s" << endl;

  // Write result in a file

  cout << "Writing file..." << endl;
  ppm_write(OUTPUT_CPU, output, width, height);
  cout << "CPU generated file written!" << endl;

  delete[] input;
  delete[] output;

  return 0;

}
