#include "ppm.h"

int main() {

  // The timer to get duration info
  clock_t start;
  double duration;

  // Image data
  int width, height;
  ppm_read_metadata(INPUT_FILE, &width, &height);

  int* pixels = (int*) malloc(3*width*height*sizeof(int));

  // Read input file and put image data into the array
  ppm_read(INPUT_FILE, pixels);

  // Start clock
  start = clock();
  for(int i = 0; i < height*width*3; i++) {

    int base = pixels[i];
    pixels[i] = 255;

    while(pixels[i] > 255-base) pixels[i]--;

  }

  duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
  cout << "Time to generate on CPU : " << duration << "s" << endl;

  // Write result in a file

  cout << "Writing file..." << endl;
  ppm_write(OUTPUT_CPU, pixels, width, height);
  cout << "CPU generated file written!" << endl;

  return 0;

}
