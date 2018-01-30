#include <iostream>
#include <ctime>
#include <fstream>
#include <stdlib.h>

#define INPUT_FILE "input.ppm"
#define OUTPUT_CPU "output_cpu.ppm"
#define OUTPUT_GPU "output_gpu.ppm"
#define BLUR_RADIUS 10

using namespace std;

/** write pixels as ppm file */
void ppm_write(const char* filename, int pixels[], int width, int height) {

  ofstream output(filename, ios::binary);

  output << "P3" << endl;
  output << width << " " << height << endl;
  output << 255 << endl;

  int index = 0;
  for(int i = 0; i < height; i++) {

    for(int j = 0; j < width; j++) {
      output << pixels[index] << " " << pixels[index+1] << " " << pixels[index+2] << " ";
      index += 3;
    }

    output << endl;

  }

}

void ppm_read_metadata(const char* filename, int* width, int* height) {

  ifstream input(filename, ios::binary);

  string dump;

  input >> dump;
  input >> *width;
  input >> *height;

  input.close();

}

void ppm_read(const char* filename, int* pixels) {

  ifstream input(filename, ios::binary);

  string dump;
  int i_dump;

  input >> dump;
  input >> i_dump;
  input >> i_dump;
  input >> i_dump;

  int value; int i = 0;
  while(input >> value)
    pixels[i++] = value;

  input.close();

}
