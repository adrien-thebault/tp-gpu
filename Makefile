all: cpu gpu

cpu:
	g++ cpu.cpp -o cpu

gpu:
	nvcc gpu.cu -o gpu
