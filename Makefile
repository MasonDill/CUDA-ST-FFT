# Load CUDA using the following command
# module load cuda
#
CC = nvcc
CFLAGS = -O3 -gencode arch=compute_80,code=sm_80
NVCCFLAGS = -O3 -gencode arch=compute_80,code=sm_80
LIBS = 

TARGETS = fft

all:	$(TARGETS)

fft: fft.o
	$(CC) -o $@ $(LIBS) fft.o

fft.o: fft.cu
	$(CC) -c $(CFLAGS) fft.cu

clean: 
	rm -f *.o $(TARGETS) *.stdout *.txt