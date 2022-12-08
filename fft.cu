#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>

#define NUM_THREADS 256

//perform a fourier transform on a buffer
__global__ void fourier(int *buffer, int *result, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        int sum = 0;
        for(int j = 0; j < size; j++){
            sum += buffer[j] * cos(2 * M_PI * i * j / size);
        }
        result[i] = sum;
    }
}

int main(){
    //gliss.ascii is a file containing the ascii representation of a song
    FILE *fp = fopen("gliss.ascii", "r");

    //read the file into a buffer
    fseek(fp, 0, SEEK_END);

    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *buffer = (char*)malloc(size);
    fread(buffer, 1, size, fp);
    fclose(fp);

    //perform a fourier transform on the buffer using CUDA
    int numBlocks = (size + NUM_THREADS - 1) / NUM_THREADS;
    int *d_buffer, *d_result;
    cudaMalloc((void**)&d_buffer, size);
    cudaMalloc((void**)&d_result, size);
    cudaMemcpy(d_buffer, buffer, size, cudaMemcpyHostToDevice);
    
    //perform the fourier transform
    fourier<<<numBlocks, NUM_THREADS>>>(d_buffer, d_result, size);
    cudaMemcpy(buffer, d_result, size, cudaMemcpyDeviceToHost);

    //write the result to a file
    fp = fopen("gliss.fft", "w");
    fwrite(buffer, 1, size, fp);
    fclose(fp);

    //free the memory
    cudaFree(d_buffer);
    cudaFree(d_result);
    free(buffer);
    return 0;
}