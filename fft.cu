#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string>
#include <cuda.h>

#define NUM_THREADS 256

//perform a fourier transform on a buffer
__global__ void fourier(int *buffer, int *result, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        float real = 0;
        float imag = 0;
        for(int j = 0; j < size; j++){
            float angle = 2 * M_PI * i * j / size;
            real += buffer[j] * cos(angle);
            imag -= buffer[j] * sin(angle);
        }
        result[i] = sqrt(real * real + imag * imag);
    }
}

void wav_to_ascii_file(std::string wav_file, std::string ascii_file){
    //convert audio.wav to a raw file
    FILE *ip = popen(("sox " + wav_file + " -t raw -r 44100 -e float -b 32 -c 1 " + ascii_file).c_str(), "r");
    pclose(ip);
}

void ascii_file_to_wav(std::string ascii_file, std::string wav_file){
    //write the buffer to a wav file
    FILE *dp = popen(("sox -t raw -r 44100 -e float -b 32 -c 1 " + ascii_file + " " + wav_file).c_str(), "w");
    pclose(dp);
}

int* ascii_file_to_buffer(std::string ascii_file){
    //read the raw file into a buffer
    FILE *dp = fopen(ascii_file.c_str(), "r");
    fseek(dp, 0, SEEK_END);
    
    int size = ftell(dp);
    fseek(dp, 0, SEEK_SET);

    int *dbuffer = (int*)malloc(size);
    fread(dbuffer, 1, size, dp);
    fclose(dp);
    return dbuffer;
}

void buffer_to_ascii_file(int *buffer, int size, std::string ascii_file){
    //write the buffer to an ascii file
    FILE *dp = fopen(ascii_file.c_str(), "w");
    fwrite(buffer, 1, size, dp);
    fclose(dp);
}

int main(){
    //convert audio.wav to a raw file
    wav_to_ascii_file("audio.wav", "audio.ascii");

    //read the raw file into a buffer
    int *dbuffer = ascii_file_to_buffer("audio.ascii");

    //perform a fourier transform on the buffer
    int *dresult;
    cudaMalloc(&dresult, 44100 * 25 * sizeof(int));
    //write the buffer to an ascii file
    buffer_to_ascii_file(dresult, 44100, "audio_result.ascii");

    //write the buffer to a wav file
    ascii_file_to_wav("audio_result.ascii", "audio_result.wav");
    return 0;
}