
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cufft.h>
#include "byteStructure.hpp"

#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>

#define LOWER_LIMIT 40
#define UPPER_LIMIT 300

#define CHUNK_SIZE 2048

using namespace std;
long* readData(long* sampleLength);
long powerOfTwo(long input);
int getIndex(int freq);


int main()
{
	long sampleLength; //Read the length of the sample
	cudaError_t cudaStatus;

	//Load wav file into memory
	long *audio = readData(&sampleLength);
	int numChunks = sampleLength / CHUNK_SIZE;

	//Host memory
	cufftComplex **wavData = new cufftComplex* [numChunks];
	cufftComplex **fftData = new cufftComplex* [numChunks];

	//Creating Complex numbers from our readings
	for (int times = 0; times < numChunks; times++)
	{
		cufftComplex *complex = new cufftComplex[CHUNK_SIZE];
		fftData[times] = new cufftComplex[CHUNK_SIZE];
		for (int i = 0; i < CHUNK_SIZE; i++)
		{
			complex[i].x = audio[times * CHUNK_SIZE + i];
			complex[i].y = 0;
		}
		wavData[times] = complex;
	}

	cufftComplex *d_wavData;
	cufftComplex *d_fftData;

	//The actual transform
	cufftHandle plan;
	cufftPlan1d(&plan, CHUNK_SIZE, CUFFT_C2C, 1);

	//FFT
	for (int i = 0; i < numChunks; i++)
	{
		cudaStatus = cudaMalloc(&d_wavData, CHUNK_SIZE  * sizeof(cufftComplex));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "d_wav malloc failed");
			return cudaStatus;
		}

		cudaStatus = cudaMalloc(&d_fftData, CHUNK_SIZE * sizeof(cufftComplex));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "d_fft malloc failed");
			return cudaStatus;
		}

		//Copy over wav data
		cudaStatus = cudaMemcpy(d_wavData, wavData[i], CHUNK_SIZE * sizeof(cufftComplex), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "d_wavData Memcpy failed");
			return cudaStatus;
		}

		cufftExecC2C(plan, (cufftComplex*)d_wavData, (cufftComplex*) d_fftData, 1);

		//Copy fft data back
		cudaStatus = cudaMemcpy(fftData[i], d_fftData, CHUNK_SIZE * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "fftData Memcpy failed\n");
			system("PAUSE");
			return cudaStatus;
		}
	}
	
	delete wavData;
	cudaFree(d_wavData);
	cudaFree(d_fftData);
	cufftDestroy(plan);
	cudaDeviceReset();

	cout << "FFT Complete." << endl;

	double **highScores = new double * [numChunks];
	for (int i = 0; i < numChunks; i ++)
	{
		highScores[i] = new double[5];
		for (int j = 0; j < 5; j++)
		{
			highScores[i][j] = 0;
		}
		
	}

	//Scale the data down
	for (int t = 0; t < numChunks; t++)
	{
		for (int freq = LOWER_LIMIT; freq < UPPER_LIMIT - 1; freq ++)
		{
			double magnitude = log(abs(fftData[t][freq].x)) + 1;

			int index = getIndex(freq);

			if (magnitude > highScores[t][index])
			{
				highScores[t][index] = magnitude;
			}
		}
	}

	cout << "\a" << endl;
	//Display to test things
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			cout << highScores[i][j] << " ";
		}
		cout << endl;
	}
	
	//Housekeeping
	delete fftData;

	system("PAUSE");
	return 0;
}

long powerOfTwo(long input)
{
	int twos[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 
		1024, 2048, 4096, 8192, 16384, 32768, 65536, 
		131072, 262144, 524288, 1048576, 2097152, 4194304,
		8388608, 16777216, 33554432, 67108864, 134217728, 268435456,
		536870912, 1073741824};

	int i = 0;

	while (twos[i] < input)
	{
		i++;
	}

	return twos[i];
}

int getIndex(int freq)
{
	int RANGE[] = {40, 80, 120, 180, UPPER_LIMIT+1};
	int i = 0; 
	while(RANGE[i] < freq) 
	{
		i++;
	}
	return i;
};

long* readData(long* sampleLength)
{
	//Open the File
	FILE *f = fopen("test.wav", "rb");
	
	long dataPointer = 40;

	fseek(f, dataPointer, SEEK_SET); // Move the File pointer to data subchunk

	//Read the size from the subchunk header
	LongFromChar val;
	byte a = fgetc(f);
	byte b = fgetc(f);
	byte c = fgetc(f);
	byte d = fgetc(f);

	long size = charToLong(a,b,c,d);
	*sampleLength = size;
	int index = 0;

	int difference = powerOfTwo(size) - size;
	long* wavData = new long[size + difference];

   /*The data subchunk is arranged with interleaved channels
	* [channel0][channel1][channel0][channel1]
	*  short	 short	   short	 short
	*/
	while (dataPointer < size + 40)
	{
		a = fgetc(f);
		b = fgetc(f);
		c = fgetc(f);
		d = fgetc(f);
		wavData[index] = charToShort(a,b);
		dataPointer += 4; //Skip to the next block
		index ++;
	}
	while (dataPointer < (size + difference - 1))
	{
		wavData[index] = 0;
		dataPointer ++;
		index++;
	}

	fclose(f);
	return wavData;
}