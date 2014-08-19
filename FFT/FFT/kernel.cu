
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
int getIndex(int freq);

int main()
{

	long sampleLength; //Read the length of the sample
	cudaError_t cudaStatus;

	//Load wav file into memory
	long *audio = readData(&sampleLength);
	int amountPossible = sampleLength / CHUNK_SIZE;

	//Host memory
	cufftComplex **wavData = new cufftComplex* [amountPossible];
	cufftComplex **fftData = new cufftComplex* [amountPossible];

	//Creating Complex numbers from our readings
	for (int times = 0; times < amountPossible; times++)
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
	//X is sample length, Y is human hearing
	cufftPlan1d(&plan, sampleLength, CUFFT_R2C, 1);

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
		     
	//FFT
	for (int i = 0; i < amountPossible; i++)
	{
		
		//Copy over wav data
		 cudaStatus = cudaMemcpy(d_wavData, wavData[i], CHUNK_SIZE * sizeof(cuComplex), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "d_wavData Memcpy failed");
			return cudaStatus;
		}

		cufftExecC2C(plan, (cufftComplex* )d_wavData, (cufftComplex*) d_fftData, 1);

		//Copy fft data back
		cudaStatus = cudaMemcpy(fftData[i], d_fftData, CHUNK_SIZE * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "fftData Memcpy failed\n");
			return cudaStatus;
		}

	}

	cudaFree(d_wavData);
	cudaFree(d_fftData);
	cufftDestroy(plan);

	cout << "FFT Complete." << endl;

	//Scale the data down
	for (int i = 0; i < amountPossible; i++)
	{
		int freq = 1;
		for (int line = 1; line < CHUNK_SIZE; line ++)
		{
			double magnitude = log(abs(fftData[i][freq].x)) + 1;

			if ((log10(line) * log10(line)) > 1)
			{
				freq += (int) (log10(line) * log10(line));
			}
			else
			{
				freq++;
			}
		}
	}
	
	

	for (int freq = LOWER_LIMIT; freq < UPPER_LIMIT - 1; freq++)
	{
		double mag = log (abs(fftData[freq].x) + 1);

	}
	

	//Housekeeping
	delete wavData;
	delete fftData;

	cudaDeviceReset();

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