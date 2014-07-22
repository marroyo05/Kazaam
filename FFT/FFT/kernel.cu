
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cufft.h>
#include "byteStructure.hpp"

#include <iostream>
#include <fstream>
using namespace std;
cufftReal* readData(long* sampleLength);

int main()
{
	long sampleLength; //Read the length of the sample
	int deltaT = 1; //timescale
	cudaError_t cudaStatus;

	//Host Memory
	cufftReal *wavData = readData(&sampleLength);
	cufftComplex *fftData = new cufftComplex[sampleLength];

	//Device Memory
	cufftReal *d_wavData;
	cufftComplex *d_fftData;


	cudaStatus = cudaMalloc(&d_wavData, sampleLength * sizeof(cufftReal));
	 if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "d_wav malloc failed");
    }

	 cudaStatus = cudaMalloc(&d_fftData, (sampleLength) * sizeof(cufftComplex));
	 if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "d_fft malloc failed");
		return cudaStatus;
    }

	//Copy over wav data
	 cudaStatus = cudaMemcpy(d_wavData, wavData, sampleLength * sizeof(cufftReal), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "d_wavData Memcpy failed");
		return cudaStatus;
    }


	//The actual transform
	cufftHandle plan;
	cufftPlan1d(&plan, sampleLength, CUFFT_R2C, 1);
	cufftExecR2C(plan, (cufftReal* )d_wavData, (cufftComplex*) d_fftData);

	//Copy fft data back
	cudaStatus = cudaMemcpy(fftData, d_fftData, sampleLength * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "fftData Memcpy failed\n");
		return cudaStatus;
    }
	
	cout << "WAV: ";
	for (int i = 0; i < 20; i++)
	{
		cout << wavData[i] << " ";
	}

	//This should be written out to be analyzed properly...
	ofstream fout("fft.txt", ofstream::out | ofstream::binary);

	cout << endl << "FFT: ";
	for (int j = 0; j < 20; j++)
	{
		//cout << fftData[j].x << " ";
		fout << fftData[j].x << endl;
	}
	cout << endl;
	cout << "\a" << endl;


	fout.close();
	//Housekeeping
	delete wavData;
	delete fftData;

	cudaFree(d_wavData);
	cudaFree(d_fftData);

	cufftDestroy(plan);
	cudaDeviceReset();

	system("PAUSE");
	return 0;
}

cufftReal* readData(long* sampleLength)
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

	cufftReal* wavData = new cufftReal[size];
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

	fclose(f);
	dataPointer = 40; // Reset data pointer
	return wavData;
}