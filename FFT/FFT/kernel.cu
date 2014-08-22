//
#include "fft.h"
//
vector<string, DataPoint> fingerPrint(long *audio, long sampleLength)
{
	//long sampleLength; //Read the length of the sample
	cudaError_t cudaStatus;

	//Load wav file into memory
	//long *audio = readData(&sampleLength);
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
		}

		cudaStatus = cudaMalloc(&d_fftData, CHUNK_SIZE * sizeof(cufftComplex));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "d_fft malloc failed");
		}

		//Copy over wav data
		cudaStatus = cudaMemcpy(d_wavData, wavData[i], CHUNK_SIZE * sizeof(cufftComplex), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "d_wavData Memcpy failed");
		}

		cufftExecC2C(plan, (cufftComplex*)d_wavData, (cufftComplex*) d_fftData, 1);

		//Copy fft data back
		cudaStatus = cudaMemcpy(fftData[i], d_fftData, CHUNK_SIZE * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "fftData Memcpy failed\n");
		}
	}
	
	delete wavData;
	cudaFree(d_wavData);
	cudaFree(d_fftData);
	cufftDestroy(plan);
	cudaDeviceReset();

	cout << "FFT Complete." << endl;

	//Create an array of hash points
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
	cout << "Reduction Complete." << endl;
	


	//Store the matches
	vector<pair<string, DataPoint>> storage;
	int fuzz = 2; // fuzziness
	for (int t = 0; t < numChunks; t++)
	{
		string hash = "";
		for (int j = 0; j < 5; j++)
		{
			//concat our hash with a fuzziness
			hash += to_string((highScores[t][j] - ((int)highScores[t][j] % fuzz)));
		}
		DataPoint d(t, 0); //Create a data poin
		pair<string, DataPoint> point (hash, d); //Pair it with the hash we calculated
		storage.push_back(point);
	}

	//Housekeeping
	delete fftData;
	delete highScores;

	return storage;
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

//This doesn't do it's job right.
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

unordered_map<string, DataPoint> ToMap(vector<pair<string,DataPoint>> data)
{
	unordered_map<string,DataPoint> storage;
	for (int t = 0; t < data.size(); t++)
	{
		
		storage.insert(data[t]);
	}
	return storage;
}