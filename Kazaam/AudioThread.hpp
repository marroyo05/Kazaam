#pragma once

#include "byteStructure.hpp"
#include "fft.h"
#include "fmodRecord.h"

#include <cstdio>
#include <exception>
#include <iostream>
#include <omp.h>
#include <string>
#include <time.h>
#include <vector>
using namespace std;

typedef 
	enum {
		waitingForFile, 
		waitingOnRead, 
		reading, 
		waitingOnFFT, 
		FFT, 
		waitingOnWrite, 
		writing
} PoolStage; 


typedef 
	enum {
		RECORDING,
		STORING
	} mode;

class 
	AudioThread
{
	friend class AudioManager;

	private:
		FILE *f;
		string WavFileName;
		long dataPointer;
		long numSamples;
		PoolStage stage;

		int channel0Len, channel1Len;
		long *channel0;
		vector<short> channel1;

		bool storing; //true if storing audio, false if recording
		int readData(); //Fill the channel buffers with data from wav file
		int writeData(FILE *f); // write out to ???
		void reset(); // reset back to original state
		
	public:
		AudioThread();
		~AudioThread();
		int audioRecord();
		void setFileName(string fname);
		void setMode(mode m);
		void analyze(); //FFT and whatnot
};

AudioThread::
	AudioThread()
{
	WavFileName = "";
	dataPointer = 40; // The byte offset where data size is
	//channel0Len = channel0.size();
	//channel1Len = channel1.size();
	stage = waitingForFile;
	storing = true;

}

AudioThread::
	~AudioThread()
{
	//Make sure that our buffers are empty
	//channel0.clear();
	channel1.clear();
}

void AudioThread::
	setFileName(string fname)
{
	WavFileName = fname;
	stage = waitingOnRead;
}

int AudioThread::
	readData()
{
	stage = reading;
	//Open the File

	f = fopen(WavFileName.c_str(), "rb");
		
	if (f != NULL)
	{
		fseek(f, dataPointer, SEEK_SET); // Move the File pointer to data subchunk
	
	
		//Read the size from the subchunk header
		LongFromChar val;
		byte a = fgetc(f);
		byte b = fgetc(f);
		byte c = fgetc(f);
		byte d = fgetc(f);

		long size = charToLong(a,b,c,d);
		numSamples = size / 2;

		channel0 = new long[numSamples];

	   /*The data subchunk is arranged with interleaved channels
		* [channel0][channel1][channel0][channel1]
		*  short	 short	   short	 short
		*/
		int i = 0;
		while (dataPointer < size + 40)
		{
			a = fgetc(f);
			b = fgetc(f);
			c = fgetc(f);
			d = fgetc(f);
			channel0[i] = charToShort(a,b); //Left channel
			channel1.push_back(charToShort(c,d)); //Right channel
			dataPointer += 4; //Skip to the next block
		}

		fclose(f);
		dataPointer = 40; // Reset data pointer
		stage = waitingOnFFT; // Move to the next section in the pool
		return 1;
	}
	else
	{
		cout << "File not found." << endl;
		return 0;
	}
}

void AudioThread::setMode(mode m)
{
	storing = m;
}

void AudioThread::
	reset()
{
	WavFileName = ""; // Clear the file name
	numSamples = -1; //Reset file size
	stage = waitingForFile; //Reset the stage
}

void AudioThread::
	analyze()
{
	if (storing)
	{
		fingerPrint(channel0, numSamples);
	}
}

