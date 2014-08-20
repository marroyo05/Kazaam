#ifndef __WAVBOT
#define __WAVBOT

#include "byteStructure.hpp"
#include "fft.h"
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <time.h>
#include <vector>
using namespace std;

typedef unsigned char byte;

//Read Shorts from Chars
struct ShortFromChar
{
	byte a, b;
};

//Read Longs from Chars
struct LongFromChar
{
	byte a, b, c, d;
};

//Little Endian only
unsigned long charToLong(byte a, byte b, byte c, byte d)
{
	LongFromChar val;
	val.a = a;
	val.b = b;
	val.c = c;
	val.d = d;
	unsigned long *l = (unsigned long*) &val;
	
	return *l;
}

//Little Endian only
unsigned short charToShort(byte a, byte b)
{
	ShortFromChar val;
	val.a = a;
	val.b = b;
	unsigned short *s = (unsigned short*) &val;

	return *s;
}

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

class 
	WavBot
{
	friend class SoundFactory;

	private:
		FILE *f;
		string WavFileName;
		long dataPointer;
		long fileSize;
		PoolStage stage;

		int channel0Len, channel1Len;
		vector<short> channel0;
		vector<short> channel1;

		void setFileName(string fname);
		int readData(); //Fill the channel buffers with data from wav file
		int writeData(FILE *f); // write out to ???
		void reset(); // reset back to original state
		void analyze(); //FFT and whatnot

		
	public:
		WavBot();
		~WavBot();
};

WavBot::
	WavBot()
{
	WavFileName = "";
	dataPointer = 40; // The byte offset where data size is
	channel0Len = channel0.size();
	channel1Len = channel1.size();
	stage = waitingForFile;

}

WavBot::
	~WavBot()
{
	//Make sure that our buffers are empty
	channel0.clear();
	channel1.clear();
}

void WavBot::
	setFileName(string fname)
{
	WavFileName = fname;
	stage = waitingOnRead;
}

int WavBot::
	readData()
{
	stage = reading;
	//Open the File
	try
	{
		f = fopen(WavFileName.c_str(), "rb");
	}
	catch(exception &e)
	{
		cout << e.what() << endl;
		return -1;
	}

	fseek(f, dataPointer, SEEK_SET); // Move the File pointer to data subchunk

	//Read the size from the subchunk header
	LongFromChar val;
	byte a = fgetc(f);
	byte b = fgetc(f);
	byte c = fgetc(f);
	byte d = fgetc(f);

	long size = charToLong(a,b,c,d);

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
		channel0.push_back(charToShort(a,b)); //Left channel
		channel1.push_back(charToShort(c,d)); //Right channel
		dataPointer += 4; //Skip to the next block
	}

	fclose(f);
	dataPointer = 40; // Reset data pointer
	stage = waitingOnFFT; // Move to the next section in the pool
	return 1;
}

void WavBot::
	reset()
{
	WavFileName = ""; // Clear the file name
	fileSize = -1; //Reset file size
	stage = waitingForFile; //Reset the stage
}

void WavBot::
	analyze()
{
	unordered_map<string, DataPoint> umm;
}

#endif