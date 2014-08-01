// WavBot.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "WavBot.hpp"

int main()
{
	FILE *f;
	unsigned char tmp = NULL;

	f = fopen("test.wav", "rb");
	long dataPointer = 40; //WAV data length is at byte 40
	fseek(f, dataPointer, SEEK_SET);

	LongFromChar val;
	byte a = fgetc(f);
	byte b = fgetc(f);
	byte c = fgetc(f);
	byte d = fgetc(f);

	long size = charToLong(a,b,c,d);
	
	cout << "Size: " << size << endl;

	vector<unsigned long> data;
	while (dataPointer < size + 40)
	{
		a = fgetc(f);
		b = fgetc(f);
		data.push_back(charToShort(a,b));
		dataPointer += 2;
	}

	fclose(f);
	for (int i = 0; i < 25; i++)
	{
		cout << (unsigned long) data.at(i) << " ";
	}
	cout << endl;
	system("PAUSE");
}