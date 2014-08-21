#include "AudioThread.hpp"
#include <iostream>

#include <string>

using namespace std;

int main(int argc, char **argv)
{
	AudioThread at;
	string fname = "test.wav";
	at.setFileName(fname);
	at.analyze();
	
	system("PAUSE");
}