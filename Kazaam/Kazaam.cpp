#include "AudioManager.hpp"
#include <iostream>

#include <string>

using namespace std;


int main(int argc, char **argv)
{
	AudioManager *kazaam = new AudioManager();
	
	cout << "Begin." << endl;
	kazaam->fingerPrintAudio();
	cout << "End." << endl;
	
	system("PAUSE");
}

