#include "AudioManager.hpp"
#include <iostream>

#include <string>

using namespace std;


int main(int argc, char **argv)
{
	cout << "Loading...This may take a while..." << endl;
	AudioManager kazaam;
	cout << "\a" << endl;
	system("CLS");
	cout << "===================" << endl;
	cout << "|     KAZAAM      |" << endl;
	cout << "===================" << endl;

	int choice;

	bool endLoop = false;

	while (!endLoop)
	{

		cout << endl << endl;
		cout << "1) Load new wav files into database" << endl;
		cout << "2) Match audio" << endl;
		cout << "3) Quit" << endl;
		cin >> choice;

		switch(choice)
		{
			case 1:
				kazaam.setMode(STORING);
				kazaam.fingerPrintAudio();
				break;
			case 2:
				kazaam.setMode(RECORDING);
				kazaam.fingerPrintAudio();
				break;
			case 3:
				endLoop = true;
				break;
			default:
				cout << "Please enter a valid choice." << endl;
		}
	}

	system("PAUSE");
}
