#include "AudioManager.hpp"
#include <iostream>

#include <string>

using namespace std;


int main(int argc, char **argv)
{
	cout << "Loading...This may take a while..." << endl;
	AudioManager kazaam;
	system("CLS");
	cout << "===================" << endl;
	cout << "|     KAZAAM      |" << endl;
	cout << "===================" << endl;

	cout << kazaam.getNumFiles() << " songs loaded." << endl;

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
				break;
			case 2:
				kazaam.setMode(RECORDING);
				break;
			case 3:
				endLoop = true;
				break;
			default:
				cout << "Please enter a valid choice." << endl;
				break;
		}
	}

	system("PAUSE");
}

vector<DataPoint> NumMatches(vector<DataPoint> allresults)
{
	vector<DataPoint> TotalMatches;
	for ( int i = 0; i < allresults.size(); i++)
	{
		bool duplicate = false;
		for (int j = 0; j < TotalMatches.size(); j++)
		{
			//increment if ID is duplicate
			if ( allresults[i].getID() == TotalMatches[j].getID() )
			{
				TotalMatches[j].setT(TotalMatches[j].getT() + 1);
				duplicate = true;
				break;
			}
		}
		if (!duplicate)
		{
			//Make new datapoint
			TotalMatches.push_back(allresults[i]);
		}
	}

	return TotalMatches;
}