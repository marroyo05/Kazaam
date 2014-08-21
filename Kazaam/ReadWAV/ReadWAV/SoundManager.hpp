#ifndef __SOUNDFACTORY
#define __SOUNDFACTORY

#include "WavBot.hpp"

#include <deque>
#include <unordered_map>
#include <fstream>
#include <sstream>


   /*
	*	TODO: We need to implement a write queue.  Callbacks would be useful
	*	Thread this sumbitch
	*/

#define NUM_OBJECTS 16 //The number of objects should = simultaneous threads

class 
	SoundFactory
{
	private:
		WavBot* pool;
		deque<string> fileList;
		FILE* outPut; //File? DB? ?? 
		void readFileList();

	public:
		SoundFactory(); //Creates default number of soundbots
		SoundFactory(int numObjects); //Creates custom number of soundbots
		~SoundFactory();
		void startEngine();

};

SoundFactory::
	SoundFactory()
{
	pool = new WavBot[NUM_OBJECTS];
	readFileList(); //Replace this with Miguel's directory to string[] code
	
}

SoundFactory::
	~SoundFactory()
{
	for (int i = 0; i < NUM_OBJECTS; i++)
	{
		this->pool[i].reset();
	}
	//Shut down the output
}

void SoundFactory::
	startEngine()
{
#pragma omp parallel
	{
		int check = 0;  // An int to check for errors
		while (!fileList.empty()) //Still stuff to process
		{
			for (int i = 0; i < NUM_OBJECTS; i++) //Loop through the pool
			{
			   // Our control switch. Decides what each object should do next.
				switch(this->pool[i].stage) //Which stage are we at??
				{
					case waitingForFile:
						this->pool[i].setFileName(fileList.front);
						fileList.pop_front();
						break;
					case waitingOnRead:
						check = this->pool[i].readData();
						if (!check)
						{
							cout << this->pool[i].WavFileName << " failed." << endl;
						}
						break;
					case reading:
						//Do nothing until the read is done.
						//This may cause a lock error...
						break;
					case waitingOnFFT:
						this->pool[i].analyze();
						break;
					case FFT:
						//Do nothing until the read is done.
						//This may cause a lock error...
						break;
					case waitingOnWrite:
						this->pool[i].writeData(outPut);
						break;
					case writing:
						//Do nothing until the read is done.
						//This may cause a lock error...
						break;
					default:
						break;
				} //Switchboard
			} //Object pool loop
		} // File reader
	} // Parallel
}



unordered_map<string, DataPoint> LoadMap()
{
	string CurrentLine;
	unordered_map<string,DataPoint> um;
	ifstream myFile;
	myFile.open("data.txt", ios::in);
	if(myFile.is_open())
	{
		while(getline(myFile,CurrentLine)) 
		{
			vector < string> tokens;
			istringstream iss (CurrentLine);
			copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter<vector<string>>(tokens));
			string hash = tokens[0];
			int songID = stoi(tokens[1]);
			int t = stoi(tokens[2]);
			DataPoint d;
			d.setID(songID);
			d.setT(t);
 
			um.insert(make_pair(hash,d));
		}
		myFile.close();	
		cout << "File was opened";
	}
	else cout<< "Unable to open file";
 
	return um;
}
 
 
void saveMap(unordered_map<string, DataPoint> um)
{
	ofstream file ("data.txt");
	if (file.is_open())
	{
		//Iterate through map
		for (auto itr = um.begin(); itr != um.end(); ++itr)
		{
				//write to file
			file << itr->first<< " " << itr->second.toString() << endl;
		}
		file.close();
	}
	else cout << "Unable to open file";
}
#endif