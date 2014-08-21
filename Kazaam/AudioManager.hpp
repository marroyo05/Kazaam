#pragma once

#include "AudioThread.hpp"

#include <queue>
#include <sstream>

#define NUM_OBJECTS 16 //The number of objects should = simultaneous threads

class 
	AudioManager
{
	private:
		AudioThread* pool;
		boolean fileLoaded;
		queue<string> fileList;
		unordered_map<string, DataPoint> database;
		queue<string> listFiles(string folder);
		int numFiles;
		vector<DataPoint> matches; 

	public:
		AudioManager(); //Creates default number of soundbots
		//AudioManager(int numObjects); //Creates custom number of soundbots
		~AudioManager();
		DataPoint query(string hash); //Queries the hashmap, returns a datapoint
		void fingerPrintAudioT();
		void fingerPrintAudio();
		void saveMap(unordered_map<string, DataPoint> um);
		unordered_map<string, DataPoint>AudioManager::LoadMap();
		int audioRecord();
};

AudioManager::
	AudioManager()
{
	//Create a pool of AudioProcessors
	pool = new AudioThread[NUM_OBJECTS];
	//Get Files
	fileList = listFiles("C:/Users/Zach/Desktop/WaveData/");
	numFiles = fileList.size();
	//Assign Files to Processors
	for (int i = 0; i < numFiles; i++)
	{
		pool[i].setFileName(fileList.front());
		fileList.pop();
	}
	//database = LoadMap();
	
}

AudioManager::
	~AudioManager()
{
	for (int i = 0; i < NUM_OBJECTS; i++)
	{
		this->pool[i].reset();
	}
	//Shut down the output
	saveMap(database);
}

void AudioManager::
	fingerPrintAudio()
{
	for (int i = 0; i < numFiles; i++)
	{
		/*
		*Call Analyze and add to the hashmap.
		*/

	}

}

void AudioManager::
	fingerPrintAudioT()
{
		int check = 0;  // An int to check for errors
		while (!fileList.empty()) //Still stuff to process
		{
			for (int i = 0; i < NUM_OBJECTS; i++) //Loop through the pool
			{
			   // Our control switch. Decides what each object should do next.
				switch(this->pool[i].stage) //Which stage are we at??
				{
					case waitingForFile:/*
						this->pool[i].setFileName(fileList.front);
						fileList.pop();*/
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
						//this->pool[i].writeData(outPut);
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
}

unordered_map<string, DataPoint>AudioManager::
	LoadMap()
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
	else cout << "Unable to open file";
 
	return um;
}
 
void AudioManager::
	saveMap(unordered_map<string, DataPoint> um)
{
	ofstream file ("data.txt");
	if (file.is_open())
	{
		//Iterate through map
		for (auto itr = um.begin(); itr != um.end(); ++itr)
		{
				//write to file
			//file << itr->first<< " " << itr->second. << endl;
		}
		file.close();
	}
	else cout << "Unable to open file";
}

DataPoint AudioManager::
	query(string hash)
{
	unordered_map<string, DataPoint>::iterator match;
	match = database.find(hash);
	if (match == database.end()) //Nothing found
	{
		return DataPoint(-1, -1); //Return invalid match
	}
	else //Found something
	{
		return match->second; //Return our DataPoint
	}
}

queue<string> AudioManager::
	listFiles(string folder)
{
    queue<string> names;
    char search_path[200];
    sprintf(search_path, "%s*.*", folder.c_str());
    WIN32_FIND_DATA fd; 
    HANDLE hFind = ::FindFirstFile(search_path, &fd); 
    if(hFind != INVALID_HANDLE_VALUE) 
    { 
        do 
        { 
            // read all (real) files in current folder
            // , delete '!' read other 2 default folder . and ..
            if(! (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ) 
            {
                names.push(fd.cFileName);
            }
        }while(::FindNextFile(hFind, &fd)); 
        ::FindClose(hFind); 
    } 
    return names;
}

int AudioManager::
	audioRecord()
{
	//Right now this drops a wav file.  It should fingerprint it first.
	return recordMic();
}