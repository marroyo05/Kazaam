#pragma once

#include "AudioThread.hpp"

#include <omp.h>
#include <queue>
#include <sstream>

#define NUM_OBJECTS 16 //The number of objects should = simultaneous threads

typedef 
	enum {
		RECORDING,
		STORING
	} mode;

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
		string musicPath;
		bool storing;
		void loadFileList(string directory);

	public:
		AudioManager(); //Creates default number of soundbots
		//AudioManager(int numObjects); //Creates custom number of soundbots
		~AudioManager();
		DataPoint query(string hash); //Queries the hashmap, returns a datapoint
		void fingerPrintAudioT();
		void fingerPrintAudio();
		void saveMap(unordered_map<string, DataPoint> um);
		unordered_map<string, DataPoint>AudioManager::LoadMap();
		vector<DataPoint> Match(vector<pair<string,DataPoint>> hashes);
		vector<DataPoint> NumMatches(vector<DataPoint> allresults);
		int audioRecord();
		void setMode( mode m);
};

AudioManager::
	AudioManager()
{
	FILE* data;
	data = fopen("data.txt", "r");
	if (data == NULL)
	{
		storing = true;
		loadFileList("C:/Users/Zach/Desktop/WaveData/");
		fingerPrintAudio();
	}
	else
	{
		fclose(data);
		database = LoadMap();
	}
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
	if (storing)  //We're adding hashes to our database
		{
		for (int i = 0; i < numFiles; i++)
		{
			vector<pair<string, DataPoint>> hashes;
			//Load a wav file
			pool[i].readData();
			pool[i].setSongID(i);
			//Convert it to hashes
			hashes = pool[i].analyze();
			//Add our hashes to the map
			for (int i = 0; i < hashes.size(); i++)
			{
				database.emplace(hashes[i]);
			}
		}
	}
	else //We're recording things
	{
		//Make an audio thread
		AudioThread at;
		//Record a wav file
		audioRecord();
		//Load it into an analyzer
		at.setFileName("temp.wav");
		at.readData();
		//Fingerprint
		vector<pair<string, DataPoint>> hashes;
		vector<DataPoint> results;
		hashes = at.analyze();
		//Match
		results = Match(hashes);

		//Group by number of results
		results = NumMatches(results);
		sort(results.begin(), results.end());
		cout << "The match is probably " << results[0].getID() << endl;
	}

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
			file << itr->first<< " " << itr->second.toString() << endl;
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
	return recordMic(20);
}

void AudioManager::setMode(mode m)
{
	storing = m;
}

void AudioManager::loadFileList(string directory)
{
		//musicPath = "C:/Users/Zach/Desktop/WaveData/";
		musicPath = directory;
		//Create a pool of AudioProcessors
		pool = new AudioThread[NUM_OBJECTS];
		//Get Files
		fileList = listFiles(musicPath);
		numFiles = fileList.size();
		//Assign Files to Processors
		for (int i = 0; i < numFiles; i++)
		{
			pool[i].setFileName(musicPath + fileList.front());
			fileList.pop();
		}
}

vector<DataPoint> AudioManager::Match(vector<pair<string,DataPoint>> Hashes)
{
	vector<DataPoint> Matches;

	for ( int i = 0; i < Hashes.size(); i++)
	{
		Matches.push_back(query(Hashes[i].first));
	}
	return Matches;
}

vector<DataPoint> AudioManager::NumMatches(vector<DataPoint> allresults)  //returns list of matches with count of song occurrences
{
	vector<DataPoint> TotalMatches;
	for ( int i = 0; i < allresults.size(); i++)
	{
		bool duplicate = false;
		for (int j = 0; j < TotalMatches.size(); j++)
		{
			//Strip out our invalid results.
			if (allresults[i].getID() == -1)
			{
				break;
			}
			//if ID is duplicate increment T 
			else if ( allresults[i].getID() == TotalMatches[j].getID() )
			{
				TotalMatches[j].setT(TotalMatches[j].getT() + 1);
				duplicate = true;
				break;
			}
		}
		if (!duplicate)
		{
			//Make new datapoint
			int id = allresults[i].getID();
			if (id != -1)
			{
				DataPoint d( 1, id);
				TotalMatches.push_back(d);
			}
		}
	}

	return TotalMatches;
}
