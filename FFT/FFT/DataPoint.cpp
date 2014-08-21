#include "DataPoint.h"

DataPoint::
	DataPoint(){};

DataPoint::
	DataPoint(int time, int id)
{
	t = time;
	songId = songId;
};

DataPoint::
	~DataPoint()
{
};

string DataPoint::
	getHash()
{
	return hash;
}

int DataPoint::
	getID()
{
	return songId;
}

int DataPoint::
	getT()
{
	return t;
}

void DataPoint::
	setHash(string hash)
{
	hash = hash;
}

void DataPoint::
	setID(int id)
{
	songId = id;
}

void DataPoint::
	setT(int time)
{
	t = time;
}


string DataPoint::
	toString()
{
 
	return to_string(songId) + " " + to_string(t);
}







