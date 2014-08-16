#include <fmod.hpp>
#include <fmod_errors.h>
#include <iostream>
#include <Windows.h>

using std::cout;
using std::endl;

class Fmod
{
private:
	FMOD::System *sys;
	FMOD_RESULT result;

public:
	Fmod();
	void playFile(const char* fName);

};
void FMODErrorCheck(FMOD_RESULT result)
{
	if (result != FMOD_OK)
	{
		cout << "FMOD ERROR (" << result << ") " << FMOD_ErrorString(result) << endl;
		exit(-1);
	}
}

Fmod::Fmod(){
	unsigned int version;
	int numDrivers;
	FMOD_SPEAKERMODE speakerMode;
	FMOD_CAPS caps;

	//create FMOD interface object
	result = FMOD::System_Create(&sys);
	FMODErrorCheck(result);

	//Check version
	result = sys->getVersion(&version);
	FMODErrorCheck(result);
	
	if (version < FMOD_VERSION)
	{
		cout << "Using old version " << version << endl;
	}

	result = sys->getNumDrivers(&numDrivers);
	FMODErrorCheck(result);

	if (numDrivers == 0)
	{
		result = sys->setOutput(FMOD_OUTPUTTYPE_NOSOUND);
		FMODErrorCheck(result);
	}
	else{
		result = sys->getDriverCaps(0, &caps, 0, &speakerMode);
		FMODErrorCheck(result);

		result = sys->setSpeakerMode(speakerMode);
		FMODErrorCheck(result);
	}
	if (caps & FMOD_CAPS_HARDWARE_EMULATED)
	{
		result = sys->setDSPBufferSize(1024, 10);
		FMODErrorCheck(result);
	}

	result = sys->init(100, FMOD_INIT_NORMAL, 0);

	if (result == FMOD_ERR_OUTPUT_CREATEBUFFER)
	{
		result = sys->setSpeakerMode(FMOD_SPEAKERMODE_STEREO);
		FMODErrorCheck(result);

		result = sys->init(100, FMOD_INIT_NORMAL, 0);
	}
	FMODErrorCheck(result);
}

void Fmod::playFile(const char* fName)
{
	cout << fName << endl;
	FMOD::Sound *audio;
	FMOD::Channel *channel;

	sys->createSound(fName, FMOD_DEFAULT, 0, &audio);
	sys->playSound(FMOD_CHANNEL_FREE, audio, false, &channel);
}


