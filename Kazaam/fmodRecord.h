#include <fmod.hpp>
#include <fmod_errors.h>

#include <Windows.h>
#include <iostream>
#include <conio.h>
using namespace std;

void ERRCHECK(FMOD_RESULT result)
{
	if (result != FMOD_OK)
	{
		cout << "FMOD error!" << result << FMOD_ErrorString(result);
	}
}

#if defined(WIN32) || defined(__WATCOMC__) || defined(_WIN32) || defined(__WIN32__)
    #define __PACKED                         /* dummy */
#else
    #define __PACKED __attribute__((packed)) /* gcc packed */
#endif


void SaveToWav(FMOD::Sound *sound)
{
    FILE *fp;
    int             channels, bits;
    float           rate;
    void           *ptr1, *ptr2;
    unsigned int    lenbytes, len1, len2;

    if (!sound)
    {
        return;
    }

    sound->getFormat  (0, 0, &channels, &bits);
    sound->getDefaults(&rate, 0, 0, 0);
    sound->getLength  (&lenbytes, FMOD_TIMEUNIT_PCMBYTES);

    {
        #if defined(WIN32) || defined(_WIN64) || defined(__WATCOMC__) || defined(_WIN32) || defined(__WIN32__)
        #pragma pack(1)
        #endif
        
        /*
            WAV Structures
        */
        typedef struct
        {
	        signed char id[4];
	        int 		size;
        } RiffChunk;
    
        struct
        {
            RiffChunk       chunk           __PACKED;
            unsigned short	wFormatTag      __PACKED;    /* format type  */
            unsigned short	nChannels       __PACKED;    /* number of channels (i.e. mono, stereo...)  */
            unsigned int	nSamplesPerSec  __PACKED;    /* sample rate  */
            unsigned int	nAvgBytesPerSec __PACKED;    /* for buffer estimation  */
            unsigned short	nBlockAlign     __PACKED;    /* block size of data  */
            unsigned short	wBitsPerSample  __PACKED;    /* number of bits per sample of mono data */
        } FmtChunk  = { {{'f','m','t',' '}, sizeof(FmtChunk) - sizeof(RiffChunk) }, 1, channels, (int)rate, (int)rate * channels * bits / 8, 1 * channels * bits / 8, bits } __PACKED;

        struct
        {
            RiffChunk   chunk;
        } DataChunk = { {{'d','a','t','a'}, lenbytes } };

        struct
        {
            RiffChunk   chunk;
	        signed char rifftype[4];
        } WavHeader = { {{'R','I','F','F'}, sizeof(FmtChunk) + sizeof(RiffChunk) + lenbytes }, {'W','A','V','E'} };

        #if defined(WIN32) || defined(_WIN64) || defined(__WATCOMC__) || defined(_WIN32) || defined(__WIN32__)
        #pragma pack()
        #endif

        fp = fopen("temp.wav", "wb");
       
        /*
            Write out the WAV header.
        */
        fwrite(&WavHeader, sizeof(WavHeader), 1, fp);
        fwrite(&FmtChunk, sizeof(FmtChunk), 1, fp);
        fwrite(&DataChunk, sizeof(DataChunk), 1, fp);

        /*
            Lock the sound to get access to the raw data.
        */
        sound->lock(0, lenbytes, &ptr1, &ptr2, &len1, &len2);

        /*
            Write it to disk.
        */
        fwrite(ptr1, len1, 1, fp);

        /*
            Unlock the sound to allow FMOD to use it again.
        */
        sound->unlock(ptr1, ptr2, len1, len2);

        fclose(fp);
    }
}

int recordMic()
{
	FMOD::System *sys = 0;
	FMOD::Sound *sound = 0;
	FMOD::Channel *channel = 0;
	FMOD_RESULT result;
	FMOD_CREATESOUNDEXINFO exinfo;
	int key, driver, recorddriver, numdrivers, count;
	unsigned int version;

	result = FMOD::System_Create(&sys);
	ERRCHECK(result);

	result = sys->getVersion(&version);
	ERRCHECK(result);

	if (version < FMOD_VERSION)
	{
		cout << "You are using an old version of FMOD!" << endl;
		return 0;
	}

	result = sys->setOutput(FMOD_OUTPUTTYPE_WINMM);
	ERRCHECK(result);

	result = sys->getRecordNumDrivers(&numdrivers);
	ERRCHECK(result);

	recorddriver = 0;

	result = sys->init(32, FMOD_INIT_NORMAL, 0);
	ERRCHECK(result);

	memset(&exinfo, 0, sizeof(FMOD_CREATESOUNDEXINFO));

	exinfo.cbsize = sizeof(FMOD_CREATESOUNDEXINFO);
	exinfo.numchannels = 1;
	exinfo.format = FMOD_SOUND_FORMAT_PCM16;
	exinfo.defaultfrequency = 44100;
	exinfo.length = exinfo.defaultfrequency * sizeof(short) * exinfo.numchannels * 5;

	result = sys->createSound(0, FMOD_2D | FMOD_SOFTWARE | FMOD_OPENUSER, &exinfo, &sound);
	ERRCHECK(result);

	static bool looping = false;
	bool recording = false;
	bool playing = false;
	unsigned int recordpos = 0;
	unsigned int playpos = 0;
	unsigned int length;
	cout << "Press enter to start" << endl;
	system("PAUSE > NUL");
	result = sys->recordStart(recorddriver, sound, looping);
	ERRCHECK(result);
	cout << "Press enter to stop" << endl;
	system("PAUSE > NUL");

	result = sys->recordStop(recorddriver);
	if (channel)
	{
		channel->stop();
		channel = 0;
	}

	SaveToWav(sound);

	result = sound->release();
    ERRCHECK(result);

    result = sys->release();
    ERRCHECK(result);
	return 0;	
}
