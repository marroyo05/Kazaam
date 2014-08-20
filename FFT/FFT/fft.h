#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cufft.h>
#include "DataPoint.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>

#define LOWER_LIMIT 40
#define UPPER_LIMIT 300

#define CHUNK_SIZE 2048

using namespace std;

unordered_map<string, DataPoint> fingerPrint(long *audio, long sampleLength);
long* readData(long* sampleLength);
long powerOfTwo(long input);
int getIndex(int freq);