#include <string>
using namespace std;

class DataPoint
{
public:
	DataPoint();
	DataPoint(int time, int id);
	~DataPoint();
	string getHash();
	int getID();
	int getT();

	void setHash(string hash);
	void setID(int id);
	void setT(int time);
	string toString();
private:
	string hash;
	int songId;
	int t;
};