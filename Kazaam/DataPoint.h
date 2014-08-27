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

		friend bool operator<(DataPoint &a, DataPoint &b)
		{
			//We sort backwards so the biggest match is up front
			return a.getID() > b.getID();
		}

	private:
		string hash;
		int songId;
		int t;
};