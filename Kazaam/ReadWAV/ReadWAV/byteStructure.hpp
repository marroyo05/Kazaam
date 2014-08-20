typedef unsigned char byte;

//Read Shorts from Chars
struct ShortFromChar
{
	byte a, b;
};

//Read Longs from Chars
struct LongFromChar
{
	byte a, b, c, d;
};

//Little Endian only
unsigned long charToLong(byte a, byte b, byte c, byte d)
{
	LongFromChar val;
	val.a = a;
	val.b = b;
	val.c = c;
	val.d = d;
	unsigned long *l = (unsigned long*) &val;
	
	return *l;
}

//Big Endian only
unsigned long bigCharToLong(byte a, byte b, byte c, byte d)
{
	LongFromChar val;
	val.a = d;
	val.b = c;
	val.c = b;
	val.d = a;
	unsigned long *l = (unsigned long*) &val;
	
	return *l;
}

//Little Endian only
unsigned short charToShort(byte a, byte b)
{
	ShortFromChar val;
	val.a = a;
	val.b = b;
	unsigned short *s = (unsigned short*) &val;

	return *s;
}

//Big Endian only
unsigned short bigCharToShort(byte a, byte b)
{
	ShortFromChar val;
	val.a = b;
	val.b = a;
	unsigned short *s = (unsigned short*) &val;

	return *s;
}