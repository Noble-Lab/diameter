/* This executable takes as input a tab-delimited file whose first column 
 * contains the elemental composition of a peptide and second column contains
 * its charge. The other inputs are the minmz, maxmz, and binwidth.
 * The output will be a textfile containing, for each elemental composition,
 * a list of peak m/z and peak intensity for each peptide.
*/
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include "CMercury8.h"

using namespace std;

int main(int argc, char* argv[]){
	
//	char isotopefile[] = "ISOTOPE.DAT";

	CMercury8 *mercury;
	mercury = new CMercury8(argv[2].c_str());
	mercury->Echo(1);

	string line;
	string formula;
	string sequence;
	string mz;
	int charge;
	ifstream formulafile( argv[1] );
	while( getline(formulafile,line) ){
		istringstream iss(line);
		iss >> formula;
		iss >> charge;
		iss >> sequence;
		iss >> mz;
		printf( "Sequence\t%s\n", sequence.c_str() );
		printf( "Charge\t%d\n", charge );
		mercury->GoMercury( (char *)formula.c_str(),charge );
	}
	
	return 0;
};
