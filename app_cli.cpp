/***************************************************************************
 *   Copyright (C) 2009 France Telecom SA - Orange Labs                    *
 *   by Changlin Liu and Luca Muscariello                                  *
 *   luca.muscariello@orange-ftgroup.com                                   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU Lesser General Public License as        *
 *   published by the Free Software Foundation; either version 2 of the    *
 *   License, or (at your option) any later version.                       *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU Lesser General Public      *
 *   License along with this program; if not, write to the                 *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#include "src/crnn_t.h"
#include <string.h>
#include <sstream>
using namespace std;
using namespace rnn_lite;

void parse_args_codec_cmd(string &strInput) throw (argument_exception);
int parse_args(rnn_lite::CRNN_T* rnn, int argc, char** argv) throw (argument_exception);
void printUsageCli();

int main(int argc, char** argv)
{//   double temp = 200;
try{
    CRNN_T rnn;
    if(parse_args(&rnn,argc, argv)){
        printUsageCli();
        return 0;
    }
    rnn.test();
    return 0;
}
catch (argument_exception& e){
    cout<<"invalid argument: "<<e.what()<<endl;
    return 1;
}
catch (rnn_exception& e){
    cout<<"##Excepntion: "<<e.what()<<endl;
    return 1;
}

}

//*************************************************************************************
//  Parse codec, convert into 0.1+N*0.2
//*************************************************************************************
void parse_args_codec_cmd(string &strInput) throw (argument_exception)
{
    stringstream ss;
    fstream fileCodec;
    string strTemp,strTemp2;
    int intEntryCount = 0;
    int intFound = -1;
    int intCodecSize = 0;

    //open corresponding files
    fileCodec.open("codec.conf",ios::in);
    if(!fileCodec.is_open())
        throw argument_exception("failed to open codec conf file");

    //load codec table, store to ssc
    while(getline(fileCodec,strTemp))
        intEntryCount++;
    intCodecSize = intEntryCount;
    stringstream* ssc = new stringstream[intCodecSize];
    fileCodec.clear();
    fileCodec.seekg(0,ios::beg);
    for(int i=0;i<intCodecSize;i++){
        getline(fileCodec,strTemp);
        ssc[i]<<strTemp;
    }

    for(int i=0;i<intCodecSize;i++){    //for every codec
        while(intFound < 0 && ssc[i]>>strTemp){    //for every variation of certain codec
            intFound = strInput.find(strTemp);
        }
//        ssc[i].clear();
//        ssc[i].seekg(0,ios::beg);
        if(intFound >= 0){      //found codec column, match i'th codec
            ss.clear();
            ss<<0.1+0.2*i;
            ss>>strTemp2;
            strInput.replace(intFound,strTemp.length(),strTemp2);
//            intFound = -1;
            break;              //no need to check other codec
        }
    }
}





//*************************************************************************************
//  Parse agrument, define network topology and normalize_vector
//*************************************************************************************
int parse_args(rnn_lite::CRNN_T* rnn, int argc, char** argv)throw(argument_exception)
{
    string strInput = "";

    //parse parameters
    if (argc == 1)
        return 1;
    for(int i=1; i<argc; i++){
        if (!strcmp(argv[i],"-inline")){
            if(++i == argc)
                throw argument_exception("argument '-inline' expects an argument");
//            rnn->setCmdMode(1);
            strInput = argv[i];
            parse_args_codec_cmd(strInput);
            continue;
        }
        if (!strcmp(argv[i],"-debug")){
            rnn->setDebugMode(1);
            continue;
        }
//        if (!strcmp(argv[i],"-recursive")){
//            rnn->setRecursiveMode(1);
//            continue;
//        }
        if (!strcmp(argv[i],"-o")){
            if(++i == argc)
                throw argument_exception("argument '-o' expects an argument");
            rnn-> Out_File_Name = argv[i];
            rnn-> setQuietMode(0);
            continue;
        }
        if (!strcmp(argv[i],"-w")){
            if(++i == argc)
                throw argument_exception("argument '-w' expects an argument");
            rnn-> Weights_File_Name = argv[i];
            continue;
        }
        if (!strcmp(argv[i],"-h")){
            return 1;
        }
        else
            return 1;
    }
    if (rnn->getMode() == CRNN_T::IDLE)
        throw argument_exception("unspecified mode");
    rnn->loadInputCmd(strInput);

    return 0;
}

void printUsageCli(){
    cout<<"app_cli [options]"<<endl<<endl
        <<" -inline [INPUT_VECTOR]"<<endl
        <<"                 provide input parameter vectors here and generate output for it directly."<<endl
        <<"                 seperate different parameters in [INPUT_VECTOR] by ',' Example:"<<endl
        <<"                 	./app_cli -inline 264,1024,1.6,1"<<endl
        <<" -o [FILENAME_OUTPUT]"<<endl
        <<"                 Designate a filename to print the evaluation output instead of using cout directly."<<endl
        <<"                 Also deadtivate quiet mode so that some debug info might be printed to cout"<<endl
        <<" -w [FILENAME_WEIGHT]"<<endl
        <<"                 Override default \"weight.dat\" weight_matrix "<<endl
        <<" -h"<<endl
        <<"                 print help msg."<<endl;

}
