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

#include <iostream>
#include "src/crnn.h"
#include "src/crnn_t.h"
//
#include <cassert>
#include <string.h>
#include <fstream>
#include <sstream>
#include <vector>
//#include <cstring>

#include "src/argument_exception.h"
using namespace std;
using namespace rnn_lite;
void printUsage();
void printUsage2();
void parse_args_codec_cmd(string &strInput) throw (argument_exception);
int parse_args_codec(string strFilename) throw (argument_exception); //parse protocol phrase
int sub_parse_double(string strInput,double* dblOutput, int* pCount);
int parse_argsExt(rnn_lite::CRNN* rnn, int argc, vector<string> argv) throw (argument_exception);
int parse_args(rnn_lite::CRNN* rnn, int argc, char** argv) throw (argument_exception, rnn_exception);
void parse_args2(rnn_lite::CRNN* rnn, int argc, char** argv, double *pThreshold) throw (argument_exception);
int beta(int a, int b=1){
    return a+b;
}


//*************************************************************************************
//  ONLY for cmdline program with input data within the line
//  Obsolete.
//*************************************************************************************
void parse_args2(rnn_lite::CRNN* rnn, int argc, char** argv, double* pThreshold)throw(argument_exception)
{
    string strInput,strTarget, strArchitecture;
    for(int i=1; i<argc; i++){
        if (!strcmp(argv[i],"-train")){
            rnn->setMode(CRNN::TRAINING);
            continue;
        }
        if (!strcmp(argv[i],"-test")){
            rnn->setMode(CRNN::TEST);
            continue;
        }
        if (!strcmp(argv[i],"-recursive")){
            rnn->setRecursiveMode(1);
            continue;
        }
        if (!strcmp(argv[i],"-threshold")){
            if(++i == argc)
                throw argument_exception("argument '-threshold' expects an argument");
            stringstream ss(argv[i]);
            ss>>*pThreshold;
            continue;
        }
/*
        if (!strcmp(argv[i],"-continue")){
            rnn->setContinueMode(1);
            continue;
        }*/
        if (!strcmp(argv[i],"-i")){
            if(++i == argc)
                throw argument_exception("argument '-i' expects an argument");
            strInput = argv[i];
            continue;
        }
        if (!strcmp(argv[i],"-t")){
            if(++i == argc)
                throw argument_exception("argument '-t' expects an argument");
            strTarget = argv[i];
            continue;
        }
        if (!strcmp(argv[i],"-o")){
            if(++i == argc)
                throw argument_exception("argument '-o' expects an argument");
            rnn-> Out_File_Name = argv[i];
            continue;
        }
        if (!strcmp(argv[i],"-w")){
            if(++i == argc)
                throw argument_exception("argument '-w' expects an argument");
            rnn-> Weights_File_Name = argv[i];
            continue;
        }
        if (!strcmp(argv[i],"-p")){
            if(++i == argc)
                throw argument_exception("argument '-p' expects an argument");
            strArchitecture = argv[i];
            continue;
        }
        else{
            printUsage();
            throw argument_exception("invalid arguments");
        }

    }
    if (rnn->getMode() == CRNN::IDLE)
        throw argument_exception("unspecified mode");

}

//*************************************************************************************
//  ONLY for cmdline program with input data within the line
//  Obsolete.
//*************************************************************************************
void printUsage2(){
    cout<<"rnn_lite [options]"<<endl<<endl
        <<" -train          Specify training run."<<endl
        <<"    -continue    Continue training after previous iteration(load previous weight matrix as initial weights)"
        <<" -test           Specify test run."<<endl
        <<" -recursive      Specify Recursive RNN. Note that Hidden Node count should exceed input and/or output nodes"<<endl
        <<"                 Program use Feed Forward RNN as default setting"
        <<" -i [INPUT_VECTOR]"<<endl
        <<"                 Provide input for either train or test. "<<endl
        <<"                 [INPUT_VECTOR] take the form as 1.1,2.2,3.3,4.4"<<endl
        <<"                 Dimension must match with that provided by \"-p\" option"<<endl
        <<" -t [TARGET_VECTOR]"<<endl
        <<"                 Provide desired output for corresponding input data for train mode."<<endl
        <<"                 [TARGET_VECTOR] take the form as 1.1,2.2"<<endl
        <<"                 Dimension must match with that provided by \"-p\" option"<<endl
        <<" -p [TOPOLOGY_VECTOR]"<<endl
        <<"                 Provide RNN network topology specification."<<endl
        <<"                 [TOPOLOGY_VECTOR] take the form as 4,5,2 to indicate 4 input node, 5 hidden node "<<endl
        <<"                 and 2 outupt node"<<endl
        <<" -threshold [THRESHOLD]"<<endl
        <<"                 override default 0.0005 threshold. Accept float input"<<endl
        <<" -o [FILENAME_OUTPUT]"<<endl
        <<"                 Override default \"output.txt\" outfile filename for test run."<<endl;
}


//*************************************************************************************
//  Standard routine with input data bulk in a stored file
//
//*************************************************************************************
void printUsage(){
    cout<<"rnn_lite [options]"<<endl<<endl
        <<" -train          Specify training run."<<endl
        <<"    -compact     Specify training submode compact input while input and target are in the same file"<<endl
        <<"                !This mode is also valid in test run but the included target data WILL be ignored"<<endl
        <<" -test           Specify test run."<<endl
        <<" -recursive      Specify Recursive RNN. Note that Hidden Node count should exceed input and/or output nodes"<<endl
        <<"                 Program use Feed Forward RNN as default setting"<<endl
        <<" -inline [INPUT_VECTOR]"<<endl
        <<"                 provide input parameter vectors here and generate output for it directly."<<endl
        <<"                 seperate different parameters in [INPUT_VECTOR ]by ',' Example:"<<endl
        <<"                 	./rnn_lite -inline 264,1024,1.6,1"<<endl
        <<" -verbose        verbose output with additional debug info"<<endl
        <<" -i [FILENAME_TRAIN]"<<endl
        <<"                 Provide train set FILENAME. "<<endl
        <<"                 [FILENAME_INPUT] is a string such as train.txt"<<endl
        <<"                 Dimension must match with that provided by \"-p\" option"<<endl
        <<" -v [FILENAME_TEST]"<<endl
        <<"                 Provide test FILENAME "<<endl
        <<"                 [FILENAME_TEST] take the form as test.txt"<<endl
        <<"                 Dimension must match with that provided by \"-p\" option"<<endl
        <<" -t [FILENAME_TARGET]"<<endl
        <<"                 Provide desired output for corresponding input data for train mode."<<endl
        <<"                 [FILENAME_TARGET] take the form as TARGET.TXT"<<endl
        <<"                 Dimension must match with that provided by \"-p\" option"<<endl
        <<"                !DO NOT set this in COMPACT mode. Not supported"<<endl
        <<" -o [FILENAME_OUTPUT]"<<endl
        <<"                 Override default \"output.txt\" outfile filename for test run."<<endl
        <<" -w [FILENAME_WEIGHT]"<<endl
        <<"                 Override default \"weight.dat\" weight_matrix filename for train mode."<<endl
        <<" -p [TOPOLOGY_VECTOR]"<<endl
        <<"                 Provide RNN network topology specification."<<endl
        <<"                 [TOPOLOGY_VECTOR] take the form as 4,5,2 to indicate 4 input node, 5 hidden node "<<endl
        <<"                 and 2 outupt node"<<endl
        <<" -n [NORMALIZE_VECTOR]"<<endl
        <<"                 Provide input data normalization scale for BOTH input and output."<<endl
        <<"                 [NORMALIZE_VECTOR] use comma to seperate different scale. "<<endl
        <<"                 Dimension must match with that provided by \"-p\" option"<<endl
        <<" -m [MAX_ITERATION]"<<endl
        <<"                 override maximum iteration limit. 2000 by default"<<endl
        <<" -threshold [THRESHOLD]"<<endl
        <<"                 override default 0.0005 threshold. Accept float input"<<endl
        <<" -h"<<endl
        <<"                 print help msg."<<endl;

}


//*************************************************************************************
//  Standard routine with input data bulk in a stored file
//  Parse agrument, define network topology and normalize_vector
//*************************************************************************************
int parse_args(rnn_lite::CRNN* rnn, int argc, char** argv)throw(argument_exception, rnn_exception)
{
    double temp;
    int intTemp;
    int cntArg=0;
    int j=0;
    vector<string> vctArg;
    string strInput,strExternalArg;


    //load default parameters from external file "arg.conf"
    fstream file_arg;
    file_arg.open("arg.conf",ios::in);
    while (file_arg >> strExternalArg) {cntArg++;}
    if (argc==1 && cntArg ==0)      //no parameters, print help and exit
        return 1;
    vctArg.resize(cntArg);
    file_arg.clear();
    file_arg.seekg(0, ios::beg);
    while (file_arg >> strExternalArg) {
        vctArg[j]=strExternalArg;
        j++;
        }
    file_arg.close();
    if(parse_argsExt(rnn,cntArg,vctArg))
        return 1;

    //parse parameters
    for(int i=1; i<argc; i++){
        if (!strcmp(argv[i],"-inline")){
            if(++i == argc)
                throw argument_exception("argument '-inline' expects an argument");
            rnn->setCmdMode(1);
            strInput = argv[i];
            continue;
        }

        if (!strcmp(argv[i],"-debug")){
            rnn->setDebugMode(1);
            continue;
        }
        if (!strcmp(argv[i],"-train")){
            rnn->setMode(CRNN::TRAINING)  ;
            continue;
        }
        if (!strcmp(argv[i],"-test")){
            rnn->setMode(CRNN::TEST);
            continue;
        }
        if (!strcmp(argv[i],"-compact")){
            rnn->setCompactMode(1);
            continue;
        }
        if (!strcmp(argv[i],"-recursive")){
            rnn->setRecursiveMode(1);
            continue;
        }
        if (!strcmp(argv[i],"-verbose")){
            rnn-> setQuietMode(0);
            continue;
        }
        if (!strcmp(argv[i],"-i")){
            if(++i == argc)
                throw argument_exception("argument '-i' expects an argument");
            rnn-> Trn_File_Name = argv[i];
            parse_args_codec(rnn-> Trn_File_Name);
            continue;
        }
        if (!strcmp(argv[i],"-v")){
            if(++i == argc)
                throw argument_exception("argument '-v' expects an argument");
            rnn-> Tst_File_Name = argv[i];
            parse_args_codec(rnn-> Tst_File_Name);
            continue;
        }
        if (!strcmp(argv[i],"-t")){
            if(++i == argc)
                throw argument_exception("argument '-t' expects an argument");
            rnn-> Tgt_File_Name = argv[i];
            continue;
        }
        if (!strcmp(argv[i],"-o")){
            if(++i == argc)
                throw argument_exception("argument '-o' expects an argument");
            rnn-> Out_File_Name = argv[i];
            continue;
        }
        if (!strcmp(argv[i],"-w")){
            if(++i == argc)
                throw argument_exception("argument '-w' expects an argument");
            rnn-> Weights_File_Name = argv[i];
            continue;
        }
        if (!strcmp(argv[i],"-p")){
            if(++i == argc)
                throw argument_exception("argument '-p' expects an argument");
            rnn->strLayout = argv[i];
            continue;
        }
        if (!strcmp(argv[i],"-n")){ //normalization vector
            if(++i == argc)
                throw argument_exception("argument '-n' expects an argument");
            rnn->strNormalize = argv[i];
            continue;
        }
        if (!strcmp(argv[i],"-m")){
            if(++i == argc)
                throw argument_exception("argument '-m' expects an argument");
            stringstream ss(argv[i]);
            ss>>intTemp;
            rnn->setIteration(intTemp);
            continue;
        }
        if (!strcmp(argv[i],"-threshold")){
            if(++i == argc)
                throw argument_exception("argument '-threshold' expects an argument");
            stringstream ss(argv[i]);
            ss>>temp;
            rnn->setThreshold(temp);
            continue;
        }
        if (!strcmp(argv[i],"-h")){
            return 1;
        }
        else{
            char strTemp[128] = "invalid arguments:";
            throw argument_exception(strcat(strTemp,argv[i]));
        }
    }
    if (rnn->getMode() == CRNN::IDLE)
        throw argument_exception("unspecified mode");
    if (rnn->isCmdMode()){
        parse_args_codec_cmd(strInput);
        rnn->loadInputCmd(strInput);
    }

    return 0;
}

//*************************************************************************************
//  Similiar routine used to parse external arguments in "arg.conf"
//  Parse agrument, define network topology and normalize_vector
//*************************************************************************************
int parse_argsExt(rnn_lite::CRNN* rnn, int argc, vector<string> argv) throw (argument_exception){
    int intTemp;
    double dblTemp;
    for(int i=0; i<argc; i++){
/*
        if (argv[i]=="-inline"){
            if(++i == argc)
                throw argument_exception("argument '-inline' expects an argument");
            rnn->setCmdMode(1);
            strInput = argv[i];
        }
*/
        if (argv[i]=="-debug"){
            rnn->setDebugMode(1);
            continue;
        }
        if (argv[i]=="-train"){
            rnn->setMode(CRNN::TRAINING)  ;
            continue;
        }
        if (argv[i]=="-test"){
            rnn->setMode(CRNN::TEST);
            continue;
        }
        if (argv[i]=="-compact"){
            rnn->setCompactMode(1);
            continue;
        }
        if (argv[i]=="-recursive"){
            rnn->setRecursiveMode(1);
            continue;
        }
        if (argv[i]=="-verbose"){
            rnn-> setQuietMode(0);
            continue;
        }
        if (argv[i]=="-i"){
            if(++i == argc)
                throw argument_exception("argument '-i' expects an argument");
            rnn-> Trn_File_Name = argv[i];
            parse_args_codec(rnn-> Trn_File_Name);
            continue;
        }
        if (argv[i]=="-v"){
            if(++i == argc)
                throw argument_exception("argument '-v' expects an argument");
            rnn-> Tst_File_Name = argv[i];
            parse_args_codec(rnn-> Tst_File_Name);
            continue;
        }
        if (argv[i]=="-t"){
            if(++i == argc)
                throw argument_exception("argument '-t' expects an argument");
            rnn-> Tgt_File_Name = argv[i];
            continue;
        }
        if (argv[i]=="-o"){
            if(++i == argc)
                throw argument_exception("argument '-o' expects an argument");
            rnn-> Out_File_Name = argv[i];
            continue;
        }
        if (argv[i]=="-w"){
            if(++i == argc)
                throw argument_exception("argument '-w' expects an argument");
            rnn-> Weights_File_Name = argv[i];
            continue;
        }
        if (argv[i]=="-p"){
            if(++i == argc)
                throw argument_exception("argument '-p' expects an argument");
            rnn->strLayout = argv[i];
            continue;
        }
        if (argv[i]=="-n"){ //normalization vector
            if(++i == argc)
                throw argument_exception("argument '-n' expects an argument");
            rnn->strNormalize = argv[i];
            continue;
        }
        if (argv[i]=="-m"){
            if(++i == argc)
                throw argument_exception("argument '-m' expects an argument");
            stringstream ss(argv[i]);
            ss>>intTemp;
            rnn->setIteration(intTemp);
            continue;
        }
        if (argv[i]=="-threshold"){
            if(++i == argc)
                throw argument_exception("argument '-threshold' expects an argument");
            stringstream ss(argv[i]);
            ss>>dblTemp;
            rnn->setThreshold(dblTemp);
            continue;
        }
        if (argv[i]=="-h"){
            return 1;
        }
        else{
            printUsage();
            throw argument_exception("invalid arguments "+argv[i]);
        }
    }
    return 0;
}

//*************************************************************************************
//  Sub-routine for parse_args()
//  preprocess protocol string, transform them into float [0,1],
//  override normalize_vector as well.
//  In:     string strIn    --filename of the file to be processed.
//  return: column index of protocol data, used to override normalize_vector
//          0 if no such column exists.
//*************************************************************************************
int parse_args_codec(string strFilename)  throw (argument_exception){
    fstream fileIn, fileOut, fileCodec;
    string strPostfix=".dat";
    int intInputSize, intEntrySize, intCodecSize;
    int intEntryCount = 0;;
    int intFound = -1;
    stringstream ss;
    string strDebug;
    string strTemp,strTemp2,strTemp3;

    //open corresponding files
    fileIn.open(strFilename.c_str(),ios::in);
    fileOut.open((strFilename+=strPostfix).c_str(),ios::out|ios::trunc);
    fileCodec.open("codec.conf",ios::in);
    if(!fileCodec.is_open())
        throw argument_exception("failed to open codec conf file");
    if(!fileIn.is_open())
        throw argument_exception("failed to open test data file");

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

    intEntryCount = 0;
    getline(fileIn,strTemp);
    ss.clear();
    ss<<strTemp;
    ss>>intInputSize;
    ss>>intEntrySize;
    fileOut<<strTemp;
    while(getline(fileIn,strTemp)){ //for every line of raw data
        if (strTemp == "")          //ignore empty line
            continue;
        for(int i=0;i<intCodecSize;i++){    //for every codec
//            ss.clear();
//            strDebug = strCodecArray[i];
//            ss<<strCodecArray[i];
//            ss>>strDebug;
            while(intFound < 0 && ssc[i]>>strTemp2){    //for every variation of certain codec
                intFound = strTemp.find(strTemp2);
            }
            ssc[i].clear();
            ssc[i].seekg(0,ios::beg);
            if(intFound >= 0){      //found codec column, match i'th codec
                ss.clear();
                ss<<0.1+0.2*i;
                ss>>strTemp3;
                strTemp.replace(intFound,strTemp2.length(),strTemp3);
                intFound = -1;
                break;              //no need to check other codec, continue with next raw data
            }
        }
        fileOut<<endl<<strTemp;
        intEntryCount++;
    }
    if (intEntryCount != intEntrySize)
        throw argument_exception("inconsistent input entries count.");
    fileIn.close();
    fileOut.close();
    return 0;

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

int main(int argc, char** argv)
{
try{
    double MSEthreshold = 0.0003;
    CRNN rnn;

    rnn.dbgPrint("::reading arguments...");

    if(1==parse_args(&rnn,argc,argv)){  //print help, normal exit
        printUsage();
        return 0;
    }

    if(rnn.isDebugMode()){
        fstream fileDebug;
        fileDebug.open("shaped.txt",ios::out|ios::trunc);
        int length = 400;
        int check = 50;
        for(int i=0;i<length;i++){
            fileDebug<<"H.264\t";
            fileDebug<<"1024\t";
//            fileDebug<<"5\t";
//            fileDebug<<"2"<<endl;
            if (i>check){
                double min = 0;
                double max = 35;
                double val = min+(double)(i-check)/(double)(length-check)*(double)(max-min);
                fileDebug<<val<<"\t";
            }
            else
                fileDebug<<"1\t";
//            fileDebug<<"5\t"<<endl;
//            fileDebug<<"5\t";
            fileDebug<<"10"<<endl;
        }
        return 0;
    }

//    if(rnn.isDebugMode()){
//    if(0){
//        CSEQ abc(8);
//        abc.work();
//        rnn.printMemoryI((unsigned int) abc.output,8,8);
//        return 0;
//    }
//    }


    rnn.dbgPrint("::argument loaded, press any key to load input, ctrl+c to break\n");
//    getchar();
    if(rnn.isCmdMode()){
        rnn.test();
        return 0;
    }

    if(rnn.loadInput())
        throw rnn_exception("train set inconsistency");

    rnn.dbgPrint("::input file loaded, press any key to train/load, ctrl+c to break");
//    getchar();



    if (rnn.getMode()==CRNN::TRAINING)
        rnn.train(&MSEthreshold,0);
    else if(rnn.getMode()==CRNN::TEST)
        rnn.test();
    //rnn.train(&MSEthreshold);
//    if (rnn.file_tst.is_open()){
//        cout<<"::testing"<<endl;
//        rnn.test();
//    }

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


