#include "crnn.h"
#include "cseq.h"
#include <sstream>
#include "string.h"
#include <stdlib.h>
#include <cmath>

using namespace std;
using namespace rnn_lite;

CRNN::CRNN(){
    dbgPrint("building CRNN");

    M_COMPACT = 0;
//    M_CONTINUE = 0;
    M_CMD = 0;

    N_MaxIterations = 2000;                         //Maximum number of Iterations
    N_TrainPatternCount = 0;

    TARGET  = NULL;
    Applied_lambda  = NULL;
    Applied_LAMBDA  = NULL;
    y   = NULL;
    Applied_y   = NULL;
    TRAIN_INPUT = NULL;
//!*****************************************************************
    Mse_Threshold  = .0005;                     //Required stop mean square error value
//    Eta = 0.1;                                  //Learinig Rate
    Eta = 0.0125;                                  //Learinig Rate


    RAND_RANGE = 0.2;                           //Range of weights initialization
//    RAND_RANGE = 0.8;
//    FIX_RIN = 0;                                //DO NOT CHANGE (Related to RNN function approximation)
    N_Saved_Iterations = 100;                    //Number of ²iterations after which weights will be saved automatically

    Trn_File_Name = "train.txt";
    Tst_File_Name = "test.txt";
    Tgt_File_Name = "target.txt";
    Out_File_Name = "output.txt";

    strLayout = "";
    strNormalize = "";

    MSEaveg = 0;
}

CRNN::~CRNN(){
    dbgPrint("destructing CRNN");
    delete[] TRAIN_INPUT;
    delete[] TARGET;
    delete[] Applied_LAMBDA;
    delete[] Applied_lambda;
    if(file_trn.is_open())
        file_trn.close();
    if(file_tgt.is_open())
        file_tgt.close();
    if(file_tst.is_open())
        file_tst.close();
    if(file_out.is_open())
        file_out.close();
    remove((Tst_File_Name+".dat").c_str());
    remove((Trn_File_Name+".dat").c_str());
}
/*
void CRNN::setContinueMode(int i){
    if (i>0)
        M_CONTINUE = 1;
    else
        M_CONTINUE = 0;
}
*/
void CRNN::setCompactMode(int i){
    if (i>0)
        M_COMPACT = 1;
    else
        M_COMPACT = 0;
}
void CRNN::setCmdMode(int i){
    if (i>0)
        M_CMD = 1;
    else
        M_CMD = 0;
}

int CRNN::initialize_architecture(string strInput){
//!     in:   string
//!     out:  N_InputSize, N_OutputSize, N_HiddenSize, N_Total,
//!           N_AllLayerCount, N_LayerArray[]
//!
//!    take string in the form as "5.7.2"
//!    interpret as layer specification
//!    output array as a[]={5,7,2,0}
//!    0 is added to its tail for easier loop control

    size_t intPos = 0;
    size_t intPosP = 0;
    int intSize = 0;
    int intSum = 0;
    int temp;

    // check layer count
    while(intPos < strInput.size()){
        intSize ++;
        intPos++;
        intPos = strInput.find(',',intPos);
    }
    if(intSize < 3 && !M_RECURSIVE){
        printf("argument '-p' invalid: FF RNN requires at least 3 layers");
        return 1;
    }
    N_AllLayerCount = intSize;
    N_LayerArray = new int[intSize + 1];

    //parse the sentence
    intPos = -1;
    for(int i=0;i<intSize;i++){
        intPos++;
        intPosP = intPos;
        intPos = strInput.find(',',intPos);
        stringstream ss(strInput.substr(intPosP,intPos - intPosP));
        ss>>temp;
        if(i == 0 && N_InputSize > 0 && N_InputSize != temp){
            printf("CRNN::initialize_architecture() error: Inconsistent Input layer count\n");
            return 1;
        }
        else if (i == intSize - 1 && N_OutputSize > 0 && N_OutputSize != temp){
            printf("CRNN::initialize_architecture() error: Inconsistent Output layer count\n");
            return 1;
        }
        N_LayerArray[i] = temp;
        intSum += N_LayerArray[i];
//        cout<<intSum<<endl;
    }

    //************************************************************
    //In recursive mode, take input 4.4.1 to build 4X4 weight matrix
    //************************************************************
    if(M_RECURSIVE){
        N_LayerArray[intSize] = 0;
        N_InputSize = N_LayerArray[0];
        N_OutputSize = N_LayerArray[intSize - 1];
        N_Total = intSum - N_InputSize - N_OutputSize;  //Assuming N_Hidden>N_input & >N_output
        N_HiddenSize = N_Total;
        if(N_InputSize > N_Total){
            printf("error: More input nodes than available nodes\n");
            return 1;
        }
        FLAG |= M_TOPOLOGY;
        return 0;
    }
    N_LayerArray[intSize] = 0;
    N_InputSize = N_LayerArray[0];
    N_OutputSize = N_LayerArray[intSize - 1];
    N_Total = intSum;
    N_HiddenSize = N_Total - N_InputSize - N_OutputSize;
    FLAG |= M_TOPOLOGY;
    printMemoryI(N_LayerArray,N_AllLayerCount+1,N_AllLayerCount+1);
    return 0;
}

int CRNN::initialize_normalization_vector(string strInput){
    unsigned int intPos = 0;
    int intSize = 0;
    int intPosP;
    double temp;
    Normalize_Vector = new double[N_InputSize+N_OutputSize];

    // check vector dimension
    // for "1,2,3" will result in
    // intSize=3
    while(intPos < strInput.size()){
        intSize ++;
        intPos++;
        intPos = strInput.find(',',intPos);
    }
    if(intSize != N_InputSize + N_OutputSize){
//        delete[] Normalize_Vector;
        printf("'-n' dimension mismatch");
        return 1;
    }

    //parse the sentence
    dbgPrint("*****************normalize vector:");
    intPos = -1;
    for(int i=0;i<intSize;i++){
        intPos++;
        intPosP = intPos;
        intPos = strInput.find(',',intPos);
        stringstream ss(strInput.substr(intPosP,intPos - intPosP));
        ss>>temp;
        Normalize_Vector[i] = temp;
    }
    printMemoryD( Normalize_Vector,N_InputSize+N_OutputSize,N_InputSize+N_OutputSize);
    FLAG |= M_NORMALIZE;
    return 0;
}

int CRNN::initialize_weights()
//!  in:    N_Total
//!  out:   wplus, wminus
//!  mem allocated: wplus, wminus,

{   if (FLAG & M_WEIGHT){   //weight matrix re-enter
        delete[] wplus;
        delete[] wminus;
    }

    int i,j;
    wplus = new double[N_Total*N_Total];
    wminus = new double[N_Total*N_Total];
    memset(wplus,0,sizeof(double)* N_Total*N_Total );
    memset(wminus,0,sizeof(double)* N_Total*N_Total );
    /*Random staff here!!!*/
    //
    //
    //
    printMemoryI(N_LayerArray,N_AllLayerCount+1,N_AllLayerCount+1);
    int intCount,uu,index_start;
    int *p = N_LayerArray;
    index_start = *p;
    uu = 0;
    if(M_RECURSIVE){
        for(i=0;i<N_Total;i++)
            for(j=0;j<N_Total;j++)
                if(i!=j){
//                {
                    *(wplus+N_Total*i+j) = ((double)(rand()%((int)(RAND_RANGE*10000))))/10000;
                    *(wminus+N_Total*i+j) = ((double)(rand()%((int)(RAND_RANGE*10000))))/10000;
                }
//        printMemoryD( wplus,N_Total*N_Total,N_Total);
        FLAG |= M_WEIGHT;
        return 0;
    }
    else
    for(i=0;i<N_Total;i++){
         intCount= 0;
        for(j=index_start;intCount<*(p+1);j++){
//            printf("u=%d\tv=%d\n",i,j);
            *(wplus+N_Total*i+j) = ((double)(rand()%((int)(RAND_RANGE*10000))))/10000;
            *(wminus+N_Total*i+j) = ((double)(rand()%((int)(RAND_RANGE*10000))))/10000;
            intCount++;
        }
        uu++;
        if(uu==*p){
            p++;
            index_start += *p;
            uu = 0;
        }

    }
/*
    for(i=0;i<N_InputSize+N_HiddenSize;i++){
        if (i<N_InputSize)
            for(j=N_InputSize;j<N_InputSize+N_HiddenSize;j++)
                *(wplus+N_Total*i+j) = ((double)(rand()%((int)(RAND_RANGE*10000))))/10000;
        else if(i>= N_InputSize && i< N_InputSize+N_HiddenSize)
            for(j=N_InputSize + N_HiddenSize;j<N_Total;j++)
                *(wplus+N_Total*i+j) = ((double)(rand()%((int)(RAND_RANGE*10000))))/10000;
    }
    for(i=0;i<N_InputSize+N_HiddenSize;i++){
        if (i<N_InputSize)
            for(j=N_InputSize;j<N_InputSize+N_HiddenSize;j++)
                *(wminus+N_Total*i+j) = ((double)(rand()%((int)(RAND_RANGE*10000))))/10000;
        else if(i>= N_InputSize && i< N_InputSize+N_HiddenSize)
            for(j=N_InputSize + N_HiddenSize;j<N_Total;j++)
                *(wminus+N_Total*i+j) = ((double)(rand()%((int)(RAND_RANGE*10000))))/10000;
    }
*/

/*
    *(wplus+14*0+7) = 0;      *(wplus+14*0+8) = 0.2711;   *(wplus+14*0+9) = 0.0781;    *(wplus+14*0+10) = 0.1177;      *(wplus+14*0+11) = 0;
    *(wplus+14*1+7) = 0.0962;  *(wplus+14*1+8) = 0;        *(wplus+14*1+9) = 0;         *(wplus+14*1+10) = 0.1647;      *(wplus+14*1+11) = 0.2183;
    *(wplus+14*2+7) = 3;       *(wplus+14*2+8) = 0.2762;   *(wplus+14*2+9) = 0.2328;    *(wplus+14*2+10) = 0.0108;      *(wplus+14*2+11) = 0;
    *(wplus+14*3+7) = 3;       *(wplus+14*3+8) = 0.1675;   *(wplus+14*3+9) = 0.2276;    *(wplus+14*3+10) = 0.0844;      *(wplus+14*3+11) = 0.0310;
    *(wplus+14*4+7) = 0.2105;  *(wplus+14*4+8) = 0;        *(wplus+14*4+9) = 0.0669;    *(wplus+14*4+10) = 0.1689;      *(wplus+14*4+11) = 0.1677;
    *(wplus+14*5+7) = 0.2258;  *(wplus+14*5+8) = 0;        *(wplus+14*5+9) = 0;         *(wplus+14*5+10) = 0.1160;      *(wplus+14*5+11) = 0.0643;
    *(wplus+14*6+7) = 0.1971;  *(wplus+14*6+8) = 0;        *(wplus+14*6+9) = 0;         *(wplus+14*6+10) = 0;           *(wplus+14*6+11) = 0.0240;

    *(wplus+14*7+12) = 0.2783; *(wplus+14*7+13) = 0;
    *(wplus+14*8+12) = 0;      *(wplus+14*8+13) = 0.2778;
    *(wplus+14*9+12) = 0;      *(wplus+14*9+13) = 0.0994;
    *(wplus+14*10+12) = 0.0608; *(wplus+14*10+13) = 0.0000;
    *(wplus+14*11+12) = 0.1820; *(wplus+14*11+13) = 0.0000;

    *(wminus+14*0+7) = 0.1263;  *(wminus+14*0+8) = 0.0540;   *(wminus+14*0+9) = 0.1169;    *(wminus+14*0+10) = 0.1730;      *(wminus+14*0+11) = 0.2095;
    *(wminus+14*1+7) = 0.0123;  *(wminus+14*1+8) = 0.1531;   *(wminus+14*1+9) = 0.1626;    *(wminus+14*1+10) = 0.0462;      *(wminus+14*1+11) = 0.0093;
    *(wminus+14*2+7) = 0.1148;  *(wminus+14*2+8) = 0.1382;   *(wminus+14*2+9) = 0.0391;    *(wminus+14*2+10) = 0.1353;      *(wminus+14*2+11) = 0.0924;
    *(wminus+14*3+7) = 0.1704;  *(wminus+14*3+8) = 0.0065;   *(wminus+14*3+9) = 0.0801;    *(wminus+14*3+10) = 0.1626;      *(wminus+14*3+11) = 0.2289;
    *(wminus+14*4+7) = 0.1592;  *(wminus+14*4+8) = 0.1331;   *(wminus+14*4+9) = 0.0651;    *(wminus+14*4+10) = 0.0503;      *(wminus+14*4+11) = 0.0451;
    *(wminus+14*5+7) = 0.1391;  *(wminus+14*5+8) = 0.0737;   *(wminus+14*5+9) = 0.1982;    *(wminus+14*5+10) = 0.0687;      *(wminus+14*5+11) = 0.1815;
    *(wminus+14*6+7) = 0.0063;  *(wminus+14*6+8) = 0.0357;   *(wminus+14*6+9) = 0.0610;    *(wminus+14*6+10) = 0.1161;      *(wminus+14*6+11) = 0.0595;

    *(wminus+14*7+12) = 0.0136; *(wminus+14*7+13) = 0.2523;
    *(wminus+14*8+12) = 0.2641; *(wminus+14*8+13) = 0.0305;
    *(wminus+14*9+12) = 0.3308; *(wminus+14*9+13) = 0.0881;
    *(wminus+14*10+12) = 0.2304;*(wminus+14*10+13) = 0.2863;
    *(wminus+14*11+12) = 0.2101;*(wminus+14*11+13) = 0.2501;
*/
//    printMemoryD( wplus,N_Total*N_Total,N_Total);
//    printMemory( wminus,N_Total*N_Total);
    FLAG |= M_WEIGHT;
    return 0;
}

int CRNN::save_weights(){
//! in:     Weights_File_Name, N_Total, N_AllLayerCount, N_LayerArray
//! out:    files on disk
//!  mem allocated: none
//weight file layout as follow:
//      | M_RECURSIVE | N_AllLayerCount  | N_LayerArray[N_AllLayerCount+1] | Normalize_Vector[N_InputSize+N_OutputSize] |
//      | wplus[N_Total*N_Total] | wminus[N_Total*N_Total] |
//
    char *p1;
    file_weight.open(Weights_File_Name.c_str(),ios::out|ios::binary|ios::trunc);
    file_weight.seekp(0,ios::beg);
    //save topology info
    p1 = (char*)&M_RECURSIVE;
    file_weight.write(p1,sizeof(int));
    p1 = (char*)&N_AllLayerCount;
    file_weight.write(p1,sizeof(int));
    p1 = (char*)N_LayerArray;
    file_weight.write(p1,sizeof(int)*(N_AllLayerCount+1));
    //save normalize vector
    p1 = (char*)Normalize_Vector;
    file_weight.write(p1,sizeof(double)*(N_InputSize+N_OutputSize));
    //save weights
    p1 = (char*)wplus;
    file_weight.write(p1,sizeof(double)*N_Total*N_Total);
    p1 = (char*)wminus;
    file_weight.write(p1,sizeof(double)*N_Total*N_Total);
    file_weight.close();
    return 0;
}

int CRNN::loadInput() throw (rnn_exception,argument_exception)
//!  in:   test.txt train.txt target.txt output.txt weight.dat
//!  out:  Corresponding file
//!        --nonstream-- TRAIN_INPUT, TEST_INPUT, TARGET,
//!  sub:  initialize_weight();
{
    int temp;
    double dblTemp;

    int intEntryCounter=0;
    int intItemIndex=0;
    double *p,*pTrain,*pTarget,*pTest;
    double dummy;

    //**********************************
    //define RNN architecture and normalize vector
    //load/randomize weights
    //During test run, most initiation is done by load_weights()
    //**********************************
    if(mode == TEST){
        load_weights();
    }

    else if(mode == TRAINING){
        if(initialize_architecture(strLayout))
            throw rnn_exception("argument '-p' caused error during RNN topology setup");
        if(initialize_normalization_vector(strNormalize))
            throw rnn_exception("argument '-n' caused error during Normalize_Vector setup");
         if(initialize_weights())
            throw rnn_exception("failed to initialize weight matrix in initialize_weights()");
//    printMemoryD(wplus,N_Total * N_Total,N_Total);
//    cout<<endl;
//    printMemoryD(wminus,N_Total * N_Total,N_Total);
    }




    //**********************************
    //check conflict
    //**********************************
/*    if (M_CONTINUE){
        if (mode != TRAINING)
            throw rnn_exception("argument '-c' failed: Not valid for Non-training mode");
        else if(M_COMPACT)
            throw rnn_exception("Undefined argument combination: -continue -compact");
    }*/
    // Input format must be pre-defined to correctly recognize input pattern inside INPUT file
    if (N_InputSize<1 || N_OutputSize <1 || N_HiddenSize <1)
        throw rnn_exception("invalid RNN architecture: Plz verify '-p' parameter");



    //**********************************
    //read input file
    //prioritize pre-processed *.dat
    //header is mandatory
    //**********************************
    file_tst.open((Tst_File_Name+".dat").c_str(),ios::in);
    if (!file_tst.is_open())
        file_tst.open(Tst_File_Name.c_str(),ios::in);
    if(mode == TRAINING){
        file_trn.open((Trn_File_Name+".dat").c_str(),ios::in);
        if (!file_trn.is_open())
            file_trn.open(Trn_File_Name.c_str(),ios::in);
    }
    if(mode == TRAINING && !M_COMPACT)
        file_tgt.open(Tgt_File_Name.c_str(),ios::in);
    if(mode==TRAINING && !file_trn.is_open())
        throw rnn_exception("failed to open train data file");
    if(mode==TRAINING && !M_COMPACT && !file_tgt)
        throw rnn_exception("failed to open target data file");
    if(mode==TEST && !file_tst.is_open())
        throw rnn_exception("failed to open test data file");


    //compact mode: training and target in a single file;
    if (M_COMPACT){
        if(file_tst.is_open()){
            file_tst>>temp;
            if(N_InputSize>0 && N_InputSize!=temp){
                throw rnn_exception("loadinput error: inconsistent Input size");
            }
            N_InputSize = temp ;
            file_tst>>N_TestPatternCount;
            pos_data_tst = file_tst.tellg();
            TEST_INPUT = new double[N_InputSize * N_TestPatternCount];
            memset(TEST_INPUT,0,sizeof(double)* N_InputSize*N_TestPatternCount);
            intEntryCounter = 0;
            intItemIndex = 0;
            pTest = TEST_INPUT;
            p = pTest;

            while (file_tst >> dblTemp) {
                if (intEntryCounter >= (N_InputSize+N_OutputSize)*N_TestPatternCount){   //boundary check
                    throw rnn_exception("test set exceed claimed size");
                }
                dblTemp = dblTemp/Normalize_Vector[intItemIndex];
                *p = dblTemp;
                p++;
                intItemIndex++;
                intEntryCounter++;
                if(intItemIndex==N_InputSize){          //swap receiver pointer, drop validation data
                    pTest = p;
                    p = &dummy;
                }
                if(intItemIndex==N_InputSize+N_OutputSize){//swap receiver pointer back to testdata buffer
                    p = pTest;
                }
                intItemIndex = intItemIndex%(N_InputSize+N_OutputSize);
            }
            if (intItemIndex!=0)
                throw rnn_exception("Mismatch between test dimension and RNN architecture");
            if (intEntryCounter != (N_InputSize+N_OutputSize)*N_TestPatternCount)
                throw rnn_exception("warning!!! test set pattern count inconsistency");
            FLAG |= M_TESTINPUT |M_VALIDATE;
        }
        if(file_trn.is_open()){
            file_trn>>temp;
            if(N_InputSize>0 && N_InputSize!=temp){
                throw rnn_exception("loadinput error: inconsistent Input size");
            }
            N_InputSize = temp ;
            file_trn>>N_TrainPatternCount;
            pos_data_trn = file_trn.tellg();
            TRAIN_INPUT = new double[N_InputSize*N_TrainPatternCount];
            TARGET = new double[N_OutputSize * N_TrainPatternCount];
            memset(TRAIN_INPUT,0,sizeof(double)* N_InputSize*N_TrainPatternCount);
            memset(TARGET,0,sizeof(double)* N_OutputSize*N_TrainPatternCount);
            intEntryCounter = 0;
            intItemIndex = 0;
            pTrain  = TRAIN_INPUT;
            pTarget = TARGET;
            p = pTrain;

            while (file_trn >> dblTemp) {
//                if(p==TRAIN_INPUT+N_InputSize*N_TrainPatternCount){
                if(intEntryCounter >= (N_InputSize+N_OutputSize)*N_TrainPatternCount){  //Train buffer boundary check
                    throw rnn_exception("train set exceed claimed size");
                }
                dblTemp = dblTemp/Normalize_Vector[intItemIndex];
                *p = dblTemp;
                p++;
                intItemIndex++;
                intEntryCounter++;
                if(intItemIndex==N_InputSize){  //swap receiver pointer to Targetdata buffer
                    pTrain = p;
                    p = pTarget;
                }
                if(intItemIndex==N_InputSize+N_OutputSize){ //swap receiver pointer to Traindata buffer
                    pTarget = p;
                    p = pTrain;
                }
                intItemIndex = intItemIndex%(N_InputSize+N_OutputSize);
            }
            if (intItemIndex!=0){       // if data stoped in middle of a record...
                throw rnn_exception("Mismatch between input dimension and RNN architecture");
                return 1;
            }
            if (intEntryCounter != (N_InputSize+N_OutputSize)*N_TrainPatternCount)
                throw rnn_exception("warning!!! train set pattern count inconsistency");
            FLAG = FLAG| M_TRAININPUT |M_TARGET;
        }
//        else
//            throw rnn_exception("Input file open failed");
    }

    //standard mode, training set and/or target provided in seperate files
    if((!M_COMPACT) && file_trn.is_open()){
        file_trn>>temp;
        if(N_InputSize>0 && N_InputSize!=temp)
            throw rnn_exception("CRNN::loadInput error: inconsistent Input size in train data file");
        N_InputSize = temp ;
        file_trn>>N_TrainPatternCount;
        pos_data_trn = file_trn.tellg();
        TRAIN_INPUT = new double[N_InputSize*N_TrainPatternCount];
        memset(TRAIN_INPUT,0,sizeof(double)* N_InputSize*N_TrainPatternCount);
        intEntryCounter = 0;
        intItemIndex = 0;
        p = TRAIN_INPUT;
        while (file_trn >> dblTemp) {
            if (intEntryCounter >= N_InputSize*N_TrainPatternCount){
                throw rnn_exception("train set exceeds claimed size");
            }
            dblTemp = dblTemp/Normalize_Vector[intItemIndex];
            *p = dblTemp;
            p++;
            intEntryCounter++;
            intItemIndex++;
            intItemIndex = intItemIndex%(N_InputSize);
        }
//           cout<<"counter: ";
//            cout<<intEntryCounter<<endl;
        if (intEntryCounter != N_InputSize*N_TrainPatternCount){
//               throw rnn_exception("train set inconsistency");
            return 1;
        }
        FLAG |= M_TRAININPUT;
    }
    if((!M_COMPACT) && file_tst.is_open()){
        file_tst>>temp;
        if(N_InputSize>0 && N_InputSize!=temp)
            cout<<"loadinput error: inconsistent Input size"<<endl;
        N_InputSize = temp;
        file_tst>>N_TestPatternCount;
        pos_data_tst = file_tst.tellg();
        TEST_INPUT = new double[N_InputSize * N_TestPatternCount];
        memset(TEST_INPUT,0,sizeof(double)* N_InputSize*N_TestPatternCount);
        intEntryCounter = 0;
        intItemIndex = 0;
        p = TEST_INPUT;
        while (file_tst >> dblTemp) {
            if (intEntryCounter >= N_InputSize*N_TestPatternCount){
                throw rnn_exception("test set exceeded claimed size");
            }
            dblTemp = dblTemp/Normalize_Vector[intItemIndex];
            *p = dblTemp;
            p++;
            intEntryCounter++;
            intItemIndex++;
            intItemIndex = intItemIndex%(N_InputSize);
        }
        if (intEntryCounter != N_InputSize*N_TestPatternCount){
            throw rnn_exception("test set inconsistency");
        }
        FLAG |= M_TESTINPUT;

    }
    if((!M_COMPACT) && file_tgt.is_open()){
        file_tgt>>temp;
        if(N_OutputSize>0 && N_OutputSize!=temp)
            cout<<"loadinput error: inconsistent Output size"<<endl;
        N_OutputSize = temp;
        file_tgt>>N_TrainPatternCount;
        pos_data_tgt = file_tgt.tellg();
        TARGET = new double[N_OutputSize * N_TrainPatternCount];
        memset(TARGET,0,sizeof(double)* N_OutputSize*N_TrainPatternCount);
        intEntryCounter = 0;
        intItemIndex = 0;
        p = TARGET;
        while (file_tgt >> dblTemp) {
            if(p>=TARGET+N_OutputSize*N_TrainPatternCount){
                throw rnn_exception("target data exceeded claimed size");
            }
            dblTemp = dblTemp/Normalize_Vector[N_InputSize+intItemIndex];
            *p = dblTemp;
            p++;
            intEntryCounter++;
            intItemIndex++;
            intItemIndex = intItemIndex%(N_OutputSize);

        }
//            cout<<"counter: ";
//            cout<<intEntryCounter<<endl;
        if (intEntryCounter != N_OutputSize*N_TrainPatternCount)
            throw rnn_exception("target data inconsistency");
        FLAG |= M_TARGET;
    }

        //!    other allocations were done already during read_file
        //!
        //!
    OUTPUT = new double[N_OutputSize * N_TestPatternCount];

    Applied_lambda = new double[N_InputSize * N_TrainPatternCount]; //this is for train run
    Applied_LAMBDA = new double[N_InputSize * N_TrainPatternCount];
    Input_lambda = new double[N_InputSize * N_TestPatternCount];    //this is for test run
    Input_LAMBDA = new double[N_InputSize * N_TestPatternCount];

    N = new double[N_Total];
    D = new double[N_Total];
    q = new double[N_Total];
    r = new double[N_Total];
    W = new double[N_Total * N_Total];
    memset(r,0,sizeof(double)* N_Total );
    memset(W,0,sizeof(double)* N_Total * N_Total);

    dbgPrint("train*********************");
    printMemoryD(TRAIN_INPUT,N_InputSize*N_TrainPatternCount,N_InputSize);
    dbgPrint("test**********************");
    printMemoryD(TEST_INPUT,N_InputSize*N_TestPatternCount,N_InputSize);
    dbgPrint("target*********************");
    printMemoryD(TARGET,N_OutputSize*N_TrainPatternCount,N_OutputSize);

/*
    p=TRAIN_INPUT;
    *p=-1;
    *(p+1)=1;
    *(p+2)=-1;
    *(p+3)=-1;
    *(p+4)=1;
    *(p+5)=1;
    *(p+6)=1;
    *(p+7)=1;
    *(p+8)=-1;
    *(p+9)=1;
    *(p+10)=1;
    *(p+11)=-1;
    *(p+12)=-1;
    *(p+13)=-1;


    p=TARGET;
    *p=1;
    *(p+1)=0;
    *(p+2)=0;
    *(p+3)=1;

    p = TEST_INPUT;
    *p=0.01;
    *(p+1)=-0.01;
    *(p+2)=-0.01;
    *(p+3)=-0.01;
    *(p+4)=0.01;
    *(p+5)=0.01;
    *(p+6)=0.01;
*/
    return 0;
}


double CRNN::test() throw (rnn_exception)
//  in:    wplus, wminus, R_OUT, LAMBDA, lambda
//  out:   r, q
//
{   if ((FLAG&0X1F) != 0X1F)
        throw rnn_exception("CRNN::test() invalid state, check input parameters");
    int ret,k;
    file_out.open(Out_File_Name.c_str(),ios::out|ios::trunc);
    prepare_tst_patterns();
    ret = calculate_rate();

//        printMemoryD(LAMBDA,N_InputSize,N_InputSize);
//        printMemoryD(lambda,N_InputSize,N_InputSize);
//        printMemoryD(r,N_Total,N_Total);
//        printMemoryD(wplus,N_Total*N_Total,N_Total);
//        cout<<endl;
//        printMemoryD(wminus,N_Total*N_Total,N_Total);
    for (int j=0;j<N_TestPatternCount;j++){
//          k = Z(j);       //Z stores sample selection masks and preferences
        k = j;
        if(M_RECURSIVE)
            ret = solve_nonlinear_equations(k);
        else
            ret = calculate_ffm_output(k);
//                printMemoryD(q+N_Total*k,N_Total,N_Total);
//                cout<<endl;
/*
//!!!!!!!!!!! loop 2
                memset(Input_lambda,0,sizeof(double)* N_InputSize * N_TestPatternCount );
                memset(Input_LAMBDA,0,sizeof(double)* N_InputSize * N_TestPatternCount );
                memcpy(Input_LAMBDA,q,sizeof(double)* N_Total*N_TestPatternCount);
*/
        for(int i =0;i<N_OutputSize;i++){
            OUTPUT[N_OutputSize*k+i] = q[i+N_Total-N_OutputSize];
//            if(M_CMD)
            if(!M_QUIET)
                cout<<OUTPUT[N_OutputSize*k+i]*Normalize_Vector[i+N_InputSize]<<"\t";
            cout.flush();
            file_out<<OUTPUT[N_OutputSize*k+i]*Normalize_Vector[i+N_InputSize]<<"\t";
        }
//                cout<<"test q coming out"<<endl;
//        printMemoryD(q,N_Total,N_Total);
        file_out<<endl;
    }
//        printMemoryD(OUTPUT,N_OutputSize * N_TestPatternCount,N_OutputSize);
    file_out.close();
    if(M_QUIET && !M_CMD && mode == TEST)
        cout<<"test done"<<endl;

//    if(M_COMPACT && (!M_CMD)){      //calculate test score as well if test target is provided inside "test.txt"
    if(FLAG & M_VALIDATE){      //calculate test score as well if test validation MOS is provided

        return score();
    }

    return 0;
}

int CRNN::train(double* MSE_RESULT,int num_iter ) throw (rnn_exception){
    if ((FLAG&0xcf) != 0xcf)
        throw rnn_exception("CRNN::train() invalid state, check input parameters");
    int ret = 0;
    int k;
    double MSEaveg_Old=0;
    double MSEaveg_delta = 0;
    double MSEaveg_delta_old = 0;
    double temp_sco;
    int intIteration = 0;
    int etaStart = 0;       // 1 if eta is under observation
    int etaSign = -1;
    double etaFactor = 0.1;
    double etaPeak = 0;     //store MSE value while Eta shift take place.
    fstream file_mse;
    fstream file_mse_delta;
    fstream file_mse_ratio;
    fstream file_mse_eta;
    fstream file_mse_sco;
    fstream file_mse_temp;
    stringstream ss;
    string strTemp;
    file_mse.open("mse.txt",ios::out|ios::trunc);
    file_mse_delta.open("mse_delta.txt",ios::out|ios::trunc);
    file_mse_ratio.open("mse_ratio.txt",ios::out|ios::trunc);
    file_mse_eta.open("mse_eta.txt",ios::out|ios::trunc);
    file_mse_sco.open("mse_sco.txt",ios::out|ios::trunc);
    file_mse_temp.open("mse_temp.txt",ios::out|ios::trunc);

    dbgPrint("--------------train started-------------");
    if(M_QUIET)
        cout<<"training in progress";
    cout.flush();

    if (*MSE_RESULT > 0)
        Mse_Threshold = *MSE_RESULT;

    ret = prepare_trn_patterns();

    if(num_iter >= 1)        //stop condition set to iter_num
        for (int i=0;i<num_iter;i++){   // Note adaptable Eta is not implemented in this mode
            MSEaveg_Old = MSEaveg;
            MSEaveg = 0;
            intIteration++;
            ss.clear();
            ss<<"#";
            ss<<intIteration;
            if (intIteration >2)
                Eta = Eta * 0.8;
//                Eta = 0.0125;
            if(intIteration%150){
//                getchar();
                Eta = 0.0125;
            }

            for (int j=0;j<N_TrainPatternCount;j++){
//                 k = Z(j);       //Z stores sample selection masks and preferences
                k = j;
                lambda = Applied_lambda;
                LAMBDA = Applied_LAMBDA;
//                printf("k= %d\n",k);
//                printMemoryD(wplus,N_Total*N_Total,N_Total);
//                cout<<endl;
//                printMemoryD(wminus,N_Total*N_Total,N_Total);

                ret = calculate_rate();
//                    printMemoryD(r,N_Total,N_Total);
                if(M_RECURSIVE)
                    ret = solve_nonlinear_equations(k);
                else
                    ret = calculate_ffm_output(k);
//                printMemoryD(q,N_Total,N_Total);
//                MSEaveg_delta = calculate_mse(k)/(j+1) - MSEaveg/(j+1);
                MSEaveg = MSEaveg + calculate_mse(k)/(j+1) - MSEaveg/(j+1);
                ret = calculate_inv();
                ret = update_wplus_1(k);
                ret = update_wminus_1(k);
            }

            ss<<"\tEta = ";
            ss<<Eta;
            file_mse_eta<<Eta<<endl;
            if(intIteration>2)
                MSEaveg_delta_old = MSEaveg_delta;
            else
                MSEaveg_delta_old = 1;
            if(intIteration>1)
                MSEaveg_delta = MSEaveg - MSEaveg_Old;
            else
                MSEaveg_delta = 1;
/*
            if(intIteration>2)
                if((MSEaveg_delta>=MSEaveg_delta_old)){      // we may be moving Eta in the wrong way or right way, can't tell
                    Eta = Eta * 1.25;                       // so simply undo the change and keep watching
                }
                else  {                                      // Otherwise we're definately correct,

//                cout<<"***********";                    //keep current value
                }
*/
        }
    else{           //stop condition set to mse_threshold
        ss.clear();
        ss<<"mse_threshold = ";
        ss<<Mse_Threshold<<endl;
        getline(ss,strTemp);
        dbgPrint(strTemp.c_str());
        MSEaveg = 99999;
        while(MSEaveg>Mse_Threshold){
//        while(1){
            // randomize load order
            CSEQ cseq(N_TrainPatternCount);
            cseq.work();
//            printMemoryI((unsigned int) cseq.output,N_TrainPatternCount,N_TrainPatternCount);
            // proceed
            MSEaveg_Old = MSEaveg;
            MSEaveg = 0;
            intIteration++;
            ss.clear();
            ss<<"#";
            ss<<intIteration;
            if (intIteration >2)
//                Eta += etaSign * etaFactor * Eta;
                Eta = 0.0125;
//                Eta = Eta * 0.8;
//                Eta = 0.05;
            if((intIteration == 453)){
//                getchar();
//                Eta = 0.005;
            }

            for (int j=0;j<N_TrainPatternCount;j++){
//                k = Z(j);       //Z stores sample selection masks and preferences
//                k = cseq.output[j]-1; // [1,N] in random order. Re-generated per iteration
                k = j;

                    //make sure lambda pointer has returned
                lambda = Applied_lambda;
                LAMBDA = Applied_LAMBDA;

                ret = calculate_rate();
                if(M_RECURSIVE)
                    ret = solve_nonlinear_equations(k);
                else
                    ret = calculate_ffm_output(k);
//                    MSEaveg_delta = calculate_mse(k)/(j+1) - MSEaveg/(j+1);
                MSEaveg = MSEaveg + calculate_mse(k)/(j+1) - MSEaveg/(j+1);
//                    cout<<"train q coming out"<<endl;
//                    printMemoryD(q,N_Total,N_Total);
                ret = calculate_inv();
                ret = update_wplus_1(k);
                ret = update_wminus_1(k);
            }

            ss<<"\tEta = ";
            ss<<Eta;
            file_mse_eta<<Eta<<endl;

            if(!(intIteration % 1000)){
                Eta = Eta / 10;
            }

            if(intIteration>2)
                MSEaveg_delta_old = MSEaveg_delta;
            else
                MSEaveg_delta_old = 1;
            if(intIteration>1)
                MSEaveg_delta = MSEaveg - MSEaveg_Old;
            else
                MSEaveg_delta = 1;
/*
            if(intIteration>2)
                if((MSEaveg_delta > MSEaveg_delta_old)){      // we're moving eta in wrong way
                    etaSign = -etaSign;                       // reverse direction
                }

*/

//            if(intIteration>2)
//                if((MSEaveg_delta>=MSEaveg_delta_old)){      // we may be moving Eta in the wrong way or right way, can't tell
//                    Eta = Eta * 1.25;                       // so simply undo the change and keep watching
//                }
//                else  {                                      // Otherwise we're definately correct,
//                    cout<<"***********";                    //keep current value
//                }



            /*
            if(intIteration>2)
                if((MSEaveg_delta>MSEaveg_delta_old)){      // we may be moving Eta in the wrong way or right way, can't tell
                    if(etaStart) {                              // since we're tracking good trent, we're doing wrong to steabalize the Eta
                        etaStart = 0;                           // Terminate the track
                    }                                           // Reduce the Eta, see if it works;
                    else{                                       // Otherwise we don't know what's happening
                        Eta = Eta * 1.25;                       // so simply undo the change and keep watching
                    }

                }

                else if((MSEaveg_delta == MSEaveg_delta_old))   // the same as before, undo the change
                    Eta = Eta * 1.25;
                else  {                                      // Otherwise we're definately correct,
                    etaStart = 1;                           // track the trend
//                    cout<<"***********";                    //keep reducing
                }
            */
            ss<<"\tcurrent MSE = ";
            ss<<MSEaveg;
            ss<<"\tMSE_delta = ";
            ss<<MSEaveg_delta;
            ss<<"\tratio= ";
            ss<<(-(MSEaveg_delta/MSEaveg));//<<endl;
            file_mse<<MSEaveg;
            file_mse<<"\n";
            file_mse_delta<<-MSEaveg_delta;
            file_mse_delta<<"\n";
            file_mse_ratio<<(-(MSEaveg_delta/MSEaveg));
            file_mse_ratio<<"\n";

            if(intIteration == 200){
//                break;
            }


            if(!(intIteration % N_Saved_Iterations)){
                dbgPrint("saving...*************************************************************************");
                save_weights();
                if(M_QUIET)
                    cout<<".";
                cout.flush();

            }
            if(intIteration == N_MaxIterations){
                dbgPrint("#debug: maximum iterations reached");

 //               getchar();
                break;
            }


//            if((-(MSEaveg_delta/MSEaveg))<0.001)
//                break;
/*            if(intIteration>2 && fabs(MSEaveg_delta)<1E-5){        // secondary threshold, MSE is not moving
                cout<<"stable MSE, shifting Eta"<<endl;
                if(etaPeak){                                       // only when previous peak is available
                    if((etaPeak - MSEaveg)<0.001*MSEaveg){         //Eta shift doesn't work, break loop;

//                        break;
                    }
//                    double R = 0.2*MSEaveg/(etaPeak - MSEaveg + 0.02*MSEaveg);   //determine shift rate
                    double R = 10*(etaPeak - MSEaveg)/(0.02*MSEaveg)+9;   //determine shift rate
                    file_mse_temp<<R<<endl;
                    etaPeak = MSEaveg;
                    Eta = R * Eta;
                }
                else{
                    etaPeak = MSEaveg;
                    Eta = 20*Eta;
                }
//                break;
            }
            */
            //if(MSEaveg<0.0710)
            //    break;
            temp_sco = test();
            ss<<"\tsco= ";
            ss<<temp_sco<<endl;
            if(!M_QUIET){
                getline(ss,strTemp);
                ss.clear();
                cout<<strTemp<<endl;
            }
            cout.flush();
//            file_mse_sco<<temp_sco*0.08<<endl;
            file_mse_sco<<temp_sco<<endl;

//            printMemoryD(q,N_Total,N_Total);
//            cout<<endl;

        }
    }

    printMemoryD(q,N_Total,N_Total);

    file_mse.close();
    file_mse_delta.close();
    file_mse_ratio.close();
    file_mse_eta.close();
    file_mse_sco.close();
    file_mse_temp.close();

//!
//!  begin to save weight into .dat file
//!  for furture usage
//!
    save_weights();
    printMemoryD(wplus,N_Total * N_Total,N_Total);
    ss<<endl;
    printMemoryD(wminus,N_Total * N_Total,N_Total);

    *MSE_RESULT = MSEaveg;               //can we drop this variable?

    dbgPrint("--------------train finished------------");
    if(M_QUIET)
        cout<<"done"<<endl;

    return 0;
}

double CRNN::score() throw (rnn_exception){
    //code for state checking still to be fnished
        if (!M_COMPACT)
            throw rnn_exception("CRNN::score() need COMPACT mode");
        fstream file_a,file_b;
        double *a, *b;
        double honey=0;
        double dummy;
        int cnt = 0;
        int cntt = 0;
        file_a.open((Tst_File_Name+".dat").c_str(),ios::in);
        if(!file_a.is_open())
            file_a.open(Tst_File_Name.c_str(),ios::in);
        file_b.open(Out_File_Name.c_str(),ios::in);
        if(!file_a.is_open() || !file_b.is_open())
            throw rnn_exception("CRNN::score() test or output file open failed");
        a = new double[N_OutputSize*N_TestPatternCount];
        b = new double[N_OutputSize*N_TestPatternCount];
        double* pp1 = a;
        double* pp2 = b;
        file_a >> dummy;
        file_a >> dummy;
        while (file_a >> dummy ) {
            if(cnt>=N_InputSize){   //!
                *pp1 = dummy;
                pp1++;
            }

            cnt++;
            cntt++;
            cnt = cnt % (N_InputSize + N_OutputSize);

        }
        while (file_b >> *pp2 ) {
            pp2++;
        }
        pp1=a;pp2=b;
        for(int ii = 0;ii<N_OutputSize*N_TestPatternCount;ii++){
            honey = honey +  (a[ii]- b[ii])*(a[ii]- b[ii])/(ii+1) - honey/(ii+1);
        }
        file_a.close();
        file_b.close();
        delete[] a;
        delete[] b;
        return honey;
}

int CRNN::prepare_trn_patterns(){
    int N_TrainCount = N_InputSize * N_TrainPatternCount;
    double *p1,*p2, *p3;

//    printMemoryD(TARGET,N_OutputSize * N_TrainPatternCount, N_OutputSize);
    memset(Applied_lambda,0,sizeof(double)* N_InputSize * N_TrainPatternCount );
    memset(Applied_LAMBDA,0,sizeof(double)* N_InputSize * N_TrainPatternCount );
    p1=TRAIN_INPUT;
    p2=Applied_LAMBDA;
    p3=Applied_lambda;
    for(int i=0;i<N_TrainCount;i++){
        if(*p1>=0)
            *p2=*p1;
        else
            *p3=-(*p1);
        p1++;
        p2++;
        p3++;
    }
    LAMBDA = Applied_LAMBDA;
    lambda = Applied_lambda;
    Applied_y = TARGET;
    y = Applied_y;
    return 0;
}

double CRNN::calculate_mse(int k){
//  in:   q, y ,MSEaveg
//  out:  MSEaveg
//  sub:  None

    double MSE = 0;
    double a;
    for(int i=0;i<N_OutputSize;i++){
        a = *(q+i+N_Total-N_OutputSize) - *(y+N_OutputSize*k+i);
        MSE = MSE + a*a;
    }
    return MSE;
}

int CRNN::update_wplus_1(int k){
//  in:    N_Total,D, gammaplus, Winv
//  out:   r, q
//
    int u,uu,v,i,intCount,index_start ;
    int* p=N_LayerArray;
//    printMemoryD( wplus,N_Total*N_Total,N_Total);
//    cout<<endl;
    double vmplus,sum1;
    double* gammaplus = new double[N_Total];        //initialized during process
    double* wplus_result = new double[N_Total*N_Total];
    memset(wplus_result,0,sizeof(double)* N_Total*N_Total );
    index_start = *p;
    uu = 0;
//    printMemoryD(wplus,N_Total*N_Total,N_Total);
    if(M_RECURSIVE){
        for(u=0;u<N_Total;u++)
            for(v=0;v<N_Total;v++){
                if(u==v){
                    wplus_result[N_Total*u+v] = 0;
                    continue;
                }
                memset(gammaplus,0,sizeof(double)* N_Total );
                for(i=0;i<N_Total;i++){
                    gammaplus[i] = 0;
                    if(u!=i && v == i)
                        gammaplus[i] = 1/D[i];
                    else if(u==i && v!=i)
                        gammaplus[i] = -1/D[i];
                }
                sum1 = 0;
                for(i=N_Total - N_OutputSize;i<N_Total;i++){
 //                 printf("check1 %f * %f + %f * %f  \n",gammaplus[u],Winv[N_Total*u+v],gammaplus[v],Winv[N_Total*v+i]);
                    vmplus = gammaplus[u] * Winv[N_Total*u+i] + gammaplus[v] * Winv[N_Total*v+i];
 //                 printf("check2 %f\n",vmplus);
//                  printf("check1 sum1= %f + %f *( %f-%f)*%f  \n",sum1,vmplus,q[i],TARGET[N_OutputSize*k+i],q[u]);
                    sum1 = sum1 + vmplus * (q[i]-TARGET[N_OutputSize*k+i-(N_Total-N_OutputSize)]) * q[u];
//                  printf("check3 sum1 = %f\n",sum1);
                    sum1 = sum1;
                }

                wplus_result[N_Total*u+v] = wplus[N_Total*u+v] - Eta * sum1;
//              cout<<"wplus_check"<<endl;
//              printMemoryD(wplus_result,N_Total*N_Total,N_Total);
                if(wplus_result[N_Total*u+v] < 0)
                    wplus_result[N_Total*u+v] = 0;
                //double check end

            }
    }
else
    for(u=0;u<N_Total;u++){
        intCount = 0;
        for(v=index_start;intCount<*(p+1);v++){
//            printf("u=%d\tv=%d\n",u,v);
            intCount++;

            //double check begin
            memset(gammaplus,0,sizeof(double)* N_Total );
            for(i=0;i<N_Total;i++){
                gammaplus[i] = 0;
                if(u!=i && v == i)
                    gammaplus[i] = 1/D[i];
                else if(u==i && v!=i)
                    gammaplus[i] = -1/D[i];
            }
            sum1 = 0;
            for(i=N_Total - N_OutputSize;i<N_Total;i++){
 //               printf("check1 %f * %f + %f * %f  \n",gammaplus[u],Winv[N_Total*u+v],gammaplus[v],Winv[N_Total*v+i]);
                vmplus = gammaplus[u] * Winv[N_Total*u+i] + gammaplus[v] * Winv[N_Total*v+i];
 //               printf("check2 %f\n",vmplus);
//                printf("check1 sum1= %f + %f *( %f-%f)*%f  \n",sum1,vmplus,q[i],TARGET[N_OutputSize*k+i],q[u]);
                sum1 = sum1 + vmplus * (q[i]-TARGET[N_OutputSize*k+i+N_OutputSize-N_Total]) * q[u];
//                printf("check3 sum1 = %f\n",sum1);
                sum1 = sum1;
            }
//            printf("check3 sum1 = %f\n",sum1);
//            cout<<"middle check"<<endl;
//            printf("%f - %f * %f",wplus[N_Total*u+v],Eta, sum1);
            wplus_result[N_Total*u+v] = wplus[N_Total*u+v] - Eta * sum1;
//            cout<<"wplus_check"<<endl;
//            printMemoryD(wplus_result,N_Total*N_Total,N_Total);
            if(wplus_result[N_Total*u+v] < 0)
                wplus_result[N_Total*u+v] = 0;
            //double check end
        }
        uu++;
        if(uu==*p){
            p++;
            index_start += *p;
            uu = 0;
        }
    }

//    cout<<"wplus"<<endl;
//    printMemoryD(wplus_result,N_Total*N_Total,N_Total);
//    cout<<endl;
    memcpy(wplus,wplus_result,sizeof(double)*N_Total*N_Total);
    delete[] gammaplus;
    delete[] wplus_result;
    return 0;
}

int CRNN::update_wminus_1(int k){
    int u,uu,v,i,intCount,index_start ;
    int* p=N_LayerArray;

    double vmminus,sum1;
    double* gammaminus = new double[N_Total];        //initialized during process
    double* wminus_result = new double[N_Total*N_Total];
    memset(wminus_result,0,sizeof(double)* N_Total*N_Total );
    index_start = *p;
    uu = 0;

//    printMemoryD(wminus,N_Total*N_Total,N_Total);

    if(M_RECURSIVE){
        for(u=0;u<N_Total;u++)
            for(v=0;v<N_Total;v++){
                if(u==v){
                    wminus_result[N_Total*u+v] = 0;
                    continue;
                }
                memset(gammaminus,0,sizeof(double)* N_Total );
                for(i=0;i<N_Total;i++){
                    gammaminus[i] = 0;
                    if(u!=i && v == i)
                        gammaminus[i] = -q[i]/D[i];
                    else if(u==i && v!=i)
                        gammaminus[i] = -1/D[i];
                    else    //loop. Not supposed to happen in FF RNN
                        gammaminus[i] = -(1.0+q[i])/D[i];
                }
                sum1 = 0;
                for(i=N_Total - N_OutputSize;i<N_Total;i++){
 //               printf("check1 %f * %f + %f * %f  \n",wminus[N_Total*u+v],Eta, sum1);
                    vmminus = gammaminus[u] * Winv[N_Total*u+i] + gammaminus[v] * Winv[N_Total*v+i];
 //               printf("check2 %f\n",vmminus);
//                printf("check1 sum1= %f + %f *( %f-%f)*%f  \n",sum1,vmminus,q[i],TARGET[N_OutputSize*k+i],q[u]);
                    sum1 = sum1 + vmminus * (q[i]-TARGET[N_OutputSize*k+i-(N_Total-N_OutputSize)]) * q[u];
//                printf("check3 sum1 = %f",sum1);
                    sum1 = sum1;
                }

                //            cout<<"middle check"<<endl;
//              printf("%f - %f * %f",wminus[N_Total*u+v],Eta, sum1);
                wminus_result[N_Total*u+v] = wminus[N_Total*u+v] - Eta * sum1;
//              cout<<"wminus_check"<<endl;
//              printMemoryD(wminus_result,N_Total*N_Total,N_Total);
                if(wminus_result[N_Total*u+v] < 0)
                wminus_result[N_Total*u+v] = 0;
                //double check end

            }
    }
    else{
        for(u=0;u<N_Total;u++){
            intCount = 0;
            for(v=index_start;intCount<*(p+1);v++){
                intCount++;

            //double check begin
                memset(gammaminus,0,sizeof(double)* N_Total );
                for(i=0;i<N_Total;i++){
                    gammaminus[i] = 0;
                    if(u!=i && v == i)
                        gammaminus[i] = -q[i]/D[i];
                    else if(u==i && v!=i)
                        gammaminus[i] = -1/D[i];
                    else    //loop. Not supposed to happen in FF RNN
                        gammaminus[i] = -(1.0+q[i])/D[i];
                }
//            printMemoryD(q,N_Total*N_Total,N_Total);
                sum1 = 0;
                for(i=N_Total - N_OutputSize;i<N_Total;i++){
 //               printf("check1 %f * %f + %f * %f  \n",wminus[N_Total*u+v],Eta, sum1);
                    vmminus = gammaminus[u] * Winv[N_Total*u+i] + gammaminus[v] * Winv[N_Total*v+i];
 //               printf("check2 %f\n",vmminus);
//                printf("check1 sum1= %f + %f *( %f-%f)*%f  \n",sum1,vmminus,q[i],TARGET[N_OutputSize*k+i],q[u]);
                    sum1 = sum1 + vmminus * (q[i]-TARGET[N_OutputSize*k+i+N_OutputSize-N_Total]) * q[u];
//                printf("check3 sum1 = %f",sum1);
                    sum1 = sum1;
                }

//            cout<<"middle check"<<endl;
//            printf("%f - %f * %f",wminus[N_Total*u+v],Eta, sum1);
                wminus_result[N_Total*u+v] = wminus[N_Total*u+v] - Eta * sum1;
//            cout<<"wminus_check"<<endl;
//            printMemoryD(wminus_result,N_Total*N_Total,N_Total);
                if(wminus_result[N_Total*u+v] < 0)
                    wminus_result[N_Total*u+v] = 0;
            //double check end
            }
            uu++;
            if(uu==*p){
                p++;
                index_start += *p;
                uu = 0;
            }
        }
    }       //end of FFM RNN

//    cout<<"wminus"<<endl;
//    printMemoryD(wminus_result,N_Total*N_Total,N_Total);
    memcpy(wminus,wminus_result,sizeof(double)*N_Total*N_Total);
    delete[] gammaminus;
    delete[] wminus_result;
    return 0;
}
