#include "crnn_t.h"
#include <sstream>
#include "string.h"
#include <stdlib.h>
#include <cmath>
//#include <math.h>

using namespace std;
using namespace rnn_lite;

CRNN_T::CRNN_T(){


    FLAG = 0;
    M_DEBUG = 0;
    M_RECURSIVE = 0;
    M_QUIET = 1;

    dbgPrint("building CRNN_T");

    N_InputSize = 0;
    N_HiddenSize = 0;
    N_OutputSize = 0;
    N_TestPatternCount = 0;

    OUTPUT  = NULL;
    wplus   = NULL;
    wminus  = NULL;
    LAMBDA  = NULL;
    lambda  = NULL;
    Input_lambda    = NULL;
    Input_LAMBDA    = NULL;
    r   = NULL;
    W   = NULL;
    N   = NULL;
    D   = NULL;
    q   = NULL;
    Winv= NULL;
    TEST_INPUT  = NULL;
    N_LayerArray= NULL;
    Normalize_Vector = NULL;

    Weights_File_Name = "weight.dat";
    Out_File_Name = "";                         //don't print to file by default

//    Log_File_Name = "rnn_gen_log1.txt";         //Results file name (ASCII format)
//    Temp_Log_File_Name = "rnn_gen_log1.m";      //Results file name (MATLAB format)
//    Res = .0001;                               //Resolution of solving the non linear equations of the model
//    AUTO_MAPPING = 0;                          //Flag = 1 only if the network is recurrent with shared inputs/outputs
//    AUTO_MAPPING = 0;                          //Flag = 0 for FF networks and 1 for fully recurrent networks
    FIX_RIN = 0;                                //fixed input firing rate, set 1 only for recurrent network.

    R_IN = 1;                                   //DO NOT CHANGE (Related to RNN function approximation)
    R_Out = .1;                                 //Firing Rate of the Output Neurons
}


CRNN_T::~CRNN_T(){
//    cout<<"destructing CRNN_T"<<endl;
    delete[] Input_LAMBDA;
    delete[] Input_lambda;
    delete[] wplus;
    delete[] wminus;
    delete[] r;
    delete[] N;
    delete[] D;
    delete[] q;
    delete[] W;
    delete[] TEST_INPUT;
    delete[] OUTPUT;
    delete[] N_LayerArray;
    delete[] Normalize_Vector;
    if(file_weight.is_open())
        file_weight.close();
}

void CRNN_T::setMode(RNN_MODE i){
    mode = i;
    FLAG |= M_MODE;
}
int CRNN_T::getMode(){
    return mode;
}
void CRNN_T::setRecursiveMode(int i){
    if (i>0){
        M_RECURSIVE = 1;
    }
    else{
        M_RECURSIVE = 0;
    }
}
void CRNN_T::setDebugMode(int i){
    if (i>0)
        M_DEBUG = 1;
    else
        M_DEBUG = 0;
}
void CRNN_T::setQuietMode(int i){
    if (i>0)
        M_QUIET = 1;
    else
        M_QUIET = 0;
}
int CRNN_T::load_weights() throw (argument_exception)
//! in:     Weights_File_Name
//! out:    N_Total, N_InputSize, N_OutputSize, N_HiddenSize,
//!         N_AllLayerCount, N_LayerArray,
//!         wplus, wminus
//!  mem allocated: wplus, wminus, N_LayerArray
{   if (FLAG & M_WEIGHT){       //weight matrix re-enter
        delete[] wplus;
        delete[] wminus;
        FLAG &= ~M_WEIGHT;
    }
    if (FLAG & M_TOPOLOGY){     //topology def re-enter
        N_InputSize = 0;
        N_OutputSize = 0;
        N_HiddenSize = 0;
        N_Total = 0;
        N_AllLayerCount = 0;
        delete[] N_LayerArray;
        FLAG &= ~M_TOPOLOGY;
    }
    if (FLAG & M_NORMALIZE){     //Normalize vector def re-enter
        delete[] Normalize_Vector;
        FLAG &= ~M_NORMALIZE;
    }
    double* p;

    file_weight.open(Weights_File_Name.c_str(),ios::in|ios::binary);
    if (!file_weight.is_open())
        throw argument_exception("'-test' failed: cannot open designated weights.");

    //process topology info
    int* pInt = &M_RECURSIVE;
    file_weight.read((char*)pInt,sizeof(int));
    pInt = &N_AllLayerCount;
    file_weight.read((char*)pInt,sizeof(int));
    N_LayerArray = new int[N_AllLayerCount + 1];
    pInt = N_LayerArray;
    file_weight.read((char*)pInt,sizeof(int) * (N_AllLayerCount+1));

    if(N_LayerArray[N_AllLayerCount] != 0){
        throw argument_exception("weight file corrupted");
    }
    N_InputSize = N_LayerArray[0];
    N_OutputSize = N_LayerArray[N_AllLayerCount-1];
    N_Total = 0;
    for(int i=0;i<N_AllLayerCount;i++)
        N_Total += N_LayerArray[i];
    if(M_RECURSIVE){
        N_Total = N_Total - N_InputSize - N_OutputSize;
        N_HiddenSize = N_Total;
    }
    else
        N_HiddenSize = N_Total - N_InputSize - N_OutputSize;
    FLAG |= M_TOPOLOGY;

    //process Normalize_Vector
    Normalize_Vector = new double[N_InputSize + N_OutputSize];
    p = Normalize_Vector;
    file_weight.read((char*)p,sizeof(double)*(N_InputSize + N_OutputSize));
    FLAG |= M_NORMALIZE;

    //Process weights
    wplus = new double[N_Total*N_Total];
    wminus = new double[N_Total*N_Total];
    memset(wplus,0,sizeof(double)* N_Total*N_Total );
    memset(wminus,0,sizeof(double)* N_Total*N_Total );
    p = wplus;
    file_weight.read((char*)p,sizeof(double)*N_Total * N_Total);
    p = wminus;
    file_weight.read((char*)p,sizeof(double)*N_Total * N_Total);
    file_weight.close();
    dbgPrint("weight loaded;");

    printMemoryD(wplus,N_Total * N_Total,N_Total);
    dbgPrint("");
    printMemoryD(wminus,N_Total * N_Total,N_Total);

    FLAG |= M_WEIGHT;
    return 0;
}



//use only when input data is fed via parameters directly
//for test run only
int CRNN_T::loadInputCmd(string strInput, string strTarget ) throw (argument_exception,rnn_exception){
//!     in:   string
//!     out:  N_InputSize, N_OutputSize, N_HiddenSize, N_Total,
//!           N_AllLayerCount, N_LayerArray[]
//!
//!    take string in the form as "5.7.2"
//!    interpret as layer specification
//!    output array as a[]={5,7,2,0}
//!
    setMode(TEST);

    size_t intPos = 0;
    size_t intPosP = 0;
    int intSize = 0;
//    int intSum = 0;
    if (strInput=="")
        throw rnn_exception("No valid input");

    //**********************************
    //Define RNN architecture
    //Define Normalization vector
    //will define   N_InputSize
    //              N_HiddenSize
    //              N_OutputSize
    //              N_Total
    //              N_AllLayerCount
    //              N_LayerArray[]
    //              Normalize_Vector[]
    //**********************************
    if (mode == TEST ){
        load_weights();
    }
    else
        throw rnn_exception("Non test mode??");

    //**********************************
    //Variable allocation
    //will define   N_testPatternCount
    //              OUTPUT
    //**********************************
    N_TestPatternCount = 1;
    OUTPUT = new double[N_OutputSize * N_TestPatternCount];
//        Applied_lambda = new double[N_InputSize * N_TrainPatternCount];
//        Applied_LAMBDA = new double[N_InputSize * N_TrainPatternCount];
    Input_lambda = new double[N_InputSize * N_TestPatternCount];
    Input_LAMBDA = new double[N_InputSize * N_TestPatternCount];
    N = new double[N_Total];
    D = new double[N_Total];
    q = new double[N_Total];
    r = new double[N_Total];
    W = new double[N_Total * N_Total];
    memset(r,0,sizeof(double)* N_Total );
    memset(W,0,sizeof(double)* N_Total * N_Total);

    //**********************************
    //handle input train/test data
    //will define   TRAIN_INPUT
    //              TARGET
    //              TEST_INPUT
    //              wplus
    //              wminus
    //**********************************

    //check vector dimension
    intPos = 0;
    intSize = 0;
    while(intPos < strInput.size()){
        intSize ++;
        intPos++;
        intPos = strInput.find(',',intPos);
    }
    if(intSize < 1)
        throw argument_exception("argument '-i' encountered an invalid argument");
    if(N_InputSize>0 && N_InputSize != intSize)
        throw argument_exception("argument '-i' conflict in variable N_InputSize");
    N_InputSize = intSize;
    TEST_INPUT = new double[N_InputSize];

    //parse the sentence
    intPos = -1;
    for(int i=0;i<intSize;i++){
        intPos++;
        intPosP = intPos;
        intPos = strInput.find(',',intPos);
        stringstream ss(strInput.substr(intPosP,intPos - intPosP));
        ss>>TEST_INPUT[i];
//            cout<<TEST_INPUT[i]<<endl;
//            cout<<Normalize_Vector[i]<<endl;
        TEST_INPUT[i]=TEST_INPUT[i]/Normalize_Vector[i];
    }
//        printMemoryD(TEST_INPUT,N_InputSize,N_InputSize);
    FLAG |= M_TESTINPUT ;
    return 0;
}

double CRNN_T::test() throw (rnn_exception)
//  in:    wplus, wminus, R_OUT, LAMBDA, lambda
//  out:   r, q
//
//
{   if ((FLAG&0X1F) != 0X1F)    //validation MOS is not mandatory
        throw rnn_exception("CRNN_T::test() invalid state, check input parameters");
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
        printMemoryD(q,N_Total * N_TestPatternCount,N_Total);
//        dbgPrint("begin output\n");
        for(int i =0;i<N_OutputSize;i++){
            OUTPUT[N_OutputSize*k+i] = q[i+N_Total-N_OutputSize];
//            cout<<OUTPUT[N_OutputSize*k+i]*Normalize_Vector[i+N_InputSize]<<"\t";
            cout<<OUTPUT[N_OutputSize*k+i]*Normalize_Vector[i+N_InputSize]<<"\t"<<endl;
            file_out<<OUTPUT[N_OutputSize*k+i]*Normalize_Vector[i+N_InputSize]<<"\t";
        }
        file_out<<endl;
    }
//        printMemoryD(OUTPUT,N_OutputSize * N_TestPatternCount,N_OutputSize);


    file_out.close();

    return 0;
}

int CRNN_T::prepare_tst_patterns(){

    int N_TestCount = N_InputSize * N_TestPatternCount;
    double *p1,*p2, *p3;

    memset(Input_lambda,0,sizeof(double)* N_InputSize * N_TestPatternCount );
    memset(Input_LAMBDA,0,sizeof(double)* N_InputSize * N_TestPatternCount );
    p1=TEST_INPUT;
    p2=Input_LAMBDA;
    p3=Input_lambda;
    for(int i=0;i<N_TestCount;i++){
        if(*p1>=0)
            *p2=*p1;
        else
            *p3=-(*p1);
        p1++;
        p2++;
        p3++;
    }
    LAMBDA = Input_LAMBDA;
    lambda = Input_lambda;
//    printMemoryD(LAMBDA,N_InputSize*N_TestPatternCount,N_InputSize);
//    printMemoryD(lambda,N_InputSize*N_TestPatternCount,N_InputSize);

    return 0;
}

int CRNN_T::calculate_rate(){
//  in:    wplus, wminus, R_OUT
//  out:   r
//
//
//
    int i;
    memset(r,0,sizeof(double)* N_Total );
    for(i=0;i<N_Total;i++)
        for(int j=0;j<N_Total;j++)
            r[i]=r[i]+*(wplus+N_Total*i+j)+*(wminus+N_Total*i+j);
    if(FIX_RIN == 1)
        for(i=0;i<N_InputSize;i++)
            r[i] = R_IN;
//    if (AUTO_MAPPING == 0)
    if (M_RECURSIVE == 0)
        for(i=N_Total - N_OutputSize;i<N_Total;i++)
            r[i] = R_Out;
//    printMemoryD(r,N_Total,N_Total);
    return 0;
}

int CRNN_T::calculate_ffm_output(int k){
//    in:   wplus, wminus, LAMBDA, lambda, r
//    temp: N, D
//    out:  q, N, D
//    sub:  None
//
//
//    printMemoryD(r,N_Total,N_Total);
    memset(q,0,sizeof(double)* N_Total );

    for(int i=0;i<N_Total;i++){
        N[i]=0;
        D[i]=0;
        for(int j=0;j<N_Total;j++){
//            printMemoryD(D,N_Total,N_Total);
            *(N+i)=*(N+i)+*(q+j) * *(wplus+N_Total*j+i);
            *(D+i)=*(D+i)+*(q+j) * *(wminus+N_Total*j+i);
        }
//        printMemoryD(D,N_Total,N_Total);
//        printMemoryD(N,N_Total,N_Total);
        if (i<N_InputSize){
            *(N+i) = *(N+i) + *(LAMBDA+ N_InputSize*k +i);
//            *(D+i) = *(D+i) + *(lambda+ N_InputSize*k +i) + *(r+i);
            *(D+i) = *(D+i) + *(lambda+ N_InputSize*k +i);

        }
//        printMemoryD(D,N_Total,N_Total);
        *(D+i) = *(D+i) + *(r+i);

        if (*(D+i)!=0)
            *(q+i) = *(N+i) / *(D+i);
        else
            *(q+i) = 1.0;
        if (*(q+i) > 1.0)
            *(q+i) = 1.0;
        else if (*(q+i) < 0)
            *(q+i) = 0;

    }
//    printMemoryD(N,N_Total,N_Total);
//    printMemoryD(D,N_Total,N_Total);
//    printMemoryD(q,N_Total,N_Total);
    return 0;
}

int CRNN_T::solve_nonlinear_equations(int k){
//    in:   N_Total, r, wplus, wminus, LAMBDA, lambda
//    temp: lambda_minus, lambda_plus, Pplus, Pminus, F, G,
//    out:  q, D,
//    sub:  None
//

    // we patch through only the kth entry to this routine
    double* lambda_temp = new double[N_Total];
    double* LAMBDA_temp = new double[N_Total];
    memset(lambda_temp,0,sizeof(double)*N_Total);
    memset(LAMBDA_temp,0,sizeof(double)*N_Total);
    for(int i=0;i<N_Total;i++){
//        lambda_temp[i]=0.5;
        LAMBDA_temp[i]=0.5;
    }

    memcpy(lambda_temp,lambda+k*N_InputSize,sizeof(double)*N_InputSize);
    memcpy(LAMBDA_temp,LAMBDA+k*N_InputSize,sizeof(double)*N_InputSize);
//    printMemoryD(LAMBDA_temp,N_Total,N_Total);

    int i,j;
    double *temp;
    double* lambda_minus = new double[N_Total];
//    double* lambda_plus = new double[N_Total];
    double *lambda_plus, *G;
    double* Pplus = new double[N_Total * N_Total];
    double* Pminus = new double[N_Total * N_Total];
    double* F = new double[N_Total * N_Total];
    for(i=0;i<N_Total;i++){
        *(lambda_minus+i) = ((double)(rand()%((int)(0.2 *10000))))/10000;
    }
    for(i=0;i<N_Total;i++)
        for(j=0;j<N_Total;j++){
            if(r[i] != 0){
                Pplus[i*N_Total+j] = wplus[i*N_Total + j]/ r[i];
                Pminus[i*N_Total+j] = wminus[i*N_Total+j]/ r[i];
            }
            else{
                Pplus[i*N_Total+j] = 0;
                Pminus[i*N_Total+j]= 0;
            }

        }

    for(int llll = 0;llll<200;llll++){
        for(i=0;i<N_Total;i++)
            for(j=0;j<N_Total;j++)
                if(i==j)
                    F[i*N_Total+j] = r[i]/(r[i]+ lambda_minus[i]);
                else
                    F[i*N_Total+j] = 0;
//        printMemoryD(F,N_Total*N_Total,N_Total);
//        printMemoryD(Pplus,N_Total*N_Total,N_Total);
        temp = matrix_mul(F,Pplus,N_Total,N_Total,N_Total);
//        printMemoryD(temp,N_Total*N_Total,N_Total);
        for(i=0;i<N_Total;i++)
            for(j=0;j<N_Total;j++)
                if(i==j)
                    temp[i*N_Total+j] = 1-(temp[i*N_Total+j]);
                else
                    temp[i*N_Total+j] = -(temp[i*N_Total+j]);
        matrix_inv(temp,N_Total);
//        lambda_plus = matrix_mul(LAMBDA_temp+k*N_Total,temp,1,N_Total,N_Total);
//        printMemoryD(LAMBDA_temp,N_Total,N_Total);
//        printMemoryD(temp,N_Total*N_Total,N_Total);
        lambda_plus = matrix_mul(LAMBDA_temp,temp,1,N_Total,N_Total);
        delete[] temp;
//        printMemoryD(lambda_plus,N_Total,N_Total);
        temp = matrix_mul(lambda_plus,F,1,N_Total,N_Total);
        delete[] lambda_plus;
        G = matrix_mul(temp, Pminus,1,N_Total,N_Total);
//        printMemoryD(G,N_Total,N_Total);
        for(i=0;i<N_Total;i++)
//            G[i] += lambda_temp[k*N_Total+i];
            G[i] += lambda_temp[i];
//        printMemoryD(G,N_Total,N_Total);
//        cout<<endl;
//        printMemoryD(lambda_minus,N_Total,N_Total);
        delete[] temp;
        int iii = 0;
        for(int ll=0;ll<N_Total;ll++)
            if (fabs(lambda_minus[ll] - G[ll])>0.001 || lambda_minus[ll]<0)
                iii++;
        if (iii>0)
            memcpy(lambda_minus,G,sizeof(double)*N_Total);
        else{
//            getchar();
            break;
        }
    }
//    printMemoryD(r,N_Total,N_Total);
//    printMemoryD(lambda_minus,N_Total,N_Total);
    for (i=0;i<N_Total;i++)
        for(j=0;j<N_Total;j++)
            if(i==j)
                F[i*N_Total+j] = r[i] / (r[i]+lambda_minus[i]);
            else
                F[i*N_Total+j] = 0;
//    printMemoryD(F,N_Total*N_Total,N_Total);
//    printMemoryD(Pplus,N_Total*N_Total,N_Total);
    temp = matrix_mul(F,Pplus,N_Total,N_Total,N_Total);
    for(i=0;i<N_Total;i++)
        for(j=0;j<N_Total;j++)
            if(i==j)
                temp[i*N_Total+j] = 1-(temp[i*N_Total+j]);
            else
                temp[i*N_Total+j] = -(temp[i*N_Total+j]);
    matrix_inv(temp,N_Total);
//    lambda_plus = matrix_mul(LAMBDA_temp+k*N_Total,temp,1,N_Total,N_Total);
//    printMemoryD(LAMBDA_temp,N_Total,N_Total);
//    printMemoryD(temp,N_Total*N_Total,N_Total);
    lambda_plus = matrix_mul(LAMBDA_temp,temp,1,N_Total,N_Total);
    delete[] temp;

//    printMemoryD(lambda_plus,N_Total,N_Total);
//    cout<<endl;
//    printMemoryD(lambda_minus,N_Total,N_Total);
//    printMemoryD(lambda_plus,N_Total,N_Total);
    for(i=0;i<N_Total;i++){
        q[i] = lambda_plus[i] / (r[i]+lambda_minus[i]);
        if (q[i]>1)
            q[i] = 1;
        if (q[i]<0)
            q[i] = 0;
        D[i] = lambda_minus[i] + r[i];
    }
//    printMemoryD(q,N_Total,N_Total);


    delete[] Pplus;
    delete[] Pminus;
    delete[] lambda_minus;
    delete[] lambda_temp;
    delete[] LAMBDA_temp;

    delete[] G;
    return 0;
}

int CRNN_T::calculate_inv(){
//  in:   wplus, wminus, q, D
//  out:  W, Winv;
//  sub:  matrix_inv();

    int i,j;
    for(j=0;j<N_Total;j++)
        for(i=0;i<N_Total;i++){
            *(W+N_Total*j+i) = - ( *(wplus+N_Total*j+i) - *(wminus+N_Total*j+i)* *(q+i)) / *(D+i);
//            printMemoryD(W,N_Total*N_Total,N_Total);
        }
    for(i=0;i<N_Total;i++)
        *(W+N_Total*i+i) = *(W+N_Total*i+i) + 1;
//    printMemoryD(wplus,N_Total*N_Total,N_Total);
    int ret = matrix_inv((double *)W,N_Total);
    Winv = W;
    return ret;
}

int CRNN_T::matrix_inv(double a[], int n){
    //n as matrix dimension
    //
    int *is,*js,i,j,k,l,u,v;
    double d,p;
    is=(int*)malloc(n*sizeof(int));
    js=(int*)malloc(n*sizeof(int));
    for (k=0; k<=n-1; k++)
      { d=0.0;
        for (i=k; i<=n-1; i++)
        for (j=k; j<=n-1; j++)
          { l=i*n+j; p=fabs(a[l]);
            if (p>d) { d=p; is[k]=i; js[k]=j;}
          }
        if (d+1.0==1.0)
          { free(is); free(js); printf("err**not inv\n");
            return(1);
          }
        if (is[k]!=k)
          for (j=0; j<=n-1; j++)
            { u=k*n+j; v=is[k]*n+j;
              p=a[u]; a[u]=a[v]; a[v]=p;
            }
        if (js[k]!=k)
          for (i=0; i<=n-1; i++)
            { u=i*n+k; v=i*n+js[k];
              p=a[u]; a[u]=a[v]; a[v]=p;
            }
        l=k*n+k;
        a[l]=1.0/a[l];
        for (j=0; j<=n-1; j++)
          if (j!=k)
            { u=k*n+j; a[u]=a[u]*a[l];}
        for (i=0; i<=n-1; i++)
          if (i!=k)
            for (j=0; j<=n-1; j++)
              if (j!=k)
                { u=i*n+j;
                  a[u]=a[u]-a[i*n+k]*a[k*n+j];
                }
        for (i=0; i<=n-1; i++)
          if (i!=k)
            { u=i*n+k; a[u]=-a[u]*a[l];}
      }
    for (k=n-1; k>=0; k--)
      { if (js[k]!=k)
          for (j=0; j<=n-1; j++)
            { u=k*n+j; v=js[k]*n+j;
              p=a[u]; a[u]=a[v]; a[v]=p;
            }
        if (is[k]!=k)
          for (i=0; i<=n-1; i++)
            { u=i*n+k; v=i*n+is[k];
              p=a[u]; a[u]=a[v]; a[v]=p;
            }
      }
    free(is); free(js);
    return(0);
}

double* CRNN_T::matrix_mul(double a[], double b[], int d1, int d2, int d3){
    //matrix a is d1 X d2
    //matrix b is d2 X d3
    //result matrix is d1 X d3
    //Do Remember to release resource for Result Matrix
    double *result = new double[d1 * d3];
    double sum;
    for(int i=0;i<d1;i++)
        for(int j=0;j<d3;j++){
            sum = 0;
            for(int k=0;k<d2;k++){
                sum += (a[i*d2+k]) * (b[k*d3+j]);
            }
            result[i*d3+j] = sum;
        }
    return result;
}

void CRNN_T::printMemory(void* in_addr, int intCount){
    int i=0;
    char* addr;
    addr = (char*)in_addr;
//    printf("%X %X \n",addr, addr+1);
    while(i<intCount * 8){
        printf("%x : %2x %2x %2x %2x %2x %2x %2x %2x ",addr+i, 0xff&*(addr+i),0xff&*(addr+i+1),0xff&*(addr+i+2),0xff&*(addr+i+3), 0xff&*(addr+i+4),0xff&*(addr+i+5),0xff&*(addr+i+6),0xff&*(addr+i+7));
        i+=8;
        printf("%2x %2x %2x %2x %2x %2x %2x %2x \n",0xff&*(addr+i),0xff&*(addr+i+1),0xff&*(addr+i+2),0xff&*(addr+i+3), 0xff&*(addr+i+4),0xff&*(addr+i+5),0xff&*(addr+i+6),0xff&*(addr+i+7));
        i+=8;
    }
}

void CRNN_T::printMemoryI(void* in_addr, int intCount, int bar){
    if (M_QUIET)
        return;
    int i=0;
    int* addr;
    addr = (int*)in_addr;
//    printf("%X %X \n",addr, addr+1);
    while(i<intCount ){
        printf("%6d \t",*(addr+i));
        i++;
        if(i%bar==0)
            printf("\n");
    }
}
void CRNN_T::printMemoryD(void* in_addr, int intCount, int bar){
    if (M_QUIET)
        return;
    int i=0;
    double* addr;
    addr = (double*)in_addr;
//    printf("%X %X \n",addr, addr+1);
    while(i<intCount ){
        printf("%6f \t",*(addr+i));
        i++;
        if(i%bar==0)
            printf("\n");
    }
}
void CRNN_T::dbgPrint(string strIn){
    if(!M_QUIET)
        cout<<strIn<<endl;;
    cout.flush();
}
