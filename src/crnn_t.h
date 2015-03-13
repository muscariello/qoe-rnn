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


#ifndef CRNN_T_H_INCLUDED
#define CRNN_T_H_INCLUDED


#include <iostream>
#include <fstream>
#include "argument_exception.h"
#include "rnn_exception.h"
using namespace std;
namespace rnn_lite{

//macro def for FLAG
#define M_TOPOLOGY 0X1
#define M_NORMALIZE 0X2
#define M_MODE 0x4
#define M_WEIGHT 0x8
#define M_TESTINPUT 0X10    //set if test data is loaded
#define M_VALIDATE 0X20     //set if validation MOS is loaded
#define M_TRAININPUT 0X40   //set if training data is loaded
#define M_TARGET 0X80       //set if target MOS is loaded

class CRNN_T
{
    public:
    enum RNN_MODE{IDLE,TRAINING,TEST};
    CRNN_T();
    ~CRNN_T();

    //************************************************************************
    //  [i] 1 or 0, set or unset M_RECURSIVE flag
    //  Recursive RNN. Still coding, don't use.
    void setRecursiveMode(int i);

    //************************************************************************
    //  [i] 1 or 0, set or unset M_DEBUG flag
    //  Mode for various test.
    //  Don't use
    void setDebugMode(int i);

    //************************************************************************
    //  [i] 1 or 0, set or unset M_QUIET flag
    //  Print no non-essential info on cout
    void setQuietMode(int i);

    //************************************************************************
    //  return 1 if is in Recursive mode
    //  return 0 if otherwise
    int isRecursiveMode(){return M_RECURSIVE; }

    //************************************************************************
    //  return 1 if is in Debug Mode
    //  return 0 otherwise
    int isDebugMode(){return M_DEBUG;    }

    //************************************************************************
    //  return 1 if is in Quiet Mode
    //  return 0 otherwise
    int isQuietMode(){return M_QUIET;    }

    //************************************************************************
    //  return value is of type RNN_MODE
    //  return the current operation mode
    int getMode();

    //************************************************************************
    //  [i] type RNN_MODE, desired Operation Mode
    //  set network Operation Mode to desired value
    //  will set variable CRNN_T::mode;
    void setMode(RNN_MODE i);

    //************************************************************************
    //  Obsolete. For cmdline app. with data provided directly as parameters.
    //  load external train/test data
    int loadInputCmd(string strInput, string strTarget = "") throw (argument_exception, rnn_exception);

    //************************************************************************
    //  [return_value]  calculated MSE from test data. range 0 to 25(if CRNN::score() is invoked)
    //  Do a predication test run with TEST_INPUT and corresponding weight matrix
    //  Input data, weight and nomalize_vector must be preloaded in order to use
    //  this method (use loadInput() and prepare_tst_patterns())
    double test() throw (rnn_exception);

    //************************************************************************
    //  [strIn]     string to print
    //  print debug to cout
    //  but print nothing is M_QUIET is set
    void dbgPrint(string strIn);

//    void printme(){cout<<"this is CRNN_T" <<endl;   }

//=================================================================================================================
// ***************************************************************************************************************
//                   public member variable interface
//                  should be customized before utlizing most public methods above
// ***************************************************************************************************************
//=================================================================================================================
    unsigned int FLAG;          //32 bit flag for state check. see macro definition for detail.
                                //corresponding flag will be set automatically when certain initiation routine is called
                                //otherwise certain methods such as CRNN::train() WILL refuse to operate
    string Out_File_Name;       //filename to save network output
    string Weights_File_Name;   //filename to store weight matrix (after training or before testing)
    fstream file_weight,file_out;        //stream obj for weight file

//=================================================================================================================
// ***************************************************************************************************************
//                   protected member variable and method
// ***************************************************************************************************************
//=================================================================================================================
protected:
//public:
    int load_weights() throw (argument_exception);         //load weight from external file
    int prepare_tst_patterns(); //parse TEST_INPUT into releated data structure
    int calculate_rate();       //calculate node firing rate.

    //************************************************************************
    //  [k]     index of data being examined, ie. the [k]th row in data file
    //  main calculation for Recursive RNN
    int solve_nonlinear_equations(int k);

    //************************************************************************
    //  [k]     index of data being examined, ie. the [k]th row in data file
    //  main calculation for FeedForward RNN
    int calculate_ffm_output(int k);

    //************************************************************************
    //  calculate the inverse of
    //  I-(wplus-wminus*q)/D
    //  result matrix stored in Winv.
    int calculate_inv();

    //************************************************************************
    //  [a]     double* serves as both input and output
    //  [n]     matrix dimension
    //  This will calculate the inverse of matrix [a] of size [n]*[n]
    //  Output will overwrite original [a] matrix
    int matrix_inv(double a[], int n);

    //************************************************************************
    //  [a]     double, float to be operated on
    //  [ret]   return abs() result of [a]
    //  This will calculate absolute value of [a]
    double my_abs(double a);

    //************************************************************************
    //  [a]     double*, input matrix a, size d1 * d2
    //  [b]     double*, input matrix b, size d2 * d3
    //  [d1]    integer, row count of matrix a;
    //  [d2]    integer, column count of matrix a, row count of matrix b;
    //  [d3]    integer, column count of matrix b;
    //  [return]    double*, the result matrix of size d1 * d3
    //  CAUTION the returned matrix memory is allocated by new(), so release the
    //  resource MANUALLY afterwards.
    double* matrix_mul(double a[], double b[], int d1, int d2, int d3);

    //************************************************************************
    //  [in_addr]   base address to be printed
    //  [intCount]  number of bytes to be printed
    //  print a bulk of memory, byte by byte in hex format
    void printMemory(void* in_addr, int intCount);

    //************************************************************************
    //  [in_addr]   base address to be printed
    //  [intCount]  number of double float structures to be printed
    //  [bar]       integer, indicate how many float to be printed per row
    //  print a bulk of memory in the format of double float.
    //  Very useful for checking float matrix value, ie. for a m*n matrix_a, use
    //      printMemoryD(matrix_a , m*n , n);
    void printMemoryD(void* in_addr, int intCount, int bar);

    //************************************************************************
    //  [in_addr]   base address to be printed
    //  [intCount]  number of integer structure to be printed
    //  [bar]       integer, indicate how many integer to be printed per row
    //  print a bulk of memory in the format of long int.
    //  Very useful for checking integer matrix value, ie. for a m*n matrix_a, use
    //      printMemoryI(matrix_a , m*n , n);
    void printMemoryI(void* in_addr, int intCount, int bar);



    //************************************************************************
    //  mode flag and M_XXX sub-option flags
    RNN_MODE mode;      //enumeration structure to record main operation mode:TRAINING|TEST|IDLE
    int M_DEBUG;        //test flag
    int M_RECURSIVE;    //recursive RNN flag
    int M_QUIET;        //no debug print

    int R_IN;               //fixed firing rate for input node if FIX_RIN flag is set.
    int FIX_RIN;
    double R_Out;       //Firing rate of output node in FF RNN
//    int AUTO_MAPPING;       //not used

    double* wplus, *wminus, *r, MSEaveg, *LAMBDA, *lambda, *y,*W,*Winv;

    double *Input_lambda, *Input_LAMBDA;
    double* TEST_INPUT,*OUTPUT;

    double *N,*D, *q;
    double* QP_train,*QP_target;

    int N_Total;            //Total number of nodes in network.
    int N_InputSize,N_OutputSize,N_HiddenSize;  //number of nodes in corresponding layer.
    int N_AllLayerCount;                        //how many layers in the network.
    int* N_LayerArray;                          //one-dimension vector to store node count in every layer.
    int N_TestPatternCount; //number of data set in test data.

    // one dimension vector of size [N_InputSize + N_OutputSize];
    // When scale all data into [0,1], store here the scaling factors
    double* Normalize_Vector;
};
}

#endif
