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


#ifndef CRNN_H_INCLUDED
#define CRNN_H_INCLUDED


#include "crnn_t.h"
namespace rnn_lite{

class CRNN : public CRNN_T{
    public:
    CRNN();
    ~CRNN();

    //************************************************************************
    //  [i] 1 or 0, set or unset M_COMPACT flag
    //  In this mode train data and target data are stored in the same file.
    //  thus not necessary to provide target data seperately
    //  validation target can also be stored with test data. In this case test
    //  score will be calculated as well.
    void setCompactMode(int i);

    //************************************************************************
    //  [i] 1 or 0, set or unset M_CONTINUE flag
    //  In this mode the initial weight matrix will be read from external file
    //  instead of generating a new random one.
    void setContinueMode(int i);

    //************************************************************************
    //  [i] 1 or 0, set or unset M_CMD flag
    //  Mode for feeding input data as program parameters
    void setCmdMode(int i);

    //************************************************************************
    //  [i] double float, desired MSE threshold
    //  will set Mse_Threshold to designated value.
    //  Effective only when stop condition set to MSE threshold(By default).
    void setThreshold(double i){Mse_Threshold = i;    }

    //************************************************************************
    //  [i] integer, desired maximum iteration before forced break;
    //  will set variable CRNN::N_MaxIterations
    void setIteration(int i){N_MaxIterations = i;    }

    //************************************************************************
    //  return 1 if is in Compact Mode
    //  return 0 otherwise
    int isCompactMode(){return M_COMPACT;    }

    //************************************************************************
    //  return 1 if is in Continue Mode
    //  return 0 otherwise
    int isContinueMode(){return M_CONTINUE;    }

    //************************************************************************
    //  return 1 if input fed as parameters
    //  return 0 otherwise
    int isCmdMode(){return M_CMD;    }

     //************************************************************************
    //  load external train/test data
    int loadInput() throw (rnn_exception, argument_exception);

    //************************************************************************
    //! Reloaded method
    //  [return_value]  calculated MSE from test data. range 0 to 25(if CRNN::score() is invoked)
    //  Do a predication test run with TEST_INPUT and corresponding weight matrix
    //  Input data, weight and nomalize_vector must be preloaded in order to use
    //  this method (use loadInput() and prepare_tst_patterns())
    double test() throw (rnn_exception);

    //************************************************************************
    //  [MSE_RESULT]    pointer to a double variable receiving training MSE.
    //                  however if pointed variable is not zero its value will be
    //                  treated as MSE threshold. Training phase will loop until
    //                  MSE less than [MSE_RESULT] or reaches N_MaxIterations.
    //  [num_iter]      integer. Set this value to 0 if wish the training to run
    //                  with MSE threshold.
    //                  Set this value greater than 0 will set training stop condition
    //                  to iteration count. Training phase will loop until [num_iter]
    //                  iterations are finished.
    //  This method will use input data in TRAIN_INPUT and TARGET (solely TRAIN_INPUT
    //  in Compact Mode) to training RNN network with designated stop condition.
    //  Input data will first be normalized with Normalize_Vector
    //  Under both cases the resulting MSE after training will always be passed to
    //  [*MSE_RESULT]. And the trained weight matrix saved to external file.
    //  Input data and nomalize_vector must be preloaded in order to use this method
    //  (use CRNN::loadInput())
    int train(double* MSE_RESULT,int num_iter = 1) throw (rnn_exception);

    //************************************************************************
    //  Calculate Mean Square Error from test target data and RNN test output.
    //  Will automatically be called by CRNN::test() in Compact Mode
    //  Works only in CompactMode when test target data is included in test data
    //  Make sure the validity of test file and output file before usage.
    double score() throw (rnn_exception);

    //************************************************************************
    //  [strInput]  string to specify RNN architecture. For a 4 layer 4-5-3-2 network
    //              set the string as "4,5,3,2"
    //  [return_value]  calculated MSE from test data. range 0 to 25
    //  Set RNN architecture. Must be called before loadInput()
    int initialize_architecture(string strInput);

    //************************************************************************
    //  [strInput]  string to specify upper range of input data.Seperate by comma
    //              Ex: "1024,25,5,5"
    //              Vector should contain values from 1st input to last output.
    //              ie. the dimension should be N_inputSize+N_outputSize
    //  The program is making assumption that no data should be negative.
    //  Besides that it's best to normalize input to range [0,1]
    int initialize_normalization_vector(string strInput);

    void printme(){cout<<"this is CRNN"<<endl;    }

//=================================================================================================================
// ***************************************************************************************************************
//                   public member variable interface
//                  should be customized before utlizing most public methods above
// ***************************************************************************************************************
//=================================================================================================================

    string Trn_File_Name;       //filename to load train data
    string Tst_File_Name;       //filename to load test data
    string Tgt_File_Name;       //filename to load train target data
    string strLayout;           //store RNN architecture description such as "5,8,2"
    string strNormalize;        //store normalization vector such as "264,1024,10,35,25,5,5"
    fstream file_trn, file_tst, file_tgt; //corresponding fstream object

//=================================================================================================================
// ***************************************************************************************************************
//                   private member variable and method
// ***************************************************************************************************************
//=================================================================================================================
    private:
    int initialize_weights();   //generate random weight matrix
    int save_weights();         //save weight to external file
//    int prepare_tst_patterns(); //Reloaded parse TEST_INPUT into releated data structure
    int prepare_trn_patterns(); //parse TRAIN_INPUT into releated data structure

    //************************************************************************
    //  [k]     index of data being examined, ie. the [k]th row in data file
    //  cumulative calculation of MSE
    //  Given the MSE before counting in the [k]th data, this method returns
    //  the MSE of all k data sample. So be sure to set MSE_aveg to 0 before
    //  feeding the first data sample.
    double calculate_mse(int k);

    //************************************************************************
    //  [k]     index of data being examined, ie. the [k]th row in data file
    //  update wplus weight matrix, using gradient descent algorithm
    int update_wplus_1(int k);

    //************************************************************************
    //  [k]     index of data being examined, ie. the [k]th row in data file
    //  update minus weight matrix, using gradient descent algorithm
    int update_wminus_1(int k);

    //************************************************************************
    //  mode flag and M_XXX sub-option flags
    int M_COMPACT;      //input train/test file contains target value in the last few column.
    int M_CONTINUE;     //continue from last training flag
    int M_CMD;          //input prepared as program parameters

    /* following are for initialization*/
    double Mse_Threshold;
    double Eta;         //Gradient Descent step length
    int N_MaxIterations;    //maximum iteration limit before force break;
    double RAND_RANGE;  //Random generator range, from 0 to RAND_RANGE
    int N_Saved_Iterations; //round length before auto-save weight matrix

    double *Applied_lambda, *Applied_LAMBDA, *Applied_y;
    double *TRAIN_INPUT,*TARGET;

    int N_TrainPatternCount;

    // these file_pos is used to remember the data offset in input file in order to skip the file header
    streampos pos_data_trn, pos_data_tgt, pos_data_tst;
};
}

#endif
