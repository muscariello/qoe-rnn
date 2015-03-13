/**********************************************************************************
Class used to generate randomized index sequence, 0 is excluded
Constructor use parameter <in_length> to specify sequence length
After calling CSEQ::work() routine, randomized vector will be available at CSEQ::output

[Comment]
    0 is excluded. You may need to do a "-1" operation manually to all members to obtained
index starting from 0;
    This class utilized rand(), so one may use srand(int seed) to initialize random seed.
    Be caution while using public variable output[] since it will be recycled upon class
deconstruction. Copy its content to a safe place if necessary.
    One may call CSEQ::work() for as many times as he/she desire, given that vector length
remains constant. Otherwise, create another class instance.

[Example]
    CSEQ abc(10);
    abc.work();
    for(int i=0;i<10;i++)
        cout<<abd.output[i]<<"\t";

    Will get output like
    "3  7   5   2   4   1   9   6   10  8"

***********************************************************************************/

#include "cseq.h"
#include "cstdlib"
CSEQ::CSEQ(int in_length){
    length = in_length;
    array = new long[length];
    output = new long [length];
}
CSEQ::~CSEQ(){
    delete[] array;
    delete[] output;
}

void CSEQ::work(){
    int index = 0;
    int indexn = 0;
    for(int i=0;i<length;i++)
        array[i] = i+1;         //all links point to next adjacent slot
    array[length-1]=0;          //make it a loop
    for(int i=0;i<length;i++){
        index = rand()%length;          //grab random slot
        if (0 == array[index]>>31){     //current slot is available
            output[i] = index+1;        //push to output
            array[index] |= 0x80000000; //mark current slot as occupied
        }
        else{
            indexn = getnext(index);    //look for next available slot
            output[i] = indexn+1;       //push lookup result to output
        }
    }
}

//return next available index
long CSEQ::getnext(int index){
    int indexn=0;
    if (0 == array[index]>>31){         //current slot is available
        array[index] |= 0x80000000;     //mark as occupied
        return index;                   //push to return value
    }
    else{                               //current slot is occupied
        indexn = getnext(array[index]&0x7fffffff);    //look for next available slot via next_available link
        array[index] = array[indexn];   //update next_available link with lookup result;
        return indexn;                  //push to return value
    }
}
