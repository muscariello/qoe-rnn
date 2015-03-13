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

#ifndef CSEQ_H_INCLUDED
#define CSEQ_H_INCLUDED

class CSEQ{
public:

    long* output;   //output vector
    long length;   //sequence length

    CSEQ(int in_length);
    ~CSEQ();
    void work();
private:
    long* array;    //input vector(link table)
    long getnext(int index);    //return index of the next valid slot

};

#endif
