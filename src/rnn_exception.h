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


#ifndef RNN_EXCEPTION_H
#define RNN_EXCEPTION_H


#include <exception>
#include <string>

namespace rnn_lite
{
    class rnn_exception : public std::exception
    {
        std::string m_arg;

    public:
        rnn_exception( const std::string& arg ) throw ()
        : m_arg( arg ) {}

    virtual ~rnn_exception() throw() {}

    virtual const char* what() const throw ()
        { return m_arg.c_str(); }
    };
}
#endif // RNN_EXCEPTION_H
