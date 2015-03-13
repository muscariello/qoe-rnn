VPATH = ./src
CC = g++
#CFLAGS = -ggdb3
CFLAGS =
PROGS := rnnlite app_cli
all: clean $(PROGS)
rnnlite: main.o cseq.o crnn_t.o crnn.o 
	$(CC) $(CFLAGS) main.o cseq.o crnn_t.o crnn.o -o rnnlite
app_cli: app_cli.o crnn_t.o
	$(CC) $(CFLAGS) app_cli.o crnn_t.o -o app_cli
app_cli.o: app_cli.cpp
	$(CC) $(CFLAGS) -c $< -o app_cli.o
main.o: main.cpp 
	$(CC) $(CFLAGS) -c $< -o main.o 
cseq.o: cseq.cpp cseq.h
	$(CC) $(CFLAGS) -c $< -o cseq.o
crnn_t.o: crnn_t.cpp crnn_t.h argument_exception.h rnn_exception.h
	$(CC) $(CFLAGS) -c $< -o crnn_t.o 
crnn.o:	crnn.cpp crnn.h crnn_t.o cseq.o
	$(CC) $(CFLAGS) -c ./src/crnn.cpp crnn_t.o -o crnn.o

clean:
	rm -f rnnlite app_cli crnn_t.o crnn.o main.o app_cli.o cseq.o
