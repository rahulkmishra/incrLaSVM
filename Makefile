#Makefile for la_svm software (c) 2005 
#SHELL=/bin/sh
CXX= g++
CFLAGS= -g 
TOOLS= vector.c messages.c kernel.c kcache.c lasvm.c
TOOLSINCL= vector.h messages.h kernel.h kcache.h lasvm.h
LASVMSRC= la_svm.cpp
LATESTSRC= la_test.cpp
LAINCRSRC= la_incr.cpp
LSVM2BINSRC= libsvm2bin.cpp
BIN2LSVMSRC= bin2libsvm.cpp

all: la_svm la_incr la_test libsvm2bin bin2libsvm

la_svm: $(LASVMSRC) $(TOOLS) $(TOOLSINCL)
	$(CXX) $(CFLAGS) -o la_svm $(LASVMSRC) $(TOOLS)  -lm

la_incr: $(LAINCRSRC) $(TOOLS) $(TOOLSINCL)
	$(CXX) $(CFLAGS) -o la_incr $(LAINCRSRC) $(TOOLS)  -lm
	
la_test: $(LATESTSRC) $(TOOLS) $(TOOLSINCL)
	$(CXX) $(CFLAGS) -o la_test $(LATESTSRC) $(TOOLS)  -lm

libsvm2bin: $(LSVM2BINSRC) $(TOOLS) $(TOOLSINCL)
	$(CXX) $(CFLAGS) -o libsvm2bin $(LSVM2BINSRC) $(TOOLS) -lm

bin2libsvm: $(BIN2LSVMSRC) $(TOOLS) $(TOOLSINCL)
	$(CXX) $(CFLAGS) -o bin2libsvm $(BIN2LSVMSRC) $(TOOLS) -lm

clean: FORCE
	rm 2>/dev/null la_svm la_incr la_test libsvm2bin bin2libsvm

FORCE:

