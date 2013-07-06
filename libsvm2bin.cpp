// -*- Mode: c++; c-file-style: "stroustrup"; -*-

using namespace std;

#include <stdio.h>  
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <cstring>

#include "vector.h"

vector <lasvm_sparsevector_t*> X; // feature vectors
vector <int> Y;                   // labels
int m;                            // number of examples
int sparse=1;
int	max_index = 0;

void libsvm_load_data(char *filename)
// loads the same format as LIBSVM
{ 
    int index; double value;
    int elements, i;
    FILE *fp = fopen(filename,"r");
    lasvm_sparsevector_t* v;

    if(fp == NULL)
    {
        fprintf(stderr,"Can't open input file \"%s\"\n",filename);
        exit(1);
    }
    else
        printf("\"%s\"..  ",filename);

    int msz = 0;
    elements = 0;
    while(1)
    {
        int c = fgetc(fp);
        switch(c)
        {
        case '\n':
            v=lasvm_sparsevector_create();
            X.push_back(v); 
            ++msz; 
            //printf("%d\n",m);
            elements=0;
            break;  
        case ':':
            ++elements;
            break;
        case EOF:
            goto out;
        default:
            ;
        }
    }
 out:
    rewind(fp);

    
    for(i=0;i<msz;i++)
    {
        int label;
        fscanf(fp,"%d",&label);
        Y.push_back(label);
        while(1) 
        {   
            int c;
            do {
                c = getc(fp);
                if(c=='\n') goto out2;
            } while(char(c)==' ');  //(isspace(c));
            ungetc(c,fp);
            fscanf(fp,"%d:%lf",&index,&value);
		
            lasvm_sparsevector_set(X[m+i],index,value);
            if (index>max_index) max_index=index;
        }	
    out2:
        label=1; // dummy
    }

    fclose(fp); 
	
    m=X.size();
    printf("examples: %d   features: %d\n",m,max_index);
}


void fullbin_save(char *fname)
{	
    int i=0,j;
    ofstream f;
    f.open(fname,ios::out|ios::binary);

    // write number of examples and number of features
    int sz[2]; sz[0]=m; sz[1]=max_index;
    f.write((char*)sz,2*sizeof(int));
    if (!f) { printf("File writing error in line %d.\n",i); exit(1);}
    
    float buf[max_index+1]; 
    for(i=0;i<m;i++) 
    {
        sz[0]=Y[i];       // write label
        f.write((char*)sz,1*sizeof(int));

        // write out features for each example
        for(j=0;j<max_index;j++) buf[j]=0;
        lasvm_sparsevector_pair_t *p = X[i]->pairs;
        while (p )
        { 
            if(p->index>max_index)
            {
                printf("error! index %d??\n",p->index); exit(1);
            }
            buf[p->index]=p->data;
            p = p->next; 	
        }
        f.write((char*)buf,max_index*sizeof(float));
    }
    f.close();
}



void bin_save(char *fname)
{	
    int i=0;
    ofstream f;
    f.open(fname,ios::out|ios::binary);

    // write number of examples and a 0 to say that the matrix is sparse
    int sz[2]; sz[0]=m; sz[1]=0;
    f.write((char*)sz,2*sizeof(int));
    if (!f) {printf("File writing error in line %d.\n",i); exit(1);}

    float buf[max_index];
    int   ind[max_index];
	
    for(i=0;i<m;i++)
    {   
        lasvm_sparsevector_pair_t *p = X[i]->pairs;
        max_index=0;
        while (p )
        { 
            //printf("%d:%g ",p->index,p->data);
            buf[max_index]=p->data;
            ind[max_index]=p->index;
            p = p->next; max_index++;	
        }

        sz[0]=Y[i];       // write label
        sz[1]=max_index;  // write length of example (how many nonsparse entries)
        f.write((char*)sz,2*sizeof(int));
        f.write((char*)ind,max_index*sizeof(int));   // write indices
        f.write((char*)buf,max_index*sizeof(float)); // write values
        if (!f) {printf("File writing error in line %d.\n",i); exit(1);}
    }
    f.close();
}



int main(int argc, char **argv)  
{
    printf("\n");
    printf("libsvm2bin file converter\n");
    printf("_________________________\n");
    
    if(argc<3 || (argc==3 && strcmp("-F",argv[1])==0))
    {
        printf("usage: %s [-F] <input file> <output file>\n",argv[0]);
        printf("	-F full matrix : forces full rather than sparse matrix storage.\n");
        exit(0);
    }

    if(strcmp(argv[1],"-F")==0)  
    {
        sparse=0;printf("[storing as a full matrix]\n");
        libsvm_load_data(argv[2]);
        fullbin_save(argv[3]);
    }
    else
    {
        libsvm_load_data(argv[1]);
        bin_save(argv[2]);
    }	
}


