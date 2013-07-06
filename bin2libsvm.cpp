// -*- Mode: c++; c-file-style: "stroustrup"; -*-

using namespace std;

#include <stdio.h>  
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <cmath>

#include "vector.h"

vector <lasvm_sparsevector_t*> X; // feature vectors
vector <int> Y;                   // labels
int m;                            // number of examples
int sparse=1;
int max_index = 0;

int binary_load_data(char *filename)
{
    int msz,i=0,j;
    lasvm_sparsevector_t* v;
    int nonsparse=0;

    ifstream f;
    f.open(filename,ios::in|ios::binary);
    
    // read number of examples and number of features
    int sz[2]; 
    f.read((char*)sz,2*sizeof(int));
    if (!f) { printf("File writing error in line %d.\n",i); exit(1);}
    msz=sz[0]; max_index=sz[1];

    vector <float> val;
    vector <int>   ind;
    val.resize(max_index);
    if(max_index>0) nonsparse=1;
    
    for(i=0;i<msz;i++) 
    {
        v=lasvm_sparsevector_create(); 
        X.push_back(v);
        if(nonsparse) // non-sparse binary file
        {
            f.read((char*)sz,1*sizeof(int)); // get label
            Y.push_back(sz[0]);
            f.read((char*)(&val[0]),max_index*sizeof(float));
            for(j=0;j<max_index;j++) // set features for each example
                lasvm_sparsevector_set(v,j,val[j]);
        }
        else			// sparse binary file
        {
            f.read((char*)sz,2*sizeof(int)); // get label & sparsity of example i
            Y.push_back(sz[0]);
            val.resize(sz[1]); 
            ind.resize(sz[1]);
            f.read((char*)(&ind[0]),sz[1]*sizeof(int));
            f.read((char*)(&val[0]),sz[1]*sizeof(float));
            for(j=0;j<sz[1];j++) // set features for each example
            {
                if (val[j]!=0)
                    lasvm_sparsevector_set(v,ind[j],val[j]);
                if(ind[j]>max_index)
                    max_index=ind[j];
            }
        }		
    }
    f.close();
    
    msz=X.size();
    printf("examples: %d   features: %d\n",msz,max_index);
    
    return msz;
}


void libsvm_save(char *fname)
{	
    FILE *f = fopen(fname,"w");
    for(int i=0;i<m;i++) 
    {
        fprintf(f,"%d", (int) Y[i]);
        lasvm_sparsevector_pair_t *p = X[i]->pairs;
        while (p )
        { 
            if(p->index>max_index)
            {
                printf("error! index %d??\n",p->index); 
                exit(1);
            }
            fprintf(f," %d:%.15g", p->index, p->data);
            p = p->next; 	
        } 
        fprintf(f,"\n");
    }
    fclose(f);
}

int main(int argc, char **argv)  
{
    printf("\n");
    printf("bin2libsvm file converter\n");
    printf("_________________________\n");
    
    if(argc!=3)
    {
        printf("usage: %s <input file> <output file>\n",argv[0]);
        exit(10);
    }
    
    m = binary_load_data(argv[1]);
    libsvm_save(argv[2]);
    return 0;
}


