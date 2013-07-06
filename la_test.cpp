// -*- mode: c++; c-file-style: "stroustrup"; -*-


using namespace std;
#include <stdio.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <cctype>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "vector.h"

#define LINEAR  0
#define POLY    1
#define RBF     2
#define SIGMOID 3 

const char *kernel_type_table[] = {"linear","polynomial","rbf","sigmoid"};

class ID // class to hold split file indices and labels
{
public:
	int x;
	int y;
	ID() : x(0), y(0) {}
	ID(int x1,int y1) : x(x1), y(y1) {}
};
// IDs will be sorted by index, not by label.
bool operator<(const ID& x, const ID& y)
{
	return x.x < y.x;
}

int m,msv,history_size;                         // training and test set sizes
vector <lasvm_sparsevector_t*> X; // feature vectors for test set
vector <lasvm_sparsevector_t*> Xsv;// feature vectors for SVs
vector <int> Y;                   // labels
vector <double> alpha;            // alpha_i, SV weights
double b0;                        // threshold
int use_b0=1;                     // use threshold via constraint \sum a_i y_i =0
int kernel_type=RBF;              // LINEAR, POLY, RBF or SIGMOID kernels
double degree=3,kgamma=-1,coef0=0;// kernel params
vector <double> x_square;         // norms of input vectors, used for RBF
vector <double> xsv_square;        // norms of test vectors, used for RBF
char split_file_name[1024]="\0";         // filename for the splits
int binary_files=0;
vector <ID> splits;             
int max_index=0;




void exit_with_help()
{
	fprintf(stdout,
			"\nUsage: la_test [options] test_set_file model_file output_file\n"
			"options:\n"
			"-B file format : files are stored in the following format:\n"
			"	0 -- libsvm ascii format (default)\n"
			"	1 -- binary format\n"
			"	2 -- split file format\n");

	exit(1);
}





int split_file_load(char *f)
{
	int binary_file=0,labs=0,inds=0;
	FILE *fp;
	fp=fopen(f,"r");

	if(fp==NULL) {printf("[couldn't load split file: %s]\n",f); exit(1);}
	char dummy[100],dummy2[100];
	unsigned int i,j=0; for(i=0;i<strlen(f);i++) if(f[i]=='/') j=i+1;
	fscanf(fp,"%s %s",dummy,dummy2);
	strcpy(&(f[j]),dummy2);

	fscanf(fp,"%s %d",dummy,&binary_file);
	fscanf(fp,"%s %d",dummy,&inds);
	fscanf(fp,"%s %d",dummy,&labs);
	printf("[split file: load:%s binary:%d new_indices:%d new_labels:%d]\n",dummy2,binary_file,inds,labs);
	//printf("[split file:%s binary=%d]\n",dummy2,binary_file);
	if(!inds) return binary_file;
	while(1)
	{
		int i,j;
		int c=fscanf(fp,"%d",&i);
		if(labs) c=fscanf(fp,"%d",&j);
		if(c==-1) break;
		if (labs)
			splits.push_back(ID(i-1,j)); 
		else 
			splits.push_back(ID(i-1,0));
	}

	sort(splits.begin(),splits.end());

	return binary_file;
}


int libsvm_load_data(char *filename)
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
		printf("loading \"%s\"..  \n",filename);
	int splitpos=0;

	int msz = 0;
	elements = 0;
	while(1)
	{
		int c = fgetc(fp);
		switch(c)
		{
		case '\n':
		if(splits.size()>0)
		{
			if(splitpos<(int)splits.size() && splits[splitpos].x==msz)
			{
				v=lasvm_sparsevector_create();
				X.push_back(v);	splitpos++;
			}
		}
		else
		{
			v=lasvm_sparsevector_create();
			X.push_back(v);
		}
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


	max_index = 0;splitpos=0;
	for(i=0;i<msz;i++)
	{

		int write=0;
		if(splits.size()>0)
		{
			if(splitpos<(int)splits.size() && splits[splitpos].x==i)
			{
				write=2;splitpos++;
			}
		}
		else
			write=1;

		int label;
		fscanf(fp,"%d",&label);
		//	printf("%d %d\n",i,label);
		if(write)
		{
			if(splits.size()>0)
			{
				if(splits[splitpos-1].y!=0)
					Y.push_back(splits[splitpos-1].y);
				else
					Y.push_back(label);
			}
			else
				Y.push_back(label);
		}

		while(1)
		{
			int c;
			do {
				c = getc(fp);
				if(c=='\n') goto out2;
			} while(isspace(c));
			ungetc(c,fp);
			fscanf(fp,"%d:%lf",&index,&value);

			if (write==1) lasvm_sparsevector_set(X[m+i],index,value);
			if (write==2) lasvm_sparsevector_set(X[splitpos-1],index,value);
			if (index>max_index) max_index=index;
		}
		out2:
		label=1; // dummy
	}

	fclose(fp);

	msz=X.size()-m;
	printf("examples: %d   features: %d\n",msz,max_index);

	return msz;
}

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
	int splitpos=0;

	for(i=0;i<msz;i++)
	{
		int mwrite=0;
		if(splits.size()>0)
		{
			if(splitpos<(int)splits.size() && splits[splitpos].x==i)
			{
				mwrite=1;splitpos++;
				v=lasvm_sparsevector_create(); X.push_back(v);
			}
		}
		else
		{
			mwrite=1;
			v=lasvm_sparsevector_create(); X.push_back(v);
		}

		if(nonsparse) // non-sparse binary file
				{
			f.read((char*)sz,1*sizeof(int)); // get label
			if(mwrite)
			{
				if(splits.size()>0 && splits[splitpos-1].y!=0)
					Y.push_back(splits[splitpos-1].y);
				else
					Y.push_back(sz[0]);
			}
			f.read((char*)(&val[0]),max_index*sizeof(float));
			if(mwrite)
				for(j=0;j<max_index;j++) // set features for each example
					lasvm_sparsevector_set(v,j,val[j]);
				}
		else			// sparse binary file
		{
			f.read((char*)sz,2*sizeof(int)); // get label & sparsity of example i
			if(mwrite)
			{
				if(splits.size()>0 && splits[splitpos-1].y!=0)
					Y.push_back(splits[splitpos-1].y);
				else
					Y.push_back(sz[0]);
			}
			val.resize(sz[1]); ind.resize(sz[1]);
			f.read((char*)(&ind[0]),sz[1]*sizeof(int));
			f.read((char*)(&val[0]),sz[1]*sizeof(float));
			if(mwrite)
				for(j=0;j<sz[1];j++) // set features for each example
				{
					lasvm_sparsevector_set(v,ind[j],val[j]);
					//printf("%d=%g\n",ind[j],val[j]);
					if(ind[j]>max_index) max_index=ind[j];
				}
		}
	}
	f.close();

	msz=X.size()-m;
	printf("examples: %d   features: %d\n",msz,max_index);

	return msz;
}


void load_data_file(char *filename)
{
	int msz,i,ft;
	splits.resize(0);

	int bin=binary_files;
	if(bin==0) // if ascii, check if it isn't a split file..
	{
		FILE *f=fopen(filename,"r");
		if(f == NULL)
		{
			fprintf(stderr,"Can't open input file \"%s\"\n",filename);
			exit(1);
		}
		char c; fscanf(f,"%c",&c);
		if(c=='f') bin=2; // found split file!
	}

	switch(bin)  // load diferent file formats
	{
	case 0: // libsvm format
		msz=libsvm_load_data(filename); break;
	case 1:
		msz=binary_load_data(filename); break;
	case 2:
		ft=split_file_load(filename);
		if(ft==0)
		{msz=libsvm_load_data(filename); break;}
		else
		{msz=binary_load_data(filename); break;}
	default:
		fprintf(stderr,"Illegal file type '-B %d'\n",bin);
		exit(1);
	}

	if(kernel_type==RBF)
	{
		x_square.resize(m+msz);
		for(i=0;i<msz;i++)
			x_square[i+m]=lasvm_sparsevector_dot_product(X[i+m],X[i+m]);
	}

	if(kgamma==-1)
		kgamma=1.0/ ((double) max_index); // same default as LIBSVM

	m+=msz;
}



void libsvm_load_sv_data(FILE *fp)
// loads the same format as LIBSVM
{ 
	int max_index; int oldindex=0;
	int index; double value; int i;
	lasvm_sparsevector_t* v;

	alpha.resize(msv);
	for(i=0;i<msv;i++)
	{
		v=lasvm_sparsevector_create();
		Xsv.push_back(v);
	}

	max_index = 0;
	for(i=0;i<msv;i++)
	{
		double label;
		fscanf(fp,"%lf",&label);
		//printf("%d:%g\n",i,label);
		alpha[i] = label;
		while(1)
		{
			int c;
			do {
				c = getc(fp);
				if(c=='\n') goto out2;
			} while(isspace(c));
			ungetc(c,fp);
			fscanf(fp,"%d:%lf",&index,&value);
			if(index!=oldindex)
			{
				lasvm_sparsevector_set(Xsv[i],index,value);
			}
			oldindex=index;
			if (index>max_index) max_index=index;
		}
		out2:
		label=1; // dummy
	}

	printf("loading model: %d svs\n",msv);

	if(kernel_type==RBF)
	{
		xsv_square.resize(msv);
		for(i=0;i<msv;i++)
			xsv_square[i]=lasvm_sparsevector_dot_product(Xsv[i],Xsv[i]);
	}

}




int libsvm_load_model(const char *model_file_name)
// saves the model in the same format as LIBSVM
{
	int i;

	FILE *fp = fopen(model_file_name,"r");

	if(fp == NULL)
	{
		fprintf(stderr,"Can't open input file \"%s\"\n",model_file_name);
		exit(1);
	}

	static char tmp[1001];
	double dtmp;
	fscanf(fp,"%1000s",tmp); //text history_size
	fscanf(fp,"%d",&history_size);//number history_size
	fscanf(fp,"%1000s",tmp); //text C
	fscanf(fp,"%lf %lf %lf",&dtmp,&dtmp,&dtmp); //number C
	fscanf(fp,"%1000s",tmp); //svm_type
	fscanf(fp,"%1000s",tmp); //c_svc
	fscanf(fp,"%1000s",tmp); //kernel_type
	fscanf(fp,"%1000s",tmp); //rbf,poly,..

	kernel_type=LINEAR;
	for(i=0;i<4;i++)
		if (strcmp(tmp,kernel_type_table[i])==0) kernel_type=i;

	if(kernel_type == POLY)
	{
		fscanf(fp,"%1000s",tmp);
		fscanf(fp,"%lf", &degree);
	}
	if(kernel_type == POLY || kernel_type == RBF || kernel_type == SIGMOID)
	{
		fscanf(fp,"%1000s",tmp);
		fscanf(fp,"%lf",&kgamma);
	}
	if(kernel_type == POLY || kernel_type == SIGMOID)
	{
		fscanf(fp,"%1000s",tmp);
		fscanf(fp,"%lf", &coef0);
	}

	fscanf(fp,"%1000s",tmp); // nr_class
	fscanf(fp,"%1000s",tmp); // 2
	fscanf(fp,"%1000s",tmp); // total_sv
	fscanf(fp,"%d",&msv);

	fscanf(fp,"%1000s",tmp); //rho
	fscanf(fp,"%lf\n",&b0);

	fscanf(fp,"%1000s",tmp); // label
	fscanf(fp,"%1000s",tmp); // 1
	fscanf(fp,"%1000s",tmp); // -1
	fscanf(fp,"%1000s",tmp); // nr_sv
	fscanf(fp,"%1000s",tmp); // num
	fscanf(fp,"%1000s",tmp); // num
	fscanf(fp,"%1000s",tmp); // SV

	// now load SV data...

	libsvm_load_sv_data(fp);

	// finished!

	fclose(fp);
	return 0;
}

double kernel(int i, int j, void *kparam)
{
	double dot;
	dot=lasvm_sparsevector_dot_product(X[i],Xsv[j]);

	// sparse, linear kernel
	switch(kernel_type)
	{
	case LINEAR:
		return dot;
	case POLY:
		return pow(kgamma*dot+coef0,degree);
	case RBF:
		return exp(-kgamma*(x_square[i]+xsv_square[j]-2*dot));
	case SIGMOID:
		return tanh(kgamma*dot+coef0);
	}
	return 0;
}  


void test(char *output_name)
{	
	FILE *fp=fopen(output_name,"w");
	int i,j; double y; double acc=0;
	double err = 0, TP = 0, TN = 0, FP = 0, FN = 0;
	for(i=0;i<m;i++)
	{
		y=-b0;
		for(j=0;j<msv;j++)
		{
			y+=alpha[j]*kernel(i,j,NULL);
		}
		if(y>=0) y=1; else y=-1;

		if(((int)y) == Y[i])
		{
			acc++;
			if(y==1)
				TP++;
			else
				TN++;
		}
		else
		{
			err++;
			if(y==1)
				FP++;
			else
				FN++;
		}
	}

	double precision = TP/(TP+FP);
	double recall = TP/(TP+FN);
	double fscore = 2*precision*recall/(precision+recall);

	printf("accuracy= %g (%d/%d)\n",(acc/m)*100,((int)acc),m);

	std::cout << "TP: " << TP << "\tFP: " << FP << std::endl;
	std::cout << "FN: " << FN << "\tTN: " << TN << std::endl;
	std::cout << "Precision: " << precision << "\tRecall: " << recall << "\nF-score: " << fscore << std::endl;
	fclose(fp);
}


void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name, char *output_file_name)
{
	int i;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
		case 'B':
			binary_files=atoi(argv[i]);
			break;
		default:
			fprintf(stderr,"unknown option\n");
			exit_with_help();
		}
	}

	// determine filenames

	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}

	if(argc<i+3) exit_with_help();

	strcpy(input_file_name, argv[i]);
	strcpy(model_file_name, argv[i+1]);
	strcpy(output_file_name, argv[i+2]);

}


int main(int argc, char **argv)  
{

	printf("\n");
	printf("la test\n");
	printf("_______\n");
	clock_t startTime = clock();
	char input_file_name[1024];
	char model_file_name[1024];
	char output_file_name[1024];
	parse_command_line(argc, argv, input_file_name, model_file_name, output_file_name);

	libsvm_load_model(model_file_name);// load model
	load_data_file(input_file_name); // load test data

	test(output_file_name);
	std::cout << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;
}


