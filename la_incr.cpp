// -*- Mode: c++; c-file-style: "stroustrup"; -*-

using namespace std;

#include <stdio.h>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "vector.h"
#include "lasvm.h"

#define LINEAR  0
#define POLY    1
#define RBF     2
#define SIGMOID 3

#define ONLINE 0
#define ONLINE_WITH_FINISHING 1

#define RANDOM 0
#define GRADIENT 1
#define MARGIN 2

#define ITERATIONS 0
#define SVS 1
#define TIME 2

#if USE_FLOAT
# define real_t float
#else
# define real_t double
#endif

const char *kernel_type_table[] = {"linear","polynomial","rbf","sigmoid"};

class stopwatch
{
public:
	stopwatch() : start(std::clock()){} //start counting time
	~stopwatch();
	double get_time()
	{
		clock_t total = clock()-start;;
		return double(total)/CLOCKS_PER_SEC;
	};
private:
	std::clock_t start;
};
stopwatch::~stopwatch()
{
	clock_t total = clock()-start; //get elapsed time
	cout<<"Time(secs): "<<double(total)/CLOCKS_PER_SEC<<endl;
}
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



/* Data and model */
int m=0,history_size=0,msv;       // training set size
vector <lasvm_sparsevector_t*> X; // feature vectors
vector <int> Y;                   // labels
vector <double> kparam;           // kernel parameters
vector <double> alpha;            // alpha_i, SV weights
double b0;                        // threshold

/* Hyperparameters */
int kernel_type=RBF;              // LINEAR, POLY, RBF or SIGMOID kernels
double degree=3,kgamma=-1,coef0=0;// kernel params
int use_b0=1;                     // use threshold via constraint \sum a_i y_i =0
int selection_type=RANDOM;        // RANDOM, GRADIENT or MARGIN selection strategies
int optimizer=ONLINE_WITH_FINISHING; // strategy of optimization
double C=1;                       // C, penalty on errors
double C_neg=1;                   // C-Weighting for negative examples
double C_pos=1;                   // C-Weighting for positive examples
int epochs=1;                     // epochs of online learning
int candidates=50;				  // number of candidates for "active" selection process
double deltamax=1000;			  // tolerance for performing reprocess step, 1000=1 reprocess only
vector <double> select_size;      // Max number of SVs to take with selection strategy (for early stopping)
vector <double> x_square;         // norms of input vectors, used for RBF

/* Programm behaviour*/
int verbosity=1;                  // verbosity level, 0=off
int saves=1;
char report_file_name[1024];             // filename for the training report
char split_file_name[1024]="\0";         // filename for the splits
long long cache_size=256;               // 256Mb cache size as default
int incrmode = 0;						// no incremental mode as default
double epsgr=1e-3;                       // tolerance on gradients
long long kcalcs=0;                      // number of kernel evaluations
int binary_files=0;
vector <ID> splits;
int max_index=0;
vector <int> iold, inew;		  // sets of old (already seen) points + new (unseen) points
int termination_type=0;

void exit_with_help()
{
	fprintf(stdout,
			"Usage: la_incr [options] training_set_file model_file\n"
			"options:\n"
			"-i incremental mode : set increment training mode(default 2)\n"
			"	1 -- increment with persistence\n"
			"	2 -- increment without persistence(default)\n"
			"-B file format : files are stored in the following format:\n"
			"	0 -- libsvm ascii format (default)\n"
			"	1 -- binary format\n"
			"	2 -- split file format\n"
			"-o optimizer: set the type of optimization (default 1)\n"
			"	0 -- online \n"
			"	1 -- online with finishing step \n"
			"-s selection: set the type of selection strategy (default 0)\n"
			"	0 -- random \n"
			"	1 -- gradient-based \n"
			"	2 -- margin-based \n"
			"-T termination: set the type of early stopping strategy (default 0)\n"
			"	0 -- number of iterations \n"
			"	1 -- number of SVs \n"
			"	2 -- time-based \n"
			"-C candidates : set number of candidates to search for selection strategy (default 50)\n"
			"-m cachesize : set cache memory size in MB (default 256)\n"
			"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
			"-p epochs : number of epochs to train in online setting (default 1)\n"
			"-D deltamax : set tolerance for reprocess step, 1000=1 call to reprocess >1000=no calls to reprocess (default 1000)\n"
	);
	exit(1);
}


void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i; int clss; double weight;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
		case 'o':
			optimizer = atoi(argv[i]);
			break;
		case 's':
			selection_type = atoi(argv[i]);
			break;
		case 'm':
			cache_size = (int) atof(argv[i]);
			break;
		case 'i':
			incrmode = (int) atof(argv[i]);
			break;
		case 'B':
			binary_files=atoi(argv[i]);
			break;
		case 'e':
			epsgr = atof(argv[i]);
			break;
		case 'p':
			epochs = atoi(argv[i]);
			break;
		case 'D':
			deltamax = atoi(argv[i]);
			break;
		case 'C':
			candidates = atoi(argv[i]);
			break;
		case 'T':
			termination_type = atoi(argv[i]);
			break;
		default:
			fprintf(stderr,"unknown option\n");
			exit_with_help();
		}
	}

	saves=select_size.size();
	if(saves==0) select_size.push_back(100000000);

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
//		if(i%10==0)
//			printf("%d..",i);
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


int sv1,sv2; double max_alpha,alpha_tol;

int count_svs()
{
	int i;
	max_alpha=0;
	sv1=0;sv2=0;

	for(i=0;i<m;i++) 	// Count svs..
	{
		if(alpha[i]>max_alpha) max_alpha=alpha[i];
		if(-alpha[i]>max_alpha) max_alpha=-alpha[i];
	}

	alpha_tol=max_alpha/1000.0;

	for(i=0;i<m;i++)
	{
		if(Y[i]>0)
		{
			if(alpha[i] >= alpha_tol) sv1++;
		}
		else
		{
			if(-alpha[i] >= alpha_tol) sv2++;
		}
	}
	return sv1+sv2;
}

int libsvm_save_model(const char *model_file_name)
// saves the model in the same format as LIBSVM
{
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	count_svs();

	// printf("nSV=%d\n",sv1+sv2);
	fprintf(fp,"history_size %d\n",m);
	fprintf(fp,"C %lf %lf %lf\n",C,C_pos,C_neg);
	fprintf(fp,"svm_type c_svc\n");
	fprintf(fp,"kernel_type %s\n", kernel_type_table[kernel_type]);

	if(kernel_type == POLY)
		fprintf(fp,"degree %g\n", degree);

	if(kernel_type == POLY || kernel_type == RBF || kernel_type == SIGMOID)
		fprintf(fp,"gamma %g\n", kgamma);

	if(kernel_type == POLY || kernel_type == SIGMOID)
		fprintf(fp,"coef0 %g\n", coef0);

	fprintf(fp, "nr_class %d\n",2);
	fprintf(fp, "total_sv %d\n",sv1+sv2);

	{
		fprintf(fp, "rho %g\n",b0);
	}

	fprintf(fp, "label 1 -1\n");
	fprintf(fp, "nr_sv");
	fprintf(fp," %d %d",sv1,sv2);
	fprintf(fp, "\n");
	fprintf(fp, "SV\n");

	for(int j=0;j<2;j++)
		for(int i=0;i<m;i++)
		{
			if (j==0 && Y[i]==-1) continue;
			if (j==1 && Y[i]==1) continue;
			if (alpha[i]*Y[i]< alpha_tol) continue; // not an SV

			fprintf(fp, "%.16g ",alpha[i]);

			lasvm_sparsevector_pair_t *p1 = X[i]->pairs;
			while (p1)
			{
				fprintf(fp,"%d:%.8g ",p1->index,p1->data);
				p1 = p1->next;
			}
			fprintf(fp, "\n");
		}

	fclose(fp);
	return 0;
}

double kernel(int i, int j, void *kparam)
{
	double dot;
	kcalcs++;
	dot=lasvm_sparsevector_dot_product(X[i],X[j]);

	// sparse, linear kernel
	switch(kernel_type)
	{
	case LINEAR:
		return dot;
	case POLY:
		return pow(kgamma*dot+coef0,degree);
	case RBF:
		return exp(-kgamma*(x_square[i]+x_square[j]-2*dot));
	case SIGMOID:
		return tanh(kgamma*dot+coef0);
	}
	return 0;
}

void finish(lasvm_t *sv)
{
	int i,l;

	if (optimizer==ONLINE_WITH_FINISHING)
	{
		fprintf(stdout,"..[finishing]");

		int iter=0;

		do {
			iter += lasvm_finish(sv, epsgr);
		} while (lasvm_get_delta(sv)>epsgr);

	}

	l=(int) lasvm_get_l(sv);
	int *svind,svs; svind= new int[l];
	svs=lasvm_get_sv(sv,svind);
	alpha.resize(m);
	for(i=0;i<m;i++) alpha[i]=0;
	double *svalpha; svalpha=new double[l];
	lasvm_get_alpha(sv,svalpha);
	for(i=0;i<svs;i++) alpha[svind[i]]=svalpha[i];
	b0=lasvm_get_b(sv);
}

void make_old(int val)
// move index <val> from new set into old set
{
	int i,ind=-1;
	for(i=0;i<(int)inew.size();i++)
	{
		if(inew[i]==val) {ind=i; break;}
	}

	if (ind>=0)
	{
		inew[ind]=inew[inew.size()-1];
		inew.pop_back();
		iold.push_back(val);
	}
}

int select(lasvm_t *sv) // selection strategy
{
	int s=-1;
	int t,i,r,j;
	double tmp,best; int ind=-1;

	switch(selection_type)
	{
	case RANDOM:   // pick a random candidate
		s=rand() % inew.size();
		break;

	case GRADIENT: // pick best gradient from 50 candidates
		j=candidates; if((int)inew.size()<j) j=inew.size();
		r=rand() % inew.size();
		s=r;
		best=1e20;
		for(i=0;i<j;i++)
		{
			r=inew[s];
			tmp=lasvm_predict(sv, r);
			tmp*=Y[r];
			//printf("%d: example %d   grad=%g\n",i,r,tmp);
			if(tmp<best) {best=tmp;ind=s;}
			s=rand() % inew.size();
		}
		s=ind;
		break;

	case MARGIN:  // pick closest to margin from 50 candidates
		j=candidates; if((int)inew.size()<j) j=inew.size();
		r=rand() % inew.size();
		s=r;
		best=1e20;
		for(i=0;i<j;i++)
		{
			r=inew[s];
			tmp=lasvm_predict(sv, r);
			if (tmp<0) tmp=-tmp;
			//printf("%d: example %d   grad=%g\n",i,r,tmp);
			if(tmp<best) {best=tmp;ind=s;}
			s=rand() % inew.size();
		}
		s=ind;
		break;
	}

	t=inew[s];
	inew[s]=inew[inew.size()-1];
	inew.pop_back();
	iold.push_back(t);

	//printf("(%d %d)\n",iold.size(),inew.size());

	return t;
}

/*
 * Dump the current lasvm state into a file
 */
void dump_lasvm_state(lasvm_t *sv, char *model_file_name)
{

	char t[1024];
	strcpy(t,model_file_name);
	strcat(t,".lasvm");

	ofstream f;
	f.open(t,ios::trunc);
	if(f.fail())
	{
		fprintf(stderr,"Can't create lasvm state file \"%s\"\n",t);
		exit(1);
	}

	// Get each fields and dump along with counters
	// counters would be helpful for list while loading this data from file to structure

	//write it to text file
	f<<sv->sumflag<<endl;
	f<<sv->cp<<endl;
	f<<sv->cn<<endl;
	f<<sv->maxl<<endl;
	f<<sv->s<<endl;
	f<<sv->l<<endl;
	int length = sv->maxl;
	for(int i=0; i<length;i++)
		f<<sv->alpha[i]<<" ";
	f<<endl;

	for(int i=0; i<length;i++)
		f<<sv->cmin[i]<<" ";
	f<<endl;

	for(int i=0; i<length;i++)
		f<<sv->cmax[i]<<" ";
	f<<endl;

	for(int i=0; i<length;i++)
		f<<sv->g[i]<<" ";
	f<<endl;

	f<<sv->gmin<<endl;
	f<<sv->gmax<<endl;
	f<<sv->imin<<endl;
	f<<sv->imax<<endl;
	f<<sv->minmaxflag<<endl;

	f.close();
}

/*
 * Dump the current kernel cache state into a file
 */
void dump_kcache_state(lasvm_kcache_t *kcache, char *model_file_name)
{
	char t[1024];
	strcpy(t,model_file_name);
	strcat(t,".lacache");

	ofstream f;
	f.open(t,ios::out|ios::trunc);
	if(f.fail())
	{
		fprintf(stderr,"Can't create kcache file \"%s\"\n",t);
		exit(1);
	}
	// Get each fields and dump

	//write it to text file
	f<<kcache->maxsize<<endl;
	f<<kcache->cursize<<endl;
	f<<kcache->l<<endl;
	int length = kcache->l;

	for(int i=0; i<length;i++)
		f<<kcache->i2r[i]<<" ";
	f<<endl;

	for(int i=0; i<length;i++)
		f<<kcache->r2i[i]<<" ";
	f<<endl;

	for(int i=0; i<length;i++)
		f<<kcache->rsize[i]<<" ";
	f<<endl;

	for(int i=0; i<length;i++)
		f<<kcache->rdiag[i]<<" ";
	f<<endl;


	for(int i=0; i<length;i++)
		f<<kcache->rnext[i]<<" ";
	f<<endl;

	for(int i=0; i<length;i++)
			f<<kcache->rprev[i]<<" ";
	f<<endl;

	for(int i=0; i<length;i++)
			f<<kcache->qnext[i]<<" ";
	f<<endl;

	for(int i=0; i<length;i++)
			f<<kcache->qprev[i]<<" ";
	f<<endl;

	for(int i=0; i<length;i++)
	{
		for(int j=0;j<kcache->rsize[i];j++)
			f<<kcache->rdata[i][j]<<" ";
		if(kcache->rsize[i]!=0)
			f<<endl;
	}

	f.close();

}

/*
 * load kernel cache  state into respective data structure from a persistent file
 */
void load_kcache_state(lasvm_kcache_t * &kcache, char* model_file_name)
{
	char t[1024];
	strcpy(t,model_file_name);
	strcat(t,".lacache");

	ifstream f;
	f.open(t,ios::in);
	if(f.fail())
	{
		fprintf(stderr,"Error: Can't open kcache file \"%s\"\n",t);
		exit(1);
	}

	kcache=lasvm_kcache_create(kernel, NULL);

	if(kcache==NULL)
	{
		fprintf(stderr,"Error: Cannot load lasvm state \n");
		exit(1);
	}

	lasvm_kcache_set_maximum_size(kcache, cache_size*1024*1024);

	f >> kcache->maxsize;
	f >> kcache->cursize;


	//TODO experiment with cursize instead of maxsize
	if(cache_size*1024*1024<kcache->maxsize)
	{
		fprintf(stderr,"Error: Cache size smaller than previous\"%lld\"\n",kcache->maxsize);
		exit(1);
	}

	f >> kcache->l;

	int nl = kcache->l;

	//allocating memory first
	initialize_kcache_load(kcache);

	int i;
	for(i=0;i<nl;i++)
		f >> kcache->i2r[i];

	for(i=0;i<nl;i++)
		f >> kcache->r2i[i];

	for(i=0;i<nl;i++)
		f >> kcache->rsize[i];

	for(i=0;i<nl;i++)
		f >> kcache->rdiag[i];

	for(i=0;i<nl;i++)
		f >> kcache->rnext[i];

	for(i=0;i<nl;i++)
		f >> kcache->rprev[i];

	for(i=0;i<nl;i++)
		f >> kcache->qnext[i];

	for(i=0;i<nl;i++)
		f >> kcache->qprev[i];

	int maxsize = 0;
	int krsize = 0;
	for(i=0;i<nl;i++)
	{
		//	printf("rdata %d",i);fflush(stdout);
		krsize = kcache->rsize[i];
		if(krsize>maxsize)
			maxsize=krsize;

		if(krsize>0)
		{
			float *ndata = (float*)malloc(maxsize*sizeof(float));
			for(int j=0;j<krsize;j++)
				f >> ndata[j];
			kcache->rdata[i]=ndata;
		}
		else
			kcache->rdata[i]=NULL;
	}
	f.close();
	//return kcache;
}

/*
 * load lasvm state into respective data structure from a persistent file
 */
void load_lasvm_state(lasvm_t * &sv, lasvm_kcache_t *kcache, char* model_file_name)
{
	char t[1024];
	strcpy(t,model_file_name);
	strcat(t,".lasvm");

	ifstream f;
	f.open(t,ios::in);
	if(f.fail())
	{
		fprintf(stderr,"Can't open lasvm state file \"%s\"\n",t);
		exit(1);
	}

	int sumflag;
	f >> sumflag;

	//int cp,cn;
	f >> C_pos;
	f >> C_neg;


	//initialize
	sv=lasvm_create(kcache,sumflag,C*C_pos,C*C_neg);
	initialize_lasvm_load(sv);



	if(sv==NULL)
	{
		fprintf(stderr,"Cannot load lasvm state \n");
		exit(1);
	}

	//allocating memory first
	initialize_lasvm_load(sv);

	sv->sumflag = sumflag;
	sv->cp = C_pos;
	sv->cn = C_neg;
	f >> sv->maxl;
	f >> sv->s;
	f >> sv->l;

	int length=sv->maxl,i;

	for(i=0;i<length;i++)
		f >> sv->alpha[i];

	for(i=0;i<length;i++)
		f >> sv->cmin[i];

	for(i=0;i<length;i++)
		f >> sv->cmax[i];

	for(i=0;i<length;i++)
		f >> sv->g[i];

	f >> sv->gmin;
	f >> sv->gmax;
	f >> sv->imin;
	f >> sv->imax;
	f >> sv->minmaxflag;

	f.close();
	//return sv;
}

void train_online(char *model_file_name, char *temp_file_name)
{
	time_t st = time(NULL);
	time_t et;
	double timer=0;
	double elapsedTime = 0.0f;
	int t1,t2=0,i,s,l,j,k;
	//double timer=0;
	stopwatch *sw; // start measuring time after loading is finished
	sw=new stopwatch;    // save timing information
	char t[1000];
	strcpy(t,model_file_name);
	strcat(t,".time");

	lasvm_kcache_t *kcache;
	lasvm_t *sv;

	// everything is new when we start
	for(i=0;i<m;i++) inew.push_back(i);
	make_old(0);


	if(incrmode==2)//without persistence
	{
		kcache=lasvm_kcache_create(kernel, NULL);

		lasvm_kcache_set_maximum_size(kcache, cache_size*1024*1024);
		sv=lasvm_create(kcache,use_b0,C*C_pos,C*C_neg);
		printf("set cache size %lld\n",cache_size);

		// first add 5 examples of each class, just to balance the initial set
		int c1=0,c2=0;
		for(i=1;i<m;i++)
		{
			if(Y[i]==1 && c1<5) {lasvm_process(sv,i,(double) Y[i]); c1++; make_old(i);}
			if(Y[i]==-1 && c2<5){lasvm_process(sv,i,(double) Y[i]); c2++; make_old(i);}
			if(c1==5 && c2==5) break;
		}

		int index2;
		//process all the old SVs
		for(i=1;i<msv;i++)
		{
			if(inew.size()==0) break;

			if(i<=inew.size())
			{
				lasvm_process(sv,i,(double) Y[i]); c1++; make_old(i);
			}
			else
			{
				index2=inew.size()-1;
				lasvm_process(sv,index2,(double) Y[index2]);  make_old(index2);
			}
			//	lasvm_reprocess(sv,epsgr); //TODO: decide whether to reprocess or not
		}
	}
	else if(incrmode==1)//with persistence
	{
		load_kcache_state(kcache, model_file_name);
		lasvm_kcache_set_maximum_size(kcache, cache_size*1024*1024);
		printf("set cache size %lld\n",cache_size);
		load_lasvm_state(sv, kcache, model_file_name);
		sv->kernel = kcache;

		printf("\n--- data loaded--- \n");

		int index2;

		for(i=1;i<history_size;i++)
		{
			if(inew.size()==0) break;
		//	printf("%d..",i);fflush(stdout);
			if(i<=inew.size())
			{
				make_old(i);
			}
			else
			{
				index2=inew.size()-1;
				make_old(inew[index2]);
			}
			//	lasvm_reprocess(sv,epsgr); //TODO: decide whether to reprocess or not
		}
	}
	else
	{
		printf("Warning!! Not Running in Increment Mode \n");
	}

	for(j=0;j<epochs;j++)
	{
//		t2=lasvm_reprocess(sv,epsgr);

		for(i=history_size;i<m;i++)
		{
			if(inew.size()==0) break; // nothing more to select
			s=select(sv);            // selection strategy, select new point

			t1=lasvm_process(sv,s,(double) Y[s]);

			if (deltamax<=1000) // potentially multiple calls to reprocess..
			{
				//printf("%g %g\n",lasvm_get_delta(sv),deltamax);
				t2=lasvm_reprocess(sv,epsgr);// at least one call to reprocess
				while (lasvm_get_delta(sv)>deltamax && deltamax<1000)
				{
					t2=lasvm_reprocess(sv,epsgr);
				}
			}

			if (verbosity==2)
			{
				l=(int) lasvm_get_l(sv);
				printf("l=%d process=%d reprocess=%d\n",l,t1,t2);
			}
			else
				if(verbosity==1)
					if( (i%1000)==0){ fprintf(stdout, "..%d",i); fflush(stdout); }

			l=(int) lasvm_get_l(sv);
			for(k=0;k<(int)select_size.size();k++)
						{
							if   ( (termination_type==ITERATIONS && i==select_size[k])
									|| (termination_type==SVS && l>=select_size[k])
									|| (termination_type==TIME && sw->get_time()>=select_size[k])
							)

							{
								if(saves>1) // if there is more than one model to save, give a new name
										{
									// save current version before potential finishing step
									int save_l,*save_sv; double *save_g, *save_alpha;
									save_l=(int)lasvm_get_l(sv);
									save_alpha= new double[l];lasvm_get_alpha(sv,save_alpha);
									save_g= new double[l];lasvm_get_g(sv,save_g);
									save_sv= new int[l];lasvm_get_sv(sv,save_sv);

									finish(sv);
									char tmp[1000];

									timer+=sw->get_time();
									//f << i << " " << count_svs() << " " << kcalcs << " " << timer << endl;

									if(termination_type==TIME)
									{
										sprintf(tmp,"%s_%dsecs",model_file_name,i);
										fprintf(stdout,"..[saving model_%d secs]..",i);
									}
									else
									{
										fprintf(stdout,"..[saving model_%d pts]..",i);
										sprintf(tmp,"%s_%dpts",model_file_name,i);
									}
									libsvm_save_model(tmp);

									// get back old version
									//fprintf(stdout, "[restoring before finish]"); fflush(stdout);
									lasvm_init(sv, save_l, save_sv, save_alpha, save_g);
									delete save_alpha; delete save_sv; delete save_g;
									delete sw; sw=new stopwatch;    // reset clock
										}
								select_size[k]=select_size[select_size.size()-1];
								select_size.pop_back();
							}
						}
			if(select_size.size()==0) break; // early stopping, all intermediate models saved
		}

		inew.resize(0);iold.resize(0); // start again for next epoch..
		for(i=0;i<m;i++) inew.push_back(i); //TODO: Do we reprocess entire data or only new data?

		et = time(NULL);
		elapsedTime = difftime(et,st);
		std::cout <<std::endl<< elapsedTime<< " seconds. EPOCH="<<(j+1)<< std::endl;
	}

	if(saves<2)
	{
		finish(sv); // if haven't done any intermediate saves, do final save
		//timer+=sw->get_time();
		//f << m << " " << count_svs() << " " << kcalcs << " " << timer << endl;
	}

	if(verbosity>0) printf("\n");
	l=count_svs();
	printf("nSVs=%d\n",l);
	printf("||w||^2=%g\n",lasvm_get_w2(sv));
	printf("kcalcs="); cout << kcalcs << endl;
	//f.close();

	//persist the 2 structures if incrmode with persistence
	if(incrmode==1)
	{
		dump_kcache_state(kcache, model_file_name);
		dump_lasvm_state(sv, model_file_name);
	}

	lasvm_destroy(sv);
	lasvm_kcache_destroy(kcache);
}

void libsvm_save_history(char *temp_file_name, char *model_file_name)
{
	char t[1024];
	strcpy(t,model_file_name);
	strcat(t,".history");

	ofstream f;
	f.open(t,ios::out|ios::trunc);
	if(f.fail())
	{
		fprintf(stderr,"Can't create history file \"%s\"\n",t);
		exit(1);
	}

	ifstream histTemp;
	histTemp.open(temp_file_name);

	if(f.fail())
	{
		fprintf(stderr,"Can't Read History.temp file \"%s\"\n",temp_file_name);
		exit(1);
	}

	ofstream hist;
	hist.open(t,ios::out|ios::trunc);
	hist << histTemp.rdbuf();

	f.close();
}



int preprocess_sv_data(FILE *fp_temp, FILE *fp)
// loads the same format as LIBSVM
{
	int max_index; int oldindex=0;
	int index; double value; int i;

	int count = 0;

	max_index = 0;
	for(i=0;i<msv;i++)
	{
		count++;
		double label;
		fscanf(fp,"%lf",&label);

		//caution: converting alpha values with label 0 to +ve
		if(label>=0)
		{
			fprintf(fp_temp,"+1 ");
		}
		else
		{
			fprintf(fp_temp,"-1 ");
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
			if(index!=oldindex)
			{
				fprintf(fp_temp,"%d:%.8g ",index,value);
			}
			oldindex=index;
			if (index>max_index) max_index=index;
		}
		out2:
		fprintf(fp_temp,"\n");
	}

	return count;

}


int preprocess_model(FILE* fp_temp, const char *model_file_name)
{
	int i;

	FILE *fp = fopen(model_file_name,"r");


	if(fp == NULL)
	{
		fprintf(stderr,"Can't open model file \"%s\"\n",model_file_name);
		exit(1);
	}

	static char tmp[1001];
	double dtmp;
	fscanf(fp,"%1000s",tmp); //history_size
	fscanf(fp,"%d",&history_size); //history_size
	fscanf(fp,"%1000s",tmp); //text C
	fscanf(fp,"%lf %lf %lf",&C,&C_pos,&C_neg);//number C
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
		fscanf(fp,"%lf", &dtmp);
	}
	if(kernel_type == POLY || kernel_type == RBF || kernel_type == SIGMOID)
	{
		fscanf(fp,"%1000s",tmp);
		fscanf(fp,"%lf",&dtmp);
	}
	if(kernel_type == POLY || kernel_type == SIGMOID)
	{
		fscanf(fp,"%1000s",tmp);
		fscanf(fp,"%lf", &dtmp);
	}

	fscanf(fp,"%1000s",tmp); // nr_class
	fscanf(fp,"%1000s",tmp); // 2
	fscanf(fp,"%1000s",tmp); // total_sv
	fscanf(fp,"%d",&msv);

	fscanf(fp,"%1000s",tmp); //rho
	fscanf(fp,"%lf\n",&dtmp);

	fscanf(fp,"%1000s",tmp); // label
	fscanf(fp,"%1000s",tmp); // 1
	fscanf(fp,"%1000s",tmp); // -1
	fscanf(fp,"%1000s",tmp); // nr_sv
	fscanf(fp,"%1000s",tmp); // num
	fscanf(fp,"%1000s",tmp); // num
	fscanf(fp,"%1000s",tmp); // SV

	// now load SV data...

	int count = preprocess_sv_data(fp_temp, fp);

	// finished!

	fclose(fp);
	return count;
}

int append_data(FILE *fp_temp, const char *input_file_name)
{
	//hoping that the input data is sanctified ... it anyways needs to go through load_data util

	FILE *fp = fopen(input_file_name,"r");
	if(fp == NULL)
	{
		fprintf(stderr,"Can't open file \"%s\"\n",input_file_name);
		exit(1);
	}

	char line [ 4096 ]; /*Assuming 4096 chars max length of a line */
	int count=0;
	while ( fgets ( line, sizeof line, fp ) != NULL ) /* read a line */
	{
		count++;
		fputs ( line, fp_temp ); /* write the line */
	}

	fclose (fp);

	return count;
}

int preprocess_data(const char* temp_file_name, const char *input_file_name, const char *model_file_name)
{

	FILE *fp_temp = fopen(temp_file_name,"w");

	if(fp_temp==NULL)
	{
		fprintf(stderr,"Can't create temp file \"%s\"\n",temp_file_name);
		exit(1);
	}

	int examples_old=0,examples_incr=0;

	if(incrmode==2) //append new data to previous SVs
	{
		examples_old = preprocess_model(fp_temp, model_file_name);
		examples_incr = append_data(fp_temp, input_file_name);
	}
	else //append new data to historical data
	{
		char history_file_name[1024];
		strcpy(history_file_name,model_file_name);
		strcat(history_file_name,".history");

		examples_old = append_data(fp_temp, history_file_name);
		examples_incr = append_data(fp_temp, input_file_name);
	}

	fclose(fp_temp);

	return examples_incr+examples_old;
}

void setPreviousConfig(char* model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");


	if(fp == NULL)
	{
		fprintf(stderr,"Can't open model file \"%s\"\n",model_file_name);
		exit(1);
	}

	static char tmp[1001];
	fscanf(fp,"%1000s",tmp); //text history_size
	fscanf(fp,"%d",&history_size); //number history_size
	fscanf(fp,"%1000s",tmp); //text C
	fscanf(fp,"%lf %lf %lf",&C,&C_pos,&C_neg); //number C, C_pos, C_neg
	fscanf(fp,"%1000s",tmp); //text svm_type
	fscanf(fp,"%1000s",tmp);//c_svc
	fscanf(fp,"%1000s",tmp); //text kernel_type
	fscanf(fp,"%1000s",tmp);//kernel_type

	kernel_type=LINEAR;
	int i;
	for(i=0;i<4;i++)
		if (strcmp(tmp,kernel_type_table[i])==0)
		{
			kernel_type=i;
			break;
		}

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

	fclose(fp);
}

int main(int argc, char **argv)
{
	printf("\n");
	printf("la INCR\n");
	printf("______\n");
	time_t st = time(NULL);
	char input_file_name[1024];
	char model_file_name[1024];
	parse_command_line(argc, argv, input_file_name, model_file_name);

	char temp_file_name[1024];
	strcpy(temp_file_name,model_file_name);
	strcat(temp_file_name,".history.temp");
	setPreviousConfig(model_file_name);

	preprocess_data(temp_file_name, input_file_name, model_file_name);

	load_data_file(temp_file_name);

	printf("\n--- data loaded--- \n");

//printing if any error in loading data precautionary.
//	for(int i=0;i<m;i++)
//	{
//		printf("%d..%d..%lf\n",i,(X[i]->pairs)->index,(X[i]->pairs)->data);
//		if(i%50==0)
//		{
//		  printf("-----\n");
//		}
//		if(X[i]->pairs == NULL)
//		{
//			printf("Error at X[%d]..",i);
//			exit(1);
//		}
//	}

	train_online(model_file_name,temp_file_name);

	std::cout <<"Saving Model...." << std::endl;
	libsvm_save_model(model_file_name);


	if(incrmode==1)//only in persistence mode
	{
		libsvm_save_history(temp_file_name,model_file_name);
	}

	time_t et;
	et = time(NULL);
	double elapsedTime = difftime(et,st);
	std::cout << elapsedTime<< " seconds." << std::endl;
	//std::cout << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;
}
