// Copyright (C) 2017 Zeke Wang- Systems Group, ETH Zurich

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//*************************************************************************
/*
zipml_data_representation is mainly 
*/
#ifndef ZIP_SGD_PM_CPP
#define ZIP_SGD_PM_CPP

#include "sgd_pm.h"

#define AVX2_EN
#define CPU_BINDING_EN

#ifdef AVX2_EN
#include "hazy/vector/mlweaving_avx2.h"      //mlweaving_8mr mlweaving_8mr_avx2     mlweaving mlweaving_avx2
#include "hazy/vector/dot_mlweaving_avx2.h"      //mlweaving_8mr mlweaving_8mr_avx2     mlweaving mlweaving_avx2

#include "hazy/vector/operations-inl_avx2.h"
#include "hazy/vector/scale_add-inl_avx2.h"
#include "hazy/vector/dot-inl_avx2.h"        //#include "hazy/vector/dot-inl.h" //
#else
#include "hazy/vector/operations-inl.h"
#include "hazy/vector/scale_add-inl.h"
#include "hazy/vector/dot-inl.h"
#endif

#ifdef CPU_BINDING_EN
#include "hazy/thread/thread_pool-inl_binding.h"
#else
#include "hazy/thread/thread_pool-inl.h"
#endif

#include "hazy/util/clock.h"
//#include "utils.h"


#include "perf_counters.h"
struct Monitor_Event inst_Monitor_Event = {
  {
    {0x2e,0x41},
    {0x24,0x21},
    {0xc5,0x00},
    {0x24,0x41},
  },
  1,
  {
    "L3 cache misses: ",
    "L2 cache misses: ",
    "Mispredicted branchs: ",
    "L2 cache hits: ",
  },
  {
    {0,0},
    {0,0},
    {0,0},
    {0,0},    
  },
  2,
  {
    "MIC_0",
    "MIC_1",
    "MIC_2",
    "MIC_3",
  },
    0  
};




using namespace std;

inline size_t GetStartIndex(size_t total, unsigned tid, unsigned nthreads) {
  return (total / nthreads) * tid;
}

/*! Returns the ending index + 1 for the given thread to use
 * Loop using for (size_t i = GetStartIndex; i < GetEndIndex(); i++)
 * \param total the total number of examples
 * \param tid the thread id [0, 1, 2, ...]
 * \param total number of threads
 * \return last index to process PLUS ONE
 */
inline size_t GetEndIndex(size_t total, unsigned tid, unsigned nthreads) {
  size_t block_size = total / nthreads;
  size_t start = block_size * tid;
  if (nthreads == tid+1) {
    return total;
  }
  if (start + block_size > total) {
    return total;
  }
  return start + block_size;
}


////////////////////////Zip_sgd constructor/////////////////////
zipml_sgd_pm::zipml_sgd_pm(bool usingFPGA, uint32_t _b_toIntegerScaler) {

	srand(7); //for whta..

	dr_a  = NULL;  dr_a_norm = NULL; a_bitweaving_fpga = NULL; 
	dr_bi = NULL; dr_bi   = NULL;

	dr_a_min = NULL; dr_a_max = NULL; 

	dr_numFeatures = 0;
	dr_numSamples  = 0;

	b_toIntegerScaler = _b_toIntegerScaler;
	
	if (usingFPGA)
	{
		if (gotFPGA == 0) {  //1: should be this one...
			myfpga = new FPGA();
			gotFPGA = 1;
		}
		else
			gotFPGA = 0;			
	}

}

////////////////////////Zip_sgd destructor/////////////////////
zipml_sgd_pm::~zipml_sgd_pm() {
	if (dr_a != NULL)
		free(dr_a);

	//if (dr_a_norm != NULL)
	//	free(dr_a_norm);
    //How to 

	if (dr_a_min != NULL)
		free(dr_a_min);

	if (dr_a_max != NULL)
		free(dr_a_max);

	if (dr_b != NULL)
		free(dr_b);

	if (dr_bi != NULL)
		free(dr_bi);

	if (gotFPGA == 1) //should delete the class....
		delete myfpga;
/**/		
}

void zipml_sgd_pm::load_dense_data(char* pathToFile, uint32_t numSamples, uint32_t numFeatures)
{
	cout << "Reading " << pathToFile << endl;

	dr_numSamples 		= numSamples;
	dr_numFeatures		= numFeatures; // For the bias term
	dr_numFeatures_algin= ((dr_numFeatures+63)&(~63));


	dr_a 				= (float*)malloc(numSamples*numFeatures*sizeof(float)); 
	if (dr_a == NULL)
	{
		printf("Malloc dr_a failed in load_tsv_data\n");
		return;
	}

	dr_b 				= (float*)malloc(numSamples*sizeof(float));
	if (dr_b == NULL)
	{
		printf("Malloc dr_b failed in load_tsv_data\n");
		return;
	}

	dr_bi 				= (int*)malloc(numSamples*sizeof(int));
	if (dr_bi == NULL)
	{
		printf("Malloc dr_bi failed in load_tsv_data\n");
		return;
	}

	int zk_fd_src;
	zk_fd_src = open(pathToFile, O_RDWR);
	if (zk_fd_src == NULL) {
		printf("Cannot open the file with the path: %s\n", pathToFile);
	}
    //store the data to this address.        //Try to mapp the file to the memory region (zk_disk_addr, zk_total_len).
    float* source =  (float *)mmap (0, 8L*1024*1024*1024, PROT_READ|PROT_WRITE, MAP_SHARED, zk_fd_src, 0); //|MAP_HUGETLB
    if (source == MAP_FAILED) 
    {
        perror ("mmap the file error:../data/imagenet_8G_4M.dat ");
        return;
    }



	for (uint64_t i = 0; i < (1L * dr_numSamples * dr_numFeatures); i++) 
	{ // Bias term
		dr_a[i] = source[i];
	}

	//fclose(f);

	for (int i = 0; i < dr_numSamples; i++) { // Bias term
		dr_b[i] = ( ( (i%2) == 1)? 1.0:0.0); // 10 1.0:0.0 
	}

	cout << "numSamples: "  << dr_numSamples  << endl;
	cout << "numFeatures: " << dr_numFeatures << endl;


}


/////////////Load the data from file with .tsv type////////////
void zipml_sgd_pm::load_tsv_data(char* pathToFile, uint32_t numSamples, uint32_t numFeatures) {
	cout << "Reading " << pathToFile << endl;

	dr_numSamples 		= numSamples;
	dr_numFeatures		= numFeatures; // For the bias term
	dr_a 				= (float*)malloc(numSamples*numFeatures*sizeof(float)); 
	if (dr_a == NULL)
	{
		printf("Malloc dr_a failed in load_tsv_data\n");
		return;
	}
	//////initialization of the array//////
	for (int i = 0; i < dr_numSamples*dr_numFeatures; i++)
		dr_a[i] = 0.0;

	dr_b 				= (float*)malloc(numSamples*sizeof(float));
	if (dr_b == NULL)
	{
		printf("Malloc dr_b failed in load_tsv_data\n");
		return;
	}

	dr_bi 				= (int*)malloc(numSamples*sizeof(int));
	if (dr_bi == NULL)
	{
		printf("Malloc dr_bi failed in load_tsv_data\n");
		return;
	}

	FILE* f;
	f = fopen(pathToFile, "r");
	if (f == NULL) {
		printf("Cannot open the file with the path: %s\n", pathToFile);
	}

	uint32_t sample;
	uint32_t feature;
	float value;
	while(fscanf(f, "%d\t%d\t%f", &sample, &feature, &value) != EOF) {
		if (feature == -2) {
			dr_b[sample]  = value;
			dr_bi[sample] = (int)(value*(float)b_toIntegerScaler);
		}
		else
			dr_a[sample*dr_numFeatures + (feature)] = value; //+1
	}
	fclose(f);

	//for (int i = 0; i < dr_numSamples; i++) { // Bias term
	//	dr_a[i*dr_numFeatures] = 1.0;
	//}

	cout << "numSamples: "  << dr_numSamples  << endl;
	cout << "numFeatures: " << dr_numFeatures << endl;
}

///////////Load the data from file with .libsvm type///////////
void zipml_sgd_pm::load_libsvm_data(char* pathToFile, uint32_t _numSamples, uint32_t _numFeatures) {
	cout << "Reading " << pathToFile << endl;

	dr_numSamples  = _numSamples;
	dr_numFeatures = _numFeatures; // For the bias term

	dr_numFeatures_algin = ((dr_numFeatures+63)&(~63));

	dr_a  = (float*)malloc(dr_numSamples*dr_numFeatures*sizeof(float)); 
	if (dr_a == NULL)
	{
		printf("Malloc dr_a failed in load_tsv_data\n");
		return;
	}
	//////initialization of the array//////
	for (int i = 0; i < dr_numSamples*dr_numFeatures; i++)
		dr_a[i] = 0.0;

	dr_b  = (float*)malloc(dr_numSamples*sizeof(float));
	if (dr_b == NULL)
	{
		printf("Malloc dr_b failed in load_tsv_data\n");
		return;
	}

	dr_bi = (int*)malloc(dr_numSamples *sizeof(int));
	if (dr_bi == NULL)
	{
		printf("Malloc dr_bi failed in load_tsv_data\n");
		return;
	}

	string line;
	ifstream f(pathToFile);

	int index = 0;
	if (f.is_open()) 
	{
		while( index < dr_numSamples ) 
		{
			getline(f, line);
			int pos0 = 0;
			int pos1 = 0;
			int pos2 = 0;
			int column = 0;
			while ( column < dr_numFeatures ) //-1 (no bias...) //while ( pos2 < line.length()+1 ) 
			{
				if (pos2 == 0) 
				{
					pos2 = line.find(" ", pos1);
					float temp = stof(line.substr(pos1, pos2-pos1), NULL);
					dr_b[index] = temp;
					dr_bi[index] = (int)(temp*(float)b_toIntegerScaler);
				}
				else 
				{
					pos0 = pos2;
					pos1 = line.find(":", pos1)+1;
					pos2 = line.find(" ", pos1);
					column = stof(line.substr(pos0+1, pos1-pos0-1));
					if (pos2 == -1) 
					{
						pos2 = line.length()+1;
						dr_a[index*dr_numFeatures + column] = stof(line.substr(pos1, pos2-pos1), NULL);
					}
					else
						dr_a[index*dr_numFeatures + column] = stof(line.substr(pos1, pos2-pos1), NULL);
				}
			}
			index++;
		}
		f.close();
	}
	else
		cout << "Unable to open file " << pathToFile << endl;

	cout << "in libsvm, numSamples: "  << dr_numSamples << endl;
	cout << "in libsvm, numFeatures: " << dr_numFeatures << endl; 
	cout << "in libsvm, dr_numFeatures_algin: " << dr_numFeatures_algin << endl; 
}

///////////Load the data from file with .libsvm type///////////
void zipml_sgd_pm::load_libsvm_data_1(char* pathToFile, uint32_t _numSamples, uint32_t _numFeatures) {
	cout << "Reading " << pathToFile << endl;

	dr_numSamples        = _numSamples;
	dr_numFeatures       = _numFeatures; // For the bias term
	dr_numFeatures_algin = ((dr_numFeatures+63)&(~63));

	dr_a  = (float*)malloc(dr_numSamples*dr_numFeatures*sizeof(float)); 
	if (dr_a == NULL)
	{
		printf("Malloc dr_a failed in load_tsv_data\n");
		return;
	}
	//////initialization of the array//////
	for (int i = 0; i < dr_numSamples*dr_numFeatures; i++)
		dr_a[i] = 0.0;

	dr_b  = (float*)malloc(dr_numSamples*sizeof(float));
	if (dr_b == NULL)
	{
		printf("Malloc dr_b failed in load_tsv_data\n");
		return;
	}

	dr_bi = (int*)malloc(dr_numSamples *sizeof(int));
	if (dr_bi == NULL)
	{
		printf("Malloc dr_bi failed in load_tsv_data\n");
		return;
	}

	string line;
	ifstream f(pathToFile);

	int index = 0;
	if (f.is_open()) {
		while( index < dr_numSamples ) {
			getline(f, line);
			int pos0 = 0;
			int pos1 = 0;
			int pos2 = 0;
			int column = 0;
			//while ( column < dr_numFeatures-1 ) {
			while ( pos2 < line.length()+1 ) {
				if (pos2 == 0) {
					pos2 = line.find(" ", pos1);
					float temp = stof(line.substr(pos1, pos2-pos1), NULL);
					dr_b[index] = temp;
					dr_bi[index] = (int)(temp*(float)b_toIntegerScaler);
				}
				else {
					pos0 = pos2;
					pos1 = line.find(":", pos1)+1;
					pos2 = line.find(" ", pos1);
					column = stof(line.substr(pos0+1, pos1-pos0-1)); //stof
					if (pos2 == -1) {
						pos2 = line.length()+1;
						dr_a[index*dr_numFeatures + column] = stof(line.substr(pos1, pos2-pos1), NULL);
					}
					else
						dr_a[index*dr_numFeatures + column] = stof(line.substr(pos1, pos2-pos1), NULL);
				}
			}
			index++;
		}
		f.close();
	}
	else
		cout << "Unable to open file " << pathToFile << endl;

	//for (int i = 0; i < dr_numSamples; i++) { // Bias term
	//	dr_a[i*dr_numFeatures] = 1.0;
	//}
	cout << "in libsvm, numSamples: "           << dr_numSamples << endl;
	cout << "in libsvm, numFeatures: "          << dr_numFeatures << endl;
	cout << "in libsvm, dr_numFeatures_algin: " << dr_numFeatures_algin << endl; 

}


///////////Load the data from file with .libsvm type///////////
void zipml_sgd_pm::load_libsvm_data_int(char* pathToFile, uint32_t _numSamples, uint32_t _numFeatures) {
	cout << "Reading " << pathToFile << endl;

	dr_numSamples        = _numSamples;
	dr_numFeatures       = _numFeatures; // For the bias term
	dr_numFeatures_algin = ((dr_numFeatures+63)&(~63));

	dr_a  = (float*)malloc(dr_numSamples*dr_numFeatures*sizeof(float)); 
	if (dr_a == NULL)
	{
		printf("Malloc dr_a failed in load_tsv_data\n");
		return;
	}
	//////initialization of the array//////
	for (int i = 0; i < dr_numSamples*dr_numFeatures; i++)
		dr_a[i] = 0.0;

	dr_b  = (float*)malloc(dr_numSamples*sizeof(float));
	if (dr_b == NULL)
	{
		printf("Malloc dr_b failed in load_tsv_data\n");
		return;
	}

	dr_bi = (int*)malloc(dr_numSamples *sizeof(int));
	if (dr_bi == NULL)
	{
		printf("Malloc dr_bi failed in load_tsv_data\n");
		return;
	}

	string line;
	ifstream f(pathToFile);

	int index = 0;
	if (f.is_open()) {
		while( index < dr_numSamples ) {
			getline(f, line);
			int pos0 = 0;
			int pos1 = 0;
			int pos2 = 0;
			int column = 0;
			//while ( column < dr_numFeatures-1 ) {
			while ( pos2 < line.length()+1 ) {
				if (pos2 == 0) {
					pos2 = line.find(" ", pos1);
					float temp = stof(line.substr(pos1, pos2-pos1), NULL);
					dr_b[index] = temp;
					dr_bi[index] = (int)(temp*(float)b_toIntegerScaler);
				}
				else {
					pos0 = pos2;
					pos1 = line.find(":", pos1)+1;
					pos2 = line.find(" ", pos1);
					column = stof(line.substr(pos0+1, pos1-pos0-1)); //stof
					if (pos2 == -1) {
						pos2 = line.length()+1;
						dr_a[index*dr_numFeatures + column] = stof(line.substr(pos1, pos2-pos1), NULL);
					}
					else
						dr_a[index*dr_numFeatures + column] = stof(line.substr(pos1, pos2-pos1), NULL);
				}
			}
			index++;
		}
		f.close();
	}
	else
		cout << "Unable to open file " << pathToFile << endl;

	//for (int i = 0; i < dr_numSamples; i++) { // Bias term
	//	dr_a[i*dr_numFeatures] = 1.0;
	//}
	cout << "in libsvm, numSamples: "           << dr_numSamples << endl;
	cout << "in libsvm, numFeatures: "          << dr_numFeatures << endl;
	cout << "in libsvm, dr_numFeatures_algin: " << dr_numFeatures_algin << endl; 

}


///////////Load the data from file with .libsvm type///////////
void zipml_sgd_pm::load_libsvm_data_1_two(char* pathToFile, uint32_t _numSamples, uint32_t _numFeatures, uint32_t target_1, uint32_t target_2) {
	cout << "Reading " << pathToFile << endl;

	dr_numSamples        = _numSamples;
	dr_numFeatures       = _numFeatures; // For the bias term
	dr_numFeatures_algin = ((dr_numFeatures+63)&(~63));

	dr_a  = (float*)malloc(dr_numSamples*dr_numFeatures*sizeof(float)); 
	if (dr_a == NULL)
	{
		printf("Malloc dr_a failed in load_tsv_data\n");
		return;
	}
	//////initialization of the array//////
	for (int i = 0; i < dr_numSamples*dr_numFeatures; i++)
		dr_a[i] = 0.0;

	dr_b  = (float*)malloc(dr_numSamples*sizeof(float));
	if (dr_b == NULL)
	{
		printf("Malloc dr_b failed in load_tsv_data\n");
		return;
	}

	dr_bi = (int*)malloc(dr_numSamples *sizeof(int));
	if (dr_bi == NULL)
	{
		printf("Malloc dr_bi failed in load_tsv_data\n");
		return;
	}

	string line;
	ifstream f(pathToFile);

	int index      = 0;
	int real_index = 0; 
	bool skip_row  = false;

	if (f.is_open()) 
	{
		while( index < dr_numSamples ) 
		{
			getline(f, line);
			int pos0 = 0;
			int pos1 = 0;
			int pos2 = 0;
			int column = 0;

			skip_row   = false;
			//while ( column < dr_numFeatures-1 ) {
			while ( pos2 < line.length()+1 ) 
			{
				if (pos2 == 0) 
				{
					pos2 = line.find(" ", pos1);
					float temp = stof(line.substr(pos1, pos2-pos1), NULL);
					if ( (temp == (float)target_1) || (temp == (float)target_2) )
					{
						dr_b[real_index]  = temp;
						dr_bi[real_index] = (int)(temp*(float)b_toIntegerScaler);						
					}
					else
					{
						skip_row = true;
						break;
					}
				}
				else 
				{
					pos0 = pos2;
					pos1 = line.find(":", pos1)+1;
					pos2 = line.find(" ", pos1);
					column = stof(line.substr(pos0+1, pos1-pos0-1));
					if (pos2 == -1) {
						pos2 = line.length()+1;
						dr_a[real_index*dr_numFeatures + column] = stof(line.substr(pos1, pos2-pos1), NULL);
					}
					else
						dr_a[real_index*dr_numFeatures + column] = stof(line.substr(pos1, pos2-pos1), NULL);

					
				}
			}
			if (skip_row != true) 
				real_index++;

			index++;
		}
		f.close();
	}
	else
		cout << "Unable to open file " << pathToFile << endl;

	dr_numSamples = real_index;

	cout << "in libsvm, numSamples: "           << dr_numSamples << endl;
	cout << "in libsvm, numFeatures: "          << dr_numFeatures << endl;
	cout << "in libsvm, dr_numFeatures_algin: " << dr_numFeatures_algin << endl; 

}


//Normalize the training data to 0.xyz  x:0.5, y:0.25, z:0.125, so the FPGA can directly do the 
//the computation on the bit-level data representation. 
//Input: dr_a (from the input data...)
//Output: dr_a_norm (for the normalized result, also malloc the space for it...)
void zipml_sgd_pm::a_normalize(void) 
{

	//uint32_t *data  = reinterpret_cast<uint32_t*>( myfpga->malloc(100)); 
	dr_a_norm_fp = (float *)malloc(dr_numSamples*dr_numFeatures*sizeof(float)); 
	if (dr_a_norm_fp == NULL)
	{
		printf("Malloc dr_a_norm_fp failed in a_normalize\n");
		return;
	}

	//a_normalizedToMinus1_1 = toMinus1_1;
	dr_a_norm   = (uint32_t *)malloc(dr_numSamples*dr_numFeatures*sizeof(uint32_t)); //to store the normalized result....
	if (dr_a_norm == NULL)
	{
		printf("Malloc dr_a_norm failed in a_normalize\n");
		return;
	}

	dr_a_min    = (float *)malloc(dr_numFeatures*sizeof(float)); //to store the minimum value of features.....
	if (dr_a_min == NULL)
	{
		printf("Malloc dr_a_min failed in a_normalize\n");
		return;
	}

	dr_a_max    = (float *)malloc(dr_numFeatures*sizeof(float)); //to store the miaximum value of features.....
	if (dr_a_max == NULL)
	{
		printf("Malloc dr_a_max failed in a_normalize\n");
		return;
	}

	printf("dr_numFeatures = %d, dr_numSamples = %d, dr_numFeatures_algin = %d\n", dr_numFeatures, dr_numSamples, dr_numFeatures_algin);

	///Normalize the values in the whole column to the range {0, 1} or {-1, 1}/// 
	for (int j = 0; j < dr_numFeatures; j++) 
	{ // Don't normalize bias
		float amin = numeric_limits<float>::max();
		float amax = numeric_limits<float>::min();
		for (int i = 0; i < dr_numSamples; i++) 
		{
			float a_here = dr_a[i*dr_numFeatures + j];
			if (a_here > amax)
				amax = a_here;
			if (a_here < amin)
				amin = a_here;
		}
		dr_a_min[j]  = amin; //set to the global variable for pm
		dr_a_max[j]  = amax;

		float arange = amax - amin;
		if (arange > 0) 
		{
			for (int i = 0; i < dr_numSamples; i++) 
			{
				float tmp = ((dr_a[i*dr_numFeatures + j] - amin)/arange); //((dr_a[i*dr_numFeatures + j] - amin)/arange)*2.0-1.0;
			  	
			  	dr_a_norm_fp[i*dr_numFeatures + j] = tmp;
			  	dr_a_norm[i*dr_numFeatures + j]    = (uint32_t) (tmp * 4294967295.0); //4294967296 = 2^32
/*
				uint32_t tmp_buffer[4];
				( (float *)tmp_buffer )[0] = tmp;
				uint32_t exponent          = ( (tmp_buffer[0] >> 23) & 0xff);       //[30:23]
				uint32_t mantissa          = 0x800000 + (tmp_buffer[0]&0x7fffff); //[22:0 ]
				uint32_t result_before     = (mantissa << 8);

				if (exponent > 127)       //should be impossible...
				{	
					printf("The normalization value of a should be from 0 to 1.0\n");
					return; 
				}
				else if (exponent == 127) //for the case with value 1.0: 0xffff_ffff
					dr_a_norm[i*dr_numFeatures + j] = 0xffffffff;
				else 
					dr_a_norm[i*dr_numFeatures + j] = result_before >>(126-exponent);
*/					
			}
		}
	}
}

//Suppose each feature contains 32-bit value...
//With the default input is dr_numFeatures...
//Constraint: padding to the smallest power of two that's greater or equal to a given value (64, 128, 256)
uint32_t zipml_sgd_pm::compute_Bytes_per_sample() 
{
	return dr_numFeatures_algin*4;
	//With the chunk of 512 features...
	//uint32_t main_num           = (dr_numFeatures/BITS_OF_ONE_CACHE_LINE)*(BITS_OF_ONE_CACHE_LINE/8); //bytes
	//uint32_t rem_num            = 0;

	//For the remainder of dr_numFeatures...
	//uint32_t remainder_features = dr_numFeatures & (BITS_OF_ONE_CACHE_LINE - 1); 
	//if (remainder_features == 0)
	//	rem_num = 0;
	//else if (remainder_features <= 64)
	//	rem_num = 4;
	//else if (remainder_features <= 128)	
	//	rem_num = 8;
	//else if (remainder_features <= 256)	
	//	rem_num = 16;
	//else 	
	//	rem_num = 32;
	//return main_num + rem_num;
}

void zipml_sgd_pm::a_perform_bitweaving_cpu(void) {

    //Compute the number of cache lines for each sample...
    int num_Bytes_per_sample = compute_Bytes_per_sample();

	printf("1 in a_perform_bitweaving_fpga, num_Bytes_per_sample = %d\n", num_Bytes_per_sample);

    //Compute the bytes for samples: Number of bytes for one CL        CLs                Samples
    uint64_t num_bytes_for_samples = num_Bytes_per_sample * ( (dr_numSamples+15)&(~15) ); //512;//
    printf("dr_numSamples = %d\n", dr_numSamples);

	a_bitweaving_cpu              = (uint32_t*) aligned_alloc(64, num_bytes_for_samples);  //(uint32_t*) malloc(num_bytes_for_samples);//
	if (a_bitweaving_cpu == NULL)
	{
		printf("Malloc memory space for a_bitweaving_cpu failed. \n");
		return;
	}

	hazy::vector::mlweaving_on_sample(a_bitweaving_cpu, dr_a_norm, dr_numSamples, dr_numFeatures); 
/*
	uint32_t *a_fpga_tmp		   = a_bitweaving_cpu;
	uint32_t address_index         = 0;
	///Do the bitWeaving to the training data...
	for (int i = 0; i < dr_numSamples; i+=8)
	{   
		int j; 
		//Deal with the main part of dr_numFeatures.
		for (j = 0; j < (dr_numFeatures/64)*64; j += 64)
		{
			uint32_t tmp_buffer[512] = {0};
			//1: initilization off tmp buffer..
			for (int k = 0; k < 8; k++)
				for (int m = 0; m < 64; m++)
					tmp_buffer[ k*64+m ] = dr_a_norm[ (i + k)*dr_numFeatures + (j+m) ];

			//2: focus on the data from index: j...
			for (int k = 0; k < 32; k++)
			{	
				uint32_t result_buffer[16] = {0};
				//2.1: re-order the data according to the bit-level...
				for (int m = 0; m < 512; m++)
				{
					result_buffer[m>>5] = result_buffer[m>>5] | ((tmp_buffer[m] >>31)<<(m&31));
					tmp_buffer[m]       = tmp_buffer[m] << 1;				
				}
			    //2.2: store the bit-level result back to the memory...
				a_fpga_tmp[address_index++] = result_buffer[0];
				a_fpga_tmp[address_index++] = result_buffer[1];
				a_fpga_tmp[address_index++] = result_buffer[2];
				a_fpga_tmp[address_index++] = result_buffer[3];
				a_fpga_tmp[address_index++] = result_buffer[4];
				a_fpga_tmp[address_index++] = result_buffer[5];
				a_fpga_tmp[address_index++] = result_buffer[6];
				a_fpga_tmp[address_index++] = result_buffer[7];
				a_fpga_tmp[address_index++] = result_buffer[8];
				a_fpga_tmp[address_index++] = result_buffer[9];
				a_fpga_tmp[address_index++] = result_buffer[10];
				a_fpga_tmp[address_index++] = result_buffer[11];
				a_fpga_tmp[address_index++] = result_buffer[12];
				a_fpga_tmp[address_index++] = result_buffer[13];
				a_fpga_tmp[address_index++] = result_buffer[14];
				a_fpga_tmp[address_index++] = result_buffer[15];
			}
		}

		//Deal with the remainder of features, with the index from j...
		uint32_t num_r_f = dr_numFeatures - j;
		//handle the remainder....It is important...
		if (num_r_f > 0)//(j < dr_numFeatures)
		{
			uint32_t tmp_buffer[512] = {0};
			//1: initilization off tmp buffer..
			for (int k = 0; k < 8; k++)
				for (int m = 0; m < num_r_f; m++)
					tmp_buffer[k*64+m] = dr_a_norm[(i + k)*dr_numFeatures + j + m];

			//2: focus on the data from index: j...
			for (int k = 0; k < 32; k++)
			{	
				uint32_t result_buffer[16] = {0};
				//2.1: re-order the data according to the bit-level...
				for (int m = 0; m < 512; m++)
				{
					result_buffer[m>>5] = result_buffer[m>>5] | ((tmp_buffer[m] >>31)<<(m&31));
					tmp_buffer[m]       = tmp_buffer[m] << 1;				
				}
			    //2.2: store the bit-level result back to the memory...
				a_fpga_tmp[address_index++] = result_buffer[0];
				a_fpga_tmp[address_index++] = result_buffer[1];
				a_fpga_tmp[address_index++] = result_buffer[2];
				a_fpga_tmp[address_index++] = result_buffer[3];
				a_fpga_tmp[address_index++] = result_buffer[4];
				a_fpga_tmp[address_index++] = result_buffer[5];
				a_fpga_tmp[address_index++] = result_buffer[6];
				a_fpga_tmp[address_index++] = result_buffer[7];
				a_fpga_tmp[address_index++] = result_buffer[8];
				a_fpga_tmp[address_index++] = result_buffer[9];
				a_fpga_tmp[address_index++] = result_buffer[10];
				a_fpga_tmp[address_index++] = result_buffer[11];
				a_fpga_tmp[address_index++] = result_buffer[12];
				a_fpga_tmp[address_index++] = result_buffer[13];
				a_fpga_tmp[address_index++] = result_buffer[14];
				a_fpga_tmp[address_index++] = result_buffer[15];
			}
		}
	}
*/
}

//It performs the bitweaving operation on a_norm, which is 32-bit. 
//Input:  dr_a_norm (after normalization)
//Output: a_bitweaving_fpga (do the bit-weaving here...)
void zipml_sgd_pm::a_perform_bitweaving_fpga(void) {

    //printf("1 in a_perform_bitweaving_fpga\n");
    //sleep(1);

    //Compute the number of cache lines for each sample...
    int num_Bytes_per_sample = compute_Bytes_per_sample();

    //printf("2 in a_perform_bitweaving_fpga\n");
    //sleep(1);


    //Compute the bytes for samples: Number of bytes for one CL        CLs                Samples
    uint64_t num_bytes_for_samples = num_Bytes_per_sample * dr_numSamples; //512;//

    //printf("3 in a_perform_bitweaving_fpga\n");
    //sleep(1);
    
	a_bitweaving_fpga              = reinterpret_cast<uint32_t*>( myfpga->malloc(num_bytes_for_samples));  //(uint32_t*) malloc(num_bytes_for_samples);//
	if (a_bitweaving_fpga == NULL)
	{
		printf("Malloc FPGA memory space for a_bitweaving_fpga failed in a_perform_bitweaving_fpga. \n");
		return;
	}

    //printf("dr_numSamples = %d, dr_numFeatures = %d, num_bytes_for_samples = %ld\n", dr_numSamples, dr_numFeatures, num_bytes_for_samples);
    //sleep(1);
	hazy::vector::mlweaving_on_sample(a_bitweaving_fpga, dr_a_norm, dr_numSamples, dr_numFeatures); 
/*
	uint32_t *a_fpga_tmp		   = a_bitweaving_fpga;
	uint32_t address_index         = 0;
	///Do the bitWeaving to the training data...
	for (int i = 0; i < dr_numSamples; i+=8)
	{   
		int j; 
		//Deal with the main part of dr_numFeatures.
		for (j = 0; j < (dr_numFeatures/64)*64; j += 64)
		{
			uint32_t tmp_buffer[512] = {0};
			//1: initilization off tmp buffer..
			for (int k = 0; k < 8; k++)
				for (int m = 0; m < 64; m++)
					tmp_buffer[ k*64+m ] = dr_a_norm[ (i + k)*dr_numFeatures + (j+m) ];

			//2: focus on the data from index: j...
			for (int k = 0; k < 32; k++)
			{	
				uint32_t result_buffer[16] = {0};
				//2.1: re-order the data according to the bit-level...
				for (int m = 0; m < 512; m++)
				{
					result_buffer[m>>5] = result_buffer[m>>5] | ((tmp_buffer[m] >>31)<<(m&31));
					tmp_buffer[m]       = tmp_buffer[m] << 1;				
				}
			    //2.2: store the bit-level result back to the memory...
				a_fpga_tmp[address_index++] = result_buffer[0];
				a_fpga_tmp[address_index++] = result_buffer[1];
				a_fpga_tmp[address_index++] = result_buffer[2];
				a_fpga_tmp[address_index++] = result_buffer[3];
				a_fpga_tmp[address_index++] = result_buffer[4];
				a_fpga_tmp[address_index++] = result_buffer[5];
				a_fpga_tmp[address_index++] = result_buffer[6];
				a_fpga_tmp[address_index++] = result_buffer[7];
				a_fpga_tmp[address_index++] = result_buffer[8];
				a_fpga_tmp[address_index++] = result_buffer[9];
				a_fpga_tmp[address_index++] = result_buffer[10];
				a_fpga_tmp[address_index++] = result_buffer[11];
				a_fpga_tmp[address_index++] = result_buffer[12];
				a_fpga_tmp[address_index++] = result_buffer[13];
				a_fpga_tmp[address_index++] = result_buffer[14];
				a_fpga_tmp[address_index++] = result_buffer[15];
			}
		}

		//Deal with the remainder of features, with the index from j...
		uint32_t num_r_f = dr_numFeatures - j;
		//handle the remainder....It is important...
		if (num_r_f > 0)//(j < dr_numFeatures)
		{
			uint32_t tmp_buffer[512] = {0};
			//1: initilization off tmp buffer..
			for (int k = 0; k < 8; k++)
				for (int m = 0; m < num_r_f; m++)
					tmp_buffer[k*64+m] = dr_a_norm[(i + k)*dr_numFeatures + j + m];

			//2: focus on the data from index: j...
			for (int k = 0; k < 32; k++)
			{	
				uint32_t result_buffer[16] = {0};
				//2.1: re-order the data according to the bit-level...
				for (int m = 0; m < 512; m++)
				{
					result_buffer[m>>5] = result_buffer[m>>5] | ((tmp_buffer[m] >>31)<<(m&31));
					tmp_buffer[m]       = tmp_buffer[m] << 1;				
				}
			    //2.2: store the bit-level result back to the memory...
				a_fpga_tmp[address_index++] = result_buffer[0];
				a_fpga_tmp[address_index++] = result_buffer[1];
				a_fpga_tmp[address_index++] = result_buffer[2];
				a_fpga_tmp[address_index++] = result_buffer[3];
				a_fpga_tmp[address_index++] = result_buffer[4];
				a_fpga_tmp[address_index++] = result_buffer[5];
				a_fpga_tmp[address_index++] = result_buffer[6];
				a_fpga_tmp[address_index++] = result_buffer[7];
				a_fpga_tmp[address_index++] = result_buffer[8];
				a_fpga_tmp[address_index++] = result_buffer[9];
				a_fpga_tmp[address_index++] = result_buffer[10];
				a_fpga_tmp[address_index++] = result_buffer[11];
				a_fpga_tmp[address_index++] = result_buffer[12];
				a_fpga_tmp[address_index++] = result_buffer[13];
				a_fpga_tmp[address_index++] = result_buffer[14];
				a_fpga_tmp[address_index++] = result_buffer[15];
			}
		}
	}
*/
}

void zipml_sgd_pm::b_normalize(char toMinus1_1, char binarize_b, int shift_bits) 
{
	//b_normalizedToMinus1_1 = toMinus1_1;
	//for (int i = 0; i < 10; i++)
	//	printf("b[%d] = %d\n", i, dr_b[i]);
	if (binarize_b == 100)
	{
		for (int i = 0; i < dr_numSamples; i++) 
		{
			if( ((int)dr_b[i])%2  == 1 )
				dr_b[i] = 1.0;
			else
				dr_b[i] = 0.0;
	
			dr_bi[i] = ( ((int)dr_b[i])<<shift_bits ); //(int)(dr_b[i]*(float)b_toBinarizeTo);
		}
	}
	else
	{
		for (int i = 0; i < dr_numSamples; i++) 
		{
			if(dr_b[i] == (float)binarize_b)
				dr_b[i] = 1.0;
			else
				dr_b[i] = 0.0;
	
			dr_bi[i] = ( ((int)dr_b[i])<<shift_bits ); //(int)(dr_b[i]*(float)b_toBinarizeTo);
		}
	}


	dr_b_min   =  0.0;
	dr_b_range =  1.0;

}

uint32_t zipml_sgd_pm::b_copy_to_fpga(void)
{
	bi_fpga  = reinterpret_cast<int*>( myfpga->malloc(dr_numSamples*sizeof(int))); 
	if (bi_fpga == NULL)
	{
		printf("Malloc FPGA-accessable memory space (size: %lu) failed in b_copy_to_fpga\n", dr_numSamples*sizeof(int) );
	}

	//copy the dr_dr_bi to bi_fpga...
	for (int i = 0; i < dr_numSamples; i++)
	{
		bi_fpga[i] = dr_bi[i];
	}
}



// Provide: float x[numFeatures]
void zipml_sgd_pm::bitFSGD(uint32_t number_of_bits, uint32_t numberOfIterations, uint32_t mini_batch_size, uint32_t stepSize, int binarize_b, float b_toBinarizeTo) 
{
    /////1:::Setup FPGA/////
    //1.1: set up afu augument configuration for the SGD on FPGA
    SGD_AFU_CONFIG * afu_cfg         = (struct SGD_AFU_CONFIG*)(myfpga->malloc( sizeof(SGD_AFU_CONFIG) ));
    if (afu_cfg == NULL)
    {
    	printf("Malloc afu_cfg in the FPGA memory space failed in bitFSGD\n");
    	return;
    }

	dr_numFeatures_algin = ((dr_numFeatures+63)&(~63));
	x_fpga = reinterpret_cast<int*>( myfpga->malloc(sizeof(int) * numberOfIterations * dr_numFeatures_algin ) );   //((dr_numFeatures+63)&(~63))
    if (x_fpga == NULL)
    {
    	printf("Malloc x_fpga in the FPGA memory space failed in bitFSGD\n");
    	return;
    }

    printf("before step_shifter = %d\n", stepSize);
    uint32_t mini_batch_size_tmp = mini_batch_size;
    while (mini_batch_size_tmp >>= 1)
    	stepSize++;
    printf("after step_shifter = %d\n", stepSize);

    afu_cfg->addr_a                  = a_bitweaving_fpga;    //The address of training set...
    afu_cfg->addr_b                  = bi_fpga;      	     //
    afu_cfg->addr_model              = x_fpga;               // do not care about it...
    afu_cfg->mini_batch_size         = mini_batch_size;      //^_^ It should be parametered...
    afu_cfg->step_size               = stepSize;             //1/2^stepSize to the 
    afu_cfg->number_of_epochs        = numberOfIterations;
    afu_cfg->dimension               = dr_numFeatures;       // add the predicator "b"
    afu_cfg->number_of_samples       = dr_numSamples;        // 1; 
    afu_cfg->number_of_bits          = number_of_bits;		 // doGather; No use of this feature, since the data is from the same dataset.
    afu_cfg->binarize_b_value        = 0;
    afu_cfg->b_value_to_binarize_to  = 0.0;
    afu_cfg->gather_depth            = 0;                    //gatherDepth: No use of this feature.
	afu_cfg->number_of_CL_to_process = 1;                    //No use of this feature. Can be replaced.

	//cout << "numCacheLines: " << numCacheLines << endl;
    //start = get_time(); //Start the timer.

    FthreadRec* cpOp =  new FthreadRec(myfpga, SGD_AFU_BITWEAVING, reinterpret_cast<unsigned char*>(afu_cfg), sizeof(SGD_AFU_CONFIG) );
    //printf("after new FthreadRec\n");
    ///malloc the new Fthread in the ZipML_SGD class/// 
    mythread = new Fthread(cpOp);
    //printf("after new Fthread\n");    
    
    //wait for the ZipML SGD to finish the computation.
    mythread->join();
    //SleepMilli(3);
    //end = get_time(); //Stop the timer.
    //printf("after new join\n");  

    mythread->printStatusLine();

	//for (int i = 0; i < 10; i++)
	//	printf("%d = %d, %f\n", i, x_fpga[i], (float)(x_fpga[i])/8388608.0  );


}


//pure software implementation on CPU...
void zipml_sgd_pm::float_linreg_SGD(uint32_t numberOfIterations, float stepSize) {

	dr_numFeatures_algin = ((dr_numFeatures+63)&(~63));
    x = (float *) malloc(sizeof(float)*numberOfIterations * dr_numFeatures_algin ); //numFeatures
	float x_tmp[dr_numFeatures_algin];
	for (int j = 0; j < dr_numFeatures_algin; j++) {
		x_tmp[j] = 0.0;
	}
	
	float loss_value = calculate_loss(x_tmp);
	cout << "init_loss: "<< loss_value <<endl;


	for(int epoch = 0; epoch < numberOfIterations; epoch++) {

		for (int i = 0; i < dr_numSamples; i++) {
			float dot = 0;
			for (int j = 0; j < dr_numFeatures; j++) {
				dot += x_tmp[j]*dr_a_norm_fp[i*dr_numFeatures + j];
			}
			
			//printf("dot = %f\n", dot);

			for (int j = 0; j < dr_numFeatures; j++) {
				x_tmp[j] -= stepSize*(dot - dr_b[i])*dr_a_norm_fp[i*dr_numFeatures + j];
			}


		}

		for (int j = 0; j < dr_numFeatures; j++) {
			x[epoch*dr_numFeatures + j] = x_tmp[j];

			//if (j < 10)
			//	printf("%d = %f\n", j, x_tmp[j]);
		}

		float loss_value = calculate_loss(x_tmp);
		cout << epoch << "_loss: "<< loss_value <<endl;
	}
}


//pure software implementation on CPU...
void zipml_sgd_pm::float_linreg_SGD_batch(uint32_t numberOfIterations, float stepSize, int mini_batch_size) {

	dr_numFeatures_algin = ((dr_numFeatures+63)&(~63));
    x = (float *) malloc(sizeof(float)*numberOfIterations * dr_numFeatures_algin ); //numFeatures

    float x_tmp[dr_numFeatures_algin];
	for (int j = 0; j < dr_numFeatures_algin; j++) 
		x_tmp[j] = 0.0;

    float x_gradient[dr_numFeatures_algin];
	for (int j = 0; j < dr_numFeatures_algin; j++) 
		x_gradient[j] = 0.0;
	

	//////Initialized loss...///
	float loss_value = calculate_loss(x_tmp);
	cout << "init_loss: "<< loss_value <<endl;

	//Iterate over each epoch...
	for(int epoch = 0; epoch < numberOfIterations; epoch++) 
	{

		//for one mini_batch...
		for (int i = 0; i < (dr_numSamples/mini_batch_size)*mini_batch_size; i += mini_batch_size) 
		{
			///set the gradient to 0.
			for (int k = 0; k < dr_numFeatures_algin; k++) 
				x_gradient[k] = 0.0;

			///compute the gradient for this mini batch.
			for (int k = 0; k < mini_batch_size; k++)
			{
				float dot = 0;
				for (int j = 0; j < dr_numFeatures; j++) 
					dot += x_tmp[j]*dr_a_norm_fp[(i+k)*dr_numFeatures + j];

				for (int j = 0; j < dr_numFeatures; j++) 
					x_gradient[j] += stepSize*(dot - dr_b[i+k])*dr_a_norm_fp[(i+k)*dr_numFeatures + j];
			}

			///update the model with the computed gradient..
			for (int k = 0; k < dr_numFeatures_algin; k++) 
				x_tmp[k] -= x_gradient[k];
		}


		//Store to the global model pool...
		for (int j = 0; j < dr_numFeatures; j++) {
			x[epoch*dr_numFeatures + j] = x_tmp[j];
		}

		float loss_value = calculate_loss(x_tmp);
		cout << epoch << "_loss: "<< loss_value <<endl;
	}
}



float zipml_sgd_pm::calculate_loss(float x[]) {
 	//cout << "numSamples: "  << numSamples << endl;
	//cout << "numFeatures: " << numFeatures << endl;
    //numSamples  = 10;
	//cout << "For debugging: numSamples=" << numFeatures << endl;

	float loss = 0;
	for(int i = 0; i < dr_numSamples; i++) {
		float dot = 0.0;
		for (int j = 0; j < dr_numFeatures; j++) {
			dot += x[j]*dr_a_norm_fp[i*dr_numFeatures + j];
			//cout << "x["<< j <<"] =" << x[j] << "   a="<< a[i*numFeatures+ j];
		}
		loss += (dot - dr_b[i])*(dot - dr_b[i]);
		//cout << "b[i]" << b[i] << endl;
        //cout << loss << endl;
	}

	loss /= (float)(2*dr_numSamples);
	return loss;
}


void zipml_sgd_pm::compute_loss_and_printf(uint32_t numberOfIterations, uint32_t num_fractional_bits)
{
	float scale_f = (float)(1 << num_fractional_bits);
	float x_tmp[dr_numFeatures];
       memset( x_tmp, 0, dr_numFeatures*sizeof(float) );
	float loss_final = calculate_loss(x_tmp);
        printf("Before training, loss is %f\n", loss_final);

	for (uint32_t i = 0; i < numberOfIterations; i++)
	{
		for (int j = 0; j < dr_numFeatures; j++) {
			x_tmp[j] = (float)(x_fpga[i*dr_numFeatures_algin + j])/scale_f; //8388608.0  ;
		}

	        loss_final = calculate_loss(x_tmp);	
		printf("%d-th iteration, loss is %f\n", i, loss_final);
	}
}

//single thread implementation. 
void RunHogwildPerThread(ThreadArgs &args, unsigned tid, unsigned total)
{
	float *a_norm_fp     = args.a_norm_fp;	
	float *b_norm_fp     = args.b_norm_fp;

	float *x_gradient    = args.x_gradient[tid];
	float *x_global      = args.x_global;
	uint32_t numSamples  = args.numSamples;
	uint32_t numFeatures = args.numFeatures;
	uint32_t batch_size  = args.batch_size;

    hazy::vector::FVector<float> f_x_gradient(x_gradient, numFeatures); 
    hazy::vector::FVector<float> f_x_global(x_global,     numFeatures); 

    size_t start         = GetStartIndex(numSamples, tid, total); 
    size_t end           = GetEndIndex(numSamples, tid, total);

	float stepSize       = args.stepSize;
	float scale          = -(stepSize/(float)batch_size); ///(float)params.batch_size;


    ////////////main loop for each epoch////////////
    for (size_t i = start; i < end; i+= batch_size) {
    	//printf("i = %d\n", i);
    	uint32_t actual_batch_size = ( (i+batch_size) > end? (end-i):batch_size ); //It is no larger than batch size.
    	//printf("actual_batch_size = %d\n", actual_batch_size);

        hazy::vector::Zero(f_x_gradient);

        for (size_t j = 0; j < actual_batch_size; j++)
        {	
        	size_t k = i + j;
			hazy::vector::FVector<float> f_sample(a_norm_fp + (uint64_t)(k*numFeatures), numFeatures); 
			//printf("jj = %d\n", j);	  
			float dot_product = hazy::vector::Dot( f_x_global, f_sample);
			float delta;
			delta = scale * (dot_product - b_norm_fp[k]); //
			//printf("j = %d\n", j);	  
				// linear regression
			hazy::vector::ScaleAndAdd(
				f_x_gradient,				
				f_sample,
				delta
			);

        }

        //threadArgs.communicate_times_[tid].ptr->Start();
		hazy::vector::ScaleAndAdd(
		    f_x_global,
		    f_x_gradient,
		    1.0
		);
        //threadArgs.communicate_times_[tid].ptr->Pause();
    } 
   // for (int i = 0; i < 10; i++)
   // 	printf("%f ", x_global[i]); 
  }



//Hogwild...
void zipml_sgd_pm::float_linreg_SGD_hogwild(uint32_t numberOfIterations, float stepSize, int mini_batch_size, uint32_t nthreads, uint32_t num_epochs_a_decay, float decay_initial) 
{
	// Init 
	ThreadArgs args; 
    hazy::util::Clock epoch_time_;

	//Global model x.
	dr_numFeatures_algin = ((dr_numFeatures+63)&(~63));
	//malloc the memory space for the model x. 
    x = (float *) aligned_alloc(64, sizeof(float) * dr_numFeatures_algin ); 
    if (x == NULL) {
    	printf("Malloc for the model x failed\n");
    	return;
    }
	for (int j = 0; j < dr_numFeatures_algin; j++)
		x[j] = 0.0;

 	args.x_global    = this->x;
 	args.numSamples  = dr_numSamples;
 	args.numFeatures = dr_numFeatures;
	args.batch_size  = mini_batch_size;
	args.stepSize    = stepSize;

	args.a_norm_fp   = dr_a_norm_fp;
	args.b_norm_fp   = dr_b;


	float loss_value = calculate_loss(x);
	cout << "Hogwild init_loss: "<< loss_value <<endl;

	//Malloc and initialize the gradient for each thread.
	for (int i = 0; i < nthreads; i++)
	{
		args.x_gradient[i] = (float *) aligned_alloc(64, sizeof(float) * dr_numFeatures_algin ); 
    	if (args.x_gradient[i] == NULL) {
    		printf("Malloc for the gradient[%d] failed\n", i);
    		return;
    	}
    	for (int j = 0; j < dr_numFeatures_algin; j++)
    		 (args.x_gradient[i])[j] = 0.0;
	}
	//printf("1\n");
	/////Initlize the threadBlock
    hazy::thread::ThreadPool* threadPool_;
	threadPool_ = new hazy::thread::ThreadPool(nthreads);
	threadPool_->Init();

	//printf("2\n");
	//Do the training....
	float total_time = 0.0;
	for(int epoch = 0; epoch < numberOfIterations; epoch++) 
	{
		//printf("3\n");
		if (epoch == numberOfIterations-2) 
		{
		  PCM_initPerformanceMonitor(&inst_Monitor_Event, NULL);
		  PCM_start();
		}

		//Decay the learning rate for better convergence.... 
		args.stepSize    = stepSize * pow( decay_initial, sqrt((float)epoch+1.0) );
		//if ( (epoch != 0) && (epoch % num_epochs_a_decay == 0) )
		//{
		//	stepSize/=2.0;
		//	args.stepSize    = stepSize;
		//}

		epoch_time_.Start();

		threadPool_->Execute( args, RunHogwildPerThread );
		threadPool_->Wait();

		epoch_time_.Stop();


		if (epoch == numberOfIterations-2) 
		{
		  PCM_stop();
		} 

		total_time += epoch_time_.value;

		float loss_value = calculate_loss(x);
		printf("Hogwild: %d-th loss is %f,stepsize = %f,epoch time: %.7f,total time: %.7f\n", epoch, loss_value, args.stepSize, epoch_time_.value, total_time); // epoch_time_.value shows the elapsed time for each epoch...
	}


	if (1) 
	{
	  printf("=====print the profiling result==========\n");
	  PCM_printResults();   
	  PCM_cleanup();
	} 

}

/*
//single thread implementation. 
void ModelSyncPerThread(ThreadArgs &args, unsigned tid, unsigned total)
{
	float *a_norm_fp     = args.a_norm_fp;	
	float *b_norm_fp     = args.b_norm_fp;

	uint32_t numSamples  = args.numSamples;
	uint32_t numFeatures = args.numFeatures;
	uint32_t batch_size  = args.batch_size;

	float *x_global      = args.x_global;
	float *x_local       = args.x_local[tid];

    hazy::vector::FVector<float> f_x_local (x_local,  numFeatures); 
    hazy::vector::FVector<float> f_x_global(x_global, numFeatures); 

    size_t start         = GetStartIndex(numSamples, tid, total); 
    size_t end           = GetEndIndex(numSamples, tid, total);

	float stepSize       = args.stepSize;
	float scale          = -stepSize; 

	//1,  Load the local model from the global model 
	hazy::vector::CopyInto(f_x_global, f_x_local);

    ////////////main loop for each epoch////////////
    for (size_t i = start; i < end; i++) {
    	//printf("i = %d\n", i);
		hazy::vector::FVector<float> f_sample(a_norm_fp + (uint64_t)(i*numFeatures), numFeatures); 
			//printf("jj = %d\n", j);	  
		float dot_product = hazy::vector::Dot( f_x_local, f_sample);
		float delta;
		delta = scale * (dot_product - b_norm_fp[i]); //
			//printf("j = %d\n", j);	  
				// linear regression
		hazy::vector::ScaleAndAdd(
			f_x_local,				
			f_sample,
			delta
		);

        //threadArgs.communicate_times_[tid].ptr->Pause();
    } 
   // for (int i = 0; i < 10; i++)
   // 	printf("%f ", x_global[i]); 
  }
*/

//single thread implementation. 
void ModelSyncPerThread(ThreadArgs &args, unsigned tid, unsigned total)
{
	float *a_norm_fp     = args.a_norm_fp;	
	float *b_norm_fp     = args.b_norm_fp;

	float *x_local       = args.x_local[tid];
	float *x_gradient    = args.x_gradient[tid];
	float *x_global      = args.x_global;
	uint32_t numSamples  = args.numSamples;
	uint32_t numFeatures = args.numFeatures;
	uint32_t batch_size  = args.batch_size;

    hazy::vector::FVector<float> f_x_local(   x_local,    numFeatures); 
    hazy::vector::FVector<float> f_x_gradient(x_gradient, numFeatures); 
	hazy::vector::FVector<float> f_x_global(  x_global,   numFeatures); 

    size_t start         = GetStartIndex(numSamples, tid, total); 
    size_t end           = GetEndIndex(numSamples, tid, total);

	float stepSize       = args.stepSize;
	float scale          = -(stepSize/(float)batch_size); ///(float)params.batch_size;

	hazy::vector::CopyInto(f_x_global, f_x_local);

    ////////////main loop for each epoch////////////
    for (size_t i = start; i < end; i+= batch_size) {
    	//printf("i = %d\n", i);
    	uint32_t actual_batch_size = ( (i+batch_size) > end? (end-i):batch_size ); //It is no larger than batch size.
    	//printf("actual_batch_size = %d\n", actual_batch_size);

        hazy::vector::Zero(f_x_gradient);

        for (size_t j = 0; j < actual_batch_size; j++)
        {	
        	size_t k = i + j;
			hazy::vector::FVector<float> f_sample(a_norm_fp + (uint64_t)(k*numFeatures), numFeatures); 
			float dot_product = hazy::vector::Dot(f_x_local, f_sample); //f_x_global
			float delta       = scale * (dot_product - b_norm_fp[k]); //
			hazy::vector::ScaleAndAdd(
				f_x_gradient,				
				f_sample,
				delta
			);

        }

        //threadArgs.communicate_times_[tid].ptr->Start();
		hazy::vector::ScaleAndAdd(
		    f_x_local,    //f_x_global,
		    f_x_gradient,
		    1.0
		);
        //threadArgs.communicate_times_[tid].ptr->Pause();
    } 
  }




//ModelAverage...
void zipml_sgd_pm::float_linreg_SGD_modelaverage(uint32_t numberOfIterations, float stepSize, int mini_batch_size, uint32_t nthreads, uint32_t num_epochs_a_decay, float decay_initial) {

	// Init 
	ThreadArgs args; 
    hazy::util::Clock epoch_time_;

	//Global model x.
	dr_numFeatures_algin = ((dr_numFeatures+63)&(~63));
	//malloc the memory space for the model x. 
    x = (float *) aligned_alloc(64, sizeof(float) * dr_numFeatures_algin ); 
    if (x == NULL) {
    	printf("Malloc for the model x failed\n");
    	return;
    }
	for (int j = 0; j < dr_numFeatures_algin; j++)
		x[j] = 0.0;

 	args.x_global    = this->x;
 	args.numSamples  = dr_numSamples;
 	args.numFeatures = dr_numFeatures;
	args.batch_size  = mini_batch_size;
	args.stepSize    = stepSize;

	args.a_norm_fp   = dr_a_norm_fp;
	args.b_norm_fp   = dr_b;


	float *x_global  = args.x_global;
    hazy::vector::FVector<float> f_x_global(x_global, dr_numFeatures); 

	hazy::vector::FVector<float> *f_x_local;
	f_x_local        = new hazy::vector::FVector<float>[nthreads];


	float loss_value = calculate_loss(x);
	cout << "ModelAverage init_loss: "<< loss_value <<endl;

	//Malloc and initialize the gradient for each thread.
	for (int i = 0; i < nthreads; i++)
	{
		args.x_local[i] = (float *) aligned_alloc(64, sizeof(float) * dr_numFeatures_algin ); 
    	if (args.x_local[i] == NULL) {
    		printf("Malloc for local[%d] failed\n", i);
    		return;
    	}
    	hazy::vector::FVector<float> f_local_tmp(args.x_local[i], dr_numFeatures);
    	f_x_local[i] = f_local_tmp;
    	for (int j = 0; j < dr_numFeatures_algin; j++)
    		 (args.x_local[i])[j] = 0.0;


		args.x_gradient[i] = (float *) aligned_alloc(64, sizeof(float) * dr_numFeatures_algin ); 
    	if (args.x_gradient[i] == NULL) {
    		printf("Malloc for gradient[%d] failed\n", i);
    		return;
    	}
    	for (int j = 0; j < dr_numFeatures_algin; j++)
    		 (args.x_gradient[i])[j] = 0.0;    		
	}


	//printf("1\n");
	/////Initlize the threadBlock
    hazy::thread::ThreadPool* threadPool_;
	threadPool_ = new hazy::thread::ThreadPool(nthreads);
	threadPool_->Init();

	//printf("2\n");
	//Do the training....
	float total_time = 0.0;
	for(int epoch = 0; epoch < numberOfIterations; epoch++) 
	{
		//printf("3\n");
		if (epoch == numberOfIterations-2) 
		{
		  PCM_initPerformanceMonitor(&inst_Monitor_Event, NULL);
		  PCM_start();
		}


		//Decay the learning rate for better convergence.... 
		args.stepSize    = stepSize * pow( decay_initial, sqrt((float)epoch+1.0) );
		//if ( (epoch != 0) && (epoch % num_epochs_a_decay == 0) )
		//{
		//	stepSize/=2.0;
		//	args.stepSize    = stepSize;
		//}

		epoch_time_.Start();

		//1, each thread works on its local model, its performance is great due to no cache coherence protocol overhead.
		threadPool_->Execute( args, ModelSyncPerThread );
		threadPool_->Wait();
		//2, After the computing thread finishes the computation of local models, the main thread will aggregate the local models from the local threads. 
		hazy::vector::avg_list(f_x_global, f_x_local, nthreads);

		epoch_time_.Stop();


		if (epoch == numberOfIterations-2) 
		{
		  PCM_stop();
		} 

		total_time += epoch_time_.value;

		float loss_value = calculate_loss(x);
		printf("ModelAverage: %d-th loss is %f, stepsize = %f,  epoch time: %.7f, total time: %.7f\n", epoch, loss_value, args.stepSize, epoch_time_.value, total_time); // epoch_time_.value shows the elapsed time for each epoch...
	}


	if (1) 
	{
	  printf("=====print the profiling result==========\n");
	  PCM_printResults();   
	  PCM_cleanup();
	} 

}



//single thread implementation. 
void mlweavingFunctionMergeModelSyncPerThread(ThreadArgs &args, unsigned tid, unsigned total)
{
	float *a_norm_fp     = args.a_norm_fp;	
	float *b_norm_fp     = args.b_norm_fp;

	float *x_local       = args.x_local[tid];
	float *x_gradient    = args.x_gradient[tid];
	float *x_global      = args.x_global;
	uint32_t numSamples  = args.numSamples;
	uint32_t numFeatures = args.numFeatures;
	uint32_t batch_size  = args.batch_size;
	uint32_t numBits     = args.numBits;
	uint32_t*a_mlweaving = args.a_mlweaving;

    hazy::vector::FVector<float> f_x_local(   x_local,    numFeatures); 
    hazy::vector::FVector<float> f_x_gradient(x_gradient, numFeatures); 
	hazy::vector::FVector<float> f_x_global(  x_global,   numFeatures); 

    size_t start         = GetStartIndex(numSamples, tid, total); 
    size_t end           = GetEndIndex(numSamples, tid, total);

	float stepSize       = args.stepSize;
	float scale          = -(stepSize/(float)batch_size); ///(float)params.batch_size;

	hazy::vector::CopyInto(f_x_global, f_x_local);


    if (numBits <= 8)
    {
		float b_base     = 256.0;  //65536.0; //1.0; // //2^16 or  1
		scale            = scale/((float)b_base*b_base); ///(float)params.batch_size;
	  
	   uint8_t dest[512+numFeatures];
	   hazy::vector::FVector<uint8_t> dest_char_vector (dest, numFeatures);

    	////////////main loop for each epoch////////////
		for (size_t i = start; i < end; i+= batch_size) 
    	{
    		//printf("i = %d\n", i);
    		uint32_t actual_batch_size = ( (i+batch_size) > end? (end-i):batch_size ); //It is no larger than batch size.
    		//printf("actual_batch_size = %d\n", actual_batch_size);
	
    	    hazy::vector::Zero(f_x_gradient); 
	
    	    for (size_t j = 0; j < actual_batch_size; j++)
    	    {	
    	    	size_t k = i + j;
				//hazy::vector::FVector<float> f_sample(a_norm_fp + (uint64_t)(k*numFeatures), numFeatures);
				hazy::vector::retrieve_from_mlweaving(dest_char_vector, a_mlweaving, k, numBits, numSamples);  //src_#####
				float dot_product = hazy::vector::Dot(f_x_local, dest_char_vector ); //f_sample
				//float dot_product = hazy::vector::Dot_mlweaving(dest_char_vector, f_x_local, a_mlweaving, k, numBits, numSamples ); //f_sample
				float delta       = scale * (dot_product - b_base*b_norm_fp[k]); //
				hazy::vector::ScaleAndAdd(
					f_x_gradient,				
					dest_char_vector,     //f_sample,
					delta
				);
    	    }
	
    	    //threadArgs.communicate_times_[tid].ptr->Start();
			hazy::vector::ScaleAndAdd(
			    f_x_local,    //f_x_global,
			    f_x_gradient,
			    1.0
			);
    	    //threadArgs.communicate_times_[tid].ptr->Pause();
    	}
    } 
}


//single thread implementation. 
void mlweavingModelSyncPerThread(ThreadArgs &args, unsigned tid, unsigned total)
{
	float *a_norm_fp     = args.a_norm_fp;	
	float *b_norm_fp     = args.b_norm_fp;

	float *x_local       = args.x_local[tid];
	float *x_gradient    = args.x_gradient[tid];
	float *x_global      = args.x_global;
	uint32_t numSamples  = args.numSamples;
	uint32_t numFeatures = args.numFeatures;
	uint32_t batch_size  = args.batch_size;
	uint32_t numBits     = args.numBits;
	uint32_t*a_mlweaving = args.a_mlweaving;

    hazy::vector::FVector<float> f_x_local(   x_local,    numFeatures); 
    hazy::vector::FVector<float> f_x_gradient(x_gradient, numFeatures); 
	hazy::vector::FVector<float> f_x_global(  x_global,   numFeatures); 

    size_t start         = GetStartIndex(numSamples, tid, total); 
    size_t end           = GetEndIndex(numSamples, tid, total);

	float stepSize       = args.stepSize;
	float scale          = -(stepSize/(float)batch_size); ///(float)params.batch_size;

	hazy::vector::CopyInto(f_x_global, f_x_local);


    if (numBits <= 8)
    {
		float b_base     = 256.0;  //65536.0; //1.0; // //2^16 or  1
		scale            = scale/((float)b_base*b_base); ///(float)params.batch_size;
	  
	   uint8_t dest[512+numFeatures];
	   hazy::vector::FVector<uint8_t> dest_char_vector (dest, numFeatures);

    	////////////main loop for each epoch////////////
		for (size_t i = start; i < end; i+= batch_size) 
    	{
    		//printf("i = %d\n", i);
    		uint32_t actual_batch_size = ( (i+batch_size) > end? (end-i):batch_size ); //It is no larger than batch size.
    		//printf("actual_batch_size = %d\n", actual_batch_size);
	
    	    hazy::vector::Zero(f_x_gradient); 
	
    	    for (size_t j = 0; j < actual_batch_size; j++)
    	    {	
    	    	size_t k = i + j;
				//hazy::vector::FVector<float> f_sample(a_norm_fp + (uint64_t)(k*numFeatures), numFeatures);
				hazy::vector::retrieve_from_mlweaving(dest_char_vector, a_mlweaving, k, numBits, numSamples);  //src_#####
				float dot_product = hazy::vector::Dot(f_x_local, dest_char_vector ); //f_sample
				float delta       = scale * (dot_product - b_base*b_norm_fp[k]); //
				hazy::vector::ScaleAndAdd(
					f_x_gradient,				
					dest_char_vector,     //f_sample,
					delta
				);
    	    }
	
    	    //threadArgs.communicate_times_[tid].ptr->Start();
			hazy::vector::ScaleAndAdd(
			    f_x_local,    //f_x_global,
			    f_x_gradient,
			    1.0
			);
    	    //threadArgs.communicate_times_[tid].ptr->Pause();
    	}
    } 
}

//ModelAverage...
void zipml_sgd_pm::mlweaving_linreg_SGD_modelaverage(uint32_t numberOfIterations, float stepSize, int mini_batch_size, uint32_t nthreads, uint32_t num_epochs_a_decay, float decay_initial, uint32_t numberOfBits) {

	// Init 
	ThreadArgs args; 
    hazy::util::Clock epoch_time_;

	//Global model x.
	dr_numFeatures_algin = ((dr_numFeatures+63)&(~63));
	//malloc the memory space for the model x. 
    x = (float *) aligned_alloc(64, sizeof(float) * dr_numFeatures_algin ); 
    if (x == NULL) {
    	printf("Malloc for the model x failed\n");
    	return;
    }
	for (int j = 0; j < dr_numFeatures_algin; j++)
		x[j] = 0.0;

 	args.x_global    = this->x;
 	args.numSamples  = dr_numSamples;
 	args.numFeatures = dr_numFeatures;
	args.batch_size  = mini_batch_size;
	args.stepSize    = stepSize;

	args.a_norm_fp   = dr_a_norm_fp;
	args.b_norm_fp   = dr_b;

	args.a_mlweaving = a_bitweaving_cpu;
	args.numBits     = numberOfBits;

	float *x_global  = args.x_global;
    hazy::vector::FVector<float> f_x_global(x_global, dr_numFeatures); 

	hazy::vector::FVector<float> *f_x_local;
	f_x_local        = new hazy::vector::FVector<float>[nthreads];


	float loss_value = calculate_loss(x);
	cout << "ModelAverage init_loss: "<< loss_value <<endl;

	//Malloc and initialize the gradient for each thread.
	for (int i = 0; i < nthreads; i++)
	{
		args.x_local[i] = (float *) aligned_alloc(64, sizeof(float) * dr_numFeatures_algin ); 
    	if (args.x_local[i] == NULL) {
    		printf("Malloc for local[%d] failed\n", i);
    		return;
    	}
    	hazy::vector::FVector<float> f_local_tmp(args.x_local[i], dr_numFeatures);
    	f_x_local[i] = f_local_tmp;
    	for (int j = 0; j < dr_numFeatures_algin; j++)
    		 (args.x_local[i])[j] = 0.0;


		args.x_gradient[i] = (float *) aligned_alloc(64, sizeof(float) * dr_numFeatures_algin ); 
    	if (args.x_gradient[i] == NULL) {
    		printf("Malloc for gradient[%d] failed\n", i);
    		return;
    	}
    	for (int j = 0; j < dr_numFeatures_algin; j++)
    		 (args.x_gradient[i])[j] = 0.0;    		
	}


	//printf("1\n");
	/////Initlize the threadBlock
    hazy::thread::ThreadPool* threadPool_;
	threadPool_ = new hazy::thread::ThreadPool(nthreads);
	threadPool_->Init();

	//printf("2\n");
	//Do the training....
	float total_time = 0.0;
	for(int epoch = 0; epoch < numberOfIterations; epoch++) 
	{
		//printf("3\n");
		if (epoch == numberOfIterations-2) 
		{
		  PCM_initPerformanceMonitor(&inst_Monitor_Event, NULL);
		  PCM_start();
		}


		//Decay the learning rate for better convergence.... 
		args.stepSize    = stepSize * pow( decay_initial, sqrt((float)epoch+1.0) );


		epoch_time_.Start();

		//1, each thread works on its local model, its performance is great due to no cache coherence protocol overhead.
		threadPool_->Execute( args, mlweavingFunctionMergeModelSyncPerThread); //mlweavingModelSyncPerThread  //
		threadPool_->Wait();
		//2, After the computing thread finishes the computation of local models, the main thread will aggregate the local models from the local threads. 
		hazy::vector::avg_list(f_x_global, f_x_local, nthreads);

		epoch_time_.Stop();


		if (epoch == numberOfIterations-2) 
		{
		  PCM_stop();
		} 

		total_time += epoch_time_.value;

		float loss_value = calculate_loss(x);
		printf("ModelAverage+MLWeaving: %d-th loss is %f, stepsize = %f,  epoch time: %.7f, total time: %.7f\n", epoch, loss_value, args.stepSize, epoch_time_.value, total_time); // epoch_time_.value shows the elapsed time for each epoch...
	}


	if (1) 
	{
	  printf("=====print the profiling result==========\n");
	  PCM_printResults();   
	  PCM_cleanup();
	} 

}


//single thread implementation. 
void charModelSyncPerThread(ThreadArgs &args, unsigned tid, unsigned total)
{
	float *a_norm_fp     = args.a_norm_fp;	
	float *b_norm_fp     = args.b_norm_fp;
	uint8_t *a_norm_char = args.a_norm_char;	

	float *x_local       = args.x_local[tid];
	float *x_gradient    = args.x_gradient[tid];
	float *x_global      = args.x_global;
	uint32_t numSamples  = args.numSamples;
	uint32_t numFeatures = args.numFeatures;
	uint32_t batch_size  = args.batch_size;
	uint32_t numBits     = args.numBits;
	uint32_t*a_mlweaving = args.a_mlweaving;

	uint32_t numFeatures_algin = args.numFeatures_algin;


    hazy::vector::FVector<float> f_x_local(   x_local,    numFeatures); 
    hazy::vector::FVector<float> f_x_gradient(x_gradient, numFeatures); 
	hazy::vector::FVector<float> f_x_global(  x_global,   numFeatures); 

    size_t start         = GetStartIndex(numSamples, tid, total); 
    size_t end           = GetEndIndex(numSamples, tid, total);

	float stepSize       = args.stepSize;
	float scale          = -(stepSize/(float)batch_size); ///(float)params.batch_size;

	hazy::vector::CopyInto(f_x_global, f_x_local);


    //if (numBits <= 8)
    {
		float b_base     = 256.0;  //65536.0; //1.0; // //2^16 or  1
		scale            = scale/((float)b_base*b_base); ///(float)params.batch_size;

	   //uint8_t dest[512+numFeatures];
	   //hazy::vector::FVector<uint8_t> dest_char_vector (dest, numFeatures);

    	////////////main loop for each epoch////////////
		for (size_t i = start; i < end; i+= batch_size) 
    	{
    		//printf("i = %d\n", i);
    		uint32_t actual_batch_size = ( (i+batch_size) > end? (end-i):batch_size ); //It is no larger than batch size.
    		//printf("actual_batch_size = %d\n", actual_batch_size);
	
    	    hazy::vector::Zero(f_x_gradient); 
	
    	    for (size_t j = 0; j < actual_batch_size; j++)
    	    {	
    	    	size_t k = i + j;
				//hazy::vector::FVector<float> f_sample(a_norm_fp + (uint64_t)(k*numFeatures), numFeatures);
				//hazy::vector::retrieve_from_mlweaving(dest_char_vector, a_mlweaving, k, numBits, numSamples);  //src_#####
				hazy::vector::FVector<uint8_t> dest_char_vector (a_norm_char+k*numFeatures_algin, numFeatures);
				float dot_product = hazy::vector::Dot(f_x_local, dest_char_vector ); //f_sample
				float delta       = scale * (dot_product - b_base*b_norm_fp[k]); //
				hazy::vector::ScaleAndAdd(
					f_x_gradient,				
					dest_char_vector,     //f_sample,
					delta
				);
    	    }
	
    	    //threadArgs.communicate_times_[tid].ptr->Start();
			hazy::vector::ScaleAndAdd(
			    f_x_local,    //f_x_global,
			    f_x_gradient,
			    1.0
			);
    	    //threadArgs.communicate_times_[tid].ptr->Pause();
    	}
    } 
}

//input dataset is char (8-bit)...
void zipml_sgd_pm::char_linreg_SGD_modelaverage(uint32_t numberOfIterations, float stepSize, int mini_batch_size, uint32_t nthreads, uint32_t num_epochs_a_decay, float decay_initial, uint32_t numberOfBits) {

	// Init 
	ThreadArgs args; 
    hazy::util::Clock epoch_time_;

	//Global model x.
	dr_numFeatures_algin         = ((dr_numFeatures+63)&(~63));
	uint32_t dr_numSamples_align = ((dr_numSamples+7)&(~7));	


	//malloc the memory space for the model x. 
    x = (float *) aligned_alloc(64, sizeof(float) * dr_numFeatures_algin ); 
    if (x == NULL) {
    	printf("Malloc for the model x failed\n");
    	return;
    }
	for (int j = 0; j < dr_numFeatures_algin; j++)
		x[j] = 0.0;

	//malloc the memory space for the model x. 
	uint8_t* a_norm_char = (uint8_t *) aligned_alloc(64, sizeof(uint8_t)*dr_numFeatures_algin*dr_numSamples_align );
    if (a_norm_char == NULL) {
    	printf("Malloc for the char-based dataset a_norm_char failed\n");
    	return;
    }
	for (uint32_t i = 0; i < dr_numSamples; i++)
	{
		for (uint32_t j = 0; j < dr_numSamples; j++)
			a_norm_char[i*dr_numFeatures_algin+j] = ( dr_a_norm[i*dr_numFeatures+j]>>24 );
	}

 	args.x_global    = this->x;
 	args.numSamples  = dr_numSamples;
 	args.numFeatures = dr_numFeatures;
	args.batch_size  = mini_batch_size;
	args.stepSize    = stepSize;

	args.a_norm_char = a_norm_char;
	args.a_norm_fp   = dr_a_norm_fp;
	args.b_norm_fp   = dr_b;

	args.a_mlweaving = a_bitweaving_cpu;
	args.numBits     = numberOfBits;

 	args.numFeatures_algin = dr_numFeatures_algin;


	float *x_global  = args.x_global;
    hazy::vector::FVector<float> f_x_global(x_global, dr_numFeatures); 

	hazy::vector::FVector<float> *f_x_local;
	f_x_local        = new hazy::vector::FVector<float>[nthreads];


	float loss_value = calculate_loss(x);
	cout << "ModelAverage init_loss: "<< loss_value <<endl;

	//Malloc and initialize the gradient for each thread.
	for (int i = 0; i < nthreads; i++)
	{
		args.x_local[i] = (float *) aligned_alloc(64, sizeof(float) * dr_numFeatures_algin ); 
    	if (args.x_local[i] == NULL) {
    		printf("Malloc for local[%d] failed\n", i);
    		return;
    	}
    	hazy::vector::FVector<float> f_local_tmp(args.x_local[i], dr_numFeatures);
    	f_x_local[i] = f_local_tmp;
    	for (int j = 0; j < dr_numFeatures_algin; j++)
    		 (args.x_local[i])[j] = 0.0;


		args.x_gradient[i] = (float *) aligned_alloc(64, sizeof(float) * dr_numFeatures_algin ); 
    	if (args.x_gradient[i] == NULL) {
    		printf("Malloc for gradient[%d] failed\n", i);
    		return;
    	}
    	for (int j = 0; j < dr_numFeatures_algin; j++)
    		 (args.x_gradient[i])[j] = 0.0;    		
	}


	//printf("1\n");
	/////Initlize the threadBlock
    hazy::thread::ThreadPool* threadPool_;
	threadPool_ = new hazy::thread::ThreadPool(nthreads);
	threadPool_->Init();

	//printf("2\n");
	//Do the training....
	float total_time = 0.0;
	for(int epoch = 0; epoch < numberOfIterations; epoch++) 
	{
		//printf("3\n");
		if (epoch == numberOfIterations-2) 
		{
		  PCM_initPerformanceMonitor(&inst_Monitor_Event, NULL);
		  PCM_start();
		}


		//Decay the learning rate for better convergence.... 
		args.stepSize    = stepSize * pow( decay_initial, sqrt((float)epoch+1.0) );


		epoch_time_.Start();

		//1, each thread works on its local model, its performance is great due to no cache coherence protocol overhead.
		threadPool_->Execute( args, charModelSyncPerThread );
		threadPool_->Wait();
		//2, After the computing thread finishes the computation of local models, the main thread will aggregate the local models from the local threads. 
		hazy::vector::avg_list(f_x_global, f_x_local, nthreads);

		epoch_time_.Stop();


		if (epoch == numberOfIterations-2) 
		{
		  PCM_stop();
		} 

		total_time += epoch_time_.value;

		float loss_value = calculate_loss(x);
		printf("ModelAverage+MLWeaving: %d-th loss is %f, stepsize = %f,  epoch time: %.7f, total time: %.7f\n", epoch, loss_value, args.stepSize, epoch_time_.value, total_time); // epoch_time_.value shows the elapsed time for each epoch...
	}


	if (1) 
	{
	  printf("=====print the profiling result==========\n");
	  PCM_printResults();   
	  PCM_cleanup();
	} 

}


//single thread implementation. 
void shortModelSyncPerThread(ThreadArgs &args, unsigned tid, unsigned total)
{
	float *a_norm_fp     = args.a_norm_fp;	
	float *b_norm_fp     = args.b_norm_fp;
	uint16_t*a_norm_short= args.a_norm_short;	

	float *x_local       = args.x_local[tid];
	float *x_gradient    = args.x_gradient[tid];
	float *x_global      = args.x_global;
	uint32_t numSamples  = args.numSamples;
	uint32_t numFeatures = args.numFeatures;
	uint32_t batch_size  = args.batch_size;
	uint32_t numBits     = args.numBits;
	uint32_t*a_mlweaving = args.a_mlweaving;

	uint32_t numFeatures_algin = args.numFeatures_algin;


    hazy::vector::FVector<float> f_x_local(   x_local,    numFeatures); 
    hazy::vector::FVector<float> f_x_gradient(x_gradient, numFeatures); 
	hazy::vector::FVector<float> f_x_global(  x_global,   numFeatures); 

    size_t start         = GetStartIndex(numSamples, tid, total); 
    size_t end           = GetEndIndex(numSamples, tid, total);

	float stepSize       = args.stepSize;
	float scale          = -(stepSize/(float)batch_size); ///(float)params.batch_size;

	hazy::vector::CopyInto(f_x_global, f_x_local);



		float b_base     = 256.0*256.0;  //65536.0; //1.0; // //2^16 or  1
		scale            = scale/((float)b_base*b_base); ///(float)params.batch_size;

	   //uint8_t dest[512+numFeatures];
	   //hazy::vector::FVector<uint8_t> dest_char_vector (dest, numFeatures);

    	////////////main loop for each epoch////////////
		for (size_t i = start; i < end; i+= batch_size) 
    	{
    		//printf("i = %d\n", i);
    		uint32_t actual_batch_size = ( (i+batch_size) > end? (end-i):batch_size ); //It is no larger than batch size.
    		//printf("actual_batch_size = %d\n", actual_batch_size);
	
    	    hazy::vector::Zero(f_x_gradient); 
	
    	    for (size_t j = 0; j < actual_batch_size; j++)
    	    {	
    	    	size_t k = i + j;
				//hazy::vector::FVector<float> f_sample(a_norm_fp + (uint64_t)(k*numFeatures), numFeatures);
				//hazy::vector::retrieve_from_mlweaving(dest_char_vector, a_mlweaving, k, numBits, numSamples);  //src_#####
				hazy::vector::FVector<uint16_t> dest_char_vector (a_norm_short+k*numFeatures_algin, numFeatures);
				float dot_product = hazy::vector::Dot(f_x_local, dest_char_vector ); //f_sample
				float delta       = scale * (dot_product - b_base*b_norm_fp[k]); //
				hazy::vector::ScaleAndAdd(
					f_x_gradient,				
					dest_char_vector,     //f_sample,
					delta
				);
    	    }
	
    	    //threadArgs.communicate_times_[tid].ptr->Start();
			hazy::vector::ScaleAndAdd(
			    f_x_local,    //f_x_global,
			    f_x_gradient,
			    1.0
			);
    	    //threadArgs.communicate_times_[tid].ptr->Pause();
    	}

}

//input dataset is short (16-bit)...
void zipml_sgd_pm::short_linreg_SGD_modelaverage(uint32_t numberOfIterations, float stepSize, int mini_batch_size, uint32_t nthreads, uint32_t num_epochs_a_decay, float decay_initial, uint32_t numberOfBits) {

	// Init 
	ThreadArgs args; 
    hazy::util::Clock epoch_time_;

	//Global model x.
	        dr_numFeatures_algin = ((dr_numFeatures+63)&(~63));
	uint32_t dr_numSamples_align = ((dr_numSamples+7)&(~7));	


	//malloc the memory space for the model x. 
    x = (float *) aligned_alloc(64, sizeof(float) * dr_numFeatures_algin ); 
    if (x == NULL) {
    	printf("Malloc for the model x failed\n");
    	return;
    }
	for (int j = 0; j < dr_numFeatures_algin; j++)
		x[j] = 0.0;

	//malloc the memory space for the model x. 
	uint16_t* a_norm_short = (uint16_t *) aligned_alloc(64, sizeof(uint16_t)*dr_numFeatures_algin*dr_numSamples_align );
    if (a_norm_short == NULL) {
    	printf("Malloc for the char-based dataset a_norm_char failed\n");
    	return;
    }
	for (uint32_t i = 0; i < dr_numSamples; i++)
	{
		for (uint32_t j = 0; j < dr_numSamples; j++)
			a_norm_short[i*dr_numFeatures_algin+j] = ( dr_a_norm[i*dr_numFeatures+j]>>16 );
	}

 	args.x_global    = this->x;
 	args.numSamples  = dr_numSamples;
 	args.numFeatures = dr_numFeatures;
	args.batch_size  = mini_batch_size;
	args.stepSize    = stepSize;

	args.a_norm_short= a_norm_short;
	args.a_norm_fp   = dr_a_norm_fp;
	args.b_norm_fp   = dr_b;

	args.a_mlweaving = a_bitweaving_cpu;
	args.numBits     = numberOfBits;

 	args.numFeatures_algin = dr_numFeatures_algin;


	float *x_global  = args.x_global;
    hazy::vector::FVector<float> f_x_global(x_global, dr_numFeatures); 

	hazy::vector::FVector<float> *f_x_local;
	f_x_local        = new hazy::vector::FVector<float>[nthreads];


	float loss_value = calculate_loss(x);
	cout << "ModelAverage init_loss: "<< loss_value <<endl;

	//Malloc and initialize the gradient for each thread.
	for (int i = 0; i < nthreads; i++)
	{
		args.x_local[i] = (float *) aligned_alloc(64, sizeof(float) * dr_numFeatures_algin ); 
    	if (args.x_local[i] == NULL) {
    		printf("Malloc for local[%d] failed\n", i);
    		return;
    	}
    	hazy::vector::FVector<float> f_local_tmp(args.x_local[i], dr_numFeatures);
    	f_x_local[i] = f_local_tmp;
    	for (int j = 0; j < dr_numFeatures_algin; j++)
    		 (args.x_local[i])[j] = 0.0;


		args.x_gradient[i] = (float *) aligned_alloc(64, sizeof(float) * dr_numFeatures_algin ); 
    	if (args.x_gradient[i] == NULL) {
    		printf("Malloc for gradient[%d] failed\n", i);
    		return;
    	}
    	for (int j = 0; j < dr_numFeatures_algin; j++)
    		 (args.x_gradient[i])[j] = 0.0;    		
	}


	//printf("1\n");
	/////Initlize the threadBlock
    hazy::thread::ThreadPool* threadPool_;
	threadPool_ = new hazy::thread::ThreadPool(nthreads);
	threadPool_->Init();

	//printf("2\n");
	//Do the training....
	float total_time = 0.0;
	for(int epoch = 0; epoch < numberOfIterations; epoch++) 
	{
		//printf("3\n");
		if (epoch == numberOfIterations-2) 
		{
		  PCM_initPerformanceMonitor(&inst_Monitor_Event, NULL);
		  PCM_start();
		}


		//Decay the learning rate for better convergence.... 
		args.stepSize    = stepSize * pow( decay_initial, sqrt((float)epoch+1.0) );


		epoch_time_.Start();

		//1, each thread works on its local model, its performance is great due to no cache coherence protocol overhead.
		threadPool_->Execute( args, shortModelSyncPerThread );
		threadPool_->Wait();
		//2, After the computing thread finishes the computation of local models, the main thread will aggregate the local models from the local threads. 
		hazy::vector::avg_list(f_x_global, f_x_local, nthreads);

		epoch_time_.Stop();


		if (epoch == numberOfIterations-2) 
		{
		  PCM_stop();
		} 

		total_time += epoch_time_.value;

		float loss_value = calculate_loss(x);
		printf("ModelAverage+MLWeaving: %d-th loss is %f, stepsize = %f,  epoch time: %.7f, total time: %.7f\n", epoch, loss_value, args.stepSize, epoch_time_.value, total_time); // epoch_time_.value shows the elapsed time for each epoch...
	}


	if (1) 
	{
	  printf("=====print the profiling result==========\n");
	  PCM_printResults();   
	  PCM_cleanup();
	} 

}



#endif
