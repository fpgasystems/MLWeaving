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
//

#ifndef ZIP_SGD_PM_H
#define ZIP_SGD_PM_H

#include "config_pm.h"
#include <immintrin.h>

using namespace std;

//define the struct of the user's arguments for FPGA implementation.
struct SGD_AFU_CONFIG {
  // CL #1
  union {
    uint64_t    qword0[8];         // make it a whole cacheline
    struct {
      uint32_t* addr_a;                       //8  63:0 
      int*      addr_b;                       //8  127:64
      int*      addr_model;                   //8  191:128

      uint32_t  mini_batch_size;         //4       223:192
      uint32_t  step_size;               //4  //8  255:224
      uint32_t  number_of_epochs;        //4       287:256
      uint32_t  dimension;               //4  //8  319:288
      uint32_t  number_of_samples;       //4       351:320
      uint32_t  number_of_bits;          //4  //8  383:352
      
      uint32_t  gather_depth;            //4  //Below slot, we can modify it....
      uint32_t  number_of_CL_to_process; //4  //8
      uint32_t  binarize_b_value;        //4
      float     b_value_to_binarize_to;  //4  //8

    };
  };
};


struct ThreadArgs
{
  float *x_global;
  float *x_local[50];
  float *x_gradient[50];

  float *a_norm_fp;
  float *b_norm_fp;
  uint8_t   *a_norm_char;       //norm -->  8-bit char
  uint16_t  *a_norm_short;      //norm --> 16-bit short

  uint32_t *a_mlweaving;

  uint32_t numSamples;
  uint32_t numFeatures;
  uint32_t numFeatures_algin;
  uint32_t batch_size;
  float    stepSize; 
  uint32_t numBits; 
};



class zipml_sgd_pm {
private: 
	float        *dr_a;	                //The original data set features matrix: numSamples x numFeatures
	float        *dr_a_norm_fp;         //The dataset after the normalization to (0, 1)
	uint32_t     *dr_a_norm;            //The dataset after the normalization to (0, 1)


	uint32_t     *a_bitweaving_fpga;    //Perform the bitWeaving on a_norm, input data format for the FPGA... 
	uint32_t     *a_bitweaving_cpu;     //Perform the bitWeaving on a_norm, input data format for the CPU... 
    
	float        *dr_b;	                // Data set labels vector: numSamples
	int          *dr_bi;	            // Integer version of dr_bi
	int          *bi_fpga;	        // Integer version of dr_bi, accessed by FPGA.
	
	//int* x;	// Address of the model which can be only referenced by CPU....
	int          *x_fpga;               //Address of the model which can be referenced by CPU+FPGA....
	float        *x;                    //for floating point implementation. 

    float        *dr_a_min;             //Minimum value of dr_a
    float        *dr_a_max;             //Maimum value of dr_a
	float         dr_b_range;
	float         dr_b_min;

	uint32_t      dr_numFeatures;       //Number of features in the machine learning training task...
	uint32_t      dr_numSamples;        //Number of samples in the machine learning training task...
	uint32_t      dr_numFeatures_algin;

	uint32_t      b_toIntegerScaler;

	char          gotFPGA = 0;
	FPGA          *myfpga;
	Fthread       *mythread;
	//Number of cache lines for all the training datasets.
    uint32_t      compute_Bytes_per_sample();

public:

	//Constructor/destructor
	zipml_sgd_pm(bool usingFPGA, uint32_t _b_toIntegerScaler);
	~zipml_sgd_pm();


	// Data loading functions
	void load_tsv_data(     char* pathToFile, uint32_t _numSamples, uint32_t _numFeatures);
	void load_libsvm_data(  char* pathToFile, uint32_t _numSamples, uint32_t _numFeatures);
	void load_libsvm_data_1(char* pathToFile, uint32_t _numSamples, uint32_t _numFeatures);
	void load_libsvm_data_int(char* pathToFile, uint32_t _numSamples, uint32_t _numFeatures);
	void load_dense_data(char* pathToFile, uint32_t _numSamples, uint32_t _numFeatures);

	void load_libsvm_data_1_two(char* pathToFile, uint32_t _numSamples, uint32_t _numFeatures, uint32_t target_1, uint32_t target_2);

	// Normalization and data shaping
	void a_normalize(void);
	void b_normalize(char toMinus1_1, char binarize_b, int shift_bits);

	//fix me...
	void a_perform_bitweaving_cpu(void);	
	// Return how many cache lines needed
	void a_perform_bitweaving_fpga(void);
	uint32_t b_copy_to_fpga(void);



	// Provide: float x[numFeatures]
	void bitFSGD(uint32_t number_of_bits, uint32_t numberOfIterations, uint32_t mini_batch_size, uint32_t stepSize, int binarize_b, float b_toBinarizeTo); 
	void compute_loss_and_printf(uint32_t numberOfIterations, uint32_t num_fractional_bits);

	//uint32_t copy_data_into_FPGA_memory_after_quantization(int quantizationBits, int _numberOfIndices, uint32_t address32offset);
	//uint32_t get_number_of_CLs_needed_for_one_index(int quantizationBits);
	void mlweaving_linreg_SGD_modelaverage(uint32_t numberOfIterations, float stepSize, int mini_batch_size, uint32_t nthreads, uint32_t num_epochs_a_decay, float decay_initial, uint32_t numBits);	
	void char_linreg_SGD_modelaverage(uint32_t numberOfIterations, float stepSize, int mini_batch_size, uint32_t nthreads, uint32_t num_epochs_a_decay, float decay_initial, uint32_t numberOfBits);
	void short_linreg_SGD_modelaverage(uint32_t numberOfIterations, float stepSize, int mini_batch_size, uint32_t nthreads, uint32_t num_epochs_a_decay, float decay_initial, uint32_t numberOfBits);

	void float_linreg_SGD_hogwild(     uint32_t numberOfIterations, float stepSize, int mini_batch_size, uint32_t nthreads, uint32_t num_epochs_a_decay, float decay_initial);	
	void float_linreg_SGD_modelaverage(uint32_t numberOfIterations, float stepSize, int mini_batch_size, uint32_t nthreads, uint32_t num_epochs_a_decay, float decay_initial);	
	void float_linreg_SGD_batch(uint32_t numberOfIterations, float stepSize, int mini_batch_size);	
	void float_linreg_SGD(uint32_t numberOfIterations, float stepSize);
	float calculate_loss(float x[]);
	// Quantization function
	//void quantize_data_integer(int aiq[], uint32_t numBits);
};

#endif
