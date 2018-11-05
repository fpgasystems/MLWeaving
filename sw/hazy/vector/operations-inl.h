// Copyright 2012 Chris Re, Victor Bittorf
//
 //Licensed under the Apache License, Version 2.0 (the "License");
 //you may not use this file except in compliance with the License.
 //You may obtain a copy of the License at
 //    http://www.apache.org/licenses/LICENSE-2.0
 //Unless required by applicable law or agreed to in writing, software
 //distributed under the License is distributed on an "AS IS" BASIS,
 //WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 //See the License for the specific language governing permissions and
 //limitations under the License.

// The Hazy Project, http://research.cs.wisc.edu/hazy/
// Author : Victor Bittorf (bittorf [at] cs.wisc.edu)

#ifndef HAZY_VECTOR_VECTOROPS_INL_H
#define HAZY_VECTOR_VECTOROPS_INL_H
#include "string.h"

#include "hazy/util/sort.h"
#include "hazy/vector/operations.h"

// See hazy/vector/operations.h for documentation

namespace hazy {
namespace vector {


#define uint32_t unsigned int
#define BITS_OF_ONE_CACHE_LINE 512

//Suppose the size of each value of training dataset is 32-bit, always true for our case...
uint32_t compute_num_CLs_per_sample(uint32_t dr_numFeatures) {
  //With the chunk of 512 features...
  uint32_t main_num           = (dr_numFeatures/BITS_OF_ONE_CACHE_LINE)*32; //It is for CLs
  uint32_t rem_num            = 0;

  //For the remainder of dr_numFeatures...
  uint32_t remainder_features = dr_numFeatures & (BITS_OF_ONE_CACHE_LINE - 1); 
  if (remainder_features == 0)
    rem_num = 0;
  else if (remainder_features <= 64)
    rem_num = 4;
  else if (remainder_features <= 128) 
    rem_num = 8;
  else if (remainder_features <= 256) 
    rem_num = 16;
  else  
    rem_num = 32;
  //printf("main_num = %d, rem_num = %d\t", main_num, rem_num);
  return main_num + rem_num;
}

void bitweaving_on_each_sample(uint32_t *dest, uint32_t *src, uint32_t numFeatures) 
{
  //Compute the number of CLs for each sample...
  int num_CLs_per_sample     = compute_num_CLs_per_sample(numFeatures);
  //printf("num_CLs_per_sample = %d\n", num_CLs_per_sample);
  //uint32_t *a_fpga_tmp       = a_bitweaving_fpga;
  uint32_t address_index     = 0;
  int num_features_main      = (numFeatures/BITS_OF_ONE_CACHE_LINE)*BITS_OF_ONE_CACHE_LINE;  

    //Deal with the main part of dr_numFeatures.
    for (int j = 0; j < num_features_main; j += BITS_OF_ONE_CACHE_LINE)
    {
      uint32_t tmp_buffer[BITS_OF_ONE_CACHE_LINE] = {0};
      //1: initilization off tmp buffer..
      for (int k = 0; k < BITS_OF_ONE_CACHE_LINE; k++)
      {
        tmp_buffer[k] = src[j + k];
        //printf("src[%d] = 0x%8x\t", j+k, src[j + k]);
      }  


      //2: focus on the data from index: j...
      for (int k = 0; k < 32; k++)
      { 
        uint32_t result_buffer[BITS_OF_ONE_CACHE_LINE/32] = {0};  //16 ints == 512 bits...
        //2.1: re-order the data according to the bit-level...
        for (int m = 0; m < BITS_OF_ONE_CACHE_LINE; m++)
        {
          result_buffer[m>>5] = result_buffer[m>>5] | ((tmp_buffer[m] >>31)<<(m&31));
          tmp_buffer[m]       = tmp_buffer[m] << 1;       
        }
        //2.2: store the bit-level result back to the memory...
        dest[address_index++] = result_buffer[0]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[1]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[2]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[3]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[4]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[5]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[6]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[7]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[8]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[9]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[10];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[11];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[12];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[13];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[14];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[15];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
      }
    }

    //Deal with the remainder of features, with the index from j...
    uint32_t num_r_f = numFeatures - num_features_main;
    //handle the remainder....It is important...
    if (num_r_f > 0)
    {
      uint32_t tmp_buffer[BITS_OF_ONE_CACHE_LINE] = {0};
      for (int k = 0; k < num_r_f; k++)
        tmp_buffer[k] = src[num_features_main + k]; //j is the existing index...

      for (int k = 0; k < 32; k++) //64 bits for each bit...
      {
        uint32_t result_buffer[BITS_OF_ONE_CACHE_LINE] = {0};
        for (int m = 0; m < num_r_f; m++)
        {
          result_buffer[m>>5] = result_buffer[m>>5] | ((tmp_buffer[m] >>31)<<(m&31));
          tmp_buffer[m]       = tmp_buffer[m] << 1;       
        }
          //1--64 
          dest[address_index++] = result_buffer[0];
          dest[address_index++] = result_buffer[1];

        if (num_r_f > 64)
        { //65--128 
          dest[address_index++] = result_buffer[2];
          dest[address_index++] = result_buffer[3];
        }

        if (num_r_f > 128)
        { //129--256 
          dest[address_index++] = result_buffer[4];
          dest[address_index++] = result_buffer[5];
          dest[address_index++] = result_buffer[6];
          dest[address_index++] = result_buffer[7];
        }

        if (num_r_f > 256)
        { //257-511
          dest[address_index++] = result_buffer[8];
          dest[address_index++] = result_buffer[9];
          dest[address_index++] = result_buffer[10];
          dest[address_index++] = result_buffer[11];
          dest[address_index++] = result_buffer[12];
          dest[address_index++] = result_buffer[13];
          dest[address_index++] = result_buffer[14];
          dest[address_index++] = result_buffer[15];
        }
      }             
    }
}




void inline Convert_from_bitweaving(FVector<unsigned short> & dest, FVector<unsigned int> &src, unsigned num_bits) 
{
	uint64_t numFeatures    = dest.size;
	unsigned short* vec_char = dest.values;
	unsigned int* vec_int    = src.values;
	
#define uint32_t unsigned int
#define BITS_OF_ONE_CACHE_LINE 512
	

	uint32_t num_features_main = (numFeatures/BITS_OF_ONE_CACHE_LINE) * BITS_OF_ONE_CACHE_LINE;
	
	for (size_t i = 0; i < num_features_main; i++) 
  {
    //vec_char[i] = extract_from_bitweaving(src.values, i, numFeatures);
	
	  //Compute the main part of numFeatures.
	  //if (i < num_features_main)
	  //{
		uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE	  ) * BITS_OF_ONE_CACHE_LINE; //
		uint32_t int_offset  = ( i&(BITS_OF_ONE_CACHE_LINE-1) )/32;
		uint32_t bit_offset  = i & 31;
	
		//The next 32 CLs contains the information of the feature. 
		unsigned short result = 0;
		unsigned int tmp;
		for (uint32_t j = 0; j < num_bits; j++)
		{
							 //main		    bit	   which ints 
		  tmp	  = vec_int[main_offset + 16 * j + int_offset]; 
		  result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (15-j)); //
		}
		vec_char[i] = result; 
	}



  for (size_t i = num_features_main; i < numFeatures; i++) 
  {
		uint32_t num_r_f = numFeatures - num_features_main;
	
		if (num_r_f <= 64)												 //////remainder <= 64
		{ 
		  uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE ) * BITS_OF_ONE_CACHE_LINE;
		  uint32_t int_offset  = ( i & (64-1) )/32;
		  uint32_t bit_offset  = i & 31;
	
		  //The next 32 CLs contains the information of the feature. 
		  unsigned short result = 0;
		  uint32_t tmp;
		  for (uint32_t j = 0; j < num_bits; j++)
		  {
							  //main		  bit	 which ints 
			tmp 	= vec_int[main_offset + 2 * j + int_offset]; 
			result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (15-j)); //
		  }
		  vec_char[i] = result; //return result;
		}
		else if (num_r_f <= 128)										  //////64 < remainder <= 128
		{ 
		  uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE ) * BITS_OF_ONE_CACHE_LINE;
		  uint32_t int_offset  = ( i&(128-1) )/32;
		  uint32_t bit_offset  = i & 31;
	
		  //The next 32 CLs contains the information of the feature. 
		  unsigned short result = 0;
		  uint32_t tmp;
		  for (uint32_t j = 0; j < num_bits; j++)
		  {
							  //main		  bit	 which ints 
			tmp 	= vec_int[main_offset + 4 * j + int_offset]; 
			result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (15-j)); //
		  }
		  vec_char[i] = result; //return result;
		}
		else if (num_r_f <= 256)										  //////128 < remainder <= 256
		{ 
		  uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE ) * BITS_OF_ONE_CACHE_LINE;
		  uint32_t int_offset  = ( i&(256-1) )/32;
		  uint32_t bit_offset  = i & 31;
	
		  //The next 32 CLs contains the information of the feature. 
		  unsigned short result = 0;
		  uint32_t tmp;
		  for (uint32_t j = 0; j < num_bits; j++)
		  {
							  //main		  bit	 which ints 
			tmp 	= vec_int[main_offset + 8 * j + int_offset]; 
			result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (15-j)); //
		  }
		  vec_char[i] = result; //return result;
		}
		else if (num_r_f < 512) 										 //////256 < remainder < 512
		{ 
		  uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE ) * BITS_OF_ONE_CACHE_LINE;
		  uint32_t int_offset  = ( i&(512-1) )/32;
		  uint32_t bit_offset  = i & 31;
		//The next 32 CLs contains the information of the feature. 
		  unsigned short result = 0;
		  uint32_t tmp;
		  for (uint32_t j = 0; j < num_bits; j++)
		  {
							  //main		  bit	 which ints 
			tmp 	= vec_int[main_offset + 16 * j + int_offset]; 
			result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (15-j)); //
		  }
		  vec_char[i] = result; //return result;
		}			
	}
}


void inline Convert_from_bitweaving(FVector<unsigned char> & dest, FVector<unsigned int> &src, unsigned num_bits) 
{
	uint64_t numFeatures    = dest.size;
	unsigned char* vec_char = dest.values;
	unsigned int* vec_int    = src.values;
	
#define uint32_t unsigned int
#define BITS_OF_ONE_CACHE_LINE 512
	

	uint32_t num_features_main = (numFeatures/BITS_OF_ONE_CACHE_LINE) * BITS_OF_ONE_CACHE_LINE;
	
	for (size_t i = 0; i < num_features_main; i++) 
  {
    //vec_char[i] = extract_from_bitweaving(src.values, i, numFeatures);
	
	  //Compute the main part of numFeatures.
	  //if (i < num_features_main)
	  //{
		uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE	  ) * BITS_OF_ONE_CACHE_LINE; //
		uint32_t int_offset  = ( i&(BITS_OF_ONE_CACHE_LINE-1) )/32;
		uint32_t bit_offset  = i & 31;
	
		//The next 32 CLs contains the information of the feature. 
		unsigned char result = 0;
		unsigned int tmp;
		for (uint32_t j = 0; j < num_bits; j++)
		{
							 //main		    bit	   which ints 
		  tmp	  = vec_int[main_offset + 16 * j + int_offset]; 
		  result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (7-j)); //
		}
		vec_char[i] = result; 
	}



  for (size_t i = num_features_main; i < numFeatures; i++) 
  {
		uint32_t num_r_f = numFeatures - num_features_main;
	
		if (num_r_f <= 64)												 //////remainder <= 64
		{ 
		  uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE ) * BITS_OF_ONE_CACHE_LINE;
		  uint32_t int_offset  = ( i & (64-1) )/32;
		  uint32_t bit_offset  = i & 31;
	
		  //The next 32 CLs contains the information of the feature. 
		  uint32_t result = 0;
		  uint32_t tmp;
		  for (uint32_t j = 0; j < num_bits; j++)
		  {
							  //main		  bit	 which ints 
			tmp 	= vec_int[main_offset + 2 * j + int_offset]; 
			result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (7-j)); //
		  }
		  vec_char[i] = result; //return result;
		}
		else if (num_r_f <= 128)										  //////64 < remainder <= 128
		{ 
		  uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE ) * BITS_OF_ONE_CACHE_LINE;
		  uint32_t int_offset  = ( i&(128-1) )/32;
		  uint32_t bit_offset  = i & 31;
	
		  //The next 32 CLs contains the information of the feature. 
		  uint32_t result = 0;
		  uint32_t tmp;
		  for (uint32_t j = 0; j < num_bits; j++)
		  {
							  //main		  bit	 which ints 
			tmp 	= vec_int[main_offset + 4 * j + int_offset]; 
			result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (7-j)); //
		  }
		  vec_char[i] = result; //return result;
		}
		else if (num_r_f <= 256)										  //////128 < remainder <= 256
		{ 
		  uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE ) * BITS_OF_ONE_CACHE_LINE;
		  uint32_t int_offset  = ( i&(256-1) )/32;
		  uint32_t bit_offset  = i & 31;
	
		  //The next 32 CLs contains the information of the feature. 
		  uint32_t result = 0;
		  uint32_t tmp;
		  for (uint32_t j = 0; j < num_bits; j++)
		  {
							  //main		  bit	 which ints 
			tmp 	= vec_int[main_offset + 8 * j + int_offset]; 
			result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (7-j)); //
		  }
		  vec_char[i] = result; //return result;
		}
		else if (num_r_f < 512) 										 //////256 < remainder < 512
		{ 
		  uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE ) * BITS_OF_ONE_CACHE_LINE;
		  uint32_t int_offset  = ( i&(512-1) )/32;
		  uint32_t bit_offset  = i & 31;
		//The next 32 CLs contains the information of the feature. 
		  uint32_t result = 0;
		  uint32_t tmp;
		  for (uint32_t j = 0; j < num_bits; j++)
		  {
							  //main		  bit	 which ints 
			tmp 	= vec_int[main_offset + 16 * j + int_offset]; 
			result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (7-j)); //
		  }
		  vec_char[i] = result; //return result;
		}			
	}
}


//add src to the destination. 
template <typename T>
void inline avg_list(FVector<T> & dest, FVector<T> *src, unsigned N) {
  T scale_factor = 1.0 /(T)N;

  for (size_t i = 0; i < dest.size; i++) 
  {
    T sum  = 0.0;
    for (unsigned j = 0; j < N; j++)  
      sum += (src[j])[i];


    dest[i] = sum * scale_factor;
  }
}

//add src to the destination. 
template <typename T>
void inline avg_list_stream(FVector<T> & dest, FVector<T> *src, unsigned N) {
  T scale_factor = 1.0 /(T)N;

  for (size_t i = 0; i < dest.size; i++) 
  {
    T sum  = 0.0;
    for (unsigned j = 0; j < N; j++)  
      sum += (src[j])[i];


    dest[i] = sum * scale_factor;
  }
}


//add src to the destination. streaming load from src, no tag for the source...
template <typename T>
void inline add(FVector<T> & dest, FVector<T> const &src) {
  unsigned vix = 0;
  unsigned i = 0;
  for (size_t i = 0; i < dest.size; i++) {
    dest.values[i]  += dest.values[i] + src.values[i];
  }
}

//add src to the destination. 
template <typename T>
void inline add_mult(FVector<T> & dest, FVector<T> const &src1, FVector<T> const &src2, T scale_factor) {
  unsigned vix = 0;
  unsigned i = 0;
  for (size_t i = 0; i < dest.size; i++) {
    dest.values[i]  += (src1.values[i] + src2.values[i])*scale_factor;
  }
}


template <typename T>
bool IsValid(SVector<T> const &v) {
  for (size_t i = 0; i < v.size; i++) {
    if (v.index[i] < 0) { assert(false); return false; }
    if (v.index[i] >= v.size) { assert(false); return false; }
  }
  return true;
}


template <typename T>
double inline Norm2(FVector<T> const &v) {
  double norm = 0;
  for (size_t i = 0; i < v.size; i++) {
    norm += v.values[i] * v.values[i];
  }
  return norm;
}

template <typename T>
double inline Norm2WithoutSquare(FVector<T> const &v) {
  double norm = 0;
  for (size_t i = 0; i < v.size; i++) {
    norm += v.values[i] * v.values[i];
  }
  return sqrt(norm);
}

template <typename T>
double inline Norm2WithoutSquare(SVector<T> const &v) {
  double norm = 0;
  for (size_t i = 0; i < v.size; i++) {
    norm += v.values[i] * v.values[i];
  }
  return sqrt(norm);
}


template <typename T, typename int_T>
void inline Project(SVector<T> const& v, FVector<int_T> const &indexes, T *out) {
  unsigned vix = 0;
  unsigned i = 0;
  while (static_cast<unsigned>(i) < indexes.size) {
    if (vix >= v.size) {
      out[i++] = 0;
      continue;
    }
    if (indexes.values[i] > v.index[vix]) {
      vix++;
      continue;
    }
    if (indexes.values[i] < v.index[vix]) {
      out[i++] = 0;
      continue;
    }
    assert(indexes.values[i] == v.index[vix]);
    out[i++] = v.values[vix];
  }
}

template <typename float_t>
void inline SimplexProject(FVector<float_t> &vec) {
  float_t v[vec.size];
  float_t *vecf = vec.values;
  for (unsigned i = 0; i < vec.size; i++) {
    v[i] = vecf[i];
  }
  util::QuickSort(v, vec.size);
  //std::vector<float_t> v(vec.values, &vec.values[vec.size]);
  //std::sort(v.begin(), v.end());
  int i     = vec.size-2;
  double ti = 0.0, ti_sum = 0.0;

  while (i >= 0) {
    ti_sum += v[i+1];
    ti  = (ti_sum - 1)/(vec.size - 1 -i);
    if(ti > v[i]) break;
    i--;
  }

  for (unsigned k = 0; k < vec.size; k++) {
    vec.values[k] = std::max(0.0, vec.values[k] - ti);
  }
}

// Only apply threshold to the masked entries
void MaskThresholdZero(vector::FVector<double> &x,
                       const vector::FVector<size_t> &mask) {
  for (size_t i = 0; i < mask.size; i++) {
    if (x.values[mask.values[i]] <= 0) {
      x.values[mask.values[i]]  = 0;
    }
  }
}

template <typename float_t>
void inline Zero(FVector<float_t> &v) {
  for (unsigned i = 0; i < v.size; i++) {
    v.values[i] = 0;
  }
}

template <typename float_t>
void inline Zero(SVector<float_t> &v) {
  for (unsigned i = 0; i < v.size; i++) {
    v.values[i] = 0;
  }
}

template <typename float_t>
void inline ThresholdZero(SVector<float_t> &v) {
  for (unsigned i = 0; i < v.size; i++) {
    if (v.values[i] < 0) 
      v.values[i] = 0;
  }
}

template <typename float_u>
void inline CopyInto(FVector<float_u> const &u, FVector<float_u> &out) {
  float_u * const __restrict__ outv = out.values;
  float_u const * const __restrict__ inv = u.values;
  memcpy(outv, inv, sizeof(float_u)*u.size);
  out.size = u.size;
}

template <typename float_u>
void inline CopyInto_stream(FVector<float_u> const &u, FVector<float_u> &out) {
  float_u * const __restrict__ outv = out.values;
  float_u const * const __restrict__ inv = u.values;
  memcpy(outv, inv, sizeof(float_u)*u.size);
  out.size = u.size;
}

} // namespace vector
} // namespace hazy
#endif
