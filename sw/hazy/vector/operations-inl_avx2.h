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
#include <immintrin.h>


namespace hazy {
namespace vector {
	
#define uint32_t unsigned int
#define BITS_OF_ONE_CACHE_LINE 512
	
	//Suppose the size of each value of training dataset is 32-bit, always true for our case...
	uint32_t compute_num_CLs_per_sample(uint32_t dr_numFeatures) {
	  //With the chunk of 512 features...
	  uint32_t main_num 		  = (dr_numFeatures/BITS_OF_ONE_CACHE_LINE)*32; //It is for CLs
	  uint32_t rem_num			  = 0;
	
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
	  int num_CLs_per_sample	 = compute_num_CLs_per_sample(numFeatures);
	  //printf("num_CLs_per_sample = %d\n", num_CLs_per_sample);
	  //uint32_t *a_fpga_tmp	   = a_bitweaving_fpga;
	  uint32_t address_index	 = 0;
	  int num_features_main 	 = (numFeatures/BITS_OF_ONE_CACHE_LINE)*BITS_OF_ONE_CACHE_LINE;  
	
		//Deal with the main part of dr_numFeatures.
		for (int j = 0; j < num_features_main; j += BITS_OF_ONE_CACHE_LINE)
		{
		  uint32_t tmp_buffer[BITS_OF_ONE_CACHE_LINE] = {0};
		  //1: initilization off tmp buffer..
		  for (int k = 0; k < BITS_OF_ONE_CACHE_LINE; k += 8)
		  {
			__m256i v1 = _mm256_loadu_si256( (__m256i const *)&(src[j+k]) );
			_mm256_storeu_si256( (__m256i *)&(tmp_buffer[k]), v1);
		  }  

		  //2: focus on the data from index: j...
		  for (int k = 0; k < 32; k++)
		  { 
			  unsigned char result_buffer[BITS_OF_ONE_CACHE_LINE/8] = {0};	//16 ints == 512 bits...
			  //2.1: re-order the data according to the bit-level...
			  for (int m = 0; m < BITS_OF_ONE_CACHE_LINE; m+=8)
			  {
				__m256i v_data	   = _mm256_loadu_si256((__m256i const *)&tmp_buffer[m]);
				int tmp 		   = _mm256_movemask_ps( _mm256_castsi256_ps(v_data) );
				result_buffer[m/8] = (unsigned char)tmp;
				v_data			   = _mm256_slli_epi32(v_data, 1);
				_mm256_storeu_si256((__m256i *)&tmp_buffer[m], v_data);
			  }
			  
			  //2.2: store the bit-level result back to the memory...
			  __m256i v_data	 = _mm256_loadu_si256((__m256i const *)&result_buffer[0]);
			   _mm256_storeu_si256((__m256i *)&dest[address_index+0], v_data);
					   v_data	  = _mm256_loadu_si256((__m256i const *)&result_buffer[32]);
				_mm256_storeu_si256((__m256i *)&dest[address_index+8], v_data);
			  address_index += 16;
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
			  tmp_buffer[m] 	  = tmp_buffer[m] << 1; 	  
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
	uint64_t numFeatures	  = dest.size;
	unsigned short* vec_short = dest.values;
	unsigned int* vec_int	  = src.values;

#define uint32_t unsigned int
#define BITS_OF_ONE_CACHE_LINE 512

	__m256i v_offset = _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0); 
	__m256i v_mask	 = _mm256_set1_epi32(0x01010101);
	__m256i v_sum, v_data, v_data_1;
	__m256i v_high, v_low;
			

	uint32_t num_features_main = (numFeatures/BITS_OF_ONE_CACHE_LINE) * BITS_OF_ONE_CACHE_LINE;
			
	for (size_t base = 0; base < numFeatures; base += BITS_OF_ONE_CACHE_LINE) 
	{	
		uint32_t num_r_f       = numFeatures - num_features_main;
		uint32_t num_iteration; // << 5 
		uint32_t stride        = 0;

		if (base < num_features_main)
		{
			stride             = 16;
			num_iteration      = 16;
		}
		else
		{   
			num_iteration = ( ((num_r_f+31)>>5)); // << 5 
			if (num_r_f <= 64)												 //////remainder <= 64
						stride         = 2;
			else if (num_r_f <= 128)										  //////64 < remainder <= 128
						stride         = 4;
			else if (num_r_f <= 256)										  //////128 < remainder <= 256
						stride         = 8;
			else if (num_r_f < 512) 										 //////256 < remainder < 512
						stride         = 16;
		}
			//	printf("In the outer loop: base = %d\n", base);

				//for each 32 values.
		for (size_t offset = 0; offset < num_iteration; offset++)
		{
			if (offset < num_bits)
			{
				_mm_prefetch((char *)(&src[base + 1*BITS_OF_ONE_CACHE_LINE+ offset*16]), _MM_HINT_NTA);	//Stay at L1
				//_mm_prefetch((char *)(&src[base + 2*BITS_OF_ONE_CACHE_LINE+ offset*16]), _MM_HINT_T2);//Stay at L1
			}		
			v_sum = _mm256_set1_epi32(0);
			unsigned int data_src;
				//printf("In the inner loop\n");
	
			data_src = src[base + stride*0 + offset];          //0
			v_data	 =	_mm256_set1_epi32(data_src); 
			v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
			v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
			v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 7) );

			data_src = src[base + stride*1 + offset];          //1
			v_data	 =	_mm256_set1_epi32(data_src); 
			v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
			v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
			v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 6) );

			data_src = src[base + stride*2 + offset];          //2
			v_data	 =	_mm256_set1_epi32(data_src); 
			v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
			v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
			v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 5) );

			data_src = src[base + stride*3 + offset];          //3
			v_data	 =	_mm256_set1_epi32(data_src); 
			v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
			v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
			v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 4) );

			data_src = src[base + stride*4 + offset];          //4
			v_data	 =	_mm256_set1_epi32(data_src); 
			v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
			v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
			v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 3) );

			data_src = src[base + stride*5 + offset];          //5
			v_data	 =	_mm256_set1_epi32(data_src); 
			v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
			v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
			v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 2) );

			data_src = src[base + stride*6 + offset];          //6
			v_data	 =	_mm256_set1_epi32(data_src); 
			v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
			v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
			v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 1) );

			data_src = src[base + stride*7 + offset];          //7
			v_data	 =	_mm256_set1_epi32(data_src); 
			v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
			v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
			v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 0) );

					//unsigned char sum_array_high[64]; //32 is enough.
					//_mm256_store_si256((__m256i *)sum_array_high, v_sum);
			v_high = v_sum;
			v_sum = _mm256_set1_epi32(0);

					data_src = src[base + stride*8 + offset];          //8
					v_data	 =	_mm256_set1_epi32(data_src); 
					v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
					v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
					v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 7) );
#if 1
			if (num_bits >=10)
			{
					data_src = src[base + stride*9 + offset];          //9
					v_data	 =	_mm256_set1_epi32(data_src); 
					v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
					v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
					v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 6) );
			 if (num_bits >=11)
			 {
					data_src = src[base + stride*10 + offset];          //10
					v_data	 =	_mm256_set1_epi32(data_src); 
					v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
					v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
					v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 5) );
			  if (num_bits >=12)
			  {
					data_src = src[base + stride*11 + offset];          //11
					v_data	 =	_mm256_set1_epi32(data_src); 
					v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
					v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
					v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 4) );
				if (num_bits >=13)
				{
					data_src = src[base + stride*12 + offset];          //12
					v_data	 =	_mm256_set1_epi32(data_src); 
					v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
					v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
					v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 3) );
				if (num_bits >=14)
				{
					data_src = src[base + stride*13 + offset];          //13
					v_data	 =	_mm256_set1_epi32(data_src); 
					v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
					v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
					v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 2) );
				 if (num_bits >=15)
				 {
					data_src = src[base + stride*14 + offset];          //14
					v_data	 =	_mm256_set1_epi32(data_src); 
					v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
					v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
					v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 1) );
				  if (num_bits >=16)
				  {
					data_src = src[base + stride*15 + offset];          //15
					v_data	 =	_mm256_set1_epi32(data_src); 
					v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
					v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
					v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 0) );
				  }}}}}} }	
#endif	
				v_low = v_sum;

			__m256i v_shuffle_constant = _mm256_set_epi8 (15, 11,  7,  3, 
			 				                              14, 10,  6,  2, 
			 				                              13,  9,  5,  1, 
			 				                              12,  8,  4,  0,
			 				                              15, 11,  7,  3, 
			 				                              14, 10,  6,  2, 
			 				                              13,  9,  5,  1, 
			 				                              12,  8,  4,  0);
    		__m256i v_perm_constant = _mm256_set_epi32 (7, 3,  6, 2,   
                                                		5,  1, 4,  0); 

			__m256i v_data_2        = _mm256_shuffle_epi8(v_low, v_shuffle_constant);
			__m256i v_low_tmp       = _mm256_permutevar8x32_epi32(v_data_2, v_perm_constant);


			       v_data_2         = _mm256_shuffle_epi8(v_high, v_shuffle_constant);
			__m256i v_high_tmp      = _mm256_permutevar8x32_epi32(v_data_2, v_perm_constant);



				//Now, we have v_high (3, 1), v_low (2, 0)... objective: (3, 2), (1, 0)...
				__m256i v_data_low  = _mm256_unpacklo_epi8(v_low_tmp, v_high_tmp);				
				__m256i v_data_high = _mm256_unpackhi_epi8(v_low_tmp, v_high_tmp);

				_mm256_storeu_si256((__m256i *)(&vec_short[base + offset*32 +  0]), v_data_low );				
				_mm256_storeu_si256((__m256i *)(&vec_short[base + offset*32 + 16]), v_data_high);				


				__m128i v_data_128_low  = _mm_loadu_si128((__m128i*)(&vec_short[base + offset*32 + 16]) );
				__m128i v_data_128_high = _mm_loadu_si128((__m128i*)(&vec_short[base + offset*32 + 8]) );

				_mm_storeu_si128((__m128i *)(&vec_short[base + offset*32 +  8]), v_data_128_low );				
				_mm_storeu_si128((__m128i *)(&vec_short[base + offset*32 + 16]), v_data_128_high);	


		}
		if (base == num_features_main)
			break;
	
	}
}
	
	
void inline Convert_from_bitweaving(FVector<unsigned char> & dest, FVector<unsigned int> &src, unsigned num_bits) 
{
		uint64_t numFeatures	= dest.size;
		unsigned char* vec_char = dest.values;
		unsigned int* vec_int	 = src.values;
		
#define uint32_t unsigned int
#define BITS_OF_ONE_CACHE_LINE 512
		
    	__m256i v_offset = _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0); 
    	__m256i v_mask   = _mm256_set1_epi32(0x01010101);
    	__m256i v_sum, v_data, v_data_1, v_data_2, v_data_3, v_data_4, v_data_5, v_data_6, v_data_7, v_data_0;
	
		uint32_t num_features_main = (numFeatures/BITS_OF_ONE_CACHE_LINE) * BITS_OF_ONE_CACHE_LINE;
		
		//for (size_t i = 0; i < numFeatures; i++) 
		//vec_char[i] = extract_from_bitweaving(src.values, i, numFeatures);
		
		//Compute the main part of numFeatures.
		//For each 512-code chunk
	for (size_t base = 0; base < numFeatures; base += BITS_OF_ONE_CACHE_LINE) 
	{

		uint32_t num_r_f       = numFeatures - num_features_main;
		uint32_t num_iteration; // << 5 
		uint32_t stride        = 0;
		if (base < num_features_main)
		{
			stride             = 16;
			num_iteration      = 16;
		}
		else
		{   
			num_iteration = ( ((num_r_f+31)>>5)); // << 5 
			if (num_r_f <= 64)												 //////remainder <= 64
			{ 
				stride         = 2;
			}
			else if (num_r_f <= 128)										  //////64 < remainder <= 128
			{ 
				stride         = 4;
			}
			else if (num_r_f <= 256)										  //////128 < remainder <= 256
			{ 
				stride         = 8;
			}
			else if (num_r_f < 512) 										 //////256 < remainder < 512
			{ 
				stride         = 16;
			}
		}

	  
		  //_mm_prefetch((char *)(&src[base + 1*BITS_OF_ONE_CACHE_LINE+ 0]), _MM_HINT_NTA);	//Stay at L1
		  //_mm_prefetch((char *)(&src[base + 1*BITS_OF_ONE_CACHE_LINE+16]), _MM_HINT_NTA);	//Stay at L1
		  //_mm_prefetch((char *)(&src[base + 1*BITS_OF_ONE_CACHE_LINE+32]), _MM_HINT_NTA);	//Stay at L1
		  //_mm_prefetch((char *)(&src[base + 1*BITS_OF_ONE_CACHE_LINE+48]), _MM_HINT_NTA);	//Stay at L1
		  //_mm_prefetch((char *)(&src[base + 1*BITS_OF_ONE_CACHE_LINE+64]), _MM_HINT_NTA);	//Stay at L1
		  //_mm_prefetch((char *)(&src[base + 1*BITS_OF_ONE_CACHE_LINE+80]), _MM_HINT_NTA);	//Stay at L1
		  //_mm_prefetch((char *)(&src[base + 1*BITS_OF_ONE_CACHE_LINE+96]), _MM_HINT_NTA);	//Stay at L1
		  //_mm_prefetch((char *)(&src[base + 1*BITS_OF_ONE_CACHE_LINE+112]), _MM_HINT_NTA);	//Stay at L1
/*
	  if (num_bits = 8)
	  {
		//size_t base = num_features_main;
						//for each 32 values.
		for (size_t offset = 0; offset < num_iteration; offset++)
		{
				if (offset < num_bits)
				{
					_mm_prefetch((char *)(&src[base + 1*BITS_OF_ONE_CACHE_LINE+ offset*16]), _MM_HINT_NTA);	//Stay at L1
					//_mm_prefetch((char *)(&src[base + 2*BITS_OF_ONE_CACHE_LINE+ offset*16]), _MM_HINT_T2);	//Stay at L1
				}
				v_sum = _mm256_set1_epi32(0);
				unsigned int data_src_0, data_src_1, data_src_2, data_src_3, data_src_4, data_src_5, data_src_6, data_src_7;

				data_src_0 = src[base + stride*0 + offset];
				data_src_1 = src[base + stride*1 + offset];
				data_src_2 = src[base + stride*2 + offset];
				data_src_3 = src[base + stride*3 + offset];

				v_data_0   =  _mm256_set1_epi32(data_src_0); 
				v_data_1   =  _mm256_set1_epi32(data_src_1); 
				v_data_2   =  _mm256_set1_epi32(data_src_2); 
				v_data_3   =  _mm256_set1_epi32(data_src_3); 

				v_data_0   =  _mm256_srav_epi32(v_data_0, v_offset); //shift it...
				v_data_1   =  _mm256_srav_epi32(v_data_1, v_offset); //shift it...
				v_data_2   =  _mm256_srav_epi32(v_data_2, v_offset); //shift it...
				v_data_3   =  _mm256_srav_epi32(v_data_3, v_offset); //shift it...

				v_data_0   =  _mm256_and_si256 (v_data_0, v_mask  ); //3  v_data
				v_data_1   =  _mm256_and_si256 (v_data_1, v_mask  ); //3  v_data
				v_data_2   =  _mm256_and_si256 (v_data_2, v_mask  ); //3  v_data
				v_data_3   =  _mm256_and_si256 (v_data_3, v_mask  ); //3  v_data

				v_sum      =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_0, 7) );
				v_sum      =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 6) );
				v_sum      =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_2, 5) );
				v_sum      =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_3, 4) );


				data_src_4 = src[base + stride*4 + offset];
				data_src_5 = src[base + stride*5 + offset];
				data_src_6 = src[base + stride*6 + offset];
				data_src_7 = src[base + stride*7 + offset];

				v_data_4   =  _mm256_set1_epi32(data_src_4); 
				v_data_5   =  _mm256_set1_epi32(data_src_5); 
				v_data_6   =  _mm256_set1_epi32(data_src_6); 
				v_data_7   =  _mm256_set1_epi32(data_src_7); 

				v_data_4   =  _mm256_srav_epi32(v_data_4, v_offset); //shift it...
				v_data_5   =  _mm256_srav_epi32(v_data_5, v_offset); //shift it...
				v_data_6   =  _mm256_srav_epi32(v_data_6, v_offset); //shift it...
				v_data_7   =  _mm256_srav_epi32(v_data_7, v_offset); //shift it...

				v_data_4   =  _mm256_and_si256 (v_data_4, v_mask  ); //3  v_data
				v_data_5   =  _mm256_and_si256 (v_data_5, v_mask  ); //3  v_data
				v_data_6   =  _mm256_and_si256 (v_data_6, v_mask  ); //3  v_data
				v_data_7   =  _mm256_and_si256 (v_data_7, v_mask  ); //3  v_data

				v_sum      =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_4, 3) );
				v_sum      =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_5, 2) );
				v_sum      =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_6, 1) );
				v_sum      =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_7, 0) );

			__m256i v_shuffle_constant = _mm256_set_epi8 (15, 11,  7,  3, 
			 				                              14, 10,  6,  2, 
			 				                              13,  9,  5,  1, 
			 				                              12,  8,  4,  0,
			 				                              15, 11,  7,  3, 
			 				                              14, 10,  6,  2, 
			 				                              13,  9,  5,  1, 
			 				                              12,  8,  4,  0);
			__m256i v_data_tmp      = _mm256_shuffle_epi8(v_sum, v_shuffle_constant);
    		__m256i v_perm_constant = _mm256_set_epi32 (7, 3,  6, 2,   
                                                		5,  1, 4,  0); 
			__m256i v_result = _mm256_permutevar8x32_epi32(v_data_tmp, v_perm_constant);
			_mm256_store_si256((__m256i *)(&vec_char[base + offset*32]), v_result);
		}

	  }
	  else
*/	  {
		//size_t base = num_features_main;
						//for each 32 values.
		for (size_t offset = 0; offset < num_iteration; offset++)
		{
			if (offset < num_bits)
			{
				_mm_prefetch((char *)(&src[base + 1*BITS_OF_ONE_CACHE_LINE+ offset*16]), _MM_HINT_T0); //Stay at L1
				//_mm_prefetch((char *)(&src[base + 2*BITS_OF_ONE_CACHE_LINE+ offset*16]), _MM_HINT_T2);	//Stay at L1
			}
		
				v_sum = _mm256_set1_epi32(0);
				unsigned int data_src;

				data_src = src[base + stride*0 + offset];
				v_data   =  _mm256_set1_epi32(data_src); 
				v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
				v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
				v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 7) );
			if (num_bits >=2)
			{
				data_src = src[base + stride*1 + offset];
				v_data   =  _mm256_set1_epi32(data_src); 
				v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
				v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
				v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 6) );
			 if (num_bits >=3)
			 {
				data_src = src[base + stride*2 + offset];
				v_data   =  _mm256_set1_epi32(data_src); 
				v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
				v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
				v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 5) );
			  if (num_bits >=4)
			  {
				data_src = src[base + stride*3 + offset];
				v_data   =  _mm256_set1_epi32(data_src); 
				v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
				v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
				v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 4) );
			   if (num_bits >=5)
			   {
				data_src = src[base + stride*4 + offset];
				v_data   =  _mm256_set1_epi32(data_src); 
				v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
				v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
				v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 3) );
			   if (num_bits >=6)
			   {
				data_src = src[base + stride*5 + offset];
				v_data   =  _mm256_set1_epi32(data_src); 
				v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
				v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
				v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 2) );
			    if (num_bits >=7)
			    {
				data_src = src[base + stride*6 + offset];
				v_data   =  _mm256_set1_epi32(data_src); 
				v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
				v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
				v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 1) );
			    if (num_bits >=8)
			    {
				data_src = src[base + stride*7 + offset];
				v_data   =  _mm256_set1_epi32(data_src); 
				v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
				v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
				v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 0) );
			 
			 }}}}}} }

			__m256i v_shuffle_constant = _mm256_set_epi8 (15, 11,  7,  3, 
			 				                              14, 10,  6,  2, 
			 				                              13,  9,  5,  1, 
			 				                              12,  8,  4,  0,
			 				                              15, 11,  7,  3, 
			 				                              14, 10,  6,  2, 
			 				                              13,  9,  5,  1, 
			 				                              12,  8,  4,  0);
			__m256i v_data_2        = _mm256_shuffle_epi8(v_sum, v_shuffle_constant);
    		__m256i v_perm_constant = _mm256_set_epi32 (7, 3,  6, 2,   
                                                		5,  1, 4,  0); 
			__m256i v_result = _mm256_permutevar8x32_epi32(v_data_2, v_perm_constant);
			_mm256_store_si256((__m256i *)(&vec_char[base + offset*32]), v_result);
		}
	  }
			//Do the storing...
	}

		
}
	
	

//add src to the destination. 
template <typename T>
void inline avg_list(FVector<T> & dest, FVector<T> *src, unsigned N) {
  const uint64_t n0         = dest.size;
  const uint64_t n1         = (n0 >> 4) << 4;

  T scale_factor            = 1.0 /(T)N;
  const __m256 scale_const  = _mm256_set1_ps(1.0 /(T)N); //T scale_factor = 1.0 /(T)N;

  for (size_t i = 0; i < n1; i+= 32) 
  {
    //T sum     = 0.0;
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();    
    //for (unsigned j = 0; j < N; j++)  
    //  sum += (src[j])[i];
    for (unsigned j = 0; j < N; j++)
    {
      __m256 v1 = _mm256_loadu_ps( (float const *)&((src[j])[i + 0 ]) );
      __m256 v2 = _mm256_loadu_ps( (float const *)&((src[j])[i + 8 ]) );
      __m256 v3 = _mm256_loadu_ps( (float const *)&((src[j])[i + 16]) );
      __m256 v4 = _mm256_loadu_ps( (float const *)&((src[j])[i + 24]) );      
      sum1      = _mm256_add_ps ( sum1, v1 );
      sum2      = _mm256_add_ps ( sum2, v2 );
      sum3      = _mm256_add_ps ( sum3, v3 );
      sum4      = _mm256_add_ps ( sum4, v4 );
    }  

    //dest[i] = sum * scale_factor;
    _mm256_storeu_ps((float *)(&dest[i + 0 ]), _mm256_mul_ps(sum1, scale_const) );
    _mm256_storeu_ps((float *)(&dest[i + 8 ]), _mm256_mul_ps(sum2, scale_const) );
    _mm256_storeu_ps((float *)(&dest[i + 16]), _mm256_mul_ps(sum3, scale_const) );
    _mm256_storeu_ps((float *)(&dest[i + 24]), _mm256_mul_ps(sum4, scale_const) );
  }

  for (size_t i = n1; i < n0; i++)
  {
    T sum   = 0.0;
    for (unsigned j = 0; j < N; j++)  
      sum  += (src[j])[i];

    dest[i] = sum * scale_factor;
  }

}


//add src to the destination. 
template <typename T>
void inline avg_list_stream(FVector<T> & dest, FVector<T> *src, unsigned N) {
  const uint64_t n0         = dest.size;
  const uint64_t n1         = (n0 >> 4) << 4;

  T scale_factor            = 1.0 /(T)N;
  const __m256 scale_const  = _mm256_set1_ps(1.0 /(T)N); //T scale_factor = 1.0 /(T)N;

  for (size_t i = 0; i < n1; i+= 32) 
  {
    //T sum     = 0.0;
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();    
    //for (unsigned j = 0; j < N; j++)  
    //  sum += (src[j])[i];
    for (unsigned j = 0; j < N; j++)
    {
		//__m256 v1 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i const *) (inv + i + 0)	));
   
      __m256 v1 =  _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i const *)&((src[j])[i + 0 ]) ));
      __m256 v2 =  _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i const *)&((src[j])[i + 8 ]) ));
      __m256 v3 =  _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i const *)&((src[j])[i + 16]) ));
      __m256 v4 =  _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i const *)&((src[j])[i + 24]) ));      
      sum1      = _mm256_add_ps ( sum1, v1 );
      sum2      = _mm256_add_ps ( sum2, v2 );
      sum3      = _mm256_add_ps ( sum3, v3 );
      sum4      = _mm256_add_ps ( sum4, v4 );
    }  

    //dest[i] = sum * scale_factor;
    _mm256_stream_ps((float *)(&dest[i + 0 ]), _mm256_mul_ps(sum1, scale_const) );
    _mm256_stream_ps((float *)(&dest[i + 8 ]), _mm256_mul_ps(sum2, scale_const) );
    _mm256_stream_ps((float *)(&dest[i + 16]), _mm256_mul_ps(sum3, scale_const) );
    _mm256_stream_ps((float *)(&dest[i + 24]), _mm256_mul_ps(sum4, scale_const) );
  }

  for (size_t i = n1; i < n0; i++)
  {
    T sum   = 0.0;
    for (unsigned j = 0; j < N; j++)  
      sum  += (src[j])[i];

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

//template <typename float_t>
void inline Zero(FVector<float> &v) {
  float_t *  outv           = v.values;

  const uint64_t n0         = v.size;
  const uint64_t n1         = (n0 >> 4) << 4;
  
  //printf("In the function Zero, with SIMD\n");

  const __m256 zero_v       = _mm256_set1_ps(0.0); //T scale_factor = 1.0 /(T)N;
  float_t scale_factor            = 0.0;

  for (size_t i = 0; i < n1; i+= 32) 
  {
    //dest[i] = sum * scale_factor;
    _mm256_storeu_ps((float *)(outv + i + 0 ), zero_v );
    _mm256_storeu_ps((float *)(outv + i + 8 ), zero_v );
    _mm256_storeu_ps((float *)(outv + i + 16), zero_v );
    _mm256_storeu_ps((float *)(outv + i + 24), zero_v );
  }

  for (size_t i = n1; i < n0; i++)
  {
    outv[i] = scale_factor;
  }  

}


template <typename float_t>
void inline Zero(FVector<float_t> &v) {
  float_t *  outv           = v.values;
  const uint64_t n0         = v.size;
  //printf("In the function Zero, with Scalar\n");

  for (size_t i = 0; i < n0; i++)
  {
    outv[i] = (float_t)0;
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
  float_u *  outv           = out.values;
  float_u *  inv            = u.values;

  const uint64_t n0         = out.size;
  const uint64_t n1         = (n0 >> 5) << 5;
  const uint64_t n2         = (n0 >> 3) << 3;


  for (size_t i = 0; i < n1; i+= 32) 
  {
    __m256 v1 = _mm256_loadu_ps( (float const *) (inv + i + 0)  );
    __m256 v2 = _mm256_loadu_ps( (float const *) (inv + i + 8)  );
    __m256 v3 = _mm256_loadu_ps( (float const *) (inv + i + 16) );
    __m256 v4 = _mm256_loadu_ps( (float const *) (inv + i + 24) );      

    _mm256_storeu_ps((float *)(outv + i + 0 ), v1 );
    _mm256_storeu_ps((float *)(outv + i + 8 ), v2 );
    _mm256_storeu_ps((float *)(outv + i + 16), v3 );
    _mm256_storeu_ps((float *)(outv + i + 24), v4 );
  }

  for (size_t i = n1; i < n2; i+= 8) 
  {
    __m256 v1 = _mm256_loadu_ps( (float const *) (inv + (i +  0) )  );
    _mm256_storeu_ps((float *)(outv + (i +  0)), v1 );
  }
  

  for (size_t i = n2; i < n0; i++)
  {
    outv[i] = inv[i];
  }

}


template <typename float_u>
void inline CopyInto_stream(FVector<float_u> const &u, FVector<float_u> &out) {
  float_u *  outv           = out.values;
  float_u *  inv            = u.values;

  const uint64_t n0         = out.size;
  const uint64_t n1         = (n0 >> 5) << 5;
  const uint64_t n2         = (n0 >> 3) << 3;


  for (size_t i = 0; i < n1; i+= 32) 
  {
    __m256 v1 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i const *) (inv + i + 0)  ));
    __m256 v2 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i const *) (inv + i + 8)  ));
    __m256 v3 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i const *) (inv + i + 16) ));
    __m256 v4 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i const *) (inv + i + 24) ));      

    _mm256_stream_ps((float *)(outv + i + 0 ), v1 );
    _mm256_stream_ps((float *)(outv + i + 8 ), v2 );
    _mm256_stream_ps((float *)(outv + i + 16), v3 );
    _mm256_stream_ps((float *)(outv + i + 24), v4 );
  }
  
  for (size_t i = n1; i < n2; i+= 8) 
  {
    __m256 v1 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i const *) (inv + i + 0)  ));
   	_mm256_stream_ps((float *)(outv + i + 0 ), v1 );
  }

  for (size_t i = n1; i < n0; i++)
  {
    outv[i] = inv[i];
  }

}


} // namespace vector
} // namespace hazy
#endif
