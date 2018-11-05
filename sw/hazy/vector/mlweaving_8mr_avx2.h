// Copyright 2018 Zeke Wang, ETH, Zurich
// Author : Zeke Wang (zeke.wang [at] inf.ethz.ch) 
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//    http://www.apache.org/licenses/LICENSE-2.0
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

#ifndef MLWEAVING_H
#define MLWEAVING_H


// This file is mainly about how to compress the dataset into MLWeaving layout and 
// to get data (char or short) out of MLWeaving layout, 
// MLWeaving: pack 8 samples together, 
// Support 8 memory arrays concurrently. 

#include "string.h"
#include "hazy/vector/fvector.h"
#include <immintrin.h>


namespace hazy {
namespace vector {

#define BITS_OF_CL      512
#define NUM_BANKS       8
#define BITS_OF_BANK   (BITS_OF_CL/NUM_BANKS)

#ifndef NUM_MR	
#define NUM_MR          8
#endif

#define BITS_OF_MR     (32/NUM_MR)
#define CEIL(A, B)     ((A+B-1)&(~(B-1))) 

//This function performs weaving on the input data array: src.
//Input : src  (dense, unsigned int) 	
//Output: dest (in MLWeaving), its size is   ceil(numSamples, )*ceil()
void mlweaving_on_sample(uint32_t *dest, uint32_t *src, uint32_t numSamples, uint32_t numFeatures) 
{
	uint32_t address_index         = 0;
	uint32_t numSamples_align      = CEIL(numSamples,  NUM_BANKS    );
	uint32_t numFeatures_align     = CEIL(numFeatures, BITS_OF_BANK );
	uint64_t mr_offset             = numSamples_align*numFeatures_align/NUM_MR;

	///Do the bitWeaving to the training data...
	for (uint32_t i = 0; i < numSamples; i+=NUM_BANKS)
	{   
		uint32_t samples_in_batch = ( (i+NUM_BANKS)<numSamples )? NUM_BANKS:(numSamples-i); 
		// j; //Deal with the main part of numFeatures.
		for (uint32_t j = 0; j < numFeatures; j += BITS_OF_BANK)//(numFeatures/BITS_OF_BANK)*BITS_OF_BANK
		{
			uint32_t bits_in_batch = ( (j+BITS_OF_BANK)<numFeatures )? BITS_OF_BANK:(numFeatures-j); 
			uint32_t tmp_buffer[512] = {0};
			//1: initilization off tmp buffer..
			for (int k = 0; k < samples_in_batch; k++)//NUM_BANKS
				for (int m = 0; m < bits_in_batch; m++) //BITS_OF_BANK
					tmp_buffer[ k*BITS_OF_BANK+m ] = src[ (i + k)*numFeatures + (j+m) ];

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
			    //Only needs to change here to support multiple memory arrays. based on the value of k...
			    //                                    which MR        which index in mr
			    uint64_t common_offset = mr_offset*(k/BITS_OF_MR) + 16*(k%BITS_OF_MR);

				dest[address_index+common_offset+ 0] = result_buffer[0]; //dest[address_index++]
				dest[address_index+common_offset+ 1] = result_buffer[1]; //dest[address_index++]
				dest[address_index+common_offset+ 2] = result_buffer[2]; //dest[address_index++]
				dest[address_index+common_offset+ 3] = result_buffer[3]; //dest[address_index++]
				dest[address_index+common_offset+ 4] = result_buffer[4]; //dest[address_index++]
				dest[address_index+common_offset+ 5] = result_buffer[5]; //dest[address_index++]
				dest[address_index+common_offset+ 6] = result_buffer[6]; //dest[address_index++]
				dest[address_index+common_offset+ 7] = result_buffer[7]; //dest[address_index++]
				dest[address_index+common_offset+ 8] = result_buffer[8]; //dest[address_index++]
				dest[address_index+common_offset+ 9] = result_buffer[9]; //dest[address_index++]
				dest[address_index+common_offset+10] = result_buffer[10];//dest[address_index++]
				dest[address_index+common_offset+11] = result_buffer[11];//dest[address_index++]
				dest[address_index+common_offset+12] = result_buffer[12];//dest[address_index++]
				dest[address_index+common_offset+13] = result_buffer[13];//dest[address_index++]
				dest[address_index+common_offset+14] = result_buffer[14];//dest[address_index++]
				dest[address_index+common_offset+15] = result_buffer[15];//dest[address_index++]
			}
			address_index += 16*BITS_OF_MR; //64 handle with *i-->features*.
		}
	}
}


//This function retrives the sample from the mlweaving layout with address: src. 
//dest: destination fvector 
//src : address of mlweaving array
//index: sample index
//num_bits: number of bits to retrieve. 
//T: template is used to generalize to uchar, ushort, uint.

void inline retrieve_from_mlweaving(FVector<uint8_t> & dest, uint32_t *src, uint32_t index, uint32_t num_bits, uint32_t numSamples) 
{	
	//dest's information
	uint32_t numFeatures        = dest.size;
	uint8_t* vec_char           = dest.values;

	//aligned number of features. 
	uint32_t numFeaturesAlign   = ( (numFeatures+(BITS_OF_BANK-1))&(~(BITS_OF_BANK-1)) ); //round up to the nearest mulitple of BITS_OF_BANK
	uint32_t numSamples_align   = CEIL(numSamples,  NUM_BANKS    );
	uint64_t mr_offset          = numSamples_align*numFeaturesAlign/NUM_MR;

	//calculate the address of sample of the index: index. 
	uint32_t  chunk_offset      = (index/NUM_BANKS) * numFeaturesAlign*NUM_BANKS/NUM_MR; //chunk index * chunk size
	uint32_t  sample_offset     = (index%NUM_BANKS) * (BITS_OF_BANK/32); //chunk index * chunk size
	uint32_t* sample_addr       = src + chunk_offset + sample_offset; //identify the address of the sampel (index)

    __m256i v_offset = _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0); 
    __m256i v_mask   = _mm256_set1_epi32(0x01010101);
    __m256i v_sum, v_data, v_data_1, v_data_2, v_data_3, v_data_4, v_data_5, v_data_6, v_data_7, v_data_8, v_data_0;

	for (size_t i = 0; i < numFeatures; i+= 32) 
	{  //prepare one number for each iteration.  
		uint32_t main_offset = ( i/BITS_OF_BANK	  ) * BITS_OF_CL/NUM_MR; //main index * size of chunk
		uint32_t int_offset  = ( i&(BITS_OF_BANK-1) )/32;
		uint32_t bit_offset  = i & 31;

		//if ( ((i>>5)<num_bits) & ((index&7)==0) )
		{
		//	_mm_prefetch((char *)(&sample_addr[1*BITS_OF_ONE_CACHE_LINE+ i/2]), _MM_HINT_NTA);	//Stay at L1 offset*16
		}

		uint32_t data_src;
		v_sum    =  _mm256_set1_epi32(0);
		data_src =  sample_addr[main_offset + (BITS_OF_CL/32)*(0%BITS_OF_MR) + mr_offset*(0/BITS_OF_MR) + int_offset];
		v_data_1 =  _mm256_set1_epi32(data_src); 
		v_data_1 =  _mm256_srav_epi32(v_data_1, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data_1, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 7) );

	if (num_bits >=2)
	{
		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*(1%BITS_OF_MR) + mr_offset*(1/BITS_OF_MR) + int_offset];
		v_data_2 =  _mm256_set1_epi32(data_src); 
		v_data_2 =  _mm256_srav_epi32(v_data_2, v_offset); //shift it...
		v_data_2 =  _mm256_and_si256 (v_data_2, v_mask  ); //3  
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_2, 6) );
	 if (num_bits >=3)
	 {
		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*(2%BITS_OF_MR) + mr_offset*(2/BITS_OF_MR) + int_offset];
		v_data_3 =  _mm256_set1_epi32(data_src); 
		v_data_3 =  _mm256_srav_epi32(v_data_3, v_offset); //shift it...
		v_data_3 =  _mm256_and_si256 (v_data_3, v_mask  ); 
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_3, 5) );
	  if (num_bits >=4)
	  {
		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*(3%BITS_OF_MR) + mr_offset*(3/BITS_OF_MR) + int_offset];
		v_data_4 =  _mm256_set1_epi32(data_src); 
		v_data_4 =  _mm256_srav_epi32(v_data_4, v_offset); //shift it...
		v_data_4 =  _mm256_and_si256 (v_data_4, v_mask  ); 
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_4, 4) );
	   if (num_bits >=5)
	   {
		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*(4%BITS_OF_MR) + mr_offset*(4/BITS_OF_MR) + int_offset];
		v_data_5 =  _mm256_set1_epi32(data_src); 
		v_data_5 =  _mm256_srav_epi32(v_data_5, v_offset); //shift it...
		v_data_5 =  _mm256_and_si256 (v_data_5, v_mask  ); 
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_5, 3) );
	    if (num_bits >=6)
	    {
		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*(5%BITS_OF_MR) + mr_offset*(5/BITS_OF_MR) + int_offset];
		v_data_6 =  _mm256_set1_epi32(data_src); 
		v_data_6 =  _mm256_srav_epi32(v_data_6, v_offset); //shift it...
		v_data_6 =  _mm256_and_si256 (v_data_6, v_mask  ); 
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_6, 2) );
	    if (num_bits >=7)
	    {
		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*(6%BITS_OF_MR) + mr_offset*(6/BITS_OF_MR) + int_offset];
		v_data_7 =  _mm256_set1_epi32(data_src); 
		v_data_7 =  _mm256_srav_epi32(v_data_7, v_offset); //shift it...
		v_data_7 =  _mm256_and_si256 (v_data_7, v_mask  ); 
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_7, 1) );
	    if (num_bits >=8)
	    {
		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*(7%BITS_OF_MR) + mr_offset*(7/BITS_OF_MR) + int_offset];
		v_data_8 =  _mm256_set1_epi32(data_src); 
		v_data_8 =  _mm256_srav_epi32(v_data_8, v_offset); //shift it...
		v_data_8 =  _mm256_and_si256 (v_data_8, v_mask  ); 
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_8, 0) );
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
		_mm256_store_si256((__m256i *)(&vec_char[i]), v_result);
	}

}


void retrieve_from_mlweaving(FVector<uint16_t> & dest, uint32_t *src, uint32_t index, uint32_t num_bits) 
{	
	//dest's information
	uint32_t  numFeatures       = dest.size;
	uint16_t* vec_short         = dest.values;

	//aligned number of features. 
	uint32_t numFeaturesAlign   = ( (numFeatures+(BITS_OF_BANK-1))&(~(BITS_OF_BANK-1)) ); //round up to the nearest mulitple of BITS_OF_BANK

	//calculate the address of sample of the index: index. 
	uint32_t  chunk_offset      = (index/NUM_BANKS) * numFeaturesAlign*NUM_BANKS; //chunk index * chunk size
	uint32_t  sample_offset     = (index%NUM_BANKS) * (BITS_OF_BANK/32); //chunk index * chunk size
	uint32_t* sample_addr       = src + chunk_offset + sample_offset; //identify the address of the sampel (index)

    __m256i v_offset = _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0); 
    __m256i v_mask   = _mm256_set1_epi32(0x01010101);
    __m256i v_sum, v_data, v_data_1, v_data_2, v_data_3, v_data_4, v_data_5, v_data_6, v_data_7, v_data_0;

	for (size_t i = 0; i < numFeatures; i+= 32) 
	{  //prepare one number for each iteration.  
		uint32_t main_offset = ( i/BITS_OF_BANK	  ) * BITS_OF_CL; //main index * size of chunk
		uint32_t int_offset  = ( i&(BITS_OF_BANK-1) )/32;
		uint32_t bit_offset  = i & 31;

		uint32_t data_src;
		v_sum    =  _mm256_set1_epi32(0);


		///////////////////Upper 8 bits///////////////////////////////////
		data_src =  sample_addr[main_offset + (BITS_OF_CL/32)*0 + int_offset];
		v_data   =  _mm256_set1_epi32(data_src); 
		v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 7) );

		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*1 + int_offset];
		v_data   =  _mm256_set1_epi32(data_src); 
		v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 6) );

		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*2 + int_offset];
		v_data   =  _mm256_set1_epi32(data_src); 
		v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 5) );

		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*3 + int_offset];
		v_data   =  _mm256_set1_epi32(data_src); 
		v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 4) );

		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*4 + int_offset];
		v_data   =  _mm256_set1_epi32(data_src); 
		v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 3) );

		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*5 + int_offset];
		v_data   =  _mm256_set1_epi32(data_src); 
		v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 2) );

		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*6 + int_offset];
		v_data   =  _mm256_set1_epi32(data_src); 
		v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 1) );

		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*7 + int_offset];
		v_data   =  _mm256_set1_epi32(data_src); 
		v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 0) );

		__m256i	v_high = v_sum;


		///////////////////Lower 8 bits///////////////////////////////////
		v_sum    =  _mm256_set1_epi32(0);
		data_src =  sample_addr[main_offset + (BITS_OF_CL/32)*8 + int_offset];
		v_data   =  _mm256_set1_epi32(data_src); 
		v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 7) );

	if (num_bits >=10)
	{
		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*9 + int_offset];
		v_data   =  _mm256_set1_epi32(data_src); 
		v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 6) );
	 if (num_bits >=11)
	 {
		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*10 + int_offset];
		v_data   =  _mm256_set1_epi32(data_src); 
		v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 5) );
	  if (num_bits >=12)
	  {
		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*11 + int_offset];
		v_data   =  _mm256_set1_epi32(data_src); 
		v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 4) );
	   if (num_bits >=13)
	   {
		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*12 + int_offset];
		v_data   =  _mm256_set1_epi32(data_src); 
		v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 3) );
	    if (num_bits >=14)
	    {
		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*13 + int_offset];
		v_data   =  _mm256_set1_epi32(data_src); 
		v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 2) );
	    if (num_bits >=15)
	    {
		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*14 + int_offset];
		v_data   =  _mm256_set1_epi32(data_src); 
		v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 1) );
	    if (num_bits >=16)
	    {
		data_src = sample_addr[main_offset + (BITS_OF_CL/32)*15 + int_offset];
		v_data   =  _mm256_set1_epi32(data_src); 
		v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
		v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 0) );
	}}}}}} }
		__m256i	v_low = v_sum;

		//////////////Manipulation on the data/////////////////////
		__m256i v_shuffle_constant = _mm256_set_epi8 (15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0,
		 				                              15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0);
    	__m256i v_perm_constant = _mm256_set_epi32 (7, 3,  6, 2,   
                                               		5, 1,  4, 0); 
		__m256i v_data_2        = _mm256_shuffle_epi8(v_low, v_shuffle_constant);
		__m256i v_low_tmp       = _mm256_permutevar8x32_epi32(v_data_2, v_perm_constant);
		       v_data_2         = _mm256_shuffle_epi8(v_high, v_shuffle_constant);
		__m256i v_high_tmp      = _mm256_permutevar8x32_epi32(v_data_2, v_perm_constant);
		//Now, we have v_high (3, 1), v_low (2, 0)... objective: (3, 2), (1, 0)...
		__m256i v_data_low      = _mm256_unpacklo_epi8(v_low_tmp, v_high_tmp);				
		__m256i v_data_high     = _mm256_unpackhi_epi8(v_low_tmp, v_high_tmp);
		_mm256_storeu_si256((__m256i *)(&vec_short[i +  0]), v_data_low );				
		_mm256_storeu_si256((__m256i *)(&vec_short[i + 16]), v_data_high);	

		__m128i v_data_128_low  = _mm_loadu_si128((__m128i*)(&vec_short[i + 16]) );
		__m128i v_data_128_high = _mm_loadu_si128((__m128i*)(&vec_short[i + 8]) );
		_mm_storeu_si128((__m128i *)(&vec_short[i +  8]), v_data_128_low );				
		_mm_storeu_si128((__m128i *)(&vec_short[i + 16]), v_data_128_high);	
	}

}


void retrieve_from_mlweaving_8(FVector<uint8_t> & dest, uint32_t *src, uint32_t index, uint32_t num_bits) 
{	
	//dest's information
	uint32_t numFeatures        = dest.size;
	uint8_t* vec_char           = dest.values;

	//aligned number of features. 
	uint32_t numFeaturesAlign   = ( (numFeatures+(BITS_OF_BANK-1))&(~(BITS_OF_BANK-1)) ); //round up to the nearest mulitple of BITS_OF_BANK

	//calculate the address of sample of the index: index. 
	uint32_t  chunk_offset      = (index/NUM_BANKS) * numFeaturesAlign*NUM_BANKS; //chunk index * chunk size
	uint32_t  sample_offset     = (index%NUM_BANKS) * (BITS_OF_BANK/32); //chunk index * chunk size
	uint32_t* sample_addr       = src + chunk_offset + sample_offset; //identify the address of the sampel (index)

    __m256i v_offset = _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0); 
    __m256i v_mask   = _mm256_set1_epi32(0x01010101);
    __m256i v_data_0, v_data_1, v_data_2, v_data_3, v_data_4, v_data_5, v_data_6, v_data_7;
    __m256i v_sum;
    __m256i v_sum_0, v_sum_1, v_sum_2, v_sum_3, v_sum_4, v_sum_5, v_sum_6, v_sum_7;
	uint32_t data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7;

		__m256i v_shuffle_constant = _mm256_set_epi8 (15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0,
		 				                              15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0);
    	__m256i v_perm_constant    = _mm256_set_epi32 (7,  3,  6,  2,   
                                               		   5,  1,  4,  0); 

	for (size_t i = 0; i < numFeatures; i+= 32) 
	{  //prepare one number for each iteration.  
		uint32_t main_offset = ( i/BITS_OF_BANK	  ) * BITS_OF_CL; //main index * size of chunk
		uint32_t int_offset  = ( i&(BITS_OF_BANK-1) )/32;
		uint32_t bit_offset  = i & 31;

		//if ( ((i>>5)<num_bits) & ((index&7)==0) )
		{
		//	_mm_prefetch((char *)(&sample_addr[1*BITS_OF_ONE_CACHE_LINE+ i/2]), _MM_HINT_T2);	//Stay at L1 offset*16
		}

		v_sum    =  _mm256_set1_epi32(0);

		data_0   =  sample_addr[main_offset + (BITS_OF_CL/32)*0 + int_offset];
		v_data_0 =  _mm256_set1_epi32(data_0); 
		v_data_0 =  _mm256_srav_epi32(v_data_0, v_offset); //shift it...
		v_data_0 =  _mm256_and_si256 (v_data_0, v_mask  ); //3 
		v_sum_0  =  _mm256_slli_epi32(v_data_0, 7); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_0);

		data_1   =  sample_addr[main_offset + (BITS_OF_CL/32)*1 + int_offset];
		v_data_1 =  _mm256_set1_epi32(data_1); 
		v_data_1 =  _mm256_srav_epi32(v_data_1, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data_1, v_mask  ); //3 
		v_sum_1  =  _mm256_slli_epi32(v_data_1, 6); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_1);

		data_2   =  sample_addr[main_offset + (BITS_OF_CL/32)*2 + int_offset];
		v_data_2 =  _mm256_set1_epi32(data_2); 
		v_data_2 =  _mm256_srav_epi32(v_data_2, v_offset); //shift it...
		v_data_2 =  _mm256_and_si256 (v_data_2, v_mask  ); //3  
		v_sum_2  =  _mm256_slli_epi32(v_data_2, 5); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_2);

		data_3   =  sample_addr[main_offset + (BITS_OF_CL/32)*3 + int_offset];
		v_data_3 =  _mm256_set1_epi32(data_3); 
		v_data_3 =  _mm256_srav_epi32(v_data_3, v_offset); //shift it...
		v_data_3 =  _mm256_and_si256 (v_data_3, v_mask  ); 
		v_sum_3  =  _mm256_slli_epi32(v_data_3, 4); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_3);

		data_4   = sample_addr[main_offset + (BITS_OF_CL/32)*4 + int_offset];
		v_data_4 =  _mm256_set1_epi32(data_4); 
		v_data_4 =  _mm256_srav_epi32(v_data_4, v_offset); //shift it...
		v_data_4 =  _mm256_and_si256 (v_data_4, v_mask  ); 
		v_sum_4  =  _mm256_slli_epi32(v_data_4, 3); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_4);

		data_5   = sample_addr[main_offset + (BITS_OF_CL/32)*5 + int_offset];
		v_data_5 =  _mm256_set1_epi32(data_5); 
		v_data_5 =  _mm256_srav_epi32(v_data_5, v_offset); //shift it...
		v_data_5 =  _mm256_and_si256 (v_data_5, v_mask  ); 
		v_sum_5  =  _mm256_slli_epi32(v_data_5, 2); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_5);

		data_6   = sample_addr[main_offset + (BITS_OF_CL/32)*6 + int_offset];
		v_data_6 =  _mm256_set1_epi32(data_6); 
		v_data_6 =  _mm256_srav_epi32(v_data_6, v_offset); //shift it...
		v_data_6 =  _mm256_and_si256 (v_data_6, v_mask  ); 
		v_sum_6  =  _mm256_slli_epi32(v_data_6, 1); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_6);

		data_7   = sample_addr[main_offset + (BITS_OF_CL/32)*7 + int_offset];
		v_data_7 =  _mm256_set1_epi32(data_7); 
		v_data_7 =  _mm256_srav_epi32(v_data_7, v_offset); //shift it...
		v_data_7 =  _mm256_and_si256 (v_data_7, v_mask  ); 
		v_sum_7  =  _mm256_slli_epi32(v_data_7, 0); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_7);

		__m256i v_sum_tmp = _mm256_shuffle_epi8(v_sum, v_shuffle_constant);
		__m256i v_result  = _mm256_permutevar8x32_epi32(v_sum_tmp, v_perm_constant);
		_mm256_store_si256((__m256i *)(&vec_char[i]), v_result);
	}

}


void retrieve_from_mlweaving_7(FVector<uint8_t> & dest, uint32_t *src, uint32_t index, uint32_t num_bits) 
{	
	//dest's information
	uint32_t numFeatures        = dest.size;
	uint8_t* vec_char           = dest.values;

	//aligned number of features. 
	uint32_t numFeaturesAlign   = ( (numFeatures+(BITS_OF_BANK-1))&(~(BITS_OF_BANK-1)) ); //round up to the nearest mulitple of BITS_OF_BANK

	//calculate the address of sample of the index: index. 
	uint32_t  chunk_offset      = (index/NUM_BANKS) * numFeaturesAlign*NUM_BANKS; //chunk index * chunk size
	uint32_t  sample_offset     = (index%NUM_BANKS) * (BITS_OF_BANK/32); //chunk index * chunk size
	uint32_t* sample_addr       = src + chunk_offset + sample_offset; //identify the address of the sampel (index)

    __m256i v_offset = _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0); 
    __m256i v_mask   = _mm256_set1_epi32(0x01010101);
    __m256i v_data_0, v_data_1, v_data_2, v_data_3, v_data_4, v_data_5, v_data_6, v_data_7;
    __m256i v_sum;
    __m256i v_sum_0, v_sum_1, v_sum_2, v_sum_3, v_sum_4, v_sum_5, v_sum_6, v_sum_7;
	uint32_t data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7;

		__m256i v_shuffle_constant = _mm256_set_epi8 (15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0,
		 				                              15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0);
    	__m256i v_perm_constant    = _mm256_set_epi32 (7,  3,  6,  2,   
                                               		   5,  1,  4,  0); 

	for (size_t i = 0; i < numFeatures; i+= 32) 
	{  //prepare one number for each iteration.  
		uint32_t main_offset = ( i/BITS_OF_BANK	  ) * BITS_OF_CL; //main index * size of chunk
		uint32_t int_offset  = ( i&(BITS_OF_BANK-1) )/32;
		uint32_t bit_offset  = i & 31;

		//if ( ((i>>5)<num_bits) & ((index&7)==0) )
		{
		//	_mm_prefetch((char *)(&sample_addr[1*BITS_OF_ONE_CACHE_LINE+ i/2]), _MM_HINT_T2);	//Stay at L1 offset*16
		}

		v_sum    =  _mm256_set1_epi32(0);

		data_0   =  sample_addr[main_offset + (BITS_OF_CL/32)*0 + int_offset];
		v_data_0 =  _mm256_set1_epi32(data_0); 
		v_data_0 =  _mm256_srav_epi32(v_data_0, v_offset); //shift it...
		v_data_0 =  _mm256_and_si256 (v_data_0, v_mask  ); //3 
		v_sum_0  =  _mm256_slli_epi32(v_data_0, 7); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_0);

		data_1   =  sample_addr[main_offset + (BITS_OF_CL/32)*1 + int_offset];
		v_data_1 =  _mm256_set1_epi32(data_1); 
		v_data_1 =  _mm256_srav_epi32(v_data_1, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data_1, v_mask  ); //3 
		v_sum_1  =  _mm256_slli_epi32(v_data_1, 6); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_1);

		data_2   =  sample_addr[main_offset + (BITS_OF_CL/32)*2 + int_offset];
		v_data_2 =  _mm256_set1_epi32(data_2); 
		v_data_2 =  _mm256_srav_epi32(v_data_2, v_offset); //shift it...
		v_data_2 =  _mm256_and_si256 (v_data_2, v_mask  ); //3  
		v_sum_2  =  _mm256_slli_epi32(v_data_2, 5); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_2);

		data_3   =  sample_addr[main_offset + (BITS_OF_CL/32)*3 + int_offset];
		v_data_3 =  _mm256_set1_epi32(data_3); 
		v_data_3 =  _mm256_srav_epi32(v_data_3, v_offset); //shift it...
		v_data_3 =  _mm256_and_si256 (v_data_3, v_mask  ); 
		v_sum_3  =  _mm256_slli_epi32(v_data_3, 4); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_3);

		data_4   = sample_addr[main_offset + (BITS_OF_CL/32)*4 + int_offset];
		v_data_4 =  _mm256_set1_epi32(data_4); 
		v_data_4 =  _mm256_srav_epi32(v_data_4, v_offset); //shift it...
		v_data_4 =  _mm256_and_si256 (v_data_4, v_mask  ); 
		v_sum_4  =  _mm256_slli_epi32(v_data_4, 3); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_4);

		data_5   = sample_addr[main_offset + (BITS_OF_CL/32)*5 + int_offset];
		v_data_5 =  _mm256_set1_epi32(data_5); 
		v_data_5 =  _mm256_srav_epi32(v_data_5, v_offset); //shift it...
		v_data_5 =  _mm256_and_si256 (v_data_5, v_mask  ); 
		v_sum_5  =  _mm256_slli_epi32(v_data_5, 2); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_5);

		data_6   = sample_addr[main_offset + (BITS_OF_CL/32)*6 + int_offset];
		v_data_6 =  _mm256_set1_epi32(data_6); 
		v_data_6 =  _mm256_srav_epi32(v_data_6, v_offset); //shift it...
		v_data_6 =  _mm256_and_si256 (v_data_6, v_mask  ); 
		v_sum_6  =  _mm256_slli_epi32(v_data_6, 1); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_6);

		__m256i v_sum_tmp = _mm256_shuffle_epi8(v_sum, v_shuffle_constant);
		__m256i v_result  = _mm256_permutevar8x32_epi32(v_sum_tmp, v_perm_constant);
		_mm256_store_si256((__m256i *)(&vec_char[i]), v_result);
	}

}


void retrieve_from_mlweaving_6(FVector<uint8_t> & dest, uint32_t *src, uint32_t index, uint32_t num_bits) 
{	
	//dest's information
	uint32_t numFeatures        = dest.size;
	uint8_t* vec_char           = dest.values;

	//aligned number of features. 
	uint32_t numFeaturesAlign   = ( (numFeatures+(BITS_OF_BANK-1))&(~(BITS_OF_BANK-1)) ); //round up to the nearest mulitple of BITS_OF_BANK

	//calculate the address of sample of the index: index. 
	uint32_t  chunk_offset      = (index/NUM_BANKS) * numFeaturesAlign*NUM_BANKS; //chunk index * chunk size
	uint32_t  sample_offset     = (index%NUM_BANKS) * (BITS_OF_BANK/32); //chunk index * chunk size
	uint32_t* sample_addr       = src + chunk_offset + sample_offset; //identify the address of the sampel (index)

    __m256i v_offset = _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0); 
    __m256i v_mask   = _mm256_set1_epi32(0x01010101);
    __m256i v_data_0, v_data_1, v_data_2, v_data_3, v_data_4, v_data_5, v_data_6, v_data_7;
    __m256i v_sum;
    __m256i v_sum_0, v_sum_1, v_sum_2, v_sum_3, v_sum_4, v_sum_5, v_sum_6, v_sum_7;
	uint32_t data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7;

		__m256i v_shuffle_constant = _mm256_set_epi8 (15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0,
		 				                              15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0);
    	__m256i v_perm_constant    = _mm256_set_epi32 (7,  3,  6,  2,   
                                               		   5,  1,  4,  0); 

	for (size_t i = 0; i < numFeatures; i+= 32) 
	{  //prepare one number for each iteration.  
		uint32_t main_offset = ( i/BITS_OF_BANK	  ) * BITS_OF_CL; //main index * size of chunk
		uint32_t int_offset  = ( i&(BITS_OF_BANK-1) )/32;
		uint32_t bit_offset  = i & 31;

		//if ( ((i>>5)<num_bits) & ((index&7)==0) )
		{
		//	_mm_prefetch((char *)(&sample_addr[1*BITS_OF_ONE_CACHE_LINE+ i/2]), _MM_HINT_T2);	//Stay at L1 offset*16
		}

		v_sum    =  _mm256_set1_epi32(0);

		data_0   =  sample_addr[main_offset + (BITS_OF_CL/32)*0 + int_offset];
		v_data_0 =  _mm256_set1_epi32(data_0); 
		v_data_0 =  _mm256_srav_epi32(v_data_0, v_offset); //shift it...
		v_data_0 =  _mm256_and_si256 (v_data_0, v_mask  ); //3 
		v_sum_0  =  _mm256_slli_epi32(v_data_0, 7); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_0);

		data_1   =  sample_addr[main_offset + (BITS_OF_CL/32)*1 + int_offset];
		v_data_1 =  _mm256_set1_epi32(data_1); 
		v_data_1 =  _mm256_srav_epi32(v_data_1, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data_1, v_mask  ); //3 
		v_sum_1  =  _mm256_slli_epi32(v_data_1, 6); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_1);

		data_2   =  sample_addr[main_offset + (BITS_OF_CL/32)*2 + int_offset];
		v_data_2 =  _mm256_set1_epi32(data_2); 
		v_data_2 =  _mm256_srav_epi32(v_data_2, v_offset); //shift it...
		v_data_2 =  _mm256_and_si256 (v_data_2, v_mask  ); //3  
		v_sum_2  =  _mm256_slli_epi32(v_data_2, 5); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_2);

		data_3   =  sample_addr[main_offset + (BITS_OF_CL/32)*3 + int_offset];
		v_data_3 =  _mm256_set1_epi32(data_3); 
		v_data_3 =  _mm256_srav_epi32(v_data_3, v_offset); //shift it...
		v_data_3 =  _mm256_and_si256 (v_data_3, v_mask  ); 
		v_sum_3  =  _mm256_slli_epi32(v_data_3, 4); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_3);

		data_4   = sample_addr[main_offset + (BITS_OF_CL/32)*4 + int_offset];
		v_data_4 =  _mm256_set1_epi32(data_4); 
		v_data_4 =  _mm256_srav_epi32(v_data_4, v_offset); //shift it...
		v_data_4 =  _mm256_and_si256 (v_data_4, v_mask  ); 
		v_sum_4  =  _mm256_slli_epi32(v_data_4, 3); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_4);

		data_5   = sample_addr[main_offset + (BITS_OF_CL/32)*5 + int_offset];
		v_data_5 =  _mm256_set1_epi32(data_5); 
		v_data_5 =  _mm256_srav_epi32(v_data_5, v_offset); //shift it...
		v_data_5 =  _mm256_and_si256 (v_data_5, v_mask  ); 
		v_sum_5  =  _mm256_slli_epi32(v_data_5, 2); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_5);

		__m256i v_sum_tmp = _mm256_shuffle_epi8(v_sum, v_shuffle_constant);
		__m256i v_result  = _mm256_permutevar8x32_epi32(v_sum_tmp, v_perm_constant);
		_mm256_store_si256((__m256i *)(&vec_char[i]), v_result);
	}

}


void retrieve_from_mlweaving_5(FVector<uint8_t> & dest, uint32_t *src, uint32_t index, uint32_t num_bits) 
{	
	//dest's information
	uint32_t numFeatures        = dest.size;
	uint8_t* vec_char           = dest.values;

	//aligned number of features. 
	uint32_t numFeaturesAlign   = ( (numFeatures+(BITS_OF_BANK-1))&(~(BITS_OF_BANK-1)) ); //round up to the nearest mulitple of BITS_OF_BANK

	//calculate the address of sample of the index: index. 
	uint32_t  chunk_offset      = (index/NUM_BANKS) * numFeaturesAlign*NUM_BANKS; //chunk index * chunk size
	uint32_t  sample_offset     = (index%NUM_BANKS) * (BITS_OF_BANK/32); //chunk index * chunk size
	uint32_t* sample_addr       = src + chunk_offset + sample_offset; //identify the address of the sampel (index)

    __m256i v_offset = _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0); 
    __m256i v_mask   = _mm256_set1_epi32(0x01010101);
    __m256i v_data_0, v_data_1, v_data_2, v_data_3, v_data_4, v_data_5, v_data_6, v_data_7;
    __m256i v_sum;
    __m256i v_sum_0, v_sum_1, v_sum_2, v_sum_3, v_sum_4, v_sum_5, v_sum_6, v_sum_7;
	uint32_t data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7;

		__m256i v_shuffle_constant = _mm256_set_epi8 (15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0,
		 				                              15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0);
    	__m256i v_perm_constant    = _mm256_set_epi32 (7,  3,  6,  2,   
                                               		   5,  1,  4,  0); 

	for (size_t i = 0; i < numFeatures; i+= 32) 
	{  //prepare one number for each iteration.  
		uint32_t main_offset = ( i/BITS_OF_BANK	  ) * BITS_OF_CL; //main index * size of chunk
		uint32_t int_offset  = ( i&(BITS_OF_BANK-1) )/32;
		uint32_t bit_offset  = i & 31;

		//if ( ((i>>5)<num_bits) & ((index&7)==0) )
		{
		//	_mm_prefetch((char *)(&sample_addr[1*BITS_OF_ONE_CACHE_LINE+ i/2]), _MM_HINT_T2);	//Stay at L1 offset*16
		}

		v_sum    =  _mm256_set1_epi32(0);

		data_0   =  sample_addr[main_offset + (BITS_OF_CL/32)*0 + int_offset];
		v_data_0 =  _mm256_set1_epi32(data_0); 
		v_data_0 =  _mm256_srav_epi32(v_data_0, v_offset); //shift it...
		v_data_0 =  _mm256_and_si256 (v_data_0, v_mask  ); //3 
		v_sum_0  =  _mm256_slli_epi32(v_data_0, 7); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_0);

		data_1   =  sample_addr[main_offset + (BITS_OF_CL/32)*1 + int_offset];
		v_data_1 =  _mm256_set1_epi32(data_1); 
		v_data_1 =  _mm256_srav_epi32(v_data_1, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data_1, v_mask  ); //3 
		v_sum_1  =  _mm256_slli_epi32(v_data_1, 6); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_1);

		data_2   =  sample_addr[main_offset + (BITS_OF_CL/32)*2 + int_offset];
		v_data_2 =  _mm256_set1_epi32(data_2); 
		v_data_2 =  _mm256_srav_epi32(v_data_2, v_offset); //shift it...
		v_data_2 =  _mm256_and_si256 (v_data_2, v_mask  ); //3  
		v_sum_2  =  _mm256_slli_epi32(v_data_2, 5); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_2);

		data_3   =  sample_addr[main_offset + (BITS_OF_CL/32)*3 + int_offset];
		v_data_3 =  _mm256_set1_epi32(data_3); 
		v_data_3 =  _mm256_srav_epi32(v_data_3, v_offset); //shift it...
		v_data_3 =  _mm256_and_si256 (v_data_3, v_mask  ); 
		v_sum_3  =  _mm256_slli_epi32(v_data_3, 4); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_3);

		data_4   = sample_addr[main_offset + (BITS_OF_CL/32)*4 + int_offset];
		v_data_4 =  _mm256_set1_epi32(data_4); 
		v_data_4 =  _mm256_srav_epi32(v_data_4, v_offset); //shift it...
		v_data_4 =  _mm256_and_si256 (v_data_4, v_mask  ); 
		v_sum_4  =  _mm256_slli_epi32(v_data_4, 3); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_4);

		__m256i v_sum_tmp = _mm256_shuffle_epi8(v_sum, v_shuffle_constant);
		__m256i v_result  = _mm256_permutevar8x32_epi32(v_sum_tmp, v_perm_constant);
		_mm256_store_si256((__m256i *)(&vec_char[i]), v_result);
	}

}


void retrieve_from_mlweaving_4(FVector<uint8_t> & dest, uint32_t *src, uint32_t index, uint32_t num_bits) 
{	
	//dest's information
	uint32_t numFeatures        = dest.size;
	uint8_t* vec_char           = dest.values;

	//aligned number of features. 
	uint32_t numFeaturesAlign   = ( (numFeatures+(BITS_OF_BANK-1))&(~(BITS_OF_BANK-1)) ); //round up to the nearest mulitple of BITS_OF_BANK

	//calculate the address of sample of the index: index. 
	uint32_t  chunk_offset      = (index/NUM_BANKS) * numFeaturesAlign*NUM_BANKS; //chunk index * chunk size
	uint32_t  sample_offset     = (index%NUM_BANKS) * (BITS_OF_BANK/32); //chunk index * chunk size
	uint32_t* sample_addr       = src + chunk_offset + sample_offset; //identify the address of the sampel (index)

    __m256i v_offset = _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0); 
    __m256i v_mask   = _mm256_set1_epi32(0x01010101);
    __m256i v_data_0, v_data_1, v_data_2, v_data_3, v_data_4, v_data_5, v_data_6, v_data_7;
    __m256i v_sum;
    __m256i v_sum_0, v_sum_1, v_sum_2, v_sum_3, v_sum_4, v_sum_5, v_sum_6, v_sum_7;
	uint32_t data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7;

		__m256i v_shuffle_constant = _mm256_set_epi8 (15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0,
		 				                              15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0);
    	__m256i v_perm_constant    = _mm256_set_epi32 (7,  3,  6,  2,   
                                               		   5,  1,  4,  0); 

	for (size_t i = 0; i < numFeatures; i+= 32) 
	{  //prepare one number for each iteration.  
		uint32_t main_offset = ( i/BITS_OF_BANK	  ) * BITS_OF_CL; //main index * size of chunk
		uint32_t int_offset  = ( i&(BITS_OF_BANK-1) )/32;
		uint32_t bit_offset  = i & 31;

		//if ( ((i>>5)<num_bits) & ((index&7)==0) )
		{
		//	_mm_prefetch((char *)(&sample_addr[1*BITS_OF_ONE_CACHE_LINE+ i/2]), _MM_HINT_T2);	//Stay at L1 offset*16
		}

		v_sum    =  _mm256_set1_epi32(0);

		data_0   =  sample_addr[main_offset + (BITS_OF_CL/32)*0 + int_offset];
		v_data_0 =  _mm256_set1_epi32(data_0); 
		v_data_0 =  _mm256_srav_epi32(v_data_0, v_offset); //shift it...
		v_data_0 =  _mm256_and_si256 (v_data_0, v_mask  ); //3 
		v_sum_0  =  _mm256_slli_epi32(v_data_0, 7); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_0);

		data_1   =  sample_addr[main_offset + (BITS_OF_CL/32)*1 + int_offset];
		v_data_1 =  _mm256_set1_epi32(data_1); 
		v_data_1 =  _mm256_srav_epi32(v_data_1, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data_1, v_mask  ); //3 
		v_sum_1  =  _mm256_slli_epi32(v_data_1, 6); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_1);

		data_2   =  sample_addr[main_offset + (BITS_OF_CL/32)*2 + int_offset];
		v_data_2 =  _mm256_set1_epi32(data_2); 
		v_data_2 =  _mm256_srav_epi32(v_data_2, v_offset); //shift it...
		v_data_2 =  _mm256_and_si256 (v_data_2, v_mask  ); //3  
		v_sum_2  =  _mm256_slli_epi32(v_data_2, 5); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_2);

		data_3   =  sample_addr[main_offset + (BITS_OF_CL/32)*3 + int_offset];
		v_data_3 =  _mm256_set1_epi32(data_3); 
		v_data_3 =  _mm256_srav_epi32(v_data_3, v_offset); //shift it...
		v_data_3 =  _mm256_and_si256 (v_data_3, v_mask  ); 
		v_sum_3  =  _mm256_slli_epi32(v_data_3, 4); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_3);

		__m256i v_sum_tmp = _mm256_shuffle_epi8(v_sum, v_shuffle_constant);
		__m256i v_result  = _mm256_permutevar8x32_epi32(v_sum_tmp, v_perm_constant);
		_mm256_store_si256((__m256i *)(&vec_char[i]), v_result);
	}

}



void retrieve_from_mlweaving_3(FVector<uint8_t> & dest, uint32_t *src, uint32_t index, uint32_t num_bits) 
{	
	//dest's information
	uint32_t numFeatures        = dest.size;
	uint8_t* vec_char           = dest.values;

	//aligned number of features. 
	uint32_t numFeaturesAlign   = ( (numFeatures+(BITS_OF_BANK-1))&(~(BITS_OF_BANK-1)) ); //round up to the nearest mulitple of BITS_OF_BANK

	//calculate the address of sample of the index: index. 
	uint32_t  chunk_offset      = (index/NUM_BANKS) * numFeaturesAlign*NUM_BANKS; //chunk index * chunk size
	uint32_t  sample_offset     = (index%NUM_BANKS) * (BITS_OF_BANK/32); //chunk index * chunk size
	uint32_t* sample_addr       = src + chunk_offset + sample_offset; //identify the address of the sampel (index)

    __m256i v_offset = _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0); 
    __m256i v_mask   = _mm256_set1_epi32(0x01010101);
    __m256i v_data_0, v_data_1, v_data_2, v_data_3, v_data_4, v_data_5, v_data_6, v_data_7;
    __m256i v_sum;
    __m256i v_sum_0, v_sum_1, v_sum_2, v_sum_3, v_sum_4, v_sum_5, v_sum_6, v_sum_7;
	uint32_t data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7;

		__m256i v_shuffle_constant = _mm256_set_epi8 (15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0,
		 				                              15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0);
    	__m256i v_perm_constant    = _mm256_set_epi32 (7,  3,  6,  2,   
                                               		   5,  1,  4,  0); 

	for (size_t i = 0; i < numFeatures; i+= 32) 
	{  //prepare one number for each iteration.  
		uint32_t main_offset = ( i/BITS_OF_BANK	  ) * BITS_OF_CL; //main index * size of chunk
		uint32_t int_offset  = ( i&(BITS_OF_BANK-1) )/32;
		uint32_t bit_offset  = i & 31;

		//if ( ((i>>5)<num_bits) & ((index&7)==0) )
		{
		//	_mm_prefetch((char *)(&sample_addr[1*BITS_OF_ONE_CACHE_LINE+ i/2]), _MM_HINT_T2);	//Stay at L1 offset*16
		}

		v_sum    =  _mm256_set1_epi32(0);

		data_0   =  sample_addr[main_offset + (BITS_OF_CL/32)*0 + int_offset];
		v_data_0 =  _mm256_set1_epi32(data_0); 
		v_data_0 =  _mm256_srav_epi32(v_data_0, v_offset); //shift it...
		v_data_0 =  _mm256_and_si256 (v_data_0, v_mask  ); //3 
		v_sum_0  =  _mm256_slli_epi32(v_data_0, 7); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_0);

		data_1   =  sample_addr[main_offset + (BITS_OF_CL/32)*1 + int_offset];
		v_data_1 =  _mm256_set1_epi32(data_1); 
		v_data_1 =  _mm256_srav_epi32(v_data_1, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data_1, v_mask  ); //3 
		v_sum_1  =  _mm256_slli_epi32(v_data_1, 6); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_1);

		data_2   =  sample_addr[main_offset + (BITS_OF_CL/32)*2 + int_offset];
		v_data_2 =  _mm256_set1_epi32(data_2); 
		v_data_2 =  _mm256_srav_epi32(v_data_2, v_offset); //shift it...
		v_data_2 =  _mm256_and_si256 (v_data_2, v_mask  ); //3  
		v_sum_2  =  _mm256_slli_epi32(v_data_2, 5); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_2);

		__m256i v_sum_tmp = _mm256_shuffle_epi8(v_sum, v_shuffle_constant);
		__m256i v_result  = _mm256_permutevar8x32_epi32(v_sum_tmp, v_perm_constant);
		_mm256_store_si256((__m256i *)(&vec_char[i]), v_result);
	}

}


void retrieve_from_mlweaving_2(FVector<uint8_t> & dest, uint32_t *src, uint32_t index, uint32_t num_bits) 
{	
	//dest's information
	uint32_t numFeatures        = dest.size;
	uint8_t* vec_char           = dest.values;

	//aligned number of features. 
	uint32_t numFeaturesAlign   = ( (numFeatures+(BITS_OF_BANK-1))&(~(BITS_OF_BANK-1)) ); //round up to the nearest mulitple of BITS_OF_BANK

	//calculate the address of sample of the index: index. 
	uint32_t  chunk_offset      = (index/NUM_BANKS) * numFeaturesAlign*NUM_BANKS; //chunk index * chunk size
	uint32_t  sample_offset     = (index%NUM_BANKS) * (BITS_OF_BANK/32); //chunk index * chunk size
	uint32_t* sample_addr       = src + chunk_offset + sample_offset; //identify the address of the sampel (index)

    __m256i v_offset = _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0); 
    __m256i v_mask   = _mm256_set1_epi32(0x01010101);
    __m256i v_data_0, v_data_1, v_data_2, v_data_3, v_data_4, v_data_5, v_data_6, v_data_7;
    __m256i v_sum;
    __m256i v_sum_0, v_sum_1, v_sum_2, v_sum_3, v_sum_4, v_sum_5, v_sum_6, v_sum_7;
	uint32_t data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7;

		__m256i v_shuffle_constant = _mm256_set_epi8 (15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0,
		 				                              15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0);
    	__m256i v_perm_constant    = _mm256_set_epi32 (7,  3,  6,  2,   
                                               		   5,  1,  4,  0); 

	for (size_t i = 0; i < numFeatures; i+= 32) 
	{  //prepare one number for each iteration.  
		uint32_t main_offset = ( i/BITS_OF_BANK	  ) * BITS_OF_CL; //main index * size of chunk
		uint32_t int_offset  = ( i&(BITS_OF_BANK-1) )/32;
		uint32_t bit_offset  = i & 31;

		//if ( ((i>>5)<num_bits) & ((index&7)==0) )
		{
		//	_mm_prefetch((char *)(&sample_addr[1*BITS_OF_ONE_CACHE_LINE+ i/2]), _MM_HINT_T2);	//Stay at L1 offset*16
		}

		v_sum    =  _mm256_set1_epi32(0);

		data_0   =  sample_addr[main_offset + (BITS_OF_CL/32)*0 + int_offset];
		v_data_0 =  _mm256_set1_epi32(data_0); 
		v_data_0 =  _mm256_srav_epi32(v_data_0, v_offset); //shift it...
		v_data_0 =  _mm256_and_si256 (v_data_0, v_mask  ); //3 
		v_sum_0  =  _mm256_slli_epi32(v_data_0, 7); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_0);

		data_1   =  sample_addr[main_offset + (BITS_OF_CL/32)*1 + int_offset];
		v_data_1 =  _mm256_set1_epi32(data_1); 
		v_data_1 =  _mm256_srav_epi32(v_data_1, v_offset); //shift it...
		v_data_1 =  _mm256_and_si256 (v_data_1, v_mask  ); //3 
		v_sum_1  =  _mm256_slli_epi32(v_data_1, 6); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_1);

		__m256i v_sum_tmp = _mm256_shuffle_epi8(v_sum, v_shuffle_constant);
		__m256i v_result  = _mm256_permutevar8x32_epi32(v_sum_tmp, v_perm_constant);
		_mm256_store_si256((__m256i *)(&vec_char[i]), v_result);
	}

}

void retrieve_from_mlweaving_1(FVector<uint8_t> & dest, uint32_t *src, uint32_t index, uint32_t num_bits) 
{	
	//dest's information
	uint32_t numFeatures        = dest.size;
	uint8_t* vec_char           = dest.values;

	//aligned number of features. 
	uint32_t numFeaturesAlign   = ( (numFeatures+(BITS_OF_BANK-1))&(~(BITS_OF_BANK-1)) ); //round up to the nearest mulitple of BITS_OF_BANK

	//calculate the address of sample of the index: index. 
	uint32_t  chunk_offset      = (index/NUM_BANKS) * numFeaturesAlign*NUM_BANKS; //chunk index * chunk size
	uint32_t  sample_offset     = (index%NUM_BANKS) * (BITS_OF_BANK/32); //chunk index * chunk size
	uint32_t* sample_addr       = src + chunk_offset + sample_offset; //identify the address of the sampel (index)

    __m256i v_offset = _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0); 
    __m256i v_mask   = _mm256_set1_epi32(0x01010101);
    __m256i v_data_0, v_data_1, v_data_2, v_data_3, v_data_4, v_data_5, v_data_6, v_data_7;
    __m256i v_sum;
    __m256i v_sum_0, v_sum_1, v_sum_2, v_sum_3, v_sum_4, v_sum_5, v_sum_6, v_sum_7;
	uint32_t data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7;

		__m256i v_shuffle_constant = _mm256_set_epi8 (15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0,
		 				                              15, 11,  7,  3, 
		 				                              14, 10,  6,  2, 
		 				                              13,  9,  5,  1, 
		 				                              12,  8,  4,  0);
    	__m256i v_perm_constant    = _mm256_set_epi32 (7,  3,  6,  2,   
                                               		   5,  1,  4,  0); 

	for (size_t i = 0; i < numFeatures; i+= 32) 
	{  //prepare one number for each iteration.  
		uint32_t main_offset = ( i/BITS_OF_BANK	  ) * BITS_OF_CL; //main index * size of chunk
		uint32_t int_offset  = ( i&(BITS_OF_BANK-1) )/32;
		uint32_t bit_offset  = i & 31;

		//if ( ((i>>5)<num_bits) & ((index&7)==0) )
		{
		//	_mm_prefetch((char *)(&sample_addr[1*BITS_OF_ONE_CACHE_LINE+ i/2]), _MM_HINT_T2);	//Stay at L1 offset*16
		}

		v_sum    =  _mm256_set1_epi32(0);

		data_0   =  sample_addr[main_offset + (BITS_OF_CL/32)*0 + int_offset];
		v_data_0 =  _mm256_set1_epi32(data_0); 
		v_data_0 =  _mm256_srav_epi32(v_data_0, v_offset); //shift it...
		v_data_0 =  _mm256_and_si256 (v_data_0, v_mask  ); //3 
		v_sum_0  =  _mm256_slli_epi32(v_data_0, 7); //_mm256_or_si256  (v_sum, );
		v_sum    =  _mm256_or_si256(v_sum, v_sum_0);

		__m256i v_sum_tmp = _mm256_shuffle_epi8(v_sum, v_shuffle_constant);
		__m256i v_result  = _mm256_permutevar8x32_epi32(v_sum_tmp, v_perm_constant);
		_mm256_store_si256((__m256i *)(&vec_char[i]), v_result);
	}

}

} // namespace vector
} // namespace hazy
#endif
