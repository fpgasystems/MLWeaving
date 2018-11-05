// Copyright 2012 Chris Re, Victor Bittorf
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The Hazy Project, http://research.cs.wisc.edu/hazy/
// Author : Victor Bittorf (bittorf [at] cs.wisc.edu)

#ifndef HAZY_VECTOR_DOT_MLWEAVING_INL_H
#define HAZY_VECTOR_DOT_MLWEAVING_INL_H

#include <cmath>
#include <immintrin.h>


// See hazy/vector/dot.h for documentation

namespace hazy {
namespace vector {

#define BITS_OF_CL      512
#define NUM_BANKS       8
#define BITS_OF_BANK    (BITS_OF_CL/NUM_BANKS)


//template <typename float_u, typename float_v>
float Dot_mlweaving( FVector<unsigned char> &vVector, FVector<float> const &uVector, 
                            uint32_t *src, uint32_t index, uint32_t num_bits, uint32_t num_samples) 
{
    uint32_t numFeatures        = vVector.size;
    uint8_t* vec_char           = vVector.values;

    //aligned number of features. 
    uint32_t numFeaturesAlign   = ( (numFeatures+(BITS_OF_BANK-1))&(~(BITS_OF_BANK-1)) ); //round up to the nearest mulitple of BITS_OF_BANK

    //calculate the address of sample of the index: index. 
    uint32_t  chunk_offset      = (index/NUM_BANKS) * numFeaturesAlign*NUM_BANKS; //chunk index * chunk size
    uint32_t  sample_offset     = (index%NUM_BANKS) * (BITS_OF_BANK/32); //chunk index * chunk size
    uint32_t* sample_addr       = src + chunk_offset + sample_offset; //identify the address of the sampel (index)

    __m256i v_offset = _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0); 
    __m256i v_mask   = _mm256_set1_epi32(0x01010101);
    __m256i v_sum, v_data, v_data_1, v_data_2, v_data_3, v_data_4, v_data_5, v_data_6, v_data_7, v_data_8, v_data_0;

        const uint64_t n0 = uVector.size;
        const uint64_t n1 = n0 & 0xFFFFFFFFFFFFFFE0UL;
        const float * u            = uVector.values;
        const unsigned char * v   = vVector.values;
        __m256 acc1 = _mm256_setzero_ps(); 
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
		
		const __m256i and_const  = _mm256_set1_epi32(0xffff);
        for (uint64_t i = 0; i < n1; i += 32)
        {// 
            uint32_t main_offset = ( i/BITS_OF_BANK   ) * BITS_OF_CL; //main index * size of chunk
            uint32_t int_offset  = ( i&(BITS_OF_BANK-1) )/32;
            uint32_t bit_offset  = i & 31;
    

            uint32_t data_src;
            v_sum    =  _mm256_set1_epi32(0);
            data_src =  sample_addr[main_offset + (BITS_OF_CL/32)*0 + int_offset];
            v_data_1 =  _mm256_set1_epi32(data_src); 
            v_data_1 =  _mm256_srav_epi32(v_data_1, v_offset); //shift it...
            v_data_1 =  _mm256_and_si256 (v_data_1, v_mask  ); //3  v_data
            v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 7) );
    
        if (num_bits >=2)
        {
            data_src = sample_addr[main_offset + (BITS_OF_CL/32)*1 + int_offset];
            v_data_2 =  _mm256_set1_epi32(data_src); 
            v_data_2 =  _mm256_srav_epi32(v_data_2, v_offset); //shift it...
            v_data_2 =  _mm256_and_si256 (v_data_2, v_mask  ); //3  
            v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_2, 6) );
         if (num_bits >=3)
         {
            data_src = sample_addr[main_offset + (BITS_OF_CL/32)*2 + int_offset];
            v_data_3 =  _mm256_set1_epi32(data_src); 
            v_data_3 =  _mm256_srav_epi32(v_data_3, v_offset); //shift it...
            v_data_3 =  _mm256_and_si256 (v_data_3, v_mask  ); 
            v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_3, 5) );
          if (num_bits >=4)
          {
            data_src = sample_addr[main_offset + (BITS_OF_CL/32)*3 + int_offset];
            v_data_4 =  _mm256_set1_epi32(data_src); 
            v_data_4 =  _mm256_srav_epi32(v_data_4, v_offset); //shift it...
            v_data_4 =  _mm256_and_si256 (v_data_4, v_mask  ); 
            v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_4, 4) );
           if (num_bits >=5)
           {
            data_src = sample_addr[main_offset + (BITS_OF_CL/32)*4 + int_offset];
            v_data_5 =  _mm256_set1_epi32(data_src); 
            v_data_5 =  _mm256_srav_epi32(v_data_5, v_offset); //shift it...
            v_data_5 =  _mm256_and_si256 (v_data_5, v_mask  ); 
            v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_5, 3) );
            if (num_bits >=6)
            {
            data_src = sample_addr[main_offset + (BITS_OF_CL/32)*5 + int_offset];
            v_data_6 =  _mm256_set1_epi32(data_src); 
            v_data_6 =  _mm256_srav_epi32(v_data_6, v_offset); //shift it...
            v_data_6 =  _mm256_and_si256 (v_data_6, v_mask  ); 
            v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_6, 2) );
            if (num_bits >=7)
            {
            data_src = sample_addr[main_offset + (BITS_OF_CL/32)*6 + int_offset];
            v_data_7 =  _mm256_set1_epi32(data_src); 
            v_data_7 =  _mm256_srav_epi32(v_data_7, v_offset); //shift it...
            v_data_7 =  _mm256_and_si256 (v_data_7, v_mask  ); 
            v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_7, 1) );
            if (num_bits >=8)
            {
            data_src = sample_addr[main_offset + (BITS_OF_CL/32)*7 + int_offset];
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
            //seperate the v_result into two __m128i variables: v1_128, v3_128

                __m128i v1_128 = _mm256_extracti128_si256(v_result, 0);
                __m128i v3_128 = _mm256_extracti128_si256(v_result, 1);
               //__m128i v1_128 = _mm_loadu_si128((__m128i const* )(v + i + 0));
               //__m128i v3_128 = _mm_loadu_si128((__m128i const* )(v + i + 16));
			   __m256i v1_abs = _mm256_cvtepu8_epi32(v1_128); //lower part
			   __m256i v2_abs = _mm256_cvtepu8_epi32(_mm_bsrli_si128(v1_128,8));//upper part
			   __m256i v3_abs = _mm256_cvtepu8_epi32(v3_128); //lower part
			   __m256i v4_abs = _mm256_cvtepu8_epi32(_mm_bsrli_si128(v3_128,8));//upper part
                    __m256 v1 = _mm256_cvtepi32_ps(v1_abs);
                    __m256 v2 = _mm256_cvtepi32_ps(v2_abs);
                    __m256 v3 = _mm256_cvtepi32_ps(v3_abs);
                    __m256 v4 = _mm256_cvtepi32_ps(v4_abs);
            const __m256 u1 = _mm256_loadu_ps(u + i + 0);
            const __m256 u2 = _mm256_loadu_ps(u + i + 8);
            const __m256 u3 = _mm256_loadu_ps(u + i + 16);
            const __m256 u4 = _mm256_loadu_ps(u + i + 24);
            acc1 = _mm256_fmadd_ps(v1, u1, acc1);
            acc2 = _mm256_fmadd_ps(v2, u2, acc2);
            acc3 = _mm256_fmadd_ps(v3, u3, acc3);
            acc4 = _mm256_fmadd_ps(v4, u4, acc4);
        }
        // add the accumulators
        const __m256 tmp1 = _mm256_add_ps(acc1, acc2);
        const __m256 tmp2 = _mm256_add_ps(acc3, acc4);
        const __m256 tmp3 = _mm256_add_ps(tmp1, tmp2);
        // perform reduction
        const __m128 left  = _mm256_extractf128_ps(tmp3, 1);
        const __m128 right = _mm256_castps256_ps128(tmp3);
        const __m128 x128  = _mm_add_ps(left, right);
        const __m128 x64   = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        const __m128 x32   = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        float result       = _mm_cvtss_f32(x32);
        for (uint64_t i = n1; i < n0; i += 1) {
            result += u[i] * (float) v[i];
        }
        return result;
}





} // namespace vector
} // namespace hazy
#endif
