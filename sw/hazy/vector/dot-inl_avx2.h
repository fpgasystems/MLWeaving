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

#ifndef HAZY_VECTOR_DOT_INL_H
#define HAZY_VECTOR_DOT_INL_H

#include <cmath>
#include "hazy/vector/dot.h"
#include <immintrin.h>


// See hazy/vector/dot.h for documentation

namespace hazy {
namespace vector {

template <typename T, typename Vec>
void inline MatrixVectorMultiply(FVector<SVector<T> > const &mat, 
                                 Vec const &vec,
                                 FVector<T> &out) {
  // Note: Assume mat is padded with zeros
  out.size = mat.size;
  for (size_t i = 0; i < mat.size; i++) {
    out.values[i] = Dot(vec, mat.values[i]);
  }
}


template <typename float_u, typename float_v>
float_u inline Dot(FVector<float_u> const& u, SVector<float_v> const& v) {
  float_u p = 0.0;
  float_u const * const /*__restrict__*/ uvals = u.values;
  float_v const * const /*__restrict__*/ vvals = v.values;
  int const * const /*__restrict__*/ vidx = v.index;
  for (size_t i = v.size; i-- > 0; ) {
    p += uvals[vidx[i]] * vvals[i]; 
  }
  return p;
}

//template <typename float_u, typename float_v>
float inline Dot(FVector<float> const &uVector, FVector<unsigned char> const &vVector) 
{

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
				   //__m128i v1_128 = _mm_loadu_si128(v + i + 0);
				  // __m128i v2_128 = _mm_loadu_si128(v + i + 8);
				  // __m128i v3_128 = _mm_loadu_si128(v + i + 16);
				   //__m128i v4_128 = _mm_loadu_si128(v + i + 24);
				   //__m256i v1_abs = _mm256_cvtepu16_epi32(v1_128);
				   //__m256i v2_abs = _mm256_cvtepu16_epi32(v2_128);
				   //__m256i v3_abs = _mm256_cvtepu16_epi32(v3_128);
				   //__m256i v4_abs = _mm256_cvtepu16_epi32(v4_128);
				   _mm_prefetch((char *)(v + i + 64), _MM_HINT_NTA); //Stay at L3
				   _mm_prefetch((char *)(u + i + 64), _MM_HINT_T0);  //Stay at L1
				   _mm_prefetch((char *)(u + i + 80), _MM_HINT_T0);  //Stay at L1
				   
				   __m128i v1_128 = _mm_loadu_si128((__m128i const* )(v + i + 0));
				   __m128i v3_128 = _mm_loadu_si128((__m128i const* )(v + i + 16));
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


//template <typename float_u, typename float_v>
float inline Dot(FVector<float> const &uVector, FVector<unsigned short> const &vVector) 
{

            const uint64_t n0 = uVector.size;
            const uint64_t n1 = n0 & 0xFFFFFFFFFFFFFFE0UL;
            const float * u            = uVector.values;
            const unsigned short * v   = vVector.values;

            __m256 acc1 = _mm256_setzero_ps(); 
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
			
			const __m256i and_const  = _mm256_set1_epi32(0xffff);

            for (uint64_t i = 0; i < n1; i += 32)
            {// 
                //const __m256 v1 = _mm256_loadu_ps(v + i + 0);
                //const __m256 v2 = _mm256_loadu_ps(v + i + 8);
                //const __m256 v3 = _mm256_loadu_ps(v + i + 16);
                //const __m256 v4 = _mm256_loadu_ps(v + i + 24);
				   __m128i v1_128 = _mm_loadu_si128((const __m128i *) (v + i + 0 ));
				   __m128i v2_128 = _mm_loadu_si128((const __m128i *) (v + i + 8 ));
				   __m128i v3_128 = _mm_loadu_si128((const __m128i *) (v + i + 16));
				   __m128i v4_128 = _mm_loadu_si128((const __m128i *) (v + i + 24));
				   //__m256i v1_256 = _mm256_cvtepi16_epi32(v1_128);
				   //__m256i v2_256 = _mm256_cvtepi16_epi32(v2_128);
				   //__m256i v3_256 = _mm256_cvtepi16_epi32(v3_128);
				   //__m256i v4_256 = _mm256_cvtepi16_epi32(v4_128);
				   //__m256i v1_abs = _mm256_and_si256(v1_256, and_const);
				   //__m256i v2_abs = _mm256_and_si256(v2_256, and_const);
				   //__m256i v3_abs = _mm256_and_si256(v3_256, and_const);
				   //__m256i v4_abs = _mm256_and_si256(v4_256, and_const);
				   __m256i v1_abs = _mm256_cvtepu16_epi32(v1_128);
				   __m256i v2_abs = _mm256_cvtepu16_epi32(v2_128);
				   __m256i v3_abs = _mm256_cvtepu16_epi32(v3_128);
				   __m256i v4_abs = _mm256_cvtepu16_epi32(v4_128);
				   
                        __m256 v1 = _mm256_cvtepi32_ps(v1_abs);
                        __m256 v2 = _mm256_cvtepi32_ps(v2_abs);
                        __m256 v3 = _mm256_cvtepi32_ps(v3_abs);
                        __m256 v4 = _mm256_cvtepi32_ps(v4_abs);

                const __m256 u1 = _mm256_loadu_ps((u + i + 0 ));
                const __m256 u2 = _mm256_loadu_ps((u + i + 8 ));
                const __m256 u3 = _mm256_loadu_ps((u + i + 16));
                const __m256 u4 = _mm256_loadu_ps((u + i + 24));

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


float inline Dot(FVector<float> const &uVector, FVector<float> const &vVector) 
{

            const uint64_t n0 = uVector.size;
            const uint64_t n1 = n0 & 0xFFFFFFFFFFFFFFE0UL;
            const float * u   = uVector.values;
            const float * v   = vVector.values;

            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();

            for (uint64_t i = 0; i < n1; i += 32)
            {
                const __m256 v1 = _mm256_loadu_ps(v + i + 0);
                const __m256 v2 = _mm256_loadu_ps(v + i + 8);
                const __m256 v3 = _mm256_loadu_ps(v + i + 16);
                const __m256 v4 = _mm256_loadu_ps(v + i + 24);

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
                result += u[i] * v[i];
            }
            return result;
}

template <typename float_u, typename float_v>
float_u inline Dot(SVector<float_u> const& x, SVector<float_v> const& y) 
{
  long xi = (long)x.size - 1;
  long yi = (long)y.size - 1;
  float_u ret = 0.0;
  while(xi >= 0 && yi >= 0) {
    int diff = x.index[xi] - static_cast<int>(y.index[yi]);
    if(diff == 0) {
      ret += x.values[xi]*y.values[yi];
      xi--; yi--;
    } else {  
      if(diff > 0) {xi--;}  else { yi--; }
    }
  }
  return ret;
}

} // namespace vector
} // namespace hazy
#endif
