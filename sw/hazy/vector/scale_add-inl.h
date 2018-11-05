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

#ifndef HAZY_VECTOR_SCALE_ADD_INL_H
#define HAZY_VECTOR_SCALE_ADD_INL_H

// See for documentation
#include "hazy/vector/scale_add.h"
#include "hazy/vector/operations.h"
#include <math.h>

namespace hazy {
namespace vector {

template <typename float_u, typename float_v, typename float_>
void inline DropScaleAndAdd(SVector<float_u> &x, SVector<float_v> const& y, 
                   float_ scal) {
  size_t xi = x.size - 1;
  size_t yi = y.size - 1;
  while(xi > 0 && yi > 0) {
    int diff = x.index[xi] - static_cast<int>(y.index[yi]);
    if(diff == 0) {
      x.values[xi] += y.values[yi] * scal;
      xi--; yi--;
    } else {
      if(diff > 0) {xi--;}  else { yi--; }
    }
  }
}

template <typename float_v, typename float_>
void inline ScaleOnly(SVector<float_v> const&v, float_ const&s) {
  float_v const * const __restrict__ vvals = v.values;
  int const * const __restrict__ vidx = v.index;
  for (size_t i = v.size; i-- > 0; ) {
    vvals[i] * s;
  }
}

template <typename float_u, typename float_v, typename float_>
void inline ScaleAndAdd(FVector<float_u> &u, SVector<float_v> const &v,
                                             float_ const& s) {
  float_u * const __restrict__ uvals = u.values;
  float_v const * const __restrict__ vvals = v.values;
  int const * const __restrict__ vidx = v.index;
  for (size_t i = v.size; i-- > 0; ) {
    uvals[vidx[i]] = uvals[vidx[i]] + vvals[i] * s;
  }
}

template <typename float_u, typename float_v, typename float_>
void inline ScaleAndAdd(FVector<float_u> &u, FVector<float_v> const &v,
                                             float_ const& s) {
  float_u * const __restrict__ uvals = u.values;
  float_v const * const __restrict__ vvals = v.values;

  //#pragma omp simd 
  for (size_t i = 0; i < v.size; ++i) {
    uvals[i] = uvals[i] + (vvals[i] * s);
  }
  
}

template <typename float_u, typename float_v, typename float_,
          typename t_float>
void inline ScaleAndAddInto(FVector<float_u> const &u, 
                        FVector<float_v> const &v, float_ const& s,
                        FVector<t_float> &copyout) {
  t_float * const __restrict__ out = copyout.values;
  float_v const * const __restrict__ vvals = v.values;
  float_u const * const __restrict__ uvals = u.values;
  for (size_t i = v.size; i-- > 0; ) {
    out[i] = uvals + vvals * s;
  }
}

template <typename float_u, typename float_v>
void inline ScaleInto(FVector<float_u> const &u, float_v const &s, 
                      FVector<float_u> &copyto) {
  float_u * const __restrict__ arr = copyto.values;
  float_u const * const __restrict__ uarr = u.values;
  for (size_t i = u.size; i -- > 0; ) {
    arr[i] = uarr[i] * s;
  }
  copyto.size = u.size;
}

template <typename float_u, typename float_v>
void inline Scale(FVector<float_u> &u, float_v const &s) {
  float_u * __restrict__ const uarr = u.values;
  for (size_t i = u.size; i-- > 0; ) {
    uarr[i] *= s;
  }
}

//Dan's Quantization Stuff

//returns 1 with probability x, 0 otherwise
int inline indicator ( double x ) {
  double r_number = rand() / double(RAND_MAX);
  //returns random number between 0 and 1

  if ( r_number > x )
    return 0;
  else 
    return 1; 
}

// Not used anymore (same as QSGDQuantizeInto(v, 1)
// template <typename float_u>
// void QSGDQuantizeInto( FVector<float_u> &v ) {
//     double norm2 = vector::Norm2WithoutSquare( v );
//     float_u * const __restrict__ vvals = v.values;
//     
//     // int initial_sparsity = 0;
//     // int updated_sparsity = 0;
// 
//     for (size_t i = 0; i < v.size; i++ ) {
//       float_u val = vvals[i];
//       if( val != 0 ) {
//         double prob = (double) ((val > 0) ? val : (-val)) / (double) norm2;
//         double xi = indicator ( prob );
// 
//         //TODO: optimize later
//         vvals[i] = ((val > 0 ) - (val < 0)) * norm2 * xi;
//         //        		if( xi == 0 ) updated_sparsity++;
// 
//       }
//       // else {
//       //         	initial_sparsity++;
//       //         }
// 
//     }
//     //printf( "Sparsity stat: [initial: %d, updated: %d]\n", (initial_sparsity), (initial_sparsity + updated_sparsity) );
// }

template <typename float_u>
void QSGDQuantizeInto( FVector<float_u> &v, unsigned s ) {
    float_u * const __restrict__ vvals = v.values;
    
    double norm2 = vector::Norm2WithoutSquare( v );

    for (size_t i = 0; i < v.size; i++ ) {
      float_u val = vvals[i];
      if( val != 0 ) {

        // projected value is s * |v_i| / norm2
        double projected_value = s * ((double) ((val > 0) ? val : (-val)) / (double) norm2);
        // we now round it find out the index of its quantization level
        // the value will be quantized between L and (L + 1)
        double L = (double) floor( projected_value );
        double probability = projected_value - L;

        // double prob = (double) ((val > 0) ? val : (-val)) / (double) norm2;
        //             if( s == 1 && probability != prob )
        //             {
        //             	printf( "Error: %f %f\n", prob, probability );
        //             	exit(1);
        //             }
        //xi is 1 with probability, 0 otherwise
        double xi = indicator ( probability );
        if( xi == 1 ) {
          // if xi == 1 then we round up 
          xi = (L + 1) / (double) s;
        }
        else {
          //round down
          xi = L / (double) s;
        }

        //TODO: optimize later
        //v[i] will be sgn (v[i]) * normalizer * xi
        vvals[i] = ((val > 0 ) - (val < 0)) * norm2 * xi;

      }

    }
}

template <typename float_u, typename float_v>
void inline QSGDQuantizeOut(FVector<float_u> &u, SVector<float_v> const &v, unsigned s) {
  float_u * const __restrict__ uvals = u.values;
  float_v const * const __restrict__ vvals = v.values;
  int const * const __restrict__ vidx = v.index;

  double norm2 = vector::Norm2WithoutSquare( v );

  Zero(u);

  for (size_t i = 0; i < v.size; i++ ) {
    float_v val = vvals[i];
    if( val != 0 ) { // Should not be zero as v should be spares, nevertheless test

      double projected_value = s * ((double) ((val > 0) ? val : (-val)) / (double) norm2);
      double L = (double) floor( projected_value );
      double probability = projected_value - L;

      // 			double prob = (double) ((val > 0) ? val : (-val)) / (double) norm2;
      // 			if( s == 1 && probability != prob )
      // 			{
      // 				printf( "Error: %f %f\n", prob, probability );
      // 				exit(1);
      // 			}

      double xi = indicator ( probability );
      if( xi == 1 ) {
        xi = (L + 1) / (double) s;
      }
      else {
        xi = L / (double) s;
      }

      //TODO: optimize later
      uvals[vidx[i]] = ((val > 0 ) - (val < 0)) * norm2 * xi;     		
    }
  }
}

template <typename float_u, typename float_v>
void inline QSGDQuantizeOut(FVector<float_u> &u, FVector<float_v> const &v, unsigned s) {
  float_u * const __restrict__ uvals = u.values;
  float_v const * const __restrict__ vvals = v.values;

  double norm2 = vector::Norm2WithoutSquare( v );

  for (size_t i = 0; i < v.size; i++ ) {
    float_v val = vvals[i];
    if( val != 0 ) {

      double projected_value = s * ((double) ((val > 0) ? val : (-val)) / (double) norm2);
      double L = (double) floor( projected_value );
      double probability = projected_value - L;

      // 			double prob = (double) ((val > 0) ? val : (-val)) / (double) norm2;
      // 			if( s == 1 && probability != prob )
      // 			{
      // 				printf( "Error: %f %f\n", prob, probability );
      // 				exit(1);
      // 			}

      double xi = indicator ( probability );
      if( xi == 1 ) {
        xi = (L + 1) / (double) s;
      }
      else {
        xi = L / (double) s;
      }

      //TODO: optimize later
      uvals[i] = ((val > 0 ) - (val < 0)) * norm2 * xi;     		
    }
    else {
      uvals[i] = 0;
    }

  }
}


//TODO:: this is for the D error correction, but not yet implemented:
template <typename float_u, typename float_v>
void inline QSGDQuantizeOut(FVector<float_u> &u, FVector<float_u> &errors, FVector<float_v> const &v, unsigned s) {
  float_u * const __restrict__ uvals = u.values;
  float_u * const __restrict__ errv = errors.values;
  float_v const * const __restrict__ vvals = v.values;

  double norm2 = vector::Norm2WithoutSquare( v );

  for (size_t i = 0; i < v.size; i-- ) {
    float_u val = vvals[i];
    if( val != 0 ) {

      double projected_value = s * ((double) ((val > 0) ? val : (-val)) / (double) norm2);
      double L = (double) floor( projected_value );
      double probability = projected_value - L;

      // 			double prob = (double) ((val > 0) ? val : (-val)) / (double) norm2;
      // 			if( s == 1 && probability != prob )
      // 			{
      // 				printf( "Error: %f %f\n", prob, probability );
      // 				exit(1);
      // 			}

      double xi = indicator ( probability );
      if( xi == 1 ) {
        xi = (L + 1) / (double) s;
      }
      else {
        xi = L / (double) s;
      }

      //TODO: optimize later
      uvals[i] = ((val > 0 ) - (val < 0)) * norm2 * xi;     		
    }
    else {
      uvals[i] = 0;
      errv[i] = 0;
    }

  }
}


// 
// 
// template <typename float_u, typename float_v>
// void inline SimpleQuantizeAdd(FVector<float_u> &u, FVector<float_v> const &v, float s) {
//     float_u * const __restrict__ uvals = u.values;
//     float_v const * const __restrict__ vvals = v.values;
//     
//     double norm2 = vector::Norm2WithoutSquare( v );
//     
//     for (size_t i = v.size; i-- > 0; ) {
// 		float_v val = vvals[i];
// 		if( val != 0 ) {
// 			
// 			float prob = (float) ((val > 0) ? val : (-val)) / (float) norm2;
// 			double xi = indicator ( prob );
// 			
// 			if (xi != 0 ) {
// 				uvals[i] += ((val > 0 ) - (val < 0)) * norm2 * s;
// 			}
// 			
// 		}
// 	}
// }
// 
// 
// 
// template <typename float_u, typename float_v, typename float_>
// void inline QSGDQuantizeOutParam(FVector<float_u> &u, FVector<float_v> const &v, unsigned s) {
//     float_u * const __restrict__ uvals = u.values;
//     float_v const * const __restrict__ vvals = v.values;
//     
//     double norm2 = vector::Norm2WithoutSquare( v );
//     
//     for (size_t i = v.size; i-- > 0; ) {
// 		float_v val = vvals[i];
// 		if( val != 0 ) {
// 			float pval = s * ((val > 0) ? val : (-val)) / (float) norm2;
// 			unsigned L = (unsigned) pval;
// 			float xi = (indicator( pval - L ) == 0) ? ( (L + 1) / (float) s ) : ( L / s );
// 
// 			uvals[i] = ((val > 0 ) - (val < 0)) * norm2 * xi;
// 			
// 		}
// 	}
// }
// 
// template <typename float_u, typename float_v>
// void inline QuantizeAndAddParamScaled(FVector<float_u> &u, FVector<float_v> const &v, unsigned s, float scalar) {
//     float_u * const __restrict__ uvals = u.values;
//     float_v const * const __restrict__ vvals = v.values;
//     
//     double norm2 = vector::Norm2WithoutSquare( v );
//     
//     for (size_t i = v.size; i-- > 0; ) {
// 		float_v val = vvals[i];
// 		if( val != 0 ) {
// 			float pval = s * ((val > 0) ? val : (-val)) / (float) norm2;
// 			unsigned L = (unsigned) pval;
// 			float xi = (indicator( pval - L ) == 0) ? ( (L + 1) / (float) s ) : ( L / s );
// 
// 			uvals[i] += ((val > 0 ) - (val < 0)) * norm2 * xi * scalar;
// 			
// 		}
// 	}
// }
// 
// 
// template <typename float_v>
// void inline ParamQuantizeInto( FVector<float_v> &v, unsigned s ) {
//     float_v * const __restrict__ vvals = v.values;
//     
//     double norm2 = vector::Norm2WithoutSquare( v );
//     
//     for (size_t i = v.size; i-- > 0; ) {
// 		float_v val = vvals[i];
// 		if( val != 0 ) {
// 			double pval = s * ((val > 0) ? val : (-val)) / (double) norm2;
// 			double L = floor( pval );
// 			double pr = (pval - L);			
// 			unsigned ind = indicator (pr);
// 			double xi = (ind == 1) ? ( (L + 1) / (double) s ) : ( L / (double) s );
// 			
// 			//printf( "pval: %f, L: %f, ind: %ud, xi: %f\n", pval, L, ind, xi );
// 
// 			double my_val = ((val > 0 ) - (val < 0)) * norm2 * xi;
// 			
// 			
// 			double prob2 = (double) ((val > 0) ? val : (-val)) / (double) norm2;
// 			if( pr != prob2 ) {
// 				printf( "prob2: %f, pr: %f\n", prob2, pr );
// 				xi = indicator( pr ); 
// 			}
// 			else {
// 				xi = ind;
// 			}
// 			
// 			vvals[i] = ((val > 0 ) - (val < 0)) * norm2 * xi;
// 						
//              if( my_val != vvals[i] ) { 
//              	printf( "Error: %f versus %f\n", my_val, vvals[i] );
//              	exit(1);
//              }
//             	
// 		}
// 	}
// }
// 
// 



} // namespace vector
} // namespace hazy
#endif
