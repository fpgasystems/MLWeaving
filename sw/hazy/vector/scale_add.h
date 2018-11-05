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

#ifndef HAZY_VECTOR_SCALE_ADD_H
#define HAZY_VECTOR_SCALE_ADD_H

namespace hazy {
namespace vector {

/*! \brief Update as FVector += scalar * SVector
 * Assumes missing entries in the SVector are zero
 * \param u the vector to modify
 * \param v the vector to scale and then add
 * \param s the scalar
 */
template <typename float_u, typename float_v, typename float_>
void inline ScaleAndAdd(FVector<float_u> &u, SVector<float_v> const&v,
                        float_ const&s);


/*! \brief Update as SVector += scalar * SVector
 * Assumes missing entries in the SVector are zero, ignores missing indicies
 * of x (hence the "drop").
 * \param x the vector to modify
 * \param y the vector to scale and then add
 * \param scal the scalar
 */
template <typename float_u, typename float_v, typename float_>
void inline DropScaleAndAdd(SVector<float_u> &x, SVector<float_v> const& y, 
                   float_ scal);


/*! \brief Scales the given vector while copying it
 * Assumes destination vector is big enough to hold it, otherwise segfault
 * \param u the vector to modify
 * \param s the scalar
 * \param copyto the vector to copy into
 */
template <typename float_u, typename float_v>
void inline ScaleInto(FVector<float_u> const &u, float_v const &s, 
                      FVector<float_u> &copyto);


/*! \brief compute scalar * SVector, just to see how fast we can compute gradients
 * Assumes missing entries in the SVector are zero
 * \param v the vector to scale
 * \param s the scalar
 */
template <typename float_v, typename float_>
void inline ScaleOnly(SVector<float_v> const&v, float_ const&s);

/*! \brief Scale the vector in place
 * \param u the vector to scale
 * \param s the scalar
 */
template <typename float_u, typename float_v>
void inline Scale(FVector<float_u> &u, float_v const &s);

/*! \brief Scale and add into a destination vector
 * Caller must ensure copyout is allocated with enough memory
 * \param u the vector to add to
 * \param v the vector to scale when adding
 * \param s scalar
 * \param copyout the vector to copy into
 */
template <typename float_u, typename float_v, typename float_,
          typename t_float>
void inline ScaleAndAddInto(FVector<float_u> const &u, 
                        FVector<float_v> const &v, float_ const& s,
                        FVector<t_float> &copyout);


/*! \brief Quantizes the vector in place using random one-bit quantization
 * \param v the vector to quantize
 */
template <typename float_u>
void inline QSGDQuantizeInto( FVector<float_u> &v, unsigned s );

/*! \brief Quantizes the vector into u using random one-bit quantization
 * \param u the resulting quantized vector
 * \param v the original
 */                                         
template <typename float_u, typename float_v>
void inline QSGDQuantizeOut(FVector<float_u> &u, FVector<float_v> const &v);   

template <typename float_u, typename float_v>
void inline QSGDQuantizeOut(FVector<float_u> &u, FVector<float_v> const &v, unsigned s) ;                                     
                                                
/*! \brief Quantizes the vector into using random quantization with s levels
 * \param u the resulting quantized vector
 * \param v the original
 * \param s the quantization levels (there are 2s + 1 of them)
 */      
template <typename float_u, typename float_v>
void inline QSGDQuantizeOutParam(FVector<float_u> &u, FVector<float_v> const &v, unsigned s);

template <typename float_u, typename float_v>
void inline QuantizeAndAddParamScaled(FVector<float_u> &u, FVector<float_v> const &v, unsigned s, float scalar);

template <typename float_u, typename float_v>
void inline SimpleQuantizeAdd(FVector<float_u> &u, FVector<float_v> const &v, float s);

template <typename float_u>
void inline ParamQuantizeInto( FVector<float_u> &u, unsigned s );

} // namespace vector
} // namespace hazy
#endif
