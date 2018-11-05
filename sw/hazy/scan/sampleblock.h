// Author : Cedric Renggli (rengglic [at] student.ethz.ch)

#ifndef HAZY_HOGWILD_SAMPLE_BLOCK_H
#define HAZY_HOGWILD_SAMPLE_BLOCK_H

#include "hazy/vector/fvector.h"

namespace hazy {
namespace scan {

template< class Sample >
struct SampleBlock
{
  vector::FVector<Sample> ex;
  vector::FVector<size_t> perm;
};   

}
}

#endif
