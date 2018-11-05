// Copyright 2012 Victor Bittorf, Chris Re
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

#ifndef HAZYHTL_TYPES_ENTRY_H
#define HAZYHTL_TYPES_ENTRY_H

namespace hazy {
namespace types {

/*! \brief A single entry in a matrix
 */
struct Entry {
  int row, col;
  double rating;

  Entry() { }

  Entry(const Entry &o) {
    row = o.row;
    col = o.col;
    rating = o.rating;
  }

  inline void operator=(const Entry &o) {
    row = o.row;
    col = o.col;
    rating = o.rating;
  }
};

} // namespace types
} // namespace hazy
#endif
