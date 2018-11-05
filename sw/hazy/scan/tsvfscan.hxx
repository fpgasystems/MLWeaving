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

// Hogwild!, part of the Hazy Project
// Author : Victor Bittorf (bittorf [at] cs.wisc.edu)
// Original Hogwild! Author: Chris Re (chrisre [at] cs.wisc.edu)       
// 
// Bugfix (and remove DIM) : Cedric Renggli (rengglic [at] student.ethz.ch)     
//  > Empty files bug 

#include <assert.h>

namespace hazy {
namespace scan {

#include <assert.h>

template <int OFFSET>
OffsetTSVFileScanner<OFFSET>::OffsetTSVFileScanner(
    const char *fname) : fname_(fname), peekable_(false), max_col_(0) {
  fh = fopen(fname, "r");
  if (fh == NULL) {
    perror("Could not open file");
    assert(false);
  }
  Reset();
}

template <int OFFSET>
bool OffsetTSVFileScanner<OFFSET>::HasNext() {
  if (peekable_) {
    return true;
  }
  return !feof(fh);
}

template <int OFFSET>
const types::Entry& OffsetTSVFileScanner<OFFSET>::Peek() {
  if (peekable_) {
    return entry_;
  }

  assert(fscanf(fh, "%d\t%d\t%lf", &entry_.row, &entry_.col, &entry_.rating) == 3);

  char temp[8];
  while (fscanf(fh, "%1[ \n]s", temp)) { if (feof(fh)) break; }

  peekable_ = true;
  entry_.row += OFFSET;
  entry_.col += OFFSET;
  return entry_;
}

template <int OFFSET>
const types::Entry& OffsetTSVFileScanner<OFFSET>::Next() {
  Peek(); // make sure entry_ is set
  peekable_ = false;
  return entry_;
}

template <int OFFSET>
void OffsetTSVFileScanner<OFFSET>::Reset() {
  fclose(fh);
  fh = fopen(fname_, "r");
  peekable_ = false;

  // Skip empty lines
  char temp[8];
  while (fscanf(fh, "%1[ \n]s", temp)) { if (feof(fh)) break; }
}

} // namespace fscan
} // namespace hazy
