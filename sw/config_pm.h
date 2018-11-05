#ifndef CONGIG_PM
#define CONGIG_PM

#include <iostream>
#include <string>
#include <fstream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <limits>
#include <cmath>
//#include <malloc.h>

#include "centaur.h" ///home/zeke/centaur CENTAUR_HOME/

#define SGD_AFU_BITWEAVING  0xf

/*This file contains the general configuration for the precision manager*/

/*It determines the pipeline width of SGD on FPGA, which can consume one cache line per cycle....*/
#define BITS_OF_ONE_CACHE_LINE 512




#endif

