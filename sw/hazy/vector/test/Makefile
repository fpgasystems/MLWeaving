CXX         = icc #g++
CPP_FLAG    = -O3 -std=c++11 -lrt #-Wall -fopenmp -Wno-unknown-pragmas -mavx2 -lpthread -lm
CPP_INCLUDE = -I ../../../
CPP_LAST    = -lpthread -lm

COMMON_CPP = ../tool/rand_tool.cpp  ../tool/cpu_mapping.cpp ../tool/huge_page.cpp  ../tool/thread_tool.cpp

.PHONY: depend clean all
all: TEST_MLWEAVING_8MR_AVX2 # TEST_MLWEAVING TEST_MLWEAVING_AVX2   $(COMMON_O)

clean: 
	rm -f TEST_MLWEAVING TEST_MLWEAVING_AVX2 TEST_MLWEAVING_8MR_AVX2


TEST_MLWEAVING: test_mlweaving.cpp
	#mkdir -p $(BIN_DIR)
	${CXX} ${CPP_FLAG} ${CPP_INCLUDE} $< -o $@


TEST_MLWEAVING_AVX2: test_mlweaving_avx2.cpp
	#mkdir -p $(BIN_DIR)
	${CXX} ${CPP_FLAG} ${CPP_INCLUDE} $< -o $@


TEST_MLWEAVING_8MR_AVX2: test_mlweaving_8mr_avx2.cpp
	#mkdir -p $(BIN_DIR)
	${CXX} ${CPP_FLAG} ${CPP_INCLUDE} $< -o $@

