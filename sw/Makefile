## 
##
## 

##******************************************************************************
CPPFLAGS ?= -std=c++11 -O3 -lrt -lpthread
CXX      ?= icc #g++
LDFLAGS  ?=

ifeq (,$(CFLAGS))
CFLAGS = -g -O2
endif

ifneq (,$(ndebug))
else
CPPFLAGS += -DENABLE_DEBUG=1
endif
ifneq (,$(nassert))
else
CPPFLAGS += -DENABLE_ASSERT=1
endif

#echo $(prefix)

ifeq (,$(DESTDIR))
ifneq (,$(prefix))
CPPFLAGS += -I$(prefix)/include
LDFLAGS	 += -L$(prefix)/lib -Wl,-rpath-link -Wl,$(prefix)/lib -Wl,-rpath -Wl,$(prefix)/lib \
	    -L$(prefix)/lib64 -Wl,-rpath-link -Wl,$(prefix)/lib64 -Wl,-rpath -Wl,$(prefix)/lib64
endif
else
ifeq (,$(prefix))
prefix = /usr/local
endif
CPPFLAGS += -I$(DESTDIR)$(prefix)/include
LDFLAGS	 += -L$(DESTDIR)$(prefix)/lib -Wl,-rpath-link -Wl,$(prefix)/lib -Wl,-rpath -Wl,$(DESTDIR)$(prefix)/lib \
	    -L$(DESTDIR)$(prefix)/lib64 -Wl,-rpath-link -Wl,$(prefix)/lib64 -Wl,-rpath -Wl,$(DESTDIR)$(prefix)/lib64
endif

#-lboost_program_options
LDFLAGS +=  -lOSAL -lAAS -laalrt

CFLAGS += -DVCMAP_ENABLE_DEFAULT=false




CT_CORE_SRC_DIR=$(CENTAUR_HOME)/common/sw/core
CT_USR_SRC_DIR=$(CENTAUR_HOME)/common/sw/usr
CT_PLATFORM_SRC_DIR=$(CENTAUR_HOME)/$(PLATFORM_NAME)/sw

CORE_SRC     = $(CT_CORE_SRC_DIR)/workload_manager.cpp $(CT_CORE_SRC_DIR)/utils.cpp  $(CT_CORE_SRC_DIR)/dlmalloc.cpp $(CT_CORE_SRC_DIR)/memory_manager.cpp $(CT_USR_SRC_DIR)/fpgahandle.cpp
USR_SRC      = $(CT_USR_SRC_DIR)/fthread.cpp $(CT_USR_SRC_DIR)/fthreadRec.cpp 
PLATFORM_SRC = $(CT_PLATFORM_SRC_DIR)/platform.cpp $(CT_PLATFORM_SRC_DIR)/vendor.cpp

SOURCES      = $(PLATFORM_SRC) $(CORE_SRC) $(USR_SRC) sgd_main.cpp sgd_pm.cpp

CPPFLAGS += -I$(CT_CORE_SRC_DIR) -I$(CT_USR_SRC_DIR) -I$(CT_PLATFORM_SRC_DIR) -I$(CENTAUR_HOME)/$(PLATFORM_NAME)/rtl/mpf_files -I$(CENTAUR_HOME) -I. -I./hazy/thread

all: sgd_main

sgd_main: $(SOURCES) Makefile
	icc $(CPPFLAGS) -D__AAL_USER__=1 -g -O2 -o sgd_main $(SOURCES) $(LDFLAGS) libpcm_2_11.a #$(CXX)

clean:
	$(RM) sgd_main

.PHONY:all clean
