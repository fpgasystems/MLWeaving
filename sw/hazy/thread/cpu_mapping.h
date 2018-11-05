/*******************************************************************************
 * Copyright (c) 2016
 * The National University of Singapore, Xtra Group
 *
 * Author: Zeke Wang (wangzeke638 AT gmail.com)
 * Import form Cagri Balkesen.
 * Description: Provides cpu mapping utility function.
 *******************************************************************************/

 
#ifndef CPU_MAPPING_H
#define CPU_MAPPING_H

#include <stdio.h>  /* FILE, fopen */
#include <stdlib.h> /* exit, perror */
#include <unistd.h> /* sysconf */

//#include "cpu-mapping.txt"
#define CUSTOM_CPU_MAPPING "cpu-mapping.txt"

#define MAX_NODES 512

static int inited = 0;
static int max_cpus;
static int node_mapping[MAX_NODES];

/** 
 * Initializes the cpu mapping from the file defined by CUSTOM_CPU_MAPPING.
 * The mapping used for our machine Intel L5520 is = "8 0 1 2 3 8 9 10 11".
 */
static int
init_mappings_from_file()
{
    FILE * cfg;
	int i;

    cfg = fopen(CUSTOM_CPU_MAPPING, "r");
    if (cfg!=NULL) {
        if(fscanf(cfg, "%d", &max_cpus) <= 0) {
            perror("Could not parse input!\n");
        }

        for(i = 0; i < max_cpus; i++){
            if(fscanf(cfg, "%d", &node_mapping[i]) <= 0) {
                perror("Could not parse input!\n");
            }
        }

        fclose(cfg);
        return 1;
    }
    /* perror("Custom cpu mapping file not found!\n"); */
    return 0;
}

/** 
 * Try custom cpu mapping file first, if does not exist then round-robin
 * initialization among available CPUs reported by the system. 
 */
static void 
init_mappings()
{
    if( init_mappings_from_file() == 0 ) {
        int i;
        
        max_cpus  = sysconf(_SC_NPROCESSORS_ONLN);
        for(i = 0; i < max_cpus; i++){
            node_mapping[i] = i;
        }
    }
}

/** @} */

/**
 * Returns SMT aware logical to physical CPU mapping for a given thread id.
 */
int 
get_cpu_id(int thread_id) 
{
    if(!inited){
        init_mappings();
        inited = 1;
    }

    return node_mapping[thread_id % max_cpus];
}

#endif /* CPU_MAPPING_H */
