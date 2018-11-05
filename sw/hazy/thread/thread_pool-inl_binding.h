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
//
// Added NUMA stuff : Cedric Renggli (rengglic [at] student.ethz.ch)

#ifndef HAZY_THREAD_THREAD_POOL_INL_H
#define HAZY_THREAD_THREAD_POOL_INL_H

#include "cpu_mapping.h"

#include <assert.h>

// See for documentation
#include "thread_pool.h"

namespace hazy {
namespace thread {

namespace __threadpool {
// Hook which pthread_create will call for us
// This simply calls back into the thread pool
void* RunThread(void* thread_meta) {
  ThreadMeta &meta = *static_cast<ThreadMeta*>(thread_meta);
  ThreadPool *tp = static_cast<ThreadPool*>(meta.tpool);
  tp->ThreadLoop(meta);
  return NULL;
}

// A way to keep the types for the caller, this will be the call back
// in the thread pool (callback_)
template <class T>
void Invoke(ThreadPool &tp, unsigned thread_id) {
  tp.ThreadCallback<T>(thread_id);
}
} // namespace __threadpool

ThreadPool::~ThreadPool() {
  if (threads_ == NULL) {
    return;
  }
  delete [] threads_;
  delete [] metas_;
  barrier_destroy(&ready_);
  barrier_destroy(&finished_);
}

void ThreadPool::Init() {
  hook_ = NULL;
  callback_ = NULL;
  arg_ = NULL;

  // +1 because 'main' thread will join the barriers
  barrier_init(&ready_, NULL, n_threads_+1);
  barrier_init(&finished_, NULL, n_threads_+1);

  metas_ = new ThreadMeta[n_threads_];

  SetExitFlags(false);
  for (unsigned i = 0; i < n_threads_; i++) {
    metas_[i].thread_id = i;
    metas_[i].ready = &ready_;
    metas_[i].finished = &finished_;
    metas_[i].tpool = this;
  }
  
  //for affinity setting.
  pthread_t tid[n_threads_];
  pthread_attr_t attr;
  cpu_set_t set; //cpu_set_t *set = (cpu_set_t *) malloc (sizeof (cpu_set_t)); //
  pthread_attr_init(&attr);


  threads_ = new pthread_t[n_threads_];
  for (unsigned i = 0; i < n_threads_; i++) 
  	{
	//for affinity setting. 	
	int cpu_idx = get_cpu_id(i);
	//DEBUGMSG(1, "Assigning thread-%d to CPU-%d\n", i, cpu_idx);
	CPU_ZERO(&set);
	CPU_SET(cpu_idx, &set);
	pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
    pthread_create(&threads_[i], &attr, __threadpool::RunThread,
                   static_cast<void*>(&metas_[i]));
#ifdef _NUMA_INIT
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    for(int j = (i - (i % CPUS_PER_NODE)) / CPUS_PER_NODE; j < (CPUS_PER_NODE * NUMA_NODES); j+=NUMA_NODES)
      CPU_SET(j, &cpuset);

    int s;
    s = pthread_setaffinity_np(threads_[i], sizeof(cpu_set_t), &cpuset);
    if (s != 0)
      printf("Error while setting affinity !!!");

    /*s = pthread_getaffinity_np(threads_[i], sizeof(cpu_set_t), &cpuset);
    if (s != 0)
      printf("Error while setting affinity !!!");

    printf("%i - Set returned by pthread_getaffinity_np() contained:\n", i);
    int j;
    for (j = 0; j < (CPUS_PER_NODE * NUMA_NODES); j++)
      if (CPU_ISSET(j, &cpuset))
        printf("    CPU %d\n", j);*/

#endif
  }
  ready_flag_ = true;
}

void ThreadPool::ThreadLoop(ThreadMeta &meta) {
  while (true) {
    barrier_wait(meta.ready);
    if (meta.exit_flag) {
      break;
    }
    callback_(*this, meta.thread_id);
    barrier_wait(meta.finished);
  }
}

template <class Task>
void ThreadPool::ThreadCallback(unsigned thread_id) {
  Task *t = static_cast<Task*>(arg_);
  void (*hook)(Task&, unsigned, unsigned) = reinterpret_cast<
      void (*)(Task&, unsigned, unsigned)>(hook_);
  hook(*t, thread_id, n_threads_);
}

template <class Task>
void ThreadPool::Execute(Task &task, void (*hook)(Task&, unsigned, 
                         unsigned)) {
  assert(ready_flag_);
  // assign each 
  arg_ = &task; 
  hook_ = reinterpret_cast<void*>(hook);
  callback_ = &__threadpool::Invoke<Task>;
  ready_flag_ = false;
  barrier_wait(&ready_);
}

void ThreadPool::Wait() {
  assert(!ready_flag_);
  barrier_wait(&finished_);
  ready_flag_ = true;
}

void ThreadPool::Join() {
  assert(ready_flag_);

  SetExitFlags(true);
  barrier_wait(&ready_);
  for (unsigned i = 0; i < n_threads_; i++) {
    pthread_join(threads_[i], NULL);
  }
}

} // namespace thread
} // namespace hazy
#endif

