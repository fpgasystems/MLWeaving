// Copyright (C) 2019 Zeke Wang - Systems Group, ETH Zurich

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//*************************************************************************
//this file runs the precision manager for the SGD 
//
//
//*************************************************************************

#include "sgd_pm.h"

// #include <unistd.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////



int main(int argc, char *argv[])
{
  uint32_t dataset_index         = argc > 1 ? atoi(argv[1]) : 0; //default set to 0...
  uint32_t exec_model            = argc > 2 ? atoi(argv[2]) : 0; //default set to 0...
  uint32_t numberOfBits          = argc > 3 ? atoi(argv[3]) : 8; //default set to 0...;
  uint32_t numberOfIterations    = argc > 4 ? atoi(argv[4]) : 30; //3; 
  uint32_t stepSizeShifter       = argc > 5 ? atoi(argv[5]) : 12;//10;
  uint32_t mini_batch_size       = argc > 6 ? atoi(argv[6]) : 8;
  uint32_t num_fractional_bits   = argc > 7 ? atoi(argv[7]) : 23;
  uint32_t nthreads              = argc > 8 ? atoi(argv[8]) : 1;
  uint32_t targeted_label        = argc > 9 ? atoi(argv[9]) : 1;
  uint32_t num_epochs_a_decay    = argc > 10? atoi(argv[10]): 10000;
  float    decay_initial         = argc > 11? atof(argv[11]): 1.0;
    
//char *trainFile          = "/upb/departments/pc2/scratch/zekewang/SGD/dataset/imagenet_nor_1G_4M_2.dat";

  ////I do not know the meaning of value_to_integer_scaler////
  uint32_t value_to_integer_scaler = 0x00800000;

  printf("dataset_index = %d, exec_model = %d\n", dataset_index, exec_model);
  printf("numberOfBits = %d, numberOfIterations = %d, stepSizeShifter = %d, mini_batch_size = %d, num_fractional_bits = %d, num_threads = %d, targeted_label = %d, num_epochs_a_decay = %d, decay_initial = %f\n", 
          numberOfBits,      numberOfIterations,      stepSizeShifter,      mini_batch_size,      num_fractional_bits,      nthreads,         targeted_label,      num_epochs_a_decay,      decay_initial 
        );


  bool usingFPGA = true;
  if ( (exec_model != 1)) // || (exec_model == 2) 

    usingFPGA = false;

  // Instantiate the main object, which includes the initialization of hardware part...
  zipml_sgd_pm sgd(usingFPGA, value_to_integer_scaler);


  if (dataset_index == 0)
  {
    printf("------Training the dataset: gisette_scale (6000 samples, 5000 features)\n");
    sgd.load_libsvm_data((char*)"./Datasets/gisette_scale", 6000, 5000); //40  ~/Documents/ML/data/gisette_scale
  }
  else if (dataset_index == 1)
  {
    printf("------Training the mnist: (60000 samples, 780 features)\n");
    sgd.load_libsvm_data_1((char*)"./Datasets/mnist", 12873*4 , 780); // 60000
    //sgd.b_normalize(0,  1, 7.0);     
  }   
  else if (dataset_index == 2)
  { ///upb/departments/pc2/users/z/zekewang
    char *trainFile          = "./Datasets/imagenet_nor_128M_4M_2.dat";
    printf("------Training the ImageNet (5200 samples, 2048 features)\n");
    sgd.load_dense_data((char*)trainFile, 5200, 2048);
    //sgd.b_normalize(0,  1, 7.0);     
  }   
 else if (dataset_index == 3)
  {
    printf("------Training the epsilon: (10000 samples, 20000 features)\n");
    sgd.load_libsvm_data_1((char*)"./Datasets/epsilon_small", 10000, 2000);
    //sgd.b_normalize(0,  1, 7.0);     
  }
 else if (dataset_index == 4)
  {
    printf("------Training the madelon: (2000 samples, 500 features)\n");
    sgd.load_libsvm_data_1((char*)"./Datasets/madelon", 2000, 500);
    //sgd.b_normalize(0,  1, 7.0);     
  }
 else if (dataset_index == 5)
  {
    printf("------Training the cifar10: (50,000 samples, 3072 features)\n");
    //sgd.load_libsvm_data((char*)"./Datasets/cifar10", 50000, 3072);
    sgd.load_libsvm_data((char*)"./Datasets/cifar10", 10000, 3072);
    //sgd.b_normalize(0,  1, 7.0);     
  } 
 else if (dataset_index == 6)
  {
    printf("------Training the RCV1: (20242 samples, 47236 features)\n"); //677399
    //sgd.load_libsvm_data((char*)"./Datasets/cifar10", 50000, 3072);
    sgd.load_tsv_data((char*)"./Datasets/rcv1_train.tsv", 20242, 47236);
    //sgd.b_normalize(0,  1, 7.0);     
  }     
  else if (dataset_index == 21)
  {
    char *trainFile          ="./Datasets/imagenet_nor_1G_4M_2.dat"; // "/upb/departments/pc2/scratch/zekewang/SGD/dataset/imagenet_nor_1G_4M_2.dat";
    printf("------Training the ImageNet (83200 samples, 2048 features)\n");
    sgd.load_dense_data((char*)trainFile, 83200, 2048);
    //sgd.b_normalize(0,  1, 7.0);     
  }   
  else if (dataset_index == 11)
  {
    printf("------Training the mnist: (60000 samples, 780 features)\n");
    //sgd.load_libsvm_data_1((char*)"./Datasets/mnist_filter", 12665, 780);
    sgd.load_libsvm_data_1((char*)"./Datasets/mnist_filter_3", 60000, 780); //63325
        //void zipml_sgd_pm::load_libsvm_data_1_two(char* pathToFile, uint32_t _numSamples, uint32_t _numFeatures, uint32_t target_1, uint32_t target_2);
    //sgd.b_normalize(0,  1, 7.0);     
  }
 else if (dataset_index == 41)
  {
    printf("------Training the madelon_double_5: (10000 samples, 500 features)\n");
    sgd.load_libsvm_data_1((char*)"./Datasets/madelon_double_5", 10000, 500);
    //sgd.b_normalize(0,  1, 7.0);     
  }
 else if (dataset_index == 42)
  {
    printf("------Training the madelon_double_10: (20000 samples, 500 features)\n");
    sgd.load_libsvm_data_1((char*)"./Datasets/madelon_double_10", 20000, 500);
    //sgd.b_normalize(0,  1, 7.0);     
  }
 else if (dataset_index == 9)
  {
    printf("------Training the sythesized data: (32 samples, 126 features)\n");
    sgd.load_synthesized_data(32, 126);
    //sgd.b_normalize(0,  1, 7.0);     
  }

  //printf("test_0_0, dr_numSamples = %d, dr_numFeatures = %d\n", dr_numSamples, dr_numFeatures);
  sgd.a_normalize();        // already binarize, two classifications. 
  printf("After normalization on A\n"); //sleep(1);

  sgd.b_normalize(0,  targeted_label, num_fractional_bits );  //23   //toMinus1_1, binarize_b, b_toBinarizeTo?  ^_^     


  double start, end;

  float    stepSize              = 1.0/((float)(1<<stepSizeShifter));
  //uint32_t step_size             
  //use the software implementation, without optimization...//
  if (exec_model == 0)
  {
    //float x_history1[numberOfIterations*sgd.dr_numFeatures];
    //sgd.numSamples = 100;
    //start = get_time();
    sgd.float_linreg_SGD(numberOfIterations, stepSize); //run the CPU-version SGD...
    
    //for (int i = 0; i < 10; i++)
    //  printf("x_history1[%d] = %f\n", i, x_history1[i]); 
    
    //float loss = sgd.calculate_loss(x_history1);
    //cout << "loss:" << loss <<endl;
    //sgd.bitFSGD(numberOfBits, numberOfIterations, mini_batch_size, stepSizeShifter, 0, 0.0);
    
    //end = get_time();
  }
  //Model: floating point implementation..
  else if (exec_model == 1) //
  { 
    sgd.a_perform_bitweaving_fpga();
    printf("BitWeaving a done\n"); //sleep(1);

    sgd.b_copy_to_fpga();    
    printf("BitWeaving b done\n"); //sleep(1);
  
    sgd.bitFSGD(numberOfBits, numberOfIterations, mini_batch_size, stepSizeShifter, 0, 0.0);
    //sgd.floatFSGD(numberOfIterations, stepSize, 0, 0);

    sgd.compute_loss_and_printf(numberOfIterations, num_fractional_bits);
  }  
  else if (exec_model == 2)
  {
    sgd.float_linreg_SGD_batch(numberOfIterations, stepSize, mini_batch_size); //run the CPU-version SGD...
  }


  

  // Program parameters
  uint32_t NUM_TUPLES         = 4*1024*1024;


  return 1;
}
