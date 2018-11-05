#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>       /* time */
#ifdef __linux__
    #include <malloc.h>
#endif

//#include "sample.h"
#include "../mlweaving.h"


//The related parameters are here. 

#define NUM_FEATURES          5017  //512 *3//
#define NUM_SAMPLES           6545 //512 *3//
#define NUM_FEATURES_ALIGN ( (NUM_FEATURES+63)&(~63) )
#define NUM_SAMPLES_ALIGN  ( (NUM_SAMPLES + 7)&(~7 ) )

/*
void test_short_convert_from_bitweaving()
{
  printf("===============================================================\n");
  printf("============ Test convert_from_bitweaving on short ============\n");
  printf("===============================================================\n");

  srand (time(NULL));  
  printf("RAND_MAX = 0x%x\n", RAND_MAX);

  //Set up the source vector with random integer. 
  uint32_t *data = (unsigned int *)malloc( NUM_SAMPLES*NUM_FEATURES*sizeof(uint32_t) ); //[NUM_VALUES];
  for (uint32_t i = 0; i < NUM_SAMPLES*NUM_FEATURES; i++)
    data[i] = rand(); //(float)i;
 
  //set up the mlweaving data .
  uint32_t *data_bitweaving = (unsigned int *)malloc( NUM_SAMPLES * NUM_FEATURES_ALIGN*sizeof(uint32_t) ); //[NUM_VALUES];
  for (uint32_t i = 0; i < NUM_SAMPLES*NUM_FEATURES; i++)
    data_bitweaving[i] = 0; //(float)i;    


  //meaving the data...
  hazy::vector::mlweaving_on_sample(data_bitweaving, data, NUM_SAMPLES, NUM_FEATURES);

  //testing the data...
  //Set up the destination...
  unsigned short dest[2*NUM_FEATURES];
  hazy::vector::FVector<unsigned short> dest_v (dest, NUM_FEATURES);

  for (uint32_t i = 0; i < NUM_SAMPLES; i++)   
  {
    //get the sample in fvector.
    hazy::vector::retrieve_from_mlweaving(dest_v, data_bitweaving, i, num_bits);
    for (uint32_t j = 0; j < NUM_FEATURES; j++)
    {
      if ( ( (data[i]>>16)<<0 ) != dest_v[i])
      {
        printf("Error::::::%d: src_0x%8x, dest_0x%x\n", j, data[j], dest_v[j]);
        return;
      }    
    }    
  }

  free(data);
  free(data_bitweaving);

  hazy::vector::FVector<unsigned int> src_int_vector (data_bitweaving, NUM_FEATURES);



    hazy::vector::Convert_from_bitweaving(dest_char_vector, src_int_vector, 16);

    //    sample_char.regroup_from_bitweaving(samps[i], num_bits);


    for (int i = 0; i < NUM_FEATURES; i++)
        if ( ( (data[i]>>16)<<0 ) != dest_char_vector[i])
        {
            printf("Error::::::%d: src_0x%8x, dest_0x%x\n", i, data[i], dest_char_vector[i]);
            return;
        }
        //if (data[i] != 0.0)
        //{
        //    printf("ERROR: %d, %f\n", i, data[i]);
        //    break;
        //}
    printf("Congratuation!!! Your test is passed...\n"); 
}
*/


// This funtion tests the correcteness of mlweaving: meaving and retrieving. 
// 

template <typename T>
int test_mlweaving(uint32_t *data, uint32_t *data_bitweaving, uint32_t num_bits )
{
/*
  for (uint32_t i = 0; i < NUM_SAMPLES*NUM_FEATURES_ALIGN/16; i++)
  { //printf();
    for (uint32_t j = 0; j < 16; j++)
      printf("0x%x ", data_bitweaving[i*16+j]);   

    printf("\n");   
  }
*/
  //testing the data...
  //Set up the destination...
  T dest[2*NUM_FEATURES];
  hazy::vector::FVector<T> dest_v (dest, NUM_FEATURES);

  uint32_t T_bits = 8 * sizeof(T);
  uint32_t T_mask = ~( (1<<(T_bits-num_bits)) - 1 );

  uint32_t shift_bits = 0;
  if (T_bits == 8)
    shift_bits    = 24;
  else if (T_bits == 16)
    shift_bits    = 16;

  for (uint32_t i = 0; i < NUM_SAMPLES; i++)   
  {
    //get the sample in fvector.
    hazy::vector::retrieve_from_mlweaving(dest_v, data_bitweaving, i, num_bits);
    for (uint32_t j = 0; j < NUM_FEATURES; j++)
    {
      if ( ((data[i*NUM_FEATURES+j]>>shift_bits)&T_mask) != dest_v[j] )
      {
        printf("Error: sample_%d,feature_%d: src_0x%8x, T_mask_0x%8x, dest_0x%8x\n", i, j, (data[i*NUM_FEATURES+j]>>shift_bits), T_mask, dest_v[j]);
        return -1;
      }    
    }    
  }
   return 0;
  //

  free(data);
  free(data_bitweaving);
}



void main ()
{
  printf("===============================================================\n");
  printf("===Test mlweaving:char/short/int: samples:%d, features:%d==\n", NUM_SAMPLES, NUM_FEATURES);
  printf("===============================================================\n");

  srand (time(NULL));  

  printf("RAND_MAX = 0x%x, number_features_align = %d\n", RAND_MAX, NUM_FEATURES_ALIGN);

  //Set up the source vector with random integer. 
  uint32_t *data = (unsigned int *)malloc( NUM_SAMPLES*NUM_FEATURES*sizeof(uint32_t) ); //[NUM_VALUES];
  for (uint32_t i = 0; i < NUM_SAMPLES*NUM_FEATURES; i++)
    data[i] = rand(); //(float)i;
/* 
  for (uint32_t i = 0; i < NUM_SAMPLES; i++)
  { //printf();
    for (uint32_t j = 0; j < NUM_FEATURES; j++)
      printf("0x%x ", data[i*NUM_FEATURES+j]);   

    printf("\n");   
  }
*/
  //set up the mlweaving data .
  uint32_t *data_bitweaving = (unsigned int *)malloc( NUM_SAMPLES_ALIGN * NUM_FEATURES_ALIGN*sizeof(uint32_t) ); //[NUM_VALUES];
  for (uint32_t i = 0; i < NUM_SAMPLES_ALIGN*NUM_FEATURES_ALIGN; i++)
    data_bitweaving[i] = 0; //(float)i;    

  //printf("before mlweaving::: \n");
  //meaving the data...
  hazy::vector::mlweaving_on_sample(data_bitweaving, data, NUM_SAMPLES, NUM_FEATURES);

  //printf("after mlweaving::: \n");  


  //uint32_t num_bits  = 8;
  for (uint32_t num_bits = 1; num_bits < 8; num_bits++)
  {
    int flag = test_mlweaving<uint8_t>(data, data_bitweaving, num_bits );
    if (flag != 0)
    {
      printf("Error happens at char for bits: %d", num_bits);
      return;      
    }
  } 
  printf("Congratuation!!! Your test on char is passed...\n"); 

  for (uint32_t num_bits = 1; num_bits < 16; num_bits++)
  {
    int flag = test_mlweaving<uint16_t>(data, data_bitweaving, num_bits );
    if (flag != 0)
    {
      printf("Error happens at short for bits: %d", num_bits);
      return;      
    }
  } 
  printf("Congratuation!!! Your test on short is passed...\n"); 

  for (uint32_t num_bits = 1; num_bits < 32; num_bits++)
  {
    int flag = test_mlweaving<uint32_t>(data, data_bitweaving, num_bits );
    if (flag != 0)
    {
      printf("Error happens at int for bits: %d", num_bits);
      return;      
    }
  } 
  printf("Congratuation!!! Your test on int is passed...\n"); 



  //test_num_CLs();
  //test_pow_of_2();
  //test_srv();
  //test_blend();
  //test_permute();
  //test_norm();
}
