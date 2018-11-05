
#define BITS_OF_CL      512
#define NUM_BANKS       8
#define BITS_OF_BANK    (BITS_OF_CL/NUM_BANKS)

//dr_a_norm_fp  : output normalized FP table
//dr_a_norm     : output normalized INT table
//dr_a          : input  input table
//dr_numSamples : number of samples int he table. 
//dr_numFeatures: number of features in each sample

void a_normalize(float *dr_a_norm_fp, uint32_t *dr_a_norm,  float *dr_a, uint32_t dr_numSamples, uint32_t dr_numFeatures) 
{
/*
	//uint32_t *data  = reinterpret_cast<uint32_t*>( myfpga->malloc(100)); 
	dr_a_norm_fp = (float *)malloc(dr_numSamples*dr_numFeatures*sizeof(float)); 
	if (dr_a_norm_fp == NULL)
	{
		printf("Malloc dr_a_norm_fp failed in a_normalize\n");
		return;
	}

	//a_normalizedToMinus1_1 = toMinus1_1;
	dr_a_norm   = (uint32_t *)malloc(dr_numSamples*dr_numFeatures*sizeof(uint32_t)); //to store the normalized result....
	if (dr_a_norm == NULL)
	{
		printf("Malloc dr_a_norm failed in a_normalize\n");
		return;
	}
*/
	float *dr_a_min    = (float *)malloc(dr_numFeatures*sizeof(float)); //to store the minimum value of features.....
	if (dr_a_min == NULL)
	{
		printf("Malloc dr_a_min failed in a_normalize\n");
		return;
	}

	float *dr_a_max    = (float *)malloc(dr_numFeatures*sizeof(float)); //to store the miaximum value of features.....
	if (dr_a_max == NULL)
	{
		printf("Malloc dr_a_max failed in a_normalize\n");
		return;
	}

	///Normalize the values in the whole column to the range {0, 1} or {-1, 1}/// 
	for (int j = 0; j < dr_numFeatures; j++) 
	{ // Don't normalize bias
		float amin = numeric_limits<float>::max();
		float amax = numeric_limits<float>::min();
		for (int i = 0; i < dr_numSamples; i++) 
		{
			float a_here = dr_a[i*dr_numFeatures + j];
			if (a_here > amax)
				amax = a_here;
			if (a_here < amin)
				amin = a_here;
		}
		dr_a_min[j]  = amin; //set to the global variable for pm
		dr_a_max[j]  = amax;

		float arange = amax - amin;
		if (arange > 0) 
		{
			for (int i = 0; i < dr_numSamples; i++) 
			{
				float tmp = ((dr_a[i*dr_numFeatures + j] - amin)/arange); //((dr_a[i*dr_numFeatures + j] - amin)/arange)*2.0-1.0;
			  	
			  	dr_a_norm_fp[i*dr_numFeatures + j] = tmp;
			  	dr_a_norm[i*dr_numFeatures + j]    = (uint32_t) (tmp * 4294967295.0); //4294967296 = 2^32				
			}
		}
	}
}


//This function performs weaving on the input data array: src.
//Input : src  (dense, unsigned int) 	
//Output: dest (in MLWeaving)
void mlweaving_on_sample(uint32_t *dest, uint32_t *src, uint32_t numSamples, uint32_t numFeatures) 
{
	uint32_t address_index         = 0;
	///Do the bitWeaving to the training data...
	for (uint32_t i = 0; i < numSamples; i+=NUM_BANKS)
	{   
		uint32_t samples_in_batch = ( (i+NUM_BANKS)<numSamples )? NUM_BANKS:(numSamples-i); 
		// j; //Deal with the main part of numFeatures.
		for (uint32_t j = 0; j < numFeatures; j += BITS_OF_BANK)//(numFeatures/BITS_OF_BANK)*BITS_OF_BANK
		{
			uint32_t bits_in_batch = ( (j+BITS_OF_BANK)<numFeatures )? BITS_OF_BANK:(numFeatures-j); 
			uint32_t tmp_buffer[512] = {0};
			//1: initilization off tmp buffer..
			for (int k = 0; k < samples_in_batch; k++)//NUM_BANKS
				for (int m = 0; m < bits_in_batch; m++) //BITS_OF_BANK
					tmp_buffer[ k*BITS_OF_BANK+m ] = src[ (i + k)*numFeatures + (j+m) ];

			//2: focus on the data from index: j...
			for (int k = 0; k < 32; k++)
			{	
				uint32_t result_buffer[16] = {0};
				//2.1: re-order the data according to the bit-level...
				for (int m = 0; m < 512; m++)
				{
					result_buffer[m>>5] = result_buffer[m>>5] | ((tmp_buffer[m] >>31)<<(m&31));
					tmp_buffer[m]       = tmp_buffer[m] << 1;				
				}
			    //2.2: store the bit-level result back to the memory...
				dest[address_index++] = result_buffer[0];
				dest[address_index++] = result_buffer[1];
				dest[address_index++] = result_buffer[2];
				dest[address_index++] = result_buffer[3];
				dest[address_index++] = result_buffer[4];
				dest[address_index++] = result_buffer[5];
				dest[address_index++] = result_buffer[6];
				dest[address_index++] = result_buffer[7];
				dest[address_index++] = result_buffer[8];
				dest[address_index++] = result_buffer[9];
				dest[address_index++] = result_buffer[10];
				dest[address_index++] = result_buffer[11];
				dest[address_index++] = result_buffer[12];
				dest[address_index++] = result_buffer[13];
				dest[address_index++] = result_buffer[14];
				dest[address_index++] = result_buffer[15];
			}
		}
	}
}

//This function retrives the sample from the mlweaving layout with address: src. 
//dest:        destination array address
//numFeatures: number of features in a sample.  
//src :        address of mlweaving array
//index:       sample index
//num_bits:    number of bits to retrieve. 
//T:           template is used to generalize to uchar, ushort, uint.
template <typename T>
void inline retrieve_from_mlweaving(T* dest, uint32_t numFeatures, uint32_t *src, uint32_t index, uint32_t num_bits) 
{	
	//aligned number of features. 
	uint32_t numFeaturesAlign   = ( (numFeatures+(BITS_OF_BANK-1))&(~(BITS_OF_BANK-1)) ); //round up to the nearest mulitple of BITS_OF_BANK

	//calculate the address of sample of the index: index. 
	uint32_t  chunk_offset      = (index/NUM_BANKS) * numFeaturesAlign*NUM_BANKS; //chunk index * chunk size
	uint32_t  sample_offset     = (index%NUM_BANKS) * (BITS_OF_BANK/32); //chunk index * chunk size
	uint32_t* sample_addr       = src + chunk_offset + sample_offset; //identify the address of the sampel (index)

	uint32_t t_bits_min_1       = sizeof(T)*8 - 1;

	for (size_t i = 0; i < numFeatures; i++) 
	{  //prepare one number for each iteration.  
		uint32_t main_offset = ( i/BITS_OF_BANK	  ) * BITS_OF_CL; //main index * size of chunk
		uint32_t int_offset  = ( i&(BITS_OF_BANK-1) )/32;
		uint32_t bit_offset  = i & 31;

		//The next 32 CLs contains the information of the feature. 
		T result = 0;
		uint32_t tmp;
		for (uint32_t j = 0; j < num_bits; j++)
		{
							     //main		      bit	          which ints 
		  tmp	  = sample_addr[main_offset + (BITS_OF_CL/32)*j + int_offset]; //16=512/32
		  result |= ( ( (tmp&(1<<bit_offset)) >> bit_offset ) << (t_bits_min_1-j) ); //
		}
		dest[i]   = result; 
	}
}



