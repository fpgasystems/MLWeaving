`ifndef SGD_DEFINES_VH
`define SGD_DEFINES_VH


//
//`define USE_ASYNC                1
//`define USE_NO_CHAINING          1             

//`define USE_ASYNC_MODEL          1


//this defination is used as the tag which identifies the rd operations
// are from a or b...
`define MEM_RD_A_TAG             8'h0a
`define MEM_RD_B_TAG             8'h0b

`define NUM_BITS_PER_CL          512


//////////////////////////Start of BANK///////////////////////////////////
`define NUM_OF_BANKS             8
`define NUM_OF_BANKS_WIDTH       3
`define NUM_BITS_PER_BANK        `NUM_BITS_PER_CL/`NUM_OF_BANKS
`define BIT_WIDTH_OF_BANK        9-`NUM_OF_BANKS_WIDTH
//////////////////////////End of Bank///////////////////////////////////

`define USE_ASYNC 1


//////////////////////////Start of Model///////////////////////////////////
`define X_BIT_DEPTH              9
`define NUM_BITS_FOR_X           32
`define MAX_BIT_WIDTH_OF_X       16 //9:depth(512), 6:width(64), 1: extra
//////////////////////////End of Model///////////////////////////////////

`define A_FIFO_DEPTH_BITS       11 //2^A_FIFO_DEPTH_BITS-depth fifo for training dataset. can not be large.


`define NUM_EXTRA_BITS_AX_B      4  //More bits for arithmic operation...

`define NUM_ALIGN_TO_BITS        64 //More bits for arithmic operation...


`endif
