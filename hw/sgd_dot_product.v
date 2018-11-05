`ifndef SGD_DOT_PRODUCT_V
`define SGD_DOT_PRODUCT_V

//WARNING: include either async or sync, not both, as both have the same module name.
//`ifdef does no work, the verilog pre-processing system sucks. Therfore, I have to add the model here, not in sgd_defines.vh
`define USE_SYNC 





`ifdef USE_SYNC
  `include "sgd_dot_product_sync.v"
`else
  `include "sgd_dot_product_async.v"
`endif



`endif