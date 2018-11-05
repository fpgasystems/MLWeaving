`ifndef SGD_X_WR_V
`define SGD_X_WR_V

//WARNING: include either with chaining or wo chaining, not both, as both have the same module name.
`define USE_WITH_CHAINING 





`ifdef USE_WITH_CHAINING
  `include "sgd_x_wr_with_chaining.v"
`else
  `include "sgd_x_wr_without_chaining.v"
`endif



`endif