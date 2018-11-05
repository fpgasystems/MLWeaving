// (C) 2001-2016 Altera Corporation. All rights reserved.
// Your use of Altera Corporation's design tools, logic functions and other 
// software and tools, and its AMPP partner logic functions, and any output 
// files any of the foregoing (including device programming or simulation 
// files), and any associated documentation or information are expressly subject 
// to the terms and conditions of the Altera Program License Subscription 
// Agreement, Altera MegaCore Function License Agreement, or other applicable 
// license agreement, including, without limitation, that your use is for the 
// sole purpose of programming logic devices manufactured by Altera and sold by 
// Altera or its authorized distributors.  Please refer to the applicable 
// agreement for further details.



// synopsys translate_off
`timescale 1 ps / 1 ps
// synopsys translate_on
module  blockram_2port #(parameter DATA_WIDTH      = 64,    
                         parameter DEPTH_BIT_WIDTH = 9
)  (
    clock,
    data,
    rdaddress,
    wraddress,
    wren,
    q);

    input    clock;
    input  [DATA_WIDTH-1:0]  data;
    input  [DEPTH_BIT_WIDTH-1:0]  rdaddress;
    input  [DEPTH_BIT_WIDTH-1:0]  wraddress;
    input    wren;
    output [DATA_WIDTH-1:0]  q;
`ifndef ALTERA_RESERVED_QIS
// synopsys translate_off
`endif
    tri1     clock;
    tri0     wren;
`ifndef ALTERA_RESERVED_QIS
// synopsys translate_on
`endif

    wire [DATA_WIDTH-1:0] sub_wire0;
    wire [DATA_WIDTH-1:0] q = sub_wire0[DATA_WIDTH-1:0];

    altera_syncram  altera_syncram_component (
                .address_a (wraddress),
                .address_b (rdaddress),
                .clock0 (clock),
                .data_a (data),
                .wren_a (wren),
                .q_b (sub_wire0),
                .aclr0 (1'b0),
                .aclr1 (1'b0),
                .address2_a (1'b1),
                .address2_b (1'b1),
                .addressstall_a (1'b0),
                .addressstall_b (1'b0),
                .byteena_a (1'b1),
                .byteena_b (1'b1),
                .clock1 (1'b1),
                .clocken0 (1'b1),
                .clocken1 (1'b1),
                .clocken2 (1'b1),
                .clocken3 (1'b1),
                .data_b ({DATA_WIDTH{1'b1}}),
                .eccencbypass (1'b0),
                .eccencparity (8'b0),
                .eccstatus (),
                .q_a (),
                .rden_a (1'b1),
                .rden_b (1'b1),
                .sclr (1'b0),
                .wren_b (1'b0));
    defparam
        altera_syncram_component.address_aclr_b          = "NONE",
        altera_syncram_component.address_reg_b           = "CLOCK0",
        altera_syncram_component.clock_enable_input_a    = "NORMAL", //BYPASS
        altera_syncram_component.clock_enable_input_b    = "NORMAL", //BYPASS
        altera_syncram_component.clock_enable_output_b   = "NORMAL", //BYPASS
        altera_syncram_component.enable_ecc              = "FALSE",
        altera_syncram_component.intended_device_family  = "Arria 10",
        altera_syncram_component.lpm_type                = "altera_syncram",
        altera_syncram_component.numwords_a              = 2**DEPTH_BIT_WIDTH,//DEPTH_BIT_WIDTH**2,
        altera_syncram_component.numwords_b              = 2**DEPTH_BIT_WIDTH,//DEPTH_BIT_WIDTH**2,
        altera_syncram_component.operation_mode          = "DUAL_PORT",
        altera_syncram_component.outdata_aclr_b          = "NONE",
        altera_syncram_component.outdata_sclr_b          = "NONE",
        altera_syncram_component.outdata_reg_b            = "CLOCK0",
        altera_syncram_component.power_up_uninitialized  = "FALSE",
        altera_syncram_component.ram_block_type          = "AUTO", //M20K
        altera_syncram_component.read_during_write_mode_mixed_ports  = "DONT_CARE",
        altera_syncram_component.widthad_a        = DEPTH_BIT_WIDTH,
        altera_syncram_component.widthad_b        = DEPTH_BIT_WIDTH,
        altera_syncram_component.width_a          = DATA_WIDTH,
        altera_syncram_component.width_b          = DATA_WIDTH,
        altera_syncram_component.width_byteena_a  = 1;


endmodule


