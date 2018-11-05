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


//three cycles latency...
// synopsys translate_off
`timescale 1 ps / 1 ps
// synopsys translate_on
module  dsp_adder_4 (
    input  wire                         clk_i,
    //////////////////////Input/////////////////////////////
    input  wire                 [31:0] in_1,   // input 1
    input  wire                 [31:0] in_2,   // input 2

    input  wire                 [31:0] in_3,   // input 1
    input  wire                 [31:0] in_4,   // input 2    
    //////////////////////Output////////////////////////////

    output wire                 [31:0] out     // output  
);


wire  [1:0] aclr = 2'b0;
wire [15:0] ay   = in_1[31:16]; //upper part
wire [15:0] az   = in_2[31:16];

wire [15:0] by   = in_1[15:0];
wire [15:0] bz   = in_2[15:0];

wire [63:0] chain_in_out;
wire [31:0] resulta;

wire  [2:0] clk      = {clk_i, clk_i, clk_i};
wire  [2:0] coefsela = 3'b0;
wire  [2:0] coefselb = 3'b0;
wire  [2:0] ena      = 3'b111;

assign out = resulta[31:0];
//cahinout to dsp 1...
twentynm_mac   dsp_0 (
     .aclr (aclr),
     .ay   (ay),
     .az   (az),
     .by   (by),
     .bz   (bz),
     .chainin (), //do not use chain.
     .clk      (clk),
     .coefsela (coefsela),
     .coefselb (coefselb),
     .ena      (ena),
     .chainout (chain_in_out),
     .resulta  ()
     );
    defparam
          dsp_0.ay_scan_in_width   = 16,
          dsp_0.az_width           = 16,
          dsp_0.by_width           = 16,
          dsp_0.bz_width           = 16,
          dsp_0.operation_mode     = "m18x18_sumof2",
          dsp_0.mode_sub_location  = 0,
          dsp_0.operand_source_max = "coef",
          dsp_0.operand_source_may = "preadder",
          dsp_0.operand_source_mbx = "coef",
          dsp_0.operand_source_mby = "preadder",
          dsp_0.signed_max         = "false",
          dsp_0.signed_may         = "false",
          dsp_0.signed_mbx         = "false",
          dsp_0.signed_mby         = "false",
          dsp_0.preadder_subtract_a = "false",
          dsp_0.preadder_subtract_b = "false",
          dsp_0.ay_use_scan_in      = "false",
          dsp_0.by_use_scan_in      = "false",
          dsp_0.delay_scan_out_ay   = "false",
          dsp_0.delay_scan_out_by   = "false",
          //dsp_0.chainout_enable     = "true", //for chaining.
          dsp_0.use_chainadder      = "false",
          dsp_0.enable_double_accum = "false",
          dsp_0.load_const_value = 0,
          dsp_0.coef_a_0 = 65536,
          dsp_0.coef_a_1 = 0,
          dsp_0.coef_a_2 = 0,
          dsp_0.coef_a_3 = 0,
          dsp_0.coef_a_4 = 0,
          dsp_0.coef_a_5 = 0,
          dsp_0.coef_a_6 = 0,
          dsp_0.coef_a_7 = 0,
          dsp_0.coef_b_0 = 1,
          dsp_0.coef_b_1 = 0,
          dsp_0.coef_b_2 = 0,
          dsp_0.coef_b_3 = 0,
          dsp_0.coef_b_4 = 0,
          dsp_0.coef_b_5 = 0,
          dsp_0.coef_b_6 = 0,
          dsp_0.coef_b_7 = 0,
          dsp_0.ax_clock                  = "none",
          dsp_0.ay_scan_in_clock          = "0",
          dsp_0.az_clock                  = "0",
          dsp_0.bx_clock                  = "none",
          dsp_0.by_clock                  = "0",
          dsp_0.bz_clock                  = "0",
          dsp_0.coef_sel_a_clock          = "0",
          dsp_0.coef_sel_b_clock          = "0",
          dsp_0.sub_clock                 = "none",
          dsp_0.sub_pipeline_clock        = "none",
          dsp_0.negate_clock              = "none",
          dsp_0.negate_pipeline_clock     = "none",
          dsp_0.accumulate_clock          = "none",
          dsp_0.accum_pipeline_clock      = "none",
          dsp_0.load_const_clock          = "none",
          dsp_0.load_const_pipeline_clock = "none",
          dsp_0.input_pipeline_clock      = "none", //adder one more cycle in the pipeline.
          dsp_0.output_clock              = "0",
          dsp_0.scan_out_width            = 27,
          dsp_0.result_a_width            = 32;


wire [15:0] ay_1   = in_3[31:16]; //upper part
wire [15:0] az_1   = in_4[31:16];

wire [15:0] by_1   = in_3[15:0];
wire [15:0] bz_1   = in_4[15:0];

//chainin
twentynm_mac   dsp_1 (
               .aclr     (aclr),
               .ay       (ay_1),
               .az       (az_1),
               .by       (by_1),
               .bz       (bz_1),
               .chainin  (chain_in_out), //do not use chain.
               .clk      (clk),
               .coefsela (coefsela),
               .coefselb (coefselb),
               .ena      (ena),
               .chainout (),
               .resulta  (resulta)
               );
    defparam
          dsp_1.ay_scan_in_width = 16,
          dsp_1.az_width = 16,
          dsp_1.by_width = 16,
          dsp_1.bz_width = 16,
          dsp_1.operation_mode = "m18x18_sumof2",
          dsp_1.mode_sub_location = 0,
          dsp_1.operand_source_max = "coef",
          dsp_1.operand_source_may = "preadder",
          dsp_1.operand_source_mbx = "coef",
          dsp_1.operand_source_mby = "preadder",
          dsp_1.signed_max = "false",
          dsp_1.signed_may = "false",
          dsp_1.signed_mbx = "false",
          dsp_1.signed_mby = "false",
          dsp_1.preadder_subtract_a = "false",
          dsp_1.preadder_subtract_b = "false",
          dsp_1.ay_use_scan_in      = "false",
          dsp_1.by_use_scan_in      = "false",
          dsp_1.delay_scan_out_ay   = "false",
          dsp_1.delay_scan_out_by   = "false",
          //dsp_1.chainout_enable     = "false", //for chaining.
          dsp_1.use_chainadder      = "true",
          dsp_1.enable_double_accum = "false",
          dsp_1.load_const_value    = 0,
          dsp_1.coef_a_0 = 65536,
          dsp_1.coef_a_1 = 0,
          dsp_1.coef_a_2 = 0,
          dsp_1.coef_a_3 = 0,
          dsp_1.coef_a_4 = 0,
          dsp_1.coef_a_5 = 0,
          dsp_1.coef_a_6 = 0,
          dsp_1.coef_a_7 = 0,
          dsp_1.coef_b_0 = 1,
          dsp_1.coef_b_1 = 0,
          dsp_1.coef_b_2 = 0,
          dsp_1.coef_b_3 = 0,
          dsp_1.coef_b_4 = 0,
          dsp_1.coef_b_5 = 0,
          dsp_1.coef_b_6 = 0,
          dsp_1.coef_b_7 = 0,
          dsp_1.ax_clock         = "none",
          dsp_1.ay_scan_in_clock = "0",
          dsp_1.az_clock         = "0",
          dsp_1.bx_clock         = "none",
          dsp_1.by_clock         = "0",
          dsp_1.bz_clock         = "0",
          dsp_1.coef_sel_a_clock = "0",
          dsp_1.coef_sel_b_clock = "0",
          dsp_1.sub_clock                 = "none",
          dsp_1.sub_pipeline_clock        = "none",
          dsp_1.negate_clock              = "none",
          dsp_1.negate_pipeline_clock     = "none",
          dsp_1.accumulate_clock          = "none",
          dsp_1.accum_pipeline_clock      = "none",
          dsp_1.load_const_clock          = "none",
          dsp_1.load_const_pipeline_clock = "none",
          dsp_1.input_pipeline_clock      = "0", //adder one more cycle in the pipeline.
          dsp_1.output_clock              = "0",
          dsp_1.scan_out_width            = 27,
          dsp_1.result_a_width            = 32;

endmodule


