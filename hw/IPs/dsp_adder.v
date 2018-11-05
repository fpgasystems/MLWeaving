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
module  dsp_adder (
    input  wire                         clk_i,
    //////////////////////Input/////////////////////////////
    input  wire                 [31:0] in_1,   // input 1
    input  wire                 [31:0] in_2,   // input 2
    //////////////////////Output////////////////////////////
    output wire                 [31:0] out     // output  
);


wire  [1:0] aclr = 2'b0;
wire [15:0] ay   = in_1[31:16]; //upper part
wire [15:0] az   = in_2[31:16];

wire [15:0] by   = in_1[15:0];
wire [15:0] bz   = in_2[15:0];

wire [63:0] chainin, chainout;
wire [36:0] resulta;

wire  [2:0] clk      = {clk_i, clk_i, clk_i};
wire  [2:0] coefsela = 3'b0;
wire  [2:0] coefselb = 3'b0;
wire  [2:0] ena      = 3'b111;

assign out = resulta[31:0];

twentynm_mac        twentynm_mac_component (
                                        .aclr (aclr),
                                        .ay (ay),
                                        .az (az),
                                        .by (by),
                                        .bz (bz),
                                        .chainin (), //do not use chain.
                                        .clk (clk),
                                        .coefsela (coefsela),
                                        .coefselb (coefselb),
                                        .ena (ena),
                                        .chainout (chainout),
                                        .resulta (resulta));
    defparam
                    twentynm_mac_component.ay_scan_in_width = 16,
                    twentynm_mac_component.az_width = 16,
                    twentynm_mac_component.by_width = 16,
                    twentynm_mac_component.bz_width = 16,
                    twentynm_mac_component.operation_mode = "m18x18_sumof2",
                    twentynm_mac_component.mode_sub_location = 0,
                    twentynm_mac_component.operand_source_max = "coef",
                    twentynm_mac_component.operand_source_may = "preadder",
                    twentynm_mac_component.operand_source_mbx = "coef",
                    twentynm_mac_component.operand_source_mby = "preadder",
                    twentynm_mac_component.signed_max = "false",
                    twentynm_mac_component.signed_may = "false",
                    twentynm_mac_component.signed_mbx = "false",
                    twentynm_mac_component.signed_mby = "false",
                    twentynm_mac_component.preadder_subtract_a = "false",
                    twentynm_mac_component.preadder_subtract_b = "false",
                    twentynm_mac_component.ay_use_scan_in = "false",
                    twentynm_mac_component.by_use_scan_in = "false",
                    twentynm_mac_component.delay_scan_out_ay = "false",
                    twentynm_mac_component.delay_scan_out_by = "false",
                    twentynm_mac_component.use_chainadder = "false",
                    twentynm_mac_component.enable_double_accum = "false",
                    twentynm_mac_component.load_const_value = 0,
                    twentynm_mac_component.coef_a_0 = 65536,
                    twentynm_mac_component.coef_a_1 = 0,
                    twentynm_mac_component.coef_a_2 = 0,
                    twentynm_mac_component.coef_a_3 = 0,
                    twentynm_mac_component.coef_a_4 = 0,
                    twentynm_mac_component.coef_a_5 = 0,
                    twentynm_mac_component.coef_a_6 = 0,
                    twentynm_mac_component.coef_a_7 = 0,
                    twentynm_mac_component.coef_b_0 = 1,
                    twentynm_mac_component.coef_b_1 = 0,
                    twentynm_mac_component.coef_b_2 = 0,
                    twentynm_mac_component.coef_b_3 = 0,
                    twentynm_mac_component.coef_b_4 = 0,
                    twentynm_mac_component.coef_b_5 = 0,
                    twentynm_mac_component.coef_b_6 = 0,
                    twentynm_mac_component.coef_b_7 = 0,
                    twentynm_mac_component.ax_clock = "none",
                    twentynm_mac_component.ay_scan_in_clock = "0",
                    twentynm_mac_component.az_clock = "0",
                    twentynm_mac_component.bx_clock = "none",
                    twentynm_mac_component.by_clock = "0",
                    twentynm_mac_component.bz_clock = "0",
                    twentynm_mac_component.coef_sel_a_clock = "0",
                    twentynm_mac_component.coef_sel_b_clock = "0",
                    twentynm_mac_component.sub_clock = "none",
                    twentynm_mac_component.sub_pipeline_clock = "none",
                    twentynm_mac_component.negate_clock = "none",
                    twentynm_mac_component.negate_pipeline_clock = "none",
                    twentynm_mac_component.accumulate_clock = "none",
                    twentynm_mac_component.accum_pipeline_clock = "none",
                    twentynm_mac_component.load_const_clock = "none",
                    twentynm_mac_component.load_const_pipeline_clock = "none",
                    twentynm_mac_component.input_pipeline_clock = "none", //adder one more cycle in the pipeline.
                    twentynm_mac_component.output_clock = "0",
                    twentynm_mac_component.scan_out_width = 27,
                    twentynm_mac_component.result_a_width = 37;



endmodule


