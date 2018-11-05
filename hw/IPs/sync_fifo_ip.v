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

///Desciption: synchronous fifo with the same clock.
//The overflow/underflow checking is always on, meaning that read to empty fifo or writing to full fifo is ingored.
//Wr_en: check the almost full signal to determine whether to write to the fifo.
//Rd_en: set ~empty, when empty and rd_en are valid at the same cycle, re-send the rd_en... 
//Rd_data: it is valid after two cycles delays of rd_en. 

// synopsys translate_off
`timescale 1 ps / 1 ps
// synopsys translate_on
module  sync_fifo_ip #(
    parameter FIFO_WIDTH                = 64,
    parameter FIFO_DEPTH_BITS           = 9,
    parameter FIFO_ALMOSTFULL_THRESHOLD = 2**FIFO_DEPTH_BITS - 8
) (
    input  wire                         clk,
    input  wire                         reset_n,
    //////////////////////wr////////////////////////////
    input  wire                         we,              // input   write enable
    input  wire      [FIFO_WIDTH - 1:0] din,             // input   write data with configurable width
    output wire                         full, 
    output wire                         almostfull,      // output  configurable programmable full/ almost full    
    //////////////////////rd////////////////////////////
    input  wire                         re,              // input   read enable    
    output reg                          valid,           // dout valid
    output wire      [FIFO_WIDTH - 1:0] dout,            // output  read data with configurable width    
    output wire                         empty
);

    reg valid_r1; //, valid_r2;
    always @(posedge clk) //  or negedge reset_n
    begin
        if (~reset_n) 
        begin
            valid_r1             <= 1'b0;
            valid                <= 1'b0;
        end
        else
        begin
            valid_r1             <= re & (~empty); //re;
            valid                <= valid_r1;
        end
    end

    scfifo  scfifo_component (
                .sclr        (~reset_n),
                .aclr        (1'b0),
                .clock       (clk),

                //write_part
                .data        (din),
                .wrreq       (we),
                .almost_full (almostfull),
                .full        (full),

                //read_part
                .rdreq       (re),
                .q           (dout),
                .empty       (empty),
                .almost_empty(),

                .usedw       (),
                .eccstatus   ()
            );
    defparam
        scfifo_component.add_ram_output_register  = "ON", //one more clock cycle latency...
        scfifo_component.almost_full_value        = FIFO_ALMOSTFULL_THRESHOLD,
        scfifo_component.enable_ecc               = "FALSE",
        scfifo_component.intended_device_family   = "Arria 10",
        scfifo_component.lpm_numwords             = 2**FIFO_DEPTH_BITS,
        scfifo_component.lpm_showahead            = "OFF",
        scfifo_component.lpm_type                 = "scfifo",
        scfifo_component.lpm_width                = FIFO_WIDTH,
        scfifo_component.lpm_widthu               = FIFO_DEPTH_BITS,
        scfifo_component.overflow_checking        = "ON", //Invalidate the transaction (write data to the full fifo.)
        scfifo_component.underflow_checking       = "ON", //Invalidate the transaction (read data to the empty fifo.)
        scfifo_component.use_eab                  = "ON"; //Use on-chip memory...



endmodule


