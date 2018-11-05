/*
 * Copyright 2017 - 2018, Zeke Wang, Systems Group, ETH Zurich
 *
 * This hardware operator is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
//The objective of the module is to compute the dot products for eight banks.
//
//Fixme: we can tune the precision of computation in this part....

`include "sgd_defines.vh"

module sgd_serial_loss (
    input   wire                                   clk,
    input   wire                                   rst_n,

    //------------------------Configuration-----------------------------//
    input   wire [31:0]                            step_size,

    //------------------Input: disptach resp data to b of each bank---------------//
    input                  [32*`NUM_OF_BANKS-1:0]  dispatch_axb_b_data, //256.
    input                                          dispatch_axb_b_wr_en, 
    //input   wire                                   dispatch_axb_b_almost_full[`NUM_OF_BANKS-1:0],

    //------------------Input: dot products for all the banks. ---------------//
    input wire signed                      [31:0] dot_product_signed[`NUM_OF_BANKS-1:0],       //
    input wire                                    dot_product_signed_valid[`NUM_OF_BANKS-1:0],  //
    //------------------Output: dot products for all the banks. ---------------//
    output reg signed                      [31:0] ax_minus_b_sign_shifted_result[`NUM_OF_BANKS-1:0],         //
    output reg                                    ax_minus_b_sign_shifted_result_valid[`NUM_OF_BANKS-1:0]
);


////////////////////////////////////////fifo for b////////////////////////////////////////
//Warning: Make sure the buffer_b has the enough space for the 

reg                           buffer_b_wr_en;     //rd
reg    [32*`NUM_OF_BANKS-1:0] buffer_b_wr_data;   //rd_data

always @(posedge clk) begin
    buffer_b_wr_en      <= dispatch_axb_b_wr_en; 
    buffer_b_wr_data    <= dispatch_axb_b_data;
end


wire                          buffer_b_rd_en;     //rd
wire   [32*`NUM_OF_BANKS-1:0] buffer_b_rd_data;   //rd_data
wire                          buffer_b_data_valid;

assign buffer_b_rd_en  = dot_product_signed_valid[0]; 
//on-chip buffer for b 
distram_fifo  #( .FIFO_WIDTH      (32*`NUM_OF_BANKS), 
                 .FIFO_DEPTH_BITS (       6        ) 
) inst_b_fifo (
    .clk        (clk),
    .reset_n    (rst_n),

    //Writing side. from sgd_dispatch...
    .we         ( buffer_b_wr_en    ),
    .din        ( buffer_b_wr_data  ),
    .almostfull (                   ), 

    //reading side.....
    .re         (buffer_b_rd_en     ),
    .dout       (buffer_b_rd_data   ),
    .valid      (buffer_b_data_valid),
    .empty      (                   ),
    .count      (                   )
);



wire signed          [31:0] buffer_b_rd_data_signed[`NUM_OF_BANKS-1:0];
reg signed           [31:0] ax_dot_product_reg[`NUM_OF_BANKS-1:0], ax_dot_product_reg2[`NUM_OF_BANKS-1:0];         //cycle synchronization. 
reg                         ax_dot_product_valid_reg[`NUM_OF_BANKS-1:0], ax_dot_product_valid_reg2[`NUM_OF_BANKS-1:0];
reg signed           [31:0] ax_minus_b_result[`NUM_OF_BANKS-1:0];          //
reg                         ax_minus_b_result_valid[`NUM_OF_BANKS-1:0];

reg signed                      [31:0] ax_minus_b_sign_shifted_result_pre[`NUM_OF_BANKS-1:0];         //
reg                                    ax_minus_b_sign_shifted_result_valid_pre[`NUM_OF_BANKS-1:0];

reg                   [4:0] num_shift_bits[`NUM_OF_BANKS-1:0];

genvar i, j;
generate for( i = 0; i < `NUM_OF_BANKS; i = i + 1) begin: inst_bank

    always @(posedge clk) begin
            num_shift_bits[i]          <= step_size[4:0]; // - 5'h0  - 5'h6
    end


    assign buffer_b_rd_data_signed[i] = buffer_b_rd_data[32*(i+1)-1:32*i];
    
    always @(posedge clk) begin
        if(~rst_n) 
        begin
            ax_minus_b_result_valid[i]              <=  1'b0;
            ax_minus_b_sign_shifted_result_valid[i] <=  1'b0;
        end
        else
        begin
            //one-cycle delay
            ax_dot_product_valid_reg[i]             <= dot_product_signed_valid[i];
            ax_dot_product_reg[i]                   <= dot_product_signed[i];
         
             //two-cycles delay         
            ax_dot_product_valid_reg2[i]            <= ax_dot_product_valid_reg[i];
            ax_dot_product_reg2[i]                  <= ax_dot_product_reg[i];
             
             //three-cycles delay         
            ax_minus_b_result_valid[i]              <= ax_dot_product_valid_reg2[i];
            ax_minus_b_result[i]                    <= ax_dot_product_reg2[i] - buffer_b_rd_data_signed[i]; // buffer_b_rd_data;

             //four-cycles delay         
            ax_minus_b_sign_shifted_result_valid_pre[i] <=  ax_minus_b_result_valid[i];
            ax_minus_b_sign_shifted_result_pre[i]       <= (ax_minus_b_result[i]>>>num_shift_bits[i] ); //should be signed reg.

             //four-cycles delay         
            ax_minus_b_sign_shifted_result_valid[i]     <= ax_minus_b_sign_shifted_result_valid_pre[i];
            ax_minus_b_sign_shifted_result[i]           <= ax_minus_b_sign_shifted_result_pre[i]; //should be signed reg.

        end 
    end



end 
endgenerate



endmodule
