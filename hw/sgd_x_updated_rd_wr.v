/*
 * Copyright 2017 - 2018 Systems Group, ETH Zurich
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
//The objective of the module is to manage the read/write ports of x_updated.
//When: when the acc_gradient_valid is valid, x_updated_rd_data should be valid 
//      at the same cycle, so v_input_valid is used to increment the x_updated_rd_addr.
//How:  only need the 


`include "sgd_defines.vh"

module sgd_x_updated_rd_wr #( parameter DATA_WIDTH_IN      = 4 ,
                     parameter MAX_DIMENSION_BITS = `MAX_BIT_WIDTH_OF_X  ) ( //16
    input   wire                                   clk,
    input   wire                                   rst_n,
    //--------------------------Begin/Stop-----------------------------//
    input   wire                                   started,
    input   wire [31:0]                            dimension,

    input   wire signed                     [31:0] acc_gradient[`NUM_BITS_PER_BANK-1:0],
    input   wire                                   acc_gradient_valid[`NUM_BITS_PER_BANK-1:0],

    ///////////////////rd part of x_updated//////////////////////
    output  reg                [`X_BIT_DEPTH-1:0]  x_updated_rd_addr,
    input             [`NUM_BITS_PER_BANK*32-1:0]  x_updated_rd_data,
    ///////////////////wr part of x_updated//////////////////////
    output  reg                                    x_updated_wr_en,     
    output  reg                [`X_BIT_DEPTH-1:0]  x_updated_wr_addr,
    output  reg       [`NUM_BITS_PER_BANK*32-1:0]  x_updated_wr_data
);

/////////////////Parameters/////////////////
reg  [31:0] main_counter_minus_1, main_counter_wire;
always @(posedge clk) 
begin        
    main_counter_wire     <= dimension[31 :`BIT_WIDTH_OF_BANK] + (dimension[`BIT_WIDTH_OF_BANK-1:0] != 0);
    main_counter_minus_1  <= main_counter_wire - 1'b1;
end

/////////////////Enable to increase the rd_addr/////////////////
wire x_updated_rd_valid_en;
assign x_updated_rd_valid_en   = acc_gradient_valid[0];

/////////////////Increase the rd_addr when enable signal is valid/////////////////
always @(posedge clk) 
begin 
    if(~rst_n)
    begin
        x_updated_rd_addr         <= { `X_BIT_DEPTH{1'b0} };
    end 
    else 
    begin
        if (x_updated_rd_valid_en)
        begin
            x_updated_rd_addr     <= x_updated_rd_addr + 1'b1;
            if (x_updated_rd_addr == main_counter_minus_1)
                x_updated_rd_addr <= { `X_BIT_DEPTH{1'b0} };    
        end
    end 
end


genvar i; 
//////////////////////////////////1st cycle//////////////////////////////////
reg signed             [31:0] acc_gradient_reg[`NUM_BITS_PER_BANK-1:0];
reg        [`X_BIT_DEPTH-1:0] x_updated_rd_addr_reg;
reg                           x_updated_rd_valid_en_reg;
    always @(posedge clk) 
    begin 
        x_updated_rd_addr_reg          <= x_updated_rd_addr;
        x_updated_rd_valid_en_reg      <= x_updated_rd_valid_en;  
    end
generate 
for( i = 0; i < `NUM_BITS_PER_BANK; i = i + 1) 
begin: first_cycle
    always @(posedge clk) 
    begin 
        acc_gradient_reg[i]            <= acc_gradient[i];  
    end
end
endgenerate

//////////////////////////////////2nd cycle//////////////////////////////////
reg signed             [31:0] acc_gradient_reg2[`NUM_BITS_PER_BANK-1:0];
wire signed            [31:0] x_update_rd_data_signed_2[`NUM_BITS_PER_BANK-1:0];
reg        [`X_BIT_DEPTH-1:0] x_updated_rd_addr_reg2;
reg                           x_updated_rd_valid_en_reg2;
    always @(posedge clk) 
    begin 
        x_updated_rd_addr_reg2         <= x_updated_rd_addr_reg;
        x_updated_rd_valid_en_reg2     <= x_updated_rd_valid_en_reg;  
    end
generate 
for( i = 0; i < `NUM_BITS_PER_BANK; i = i + 1) 
begin: second_cycle
    assign x_update_rd_data_signed_2[i] = x_updated_rd_data[(i+1)*32-1: i*32];//data from bram.
    always @(posedge clk) 
    begin 
        acc_gradient_reg2[i]           <= acc_gradient_reg[i];  
    end
end
endgenerate

//////////////////////////////////3-th cycle//////////////////////////////////
reg        [`X_BIT_DEPTH-1:0] x_updated_rd_addr_reg3;
reg signed             [31:0] x_update_wr_data_signed_reg3[`NUM_BITS_PER_BANK-1:0];
reg                           x_updated_rd_valid_en_reg3;
    always @(posedge clk) 
    begin 
        x_updated_rd_addr_reg3         <= x_updated_rd_addr_reg2;  
        x_updated_rd_valid_en_reg3     <= x_updated_rd_valid_en_reg2;  
    end
generate 
for( i = 0; i < `NUM_BITS_PER_BANK; i = i + 1) 
begin: third_cycle
    always @(posedge clk) 
    begin 
        x_update_wr_data_signed_reg3[i]<= x_update_rd_data_signed_2[i] - acc_gradient_reg2[i]; 
    end
end
endgenerate

//////////////////////////////////Output//////////////////////////////////
    //addr, en.
//    assign x_updated_wr_addr                   = x_updated_rd_addr_reg3;
//    assign x_updated_wr_en                     = x_updated_rd_valid_en_reg3;
    always @(posedge clk) 
    begin 
        x_updated_wr_addr    <= x_updated_rd_addr_reg3; 
        x_updated_wr_en      <= x_updated_rd_valid_en_reg3;
    end
generate 
for( i = 0; i < `NUM_BITS_PER_BANK; i = i + 1) 
begin: output_data
    always @(posedge clk) 
    begin 
        x_updated_wr_data[(i+1)*32-1: i*32]   <= x_update_wr_data_signed_reg3[i];  
    end    
//    assign x_updated_wr_data[(i+1)*32-1: i*32] = x_update_wr_data_signed_reg3[i];
end
endgenerate


endmodule
