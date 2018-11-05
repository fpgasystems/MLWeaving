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
//The objective of the module is to seperate the data fro a and b.  
// 

`include "sgd_defines.vh"

module sgd_dispatch (
    input   wire                                   clk,
    input   wire                                   rst_n,
    //--------------------------Begin/Stop-----------------------------//
    input   wire                                   started,

    output  reg  [31:0]                            state_counters_dispatch,

    //---------------------Input: External Memory rd response-----------------//
    // RX RD response
    input   wire [7:0]                             um_rx_rd_tag,
    input   wire [511:0]                           um_rx_data,
    input   wire                                   um_rx_rd_valid,
    output  wire                                   um_rx_rd_ready,

    //banks = 8, bits_per_bank=64...
    //------------------Output: disptach resp data to a ofeach bank---------------//
    output  reg [`NUM_BITS_PER_BANK*`NUM_OF_BANKS-1:0] dispatch_axb_a_data, 
    output  reg                                        dispatch_axb_a_wr_en, 
    input   wire                                       dispatch_axb_a_almost_full, //only one of them is used to control...

    //------------------Output: disptach resp data to b of each bank---------------//
    output  reg                 [32*`NUM_OF_BANKS-1:0] dispatch_axb_b_data, 
    output  reg                                        dispatch_axb_b_wr_en
    //input   wire                                     dispatch_axb_b_almost_full[`NUM_OF_BANKS-1:0],
);

reg         started_r;
always @(posedge clk) begin
    if(~rst_n) 
    begin
        started_r <= 1'b0;
    end 
    else if (started) 
    begin
        started_r <= 1'b1;
    end
end

reg [31:0] wr_a_counter, wr_b_counter;
always @(posedge clk) begin
    if(~rst_n) 
    begin
        wr_a_counter <= 32'b0;
    end 
    else if (dispatch_axb_a_wr_en) 
    begin
        wr_a_counter <= wr_a_counter + 32'b1;
    end
end

always @(posedge clk) begin
    if(~rst_n) 
    begin
        wr_b_counter <= 32'b0;
    end 
    else if (dispatch_axb_b_wr_en) 
    begin
        wr_b_counter <= wr_b_counter + 32'b1;
    end
end



reg tmp_ready, compute_unit_full;

//output: accumulating the gradient, output to the gradient tree...
//assign state_counters_dispatch = {wr_b_counter[15:0], wr_a_counter[15:0]};
always @(posedge clk) begin
    if(~rst_n) 
        state_counters_dispatch <= 32'b0;
    else 
        state_counters_dispatch <= state_counters_dispatch + {31'b0, compute_unit_full};
end

always @(posedge clk) 
begin
    compute_unit_full <=  dispatch_axb_a_almost_full;
    tmp_ready         <= ~dispatch_axb_a_almost_full;
end

assign um_rx_rd_ready = tmp_ready; //~dispatch_axb_a_almost_full[0]; //Can insert one register between them. 


///////////////////////////////////////////////////////////////////////////////////////////////////
//------------------Output: disptach resp data to b ofeach bank---------------//
//We donot check the avaiability of the buffer for b, since we assume it has.
///////////////////////////////////////////////////////////////////////////////////////////////////
reg   [1:0] state_b; 
reg [511:0] mem_b_buffer;
reg         mem_b_received_en;
reg   [2:0] mem_b_index, mem_b_addr;
/////////FSM: generate  m_b_index, mem_b_received_en and mem_b_buffer.
parameter MEM_B_WRITING_COUNTER = 512/(32*`NUM_OF_BANKS);
localparam [1:0]
        RE_B_IDLE_STATE       = 2'b00,
        RE_B_POOLING_STATE    = 2'b01,
        RE_B_WRITING_STATE    = 2'b10,
        RE_B_END_STATE        = 2'b11;
//////////////////////////////   Finite State Machine: for b    ///////////////////////////////////
always@(posedge clk) begin
    if(~rst_n) 
    begin
        state_b                  <= RE_B_IDLE_STATE;
        //mem_b_buffer             <= 512'b0;
        mem_b_received_en        <= 1'b0;
        //mem_b_index              <= 3'b0;
        //mem_b_addr               <= 3'b0;
    end 
    else 
    begin
        mem_b_received_en        <= 1'b0;
        case (state_b)
            //This state is the beginning of  
            RE_B_IDLE_STATE: 
            begin 
                if(started_r)  // started with one cycle later...
                    state_b        <= RE_B_POOLING_STATE;  
            end

            /* This state is just polling the arriving of data for b.*/
            RE_B_POOLING_STATE: 
            begin
                mem_b_index       <= 3'b0;
                mem_b_addr        <= 3'b0;
                /* It also registers the parameters for the FSM*/
                if (um_rx_rd_valid & tmp_ready & (um_rx_rd_tag == `MEM_RD_B_TAG) )
                begin
                    mem_b_buffer  <= um_rx_data;
                    state_b         <= RE_B_WRITING_STATE;  
                end
            end

            RE_B_WRITING_STATE:
            begin
                mem_b_received_en <= 1'b1;
                mem_b_addr        <= mem_b_addr  + {2'b0, mem_b_received_en};
                mem_b_index       <= mem_b_index + 3'b1;

                if (mem_b_index == (MEM_B_WRITING_COUNTER-1))
                    state_b         <= RE_B_POOLING_STATE;  
            end
        endcase 
    end 
end

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////Output///////////////////////////////////////////////////////
//------------------aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa---------------//
always @(posedge clk) 
begin 
    if(~rst_n) 
    begin
        dispatch_axb_a_wr_en      <= 1'b0 ;        
    end 
    else 
    begin  //I do not know whether this implementation works or not...
        dispatch_axb_a_wr_en      <= um_rx_rd_valid & tmp_ready & (um_rx_rd_tag == `MEM_RD_A_TAG);
        dispatch_axb_a_data       <= um_rx_data;
    end
end
//------------------bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb---------------//
always @(posedge clk) 
begin 
    if(~rst_n) 
    begin
        //dispatch_axb_b_data[i]       <= 32'h0;
        dispatch_axb_b_wr_en      <= 1'b0 ;        
    end 
    else 
    begin  
        dispatch_axb_b_wr_en      <= mem_b_received_en;
        if (mem_b_addr == 3'b0)
            dispatch_axb_b_data   <= mem_b_buffer[32*`NUM_OF_BANKS-1:0]; //255:0
        else //if (mem_b_addr == 3'b1)
            dispatch_axb_b_data   <= mem_b_buffer[2*32*`NUM_OF_BANKS-1:32*`NUM_OF_BANKS];
    end
end


endmodule
