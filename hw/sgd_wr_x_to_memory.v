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
//The objective of the module sgd_mem_rd is to generate the memory read request for the SGD computing task...
// (number_of_epochs, number_of_samples). Memory traffic: ((features+63)/64) * bits * (samples/8). 
// It is independent of the computing pipeline since the training dataset is not changed during the training...
//
// The reason for stalling is that um_tx_rd_ready is not asserted. 
// The back pressure is from the signal um_rx_rd_ready, whose negative value can cause um_tx_rd_ready to be 0.
// The batch size should be a multiple of #Banks, i.e., 8. 


`include "sgd_defines.vh"

module sgd_wr_x_to_memory #( parameter DATA_WIDTH_IN      = 4 ,
                     parameter MAX_DIMENSION_BITS = `MAX_BIT_WIDTH_OF_X  ) ( //16
    input   wire                                   clk,
    input   wire                                   rst_n,
    //--------------------------Begin/Stop-----------------------------//
    input   wire                                   started,
    output  wire [31:0]                            state_counters_wr_x_to_memory,

    //---------Input: Parameters (where, how many) from the root module-------//
    input   wire [57:0]                            addr_model,

    //input   wire [63:0]                            addr_model,
    input   wire [31:0]                            dimension,

    input                                          writing_x_to_host_memory_en,
    output  reg                                    writing_x_to_host_memory_done,

    ///////////////////rd part of x_updated//////////////////////
    output  reg                [`X_BIT_DEPTH-1:0]  x_mem_rd_addr,
    input             [`NUM_BITS_PER_BANK*32-1:0]  x_mem_rd_data,

    //---------------------Memory Inferface:write----------------------------//
    // TX WR
    output  reg  [57:0]                            um_tx_wr_addr,
    output  reg  [7:0]                             um_tx_wr_tag,
    output  reg                                    um_tx_wr_valid,
    output  reg  [511:0]                           um_tx_data,
    input   wire                                   um_tx_wr_ready
);
//From parameters from sgd_defines.vh:::

parameter MAX_BURST_BITS = MAX_DIMENSION_BITS - 9; //7..... Each chunk contains 512 features...

//to make sure that the parameters has been assigned...
reg       started_r, started_r2, started_r3;   //one cycle delay from started...
reg [2:0] state; 
reg [3:0] error_state; //0000: ok; 0001: dimension is zero; 


always @(posedge clk) begin
    if(~rst_n)
    begin
        started_r  <= 1'b0;
        started_r2 <= 1'b0;
        started_r3 <= 1'b0; //1'b0;
    end 
    else //if (started) 
    begin
        started_r  <= started;   //1'b0;
        started_r2 <= started_r; //1'b0;
        started_r3 <= started_r2; //1'b0;
    end 
end

reg               [11:0] main_counter, main_counter_minus_1, main_index;
reg               [31:0] main_counter_wire;

reg               [31:0] addr_model_index;
reg               [57:0] addr_model_for_epoch, addr_model_reg;

reg  [`NUM_BITS_PER_BANK*32-1:0]  x_mem_rd_data_reg;
reg                      writing_x_to_host_memory_en_r, writing_x_to_host_memory_en_r2, writing_x_to_host_memory_en_r3, writing_x_to_host_memory_en_r4;

reg  rd_data_reg_avail;
reg  rd_en;
reg  rd_data_pre_valid, rd_data_valid, rd_data_reg_valid;

always @(posedge clk) begin
    if(~rst_n)
    begin
        //x_mem_rd_data_reg               <= 1'h0;

        //addr_model_reg                  <= 58'b0;
        //main_counter_wire               <= 32'h0;
        writing_x_to_host_memory_en_r   <= 1'b0;
        writing_x_to_host_memory_en_r2  <= 1'b0;
        writing_x_to_host_memory_en_r3  <= 1'b0;
        writing_x_to_host_memory_en_r4  <= 1'b0;
    end 
    else 
    begin
        if (rd_data_valid)
            x_mem_rd_data_reg           <= x_mem_rd_data;

        addr_model_reg                  <= addr_model;
        main_counter_wire               <= dimension[31 :`BIT_WIDTH_OF_BANK] + (dimension[`BIT_WIDTH_OF_BANK-1:0] != 0);

        main_counter_minus_1            <= main_counter_wire - 1'b1; //MAX_BURST_BITS'h1;


        writing_x_to_host_memory_en_r   <= writing_x_to_host_memory_en;
        writing_x_to_host_memory_en_r2  <= writing_x_to_host_memory_en_r;
        writing_x_to_host_memory_en_r3  <= writing_x_to_host_memory_en_r2;
        writing_x_to_host_memory_en_r4  <= writing_x_to_host_memory_en_r3;
    end 
end

always @(posedge clk) begin
    if(~rst_n)
    begin
        rd_data_pre_valid               <= 1'b0;
        rd_data_valid                   <= 1'b0;
    end 
    else 
    begin
        rd_data_pre_valid               <= rd_en;    //determined in FSM...
        rd_data_valid                   <= rd_data_pre_valid;
    end 
end

//////////////////////////////////////////////////////////////////////////////
///////prepare the output configuration....
//////////////////////////////////////////////////////////////////////////////
always @(posedge clk) begin
    if(~rst_n)
    begin
        //rd_en                           <= 1'b0;
        rd_data_reg_valid               <= 1'b0;
    end 
    else 
    begin
        //rd_en                           <= 1'b0;
        //if ( (~writing_x_to_host_memory_en_r2) & writing_x_to_host_memory_en_r ) //capturing rising edge.
        //    rd_en                       <= 1'b1;

        if (rd_data_valid)                            //from rd_en...  one cycle before...
            rd_data_reg_valid           <= 1'b1;
        else if ( rd_data_reg_valid & um_tx_wr_ready )
        begin
            rd_data_reg_valid           <= 1'b0;
        //    rd_en                       <= 1'b1;
        end
    end 
end



//////////////////////////////////////////////////////////////////////////////
//////////////////////////Generate the output...//////////////////////////////
//////////////////////////////////////////////////////////////////////////////

reg  [57:0]                            um_tx_wr_addr_pre, um_tx_wr_addr_pre2;
reg  [7:0]                             um_tx_wr_tag_pre,  um_tx_wr_tag_pre2;
reg                                    um_tx_wr_valid_pre, um_tx_wr_valid_pre2;
reg  [511:0]                           um_tx_data_pre, um_tx_data_pre2;


reg   [1:0] inner_index;
reg  [31:0] model_offset;

always @(posedge clk) begin
    if(~rst_n)
    begin
        inner_index               <=  2'b0;
        model_offset              <= 32'b0;

        //um_tx_wr_addr             <=  58'h0;
        //um_tx_wr_tag              <=   8'h0;
        um_tx_wr_valid            <=   1'b0;
        //um_tx_data                <= 512'h0;

        //um_tx_wr_addr_pre         <=  58'h0;
        //um_tx_wr_tag_pre          <=   8'h0;
        um_tx_wr_valid_pre        <=   1'b0;
        //um_tx_data_pre            <= 512'h0;


        //um_tx_wr_addr_pre2        <=  58'h0;
        //um_tx_wr_tag_pre2         <=   8'h0;
        um_tx_wr_valid_pre2       <=   1'b0;
        //um_tx_data_pre2           <= 512'h0;
    end 
    else 
    begin
        um_tx_wr_valid_pre             <=   1'b0; //um_tx_wr_valid
        //send out four memory writing transactions for each rd_en;
        if ( (rd_data_reg_valid & um_tx_wr_ready) | ( writing_x_to_host_memory_en_r4&(inner_index != 2'b0) ) ) 
        begin
            inner_index                <= inner_index  + 2'b1;
            model_offset               <= model_offset + 1'b1;

            um_tx_wr_addr_pre          <=  addr_model_reg + model_offset; //um_tx_wr_addr
            um_tx_wr_tag_pre           <=   8'h7; //um_tx_wr_tag
            um_tx_wr_valid_pre         <=   1'b1; //um_tx_wr_valid

            if (inner_index      == 2'b00)   //um_tx_data
                um_tx_data_pre         <= x_mem_rd_data_reg[511:0];//512'h0;
            else if (inner_index == 2'b01)
                um_tx_data_pre         <= x_mem_rd_data_reg[512+511:512];//512'h0;
            else if (inner_index == 2'b10)
                um_tx_data_pre         <= x_mem_rd_data_reg[1024+511:1024];//512'h0;
            else
                um_tx_data_pre         <= x_mem_rd_data_reg[1024+512+511:1024+512];//512'h0;
        end

        //um_tx_wr_addr_pre2             <=  um_tx_wr_addr_pre;
        //um_tx_wr_tag_pre2              <=  um_tx_wr_tag_pre;
        //um_tx_wr_valid_pre2            <=  um_tx_wr_valid_pre;
        //um_tx_data_pre2                <=  um_tx_data_pre;

        um_tx_wr_addr                  <=  um_tx_wr_addr_pre;  //2;
        um_tx_wr_tag                   <=  um_tx_wr_tag_pre;   //2;
        um_tx_wr_valid                 <=  um_tx_wr_valid_pre; //2;
        um_tx_data                     <=  um_tx_data_pre;     //2;
    end 
end



///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////       Finite State Machine      ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
localparam [2:0]
        WR_MEM_IDLE_STATE          = 3'b000,
        WR_MEM_STARTING_STATE      = 3'b001,
        WR_MEM_POLL_STATE          = 3'b010,        
        WR_MEM_IN_PROGRESS_STATE   = 3'b011,

        WR_MEM_END_STATE           = 3'b100;

always@(posedge clk) begin
    if(~rst_n) 
     begin
        state                    <=  WR_MEM_IDLE_STATE;
        //error_state              <=  4'b0;

        //rd_data_reg_avail        <=  1'b0;
        //rd_en                    <=  1'b0;

        //main_counter_minus_1     <=     0; 
        //main_index               <=     0; 
        //x_mem_rd_addr            <= 1'b0;

        //writing_x_to_host_memory_done <= 1'h0;
     end 
    else 
     begin
        rd_en                           <=  1'b0;
        writing_x_to_host_memory_done   <= 1'b0;
        case (state)
            //This state is the beginning of  
            WR_MEM_IDLE_STATE: 
            begin 
                if(started_r3)  // started with two cycles later...
                    state                <= WR_MEM_STARTING_STATE;  
            end

            /* This state is just a stopby state which initilizes the parameters...*/
            WR_MEM_STARTING_STATE: 
            begin
                //rd_data_reg_avail        <= 1'b1;
                /* It also registers the parameters for the FSM*/  //main counter, 9th bit --> 512
                //main_counter             <= main_counter_wire;        //dimension[9+MAX_BURST_BITS-1:9];        

                //addr_model_for_epoch     <= 
                state                    <= WR_MEM_POLL_STATE;
                    // Go to start state, set some flags
            end

            //This state indicates the finish of each epoch...
            WR_MEM_POLL_STATE: 
            begin
                main_index               <= 32'h0;
                //rd_data_reg_avail        <=  1'b1;
                if ( (~writing_x_to_host_memory_en_r2) & writing_x_to_host_memory_en_r ) //capturing rising edge.
                begin
                    //addr_model_for_epoch <= addr_model_reg;
                    x_mem_rd_addr        <= 1'b0;
                    rd_en                <= 1'b1;
                    state                <= WR_MEM_IN_PROGRESS_STATE;
                end
           end

            //This state indicates the finish of each epoch...
            WR_MEM_IN_PROGRESS_STATE: 
            begin
                if ( rd_data_reg_valid & um_tx_wr_ready )
                begin
                    rd_en                               <= 1'b1;
                end

                //Output:::::...
                if (rd_en == 1'b1) //one cycle delayed...
                begin
                    x_mem_rd_addr                       <= x_mem_rd_addr + 1'b1;    //determined in FSM...
                    main_index                          <= main_index + 1'b1; //MAX_BURST_BITS'h1;
                    if (main_index == main_counter_minus_1) //end of all the chunks...
                    begin
                        writing_x_to_host_memory_done   <= 1'b1;
                        //main_index                      <= 0;
                        state                           <= WR_MEM_POLL_STATE; //Back to the A's main entry.
                    end
               end
            end
            //  

            WR_MEM_END_STATE: 
            begin
                state                          <= WR_MEM_END_STATE; //end of one sample...
            end 
        endcase 
         // else kill
    end 
end
                //if ( rd_data_reg_avail )  //rd_en is valid only when the reg is available. 
                //begin
                //    rd_en                <= 1'b1;
                //    rd_data_reg_avail    <= 1'b0;
                //end
                //else 
                //if ( rd_data_reg_valid & um_tx_wr_ready) //fully pipelined design.   & (~rd_data_reg_avail) is always true...
                //begin
                //end


assign state_counters_wr_x_to_memory = {17'b0, main_index, state};


endmodule
