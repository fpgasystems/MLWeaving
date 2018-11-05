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

module sgd_mem_rd #( parameter DATA_WIDTH_IN      = 4 ,
                     parameter MAX_DIMENSION_BITS = `MAX_BIT_WIDTH_OF_X  ) ( //16
    input   wire                                   clk,
    input   wire                                   rst_n,
    //--------------------------Begin/Stop-----------------------------//
    input   wire                                   started,
    output  reg                                    mem_op_done,
    output  reg  [31:0]                            num_issued_mem_rd_reqs,

    output  reg  [31:0]                            state_counters_mem_rd,

    //---------Input: Parameters (where, how many) from the root module-------//
    input   wire [57:0]                            addr_a,
    input   wire [57:0]                            addr_b,
    //input   wire [63:0]                            addr_model,
    input   wire [31:0]                            number_of_epochs,
    input   wire [31:0]                            number_of_samples,
    input   wire [31:0]                            dimension,
    input   wire [31:0]                            number_of_bits, 
    //---------------------Memory Inferface----------------------------//
    // TX RD
    output  reg  [57:0]                            um_tx_rd_addr,
    output  reg  [7:0]                             um_tx_rd_tag,
    output  reg                                    um_tx_rd_valid,
    input   wire                                   um_tx_rd_ready
);
//From parameters from sgd_defines.vh:::
//`define NUM_OF_BANKS             8

parameter MAX_BURST_BITS = MAX_DIMENSION_BITS - 9; //7..... Each chunk contains 512 features...

//to make sure that the parameters has been assigned...
reg       started_r, started_r2;   //one cycle delay from started...
reg [2:0] state; 
reg [3:0] error_state = 0; //0000: ok; 0001: dimension is zero; 

always @(posedge clk) begin
    if(~rst_n) 
        num_issued_mem_rd_reqs   <= 32'b0;
    else if (started_r) 
        num_issued_mem_rd_reqs   <= num_issued_mem_rd_reqs + (um_tx_rd_valid == 1'b1);
end

always @(posedge clk or negedge rst_n) begin
    if(~rst_n)
    begin
        started_r  <= 1'b0;
        started_r2 <= 1'b0;
    end 
    else //if (started) 
    begin
        started_r  <= started;   //1'b0;
        started_r2 <= started_r; //1'b0;
    end 
end

reg               [11:0] main_counter, main_counter_minus_1, main_index;
reg                [4:0] numBits_minus_1, numBits_index;
reg               [31:0] num_CLs_per_bank;     
reg                [9:0] numEpochs, epoch_index;
reg               [31:0] numSamples, sample_index;
reg               [31:0] addr_b_index;
reg               [57:0] addr_a_for_each_sample;

reg               [31:0] bank_addr_offset, rem_bank_addr_offset;
reg                [4:0] bank_index, rem_bank_index;      //enough for the index...
reg               [31:0] offset_bank_of_samples;
reg                      need_to_entry_B;

reg               [31:0] number_of_epochs_r;
reg               [31:0] number_of_samples_r;
reg               [31:0] number_of_bits_r; 
reg               [31:0] main_counter_wire;
//assign                   main_counter_wire = dimension[31 :`BIT_WIDTH_OF_BANK] + (dimension[`BIT_WIDTH_OF_BANK-1:0] != 0);


always @(posedge clk ) 
begin
    begin
        number_of_epochs_r       <= number_of_epochs;
        number_of_samples_r      <= number_of_samples;
        //dimension_r         <= dimension;
        number_of_bits_r         <= number_of_bits;
        main_counter_wire        <= dimension[31 :`BIT_WIDTH_OF_BANK] + (dimension[`BIT_WIDTH_OF_BANK-1:0] != 0);

        /* It also registers the parameters for the FSM*/  //main counter, 9th bit --> 512
        main_counter             <= main_counter_wire;        //dimension[9+MAX_BURST_BITS-1:9];        
        main_counter_minus_1     <= main_counter_wire - 1'b1; //MAX_BURST_BITS'h1;
        numBits_minus_1          <= number_of_bits_r[5:0] - 6'h1;
        numEpochs                <= number_of_epochs_r;                       //  - 10'h1
        numSamples               <= number_of_samples_r;
        num_CLs_per_bank         <= {main_counter_wire[26:0], 5'b0 }; //each
    end 
end


reg need_to_entry_B_reg, is_the_last_sample;

always @(posedge clk) 
begin
    begin
        need_to_entry_B_reg  <= ( (sample_index[3:0] == 4'h0) & (sample_index[31:4] != 28'h0)  ); //& need_to_entry_B
        is_the_last_sample   <= (sample_index == numSamples);
    end 
end

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////       Finite State Machine      ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
localparam [2:0]
        MEM_IDLE_STATE          = 3'b000,
        MEM_STARTING_STATE      = 3'b001,
        MEM_EPOCH_STATE         = 3'b010,        
        MEM_A_SAMPLE_STATE      = 3'b011,
        MEM_A_READ_STATE        = 3'b100,
        MEM_B_READ_STATE        = 3'b101,
        MEM_END_STATE           = 3'b110;

always@(posedge clk) begin
    if(~rst_n) 
     begin
        state                    <=  MEM_IDLE_STATE;
        //error_state              <=  4'b0;
        mem_op_done              <=  1'b0;

        //numEpochs                <= 10'h0;
        //numSamples               <= 32'h0;
        //num_CLs_per_bank         <=     0; //15'h0; //(MAX_BURST_BITS+5)'h0;
        need_to_entry_B          <=  1'b0;

       //main_counter_minus_1     <=     0; //10'h0; //MAX_BURST_BITS'h0;
        //main_index               <=     0; //10'h0; //MAX_BURST_BITS'h0;
        //numBits_minus_1          <=  5'h0;
        //numBits_index            <=  5'h0;
        //epoch_index              <= 10'h0;
        //sample_index             <= 32'h0;

        //addr_b_index             <= 32'h0;
        //addr_a_for_each_sample   <= 58'h0;
        //offset_bank_of_samples   <= 32'h0;

        //um_tx_rd_addr            <= 58'h0;
        //um_tx_rd_tag             <=  8'h0;
        um_tx_rd_valid           <=  1'b0;
     end 
    else 
     begin
        //Do the memory read job.... 
        um_tx_rd_valid  <= 1'b0;  //always required, otherwise more rd request sent out...
        case (state)
            //This state is the beginning of  
            MEM_IDLE_STATE: 
            begin 
                if(started_r2)  // started with two cycles later...
                    state                <= MEM_STARTING_STATE;  
            end

            /* This state is just a stopby state which initilizes the parameters...*/
            MEM_STARTING_STATE: 
            begin
                epoch_index              <= 10'h0;

                state                    <= MEM_EPOCH_STATE;
                    // Go to start state, set some flags
            end

            //This state indicates the begginning of each epoch...
            MEM_EPOCH_STATE: 
            begin
                /*This state initilizes each index to zero.*/
                addr_b_index             <= 32'h0;
                addr_a_for_each_sample   <= addr_a[57:0];
                offset_bank_of_samples   <= 32'h0;

                main_index               <= 1'b0; //MAX_BURST_BITS'h0;
                numBits_index            <= 5'b0;
                sample_index             <= 32'h0;
                /* This state is used for the beginning of each epoch, check whether 
                the execution with numEpochs epochs is finished...*/

                epoch_index              <= epoch_index + 10'b1;
                if (epoch_index == numEpochs) //The SGD ends the execution..when all done here.. _minus_1
                    state                <= MEM_END_STATE;
                else
                    state                <= MEM_B_READ_STATE;
            end

            //This state indicates that "b" is loaded from memory.....
            MEM_B_READ_STATE:
            begin
                need_to_entry_B         <= 1'b0;
                //um_tx_rd_valid  <= 1'b0;  //always required, otherwise more rd request sent out...
                if (um_tx_rd_ready)       //request sent when it is ready...
                begin
                    addr_b_index        <= addr_b_index + 32'h1;
                    um_tx_rd_addr       <= addr_b[57:0] + addr_b_index;//58'h0;
                    um_tx_rd_tag        <= `MEM_RD_B_TAG; //tag: b for the destination column...
                    um_tx_rd_valid      <= 1'b1;
                    //num_issued_mem_rd_reqs  <= num_issued_mem_rd_reqs + 64'b1;
                    state               <= MEM_A_SAMPLE_STATE;           
                end
            end

            //This state indicates the beginning of sample a, but read nothing... 
            MEM_A_SAMPLE_STATE:
            begin
                main_index                     <= 1'b0; //MAX_BURST_BITS'h0;
                numBits_index                  <= 5'b0; //indices should be zero at the beginning of each sample..

                if (is_the_last_sample)//(sample_index == numSamples) //
                    state                      <= MEM_EPOCH_STATE;
                else if ( need_to_entry_B_reg & need_to_entry_B ) //Load data for each (need_to_entry_B_reg) //(sample_index[3:0] == 4'h0) & (sample_index[31:4] != 28'h0) 
                    state                      <= MEM_B_READ_STATE;
                else //Enter the sample reading....
                begin
                    sample_index               <= sample_index + `NUM_OF_BANKS;
                    need_to_entry_B            <= 1'b1;

                    offset_bank_of_samples     <= offset_bank_of_samples + num_CLs_per_bank;//{ num_CLs_per_sample, { `NUM_OF_BANKS_WIDTH{1'b0} } };
                    addr_a_for_each_sample     <= addr_a[57:0] + offset_bank_of_samples;    //{ num_CLs_per_sample; 
                    state                      <= MEM_A_READ_STATE;
                end
            end

            MEM_A_READ_STATE: 
            begin
                if (um_tx_rd_ready)       //request sent when it is ready...
                begin
                    //1, Outer loop: bit_offset.
                    numBits_index                  <= numBits_index + 5'h1;
                    if (numBits_index == numBits_minus_1)  //end of each 512-feature chunk...
                    begin
                        numBits_index              <= 5'h0;
                        //2, Innar loop: chunk index...
                        main_index                 <= main_index + 1'b1; //MAX_BURST_BITS'h1;
                        if (main_index == main_counter_minus_1) //end of all the chunks...
                        begin
                            main_index             <= 0;
                            state                  <= MEM_A_SAMPLE_STATE; //Back to the A's main entry.
                        end
                    end
                    um_tx_rd_addr                  <= addr_a_for_each_sample + {main_index, numBits_index};//58'h0;
                    um_tx_rd_tag                   <= `MEM_RD_A_TAG; //tag: a for the destination column...
                    um_tx_rd_valid                 <= 1'b1;   
                    //num_issued_mem_rd_reqs             <= num_issued_mem_rd_reqs + 64'b1;
                end
            end

            MEM_END_STATE: 
            begin
                mem_op_done                    <=  1'b1;
                state                          <= MEM_END_STATE; //end of one sample...
            end 
        endcase 
         // else kill
    end 
end

reg mem_rd_full_pre, mem_rd_full; 


always @(posedge clk) begin
    mem_rd_full_pre  <= ~um_tx_rd_ready;
    mem_rd_full      <= mem_rd_full_pre;
end

always @(posedge clk) begin
    if(~rst_n)
        state_counters_mem_rd <= 32'b0;
    else
        state_counters_mem_rd <= state_counters_mem_rd + {31'b0, mem_rd_full};
end

//assign state_counters_mem_rd = {um_tx_rd_ready, state, sample_index[19:0], epoch_index[7:0]};

//one state machine to control the rd addr, which is shared by the "a" and "b"



endmodule
