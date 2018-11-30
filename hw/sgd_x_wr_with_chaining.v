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

module sgd_x_wr #( parameter DATA_WIDTH_IN      = 4 ,
                     parameter MAX_DIMENSION_BITS = `MAX_BIT_WIDTH_OF_X  ) ( //16
    input   wire                                   clk,
    input   wire                                   rst_n,
    //--------------------------Begin/Stop-----------------------------//

    output  wire [31:0]                            state_counters_x_wr,
    input   wire                                   started,
    input   wire                            [31:0] mini_batch_size,
    input   wire                            [31:0] dimension,
    input   wire                            [31:0] number_of_epochs,
    input   wire                            [31:0] number_of_samples,

    //the credit is used to make sure the ax module reads the updated model....
    output  reg                              [7:0] x_wr_credit_counter, 


    output  reg                                    sgd_execution_done,  
    output  reg                                    sgd_x_wr_error, 

    output  reg                                    writing_x_to_host_memory_en,
    input                                          writing_x_to_host_memory_done,

    ///////////////////wr part of x_updated//////////////////////
    input   wire                                   x_updated_wr_en,     
    input   wire               [`X_BIT_DEPTH-1:0]  x_updated_wr_addr,
    input   wire      [`NUM_BITS_PER_BANK*32-1:0]  x_updated_wr_data,

    ///////////////////wr part of x//////////////////////
    output  reg                                    x_wr_en,     
    output  reg                [`X_BIT_DEPTH-1:0]  x_wr_addr,
    output  reg       [`NUM_BITS_PER_BANK*32-1:0]  x_wr_data

);

    //input   wire                                   acc_gradient_valid[`NUM_BITS_PER_BANK-1:0];

reg                     x_wr_en_end_of_batch;      
reg               [7:0] x_wr_credit_counter_pre, x_wr_credit_counter_pre2, x_wr_credit_counter_pre3, x_wr_credit_counter_pre4, x_wr_credit_counter_pre5; 
always @(posedge clk) begin
    //if(~rst_n)
    //    x_wr_credit_counter  <= 8'b0;
    //else //if (started) 
    x_wr_credit_counter_pre2  <= x_wr_credit_counter_pre;
    x_wr_credit_counter_pre3  <= x_wr_credit_counter_pre2;
    x_wr_credit_counter_pre4  <= x_wr_credit_counter_pre3;
    x_wr_credit_counter_pre5  <= x_wr_credit_counter_pre4;
    x_wr_credit_counter       <= x_wr_credit_counter_pre5;
end

//to make sure that the parameters has been assigned...
//reg       started_r;   //one cycle delay from started...
reg [2:0] state; 
reg [3:0] error_state; //0000: ok; 0001: dimension is zero; 
reg               [11:0] main_counter, main_counter_minus_1, main_index;
reg                [4:0] numBits_minus_1, numBits_index;
reg                [9:0] numEpochs, epoch_index;
reg               [31:0] numSamples, sample_index;
reg                [7:0] bank_batch_size, bank_batch_size_minus_1, batch_index; 

reg               [31:0] number_of_epochs_r;
reg               [31:0] number_of_samples_r;
reg               [31:0] mini_batch_size_r;
reg               [31:0] main_counter_wire;
//assign                   main_counter_wire        = dimension[31 :`BIT_WIDTH_OF_BANK] + (dimension[`BIT_WIDTH_OF_BANK-1:0] != 0);
always @(posedge clk) begin
    begin
        number_of_epochs_r  <= number_of_epochs;
        number_of_samples_r <= number_of_samples;
        main_counter_wire   <= dimension[31 :`BIT_WIDTH_OF_BANK] + (dimension[`BIT_WIDTH_OF_BANK-1:0] != 0);
        mini_batch_size_r   <= mini_batch_size;

        /* It also registers the parameters for the FSM*/  //main counter, 9th bit --> 512
        main_counter             <= main_counter_wire;        //dimension[9+MAX_BURST_BITS-1:9];        
        main_counter_minus_1     <= main_counter_wire - 1'b1; //MAX_BURST_BITS'h1;
        numEpochs                <= number_of_epochs_r;                       //  - 10'h1
        numSamples               <= number_of_samples_r;
        bank_batch_size          <= mini_batch_size_r[`NUM_OF_BANKS_WIDTH+7:`NUM_OF_BANKS_WIDTH];
        bank_batch_size_minus_1  <= mini_batch_size_r[`NUM_OF_BANKS_WIDTH+7:`NUM_OF_BANKS_WIDTH] - 8'b1;

    end 
end


reg                                    x_wr_en_pre;     
reg                [`X_BIT_DEPTH-1:0]  x_wr_addr_pre;
reg       [`NUM_BITS_PER_BANK*32-1:0]  x_wr_data_pre;

reg                                    x_wr_en_pre2;     
reg                [`X_BIT_DEPTH-1:0]  x_wr_addr_pre2;
reg       [`NUM_BITS_PER_BANK*32-1:0]  x_wr_data_pre2;

wire                    x_updated_wr_en_is_coming;
assign                  x_updated_wr_en_is_coming = x_wr_en_pre; //x_updated_wr_en;

////////////////////Update the x (wr part)//////////////////////
always @(posedge clk) 
begin 
    begin
        x_wr_en_pre     <= x_updated_wr_en; // & x_wr_en_end_of_batch
        x_wr_addr_pre   <= x_updated_wr_addr;
        x_wr_data_pre   <= x_updated_wr_data;   
        //x_wr_en_pre2    <= x_wr_en_pre;
        //x_wr_addr_pre2  <= x_wr_addr_pre;
        //x_wr_data_pre2  <= x_wr_data_pre;    
        //x_wr_en         <= x_wr_en_pre2;
        //x_wr_addr       <= x_wr_addr_pre2;
        //x_wr_data       <= x_wr_data_pre2;

        x_wr_en         <= x_wr_en_pre  & x_wr_en_end_of_batch;
        x_wr_addr       <= x_wr_addr_pre;
        x_wr_data       <= x_wr_data_pre;                     
    end 
end




reg       started_r, started_r2, started_r3;   //one cycle delay from started...

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

reg chain_model_en; //fix me. The value chooce of main_index (1) is trick.
reg sample_not_last_one;

always @(posedge clk) begin
    sample_not_last_one  <=  sample_index != numSamples; //happen before chan_model_en is valid
    chain_model_en       <= ( (main_index == 12'b00) & x_updated_wr_en_is_coming ) & sample_not_last_one; //12'b10: <=128 --> bug.
end
//()//(   )


//trigger: 
//Output: x_wr_credit_counter_pre, x_wr_enable
///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////       Finite State Machine      ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
localparam [2:0]
        X_WR_IDLE_STATE            = 3'b000,
        X_WR_STARTING_STATE        = 3'b001,
        X_WR_EPOCH_STATE           = 3'b010,        
        X_WR_SAMPLE_CHECK_STATE    = 3'b011, //check whether to update x.
        X_WR_SAMPLE_IDLE_STATE     = 3'b100, //Not update.
        X_WR_SAMPLE_UPDATING_STATE = 3'b101, //Update.
        X_WR_FINISH_STATE          = 3'b110,
        X_WR_GLOBAL_STATE          = 3'b111;

always@(posedge clk) begin
    if(~rst_n) 
     begin
        //writing_x_to_host_memory_en      <= 1'b0;
        //x_wr_credit_counter_pre  <= 8'b0;   
        sgd_execution_done       <=  1'b0;
        state                    <=  X_WR_IDLE_STATE;
        //bank_batch_size          <=  8'h0;  
        //bank_batch_size_minus_1  <=  8'h0;  
        //batch_index              <=  8'h0;  

        //x_wr_en_end_of_batch     <=  1'b0;

        //numEpochs                <= 10'h0;
        //numSamples               <= 32'h0;
        //main_counter_minus_1     <= 32'h0; 
        //main_counter             <= 32'h0; 

        //sample_index             <= 32'h0;
        //main_index               <= 32'h0; 
     end 
    else 
     begin
        //Do the memory read job....
        writing_x_to_host_memory_en      <= 1'b0; 
        x_wr_en_end_of_batch             <= 1'b0;
        case (state)
            //This state is the beginning of FSM... 
            X_WR_IDLE_STATE: 
            begin 
                if(started_r2)  // started with one cycle later...
                    state                <= X_WR_STARTING_STATE;  
            end

            /* This state is just a stopby state which initilizes the parameters...*/
            X_WR_STARTING_STATE: 
            begin
                epoch_index              <= 10'h0;
                sgd_execution_done       <=  1'b0;
                sgd_x_wr_error           <=  1'b0;
                error_state              <=  3'b0;


                x_wr_credit_counter_pre  <= 8'b0;

                state                    <= X_WR_EPOCH_STATE;
            end

            //This state indicates the begginning of each epoch...
            X_WR_EPOCH_STATE: 
            begin
                main_index               <= 1'b0; //MAX_BURST_BITS'h0;
                sample_index             <= 32'h0;
                batch_index              <=  8'h0;  
                /* This state is used for the beginning of each epoch, check whether 
                the execution with numEpochs epochs is finished...*/

                epoch_index              <= epoch_index + 10'b1;
                if (epoch_index == numEpochs) //The SGD ends the execution..when all done here.. _minus_1
                    state                <= X_WR_FINISH_STATE;
                else
                begin
                    //x_wr_credit_counter_pre enables the reading bank_batch_size samples for the first epoch...
                    if (epoch_index == 10'b0)
                    x_wr_credit_counter_pre <= x_wr_credit_counter_pre + bank_batch_size; 
                
                    state                   <= X_WR_SAMPLE_CHECK_STATE;
                end
            end

            //This state indicates the beginning of sample a, 
            //In this state, the signal x_updated_wr_en cannot be valid, otherwise, the code can be wrong...
            X_WR_SAMPLE_CHECK_STATE:
            begin
                main_index               <= 1'b0; //MAX_BURST_BITS'h0;
                sample_index             <= sample_index + `NUM_OF_BANKS;
                batch_index              <= batch_index  + 8'h1;  

                if (x_updated_wr_en_is_coming)
                begin
                    sgd_x_wr_error           <=  1'b1;
                    state                <= X_WR_FINISH_STATE;         //Error happens......
                end
                else if (sample_index == numSamples) //write back the model to the global memory.
                    state                <= X_WR_GLOBAL_STATE;//X_WR_EPOCH_STATE;   //Enter the state to write back the model
                else if (batch_index == bank_batch_size_minus_1)
                begin
                    batch_index          <= 8'b0;
                    state                <= X_WR_SAMPLE_UPDATING_STATE; 
                end
                else
                    state                <= X_WR_SAMPLE_IDLE_STATE;
            end

            X_WR_SAMPLE_IDLE_STATE:
            begin
                main_index               <= main_index + {31'b0, x_updated_wr_en_is_coming}; //MAX_BURST_BITS'h1;
                if (main_index == main_counter) //end of all the chunks...main_counter_minus_1
                begin
                    main_index           <= 0;
                    state                <= X_WR_SAMPLE_CHECK_STATE; //Back to the A's main entry.
                end
            end 

            X_WR_SAMPLE_UPDATING_STATE: 
            begin
                x_wr_en_end_of_batch     <= 1'b1;

                main_index               <= main_index + {31'b0, x_updated_wr_en_is_coming}; //MAX_BURST_BITS'h1;
                if (main_index == main_counter) //end of all the chunks...main_counter_minus_1
                begin
                    //main_index           <= 0;
                    state                <= X_WR_SAMPLE_CHECK_STATE; //Back to the A's main entry.
                end

                if (chain_model_en)//(  ( (main_index == 32'b1) & x_updated_wr_en_is_coming ) & (sample_index != numSamples)  ) //Make sure that the x is written to the bram....
                    x_wr_credit_counter_pre  <= x_wr_credit_counter_pre + bank_batch_size;
            end 

            //Writing back the model to global memory
            X_WR_GLOBAL_STATE:
            begin
                /////
                writing_x_to_host_memory_en  <= 1'b1; 
                if (writing_x_to_host_memory_done)
                begin
                    x_wr_credit_counter_pre      <= x_wr_credit_counter_pre + bank_batch_size;
                    state                    <= X_WR_EPOCH_STATE; //end of one sample...
                end                
            end

            X_WR_FINISH_STATE: 
            begin

                sgd_execution_done       <=  1'b1;
                /////
                state                    <= X_WR_FINISH_STATE; //end of one sample...
            end

        endcase 
         // else kill
    end 
end

//one state machine to control the rd addr, which is shared by the "a" and "b"



assign state_counters_x_wr = {x_updated_wr_en, state, sample_index[19:0], epoch_index[7:0]};


endmodule
