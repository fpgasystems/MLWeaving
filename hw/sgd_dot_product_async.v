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

module sgd_dot_product (
    input   wire                                   clk,
    input   wire                                   rst_n,
    //--------------------------Begin/Stop-----------------------------//
    input   wire                                   started,
    output  reg  [31:0]                            state_counters_dot_product,

    //------------------------Configuration-----------------------------//
    input   wire [31:0]                            mini_batch_size,
    input   wire [31:0]                            number_of_epochs,
    input   wire [31:0]                            number_of_samples,
    input   wire [31:0]                            dimension,
    input   wire [31:0]                            number_of_bits, 
    input   wire [31:0]                            step_size,

    //------------------Input: disptach resp data to a of each bank---------------//
    input [`NUM_BITS_PER_BANK*`NUM_OF_BANKS-1:0]   dispatch_axb_a_data, //512. 
    input                                          dispatch_axb_a_wr_en, 
    output wire                                    dispatch_axb_a_almost_full, 
    //------------------Input: x ---------------//
    //input                                  [x:0]  x,  
    input                                          writing_x_to_host_memory_done,  
    input                                   [7:0]  x_wr_credit_counter,  //this one is used to check the x is written.        
    output reg                 [`X_BIT_DEPTH-1:0]  x_rd_addr,
    input             [`NUM_BITS_PER_BANK*32-1:0]  x_rd_data, 


    output wire                                        buffer_a_rd_data_valid,
    output wire [`NUM_BITS_PER_BANK*`NUM_OF_BANKS-1:0] buffer_a_rd_data,

    //------------------Output: dot products for all the banks. ---------------//
    output wire signed                     [31:0] dot_product_signed[`NUM_OF_BANKS-1:0],       //
    output wire                                   dot_product_signed_valid[`NUM_OF_BANKS-1:0]  //
);

reg       started_r, started_r2, started_r3;   //one cycle delay from started...

always @(posedge clk) 
begin
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
//wire                  [8:0] buffer_a_rd_addr;

/////////////////////Wr of a_buffer///////////////////////////////////////
reg  [`NUM_BITS_PER_BANK*`NUM_OF_BANKS-1:0] buffer_a_wr_data;
reg                                         buffer_a_wr_en;
wire                                        buffer_a_almost_full;
assign dispatch_axb_a_almost_full = buffer_a_almost_full; //output...
always @(posedge clk) 
begin
    buffer_a_wr_data   <= dispatch_axb_a_data;   
    buffer_a_wr_en     <= dispatch_axb_a_wr_en; //1'b0;
end


/////////////////////rd of a_buffer///////////////////////////////////////
//wire                                        buffer_a_rd_data_valid;
//wire [`NUM_BITS_PER_BANK*`NUM_OF_BANKS-1:0] buffer_a_rd_data;
wire                     buffer_a_empty;
wire               [8:0] buffer_a_counter; 
reg                      buffer_a_rd_en;


//on-chip buffer for a 
blockram_fifo #( .FIFO_WIDTH      (`NUM_BITS_PER_BANK*`NUM_OF_BANKS), 
                 .FIFO_DEPTH_BITS ( `X_BIT_DEPTH                   ) //depth 
) inst_a_buffer (
    .clk        (clk),
    .reset_n    (rst_n),

    //Writing side....
    .we         ( buffer_a_wr_en        ), //dispatch_axb_a_wr_en
    .din        ( buffer_a_wr_data      ), //dispatch_axb_a_data
    .almostfull ( buffer_a_almost_full  ), //dispatch_axb_a_almost_full

    //reading side.....
    .re         (buffer_a_rd_en         ),
    .dout       (buffer_a_rd_data       ),
    .valid      (buffer_a_rd_data_valid ),
    .empty      (buffer_a_empty         ),
    .count      (buffer_a_counter       )
);


reg x_rd_en, x_rd_data_valid, x_rd_valid_r1; //it becomes valid after two cycles
always @(posedge clk) 
begin 
    x_rd_valid_r1             <= x_rd_en;
    x_rd_data_valid           <= x_rd_valid_r1;
end


/////////////////One FSM to control NUM_BITS_PER_BANK accumulators////////////////
//Wait for the wr_counter valid information to do the computation....
//One state for the .
reg                [2:0] state;
reg               [11:0] main_counter, main_counter_minus_1, main_index;
reg                [4:0] numBits_minus_1, numBits_index;
reg                [9:0] numEpochs, epoch_index;
reg               [31:0] numSamples, sample_index;
reg                [4:0] num_shift_bits;

reg                [7:0] x_rd_credit_counter;

reg               [31:0] number_of_epochs_r;
reg               [31:0] number_of_samples_r;
reg               [31:0] number_of_bits_r; 
reg               [31:0] main_counter_wire;
//reg               [31:0] mini_batch_size_r; 
reg               [31:0] step_size_r;
reg                      writing_x_to_host_memory_done_r, writing_x_to_host_memory_done_r2;
always @(posedge clk) begin
    begin
        writing_x_to_host_memory_done_r    <= writing_x_to_host_memory_done;
        writing_x_to_host_memory_done_r2  <= writing_x_to_host_memory_done_r;
    end 
end

always @(posedge clk) begin
    begin
        number_of_epochs_r       <= number_of_epochs;
        number_of_samples_r      <= number_of_samples;
        number_of_bits_r         <= number_of_bits;
        main_counter_wire        <= dimension[31 :`BIT_WIDTH_OF_BANK] + (dimension[`BIT_WIDTH_OF_BANK-1:0] != 0);
        //mini_batch_size_r   <= mini_batch_size;
        step_size_r              <= step_size;   

        main_counter             <= main_counter_wire;        //dimension[9+MAX_BURST_BITS-1:9];        
        main_counter_minus_1     <= main_counter_wire - 1'b1; //MAX_BURST_BITS'h1;
        numBits_minus_1          <= number_of_bits_r[5:0] - 6'h1;
        numEpochs                <= number_of_epochs_r;                       //  - 10'h1
        numSamples               <= number_of_samples_r;
        num_shift_bits           <= step_size_r[4:0]; // - 5'h0  - 5'h6
    end 
end

reg x_credit_available_pre, x_credit_available_pre2, x_credit_available_pre3, x_credit_available;
reg not_the_last_sample;

always @(posedge clk) begin
    begin
        x_credit_available_pre   <= (x_wr_credit_counter != x_rd_credit_counter); 
        x_credit_available_pre2  <= x_credit_available_pre; 
        x_credit_available_pre3  <= x_credit_available_pre2; 

        x_credit_available       <= x_credit_available_pre3; 

        not_the_last_sample      <= (sample_index != numSamples);
    end 
end

reg                      in_computing_stage;
wire                     buffer_a_rd_safe; //It can be the bottleneck.
                        //cannot read in this cycle     //Kill the potential that the next cycle can be empty.
assign                   buffer_a_rd_safe = ~(buffer_a_empty | ( (buffer_a_counter == 1'h1) & buffer_a_rd_en ));
localparam [2:0]
        BANK_IDLE_STATE          = 3'b000,
        BANK_STARTING_STATE      = 3'b001,
        BANK_EPOCH_STATE         = 3'b010,        
        BANK_A_SAMPLE_STATE      = 3'b011,
        BANK_A_COMPUTING_STATE   = 3'b100,
        BANK_CHECK_X_STATE       = 3'b101,
        BANK_END_STATE           = 3'b110,
        BANK_EPOCH_SAMPLE_STATE  = 3'b111;

always@(posedge clk) begin
    if(~rst_n) 
     begin
        state                    <=  BANK_IDLE_STATE;
     end 
    else 
     begin
        buffer_a_rd_en           <= 1'b0;
        x_rd_en                  <= 1'b0;
        in_computing_stage       <= 1'b0;

        //Do the memory read job.... 
        case (state)
            //This state is the beginning of FSM... 
            BANK_IDLE_STATE: 
            begin 
                if(started_r3)  // started with one cycle later...
                    state                 <= BANK_STARTING_STATE;  
            end

            /* This state is just a stopby state which initilizes the parameters...*/
            BANK_STARTING_STATE: 
            begin
                epoch_index               <= 10'h0;
                x_rd_credit_counter       <=  8'b0;

                /* It also registers the parameters for the FSM*/  //main counter, 9th bit --> 512
                state                     <= BANK_EPOCH_STATE;
                    // Go to start state, set some flags
            end

            //This state indicates the begginning of each epoch...
            BANK_EPOCH_STATE: 
            begin
                /*This state initilizes each index to zero.*/
                //x_rd_credit_counter      <=  8'b0;
                //main_index               <=  1'b0; //MAX_BURST_BITS'h0;
                numBits_index             <=  5'b0;
                sample_index              <= 32'h0;
                /* This state is used for the beginning of each epoch, check whether 
                the execution with numEpochs epochs is finished...*/

                epoch_index               <= epoch_index + 10'b1;
                if (epoch_index == numEpochs) //The SGD ends the execution..when all done here.. _minus_1
                    state                 <= BANK_END_STATE;
                else
                    state                 <= BANK_EPOCH_SAMPLE_STATE; //BANK_A_SAMPLE_STATE;
            end

            BANK_EPOCH_SAMPLE_STATE:
            begin
                if ( (epoch_index == 10'b1) || writing_x_to_host_memory_done_r2 )
                state                     <= BANK_A_SAMPLE_STATE;
            end

            //This state indicates the beginning of sample a, but read nothing... 
            BANK_A_SAMPLE_STATE:
            begin
                main_index                <= 1'b0; //MAX_BURST_BITS'h0;
                numBits_index             <= 5'b0; //indices should be zero at the beginning of each sample..

                if (~not_the_last_sample)//(sample_index == numSamples) //
                    state                 <= BANK_EPOCH_STATE;
                //else if ( bank_batch_size == loop_index ) //Need to check the dependency... 
                else //Enter the sample reading....
                begin
                    sample_index          <= sample_index + `NUM_OF_BANKS;
                    state                 <= BANK_CHECK_X_STATE; //BANK_A_COMPUTING_STATE; //
                end
            end

            //This state indicates that "b" is loaded from memory.....
            BANK_CHECK_X_STATE:
            begin
                if (1)  //x_credit_available   x_wr_credit_counter != x_rd_credit_counter
                begin
                    //if ( not_the_last_sample ) //sample_index != numSamples
                    x_rd_credit_counter   <= x_rd_credit_counter + 8'b1;
                    //num_issued_mem_rd_reqs  <= num_issued_mem_rd_reqs + 64'b1;
                    state                 <= BANK_A_COMPUTING_STATE;           
                end
                //May log down the cycles the computing pipeline is idle due to the RAW dependency. 
            end

            BANK_A_COMPUTING_STATE: //For the sample. 
            begin
                in_computing_stage        <= 1'b1;

                if (buffer_a_rd_safe)//With this valid, it is saft to generate the rd_en signal in the next cycle. 
                begin
                    //1, Outer loop: bit_offset.
                    numBits_index         <= numBits_index + 5'h1;
                    if (numBits_index == numBits_minus_1)  //end of each 512-feature chunk...
                    begin
                        numBits_index     <= 5'h0;  
                        //2, Innar loop: main index...
                        main_index        <= main_index + 1'b1; //MAX_BURST_BITS'h1;
                        if (main_index == main_counter_minus_1) //end of all the chunks...
                        begin
                            //main_index             <= 0;
                            state         <= BANK_A_SAMPLE_STATE; //BANK_A_SAMPLE_STATE; //Back to the A's main entry.
                        end
                    end

                    //////////////////////////////Output////////////////////////////// 
                    //buffer_a_rd_en           <= 1'b1;
                    buffer_a_rd_en        <= 1'b1; //[`NUM_OF_BANKS-1:0]
                    if (numBits_index == 5'h0)
                    begin
                        x_rd_en           <= 1'b1;  //[`NUM_OF_BANKS-1:0]
                        x_rd_addr         <= main_index[`X_BIT_DEPTH-1:0];
                    end

                    //Do some thing....
                    //num_issued_mem_rd_reqs             <= num_issued_mem_rd_reqs + 64'b1;
                end
            end

            BANK_END_STATE: 
            begin
                state                     <= BANK_END_STATE; //end of one sample...
            end 
        endcase 
         // else kill
    end 
end

/////////////////////////////////Dot products for banks.///////////////////////////////////////////////////////
//////////////////////////////Without real accumulator, just add...//////////////////////////////////
reg  signed        a_input[`NUM_OF_BANKS-1:0][`NUM_BITS_PER_BANK-1:0]; //     8*64
reg  signed [31:0] x_input[`NUM_OF_BANKS-1:0][`NUM_BITS_PER_BANK-1:0]; //32b* 8*64

reg  signed        add_tree_in_valid[`NUM_OF_BANKS-1:0]; //32b* 8*64
wire signed [31:0] add_tree_in[`NUM_OF_BANKS-1:0][`NUM_BITS_PER_BANK-1:0]; //32b* 8*64

wire signed [31:0] add_tree_out[`NUM_OF_BANKS-1:0]; //32b* 8*64
wire signed        add_tree_out_valid[`NUM_OF_BANKS-1:0]; //32b* 8*64

wire signed           [35:0] add_tree_out_shift_wire[`NUM_OF_BANKS-1:0]; //more precisin for the accumulator.

reg signed            [35:0] add_tree_out_shift[`NUM_OF_BANKS-1:0]; //more precisin for the accumulator.
reg                          add_tree_out_shift_valid[`NUM_OF_BANKS-1:0];
reg                   [ 4:0] d_numBits_index[`NUM_OF_BANKS-1:0];
reg                   [11:0] chunk_index[`NUM_OF_BANKS-1:0];

reg                          ax_dot_product_valid_pre[`NUM_OF_BANKS-1:0];
reg                          adder_tree_first_bit_en[`NUM_OF_BANKS-1:0];

reg signed            [35:0] ax_dot_product[`NUM_OF_BANKS-1:0];
reg                          ax_dot_product_valid[`NUM_OF_BANKS-1:0];


genvar i, j;
generate for( i = 0; i < `NUM_OF_BANKS; i = i + 1) begin: inst_bank
    //0:::::::Pre at input data 
    for( j = 0; j < `NUM_BITS_PER_BANK; j = j + 1) begin: inst_stage_input 
        always @(posedge clk) 
        begin 
            a_input[i][j]     <= buffer_a_rd_data[`NUM_BITS_PER_BANK*i+j]; //1 bit
            x_input[i][j]     <= x_rd_data[32*(j+1)-1:32*j]; // 
        end 
    end
    for( j = 0; j < `NUM_BITS_PER_BANK; j = j + 1) begin: inst_at_input
        assign add_tree_in[i][j] = (a_input[i][j] == 1'b1)? x_input[i][j]:32'b0;
    end
    //0:::::::Pre at input data valid ...
    always @(posedge clk) 
    begin 
        add_tree_in_valid[i]    <= buffer_a_rd_data_valid;
    end 
    //1:::::::add tree which brings a 6-cycles latency. ...
    sgd_adder_tree #(
        .TREE_DEPTH (`BIT_WIDTH_OF_BANK) //2**8 = 64 
    ) inst_ax (
        .clk              ( clk                   ),
        .rst_n            ( rst_n                 ),
        .v_input          ( add_tree_in[i]        ),
        .v_input_valid    ( add_tree_in_valid[i]  ),

        .v_output         ( add_tree_out[i]       ),   //output...
        .v_output_valid   ( add_tree_out_valid[i] ) 
    ); 

    assign add_tree_out_shift_wire[i]   = {add_tree_out[i], 4'b0 }; //{4{add_tree_out[i][0]}}&{4{add_tree_out[i][31]}}
    //at input valid ...
    always @(posedge clk) 
    begin 
        case (d_numBits_index[i])
            5'h00: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 1 );
            5'h01: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 2 );
            5'h02: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 3 );
            5'h03: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 4 );
            5'h04: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 5 );
            5'h05: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 6 );
            5'h06: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 7 );
            5'h07: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 8 );
            5'h08: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 9 );
            5'h09: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 10);
            5'h0a: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 11);
            5'h0b: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 12);
            5'h0c: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 13);
            5'h0d: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 14);
            5'h0e: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 15);
            5'h0f: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 16);
            5'h10: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 17);
            5'h11: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 18);
            5'h12: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 19);
            5'h13: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 20);
            5'h14: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 21);
            5'h15: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 22);
            5'h16: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 23);
            5'h17: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 24);
            5'h18: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 25);
            5'h19: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 26);
            5'h1a: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 27);
            5'h1b: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 28);
            5'h1c: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 29);
            5'h1d: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 30);
            5'h1e: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 31);
            5'h1f: add_tree_out_shift[i]          <= (add_tree_out_shift_wire[i] >>> 32);
        endcase 
        
        //add_tree_out_shift[i]          <= ( add_tree_out_shift_wire[i] >>> 3 ); //(d_numBits_index+5'b1)
        add_tree_out_shift_valid[i]               <= add_tree_out_valid[i];
    end 

    always @(posedge clk) begin
        if(~rst_n) 
        begin
            d_numBits_index[i]                  <=  5'b0;
            chunk_index[i]                      <= 12'h0;
        end
        else
        begin
            adder_tree_first_bit_en[i]          <= 1'b0;
            ax_dot_product_valid_pre[i]         <= 1'b0;
            if (add_tree_out_valid[i])       //add_tree_out_shift_valid
            begin
                //1, Outer loop: d_numBits_index
                d_numBits_index[i]              <= d_numBits_index[i] + 5'h1;
                if (d_numBits_index[i] == numBits_minus_1)  
                begin
                    d_numBits_index[i]          <= 5'h0;  
                    //2, Intra loop: chunk_index
                    chunk_index[i]              <= chunk_index[i] + 1'b1; 
                    if (chunk_index[i] == main_counter_minus_1) //end of all the chunks...
                    begin
                        chunk_index[i]          <= 0;
                    ax_dot_product_valid_pre[i] <= 1'b1;
                    end
                end
                //output to the next cycle....
                adder_tree_first_bit_en[i]      <= (chunk_index[i] == 12'b0)&(d_numBits_index[i]==5'b0);
           end
        end 
    end

    always @(posedge clk) begin
        ax_dot_product_valid[i]             <= ax_dot_product_valid_pre[i];
        if (add_tree_out_shift_valid[i])       //part of the result coming...
        begin
            //compute the dot product result...
            if ( adder_tree_first_bit_en[i] ) //first of vector
                ax_dot_product[i]           <= add_tree_out_shift[i] + 36'hb;
            else                                        //add  
                ax_dot_product[i]           <= ax_dot_product[i] +  ( (add_tree_out_shift[i][35:0] == 36'hfffffffff)? 36'h0:add_tree_out_shift[i] ); //add_tree_out_shift[i] ;//
        end
    end

    //Output of dot product module.
    assign dot_product_signed[i]        = ax_dot_product[i][35:4];       //
    assign dot_product_signed_valid[i]  = ax_dot_product_valid[i];  //

end 
endgenerate

endmodule
