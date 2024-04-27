#include "cnn.h"

void kernel
(
    hls::stream<axis_in_t> &in,
    hls::stream<axis_weights_t> &weights,
    hls::stream<axis_out_t> &out,
    const int8_t l1_kernels[IN_CHANNELS * L1_KERNELS][KERNEL_SIZE][KERNEL_SIZE],
    uint8_t l1_stripes[IN_CHANNELS][STRIPE_HEIGHT][L1_STRIPE_INPUT_WIDTH + 2],
    const int8_t l2_kernels[L1_KERNELS * L2_KERNELS][KERNEL_SIZE][KERNEL_SIZE],
    uint16_t l2_stripes[L1_KERNELS][STRIPE_HEIGHT][L2_STRIPE_INPUT_WIDTH + 2],
    int32_t l3_outputs[L3_OUTPUT_WIDTH]
)
{
// pipeline consisting of 33 stages, i.e. each iteration will take 33 clock cycles, but in each clock cycle an iteration will be started and finished
#pragma HLS PIPELINE II=33 

    // state variables for the first layer
    static uint32_t l1_iteration = 0;
    static uint16_t l1_write_col_offset = 1;
    static uint8_t l1_write_row_offset = 0;
    static uint8_t l1_channel_idx = 0;
    static uint8_t l1_read_row_offset = 0;
    static uint16_t l1_read_col_offset = 0;
    static int32_t l1_maxes[L1_KERNELS] = {0, };

    // state variables for the second layer
    static uint32_t l2_iteration = 0;
    static uint16_t l2_write_col_offset = 1;
    static uint8_t l2_write_row_offset = 0;
    static uint8_t l2_channel_idx = 0;
    static uint8_t l2_read_row_offset = 0;
    static uint16_t l2_read_col_offset = 0;
    static int32_t l2_maxes[2][L2_KERNELS] = {{0, }, };
    static bool l2_maxes_idx = 0;
    static int32_t l2_channel_sums[L2_KERNELS] = {0, };

    // state variable for the third layer
    static int32_t l3_iteration = -L2_KERNELS;

    // Input reading/loading:
    // - input is read 2 rows at a time (2 * 256 * 3 = 1536 bytes) and placed to a cyclic buffer containing 6 rows,
    //   2 of which are currently loaded and the remaining 4 (later referred as stripe) are processed.
    if ((l1_iteration & ITERATION_MASK) < ((2 * L1_STRIPE_INPUT_WIDTH * IN_CHANNELS) / 8)) // 2 rows have not been read yet
    {
        int high = 7;
        int low = 0;
        axis_in_t in_data = in.read(); // read 8 bytes
        for (int k = 0; k < 8; k++)    // distribute the read bytes across the 3 input color channels (RGB)
        {
            l1_stripes[l1_channel_idx][l1_write_row_offset][l1_write_col_offset] = in_data.data.range(high, low);
            l1_channel_idx++;
            if (l1_channel_idx == IN_CHANNELS) // move back to the first channel
            {
                l1_channel_idx = 0;
                l1_write_col_offset++;
            }
            high += 8;
            low += 8;
        }

        if ((l1_iteration & ITERATION_MASK) == (L1_STRIPE_INPUT_WIDTH * IN_CHANNELS / 8 - 1)) // 1 whole row read 
        {
            l1_write_row_offset += 1; // move to the next row
            l1_write_col_offset = 1;  // move to the 2nd column, 1st and last columns are 0 padding
        }
    }
    else if ((l1_iteration & ITERATION_MASK) == ITERATION_MASK) // one iteration of processing of the rows has finished, reading of the next 2 rows can start again
    {
        l1_write_col_offset = 1;  // move to the 2nd column, 1st and last columns are 0 padding
        l1_write_row_offset += 1; // move to the next row
        if (l1_write_row_offset == STRIPE_HEIGHT) // end of the cyclic buffer
        {
            l1_write_row_offset = 0; // cycle back to the first row of the buffer
        }
    }

    // First layer processing:
    // - each iteration results in 1 output value for each output channel, 
    // - in every 4th iteration max pooling of a 2x2 region is performed and the result is written to the next layer cyclic input buffer,
    // - in every 4th iteration the column index to the input buffer is incremented by 2, i.e. the 2x2 region is moved by 2 columns.
    if (l1_iteration >= 2 * ITERATION_MULTIPLE) // wait until 4 rows are read 
    {
        // Convolution is performed in the 2x2 region as follows:
        // 1. left upper corner,
        // 2. right upper corner,
        // 3. left lower corner,
        // 4. right lower corner.
        bool top_offset = l1_iteration & 2;   // ensure top offset is 0 for the first 2 iterations
        bool left_offset = l1_iteration & 1;  // ensure left offset is 0 for every odd iteration (iterations start from 0)
        uint16_t local_col_index = l1_read_col_offset + left_offset;

        int32_t kernel_sums[IN_CHANNELS][L1_KERNELS] = {{0, }, }; // results of the convolution between each input channel and each kernel
        int32_t channel_sums[L1_KERNELS] = {0,};                  // the final result of the convolution for each kernel, i.e the output channels
    // partition the arrays in memory to allow parallel access
    #pragma HLS ARRAY_PARTITION variable=partial_sums complete
    #pragma HLS ARRAY_PARTITION variable=channel_sums complete

        for (int l = 0; l < KERNEL_SIZE; l++) // move across the height of the convolutional filter
        {
            uint8_t row_idx = l1_read_row_offset + l + top_offset; // offset of the input patch in the heigh dimension of the input buffer
            if (row_idx >= STRIPE_HEIGHT) // row index out of range
            {
                row_idx -= STRIPE_HEIGHT; // cycle back at the beginning of the buffer
            }

            for (int m = 0; m < KERNEL_SIZE; m++) // move across the width of the convolutional filter
            {
                uint16_t col_idx = local_col_index + m;
                // the following for loops will be completely unrolled and the operations will be performed in parallel in the hardware
                for (int j = 0; j < IN_CHANNELS; j++) // convolve all input channels
                {
                    for (int k = 0; k < L1_KERNELS; k++) // with all kernels in each filter
                    {
                        // convolution operation
                        kernel_sums[j][k] += l1_kernels[j * L1_KERNELS + k][l][m] * l1_stripes[j][row_idx][col_idx];
                    }
                }
            }
        }

        for (int j = 0; j < IN_CHANNELS; j++)
        {
            for (int k = 0; k < L1_KERNELS; k++)
            {
                channel_sums[k] += kernel_sums[j][k]; // accumulate the results of the convolution in each output channel
            }
        }

        for (int j = 0; j < L1_KERNELS; j++) // max pooling, the maximum is updated in each of the 4 iterations
        {
            l1_maxes[j] = channel_sums[j] > l1_maxes[j] ? channel_sums[j] : l1_maxes[j];
        }

        if ((l1_iteration & L1_OUTPUT_WRITE_MASK) == L1_OUTPUT_WRITE_MASK) // every 4th iteration
        {
            for (int k = 0; k < L1_KERNELS; k++) // write to the input buffer of the next layer, all channels are written in parallel
            {
                // shift the decimal point to simulate floating point arithmetic
                l2_stripes[k][l2_write_row_offset][l2_write_col_offset] = l1_maxes[k] >> L1_OUTPUT_SHIFT;
                l1_maxes[k] = 0; // reset the value in the output buffer of the first layer, 0 ensures ReLU activation function
            }

            l2_write_col_offset++; // move to the next column in the input buffer of the next layer
            if (l2_write_col_offset == L2_STRIPE_INPUT_WIDTH + 1) // whole output row written
            {
                l2_write_col_offset = 1; // move to the 2nd column, 1st and last columns are 0 padding
                l2_write_row_offset++;   // move to the next row
                if (l2_write_row_offset == STRIPE_HEIGHT) // end of the cyclic buffer
                {
                    l2_write_row_offset = 0; // cycle back to the first row of the buffer
                }
            }
            
            l1_read_col_offset += 2; // move to the next 2x2 region in the height direction
            if (l1_read_col_offset == L1_STRIPE_INPUT_WIDTH) // whole input stripe processed
            {
                l1_read_col_offset = 0;  // move to the first column (1st value is 0 padding)
                l1_read_row_offset += 2; // move to the next 2x2 region in the width direction
                if (l1_read_row_offset == STRIPE_HEIGHT) // end of the cyclic buffer
                {
                    l1_read_row_offset = 0; // cycle back to the first row of the buffer
                }
            }
        }
    }

    // Second layer processing:
    // - every 2nd iteration results in 1 output value for each output channel, this can be done because there are half as many inputs as in the first layer,
    // - every 8th iteration max pooling of a 2x2 region is performed and the result is written to the next layer cyclic input buffer.
    if (l2_iteration >= 6 * ITERATION_MULTIPLE && !(l2_iteration & ITERATION_MULTIPLE)) // wait until the first produces 4 output rows
    {
        uint8_t channel_offset = l2_iteration & 1 ? (L1_KERNELS / 2) : 0; // process first 2 channels in even iterations and remaining 2 in odd iterations
        bool top_offset = l2_iteration & 4;  // top offset is 0 for the first 4 iterations
        bool left_offset = l2_iteration & 2; // left offset is 0 for iterations 0, 1, 4, 5 and 1 for iterations 2, 3, 6, 7
        uint16_t local_col_index = l2_read_col_offset + left_offset;

        for (int l = 0; l < KERNEL_SIZE; l++)
        {
            uint8_t row_idx = l2_read_row_offset + l + top_offset;
            if (row_idx >= STRIPE_HEIGHT)
            {
                row_idx -= STRIPE_HEIGHT;
            }
            for (int m = 0; m < KERNEL_SIZE; m++)
            {
                for (int j = 0; j < L1_KERNELS / 2; j++)
                {
                    for (int k = 0; k < L2_KERNELS; k++)
                    {
                        // convolution operation
                        l2_channel_sums[k] += l2_kernels[(j + channel_offset) * L2_KERNELS + k][l][m] * l2_stripes[j + channel_offset][row_idx][local_col_index + m];
                    }
                }
            }
        }

        if (l2_iteration & 1) // every 2nd iteration
        {
            for (int j = 0; j < L2_KERNELS; j++)
            {
                // update the maximum value in each output channel, i.e. ReLU activation function
                l2_maxes[l2_maxes_idx][j] = l2_channel_sums[j] > l2_maxes[l2_maxes_idx][j] ? l2_channel_sums[j] : l2_maxes[l2_maxes_idx][j];
                l2_channel_sums[j] = 0;
            }
        }
                
        if ((l2_iteration & L2_OUTPUT_WRITE_MASK) == L2_OUTPUT_WRITE_MASK) // every 4th iteration
        {
            l2_read_col_offset += 2; // move to the next 2x2 region in the height direction
            if (l2_read_col_offset == L2_STRIPE_INPUT_WIDTH) // whole input stripe processed
            {
                l2_read_col_offset = 0;  // move to the first column (1st value is 0 padding)
                l2_read_row_offset += 2; // move to the next 2x2 region in the width direction
                if (l2_read_row_offset == STRIPE_HEIGHT) // end of the cyclic buffer
                {
                    l2_read_row_offset = 0; // cycle back to the first row of the buffer
                }
            }
        }
    }

    // Third layer processing:
    // - each iteration an outer product by a row of the weights and 1 output value of the 2nd layer is performed,
    // - every 8th iteration all the output channels of the 2nd layer are processed.
    if (l3_iteration >= 6 * ITERATION_MULTIPLE && !(l3_iteration & ITERATION_MULTIPLE))
    {
        int maxes_idx = l3_iteration & L2_OUTPUT_WRITE_MASK;    // get the position in the 2nd layer ping-pong output buffer
        l2_maxes[!l2_maxes_idx][maxes_idx] >>= L2_OUTPUT_SHIFT; // shift the decimal point to simulate floating point arithmetic
        axis_weights_t weights_data = weights.read();           // read the periodically sent weights (to large to be stored in BRAM), FIXME: use direct DDR access
        for (int j = 0; j < L3_OUTPUT_WIDTH; j++) // this loop will be completely unrolled and the operations will be performed in parallel in the hardware
        {
            int8_t weight = weights_data.data.range(j * 8 + 7, j * 8);
            l3_outputs[j] += l2_maxes[!l2_maxes_idx][maxes_idx] * weight; // outer product
        }
        l2_maxes[!l2_maxes_idx][maxes_idx] = 0; // reset the value in the 2nd layer output buffer, ensure that max is never negative, i.e. ReLU activation function
    }

    if ((l2_iteration & L2_OUTPUT_WRITE_MASK) == L2_OUTPUT_WRITE_MASK)
    {
        l2_maxes_idx = !l2_maxes_idx; // switch the ping-pong buffer, i.e. second layer writes to "ping" and the third layer reads from "pong"
    }

    l1_iteration++;
    if (l1_iteration == (L2_STRIPE_INPUT_WIDTH + 1) * ITERATION_MULTIPLE) // whole input image processed by the first layer
    {
        l1_iteration = ITERATION_MULTIPLE; // 2 rows of the next image are already read, set the iteration such that the next 2 rows can be read before processing starts
        l1_read_row_offset += 2;
        l1_read_row_offset = l1_read_row_offset == STRIPE_HEIGHT ? 0 : l1_read_row_offset;
        l2_write_row_offset--;
    }

    l2_iteration++;
    if (l2_iteration == (L2_STRIPE_INPUT_WIDTH + 2) * ITERATION_MULTIPLE) // whole input image processed by the second layer
    {
        l2_iteration = 2 * ITERATION_MULTIPLE; // 2 rows of the next image are already processed by the first layer
        l2_read_row_offset = 0;
    }

    // Third layer output:
    // - the output of the FPGA accelerated part of the CNN, the last layer is computed on the CPU (just a 16 by 16 vector dot product).
    l3_iteration++;
    if (l3_iteration == (L2_STRIPE_INPUT_WIDTH + 2) * ITERATION_MULTIPLE) // whole input image processed by the third layer
    {
        l3_iteration = 2 * ITERATION_MULTIPLE;
        
        axis_out_t out_data;
        out_data.keep = -1;
        out_data.last = 1;
        for (int j = 0; j < L3_OUTPUT_WIDTH; j++)
        {
            l3_outputs[j] = l3_outputs[j] > 0 ? l3_outputs[j] : 0; // ReLU activation function
            out_data.data.range(j * 32 + 31, j * 32) = l3_outputs[j] >> L3_OUTPUT_SHIFT; // shift the decimal point to simulate floating point arithmetic
            l3_outputs[j] = 0; // reset the output buffer of the third layer
        }
        out.write(out_data); // write the output to the output stream
    }
    else if (l3_iteration > 2 * ITERATION_MULTIPLE && l3_iteration < 2 * ITERATION_MULTIPLE + 8) // write padding to the output stream, 
    {                                                                                            // chunks of at least 128 bytes need to be sent
        axis_out_t out_data;
        out_data.keep = -1;
        out_data.last = 0;
        for (int j = 0; j < L3_OUTPUT_WIDTH; j++)
        {
            out_data.data.range(j * 32 + 31, j * 32) = -1; // padding with -1, which is not a valid output value because of the ReLU activation function
        }
        out.write(out_data);
    }
}

void cnn(hls::stream<axis_in_t> &in, hls::stream<axis_weights_t> &weights, hls::stream<axis_out_t> &out)
{
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=weights
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE ap_ctrl_none port=return

    static const int8_t l1_kernels[IN_CHANNELS * L1_KERNELS][KERNEL_SIZE][KERNEL_SIZE] = KERNEL_WEIGHTS_L1; // kernel weights of the first layer
    static const int8_t l2_kernels[L1_KERNELS * L2_KERNELS][KERNEL_SIZE][KERNEL_SIZE] = KERNEL_WEIGHTS_L2;  // kernel weights of the second layer
#pragma HLS ARRAY_PARTITION variable=l1_kernels complete dim=1
#pragma HLS ARRAY_PARTITION variable=l2_kernels complete dim=1

    static uint8_t  l1_stripes[IN_CHANNELS][STRIPE_HEIGHT][L1_STRIPE_INPUT_WIDTH + 2] = {{0, } }; // cyclic buffer for the input of the first layer
    static uint16_t l2_stripes[L1_KERNELS][STRIPE_HEIGHT][L2_STRIPE_INPUT_WIDTH + 2] = {{0, } };  // cyclic buffer for the input of the second layer
    static int32_t  l3_outputs[L3_OUTPUT_WIDTH] = {0, };                                          // output buffer of the third layer
// ensure correct mapping to recourses and memory partitioning to allow parallel access
#pragma HLS RESOURCE variable=l1_stripes core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=l1_stripes complete dim=1
#pragma HLS ARRAY_PARTITION variable=l1_stripes complete dim=2
#pragma HLS RESET variable=l1_stripes
#pragma HLS RESOURCE variable=l2_stripes core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=l2_stripes complete dim=1
#pragma HLS ARRAY_PARTITION variable=l2_stripes complete dim=2
#pragma HLS RESET variable=l2_stripes
#pragma HLS ARRAY_PARTITION variable=l3_outputs complete
#pragma HLS RESET variable=l3_outputs

#pragma HLS PIPELINE II=36
    kernel(in, weights, out, l1_kernels, l1_stripes, l2_kernels, l2_stripes, l3_outputs);
}
