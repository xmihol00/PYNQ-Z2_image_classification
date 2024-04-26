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
#pragma HLS PIPELINE II=33

    static uint32_t l1_iteration = 0;
    static uint16_t l1_write_col_offset = 1;
    static uint8_t l1_write_row_offset = 0;
    static uint8_t l1_channel_idx = 0;
    static uint8_t l1_read_row_offset = 0;
    static uint16_t l1_read_col_offset = 0;
    static int32_t l1_maxes[L1_KERNELS] = {0, };

    static uint32_t l2_iteration = 0;
    static uint16_t l2_write_col_offset = 1;
    static uint8_t l2_write_row_offset = 0;
    static uint8_t l2_channel_idx = 0;
    static uint8_t l2_read_row_offset = 0;
    static uint16_t l2_read_col_offset = 0;
    static int32_t l2_maxes[2][L2_KERNELS] = {{0, }, };
    static bool l2_maxes_idx = 0;
    static int32_t l2_kernel_sums[L2_KERNELS] = {0, };

    static int32_t l3_iteration = -L2_KERNELS;

    if ((l1_iteration & ITERATION_MASK) < (L1_STRIPE_INPUT_WIDTH * IN_CHANNELS / 4))
    {
        int high = 7;
        int low = 0;
        axis_in_t in_data = in.read();
        for (int k = 0; k < 8; k++)
        {
            l1_stripes[l1_channel_idx][l1_write_row_offset][l1_write_col_offset] = in_data.data.range(high, low);
            l1_channel_idx++;
            if (l1_channel_idx == IN_CHANNELS)
            {
                l1_channel_idx = 0;
                l1_write_col_offset++;
            }
            high += 8;
            low += 8;
        }

        if ((l1_iteration & ITERATION_MASK) == (L1_STRIPE_INPUT_WIDTH * IN_CHANNELS / 8 - 1))
        {
            l1_write_row_offset += 1;
            l1_write_col_offset = 1;
        }
    }
    else if ((l1_iteration & ITERATION_MASK) == ITERATION_MASK)
    {
        l1_write_col_offset = 1;
        l1_write_row_offset += 1;
        if (l1_write_row_offset == STRIPE_HEIGHT)
        {
            l1_write_row_offset = 0;
        }
    }

    if (l1_iteration >= 2 * ITERATION_MULTIPLE)
    {   
        bool top_offset = l1_iteration & 2;
        bool left_offset = l1_iteration & 1;
        uint16_t local_col_index = l1_read_col_offset + left_offset;

        int32_t partial_sums[IN_CHANNELS][L1_KERNELS] = {{0, }, };
        int32_t kernel_sums[L1_KERNELS] = {0,};
    #pragma HLS ARRAY_PARTITION variable=partial_sums complete
    #pragma HLS ARRAY_PARTITION variable=kernel_sums complete

        for (int l = 0; l < KERNEL_SIZE; l++)
        {
            uint8_t row_idx = l1_read_row_offset + l + top_offset;
            if (row_idx >= STRIPE_HEIGHT)
            {
                row_idx -= STRIPE_HEIGHT;
            }
            for (int m = 0; m < KERNEL_SIZE; m++)
            {
                uint16_t col_idx = local_col_index + m;
                for (int j = 0; j < IN_CHANNELS; j++)
                {
                #pragma HLS latency min=5
                    for (int k = 0; k < L1_KERNELS; k++)
                    {
                    #pragma HLS latency min=5
                        partial_sums[j][k] += l1_kernels[j * L1_KERNELS + k][l][m] * l1_stripes[j][row_idx][col_idx];
                    }
                }
            }
        }

        for (int j = 0; j < IN_CHANNELS; j++)
        {
        #pragma HLS latency min=5
            for (int k = 0; k < L1_KERNELS; k++)
            {
            #pragma HLS latency min=5
                kernel_sums[k] += partial_sums[j][k];
            }
        }

        for (int j = 0; j < L1_KERNELS; j++)
        {
        #pragma HLS UNROLL
            l1_maxes[j] = kernel_sums[j] > l1_maxes[j] ? kernel_sums[j] : l1_maxes[j];
        }

        if ((l1_iteration & L1_OUTPUT_WRITE_MASK) == L1_OUTPUT_WRITE_MASK)
        {
            for (int k = 0; k < L1_KERNELS; k++)
            {
                l2_stripes[k][l2_write_row_offset][l2_write_col_offset] = l1_maxes[k] >> L1_OUTPUT_SHIFT;
                l1_maxes[k] = 0;
            }

            l2_write_col_offset++;
            if (l2_write_col_offset == L2_STRIPE_INPUT_WIDTH + 1)
            {
                l2_write_col_offset = 1;
                l2_write_row_offset++;
                if (l2_write_row_offset == STRIPE_HEIGHT)
                {
                    l2_write_row_offset = 0;
                }
            }
            
            l1_read_col_offset += 2;
            if (l1_read_col_offset == L1_STRIPE_INPUT_WIDTH)
            {
                l1_read_col_offset = 0;
                l1_read_row_offset += 2;
                if (l1_read_row_offset == STRIPE_HEIGHT)
                {
                    l1_read_row_offset = 0;
                }
            }
        }
    }

    if (l2_iteration >= 6 * ITERATION_MULTIPLE && !(l2_iteration & ITERATION_MULTIPLE))
    {
        uint8_t channel_offset = l2_iteration & 1 ? 2 : 0;
        bool top_offset = l2_iteration & 4;
        bool left_offset = l2_iteration & 2;
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
                        l2_kernel_sums[k] += l2_kernels[(j + channel_offset) * L2_KERNELS + k][l][m] * l2_stripes[j + channel_offset][row_idx][local_col_index + m];
                    }
                }
            }
        }

        if (l2_iteration & 1)
        {
            for (int j = 0; j < L2_KERNELS; j++)
            {
                l2_maxes[l2_maxes_idx][j] = l2_kernel_sums[j] > l2_maxes[l2_maxes_idx][j] ? l2_kernel_sums[j] : l2_maxes[l2_maxes_idx][j];
                l2_kernel_sums[j] = 0;
            }
        }
                
        if ((l2_iteration & L2_OUTPUT_WRITE_MASK) == L2_OUTPUT_WRITE_MASK)
        {
            l2_read_col_offset += 2;
            if (l2_read_col_offset == L2_STRIPE_INPUT_WIDTH)
            {
                l2_read_col_offset = 0;
                l2_read_row_offset += 2;
                if (l2_read_row_offset == STRIPE_HEIGHT)
                {
                    l2_read_row_offset = 0;
                }
            }
        }
    }

    if (l3_iteration >= 6 * ITERATION_MULTIPLE && !(l3_iteration & ITERATION_MULTIPLE))
    {
        int maxes_idx = l3_iteration & L2_OUTPUT_WRITE_MASK;
        l2_maxes[!l2_maxes_idx][maxes_idx] >>= L2_OUTPUT_SHIFT;
        axis_weights_t weights_data = weights.read();
        #pragma HLS UNROLL
        for (int j = 0; j < L3_OUTPUT_WIDTH; j++)
        {
            int8_t weight = weights_data.data.range(j * 8 + 7, j * 8);
            l3_outputs[j] += l2_maxes[!l2_maxes_idx][maxes_idx] * weight;
        }
        l2_maxes[!l2_maxes_idx][maxes_idx] = 0;
    }

    if ((l2_iteration & L2_OUTPUT_WRITE_MASK) == L2_OUTPUT_WRITE_MASK)
    {
        l2_maxes_idx = !l2_maxes_idx;
    }

    l1_iteration++;
    if (l1_iteration == (L2_STRIPE_INPUT_WIDTH + 1) * ITERATION_MULTIPLE)
    {
        l1_iteration = ITERATION_MULTIPLE;
        l1_read_row_offset += 2;
        l1_read_row_offset = l1_read_row_offset == STRIPE_HEIGHT ? 0 : l1_read_row_offset;
        l2_write_row_offset--;
    }

    l2_iteration++;
    if (l2_iteration == (L2_STRIPE_INPUT_WIDTH + 2) * ITERATION_MULTIPLE)
    {
        l2_iteration = 2 * ITERATION_MULTIPLE;
        l2_read_row_offset = 0;
    }

    l3_iteration++;
    if (l3_iteration == (L2_STRIPE_INPUT_WIDTH + 2) * ITERATION_MULTIPLE)
    {
        l3_iteration = 2 * ITERATION_MULTIPLE;
        
        axis_out_t out_data;
        out_data.keep = -1;
        out_data.last = 1;
        for (int j = 0; j < L3_OUTPUT_WIDTH; j++)
        {
            l3_outputs[j] = l3_outputs[j] > 0 ? l3_outputs[j] : 0;
            out_data.data.range(j * 32 + 31, j * 32) = l3_outputs[j] >> L3_OUTPUT_SHIFT;
            l3_outputs[j] = 0;
        }
        out.write(out_data);
    }
    else if (l3_iteration > 2 * ITERATION_MULTIPLE && l3_iteration < 2 * ITERATION_MULTIPLE + 8)
    {
        axis_out_t out_data;
        out_data.keep = -1;
        out_data.last = 0;
        for (int j = 0; j < L3_OUTPUT_WIDTH; j++)
        {
            out_data.data.range(j * 32 + 31, j * 32) = -1;
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

    static const int8_t l1_kernels[IN_CHANNELS * L1_KERNELS][KERNEL_SIZE][KERNEL_SIZE] = KERNEL_WEIGHTS_L1;
    static const int8_t l2_kernels[L1_KERNELS * L2_KERNELS][KERNEL_SIZE][KERNEL_SIZE] = KERNEL_WEIGHTS_L2;
#pragma HLS ARRAY_PARTITION variable=l1_kernels complete dim=1
#pragma HLS ARRAY_PARTITION variable=l2_kernels complete dim=1

    static uint8_t l1_stripes[IN_CHANNELS][STRIPE_HEIGHT][L1_STRIPE_INPUT_WIDTH + 2] = {{0, } };
    static uint16_t l2_stripes[L1_KERNELS][STRIPE_HEIGHT][L2_STRIPE_INPUT_WIDTH + 2] = {{0, } };
    static int32_t l3_outputs[L3_OUTPUT_WIDTH] = {0, };
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
