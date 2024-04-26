#include "cnn.h"
#include "cnn_tb.h"
#include "cnn_test_data.h"

int main()
{
    int32_t output[L3_OUTPUT_SIZE * 8 + L3_OUTPUT_SIZE];
    hls::stream<axis_in_t> in_stream;
    hls::stream<axis_weights_t> weights_stream;
    hls::stream<axis_out_t> out_stream;
    int valid_count = 0;
    bool failed = false;
    int output_idx = 0;
    int weights_idx = 0;
    int predictions_idx = 0;

    for (int m = 0; m < NUMBER_OF_INPUTS; m++)
    {
        cout << "Starting iteration " << m << endl;
        for (int n = 0; n < L2_INPUT_WIDTH + 2*(m == NUMBER_OF_INPUTS); n++)
        {
            int in_shift = n * INPUT_VALUES_PER_ITERATION;

            for (int i = 0; i < INPUT_VALUES_PER_ITERATION; i += 8)
            {
                axis_in_t in1;
                
                in1.data.range(7, 0)   = inputs[m][in_shift + i];
                in1.data.range(15, 8)  = inputs[m][in_shift + i+1];
                in1.data.range(23, 16) = inputs[m][in_shift + i+2];
                in1.data.range(31, 24) = inputs[m][in_shift + i+3];
                in1.data.range(39, 32) = inputs[m][in_shift + i+4];
                in1.data.range(47, 40) = inputs[m][in_shift + i+5];
                in1.data.range(55, 48) = inputs[m][in_shift + i+6];
                in1.data.range(63, 56) = inputs[m][in_shift + i+7];

                in_stream.write(in1);
            }

            if (n > 1 && n < 126)
            {
                for (int i = 0; i < WEIGHT_ROWS_PER_ITERATION; i++)
                {
                    axis_weights_t weights;
                    int high = 7;
                    int low = 0;
                    for (int j = 0; j < WEIGHTS_PER_ROW; j++)
                    {
                        weights.data.range(high, low) = l3_weights[weights_idx++];
                        high += 8;
                        low += 8;
                    }
                    weights_stream.write(weights);
                }
            }

            if (L3_OUTPUT_SIZE * L3_INPUT_HEIGHT * L3_INPUT_WIDTH * L2_KERNELS == weights_idx)
            {
                weights_idx = 0;
                cout << "Resetting weights index at: " << m << " " << n << endl;
            }

            for (int i = 0; i < NUMBER_OF_CNN_CALLS; i++)
            {   
                cnn(in_stream, weights_stream, out_stream);
            }

            if (!out_stream.empty())
            {
                axis_out_t out;
                out = out_stream.read();
                if (out.keep)
                {
                    int high = 31;
                    int low = 0;
                    for (int k = 0; k < L3_OUTPUT_SIZE; k++)
                    {
                        output[output_idx++] = out.data.range(high, low);
                        high += 32;
                        low += 32;
                    }
                }

                if (out.last)
                {
                    bool failed_iteration = false;
                    std::cout << "Last index: " << output_idx << ", expected last index: " << L3_OUTPUT_SIZE << std::endl;
                    output_idx = 0;
                    int error_count = 0;
                    for (int i = 0; i < L3_OUTPUT_SIZE; i++)
                    {
                        int output_idx = i + 112;
                        if (output[output_idx] != predictions[predictions_idx][i])
                        {
                            std::cout << "FAILED AT " << i << ":\tExpected - " << predictions[predictions_idx][i] << "\t Actual - " << output[output_idx] << std::endl;
                            failed = true;
                            failed_iteration = true;
                            error_count++;
                            if (error_count > 100)
                            {
                                break;
                            }
                        }
                        else
                        {
                            std::cout << "Passed at " << i << ":\tExpected - " << predictions[predictions_idx][i] << "\t Actual - " << output[output_idx] << std::endl;
                        }
                    }

                    if (failed_iteration)
                    {
                        std::cout << "Failed at iteration " << m << std::endl;
                    }
                    else
                    {
                        std::cout << "Passed at iteration " << m << std::endl;
                    }
                    std::cout << "----------------------------------------------\n" << std::endl;

                    predictions_idx++;
                }
            }
        }
    }

    if (failed)
    {
        std::cout << "Failed" << std::endl;
    }
    else
    {
        std::cout << "Passed" << std::endl;
    }
     
    return failed;
}