#ifndef _CNN_H_
#define _CNN_H_

#include <inttypes.h>
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "stdlib.h"
#ifndef __SYNTHESIS__
	#include <iostream>
    using namespace std;
#endif

// NN parameters
#define STRIPE_HEIGHT 6
#define STRIPE_OUTPUT_HEIGHT ((STRIPE_HEIGHT - 2) / 2)
#define L1_STRIPE_INPUT_WIDTH 256
#define L2_STRIPE_INPUT_WIDTH (L1_STRIPE_INPUT_WIDTH / 2)
#define L3_OUTPUT_WIDTH 16
#define AXI_INPUT_WIDTH (8 * 8)
#define AXI_WEIGHTS_WIDTH (16 * 8)
#define AXI_OUTPUT_WIDTH (16 * 32)
#define INT8_BITS 8
#define INT16_BITS 16
#define IN_CHANNELS 3
#define L1_KERNELS 4
#define L2_KERNELS 8
#define KERNEL_SIZE 3
#define OUTPUT_BATCH_SIZE 4
#define L1_OUTPUT_WRITE_MASK 0b11
#define L2_OUTPUT_WRITE_MASK 0b111
#define L1_OUTPUT_SHIFT 8
#define L2_OUTPUT_SHIFT 8
#define L3_OUTPUT_SHIFT 8
#define ITERATION_MULTIPLE (2 * L1_STRIPE_INPUT_WIDTH)
#define ITERATION_MASK (ITERATION_MULTIPLE - 1)

typedef ap_axiu<AXI_INPUT_WIDTH, 0, 0, 0> axis_in_t;
typedef ap_axiu<AXI_WEIGHTS_WIDTH, 0, 0, 0> axis_weights_t;
typedef ap_axiu<AXI_OUTPUT_WIDTH, 0, 0, 0> axis_out_t;

void cnn(hls::stream<axis_in_t> &in, hls::stream<axis_weights_t> &weights, hls::stream<axis_out_t> &out);

// trained kernel weights of the 1st layer
#define KERNEL_WEIGHTS_L1 {\
        {{    7,  -57,  -20 }, {   15,   28,  101 }, {   55,   82,   34 }},\
        {{   53,   18,  -44 }, {  -95,   25,   60 }, {   70,  -18,   11 }},\
        {{   82,  -46,   45 }, {   80,   43,  -63 }, {  -19,  -35,   31 }},\
        {{  -38,   -7,  -31 }, {   68,  -31,    5 }, {  -20,   66,  -26 }},\
        {{   15,   61,  -62 }, {   38,   91,   55 }, {  111,  110,   17 }},\
        {{   64,   -8,  -26 }, {  -68,  -96,    2 }, {  -58,   61,  -52 }},\
        {{   46,    8,  -11 }, {  -14,    0,  -11 }, {  -69,  -68,  -51 }},\
        {{  -48,   72,  -35 }, {   61,   14,   25 }, {   78,   24,   37 }},\
        {{  111,  -15,   17 }, {  -16,   40,  -49 }, {   76,  127,   76 }},\
        {{  -97,  -42,  -35 }, {   62,   48,    4 }, {   69,   79,   79 }},\
        {{   94,   29,  -21 }, {    0,   64,  -72 }, {   60,   56,  -37 }},\
        {{  -83,   -2,   38 }, {   42,    3,  -49 }, {   -9,   56,   13 }}\
    }

// trained kernel weights of the 2nd layer
#define KERNEL_WEIGHTS_L2 {\
        {{   29, -114,   50 }, {   39,  -43,  -53 }, {   87,   40,  -80 }},\
        {{   15,  -77,  -26 }, {  -81,   61,  -55 }, { -107,   32, -103 }},\
        {{  -20,  -29,   90 }, {   79,  -28,   -1 }, {  -88,   -1,  -48 }},\
        {{  -82,  111,   97 }, { -118,   67,   55 }, {  -95,   21,  -79 }},\
        {{  114,  -54,  -62 }, {   14, -127,   49 }, {  -50,  -49,   89 }},\
        {{  -71,  -91,   16 }, { -105,   10,   31 }, {   -7,  103,   84 }},\
        {{  106,   75,   24 }, {  -67,   -1, -102 }, {   -3,  -79,  -73 }},\
        {{   91,  -20, -104 }, {   53,  -86,   25 }, {  -85,   45,   81 }},\
        {{  -20,    9,   39 }, { -107,  -34,  -92 }, {  -68, -108, -108 }},\
        {{   37,   94,    9 }, {  -22,  -33,   40 }, {   94,  -62,   42 }},\
        {{   95,  -78,   -4 }, {  -81,   86,   22 }, {   85,   -8,   67 }},\
        {{  -26,  -99,  -22 }, {   -7,  -55,    3 }, {  -55,   36,   87 }},\
        {{   42,   13,  -49 }, {  -37,  -42,   -7 }, {  -75,   32,   77 }},\
        {{  -26,   90,   40 }, {   12,  -30,   90 }, {  -30,   10,   -2 }},\
        {{  -19,  -62,  -37 }, {   30,   10,  -15 }, {  -40,   74,   -4 }},\
        {{    8,  -40,   43 }, {  -52,  108,  103 }, {  -11,  -74,   88 }},\
        {{   91,    1,  -96 }, {  -76,   53,  -54 }, {   79,    7,  -71 }},\
        {{  -86,   87,  -65 }, {  -55,  -27,   46 }, {    3,  -28,   54 }},\
        {{   18,  -68,  -59 }, {   -9,   80,  -24 }, {  -96,  -61,  -13 }},\
        {{   11,   29,   -4 }, {   46,   62,   -1 }, {    5,  -38,  -19 }},\
        {{   71,   12,  -13 }, {  113,  -51,   37 }, {  -19,  -99,  -49 }},\
        {{   -9,  -69, -118 }, {   23,   20,   21 }, {  -84,  -42,   51 }},\
        {{  -89,  -52,  -46 }, {  -31,  -47,  -82 }, {   31,  103,   78 }},\
        {{   -4,  -47,   23 }, { -102,    4,   19 }, {    6,   90,   12 }},\
        {{   60,  -90,   43 }, {   54,  -19, -103 }, {  103,   24,  -22 }},\
        {{   40,   41,  -61 }, {   32,  -42,   37 }, {   72,  -74,   54 }},\
        {{  -46,   43,  -36 }, {   74,   26,  -57 }, {   38,   16,   88 }},\
        {{ -115,    2,   30 }, {  -15,  -91,   63 }, {  -76,   89,  -51 }},\
        {{   88,   50,  -31 }, {  -62,  -78,  -18 }, {   43,   62,   96 }},\
        {{  -43,    4,   89 }, {   62,  -56,   36 }, {   22,  -64,   23 }},\
        {{  -10,  -51,  -94 }, {   49,   -4,  -24 }, {  -14,  -98,   15 }},\
        {{ -107,   34,    7 }, {  -16,  -22,   12 }, {   59,   -8,   55 }}\
    }

#endif