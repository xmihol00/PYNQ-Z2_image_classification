# FPGA CNN Acceleration
This document presents the development process of a convolutional neural network (CNN) for an FPGA and a specific CNN architecture developed for binary classification of images on the the Zynq-7000 SoC XC7Z020-1CLG400C FPGA available on the PYNQ-Z2 board.

## Development Cycle
A simplified development cycle of a CNN used for accelerated inference on an FPGA consists of:
  1. Designing a CNN architecture, which can be accelerated with the limited resources of a given FPGA.
  2. Implementing the designed CNN model in a machine learning library of choice.
  3. Training the model on a selected dataset.
  4. Extracting the weights from the trained model.
  5. Quantizing the weights to integers.
  6. Implementing the model using a language of choice in integer arithmetics and verifying the correct weight extraction and quantization.
  7. Tuning the bit width of the quantized weights to achieve required accuracy and performance.
  8. Implementing the model as a finite state machine (FSM) for the FPGA.

## Model Architecture
The CNN model was design to process 256x256 RGB images. It is a CNN with the four following layers:
  1. 1st 2D convolutional layer:
     - 3 256x258 input channels (0 padded in the width dimension),
     - 4 filters with 3x3 kernels,
     - ReLU activation function,
     - 2x2 max pooling,
     - 4 126x128 output channels.
  2. 2nd 2D convolutional layer:
     - 4 126x130 input channels (0 padded in the width dimension),
     - 8 filters with 3x3 kernels,
     - ReLU activation function,
     - 2x2 max pooling.
     - 8 62x64 output channels.
  3. 1st fully connected layer:
     - flattened 8 62x64 channels into a 1x31744 input vector,
     - 31744x16 weight matrix,
     - ReLU activation
     - 1x16 output vector.
  4. 2nd fully connected layer:
     - 1x16 input vector,
     - 16x1 weight matrix,
     - Sigmoid activation function,
     - scalar output value between 0 and 1.

The equivalent of the above described CNN model in the Tensorflow library is the following:
```python
tensorflow.keras.models.Sequential(
    [
    # 1st convolutional layer
        tensorflow.keras.layers.ZeroPadding2D(padding=(0, 1), input_shape=(256, 256, 3)),
        tensorflow.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='valid'),
        tensorflow.keras.layers.MaxPooling2D((2, 2)),
    
    # 2nd convolutional layer
        tensorflow.keras.layers.ZeroPadding2D(padding=(0, 1)),
        tensorflow.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='valid'),
        tensorflow.keras.layers.MaxPooling2D((2, 2)),
    
    # 1st fully connected layer
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(16, activation='relu'),
    
    # 2nd fully connected layer
        tensorflow.keras.layers.Dense(1, activation='sigmoid')
    ]
)
```

## Software Inference
The SW inference, i.e. classification of images, is in general performed layer by layer, where the activation function and max pooling are considered as separate layers. Meaning that a 256x258x3 tensor representing the padded input image is first convolved by the filters in the first layer obtaining a new 254x256x4 tensor, then the activation function is applied in place, which is followed by the max pooling, resulting in an another 126x128x4 tensor. And similarly for the rest of the network. Moreover, usually a batch of images is inferred at a time to make the inference more efficient, which makes it even worse for some real-time processing.

## FPGA Inference
The HW, i.e. FPGA accelerated, inference differs from the SW inference in the following two main factors:
  1. The inference is performed on a stream of pixels passed through a pipeline of operations. Main stages of the pipeline are separated by small buffers to discretize the stream into small chunks, which can be processed, e.g. convolved by filters.
  2. A main stage of the pipeline consists of multiple fused operations, e.g. convolution, activation and max pooling, to accelerate the computation and save resources.

The CNN must be implemented as an FSM to achieve the above described streamed processing. State of the FSM in simple terms describes at what location in the input image is each main stage and what operations it needs to execute. There are two main groups of states, first the states, in which an initial part of the input is loaded into the buffers to allow discretization, and second the states, in which the loaded buffers are processed, while the already processed data in those buffers are rewritten with new parts of the input. The core operation executed in every processing state of the FSM is a 2D convolution of a filter with a patch of the input of the same width and heigh as the filter. The following C code implements this operation:
```c
// 'l1_iteration', 'l1_read_col_offset' and 'l1_read_row_offset' are the state variable
// 'l1_stripes' is the input buffer
// 'l1_kernels' are the trained weights of the convolutional filters
// 'kernel_sums' is the partial output of the convolution

bool top_offset = l1_iteration & 2;   // ensure top offset is 0 for the first 2 iterations
bool left_offset = l1_iteration & 1;  // ensure left offset is 0 for every odd iteration (iterations start from 0)
uint16_t local_col_index = l1_read_col_offset + left_offset; // offset of the input patch in the width dimension of the input buffer

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
```   

## HW-SW partitioning
The CNN is partitioned between the FPGA and CPU in a way that both of the convolutional layers and the 1st fully connected layer are processed on the FPGA and only the last fully connected layer with minimal compute requirements (dot product of two 16 element vectors) is realized on the CPU. The convolutional filters are fully stored on the FPGA in BRAM, while the weight matrix of the 1st fully connected layer is cyclicly streamed from the main DDR memory to the FPGA due to its large size.

## Inference performance
The following table summarizes the performance of the different implementations of the CNN model achieved on a desktop CPU and the PYNQ-Z2 board for sample by sample inference:
| Implementation          | Demo script                | Hardware                                                      | Average FPS | 
|-------------------------|----------------------------|---------------------------------------------------------------|-------------|
| FPGA accelerated        | `../demo_FPGA.ipynb`       | Dual ARM® Cortex™-A9 CPU @ 650 MHz, Artix™ 7 FPGA @ 83.33 MHz | **22.3**    |
| Tensorflow              | `../demo_tensorflow.ipynb` | Intel(R) Core(TM) i5-4670K CPU @ 3.40 GHz                     | 20.9        |
| SW integral arithmetics | `../demo_integral.ipynb`   | Intel(R) Core(TM) i5-4670K CPU @ 3.40 GHz                     | 18.1        |
