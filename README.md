# PYNQ-Z2 Image Classification
Binary image classification with a CNN implemented as an FSM on an FPGA. See the `documentation` directory and read the comments in `hls_src/cnn.cpp` for more information.

## Repository structure
```
├── dataset/            - directory containing the dataset used for the CNN training (must be downloaded and pre-processed, see the directory)
├── documentation/      - the documentation of the project, read first
├── fpga_config/        - configuration for the FPGA to run the CNN on the PYNQ-Z2 board
├── hls_src/            - C++ source files of the CNN for FPGA
├── model/              - the trained model in Tensorflow
├── py_src/             - Python source files used for data processing, training and weights extraction from the trained model
├── weights/            - the extracted weights used by the FPGA CNN
├── demo_FPGA.ipynb     - demo of the CNN running on the FPGA
├── demo_math.ipynb     - demo of the CNN inference on quantized weights in integer arithmetics 
├── demo_model.ipynb    - demo of the CNN inference using the trained Tensorflow model
└── README.md
```
