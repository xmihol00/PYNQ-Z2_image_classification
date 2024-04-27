import numpy as np
import time
from PIL import Image
import glob
from IPython.display import display, clear_output
from pynq import Clocks, Overlay, allocate

INPUT_WIDTH = 256
INPUT_HEIGHT = 256
CHANNELS = 3
OUTPUT_WIDTH = 16
WEIGHTS_HEIGHT = 31744
WEIGHTS_WIDTH = 16

print("Loading FPGA configuration", flush=True)
overlay = Overlay("./cnn.bit")
print("FPGA configuration loaded", flush=True)

print(f"CPU  running at: {Clocks.cpu_mhz} MHZ")
print(f"FPGA running at: {Clocks.fclk0_mhz} MHZ")

dma_inputs = overlay.dma1   # note that the name of the DMA in the block design is 'dma_in_out', this is an older version of the overlay
dma_weights = overlay.dma2  # note that the name of the DMA in the block design is 'dma_weights', this is an older version of the overlay

print("Allocating buffers", flush=True)
in_buffers = [ # use one input buffer to load next image and second to transfer data to FPGA, i.e. hide DMA transfer time
    allocate(shape=(INPUT_HEIGHT * INPUT_WIDTH * CHANNELS), dtype=np.uint8),
    allocate(shape=(INPUT_HEIGHT * INPUT_WIDTH * CHANNELS), dtype=np.uint8)
]
l3_weights_buffer = allocate(shape=(WEIGHTS_HEIGHT * WEIGHTS_WIDTH), dtype=np.int8)
out_buffer = allocate(shape=(OUTPUT_WIDTH * 8), dtype=np.int32)
print("Buffers allocated", flush=True)

in_ping_pong = False # ping-pong index to switch between input buffers

print("Loading weights", flush=True)
l4_weights = np.load("l4_weights.npy")
l3_weights_buffer[:] = np.load("l3_weights.npy").flatten()
print("Weights loaded", flush=True)

outputs = np.zeros((OUTPUT_WIDTH))

start = time.time()
in_buffers[in_ping_pong][:] = 0

cats = glob.glob("./dataset/cats_and_dogs_256x256/train/cat/*.jpg")
cats = list(zip(cats, ["CAT"] * len(cats)))

dogs = glob.glob("./dataset/cats_and_dogs_256x256/train/dog/*.jpg")
dogs = list(zip(dogs, ["DOG"] * len(dogs)))

images = cats + dogs
np.random.shuffle(images)

print("Initiating transfer with empty buffer", flush=True)
dma_inputs.sendchannel.transfer(in_buffers[in_ping_pong])
dma_weights.sendchannel.transfer(l3_weights_buffer)

MAX_ITERATIONS = 10
CLEAR_OUTPUT = False # set to True for larger number of iterations
accuracy = 0
counter = 0
last_expected_class = "DOG"
start = time.time()
for (sample, expected_class), i in zip(images, range(MAX_ITERATIONS)):       
    in_ping_pong = not in_ping_pong
    dma_inputs.recvchannel.transfer(out_buffer)
    
    in_buffers[in_ping_pong][:] = np.array(Image.open(sample)).flatten()

    dma_inputs.sendchannel.wait()
    dma_inputs.sendchannel.transfer(in_buffers[in_ping_pong])
    
    dma_weights.sendchannel.wait()
    dma_weights.sendchannel.transfer(l3_weights_buffer)

    dma_inputs.recvchannel.wait()
    outputs[:] = out_buffer[7 * OUTPUT_WIDTH : 8 * OUTPUT_WIDTH]
    prediction = np.dot(outputs.reshape((1, -1)), l4_weights) # compute the last layer of the network on the CPU

    time_taken = time.time() - start
    classified = "DOG" if prediction[0, 0] >= 0 else "CAT"
    accuracy += classified == last_expected_class
    last_expected_class = expected_class
    counter += 1

time_taken = time.time() - start
print(f"Classified {counter} images in {time_taken} seconds with {counter / time_taken} FPS")
print(f"Accuracy: {accuracy / MAX_ITERATIONS * 100} %")

dma_inputs.recvchannel.transfer(out_buffer)

# Free the resources
time.sleep(1)
in_buffers[0].freebuffer()
in_buffers[1].freebuffer()
out_buffer.freebuffer()