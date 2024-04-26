import os
from net import get_model
import numpy as np

KERNEL_SIZE = 3
IN_CHANNELS = 3
L1_OUT_CHANNELS = 4
L2_OUT_CHANNELS = 8

def format_array_C(arr):
    if len(arr.shape) == 1:
        return f"{{ {', '.join(map(lambda x: f'{x}', arr))} }}"
    
    if len(arr.shape) == 2:
        lines = ['{' + ', '.join(map(lambda x: f"{x: 4d}", line)) + '}' for line in arr]
        new_line = ',\\\n        '
        return f"{{\\\n        {new_line.join(lines)}\\\n    }}"

    if len(arr.shape) == 3:
        row_map = lambda x: f"{{ {', '.join(map(lambda y: f'{y: 4d}', x))} }}"
        lines = ['{' + ', '.join(map(row_map, line)) + '}' for line in arr]
        new_line = ',\\\n        '
        return f"{{\\\n        {new_line.join(lines)}\\\n    }}"

model = get_model()
model.load_weights("../model/cats_dogs_net.h5")

# get just the weights biases are not used
l1_kernels = model.layers[1].get_weights()[0]
l2_kernels = model.layers[4].get_weights()[0]
l3_weights = model.layers[7].get_weights()[0]
l4_weights = model.layers[8].get_weights()[0]

l1_max = np.max(np.abs(l1_kernels))
l2_max = np.max(np.abs(l2_kernels))
l3_max = np.max(np.abs(l3_weights))
l4_max = np.max(np.abs(l4_weights))

l1_quantized = l1_kernels / l1_max * 127
l2_quantized = l2_kernels / l2_max * 127
l3_quantized = l3_weights / l3_max * 127
l4_quantized = l4_weights / l4_max * 127

l1_kernels = l1_quantized.reshape(KERNEL_SIZE * KERNEL_SIZE, -1).T.astype(np.int8)
l2_kernels = l2_quantized.reshape(KERNEL_SIZE * KERNEL_SIZE, -1).T.astype(np.int8)
l3_weights = l3_quantized.astype(np.int8)
l4_weights = l4_quantized.astype(np.int8)

os.makedirs("../weights", exist_ok=True)
np.save("../weights/l3_weights.npy", l3_weights)
np.save("../weights/l4_weights.npy", l4_weights)
with open("../weights/weights.h", "w") as f:
    f.write("#define KERNEL_WEIGHTS_L1 ")
    f.write(format_array_C(l1_kernels.reshape(IN_CHANNELS * L1_OUT_CHANNELS, 3, 3)))
    f.write("\n\n")
    f.write("#define KERNEL_WEIGHTS_L2 ")
    f.write(format_array_C(l2_kernels.reshape(L1_OUT_CHANNELS * L2_OUT_CHANNELS, 3, 3)))
    f.write("\n")
