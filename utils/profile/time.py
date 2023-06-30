import time
import numpy as np
import torch


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def log_time_consuming(time_consuming):
    print("=" * 20, " Time Consuming ", "=" * 20)
    print(f"body_detection: {1e3 * np.mean(time_consuming['body_detection'][10:]):.2f} ms")
    print(f"hand_detection: {1e3 * np.mean(time_consuming['hand_detection'][10:]):.2f} ms")
    print(f"body3d: {1e3 * np.mean(time_consuming['body3d'][10:]):.2f} ms")
    print(f"hand3d: {1e3 * np.mean(time_consuming['hand3d'][10:]):.2f} ms")
    print(f"fusion: {1e3 * np.mean(time_consuming['fusion'][10:]):.2f} ms")
    print(f"smooth: {1e3 * np.mean(time_consuming['smooth'][10:]):.2f} ms")
    print(f"retarget: {1e3 * np.mean(time_consuming['retarget'][10:]):.2f} ms")
    print(f"render: {1e3 * np.mean(time_consuming['render'][10:]):.2f} ms")
    print(f"all: {1e3 * np.mean(time_consuming['all'][10:]):.2f} ms")
