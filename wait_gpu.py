import GPUtil
import time, datetime
import numpy as np

def wait_gpu():
    while True:
        GPUs = GPUtil.getGPUs()
        available = GPUtil.getAvailability(GPUs, maxLoad = 0.01, maxMemory = 0.01, includeNan=False, excludeID=[], excludeUUID=[])
        if np.sum(available) == np.size(available):
            # print ("GPU available") with current date time
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"GPU available at {current_time}")
            break
        else:
            # Remove the printed line in previous iteration
            print("\033[2K", end="\r")
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Print GPU unavailability with current date and time
            print(f"GPU not available at {current_time}", end="\r")
            time.sleep(10)

if __name__ == '__main__':
    wait_gpu()