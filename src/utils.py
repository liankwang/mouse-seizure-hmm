import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Union

def timestamp_to_hourmin(pi_time, start_time="15:45:00", start_timestamp=412102781.0):
    start = datetime.strptime(start_time, "%H:%M:%S")

    seconds_elapsed = (pi_time - start_timestamp) / 1000
    time = start + timedelta(seconds=seconds_elapsed)
    return time.strftime("%H:%M")

def hourmin_to_timestamp(time, start_time="15:45:00", start_timestamp=412102781.0):
    start = datetime.strptime(start_time, "%H:%M:%S")
    time = datetime.strptime(time, "%H:%M")
    if time < start:
        seconds_elapsed = (time + timedelta(days=1) - start).total_seconds()
    else:
        seconds_elapsed = (time - start).total_seconds()
    return start_timestamp + seconds_elapsed * 1000



to_t = lambda array: torch.tensor(array, device='cpu', dtype=torch.float32)
from_t = lambda tensor: tensor.to("cpu").detach().numpy().astype(np.float64)

Number = Union[int, float]