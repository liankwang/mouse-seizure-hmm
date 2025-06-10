import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Union

def timestamp_to_hourmin(pi_time: int) -> str:
    hours = pi_time // 3600
    minutes = (pi_time % 3600) // 60
    return f"{hours:02}:{minutes:02}"

def hourmin_to_timestamp(time: str) -> int:
    time_obj = datetime.strptime(time, "%H:%M")
    return time_obj.hour * 3600 + time_obj.minute * 60

def hourmin_to_timestamp_ms(time: str) -> int:
    time_obj = datetime.strptime(time, "%H:%M")
    return (time_obj.hour * 3600 + time_obj.minute * 60)*1000

to_t = lambda array: torch.tensor(array, device='cpu', dtype=torch.float32)
from_t = lambda tensor: tensor.to("cpu").detach().numpy().astype(np.float64)

Number = Union[int, float]