import json
import time
from typing import Dict

from src.logger import logger


def write_to_json(obj: Dict, path: str, **kwargs):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, sort_keys=True, **kwargs)


def calc_running_time(func):
    def calc_running_time_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Function {func.__name__} took {total_time:.4f} seconds")
        return result
    return calc_running_time_wrapper
