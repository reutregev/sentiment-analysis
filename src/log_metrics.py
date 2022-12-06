import json
import os.path
import subprocess
from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.logger import logger
from src.utils import write_to_json


class MetricsLogger:

    def __init__(self, tensorboard: bool):
        self.metrics = defaultdict(list)
        self.n_epochs = 0
        self.writer = SummaryWriter() if tensorboard else None

    def get(self, metric: str) -> float:
        """ Return the latest value of the given metric. If not exists, return np.inf """
        if self.is_metric_exist(metric):
            return self.metrics[metric][-1]
        else:
            return np.inf

    def add_metric(self, metric: str, val: float):
        """
        Add a metric.

        Parameters
        ----------
        metric : str
            Metric name
        val : float
            Value of metric to add

        """
        self.metrics[metric].append(val)

    def add_metrics(self, metrics: Dict[str, float]):
        """
        Add metrics from a dict.

        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary of metric name to value

        """
        for metric, val in metrics.items():
            self.metrics[metric].append(val)
            if self.writer:
                self.writer.add_scalar(metric, val, self.n_epochs)

    def mean(self, metric: str):
        """
        Calculate the mean of the stored values of the given metric.

        Parameters
        ----------
        metric : str
            Metric name

        """
        if self.is_metric_exist(metric):
            return round(np.mean(self.metrics[metric]), 4)

    def print_metric(self, metric: str, only_last_iter: bool = False):
        """
        Print values of the given metric.
        If only_last_iter is True, print only the latest value.

        Parameters
        ----------
        metric : str
            Metric name
        only_last_iter : bool, optional
            Whether to print only the latest value or all stored values of the metric.

        """
        if self.is_metric_exist(metric):
            if only_last_iter:
                print(f"{metric}: {self.metrics[metric][-1]}")
            else:
                print(f"{metric}: {self.metrics[metric]}")

    def print_all(self):
        for metric, vals in self.metrics.items():
            print(f"\n{metric}: {vals}")

    def print_latest_epoch(self):
        """ Print all latest values of all stored metrics """
        log = f"\nEpoch {self.n_epochs + 1}"
        for metric in self.metrics.keys():
            log += f" | {metric}: {round(self.metrics[metric][-1], 4)}"
        print(log)

    def plot(self, metric: str):
        if self.is_metric_exist(metric):
            vals = self.metrics[metric]
            x = np.arange(1, len(vals) + 1)
            plt.figure()
            plt.plot(x, vals)
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.xticks(x)
            plt.show()

    def export(self, path: str):
        os.makedirs(path, exist_ok=True)
        write_to_json(self.metrics, path)

    def is_metric_exist(self, metric: str) -> bool:
        return metric in self.metrics

    def increase_epoch(self):
        self.n_epochs += 1

    def run_tensorboard(self):
        """ If an instance is initialized with tensorboard, this method would run tensorboard """
        if self.writer:
            subprocess.call("tensorboard --logdir=./runs")
            logger.info(f"Tensorboard is running")
        else:
            logger.info(f"Tensorboard is not enabled")
