import lzma
import os
import csv
from typing import List, Tuple, Any, Dict
from math import sqrt
import logging
import random

from matplotlib import pyplot
from torch import argmax, tensor
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score


_log = logging.getLogger(__name__)


class MetricsExporter:
    "The class implements an object, to be called when a neural network generates output"

    def __enter__(self):
        "overridable"
        return self

    def __exit__(self, exception_type, exception, traceback):
        "overridable"
        pass

    def process(self, pass_name: str, epoch_number: int,
                entry_names: List[str], output_values: List[Any], target_values: List[Any]):
        "the entry_names, output_values, target_values MUST have the same length"
        pass


class MetricsExporterCollection:
    "allows a series of metrics exporters to be used at the same time"

    def __init__(self, *args: Tuple[MetricsExporter]):
        self._metrics_exporters = args

    def __enter__(self):
        for metrics_exporter in self._metrics_exporters:
            metrics_exporter.__enter__()

        return self

    def __exit__(self, exception_type, exception, traceback):
        for metrics_exporter in self._metrics_exporters:
            metrics_exporter.__exit__(exception_type, exception, traceback)

    def process(self, pass_name: str, epoch_number: int,
                entry_names: List[str], output_values: List[Any], target_values: List[Any]):
        for metrics_exporter in self._metrics_exporters:
            metrics_exporter.process(pass_name, epoch_number, entry_names, output_values, target_values)


class TensorboardBinaryClassificationExporter(MetricsExporter):
    """ Exports to tensorboard, works for binary classification only.

        Currently outputs to tensorboard:
         - Mathews Correlation Coefficient (MCC)
         - Accuracy
         - ROC area under the curve

        Outputs are done per epoch.
    """

    def __init__(self, directory_path: str):
        self._directory_path = directory_path
        self._writer = SummaryWriter(log_dir=directory_path)

    def __enter__(self):
        self._writer.__enter__()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self._writer.__exit__(exception_type, exception, traceback)

    def process(self, pass_name: str, epoch_number: int,
                entry_names: List[str], output_values: List[Any], target_values: List[Any]):
        "write to tensorboard"

        loss = cross_entropy(tensor(output_values), tensor(target_values)).item()
        self._writer.add_scalar(f"{pass_name} cross entropy loss", loss, epoch_number)

        probabilities = []
        fp, fn, tp, tn = 0, 0, 0, 0
        for entry_index, entry_name in enumerate(entry_names):
            probability = output_values[entry_index][1]
            probabilities.append(probability)

            prediction_value = argmax(tensor(output_values[entry_index]))
            target_value = target_values[entry_index]

            if prediction_value > 0.0 and target_value > 0.0:
                tp += 1

            elif prediction_value <= 0.0 and target_value <= 0.0:
                tn += 1

            elif prediction_value > 0.0 and target_value <= 0.0:
                fp += 1

            elif prediction_value <= 0.0 and target_value > 0.0:
                fn += 1

        mcc_numerator = tn * tp - fp * fn
        if mcc_numerator == 0.0:
            self._writer.add_scalar(f"{pass_name} MCC", 0.0, epoch_number)
        else:
            mcc_denominator = sqrt((tn + fn) * (fp + tp) * (tn + fp) * (fn + tp))

            if mcc_denominator != 0.0:
                mcc = mcc_numerator / mcc_denominator
                self._writer.add_scalar(f"{pass_name} MCC", mcc, epoch_number)

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        self._writer.add_scalar(f"{pass_name} accuracy", accuracy, epoch_number)

        if len(set(target_values)) == 2:
            roc_auc = roc_auc_score(target_values, probabilities)
            self._writer.add_scalar(f"{pass_name} ROC AUC", roc_auc, epoch_number)


class OutputExporter(MetricsExporter):
    """ A metrics exporter that writes CSV output tables, containing every single data point.

        Included are:
            - entry names
            - output values
            - target values

        The user can load these output tables in excel.

        Outputs are done per epoch.
    """

    def __init__(self, directory_path: str):
        self._directory_path = directory_path

    def get_filename(self, pass_name, epoch_number):
        "returns the filename for the table"
        return os.path.join(self._directory_path, f"output-{pass_name}-epoch-{epoch_number}.csv.xz")

    def process(self, pass_name: str, epoch_number: int,
                entry_names: List[str], output_values: List[Any], target_values: List[Any]):
        "write the output to the table"

        with lzma.open(self.get_filename(pass_name, epoch_number), 'wt', newline='\n') as f:
            w = csv.writer(f, delimiter=',')

            w.writerow(["entry", "output", "target"])

            for entry_index, entry_name in enumerate(entry_names):
                output_value = output_values[entry_index]
                target_value = target_values[entry_index]

                w.writerow([entry_name, str(output_value), str(target_value)])


class ScatterPlotExporter(MetricsExporter):
    """ A metrics exporter that ocasionally makes scatter plots, containing every single data point.

        On the X-axis: targets values
        On the Y-axis: output values
    """

    def __init__(self, directory_path: str, epoch_interval: int = 1):
        """ Args:
                directory_path: where to store the plots
                epoch_interval: how often to make a plot, 5 means: every 5 epochs
        """

        self._epoch_interval = epoch_interval
        self._directory_path = directory_path

    def __enter__(self):
        self._plot_data = {}
        return self

    def __exit__(self, exception_type, exception, traceback):
        self._plot_data.clear()

    def get_filename(self, epoch_number):
        "returns the filename for the table"
        return os.path.join(self._directory_path, f"scatter-{epoch_number}.png")

    @staticmethod
    def _get_color(pass_name):

        pass_name = pass_name.lower().strip()

        if pass_name in ("train", "training"):
            return "blue"

        elif pass_name in ("eval", "valid", "validation"):
            return "red"

        elif pass_name == ("test", "testing"):
            return "green"
        else:
            return random.choice(["yellow", "cyan", "magenta"])


    @staticmethod
    def _plot(epoch_number: int, data: Dict[str, Tuple[List[float], List[float]]], png_path: str):

        pyplot.title(f"Epoch {epoch_number}")

        for pass_name, (truth_values, prediction_values) in data.items():
            pyplot.scatter(truth_values, prediction_values, color=ScatterPlotExporter._get_color(pass_name), label=pass_name)

        pyplot.xlabel("truth")
        pyplot.ylabel("prediction")

        pyplot.legend()
        pyplot.savefig(png_path)
        pyplot.close()

    def process(self, pass_name: str, epoch_number: int,
                entry_names: List[str], output_values: List[Any], target_values: List[Any]):
        "make the plot, if the epoch matches with the interval"

        if epoch_number % self._epoch_interval == 0:

            if epoch_number not in self._plot_data:
                self._plot_data[epoch_number] = {}

            self._plot_data[epoch_number][pass_name] = (target_values, output_values)

            path = self.get_filename(epoch_number)
            self._plot(epoch_number, self._plot_data[epoch_number], path)
