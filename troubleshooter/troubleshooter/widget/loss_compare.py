import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from troubleshooter.common.util import validate_and_normalize_path, make_directory

LOSS_REG_EXP = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


class FileInfo:
    def __init__(self, file_dict):
        self.path = validate_and_normalize_path(file_dict.get("path"))
        self.loss_tag = file_dict.get("loss_tag")
        self.label = file_dict.get("label")
        self.losses = []
        self.parse_file()

    def parse_file(self):
        with open(self.path, "r") as f:
            content = f.readlines()
        for line in content:
            loss = self.parse_by_tag(line)
            if loss is not None:
                self.losses.append(loss)

    def parse_by_tag(self, line):
        pos = line.find(self.loss_tag)
        result = None
        if pos != -1:
            res = LOSS_REG_EXP.search(line[pos + len(self.loss_tag):].strip().split()[0])
            if res is not None:
                result = float(res[0])
            else:
                print(f"found {'loss'} text, but parse value with error: [{line}]")
        return result


class LossComparison:
    def __init__(self, left_file, right_file, title):
        self.left_file = left_file
        self.right_file = right_file
        self.output_dir = make_directory(os.path.join(os.getcwd(), "loss_compare"))
        self.title = title

    @staticmethod
    def permission_change(path, mode):
        if not os.path.exists(path) or os.path.islink(path):
            return
        try:
            os.chmod(path, mode)
        except PermissionError as e:
            raise PermissionError('Failed to change {} permissions. {}'.format(path, str(e)))

    @staticmethod
    def get_loss_stat(loss):
        loss_max = max(loss) if loss else "Nan"
        loss_min = min(loss) if loss else "Nan"
        loss_mean = sum(loss) / len(loss) if loss else "Nan"
        return loss_max, loss_min, loss_mean

    def save_csv(self, file):
        file_name = os.path.split(file.path)[-1] + ".csv"
        output_path = os.path.join(self.output_dir, file_name)
        df = pd.DataFrame({"loss": file.losses})
        df.to_csv(output_path, index=True)
        self.permission_change(output_path, 0o600)

    def plot_loss(self):
        save_path = os.path.join(self.output_dir, "loss_compare.png")
        plt.figure(dpi=150)
        plt.plot(self.left_file.losses, label=self.left_file.label)
        plt.plot(self.right_file.losses, label=self.right_file.label)
        plt.title(self.title)
        plt.xlabel(u'iter')
        plt.ylabel(u'loss')
        plt.legend(loc="best")
        plt.savefig(save_path)
        self.permission_change(save_path, 0o600)

    def plot_error(self):
        save_path = os.path.join(self.output_dir, "error.png")
        error = [left - right for (left, right) in zip(self.left_file.losses, self.right_file.losses)]
        plt.figure(dpi=150)
        plt.plot(error, label="error")
        plt.xlabel(u'iter')
        plt.ylabel(u'error')
        plt.legend(loc="best")
        plt.savefig(save_path)
        self.permission_change(save_path, 0o600)

    def get_stat_info(self):
        save_path = os.path.join(self.output_dir, "statistic.csv")
        if self.left_file.losses and self.right_file.losses:
            max_abs_error = max(abs(left - right) for (left, right) in zip(
                self.left_file.losses, self.right_file.losses))
            max_rel_error = max(abs(left - right) / left for (left, right) in zip(
                self.left_file.losses, self.right_file.losses))
        else:
            max_abs_error = "Nan"
            max_rel_error = "Nan"
        left_max, left_min, left_mean = self.get_loss_stat(self.left_file.losses)
        right_max, right_min, right_mean = self.get_loss_stat(self.right_file.losses)
        df = pd.DataFrame(
            {
                "LEFT_MAX": left_max,
                "LEFT_MIN": left_min,
                "LEFT_MEAN": left_mean,
                "RIGHT_MAX": right_max,
                "RIGHT_MIN": right_min,
                "RIGHT_MEAN": right_mean,
                "MAX_ABS_ERROR": max_abs_error,
                "MAX_REL_ERROR": max_rel_error
            },
            index=[0]
        )
        df.to_csv(save_path, index=False)
        self.permission_change(save_path, 0o600)

    def compare_loss(self):
        self.plot_loss()
        self.plot_error()
        self.get_stat_info()
        self.save_csv(self.left_file)
        self.save_csv(self.right_file)
        print(f"Loss Comparison has completed, results saved in {self.output_dir}!")


def loss_compare(left_file, right_file, title="Loss Compare"):
    left = FileInfo(left_file)
    right = FileInfo(right_file)
    compare = LossComparison(left, right, title)
    compare.compare_loss()
    
