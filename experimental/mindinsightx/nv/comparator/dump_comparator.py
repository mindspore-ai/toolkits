# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Dump value comparators."""
from enum import Enum
import numpy as np

FLOAT_EPSILON = np.finfo(float).eps

_factories = dict()


class Result(Enum):
    EQUAL = 0
    NOT_EQUAL = 2
    ERROR = 3
    INVALID_DUMP0 = 4
    INVALID_DUMP1 = 5
    SIZE_NOT_MATCH = 6


class DumpComparator:
    """Comparator base class."""

    @staticmethod
    def from_str(cmp_str):
        """
        Create dump comparator from string.

        Args:
            cmp_str (str): comparator construction string like "Euclidean(rtol=1e-3)"

        Returns:
            DumpComparator
        """
        splited = cmp_str.split('(')
        cmp_name = splited[0].strip()
        factory = _factories.get(cmp_name, None)
        if factory is None:
            return None
        if len(splited) < 2:
            return factory()
        str_kwargs = splited[1].strip().replace(')', '')
        str_kwargs_splited = str_kwargs.split(',')
        kwargs = dict()
        for splited in str_kwargs_splited:
            splited = splited.strip()
            if splited == '':
                continue
            splited2 = splited.split('=')
            if len(splited2) != 2:
                return None
            try:
                kwargs[splited2[0].strip()] = float(splited2[1].strip())
            except ValueError:
                return None

        try:
            return factory(**kwargs)
        except TypeError:
            return None

    def compare(self, dump0, dump1):
        """
        Compare the dump values.

        Args:
            dump0 (numpy.ndarray): Flattened dump 0 data.
            dump1 (numpy.ndarray): Flattened dump 1 data.

        Returns:
            Tuple[Result, Any, str], Compare result and details.
        """
        result, detail = self.check_data(dump0, dump1)
        if result:
            return result, None, detail
        return self.compare_value(dump0, dump1)

    def check_data(self, dump0, dump1):
        """Check if the data is valid."""
        if dump0 is None or dump1 is None:
            return Result.INVALID_DUMP0, "dump0 or dump1 is None"
        if not dump0.size or len(dump0.shape) != 1:
            return Result.INVALID_DUMP0, f"size of dump0:{dump0.size}, dump1:{dump1.size}"
        if not dump1.size or len(dump1.shape) != 1:
            return Result.INVALID_DUMP1, f"size of dump0:{dump0.size}, dump1:{dump1.size}"
        if dump0.size != dump1.size:
            return Result.SIZE_NOT_MATCH, f"size of dump0:{dump0.size}, dump1:{dump1.size}"
        return 0, None

    def compare_value(self, dump0, dump1):
        """
        Do compare the dump values.

        Args:
            dump0 (numpy.ndarray): Flattened dump 0 data.
            dump1 (numpy.ndarray): Flattened dump 1 data.

        Returns:
            Tuple[Result, Any, str], Compare result and details.
        """
        raise NotImplementedError


class AllClose(DumpComparator):
    """AllClose comparator."""
    def __init__(self, rtol=1e-05, atol=1e-08):
        self.rtol = rtol
        self.atol = atol

    def __str__(self):
        return type(self).__name__ + f'(rtol={self.rtol},atol={self.atol})'

    def compare_value(self, dump0, dump1):
        """
        Do compare the dump values.

        Args:
            dump0 (numpy.ndarray): Flattened dump 0 data.
            dump1 (numpy.ndarray): Flattened dump 1 data.

        Returns:
            Tuple[Result, float, str], Compare result and the detail string.
        """
        if np.allclose(dump0, dump1, rtol=self.rtol, atol=self.atol):
            return Result.EQUAL, None, None
        return Result.NOT_EQUAL, None, None


class MaxAbsError(DumpComparator):
    """Maximum absolute error comparator."""
    def __init__(self, atol=1e-8):
        self.atol = atol

    def __str__(self):
        return type(self).__name__ + f'(atol={self.atol})'

    def compare_value(self, dump0, dump1):
        """
        Do compare the dump values.

        Args:
            dump0 (numpy.ndarray): Flattened dump 0 data.
            dump1 (numpy.ndarray): Flattened dump 1 data.

        Returns:
            Tuple[Result, float, str], Compare result and the detail string.
        """
        abs_errs = np.abs(dump0 - dump1)
        max_abs_err = abs_errs.max()

        result = Result.EQUAL if max_abs_err <= self.atol else Result.NOT_EQUAL
        return result, max_abs_err, None


class NormError(DumpComparator):
    """Norm comparator."""

    def __init__(self, rtol=1e-3):
        self.rtol = rtol

    def __str__(self):
        return type(self).__name__ + f'(rtol={self.rtol})'

    def compare_value(self, dump0, dump1):
        """
        Do compare the dump values.

        Args:
            dump0 (numpy.ndarray): Flattened dump 0 data.
            dump1 (numpy.ndarray): Flattened dump 1 data.

        Returns:
            Tuple[Result, float, str], Compare result and the detail string.
        """
        norm0 = np.linalg.norm(dump0)
        norm1 = np.linalg.norm(dump1)
        if not np.isfinite(norm0) or not np.isfinite(norm1):
            return Result.ERROR, None, f'norm0:{norm0} norm1:{norm1}'

        norm_diff = np.abs(norm0 - norm1)
        if norm_diff <= FLOAT_EPSILON:
            return Result.EQUAL, 0, f'norm0:{norm0} norm1:{norm1}'

        rerr = norm_diff/(max(norm0, norm1) + FLOAT_EPSILON)
        if not np.isfinite(rerr) or rerr >= self.rtol:
            return Result.NOT_EQUAL, rerr, f'norm0:{norm0} norm1:{norm1}'
        return Result.EQUAL, rerr, f'norm0:{norm0} norm1:{norm1}'


class Euclidean(DumpComparator):
    """Euclidean distance comparator."""

    def __init__(self, rtol=1e-3):
        self.rtol = rtol

    def __str__(self):
        return type(self).__name__ + f'(rtol={self.rtol})'

    def compare_value(self, dump0, dump1):
        """
        Do compare the dump values.

        Args:
            dump0 (numpy.ndarray): Flattened dump 0 data.
            dump1 (numpy.ndarray): Flattened dump 1 data.

        Returns:
            Tuple[Result, float], Compare result and the detail string.
        """
        diff = dump0 - dump1
        diff_norm = np.linalg.norm(diff)
        if not np.isfinite(diff_norm):
            return Result.ERROR, None, f'diff_norm:{diff_norm}'
        if diff_norm <= FLOAT_EPSILON:
            return Result.EQUAL, 0, f'diff_norm:{diff_norm}'

        norm0 = np.linalg.norm(dump0)
        norm1 = np.linalg.norm(dump1)
        max_dump_norm = np.maximum(norm0, norm1)
        if not np.isfinite(max_dump_norm):
            return Result.ERROR, None, f'diff_norm:{diff_norm} norm0:{norm0} norm1:{norm1}'

        rerr = diff_norm/(max_dump_norm + FLOAT_EPSILON)
        if not np.isfinite(rerr) or rerr >= self.rtol:
            return Result.NOT_EQUAL, rerr, f'diff_norm:{diff_norm} norm0:{norm0} norm1:{norm1}'
        return Result.EQUAL, rerr, f'diff_norm:{diff_norm} norm0:{norm0} norm1:{norm1}'


class AvgAbsError(DumpComparator):
    """Average absolute error comparator."""
    def __init__(self, rtol=1e-3):
        self.rtol = rtol

    def __str__(self):
        return type(self).__name__ + f'(rtol={self.rtol})'

    def compare_value(self, dump0, dump1):
        avg_abs0 = np.mean(np.abs(dump0))
        avg_abs1 = np.mean(np.abs(dump1))
        rerr = (2 * np.abs(avg_abs0 - avg_abs1))/((avg_abs0 + avg_abs1) + FLOAT_EPSILON)
        if not np.isfinite(rerr) or rerr >= self.rtol:
            return Result.NOT_EQUAL, rerr, f'avg_abs0:{avg_abs0} avg_abs1:{avg_abs1}'
        return Result.EQUAL, rerr, f'avg_abs0:{avg_abs0} avg_abs1:{avg_abs1}'


_factories[AllClose.__name__] = AllClose
_factories[MaxAbsError.__name__] = MaxAbsError
_factories[NormError.__name__] = NormError
_factories[Euclidean.__name__] = Euclidean
_factories[AvgAbsError.__name__] = AvgAbsError
