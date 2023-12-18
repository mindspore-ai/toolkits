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
"""The utils."""
import glob
import math

import numpy as np


FORMAT_ND = ["ND"]
FORMAT_4D = ["NCHW", "NHWC", "HWCK", "HWCN"]
FORMAT_5D = ["NC1HWC0"]
FORMAT_6D = ["FRACZ", "FRACTAL_NZ"]
ALL_FORMAT = FORMAT_ND + FORMAT_4D + FORMAT_5D + FORMAT_6D

MAX_FIXED_NUM = 16
DATA_TYPE_MAPPING = {"float16": np.float16,
                     "float32": np.float32,
                     "float64": np.float64,
                     "int8": np.int8,
                     "int16": np.int16,
                     "int32": np.int32,
                     "int64": np.int64,
                     "uint8": np.uint8,
                     "uint16": np.uint16,
                     "uint32": np.uint32,
                     "uint64": np.uint64,
                     "bool": np.bool}


def get_files_list(path):
    """
    Get the files list with path.

    Args:
        path (str): The file path.

    Returns:
        dict, the files list.

    """
    return glob.glob(path + "/**/*", recursive=True)


def get_shape_format(dump_file):
    """
    Parse the dump_file then get data_format, data_type, data_shape.

    Examples:
        dump_file is:
        "conv2-Conv2d--Conv2D_output_0_shape_5_5_6_16_float32_NHWC.npy",
                                            |--------|-------|----|
                                            |--shape-|--type-|format|

    Args:
        dump_file (str): The dump file path.

    Returns:
        tuple, the result of (format, type, shape).

    """
    shape_start_index = dump_file.find("_shape_")
    if shape_start_index <= -1:
        return None, None, []

    shape_format_content = \
        dump_file[shape_start_index + len("_shape_"):-4]
    shape_format_list = shape_format_content.split("_")

    # get the shape size
    type_start_index = -2
    index = -2
    try:
        for index, data in enumerate(shape_format_list):
            int(data)
    except ValueError:
        type_start_index = index

    data_format = "-".join(shape_format_list[type_start_index + 1:])
    data_type = shape_format_list[type_start_index]
    data_shape = [int(x) for x in shape_format_list[0:type_start_index]]
    return data_format, data_type, data_shape


def check_data_format_pair(format1, format2):
    """
    Check data format pair.

    Args:
        format1 (str): The format.
        format2 (str): The format.

    Returns:
        bool, return true if equal.

    """
    format1_upper = format1.upper()
    format2_upper = format2.upper()
    target_format = select_target_format(format1_upper, format2_upper)

    if not _check_data_format_transform(format1, target_format):
        return False
    if not _check_data_format_transform(format2, target_format):
        return False

    return True


def select_target_format(format1, format2):
    """
    Select target format.

    Args:
        format1 (str): The format.
        format2 (str): The format.

    Returns:
        str, the dest format.

    """
    dst_format = format1.upper()
    if format2.upper() in FORMAT_5D + FORMAT_6D:
        dst_format = format2.upper()
    return dst_format


def _check_data_format_transform(src_format, dst_format):
    """
    Check the data format transform.

    Args:
        src_format (str): The source format.
        dst_format (str): The dest format.

    Returns:
        bool, return true if equal.

    """
    src_format = src_format.upper()
    dst_format = dst_format.upper()
    if src_format == dst_format:
        return True
    if dst_format in FORMAT_ND:
        return False
    if src_format == "FRACTAL_NZ":
        return False
    if src_format in FORMAT_5D + FORMAT_6D and dst_format not in FORMAT_4D:
        return False
    if src_format == "HWCK" and dst_format == "FRACTAL_NZ":
        return False
    return True


def dtype_from_string(str_data_type):
    """Convert str to dtype of np.ndarray."""
    try:
        data_type = DATA_TYPE_MAPPING[str_data_type.lower()]
    except KeyError:
        raise ValueError("The data type (%s) is not supported." % str_data_type)
    return data_type


def load_numpy_data_from_file(tensor_bin_file, str_data_type, src_format=None,
                              to_format=None, shape=None):
    """Load numpy data from file."""
    orign_npy_data = load_data_from_np_file(tensor_bin_file, str_data_type)
    npy_data = orign_npy_data
    if src_format and src_format != to_format:
        npy_data = convert_data_format(orign_npy_data, src_format, to_format,
                                       shape)
    return orign_npy_data, npy_data


def _rendim_from1_to_4(npy_data, shape):
    """Change ndim of ndarray from 1 to 4."""
    if shape and len(shape) == 4 and npy_data.ndim == 1:
        npy_data = npy_data.reshape(shape)
    return npy_data


def nhwc2nc1hwc0(npy_data, nhwc_shape=None):
    """Change ndarray from nhwc format to nc1hwc0 format."""
    try:
        npy_data = _rendim_from1_to_4(npy_data, nhwc_shape)
        if npy_data.ndim == 4:
            # first switch to NCHW
            npy_data = np.transpose(npy_data, [0, 3, 1, 2])
            npy_data_shape = npy_data.shape
            npy_data = npy_data.reshape(npy_data.size)
            d5_npy_data, is_pad = _nchw2nc1hwc0(npy_data, npy_data_shape)
        else:
            raise ValueError("can not convert NHWC to NC1HWC0.")
    except IndexError:
        raise ValueError("The shape and format do not match.")
    return d5_npy_data, is_pad


def nchw2nc1hwc0(npy_data, nchw_shape=None):
    """Change ndarray from nchw format to nc1hwc0 format."""
    try:
        nchw_4d_shape = npy_data.shape
        if len(nchw_4d_shape) != 4 and len(nchw_shape) == 4:
            nchw_4d_shape = nchw_shape

        if len(nchw_4d_shape) == 4:
            npy_data = npy_data.reshape(npy_data.size)
            d5_npy_data, is_pad = _nchw2nc1hwc0(npy_data, nchw_4d_shape)
        else:
            raise ValueError("can not convert NCHW to NC1HWC0.")
    except IndexError:
        raise ValueError("The shape and format do not match.")
    return d5_npy_data, is_pad


def hwck2fracz(npy_data, hwck_4d_shape=None):
    """Change ndarray from hwck format to fracz format."""
    try:
        npy_data = _rendim_from1_to_4(npy_data, hwck_4d_shape)
        if npy_data.ndim == 4:
            # fracz is c1hwnc0 format
            fz_npy_data, is_pad = _hwck2c1hwnc0(npy_data)
            fz_npy_data = fz_npy_data.reshape(fz_npy_data.size)
        else:
            raise ValueError("can not convert HWCK to FracZ.")
    except IndexError:
        raise ValueError("The shape and format do not match.")
    return fz_npy_data, is_pad


def fracz2hwck(npy_data, hwck_shape=None):
    """Change ndarray from fracz format to hwck format."""
    if hwck_shape is None or len(hwck_shape) != 4:
        raise ValueError("Shape is invalid!")
    h, w, c_out, n_out = hwck_shape

    n, _, _ = _get_high_dim_num(n_out)
    _, c0, c1 = _get_high_dim_num(c_out)

    npy_data = npy_data.reshape((c1, -1, n, c0)).transpose((1, 0, 3, 2))
    npy_data = npy_data.reshape((npy_data.shape[0], -1, npy_data.shape[3]))
    npy_data = npy_data[:, :c_out, :n_out].reshape(h, w, c_out, n_out)

    return npy_data


def nc1hwc02nhwc(npy_data, nhwc_shape=None):
    """Change ndarray from nc1hwc0 format to nhwc format."""
    # nc1hwc0 to nhwc
    if nhwc_shape is None or len(nhwc_shape) != 4:
        raise ValueError("Shape is invalid!")

    n, h, w, c_out = nhwc_shape
    _, c0, c1 = _get_high_dim_num(c_out)
    npy_data = npy_data.reshape((n, c1, -1, c0)).transpose(0, 2, 1, 3)
    npy_data = npy_data.reshape((n, h * w, -1))[:, :, :c_out].reshape(
        [-1]).reshape(n, h, w, c_out)
    return npy_data


def nd2fracnz(npy_data, nd_shape=None):
    """
    Change ndarray from nd format to fracNz format.

    Args:
        npy_data: np.array
        nd_shape: The shape.

    Returns:

    """
    is_pad = False
    if not nd_shape or len(nd_shape) > 4:
        raise ValueError("Shape is invalid!")

    npy_data = npy_data.reshape(nd_shape)

    if len(nd_shape) < 2:
        return nd2fracnz(npy_data[np.newaxis, :], (1,) + tuple(nd_shape))
    dim1, dim2 = nd_shape[-2:]
    dim1, padding_dim1 = _get_padding_num(dim1)
    dim2, padding_dim2 = _get_padding_num(dim2)

    if padding_dim1 or padding_dim2:
        is_pad = True

    padding_shape = (len(nd_shape) - 2) * ((0, 0),) + ((0, padding_dim1),) + (
        (0, padding_dim2),)

    npy_data = np.pad(npy_data, padding_shape, mode="constant",
                      constant_values=0)

    if len(npy_data.shape) == 3:
        npy_data = npy_data.reshape(((npy_data.shape[:-2]) + (dim1, MAX_FIXED_NUM, dim2, MAX_FIXED_NUM))).\
            transpose([0, 3, 1, 2, 4])

    elif len(npy_data.shape) == 4:
        npy_data = npy_data.reshape(((npy_data.shape[:-2]) + (dim1, MAX_FIXED_NUM, dim2, MAX_FIXED_NUM))).\
            transpose([0, 1, 4, 2, 3, 5])
    else:
        npy_data = npy_data.reshape((dim1, MAX_FIXED_NUM, dim2, MAX_FIXED_NUM)).transpose([2, 0, 1, 3])

    return npy_data, is_pad


def _get_padding_num(num, max_fixed_num=MAX_FIXED_NUM):
    padding_num = 0
    if num % max_fixed_num:
        padding_num = max_fixed_num - num % max_fixed_num
    num = (num + max_fixed_num - 1) // MAX_FIXED_NUM

    return num, padding_num


def _get_high_dim_num(expected_dim):
    ceiling_dim = (expected_dim + MAX_FIXED_NUM - 1) // MAX_FIXED_NUM * MAX_FIXED_NUM
    dim1, dim0 = ceiling_dim // MAX_FIXED_NUM, MAX_FIXED_NUM

    return ceiling_dim, dim0, dim1


def transform_4d(npy_data, src_shape=None, src_format="NHWC", to_format="NHWC"):
    """Transpose ndarray."""
    if src_shape is not None:
        npy_data = _rendim_from1_to_4(npy_data, src_shape)
    if npy_data.ndim != 4:
        raise ValueError("The shape and format do not match.")

    src_format = src_format.upper()
    to_format = to_format.upper()

    if src_format == to_format:
        return npy_data

    src_format = _tran_if_hwck(src_format)
    to_format = _tran_if_hwck(to_format)

    to_index = {}
    index = 0
    for i in src_format:
        to_index.update({i: index})
        index += 1

    src_to_index = []
    for j in to_format:
        src_to_index.append(to_index.get(j))

    npy_data = np.transpose(npy_data, src_to_index)

    return npy_data


def transpose_4d_format(shape, src_format, to_format):
    """Get new shape of data after convert format from src_format to to_format."""
    if len(shape) != 4:
        raise ValueError("The shape and format do not match.")

    src_format = src_format.upper()
    to_format = to_format.upper()

    if src_format == to_format:
        return shape

    src_format = _tran_if_hwck(src_format)
    to_format = _tran_if_hwck(to_format)

    src_shape = {}
    index = 0
    for i in src_format:
        src_shape.update({i: shape[index]})
        index += 1

    to_shape = []
    for j in to_format:
        to_shape.append(src_shape.get(j))

    return to_shape


def _tran_if_hwck(shape_format):
    """Transform if format if HWCK."""
    if shape_format == "HWCK":
        shape_format = "HWCN"
    return shape_format


def convert_data_format(npy_data, src_format="NHWC", to_format="NHWC",
                        src_shape=None):
    """Convert ndarray format from src_format to to_format."""
    is_pad = False
    if src_format.upper() == to_format.upper():
        return npy_data, False

    if src_format.upper() in FORMAT_4D and to_format.upper() in FORMAT_4D:
        return transform_4d(npy_data, src_shape, src_format, to_format), False

    if src_format.upper() == "NHWC" and to_format.upper() == "NC1HWC0":
        return nhwc2nc1hwc0(npy_data, src_shape)

    if src_format.upper() == "NCHW" and to_format.upper() == "NC1HWC0":
        return nchw2nc1hwc0(npy_data, src_shape)

    if src_format.upper() == "HWCK" and to_format.upper() == "NC1HWC0":
        npy_data = transform_4d(npy_data, src_shape, src_format, "NCHW")
        return nchw2nc1hwc0(npy_data)

    if src_format.upper() == "NC1HWC0" and to_format.upper() in FORMAT_4D:
        nhwc_shape = transpose_4d_format(src_shape, src_format=to_format,
                                         to_format="NHWC")
        res = nc1hwc02nhwc(npy_data, nhwc_shape)
        return transform_4d(res, res.shape, "NHWC", to_format), False

    if src_format.upper() in FORMAT_4D and to_format.upper() == "FracZ".upper():
        npy_data = transform_4d(npy_data, src_shape, src_format, "HWCK")
        return hwck2fracz(npy_data)

    if src_format.upper() in ["NCHW", "NHWC",
                              "ND"] and to_format.upper() == "FRACTAL_NZ":
        return nd2fracnz(npy_data, src_shape)

    if src_format.upper() == "FracZ".upper() and to_format.upper() in FORMAT_4D:
        # first to convert to HWCK
        hwck_shape = transpose_4d_format(src_shape, src_format=to_format,
                                         to_format="HWCK")
        res = fracz2hwck(npy_data, hwck_shape)
        # then convert to target format
        return transform_4d(res, res.shape, "HWCK", to_format), False
    return npy_data, is_pad


def _nchw2nc1hwc0(nchw_data, nchw_shape):
    """NCHW to NCL2HWC0."""
    is_pad = False
    n = nchw_shape[0]
    c = nchw_shape[1]
    h = nchw_shape[2]
    w = nchw_shape[3]
    c0 = 16
    c1 = int(math.ceil(c / 16.0))
    nc1hwc0_data = np.zeros((n * c1 * h * w * c0), np.float32)

    if int(c) != int(c0*c1):
        is_pad = True

    for nidx in range(n):
        nf = nidx * c1 * h * w * c0
        for c1idx in range(c1):
            c1f = nf + c1idx * h * w * c0
            for hidx in range(h):
                _nchw2nc1hwc0_nested(nidx, c1idx, c1f,
                                     hidx, c, h, w, c0,
                                     nc1hwc0_data, nchw_data)
    return nc1hwc0_data, is_pad


def _nchw2nc1hwc0_nested(nidx, c1idx, c1f,
                         hidx, c, h, w, c0,
                         nc1hwc0_data, nchw_data):
    """For passing pylint too many nested scope check"""
    hf = c1f + hidx * w * c0
    for widx in range(w):
        wf = hf + widx * c0
        for c0idx in range(c0):
            idx = wf + c0idx
            cidx = c0idx + c1idx * c0
            srcidx = nidx * c * h * w + cidx * h * w + hidx * w + widx
            if cidx < c:
                nc1hwc0_data[idx] = nchw_data[srcidx]


def _hwck2c1hwnc0(filter_data):
    """HWCK to C1HWNC0."""
    is_pad = False
    channel0 = 16
    f = np.shape(filter_data)
    c = f[2]
    c1 = np.int(np.ceil(f[2] / channel0))
    h = f[0]
    w = f[1]
    n = f[3]
    c0 = channel0
    nalign = np.int(np.ceil(n/channel0)) * channel0

    if int(c1) != int(c/channel0) or int(n) != int(nalign):
        is_pad = True

    output_data = np.zeros(
        [np.int(np.ceil(f[2] / channel0)) * f[0] * f[1] * nalign * channel0],
        np.float32)
    filter_data = filter_data.reshape(h * w * c * n)
    for c1loop in range(c1):
        for hloop in range(h):
            for wloop in range(w):
                _hwck2c1hwnc0_nested(nalign, c0, hloop,
                                     h, w, c, n, wloop,
                                     c1loop, output_data,
                                     filter_data)
    return output_data, is_pad


def _hwck2c1hwnc0_nested(nalign, c0, hloop,
                         h, w, c, n, wloop,
                         c1loop, output_data,
                         filter_data):
    """For fixing pylint too many nested scope issue."""
    for nloop in range(nalign):
        for c0loop in range(c0):
            srcindex = hloop * w * c * n + wloop * c * n + (c1loop * c0 + c0loop) * n + nloop
            dstindex = c1loop * h * w * nalign * c0 + hloop * w * nalign * c0 + \
                       wloop * nalign * c0 + nloop * c0 + c0loop
            if (c1loop * c0 + c0loop) >= c or nloop >= n:
                output_data[dstindex] = 0
            else:
                output_data[dstindex] = filter_data[srcindex]


def str2shape(str_shape):
    """Get real shape for str of shape."""
    # input like "(1,2,3,4)"
    if str_shape:
        shape_tuple = tuple(
            map(int, str_shape.lstrip("(").rstrip(")").split(",")))
        return shape_tuple
    return None


def load_data_from_np_file(tensor_bin_file, data_type):
    """Load data."""
    try:
        bin_data = np.load(tensor_bin_file)
    except FileNotFoundError:
        print(f"file not found, {tensor_bin_file}")
        raise
    except PermissionError:
        print(f"not permission to access file, {tensor_bin_file}")
        raise
    except (IOError, ValueError):
        # Binary file
        if not data_type:
            raise ValueError("must to assign data type for file.")
        if isinstance(data_type, str):
            data_type = dtype_from_string(data_type)
        bin_data = np.fromfile(tensor_bin_file, dtype=data_type)

    if bin_data.dtype == np.float16:
        bin_data = bin_data.astype(np.float32)

    return bin_data


def nchw2nc1hwc0_shape(shape):
    """Get nc1hwc0 format shape from nchw format shape."""
    n = shape[0]
    c = shape[1]
    h = shape[2]
    w = shape[3]
    c0 = 16
    c1 = int(math.ceil(c / 16.0))
    return tuple([n, c1, h, w, c0])


def nhwc2nc1hwc0_shape(shape):
    """Get nc1hwc0 format shape from nhwc format shape."""
    n = shape[0]
    h = shape[1]
    w = shape[2]
    c = shape[3]
    c0 = 16
    c1 = int(math.ceil(c / 16.0))
    return tuple([n, c1, h, w, c0])


def hwck2nc1hwc0_shape(shape):
    """Get nc1hwc0 format shape from hwck format shape."""
    h = shape[0]
    w = shape[1]
    c = shape[2]
    k = shape[3]
    c0 = 16
    c1 = int(math.ceil(c / 16.0))
    return tuple([k, c1, h, w, c0])


def cmp_4dtonc1hwc0_shape(shape_4d, shape_5d):
    """Estimate whether shape_5d is available to convert to shape_4d."""
    if len(shape_4d) != 4:
        raise ValueError("Sahpe is invalid")

    prod_shape_4d = np.prod(shape_4d)
    prod_shape_5d = np.prod(shape_5d)
    if prod_shape_4d != prod_shape_5d:
        return False

    sum_shape_5d = np.sum(shape_5d)
    sum_shape_4d_nchw = np.sum(nchw2nc1hwc0_shape(shape_4d))
    sum_shape_4d_nhwc = np.sum(nhwc2nc1hwc0_shape(shape_4d))
    sum_shape_4d_hwck = np.sum(hwck2nc1hwc0_shape(shape_4d))
    if sum_shape_5d not in [sum_shape_4d_nchw, sum_shape_4d_nhwc,
                            sum_shape_4d_hwck]:
        return False

    return True


def nd2fracnz_shape(nd_shape):
    """Convert N-d shape to fracNz shape."""
    if not nd_shape or len(nd_shape) > 4:
        raise ValueError("Shape is invalid!")

    if len(nd_shape) < 2:
        return nd2fracnz_shape((1,) + tuple(nd_shape))
    dim1, dim2 = nd_shape[-2:]
    dim1, _ = _get_padding_num(dim1)
    dim2, _ = _get_padding_num(dim2)

    if len(nd_shape) == 3:
        # C H W -> C W1 H1 H0 W0
        c = nd_shape[0]
        fracnz_shape = tuple([c, dim2, dim1, MAX_FIXED_NUM, MAX_FIXED_NUM])
    elif len(nd_shape) == 4:
        # N C H W -> N C W1 H1 H0 W0
        n = nd_shape[0]
        c = nd_shape[1]
        fracnz_shape = tuple([n, c, dim2, dim1, MAX_FIXED_NUM, MAX_FIXED_NUM])
    else:
        fracnz_shape = tuple([dim2, dim1, MAX_FIXED_NUM, MAX_FIXED_NUM])
    return fracnz_shape


def cmp_ndtofractal_nz_shape(shape_nd, shape_nz):
    trans_shape_for_nd = nd2fracnz_shape(shape_nd)

    if tuple(shape_nz) == tuple(trans_shape_for_nd):
        return True

    return False


def cmp_shape(shape1, shape2, auto_shape_covert=True):
    """Check if 2 shapes are equal"""
    is_equal = False
    if not is_valid_shape(shape1) or not is_valid_shape(shape2):
        return False

    if not auto_shape_covert:
        if shape1 == shape2:
            return True
        return False
    # NCHW vs NHWC vs HWCK
    if len(shape1) == len(shape2):
        return sorted(shape1) == sorted(shape2)

    small_shape = shape1 if len(shape1) < len(shape2) else shape2
    if small_shape == shape1:
        big_shape = shape2
    else:
        big_shape = shape1
    if len(small_shape) == 4:
        is_equal = cmp_4dtonc1hwc0_shape(small_shape, big_shape)
    if not is_equal:
        try:
            is_equal = cmp_ndtofractal_nz_shape(small_shape, big_shape)
        except ValueError:
            is_equal = False
    return is_equal


def is_fracz_format(format_name):
    if isinstance(format_name, str):
        if format_name.upper() == "FRACZ" or format_name.upper() == "FRACTAL_Z":
            return True
    return False


def is_valid_shape(shape):
    if not isinstance(shape, (list, tuple)):
        return False
    for s in shape:
        if not isinstance(s, int) or s < 0:
            return False
    return True
