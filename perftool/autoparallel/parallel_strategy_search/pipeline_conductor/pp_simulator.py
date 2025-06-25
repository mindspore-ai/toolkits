# Copyright 2024 Huawei Technologies Co., Ltd
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

from __future__ import annotations
from collections.abc import Iterable
import copy
import sys
import time
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib import colors
from matplotlib.transforms import ScaledTranslation
sys.setrecursionlimit(8192)

def format_2d_inputs(a, raw, col):
    if isinstance(a, (int, float)):
        return np.broadcast_to(a, (raw, col))
    if isinstance(a, (list, tuple)):
        if all(isinstance(item, (list, tuple)) for item in a):
            return np.array(a)
        elif all(isinstance(item, (int, float)) for item in a):
            return np.array([a])
        else:
            raise ValueError(f"Unsupported inputs: {a}")
    else:
        raise ValueError(f"Unsupported inputs: {a}")

def apply_color(l:list, c:list[str]):
    for i in range(len(l)):
        l[i] = f'{l[i]:.4f}' if isinstance(l[i], float) else l[i]
        l[i] = f"\033[{c[i]}m{l[i]}\033[0m"
    return l

def apply_format(l:list):
    s = f'{l[0]:^22}'
    symbol = ['=', '+', '+', '+', '+', '+']
    for i in range(len(l) - 1):
        s = f'{s}{symbol[i]}{l[i + 1]:^22}'
    return s

def color_mix(c1, c2, w1=0.5, w2=0.5):
    rgb = (np.array(colors.to_rgba(c1, 1)) * w1 + np.array(colors.to_rgba(c2, 1)) * w2) / (w1 + w2)
    return colors.to_rgba(rgb)


class CausalError(Exception):
    def __init__(self, msg, blocks:list[list[MicroBlockSim]]=None, loop:list[BlockSim]=[]) -> None:
        self.msg = msg
        self.canvas = PlotMgr(num_plots=1, figsize=(12, 6))
        self.canvas.draw_loop(blocks, loop, 0, False, False, True)
        self.canvas.ax[0].set_title("Block pipeline dependency")
        super().__init__()
        print(f"{self.canvas.msg}")

    def __str__(self):
        plt.show()
        return f"{self.msg}"
    

class CausalCommError(Exception):
    def __init__(self, msg, blocks:list[list[MicroBlockSim]]=None, loop:list[BlockSim]=[]) -> None:
        self.msg = msg
        self.canvas = PlotMgr(num_plots=1, figsize=(12, 6))
        self.canvas.draw_comm_loop(blocks, loop, 0)
        self.canvas.ax[0].set_title("Block comm pipeline dependency")
        super().__init__()
        print(f"{self.canvas.msg}")

    def __str__(self):
        plt.show()
        return f"{self.msg}"

def dfs_builder(comm=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            self = args[0]
            pre, left = (self.depend_pre, self.depend_left) if comm else (self.pre, self.left)
            if self.finish:
                return
            if pre is None or left is None:
                raise NotImplementedError
            if self.in_queue:
                raise ValueError
            self.in_queue = True
            res = func(*args, **kwargs)
            self.finish = True
            self.in_queue = False
            return res
        return wrapper
    return decorator

def timer(func:function):
    def wrapper(*args, **kwargs):
        T0 = time.time()
        res = func(*args, **kwargs)
        T1 = time.time() - T0
        print(f"function `{func.__name__}` time used: {T1:.4f} s", flush=True)
        return res
    return wrapper


class PlotMgr:
    def __init__(self, num_plots=2, ax_type='block', subplot_args=None, *args, **kwargs):
        self.fig = plt.figure(figsize=kwargs.get('figsize', (12, 8)))
        self.fig.subplots_adjust(wspace=0, hspace=0.4)
        ax_type = ax_type if isinstance(ax_type, (list, tuple)) else [ax_type] * num_plots
        self.ax = []
        for i in range(num_plots):
            if subplot_args is None:
                self.ax.append(self.fig.add_subplot(num_plots * 100 + 10 + i + 1))
            elif isinstance(subplot_args, Iterable) and len(subplot_args) >= num_plots:
                self.ax.append(self.fig.add_subplot(subplot_args[i]))
            else:
                raise ValueError(f"Unsupported subplot_args format: {subplot_args}")
                
    def _set_block_ax(self, ax:plt.Axes, pp:int) -> plt.Axes:
        ax.set_title("Pipeline Flow Timeline")
        ax.set_yticks(range(pp), [f"stage {p}" for p in range(pp)])
        for tick in ax.get_yticklabels():
            tick.set_verticalalignment('top')
            tick.set_transform(tick.get_transform() + ScaledTranslation(0, 0.05-1/pp, self.fig.dpi_scale_trans))
            tick.set_fontsize(12)
        ax.set_ylim(0, pp)
        ax.invert_yaxis()

    def _get_block_indices(self, blocks:list[list[MicroBlockSim]], mode='compact', equal_wide=False):
        if mode not in ['compact', 'joint', 'timeline']:
            raise ValueError(f"Get unsupported draw mode: {mode}")
        if mode == 'timeline' and not blocks[-1][-1].finish:
            raise ValueError(f"Block building should be finished before drawing timeline")
        block_index = []
        for p in range(len(blocks)):
            inds = []
            for block in blocks[p]:
                if mode == 'compact':
                    if block._type == 'c':
                        inds.append(1 if equal_wide else block.time)
                    else:
                        inds.append(0)
                elif mode == 'joint':
                    if block._type == 'c':
                        inds.append(1 if equal_wide else block.time)
                    else:
                        inds.append(block.time)
                else:
                    inds.append(1)
            inds.insert(0, 0)
            inds = np.cumsum(inds)
            block_index.append(inds)
        return block_index
    
    def draw_block(self, block_index:list[list[float]], blocks:list[list[MicroBlockSim]], ax_index:int = 0, equal_wide=False, width=1, phase=False):
        for p in range(len(blocks)):
            for b, block in enumerate(blocks[p]):
                if block._type == 'c':
                    block.draw(self.ax[ax_index], index=block_index[p][b], equal_wide=equal_wide, width=width, phase=phase)
        return self
    
    def draw_comm(self, block_index:list[list[float]], blocks:list[list[MicroBlockSim]], ax_index:int = 0, equal_wide=False, mode='compact'):
        for p in range(len(blocks)):
            for b, block in enumerate(blocks[p]):
                if block._type == 'c' and mode == 'compact':
                    if block.send_block:
                        block.send_block.draw(self.ax[ax_index], index=block_index[p][b], equal_wide=equal_wide)
                    if block.rec_block:
                        block.rec_block.draw(self.ax[ax_index], index=block_index[p][b], equal_wide=equal_wide)
                elif block._type in ['s', 'r'] and mode in ['joint', 'timeline']:
                    block.draw(self.ax[ax_index], index=block_index[p][b], equal_wide=equal_wide, mode=mode)
        return self

    def draw_connect(self, block_index:list[list[float]], blocks:list[list[MicroBlockSim]], ax_index:int = 0, equal_wide=False, mode='compact'):
        for p in range(len(blocks)):
            for b, block in enumerate(blocks[p]):
                if block._type == 'c' and mode == 'compact' and block.send_block:
                    dual_p = block.send_block.dual._stage
                    dual_ind = blocks[dual_p].index(block.send_block.dual.host)
                    block.send_block.draw_comm(self.ax[ax_index], index_from=block_index[p][b], index_to=block_index[dual_p][dual_ind], equal_wide=equal_wide, mode=mode)
                elif block._type == 's' and mode in ['joint', 'timeline']:
                    dual_p = block.dual._stage
                    dual_ind = blocks[dual_p].index(block.dual)
                    block.draw_comm(self.ax[ax_index], index_from=block_index[p][b], index_to=block_index[dual_p][dual_ind], equal_wide=equal_wide, mode=mode)
        return self

    def draw(self, blocks:list[list[MicroBlockSim]], ax_index:int = 0, comm=False, connect=False, equal_wide=False, mode='compact', phase=False) -> PlotMgr:
        pp = len(blocks)
        block_index = self._get_block_indices(blocks, mode=mode, equal_wide=equal_wide)
        width = max(np.max(block_index[p]) for p in range(pp)) if blocks[0][-1].end is None else max(blocks[p][-1].end for p in range(pp))
        self.draw_block(block_index, blocks, ax_index, equal_wide, width, phase=phase)
        if comm:
            self.draw_comm(block_index, blocks, ax_index, equal_wide, mode)
        if connect:
            self.draw_connect(block_index, blocks, ax_index, equal_wide, mode)
        self._set_block_ax(self.ax[ax_index], pp)
        self.ax[ax_index].set_xlim(0, width)
        return self

    def draw_loop(self, blocks:list[list[MicroBlockSim]], loop:list[BlockSim], ax_index:int = 0, comm=False, connect=False, equal_wide=False) -> PlotMgr:
        self.draw(blocks, ax_index, comm, connect, equal_wide, phase=True)
        block_index = self._get_block_indices(blocks, equal_wide=equal_wide)
        msg = 'dependency loop: '
        for b in range(len(loop) - 1):
            p = loop[b]._stage
            ind = blocks[p].index(loop[b])
            x1, y1, dx1, _ = loop[b].loc_size(block_index[p][ind], equal_wide)
            p = loop[b + 1]._stage
            ind = blocks[p].index(loop[b + 1])
            x2, y2, dx2, _ = loop[b + 1].loc_size(block_index[p][ind], equal_wide)
            msg = f'{msg} {loop[b].color_label} -> '
            self.ax[ax_index].annotate(None, xy=(x1 + dx1 / 2, y1), xytext=(x2 + dx2 / 2, y2),
                                       arrowprops=dict(fc='white', ec='r', arrowstyle='simple', shrinkA=5, shrinkB=5, connectionstyle="arc3,rad=-0.1"))
        self.msg = f'{msg} {loop[len(loop) - 1].color_label}'
        return self
    
    def draw_comm_loop(self, lines:list[list[BlockSim]], loop:list[BlockSim], ax_index:int = 0) -> PlotMgr:
        self.draw(lines, ax_index, True, True, True, 'joint', phase=True)
        block_index = self._get_block_indices(lines, mode='joint', equal_wide=True)
        msg = 'dependency loop: '
        for b in range(len(loop) - 1):
            p = loop[b]._stage
            ind = lines[p].index(loop[b])
            x1, y1, dx1, _ = loop[b].loc_size(block_index[p][ind], True, 'joint')
            p = loop[b + 1]._stage
            ind = lines[p].index(loop[b + 1])
            x2, y2, dx2, _ = loop[b + 1].loc_size(block_index[p][ind], True, 'joint')
            msg = f'{msg} {loop[b].color_label} -> '
            self.ax[ax_index].annotate(None, xy=(x1 + abs(dx1) / 2, y1), xytext=(x2 + abs(dx2) / 2, y2), size=10,
                                       arrowprops=dict(fc='white', ec='r', arrowstyle='simple', shrinkA=3, shrinkB=3, connectionstyle="arc3,rad=-0.1", lw=0.8))
        self.msg = f'{msg} {loop[len(loop) - 1].color_label}'
        return self
    
    def draw_mem(self, block_mem_list:list[np.ndarray], ax_index:int = 0) -> PlotMgr:
        for p in range(len(block_mem_list)):
            self.ax[ax_index].plot((block_mem_list[p].T)[0], (block_mem_list[p].T)[1], label=f"stage-{p}")
        self.ax[ax_index].set_title("Block Memory Timeline")
        self.ax[ax_index].set_xlim(0, max(np.max((block_mem.T)[0]) for block_mem in block_mem_list))

    def draw_info(self, bubble_info:dict, mem_info:list):
        info_list = [f'{k} bubble: {v:.4f}' for k, v in bubble_info.items()]
        self.fig.text(0.5, 0.5, ', '.join(info_list), ha='center', va='center',
                      fontdict={'fontsize':13, 'weight':'medium'}, color='C3')
        info_list = [f"{v:.2f}" for v in mem_info]
        self.fig.text(0.5, 0.05, f"peak memory: {', '.join(info_list)}", ha='center', va='center',
                      fontdict={'fontsize':13, 'weight':'medium'}, color='C0')

    def show(self):
        self.fig.legend(bbox_to_anchor=(0.22, 0.45))
        plt.show()
        plt.savefig("figure.pdf")  # 默认存储在当前目录


@dataclass
class BlockSim:
    _stage: int # p
    _state: str # s
    _id: int # m
    _virtual: int # v
    time: float
    _type: str
    start: float = None
    end: float = None
    pre: BlockSim = field(repr=False, default=None)
    left: BlockSim = field(repr=False, default=None)
    right: MicroBlockSim = field(repr=False, default=None)
    depend_pre: BlockSim = field(repr=False, default=None)
    depend_left: BlockSim = field(repr=False, default=None)
    finish = False
    in_queue = False
    _flag = False
    _color = '0;38'
    father: BlockSim = field(repr=False, default=None)
    @property
    def label(self) -> tuple:
        return (self._type, self._state, self._id, self._virtual, self._stage)
    @property
    def color_label(self) -> str:
        return f"\033[{self._color}m{self.label}\033[0m"
    @property
    def repr(self) -> str:
        raise NotImplementedError
    def draw(self, ax:plt.Axes, *args, **kwargs):
        raise NotImplementedError
    @dfs_builder(False)
    def build_without_comm(self) -> None:
        r"""Build pipeline timeline without comm blocks and dependency."""
        self.pre.build_without_comm()
        self.left.build_without_comm()
        self.start = max(self.pre.end, self.left.end)
        self.end = self.start + self.time
    @dfs_builder(True)
    def build_with_comm(self) -> None:
        r"""Build pipeline timeline with comm blocks and dependency."""
        self.depend_pre.build_with_comm()
        self.depend_left.build_with_comm()
        self.start = max(self.depend_pre.end, self.depend_left.end)
        self.end = self.start + self.time
    def reset_time_recursive(self) -> None:
        raise NotImplementedError
    def reset_time(self) -> None:
        self.start = None
        self.end = None
        self.finish = False
    def loc_size(self, x:int = 0, equal_wide=False, mode='compact') -> tuple:
        x = x if self.start is None else self.start
        dx = 1 if equal_wide else self.time
        return x, self._stage + 0.5, dx, 1
    def loop(self, comm=False) -> list[BlockSim]:
        if self._flag and not self.in_queue:
            return []
        l = []
        if self.in_queue:
            loop = [self]
            block = self.father
            while block.father and block is not self:
                block = block.father
                loop.append(block)
            return loop
        self._flag = True
        self.in_queue = True
        depends = [self.depend_pre, self.depend_left] if comm else [self.pre, self.left]
        for dep in depends:
            if dep:
                dep.father = self
                l.extend(dep.loop(comm=comm))
                dep.father = None
        self.in_queue = False
        return l
    
    def comm_loop(self) -> list[BlockSim]:
        return self.loop(True)

@dataclass
class HeadBlockSim(BlockSim):
    _stage: int # p
    _type: str = 'h'
    _id: int = field(repr=False, init=False)
    _state: str = field(repr=False, init=False)
    _virtual: int = field(repr=False, init=False)
    time: float = 0.
    start: float = 0.
    end: float = 0.
    finish = True
    @property
    def label(self) -> tuple:
        return (self._type, self._stage)
    @property
    def repr(self) -> str:
        s_list = []
        block = self
        while block:
            s_list.append(block.__repr__())
            block = block.right
        return '\n'.join(s_list)
    def draw(self, ax, *args, **kwargs):
        return
    def build_without_comm(self):
        return
    def build_with_comm(self):
        return
    def reset_time_recursive(self):
        return

@dataclass
class MicroBlockSim(BlockSim):
    _type: str = 'c'
    mem: float = 0.
    phase: str = None
    send_block: SendBlockSim = field(repr=False, default=None)
    rec_block: RecBlockSim = field(repr=False, default=None)
    def __post_init__(self):
        self._color = '1;34' if self._state == 'f' else '1;33'

    def draw(self, ax:plt.Axes, *args, **kwargs) -> None:
        x, y, dx, dy = self.loc_size(kwargs.get('index', 0), kwargs.get('equal_wide', False))
        color = (167/255, 184/255, 231/255) if self._state == 'f' else (255/255, 213/255, 143/255)
        mix_color = (240/255, 255/255, 245/255) if self._state == 'f' else (255/255, 240/255, 255/255)
        color = color_mix(mix_color, color, w1=self._virtual / 3)
        if self.phase == 'warmup' and kwargs.get('phase', False):
            edgecolor = 'lightblue'
        elif self.phase == 'cooldown' and kwargs.get('phase', False):
            edgecolor = 'orange'
        else:
            edgecolor = 'black'
        rect = Rectangle((x, y - dy / 2), dx, dy, facecolor=color, edgecolor=edgecolor, linewidth=0.4)
        if dx > 0.008 * kwargs.get('width', 0):
            ax.text(rect.xy[0] + dx / 2, rect.xy[1] + dy / 2, str(self._id), ha='center', va='center', color='black', fontdict={'fontsize':9})
        ax.add_patch(rect)

    def reset_time_recursive(self) -> None:
        if self.finish:
            self.pre.reset_time_recursive()
            self.left.reset_time_recursive()
            self.reset_time()
    
@dataclass
class CommBlockSim(BlockSim):
    host: MicroBlockSim = field(repr=False, default=None)
    dual: CommBlockSim = field(repr=False, default=None)
    def joint_loc(self, index:int = 0, equal_wide=False) -> tuple:
        raise NotImplementedError
    def get_triangle(self, x, y, dx, dy) -> tuple:
        raise NotImplementedError
    def draw(self, ax:plt.Axes, *args, **kwargs) -> None:
        color = (167/255, 184/255, 231/255) if self._state == 'f' else (255/255, 213/255, 143/255)
        mix_color = (240/255, 255/255, 255/255) if self._state == 'f' else (255/255, 240/255, 255/255)
        color = color_mix(mix_color, color, w1=1.2*self._virtual/3)
        index, equal_wide, mode = (kwargs.get('index', 0), kwargs.get('equal_wide', False), kwargs.get('mode', 'compact'))
        x, y, dx, dy = self.loc_size(index, equal_wide, mode)
        xy = self.get_triangle(x, y, dx, dy)
        tri = Polygon(xy, closed=True, facecolor=color, edgecolor='black', linewidth=0.4)
        ax.add_patch(tri)

@dataclass
class SendBlockSim(CommBlockSim):
    _type: str = 's'
    _color = '35'
    def loc_size(self, index:int = 0, equal_wide=False, mode='compact') -> tuple:
        host_x, _, host_dx, _ = self.host.loc_size(index, equal_wide)
        x, y, _, _ = super().loc_size(index, equal_wide)
        _dx = self.time
        _dy = min(np.sqrt(self.time) * 0.6, 0.6)
        if mode == 'compact':
            x = host_x + host_dx - _dx
        return x, y, _dx, _dy
    def get_triangle(self, x, y, dx, dy) -> tuple:
        return [[x, y - dy / 2], [x, y + dy / 2], [x + dx, y]]

    def draw_comm(self, ax:plt.Axes, *args, **kwargs) -> None:
        index_from, index_to = (kwargs.get('index_from', 0), kwargs.get('index_to', 0))
        equal_wide, mode = (kwargs.get('equal_wide', False), kwargs.get('mode', 'compact'))
        x, y, dx, _ = self.loc_size(index_from, equal_wide, mode)
        x_, y_, dx_, _ = self.dual.loc_size(index_to, equal_wide, mode)
        ax.annotate(None, xy=(x_ - dx_/2, y_), xytext=(x + dx/2, y), arrowprops=dict(ec='grey', arrowstyle='->', shrinkA=2, shrinkB=2))

    @dfs_builder(True)
    def build_with_comm(self) -> None:
        r"""Build pipeline timeline with comm blocks and dependency."""
        self.dual.depend_left.build_with_comm()
        self.depend_left.build_with_comm()
        self.start = max(self.depend_left.end, self.dual.depend_left.end)
        self.end = self.start + self.time

    def loop(self, comm=False) -> list[BlockSim]:
        if comm:
            return self.comm_loop()
        else:
            return super().loop(comm)
    def comm_loop(self) -> list[BlockSim]:
        if self._flag and not self.in_queue:
            return []
        l = []
        if self.in_queue:
            loop = [self]
            block = self.father
            while block.father and block is not self:
                block = block.father
                loop.append(block)
            return loop
        self._flag = True
        self.in_queue = True
        depends = [self.dual.depend_left, self.depend_left]
        for dep in depends:
            if dep:
                dep.father = self
                l.extend(dep.comm_loop())
                dep.father = None
        self.in_queue = False
        return l

@dataclass
class RecBlockSim(CommBlockSim):
    _type: str = 'r'
    _color = '32'
    def loc_size(self, index:int = 0, equal_wide=False, mode='compact') -> tuple:
        host_x, _, _, _ = self.host.loc_size(index, equal_wide)
        x, y, _, _ = super().loc_size(index, equal_wide)
        _dx = self.time
        _dy = min(np.sqrt(self.time) * 0.6, 0.6)
        if mode == 'compact':
            x = host_x
        return x, y, -_dx, -_dy
    def get_triangle(self, x, y, dx, dy) -> tuple:
        return [[x, y], [x - dx, y + dy / 2], [x - dx, y - dy / 2]]
    
    @dfs_builder(True)
    def build_with_comm(self) -> None:
        r"""Build pipeline timeline with comm blocks and dependency."""
        self.dual.build_with_comm()
        self.depend_left.build_with_comm()
        self.start = max(self.depend_left.end, self.dual.start)
        self.end = self.start + self.time

class PipelineBuild:
    @staticmethod
    def _inter_merge(a: list[MicroBlockSim], b: list[MicroBlockSim], delta: int = 0) -> list[MicroBlockSim]:
        res = []
        #res.extend(a)
        #res.extend(b)
        #return res
        if delta >= 0:
            res.extend(a[:delta])
            a = a[delta:]
        else:
            res.extend(b[:-delta])
            b = b[-delta:]
        stable_count = 0
        while len(a):
            block = a.pop(0)
            block.phase = 'stable'
            res.append(block)
            stable_count += 1
            if len(b):
                block = b.pop(0)
                block.phase = 'stable'
                res.append(block)
                stable_count += 1
            else:
                break
        if stable_count:
            res[-1].phase = 'cooldown'
        if len(a):
            res.extend(a)
        elif len(b):
            res.extend(b)
        return res

    @staticmethod
    def _build_chain(line: list[MicroBlockSim], p: int) -> list[BlockSim]:
        head = HeadBlockSim(p)
        left = head
        for item in line:
            left.right = item
            item.left = left
            left = item
        if p == 0:
            head.right.pre = head
        return line

    @staticmethod
    def build_1f1b(pp, micro_num, p, forward_time, backward_time, block_mem) -> list[BlockSim]:
        for_line = [MicroBlockSim(p, 'f', i, 0, forward_time, mem=block_mem) for i in range(micro_num)]
        back_line = [MicroBlockSim(p, 'b', i, 0, backward_time, mem=block_mem) for i in range(micro_num)]
        line = PipelineBuild._inter_merge(for_line, back_line, pp - p - 1)
        return PipelineBuild._build_chain(line, p)

    @staticmethod
    def build_virtualpipeline(pp, micro_num, vp, p, forward_time, backward_time, block_mem) -> list[BlockSim]:
        for_line = []
        back_line = []
        r = micro_num % pp
        for inter in range(micro_num // pp):
            for i in range(vp):
                if inter == 0:
                    for_line.extend([MicroBlockSim(p, 'f', m, i, forward_time[i], mem=block_mem[i], phase='warmup') for m in range(r)])
                    back_line.extend([MicroBlockSim(p, 'b', m, i, backward_time[i], mem=block_mem[i], phase='cooldown') for m in range(r)])
                for_line.extend([MicroBlockSim(p, 'f', r + m + inter * pp, i, forward_time[i], mem=block_mem[i], phase='warmup') for m in range(pp)])
                back_line.extend([MicroBlockSim(p, 'b', r + m + inter * pp, i, backward_time[i], mem=block_mem[i], phase='cooldown') for m in range(pp)])
        else:
            #back_line[14], back_line[15] = back_line[15], back_line[14]
            line = PipelineBuild._inter_merge(for_line, back_line, (vp + 1) * pp - 2 * p - 2 + r * (vp - 1))
        #if p == 13: line[34], line[33] = line[33], line[34]

        # if p == 15: line[31], line[32], line[33] = line[32],line[33],line[31]
        # if p <= 12 and p > 7: line[38+(12-p)*2], line[39+(12-p)*2] = line[39+(12-p)*2], line[38+(12-p)*2]
        # if p == 8: line[49], line[48] = line[48], line[49]
        # if p == 14: line[31], line[32]= line[32],line[31]
        return PipelineBuild._build_chain(line, p)
    
    @staticmethod
    def build_virtualpipeline2(pp, micro_num, vp, p, forward_time, backward_time, block_mem):
        for_line = []
        back_line = []
        r = micro_num % pp
        for inter in range(micro_num // pp):
            for i in range(vp):
                if inter == 0:
                    for_line.extend([MicroBlockSim(p, 'f', m, i, forward_time[i], mem=block_mem[i]) for m in range(r)])
                    back_line.extend([MicroBlockSim(p, 'b', m, i, backward_time[i], mem=block_mem[i]) for m in range(r)])
                for_line.extend([MicroBlockSim(p, 'f', r + m + inter * pp, i, forward_time[i], mem=block_mem[i]) for m in range(pp)])
                back_line.extend([MicroBlockSim(p, 'b', r + m + inter * pp, i, backward_time[i], mem=block_mem[i]) for m in range(pp)])
        line = PipelineBuild._inter_merge(for_line, back_line, vp  * pp - p - 1)
        return PipelineBuild._build_chain(line, p)


class PipelineSimulator:
    def __init__(self, block_time:list, micro_num:int, comm_time:float = 0.1, layer_recompute=False, block_mem=1, backward_ratio=2.):
        self.init(block_time, micro_num, comm_time, layer_recompute, block_mem, backward_ratio)

    def init(self, block_time, micro_num, comm_time, layer_recompute, block_mem, backward_ratio, *args, **kwargs):
        self.micro_num = micro_num
        self.pp, self.vp = self._base_init(block_time)
        self.block_num = 2 * self.vp * self.micro_num
        self.comm_time = comm_time
        self._input_format(block_time, layer_recompute, block_mem, backward_ratio)
        self._statistic_init()
        self._comm = True
        self.adjust_func_list = [self.swap_send_rec]
        # Construct pipeline blocks
        if self.vp == 1:
            self.blocks = [PipelineBuild.build_1f1b(self.pp, self.micro_num, p,
                                                    self.block_time[0, p],
                                                    self.backward_time[0, p],
                                                    self.block_mem[0, p]) for p in range(self.pp)]
        else:
            self.blocks = [PipelineBuild.build_virtualpipeline(self.pp, self.micro_num, self.vp, p,
                                                               self.block_time[:, p],
                                                               self.backward_time[:, p],
                                                               self.block_mem[:, p]) for p in range(self.pp)]
            if self.micro_num >= self.pp:
                self.adjust_func_list = [self.vpp_send_delay, self.residue_delay] + self.adjust_func_list
        self._build_block() # create connection among compute blocks
        self._build_comm_block() # create comm blocks for each compute block

    def run(self, comm=True, print_info=True) -> PipelineSimulator:
        self._comm = comm
        self._check_loop()
        if comm:
            self.lines = self._create_lines(*self.adjust_func_list)
            self._check_comm_loop()
            for b in range(self.block_num):
                for p in range(self.pp):
                    self.blocks[p][b].build_with_comm()
            self.lines[0][-1].build_with_comm()
        else:
            for p in range(self.pp):
                for block in self.blocks[p]:
                    block.build_without_comm()
        self._statistic_info()
        if print_info:
            self.print_info()
        return self
    
    def show(self, comm=True, connect=None) -> PipelineSimulator:
        self.canvas = PlotMgr(2, ['block', 'memory'])
        if self._comm:
            connect = True if connect is None else connect
            self.canvas.draw(self.lines, 0, comm, connect, False, 'timeline')
        else:
            connect = False if connect is None else connect
            self.canvas.draw(self.blocks, 0, comm, connect, False, 'timeline')
        self.canvas.draw_mem(self.states['block_mem_list'], 1)
        self.canvas.draw_info(self.bubbles, self.peak_memory)
        self.canvas.show()
        return self

    def print_info(self):
        print('\033[1;37m' + '—'*13, f'pp:{self.pp:>2}, vp:{self.vp:>2}, micro:{self.micro_num:>3} ' + '—'*12 + '\033[0m')
        print('-'*20, ' bubble ', '-'*20)
        print(apply_format(apply_color(list(self.bubbles.keys()), ['1;33', '1;32', '1;31', '1;35', '1;36'])))
        print(apply_format(apply_color(list(self.bubbles.values()), ['1;33', '1;32', '1;31', '1;35', '1;36'])))
        print('-'*20, ' memory ', '-'*20)
        print(f"peak memory: {', '.join([f'{v:.2f}' for v in self.peak_memory])}")
        return self

    def _base_init(self, block_time) -> tuple:
        if isinstance(block_time, (list, tuple)):
            if all(isinstance(item, (list, tuple)) for item in block_time):
                vp = len(block_time)
                pp = len(block_time[0])
            elif all(isinstance(item, (int, float)) for item in block_time):
                vp = 1
                pp = len(block_time)
            else:
                raise ValueError(f"Unsupported input format block_time: {block_time}")
        else:
            raise ValueError(f"Unsupported input format block_time: {block_time}")
        if self.micro_num < pp:
            raise ValueError(f" `micro_num`({self.micro_num}) should equal or larger than `pp`({pp})")
        return pp, vp
    
    def _input_format(self, block_time, layer_recompute, block_mem, backward_ratio) -> None:
        self.block_time = format_2d_inputs(block_time, self.vp, self.pp)
        if isinstance(layer_recompute, bool):
            self.layer_recompute = self.block_time if layer_recompute else format_2d_inputs(0, self.vp, self.pp)
        else:
            self.layer_recompute = format_2d_inputs(layer_recompute, self.vp, self.pp)
        if isinstance(block_mem, (int, float)):
            self.block_mem = self.block_time * block_mem
        else:
            self.block_mem = format_2d_inputs(block_mem, self.vp, self.pp)
        self.backward_ratio = format_2d_inputs(backward_ratio, self.vp, self.pp)

    def _statistic_init(self) -> None:
        self.forward_time = self.block_time
        self.backward_time = np.flip(self.block_time * self.backward_ratio + self.layer_recompute, axis=0)
        self.states = {'last_time': np.zeros(self.pp),
                       'warmup_time': np.zeros(self.pp),
                       'cooldown_time': np.zeros(self.pp),
                       'stable_free_time': (np.zeros((self.vp, self.pp)), np.zeros((self.vp, self.pp))),
                       'block_mem_list': [np.array([[0, 0]]) for _ in range(self.pp)]}
        self.model_compute_time = (np.sum(self.forward_time) + np.sum(self.forward_time * self.backward_ratio)) * self.micro_num
        self.hardware_compute_time = (np.sum(self.forward_time) + np.sum(self.backward_time)) * self.micro_num
        self.bubbles = {'real': 0,
                        'ideal': (self.pp - 1) / self.vp / self.micro_num,
                        'imba': 0,
                        'comm': 0}
        if np.sum(self.layer_recompute) > 1e-5:
            self.bubbles['recompute'] = self.hardware_compute_time / self.model_compute_time - 1
        p, v, m = self.pp, self.vp, self.micro_num
        if self.vp == 1:
            if self.pp == 2:
                self.bubbles['comm'] = 4*m
            elif self.pp %2 == 0:
                self.bubbles['comm'] = 4*p*m + 4*p**2 - 14*p
            else:
                self.bubbles['comm'] = 4*p*m + 4*p**2 - 12*p
        elif self.pp <= 5:
            comm_coef_list = [[4, -2, 0], [6, -2, -6], [4, 0, 12], [6, -2, 40]]
            self.bubbles['comm'] = np.dot(np.array([p*v*m, m*p, 1]), comm_coef_list[self.pp - 2])
        elif self.pp % 2 == 0:
            self.bubbles['comm'] = 4*p*v*m + 4*p**2 - 13*p
        else:
            self.bubbles['comm'] = 6*p*v*m - 2*v*p**2 + 4*v*p - 2*p*m + 6*p**2 -16*p

        self.bubbles['comm'] *= self.comm_time / self.model_compute_time

    def _statistic_info(self) -> None:
        for p in range(self.pp):
            blocks = self.lines[p] if self._comm else self.blocks[p]
            current_mem = 0
            for block in blocks:
                if block._type == 'c' and block._state == 'f':
                    current_mem += block.mem
                elif block._type == 'c' and block._state == 'b':
                    if not self._comm or not block.rec_block:
                        current_mem -= block.mem
                elif block._type == 'r' and block.host._state == 'b':
                    current_mem -= block.host.mem
                    block = block.host
                else:
                    continue
                self.states['block_mem_list'][p] = np.append(self.states['block_mem_list'][p], np.array([[block.end, current_mem]]), axis=0)
            self.states['block_mem_list'][p] = np.append(self.states['block_mem_list'][p], np.array([[blocks[-1].end, current_mem]]), axis=0)
        self.peak_memory = [np.max((self.states['block_mem_list'][p].T)[1]) for p in range(self.pp)]
        self.end_time = max(np.max((self.states['block_mem_list'][p].T)[0]) for p in range(self.pp))
        self.bubbles['real'] = (self.pp * self.end_time - self.model_compute_time) / self.model_compute_time
        self.bubbles['imba'] = self.bubbles['real'] - self.bubbles['ideal'] + 1e-10
        if not self._comm:
            self.bubbles.pop('comm')
        else:
            self.bubbles['imba'] -= self.bubbles['comm']
        if self.bubbles.get('recompute'):
            self.bubbles['imba'] -= self.bubbles['recompute']

    def _get_pre_label(self, label:tuple) -> tuple:
        t, s, m, v, p = label
        if (s, v, p) == ('f', 0, 0):
            return ('h', p)
        if (s, p) == ('f', 0):
            return (t, s, m, v - 1, self.pp - 1)
        if (s, p) == ('b', self.pp - 1):
            if v == 0:
                return (t, 'f', m, self.vp - 1, p)
            return (t, s, m, v - 1, 0)
        if s == 'f':
            return (t, s, m, v, p - 1)
        elif s == 'b':
            return (t, s, m, v, p + 1)
        raise ValueError(f"Illegal label: {label}")

    def _build_block(self) -> None:
        r"""Build `pre` relation for computation blocks."""
        books = {self.blocks[0][0].pre.label: self.blocks[0][0].pre}
        for p in range(self.pp):
            for item in self.blocks[p]:
                books[item.label] = item
        for p in range(self.pp):
            block = self.blocks[p][0]
            while block:
                pre_label = self._get_pre_label(block.label)
                block.pre = books[pre_label]
                block = block.right

    def _build_comm_block(self) -> None:
        r"""Build `send_block` and `rec_block` relation among a computation block and two comm blocks."""
        for p in range(self.pp):
            block = self.blocks[p][0]
            while block:
                pre = block.pre
                if pre._stage != block._stage:
                    block.rec_block = RecBlockSim(p, block._state, block._id, block._virtual, self.comm_time)
                    pre.send_block = SendBlockSim(pre._stage, pre._state, pre._id, pre._virtual, self.comm_time)
                    block.rec_block.host = block
                    block.rec_block.dual = pre.send_block
                    pre.send_block.host = pre
                    pre.send_block.dual = block.rec_block
                    block.depend_pre = block.rec_block
                    block.rec_block.depend_pre = pre.send_block
                    pre.send_block.depend_pre = pre
                else:
                    block.depend_pre = pre
                block = block.right

    def _check_loop(self) -> None:
        loop = self.blocks[0][-1].loop()
        if loop:
            raise CausalError('Block dependency exist loops!', self.blocks, loop)
        for p in range(self.pp):
            for block in self.blocks[p]:
                block._flag = False

    def _check_comm_loop(self) -> None:
        loop = self.lines[0][-1].comm_loop()
        if loop:
            raise CausalCommError('Block comm dependency exist loops!', self.lines, loop)
        for p in range(self.pp):
            for block in self.lines[p]:
                block._flag = False

    def _create_lines(self, *adjust_func) -> list[list[BlockSim]]:
        lines = [copy.copy(self.blocks[p]) for p in range(self.pp)]
        for p in range(self.pp):
            for b in range(self.block_num):
                block = self.blocks[p][b]
                pre = block.pre
                if block.rec_block:
                    lines[p].insert(lines[p].index(block), block.rec_block)
                    if pre._type == 'h':
                        lines[pre._stage].insert(0, pre.send_block)
                    else:
                        lines[pre._stage].insert(lines[pre._stage].index(pre) + 1, pre.send_block)
        for func in adjust_func:
            lines = func(lines)
        for p in range(self.pp):
            for b, block in enumerate(lines[p]):
                if b == 0:
                    block.depend_left = block.left if block.left else block.host.left
                else:
                    block.depend_left = lines[p][b - 1]
        return lines
    
    def _get_block_phase(self, p:int, b:int) -> str:
        r = self.micro_num % self.pp
        if b < (self.vp + 1) * self.pp - 2 * p - 2 + r:
            return 'warmup'
        elif b > self.block_num - (self.vp + 1) * self.pp + 2 * p:
            return 'cooldown'
        else:
            return 'stable'
        
    def _send_block_delay(self, lines, p:int, b:int, distance:int) -> None:
        i_send = lines[p].index(self.blocks[p][b].send_block)
        send_block = lines[p].pop(i_send)
        i_new = lines[p].index(self.blocks[p][b + distance]) + 1
        lines[p].insert(i_new, send_block)
    # @timer
    def swap_send_rec(self, lines) -> list[list[BlockSim]]:
        for p in range(self.pp):
            for b, block in enumerate(self.blocks[p]):
                if b >= len(self.blocks[p]) - 1:
                    continue
                i_b = lines[p].index(block)
                i_bn = lines[p].index(self.blocks[p][b + 1])
                if i_bn - i_b == 3:
                    if p % 2 == 0 and lines[p][i_b+1]._type == 'r' and lines[p][i_b+2]._type == 's':
                        lines[p][i_b+1], lines[p][i_b+2] = lines[p][i_b+2], lines[p][i_b+1]
                    if p % 2 == 1 and lines[p][i_b+1]._type == 's' and lines[p][i_b+2]._type == 'r':
                        if block.phase == 'warmup' and self.blocks[p][b + 1].phase == 'cooldown':
                            continue
                        lines[p][i_b+1], lines[p][i_b+2] = lines[p][i_b+2], lines[p][i_b+1]
                    if lines[p][i_b+1].dual._stage == lines[p][i_b+2].dual._stage:
                        pd = lines[p][i_b+1].dual._stage
                        j_b1 = lines[pd].index(lines[p][i_b+1].dual)
                        j_b2 = lines[pd].index(lines[p][i_b+2].dual)
                        if j_b1 > j_b2:
                            lines[p][i_b+1], lines[p][i_b+2] = lines[p][i_b+2], lines[p][i_b+1]
                if i_bn - i_b == 4:
                    if lines[p][i_b+1].dual._stage == lines[p][i_b+2].dual._stage and lines[p][i_b+2].dual._stage == lines[p][i_b+3].dual._stage:
                        if lines[p][i_b+1]._type == 's' and lines[p][i_b+2]._type == 's' and lines[p][i_b+3]._type == 'r':
                            lines[p][i_b+1], lines[p][i_b+2] = lines[p][i_b+2], lines[p][i_b+1]
                            # lines[p][i_b+2], lines[p][i_b+3] = lines[p][i_b+3], lines[p][i_b+2]
        return lines
    # @timer
    def vpp_send_delay(self, lines) -> list[list[BlockSim]]:
        if self.micro_num % self.pp != 0:
            return lines
        for p in range(self.pp):
            for b, block in enumerate(self.blocks[p]):
                if block.send_block is not None and block.phase == 'stable':
                    self._send_block_delay(lines, p, b, 1)
        return lines
    # @timer
    def residue_delay(self, lines) -> list[list[BlockSim]]:
        r = self.micro_num % self.pp
        if not r:
            return lines
        for p in range(self.pp):
            for b, block in enumerate(self.blocks[p]):
                if block.send_block is None:
                    continue
                if p == self.pp - 1 and block._id < self.pp + r and block._state == 'f':
                    self._send_block_delay(lines, -1, b, r + max(0, block._id - self.pp + 1))
                elif p == 0 and block._id < self.pp + r and block._state == 'b':
                    if self.micro_num // self.pp == 1:
                        self._send_block_delay(lines, 0, b, r)
                    else:
                        self._send_block_delay(lines, 0, b, r + self.pp)
                elif block.phase == 'stable':
                    self._send_block_delay(lines, p, b, 1)
        return lines

def test_comm_imba_zero():
    for pp in range(2, 10):
        for m in range(pp, 3*pp+1): # check vp=1 imba
            sim = PipelineSimulator(np.ones(pp).tolist(), m, comm_time=0.1).run(print_info=False)
            if sim.bubbles['imba'] > 0.001:
                sim.print_info()
        for vp in range(2, 6): # check vp>1 imba
            for m in range(pp, 4*pp+1, pp):
                sim = PipelineSimulator(np.ones((vp, pp)).tolist(), m, comm_time=0.1).run(print_info=False)
                if sim.bubbles['imba'] > 0.001:
                    sim.print_info()

'''
当前缺陷：
    1. 开comm与vpp时, pp要求整除micro, 否则可能成环;
'''

if __name__ == '__main__':
    layer_dis = [[2*0.6, 1.6, 1, 3, 2, 3, 4, 2], [6, 6, 6, 4, 6, 5, 5, 5+1.5]]
    recompute = [[2*0.6, 2, 1, 3, 2, 3, 4, 2], [6, 6, 6, 4, 6, 4, 5, 0]]
    PipelineSimulator(layer_dis, 128, 0.07, recompute).run(False).show(False, False)
