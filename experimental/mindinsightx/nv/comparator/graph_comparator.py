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
"""Computational graph and dump files comparator."""
import copy
import os.path

import numpy as np
import xlsxwriter

import mindinsightx.nv.comparator.dump_comparator as dcmp
from mindinsightx.nv.mapper.node_mapper import NodeMapper

# dump compare result strings
DUMP_MISSING = 'missing'
DUMP_NOT_EQUAL = 'not equal'
DUMP_EQUAL = 'equal'
DUMP_SHAPE_NOT_MATCH = 'shape not match'
DUMP_FORMAT_NOT_MATCH = 'format not match'
DUMP_SHAPE_FORMAT_NOT_MATCH = 'shape and format not match'
DUMP_SIZE_NOT_MATCH = 'size not match'
DUMP_CMP_ERROR = 'error'
DUMP_EQUAL_SOME_MISSING = 'equal but some files missing'
DUMP_NO_COMMON_INDEX = 'no common output index'


class DumpCmpResult:
    """Dump compare result."""
    def __init__(self, value_cmp_result, index=-1):
        self.value_cmp_result = value_cmp_result
        self.value_cmp_measure = None
        self.value_cmp_detail = None
        self.value_count = 0
        self.abs_diff_sum = None
        self.abs_diff_avg = None
        self.abs_diff_max = None
        self.max_value = [None, None]
        self.min_value = [None, None]
        self.index = index
        self.breakdowns = None


class GraphComparator:
    """
    Comparator of 2 graphs and their dump files.

    Notes:
        If dump_dir_0 or dump_dir_1 is not specified then no dump file comparison will be done.

    Args:
        graph0 (Graph): Graph of index 0.
        graph1 (Graph): Graph of index 1.
        dump_dir_0 (str, optional): Dump file directory of graph0.
        dump_dir_1 (str, optional): Dump file directory of graph1.
        dump_comparator (DumpComparator, optional), Dump value comparator. AllClose will be used if not specified.
        ignore_shape (bool, optional): ignore output shape when mapping graph nodes, None means auto detect.
        force (bool): Force to comparable the graphs event if they are not mappable.
        diagnostic (bool): Detect problematic nodes.

    Examples:
        >>> from mindinsightx.nv import constants
        >>> from mindinsightx.nv.graph.graph import Graph
        >>> from mindinsightx.nv.comparator.graph_comparator import GraphComparator
        >>>
        >>> me_graph = Graph.create(constants.ME_GRAPH_TYPE, '1.0.0')
        >>> me_graph.load('me_graph.pb')
        >>> tf_graph = Graph.create(constants.TF_GRAPH_TYPE, '1.14')
        >>> tf_graph.load('tf_graph.pb')
        >>>
        >>> comparator = GraphComparator(me_graph, tf_graph, 'me_dump_dir', 'tf_dump_dir')
        >>> comparator.compare()
        >>> comparator.save_as_xlsx('report.xlsx')
    """
    def __init__(self,
                 graph0,
                 graph1,
                 dump_dir_0=None,
                 dump_dir_1=None,
                 dump_comparator=None,
                 ignore_shape=None,
                 force=False,
                 diagnostic=False):

        self._graphs = (graph0, graph1)
        self._node_mapper = NodeMapper(graph0, graph1, ignore_shape=ignore_shape, force=force)

        if dump_dir_0 and dump_dir_1:
            self._dump_dirs = (dump_dir_0, dump_dir_1)
            self._dump_comparator = dcmp.AvgAbsError() if dump_comparator is None else dump_comparator

            dump_mapper0 = graph0.create_dump_mapper(dump_dir_0)
            if dump_mapper0 is None:
                raise ValueError(f'graph0({graph0.type_ver_code}) does not support dump comparison!')
            dump_mapper1 = graph1.create_dump_mapper(dump_dir_1)
            if dump_mapper1 is None:
                raise ValueError(f'graph1({graph1.type_ver_code}) does not support dump comparison!')
            self._dump_mappers = (dump_mapper0, dump_mapper1)
        else:
            self._dump_dirs = None
            self._dump_comparator = None
            self._dump_mappers = None

        if diagnostic and not self._dump_mappers:
            raise ValueError(f'dump_dir_0 and dump_dir_1 must be specified if diagnostic is True')
        self._diagnostic = diagnostic

        self._output_cmp_results = None
        self._problem_node_recs = None
        self._node_type_stats = None
        self._graph_stats = None
        self._xlsx_bold = None

    @property
    def graphs(self):
        """Get the graphs."""
        return self._graphs

    @property
    def node_mapper(self):
        """Get the node mapper."""
        return self._node_mapper

    @property
    def output_cmp_results(self):
        """Get the output dump compare results."""
        return self._output_cmp_results

    @property
    def node_type_stats(self):
        """Get the node type statistics."""
        return self._node_type_stats

    @property
    def graph_stats(self):
        """Get the graph statistics."""
        return self._graph_stats

    @property
    def compared(self):
        """Check if compare() was invoked."""
        return self._node_mapper.processed

    def compare(self, max_steps=None):
        """
        Do compare the graphs and dump files.

        Args:
            max_steps (int, optional): Maximum steps of the node mapping process. None means unlimited.
        """
        if self.compared:
            return

        if self._dump_mappers:
            for i in (0, 1):
                print(f'Building graph{i} node to dump file mapping...')
                self._dump_mappers[i].process(map_input=self._diagnostic)
                out_count = len(self._dump_mappers[i].output_map)
                out_percent = (len(self._dump_mappers[i].output_map)/len(self._graphs[i].nodes)) * 100
                if self._dump_mappers[i].input_map is None:
                    print('{}({:.2f}%) nodes have output'.format(out_count, out_percent))
                else:
                    in_count = len(self._dump_mappers[i].input_map)
                    in_percent = (len(self._dump_mappers[i].input_map) / len(self._graphs[i].nodes)) * 100
                    print('{}({:.2f}%) nodes have input, {}({:.2f}%) have output'.
                          format(in_count, in_percent, out_count, out_percent))

        print('Building node to node mapping...')
        self._node_mapper.process(max_steps=max_steps)

        if self._dump_mappers:
            self._cmp_output_dumps()

        if self._diagnostic and self._dump_mappers and self._output_cmp_results:
            self._diagnose()

        self._gather_stats()

    def save_as_xlsx(self, path, top_k=5, user_data=None):
        """
        Save as excel(.xlsx) report.

        Args:
            path (str): The file path to be saved.
            top_k (int): The k number of the top-k similar node list. Non-positive top_k means do not reports
                the top-k list.
            user_data (dict[str, str], optional): Extra key-value pairs user data to be saved in the
                summary sheet.
        """
        if not self.compared:
            raise RuntimeError(f'Function compare() not yet invoked.')

        workbook = xlsxwriter.Workbook(path)
        self._xlsx_bold = workbook.add_format({'bold': True})
        sheet = workbook.add_worksheet(name=f"summary")

        self._xlsx_save_summary(sheet, user_data)

        if self._problem_node_recs is not None:
            sheet = workbook.add_worksheet(name=f"diagnostic")
            self._xlsx_save_diagnostic(sheet)

        for i in (0, 1):
            sheet = workbook.add_worksheet(name=f'graph{i} mapped')
            self._xlsx_save_mapped(sheet, i)
            if top_k > 0:
                sheet = workbook.add_worksheet(name=f'graph{i} top-{top_k}')
                self._xlsx_save_top_k(sheet, i, top_k)
            if self._dump_mappers:
                sheet = workbook.add_worksheet(name=f'graph{i} dump')
                self._xlsx_save_dump_list(sheet, i)
        workbook.close()
        os.chmod(path, mode=0o600)

    def _xlsx_save_summary(self, sheet, user_data):
        """Save the summary sheet."""
        sheet.set_column(0, 0, width=50)
        sheet.set_column(7, 7, width=50)
        row = 0
        for i in (0, 1):
            sheet.write(row, 0, f'graph {i} ({self._graphs[i].type_ver_code}):', self._xlsx_bold)
            sheet.write(row, 1, self._graphs[i].real_path)
            sheet.write(row, 2, 'node#:', self._xlsx_bold)
            # Exclude Parameter and TupleGetItem in the table(start)
            nodes_without_para = [node for node in self._graphs[i].nodes if node.node_type != 'Parameter' and node.node_type != 'TupleGetItem']
            sheet.write(row, 3, str(len(nodes_without_para)))
            # (end)
            sheet.write(row, 4, 'edge#:', self._xlsx_bold)
            sheet.write(row, 5, str(len(self._graphs[i].edges)))
            row += 1
            sheet.write(row, 0, 'dump dir:', self._xlsx_bold)
            if self._dump_dirs:
                sheet.write(row, 1, os.path.realpath(self._dump_dirs[i]))
            row += 1

        sheet.write(row, 0, 'dump comparator:', self._xlsx_bold)
        if self._dump_comparator is not None:
            sheet.write(row, 1, str(self._dump_comparator))
        row += 1

        if user_data:
            for key, value in user_data.items():
                sheet.write(row, 0, str(key)+':', self._xlsx_bold)
                sheet.write(row, 1, str(value))
                row += 1

        row += 1
        max_stat_item_len = max(len(self._node_type_stats[0]), len(self._node_type_stats[1]))
        graph_stat_row = row + max_stat_item_len + 2
        sheet.write(graph_stat_row, 0, 'total:', self._xlsx_bold)
        sheet.write(graph_stat_row + 1, 0, 'rate:', self._xlsx_bold)

        for i in (0, 1):
            col = i * 7
            stats = list(self._node_type_stats[i].items())
            stats.sort(key=lambda x: x[1]['node_count'], reverse=True)
            tmp_row = row
            sheet.write(tmp_row, col, f'Graph {i}', self._xlsx_bold)
            sheet.write(tmp_row, col + 1, f'Total', self._xlsx_bold)
            sheet.write(tmp_row, col + 2, f'Mapped', self._xlsx_bold)
            sheet.write(tmp_row, col + 3, f'Unmapped', self._xlsx_bold)
            sheet.write(tmp_row, col + 4, f'Dumped', self._xlsx_bold)
            sheet.write(tmp_row, col + 5, f'Dump Eq.', self._xlsx_bold)
            tmp_row += 1
            # Exclude Parameter and TupleGetItem in the table(start)
            for node_type, node_type_stat in stats:
                if node_type in ['Parameter', 'TupleGetItem']:
                    continue
                sheet.write(tmp_row, col, str(node_type))
                sheet.write(tmp_row, col + 1, str(node_type_stat['node_count']))
                sheet.write(tmp_row, col + 2, str(node_type_stat['mapped']))
                sheet.write(tmp_row, col + 3, str(node_type_stat['unmapped']))
                sheet.write(tmp_row, col + 4, str(node_type_stat['dumped']))
                sheet.write(tmp_row, col + 5, str(node_type_stat['dump_equal']))
                tmp_row += 1
            # (end)
            graph_stat = self._graph_stats[i]
            sheet.write(graph_stat_row, col + 1, str(graph_stat['node_count']))
            sheet.write(graph_stat_row, col + 2, str(graph_stat['mapped']))
            sheet.write(graph_stat_row, col + 3, str(graph_stat['unmapped']))
            sheet.write(graph_stat_row, col + 4, str(graph_stat['dumped']))
            sheet.write(graph_stat_row, col + 5, str(graph_stat['dump_equal']))
            sheet.write(graph_stat_row + 1, col + 2,
                        str(graph_stat['mapped']/graph_stat['node_count']))
            sheet.write(graph_stat_row + 1, col + 3,
                        str(graph_stat['unmapped']/graph_stat['node_count']))
            sheet.write(graph_stat_row + 1, col + 4,
                        str(graph_stat['dumped'] / graph_stat['node_count']))
            sheet.write(graph_stat_row + 1, col + 5,
                        str(graph_stat['dump_equal'] / graph_stat['node_count']))

    def _xlsx_save_diagnostic(self, sheet):
        """Save diagnostic results."""
        sheet.write(0, 0, 'Node', self._xlsx_bold)
        sheet.write(0, 1, 'Topo Id', self._xlsx_bold)
        sheet.write(0, 2, 'Type', self._xlsx_bold)
        sheet.write(0, 3, 'Shape', self._xlsx_bold)
        sheet.write(0, 4, 'Mapped Node', self._xlsx_bold)
        sheet.write(0, 5, 'Topo Id', self._xlsx_bold)
        sheet.write(0, 6, 'Type', self._xlsx_bold)
        sheet.write(0, 7, 'Shape', self._xlsx_bold)
        sheet.write(0, 8, 'Similarity', self._xlsx_bold)
        sheet.write(0, 9, 'Input Cmp', self._xlsx_bold)
        sheet.write(0, 10, 'Output Cmp', self._xlsx_bold)
        sheet.write(0, 11, 'Measure', self._xlsx_bold)
        sheet.write(0, 12, 'Detail', self._xlsx_bold)

        row = 1
        for rec in self._problem_node_recs:
            node_pair = rec[0]
            input_cmp_result = rec[1]
            output_cmp_result = rec[2]
            node = node_pair[0]
            # Exclude Parameter and TupleGetItem in the table(start)
            if node.node_type in ['Parameter', 'TupleGetItem']:
                continue
            # (end)
            mapped_node = node_pair[1]
            similarity = node_pair.similarity
            self._xlsx_write_mapped_node(sheet, row, 0, node, mapped_node, similarity, None, show_details=False)
            sheet.write(row, 9, input_cmp_result.value_cmp_result)
            sheet.write(row, 10, output_cmp_result.value_cmp_result)

            if output_cmp_result.value_cmp_measure is not None:
                sheet.write(row, 11, output_cmp_result.value_cmp_measure)
            if output_cmp_result.value_cmp_detail is not None:
                sheet.write(row, 12, output_cmp_result.value_cmp_detail)
            row += 1

    def _xlsx_save_mapped(self, sheet, graph_idx):
        """Save the 1 to 1 node mapping sheet."""
        sheet.write(0, 0, 'Node', self._xlsx_bold)
        sheet.write(0, 1, 'Topo Id', self._xlsx_bold)
        sheet.write(0, 2, 'Type', self._xlsx_bold)
        sheet.write(0, 3, 'Shape', self._xlsx_bold)
        sheet.write(0, 4, 'Depth', self._xlsx_bold)
        sheet.write(0, 5, 'Homo Depth', self._xlsx_bold)
        sheet.write(0, 6, 'Mapped Node', self._xlsx_bold)
        sheet.write(0, 7, 'Topo Id', self._xlsx_bold)
        sheet.write(0, 8, 'Type', self._xlsx_bold)
        sheet.write(0, 9, 'Shape', self._xlsx_bold)
        sheet.write(0, 10, 'Similarity', self._xlsx_bold)
        if self._output_cmp_results:
            sheet.write(0, 11, 'Output Cmp', self._xlsx_bold)
            sheet.write(0, 12, 'Measure', self._xlsx_bold)
            sheet.write(0, 13, 'Detail', self._xlsx_bold)
            sheet.write(0, 14, 'Abs Diff Sum', self._xlsx_bold)
            sheet.write(0, 15, 'Abs Diff Avg', self._xlsx_bold)
            sheet.write(0, 16, 'Abs Diff Max', self._xlsx_bold)
            sheet.write(0, 17, 'Min Value', self._xlsx_bold)
            sheet.write(0, 18, 'Max Value', self._xlsx_bold)
            sheet.write(0, 19, 'Cmp Min Value', self._xlsx_bold)
            sheet.write(0, 20, 'Cmp Max Value', self._xlsx_bold)

        nodes = list(self._graphs[graph_idx].nodes)

        nodes.sort(key=lambda x: x.internal_id)

        row = 1
        cmp_graph_idx = 1 - graph_idx
        for node in nodes:
            # Exclude Parameter and TupleGetItem in the table(start)
            if node.node_type in ['Parameter', 'TupleGetItem']:
                continue
            # (end)
            dump_cmp_result = None
            if self._output_cmp_results:
                dump_cmp_result = self._output_cmp_results[graph_idx].get(node.internal_id, None)

            node_pair = self._node_mapper.map_strict(graph_idx, node.internal_id)
            mapped_node = None if node_pair is None else node_pair[cmp_graph_idx]
            similarity = None if node_pair is None else node_pair.similarity

            if dump_cmp_result is None or not dump_cmp_result.breakdowns:
                self._xlsx_write_mapped_node(sheet, row, graph_idx, node, mapped_node, similarity, dump_cmp_result)
                row += 1
            else:
                for result in dump_cmp_result.breakdowns:
                    self._xlsx_write_mapped_node(sheet, row, graph_idx, node, mapped_node, similarity, result)
                    row += 1

    def _xlsx_write_mapped_node(self, sheet, row, graph_idx, node, mapped_node,
                                similarity, dump_cmp_result, show_details=True):
        """Write mapped nodes list to sheet."""
        if dump_cmp_result is None or dump_cmp_result.index < 0:
            name_suffix = ''
        else:
            name_suffix = ':output_' + str(dump_cmp_result.index)
        col = 0
        sheet.write(row, col, node.name + name_suffix)
        col += 1
        sheet.write(row, col, str(node.topo_id))
        col += 1
        sheet.write(row, col, str(node.node_type))
        col += 1
        sheet.write(row, col, str(node.shape))
        col += 1
        if show_details:
            sheet.write(row, col, str(node.depth))
            col += 1
            sheet.write(row, col, str(node.homo_depth))
            col += 1

        if mapped_node is not None:
            sheet.write(row, col, mapped_node.name + name_suffix)
            col += 1
            sheet.write(row, col, str(mapped_node.topo_id))
            col += 1
            sheet.write(row, col, str(mapped_node.node_type))
            col += 1
            sheet.write(row, col, str(mapped_node.shape))
            col += 1
            sheet.write(row, col, '{:.5f}'.format(similarity))
            col += 1

        if dump_cmp_result is not None:
            self._xlsx_write_not_none(sheet, row, col, dump_cmp_result.value_cmp_result)
            col += 1
            if show_details:
                self._xlsx_write_not_none(sheet, row, col, dump_cmp_result.value_cmp_measure)
                col += 1
                self._xlsx_write_not_none(sheet, row, col, dump_cmp_result.value_cmp_detail)
                col += 1
                self._xlsx_write_not_none(sheet, row, col, dump_cmp_result.abs_diff_sum)
                col += 1
                self._xlsx_write_not_none(sheet, row, col, dump_cmp_result.abs_diff_avg)
                col += 1
                self._xlsx_write_not_none(sheet, row, col, dump_cmp_result.abs_diff_max)
                col += 1
                self._xlsx_write_not_none(sheet, row, col, dump_cmp_result.min_value[graph_idx])
                col += 1
                self._xlsx_write_not_none(sheet, row, col, dump_cmp_result.max_value[graph_idx])
                col += 1
                self._xlsx_write_not_none(sheet, row, col, dump_cmp_result.min_value[1 - graph_idx])
                col += 1
                self._xlsx_write_not_none(sheet, row, col, dump_cmp_result.max_value[1 - graph_idx])
                col += 1

    @staticmethod
    def _xlsx_write_not_none(sheet, row, col, val):
        if val is not None:
            sheet.write(row, col, str(val))

    def _xlsx_save_top_k(self, sheet, graph_idx, top_k):
        """Save the top-k similar node list sheet."""
        sheet.write(0, 0, 'Node', self._xlsx_bold)
        sheet.write(0, 1, 'Topo Id', self._xlsx_bold)
        sheet.write(0, 2, 'Type', self._xlsx_bold)
        sheet.write(0, 3, 'Shape', self._xlsx_bold)
        sheet.write(0, 4, 'Sim Node', self._xlsx_bold)
        sheet.write(0, 5, 'Topo Id', self._xlsx_bold)
        sheet.write(0, 6, 'Type', self._xlsx_bold)
        sheet.write(0, 7, 'Shape', self._xlsx_bold)
        sheet.write(0, 8, 'Similarity', self._xlsx_bold)
        sheet.write(0, 9, 'Mapped', self._xlsx_bold)

        nodes = list(self._graphs[graph_idx].nodes)
        nodes.sort(key=lambda x: x.internal_id)
        cmp_graph_idx = 1 - graph_idx
        row = 1
        for node in nodes:
            # Exclude Parameter and TupleGetItem in the table(start)
            if node.node_type in ['Parameter', 'TupleGetItem']:
                continue
            # (end)
            node_pairs = self._node_mapper.top_k(graph_idx, node.internal_id, top_k)
            mapped_node = self._node_mapper.map_strict(graph_idx, node.internal_id, ret_info='node')
            if node_pairs:
                for node_pair in node_pairs:

                    sheet.write(row, 0, node.name)
                    sheet.write(row, 1, str(node.topo_id))
                    sheet.write(row, 2, str(node.node_type))
                    sheet.write(row, 3, str(node.shape))

                    sim_node = node_pair[cmp_graph_idx]
                    similarity = node_pair.similarity

                    sheet.write(row, 4, sim_node.name)
                    sheet.write(row, 5, str(sim_node.topo_id))
                    sheet.write(row, 6, str(sim_node.node_type))
                    sheet.write(row, 7, str(sim_node.shape))
                    sheet.write(row, 8, '{:.5f}'.format(similarity))
                    if sim_node is mapped_node:
                        sheet.write(row, 9, 'mapped')
                    row += 1
            else:
                sheet.write(row, 0, node.name)
                sheet.write(row, 1, str(node.topo_id))
                sheet.write(row, 2, str(node.node_type))
                sheet.write(row, 3, str(node.shape))
                sheet.write(row, 4, 'no similar')
                row += 1
            row += 1

    def _xlsx_save_dump_list(self, sheet, graph_idx):
        """Save the dump list."""
        sheet.write(0, 0, 'Node', self._xlsx_bold)
        sheet.write(0, 1, 'Topo Id', self._xlsx_bold)
        sheet.write(0, 2, 'Type', self._xlsx_bold)
        sheet.write(0, 3, 'Shape', self._xlsx_bold)
        sheet.write(0, 4, 'Dump', self._xlsx_bold)
        sheet.write(0, 5, 'Index', self._xlsx_bold)
        sheet.write(0, 6, 'Shape', self._xlsx_bold)
        sheet.write(0, 7, 'DType', self._xlsx_bold)
        sheet.write(0, 8, 'Format', self._xlsx_bold)

        dump_mapper = self._dump_mappers[graph_idx]
        nodes = list(self._graphs[graph_idx].nodes)
        nodes.sort(key=lambda x: x.internal_id)

        row = 1
        for node in nodes:
            # Exclude Parameter and TupleGetItem in the table(start)
            if node.node_type in ['Parameter', 'TupleGetItem']:
                continue
            # (end)
            dump_recs = dump_mapper.output_map.get(node.internal_id, None)
            if dump_recs:
                for rec in dump_recs:
                    sheet.write(row, 0, node.name)
                    sheet.write(row, 1, str(node.topo_id))
                    sheet.write(row, 2, str(node.node_type))
                    sheet.write(row, 3, str(node.shape))
                    self._xlsx_write_not_none(sheet, row, 4, rec.filename)
                    self._xlsx_write_not_none(sheet, row, 5, rec.index)
                    self._xlsx_write_not_none(sheet, row, 6, rec.shape)
                    dtype_str = None if rec.dtype is None else rec.dtype.__name__
                    self._xlsx_write_not_none(sheet, row, 7, dtype_str)
                    self._xlsx_write_not_none(sheet, row, 8, rec.data_format)
                    row += 1
            else:
                sheet.write(row, 0, node.name)
                sheet.write(row, 1, str(node.topo_id))
                sheet.write(row, 2, str(node.node_type))
                sheet.write(row, 3, str(node.shape))
                sheet.write(row, 4, 'missing')
                row += 1

    def _cmp_output_dumps(self):
        """Compare output dumps."""
        self._output_cmp_results = (dict(), dict())
        pair_count = len(self._node_mapper.mapped_node_pairs)
        for i, node_pair in enumerate(self._node_mapper.mapped_node_pairs):
            if node_pair.ambiguous or not node_pair.homogeneous:
                continue
            progress = (i/pair_count) * 100
            print('Comparing output dump files... {:.2f}%'.format(progress), end='\r', flush=True)
            dump_recs0 = self._dump_mappers[0].output_map.get(node_pair[0].internal_id, None)
            dump_recs1 = self._dump_mappers[1].output_map.get(node_pair[1].internal_id, None)
            result = self._cmp_dump_files(dump_recs0, dump_recs1)
            self._output_cmp_results[0][node_pair[0].internal_id] = result
            self._output_cmp_results[1][node_pair[1].internal_id] = result
        print(f'Comparing output dump files... done         ', flush=True)

    def _diagnose(self):
        """Detect problematic nodes."""
        self._problem_node_recs = []
        pair_count = len(self._node_mapper.mapped_node_pairs)
        for i, node_pair in enumerate(self._node_mapper.mapped_node_pairs):
            if node_pair.ambiguous or not node_pair.homogeneous:
                continue
            progress = (i / pair_count) * 100
            print('Diagnosing... {:.2f}%'.format(progress), end='\r', flush=True)
            output_result = self._output_cmp_results[0].get(node_pair[0].internal_id, None)
            if output_result is None or output_result.value_cmp_result in (DUMP_EQUAL, DUMP_EQUAL_SOME_MISSING):
                continue

            input_recs0 = self._dump_mappers[0].input_map.get(node_pair[0].internal_id, None)
            input_recs1 = self._dump_mappers[1].input_map.get(node_pair[1].internal_id, None)
            if input_recs0 is None or input_recs1 is None:
                continue
            input_result = self._cmp_dump_files(input_recs0, input_recs1, fast=True)
            if input_result.value_cmp_result in (DUMP_EQUAL, DUMP_EQUAL_SOME_MISSING):
                self._problem_node_recs.append((node_pair, input_result, output_result))
        self._problem_node_recs.sort(key=lambda x: x[0][0].internal_id)
        print(f'Diagnosing... done         ', flush=True)
        print(f'{len(self._problem_node_recs)} problematic node(s) detected.')

    def _gather_stats(self):
        """Gather the statistics."""
        self._node_type_stats = (dict(), dict())
        empty_stat = {'node_count': 0, 'mapped': 0, 'unmapped': 0, 'dumped': 0, 'dump_equal': 0}
        self._graph_stats = (empty_stat.copy(), empty_stat.copy())
        for i in (0, 1):
            node_type_stat = self._node_type_stats[i]
            for node in self._graphs[i].nodes:
                # Exclude Parameter and TupleGetItem in the table(start)
                if node.node_type in ['Parameter', 'TupleGetItem']:
                    continue
                # (end)
                stat = node_type_stat.get(node.node_type, None)
                if stat is None:
                    stat = empty_stat.copy()
                    node_type_stat[node.node_type] = stat
                stat['node_count'] += 1
                if self._node_mapper.map_strict(i, node.internal_id) is not None:
                    stat['mapped'] += 1

                if self._dump_mappers is not None:
                    if self._dump_mappers[i].output_map.get(node.internal_id, None) is not None:
                        stat['dumped'] += 1

                if self._output_cmp_results is not None:
                    cmp_result = self._output_cmp_results[i].get(node.internal_id, None)
                    if cmp_result is not None and cmp_result.value_cmp_result in (DUMP_EQUAL, DUMP_EQUAL_SOME_MISSING):
                        stat['dump_equal'] += 1

            for stat in node_type_stat.values():
                stat['unmapped'] = stat['node_count'] - stat['mapped']
                self._graph_stats[i]['node_count'] += stat['node_count']
                self._graph_stats[i]['mapped'] += stat['mapped']
                self._graph_stats[i]['unmapped'] = self._graph_stats[i]['node_count'] - self._graph_stats[i]['mapped']
                self._graph_stats[i]['dumped'] += stat['dumped']
                self._graph_stats[i]['dump_equal'] += stat['dump_equal']

    def _cmp_dump_files(self, dump_recs0, dump_recs1, fast=False):
        """Compare the dump files."""
        if not dump_recs0 and not dump_recs1:
            return DumpCmpResult(DUMP_MISSING)

        indices0 = [-1] if dump_recs0 is None else [rec.index for rec in dump_recs0]
        indices1 = [-1] if dump_recs1 is None else [rec.index for rec in dump_recs1]

        max_index = max(max(indices0), max(indices1))
        if max_index < 0:
            print('Error: max_idx < 0 !')
            return DumpCmpResult(DUMP_CMP_ERROR)

        results = []
        for index in range(max_index+1):
            rec0 = None
            rec1 = None
            if dump_recs0:
                for rec in dump_recs0:
                    if rec.index == index:
                        rec0 = rec
                        break
            if dump_recs1:
                for rec in dump_recs1:
                    if rec.index == index:
                        rec1 = rec
                        break

            results.append(self._cmp_dump_file(rec0, rec1, index, fast))

        if not results:
            return DumpCmpResult(DUMP_MISSING)

        result = self._combine_dump_cmp_results(results, fast)
        if result.value_cmp_result == DUMP_EQUAL and \
                len(results) < max(len(indices0), len(indices1)):
            result.value_cmp_result = DUMP_EQUAL_SOME_MISSING
            result.value_cmp_detail = f'graph0: {indices0} output(s), graph1: {indices1} output(s)'
        return result

    def _cmp_dump_file(self, rec0, rec1, index, fast=False):
        """Compare dump 2 files."""
        result = DumpCmpResult('', index)

        if rec0 is not None and rec1 is not None:
            if rec0.data_format is not None and rec1.data_format is not None and\
                    rec0.data_format != rec1.data_format:
                result.value_cmp_result = DUMP_FORMAT_NOT_MATCH

            if rec0.shape is not None and rec1.shape is not None and\
                    rec0.shape != rec1.shape:
                if result.value_cmp_result == DUMP_FORMAT_NOT_MATCH:
                    result.value_cmp_result = DUMP_SHAPE_FORMAT_NOT_MATCH
                else:
                    result.value_cmp_result = DUMP_SHAPE_NOT_MATCH

        dump0, dump1 = self._load_dump_data(rec0, rec1, result)

        if dump0 is not None and dump1 is not None:
            if dump0.size > 0 and dump0.size == dump1.size:
                self._cmp_dump_data(dump0, dump1, result, fast)

        return result

    @staticmethod
    def _load_dump_data(rec0, rec1, result):
        """load dump files."""
        recs = (rec0, rec1)
        dumps = [None, None]

        for i in (0, 1):
            rec = recs[i]
            dump = None
            if rec is not None:
                try:
                    dump = rec.load()
                except FileNotFoundError:
                    result.value_cmp_result = DUMP_MISSING
                except IOError:
                    result.value_cmp_result = DUMP_CMP_ERROR
            else:
                result.value_cmp_result = DUMP_MISSING

            if dump is not None:
                if dump.dtype == np.bool_:
                    dump = dump.astype(np.float, copy=False)
                if len(dump.shape) != 1:
                    dump = dump.flatten()
                if dump.size > 0:
                    result.min_value[i] = dump.min()
                    result.max_value[i] = dump.max()
                dumps[i] = dump

        return dumps[0], dumps[1]

    def _cmp_dump_data(self, dump0, dump1, result, fast=False):
        """Compare 2 dump data."""
        result.value_count = dump0.size
        if not fast:
            abs_diff = np.abs(dump0 - dump1)
            result.abs_diff_max = abs_diff.max()
            result.abs_diff_sum = abs_diff.sum()
            result.abs_diff_avg = abs_diff.mean()

        if result.abs_diff_max is not None:
            has_nan = np.isnan(result.abs_diff_max)
        else:
            has_nan = np.any(np.isnan(dump0))
            if not has_nan:
                has_nan = np.any(np.isnan(dump1))

        if has_nan:
            result.value_cmp_result = DUMP_CMP_ERROR
            result.value_cmp_detail = 'contains NaN'
        else:
            dcmp_result, dcmp_measure, dcmp_detail = self._dump_comparator.compare(dump0, dump1)
            if dcmp_result == dcmp.Result.EQUAL:
                result.value_cmp_result = DUMP_EQUAL
            elif result.value_cmp_result == '':
                if dcmp_result == dcmp.Result.NOT_EQUAL:
                    result.value_cmp_result = DUMP_NOT_EQUAL
                elif dcmp_result == dcmp.Result.SIZE_NOT_MATCH:
                    result.value_cmp_result = DUMP_SIZE_NOT_MATCH
                else:
                    result.value_cmp_result = DUMP_CMP_ERROR
            result.value_cmp_measure = dcmp_measure
            result.value_cmp_detail = dcmp_detail

    @staticmethod
    def _combine_dump_cmp_results(results, fast=False):
        """Combine the dump compare results to a single overall result."""
        combined = copy.deepcopy(results[0])
        combined.index = -1
        combined.breakdowns = results
        for result in results[1:]:
            if result.value_cmp_result not in (DUMP_EQUAL, DUMP_MISSING) and\
                    combined.value_cmp_result != DUMP_CMP_ERROR:
                combined.value_cmp_result = result.value_cmp_result
                combined.value_cmp_measure = result.value_cmp_measure
                combined.value_cmp_detail = result.value_cmp_detail

            if fast:
                continue

            GraphComparator._process_result(combined, result)

        if not fast:
            if combined.abs_diff_sum is not None and combined.value_count > 0:
                combined.abs_diff_avg = combined.abs_diff_sum / combined.value_count

        return combined

    @staticmethod
    def _process_result(combined, result):
        """Process result."""
        combined.value_count += result.value_count
        for i in (0, 1):
            if combined.min_value[i] is not None and result.min_value[i] is not None and \
                    result.min_value[i] < combined.min_value[i]:
                combined.min_value[i] = result.min_value[i]
            if combined.max_value[i] is not None and result.max_value[i] is not None and \
                    result.max_value[i] > combined.max_value[i]:
                combined.max_value[i] = result.max_value[i]
        if combined.abs_diff_sum is not None and result.abs_diff_sum:
            combined.abs_diff_sum += result.abs_diff_sum
        if combined.abs_diff_max is not None and result.abs_diff_max is not None and \
                result.abs_diff_max > combined.abs_diff_max:
            combined.abs_diff_sum = result.abs_diff_max
