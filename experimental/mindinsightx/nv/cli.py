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
"""Commandline module."""
import argparse
import os

try:
    import cli_helper
except ImportError:
    pass

import mindinsightx
from mindinsightx.nv import constants
from mindinsightx.nv.common import viz_utils
from mindinsightx.nv.graph.graph import Graph
from mindinsightx.nv.comparator.graph_comparator import GraphComparator
from mindinsightx.nv.comparator.dump_comparator import DumpComparator


def _infer_type_ver_code(arg_graph_type, has_dump_dir):
    """Infer the type version code."""
    if arg_graph_type in (constants.ME_GRAPH_TYPE, constants.ME_TYPE_VER_X):
        return constants.ME_TYPE_VER_X

    if has_dump_dir:
        # auto append '-opt' to me graph types if dump dir is specified
        if arg_graph_type.startswith(constants.ME_GRAPH_TYPE+'-') and not arg_graph_type.endswith('-opt'):
            return arg_graph_type + '-opt'

    return arg_graph_type


def _get_cli_args():
    """Get commandline arguments."""

    cli_parser = argparse.ArgumentParser(
        prog='mindcompare',
        description='MindCompare CLI entry point (version: {})'.format(mindinsightx.__version__))

    cli_parser.add_argument(
        '--graph-file',
        type=str,
        required=True,
        help='graph0 file(.pb)')

    cli_parser.add_argument(
        '--graph-type',
        type=str,
        required=False,
        default='me-trace_code',
        help='graph0 type(default: me-trace_code)')

    cli_parser.add_argument(
        '--cmp-graph-file',
        type=str,
        required=True,
        help='graph1 file(.pb)')

    cli_parser.add_argument(
        '--cmp-graph-type',
        type=str,
        required=False,
        default='me-trace_code',
        help='graph1 type(default: me-trace_code)')

    cli_parser.add_argument(
        '--dump-dir',
        type=str,
        required=False,
        help='graph0 dump directory of a single iteration')

    cli_parser.add_argument(
        '--cmp-dump-dir',
        type=str,
        required=False,
        help='graph1 dump directory of a single iteration')

    cli_parser.add_argument(
        '--report',
        type=str,
        required=False,
        default='./report.xlsx',
        help='output xlsx report path(default: ./report.xlsx)')

    cli_parser.add_argument(
        '--force',
        action='store_const',
        default=False,
        const=True,
        help='force to compare even the graphs are not mappable')

    cli_parser.add_argument(
        '--no-diagnostic',
        action='store_const',
        default=False,
        const=True,
        help='don\'t detect problematic operators')

    cli_parser.add_argument(
        '--dump-cmp',
        type=str,
        required=False,
        default='AvgAbsError',
        help='dump comparator(default: AvgAbsError)')

    cli_parser.add_argument(
        '--max-steps',
        type=str,
        required=False,
        default='unlimited',
        help='max. no. of node mapper steps(default: unlimited)')

    cli_parser.add_argument(
        '--export-viz',
        action='store_const',
        default=False,
        const=True,
        help='export json graphs for visualization')

    cli_parser.add_argument(
        '--top-k',
        type=int,
        required=False,
        default=5,
        help='no. of most similar nodes to be listed in report(default: 5)')

    return cli_parser.parse_args()


def main():

    try:
        cli_helper.print_info()
    except NameError:
        pass

    args = _get_cli_args()

    if args.dump_dir and not args.cmp_dump_dir:
        raise ValueError('missing --cmp-dump-dir, dump-dir and cmp-dump-dir must be used together')

    if not args.dump_dir and args.cmp_dump_dir:
        raise ValueError('missing --dump-dir, dump-dir and cmp-dump-dir must be used together')

    if args.max_steps != 'unlimited':
        try:
            max_steps = int(args.max_steps)
            if max_steps <= 0:
                raise ValueError
        except ValueError:
            raise ValueError('invalid ---max-steps, must be "unlimited" or positive integer')
    else:
        max_steps = None

    has_dump_dir = bool(args.dump_dir)

    type_ver_code = _infer_type_ver_code(args.graph_type, has_dump_dir)
    graph0 = Graph.create(type_ver_code=type_ver_code)
    graph0.load(args.graph_file)

    type_ver_code = _infer_type_ver_code(args.cmp_graph_type, has_dump_dir)
    graph1 = Graph.create(type_ver_code=type_ver_code)
    graph1.load(args.cmp_graph_file)

    print(f'graph0 node#:{len(graph0.nodes)} edge#:{len(graph0.edges)}')
    print(f'graph1 node#:{len(graph1.nodes)} edge#:{len(graph1.edges)}')

    if has_dump_dir:
        diagnostic = not args.no_diagnostic
    else:
        diagnostic = False

    if has_dump_dir:
        dump_comparator = DumpComparator.from_str(args.dump_cmp)
        if dump_comparator is None:
            raise ValueError('invalid --dump-cmp')
    else:
        dump_comparator = None

    comparator = GraphComparator(graph0, graph1, args.dump_dir, args.cmp_dump_dir,
                                 force=args.force, diagnostic=diagnostic,
                                 dump_comparator=dump_comparator)

    comparator.compare(max_steps=max_steps)

    report_path = os.path.realpath(args.report)
    print(f'Saving report as {report_path}...')

    comparator.save_as_xlsx(report_path, top_k=args.top_k)

    if args.export_viz:
        for i, graph in enumerate(comparator.graphs):
            if graph.graph_type == constants.ME_GRAPH_TYPE:
                json_path = f"{report_path}.graph{i}.data.json"
                viz_utils.save_me_as_json(comparator, i, json_path)
                print(f"{json_path} saved.")

    print('Done')


if __name__ == "__main__":
    main()
