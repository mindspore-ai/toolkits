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
"""Cross graph node to node mapper."""
import copy

import numpy as np

from mindinsightx.nv import constants
from mindinsightx.nv.graph.graph import Node


class _DummyNode(Node):
    """Dummy node."""
    def __init__(self):
        super().__init__('', '', '')

    @property
    def topo_id(self):
        return None


class NodePair:
    """Node pair."""
    def __init__(self, node0, node1, match_idx, similarity, homogeneous, ambiguous):
        self.nodes = (node0, node1)
        self.match_idx = match_idx
        self.similarity = similarity
        self.homogeneous = homogeneous
        self.ambiguous = ambiguous

    def __getitem__(self, idx):
        return self.nodes[idx]


class NodeMapper:
    """
    Cross graph node to node mapper.

    Args:
        graph0 (Graph): Graph of index 0.
        graph1 (Graph): Graph of index 1.
        ignore_shape (bool): ignore output shape when mapping graph nodes. None means auto detected.
        max_walk_dist (int): Maximum walk distance for generating footprint matrix.
        dtype (numpy.dtype): Data type of matrices.

    Examples:
        >>> from mindinsightx.nv import constants
        >>> from mindinsightx.nv.graph.graph import Graph
        >>> from mindinsightx.nv.mapper.node_mapper import NodeMapper
        >>>
        >>> me_graph0 = Graph.create('me', '2.0.0')
        >>> me_graph0.load('me_graph0.pb')
        >>> me_graph1 = Graph.create('me', '2.1.0')
        >>> me_graph1.load('me_graph1.pb')
        >>> mapper = NodeMapper(me_graph0, me_graph1)
        >>> mapper.process()
        >>> me_node = me_graph0.nodes[0]
        >>> pair = mapper.map(0, me_node.internal_id)
        >>> if pair:
        >>>     print(f'mapped node: {pair[1].name} similarity: {pair.similarity}')
        >>> top_k = mapper.top_k(0, me_node.internal_id, k=3)
        >>> if top_k:
        >>>     print(f'top-k nodes: {[pair[1].name for pair in top_k]}')
    """

    # clipped similarity range for avoiding infinity values or divides by zero when computing objective function
    _CLIPPED_SIM_MIN = 0.001
    _CLIPPED_SIM_MAX = 0.99
    _SIM_EPSILON = 1e-3

    def __init__(self, graph0, graph1, ignore_shape=None, max_walk_dist=15, dtype=np.float64, force=False):
        mappable = self.is_mappable(graph0, graph1)
        if not mappable:
            if force:
                print("Graphs may not be mappable, but will force to do it.")
            else:
                raise ValueError(f"Graphs are not mappable.")

        self._graphs = (graph0, graph1)
        if ignore_shape is None:
            self._ignore_shape = self._infer_ignore_shape(graph0.graph_type, graph1.graph_type, mappable)
        else:
            self._ignore_shape = ignore_shape
        self._max_walk_dist = max_walk_dist
        self._dtype = dtype

        # intermediates
        self._mixed_part = None
        self._homo_parts = None
        self._fp_mat_dim = 0
        self._fp_mat0 = None
        self._feat_mat0_t = None
        self._feat_mat1 = None

        # results
        self._sim_matrix = None
        self._mapped_node_pairs = None
        self._node_id_map_recs = None
        self._matched_node_infos = None
        self._objectives = []

    @property
    def graphs(self):
        """Graph list in form of [graph0, graph1]."""
        return self._graphs

    @property
    def sim_matrix(self):
        """Similarity matrix."""
        return self._sim_matrix

    @property
    def mapped_node_pairs(self):
        """One to one mapped node pair, mapped_node_pairs[0] responds to graph0 node."""
        return self._mapped_node_pairs

    @property
    def objectives(self):
        """
        Objective function's historical values over the optimization process.

        Notes:
            objectives[0] is value before step 0, objectives[1] is value after step 1 an so on.
        """
        return self._objectives

    @property
    def objective(self):
        """The final objective function value."""
        return self._objectives[-1] if self._objectives else 0

    @property
    def step_count(self):
        """Number of optimization step has been taken."""
        return max(0, len(self._objectives) - 1)

    @staticmethod
    def is_mappable(graph0, graph1):
        """Check if the graphs' node types are mappable."""
        return True

    @property
    def processed(self):
        """Check if the optimization process was conducted."""
        return self._sim_matrix is not None

    def process(self, auto_stop=True, max_steps=None):
        """
        Conduct the optimization process of node mapping.

        Args:
            auto_stop (bool): Automatically stops when the objective function is maximized.
            max_steps (int, Optional): Number of maximum optimization steps to be taken.
        """
        if self.processed:
            raise RuntimeError('Function process() was invoked already.')

        if not auto_stop and max_steps is None:
            raise ValueError('Argument auto_stop and max_steps cannot be False and None at the same time.')

        if max_steps is not None and max_steps <= 0:
            raise ValueError(f'Invalid max_steps: {max_steps}, positive integer is expected.')

        self._reset_nodes()

        self._graphs[0].prepare_node_mapping(self._graphs[1], self._ignore_shape)
        self._graphs[1].prepare_node_mapping(self._graphs[0], self._ignore_shape)

        self._create_parts()
        self._create_fp_mats()
        objective, total_sim = self._update_match_sim()
        self._objectives.append(objective)
        print(f'initial total_sim:{total_sim} objective:{objective}')

        step = 0
        last_objective = -1
        non_increase_steps = 0
        while True:
            if max_steps is not None and step >= max_steps:
                print(f'Maximum {max_steps} steps reached.')
                break

            mapping, changed = self._priority_match()
            if changed == 0:
                print('No more change in footprint matrix, stopped.')
                break
            self._update_fp_mat(mapping)
            objective, total_sim = self._update_match_sim()
            self._objectives.append(objective)

            print(f'step#{step} total_sim:{total_sim} objective:{objective}')

            if auto_stop and objective <= last_objective:
                non_increase_steps += 1
                if non_increase_steps >= 2:
                    print('Objective maximized, stopped.')
                    break

            last_objective = objective
            step += 1

        self._gather_results()
        self._cleanup()

    @staticmethod
    def _infer_ignore_shape(graph_type0, graph_type1, mappable):
        if not mappable:
            graph_types = sorted((graph_type0, graph_type1))
            if graph_types[0] == constants.ME_GRAPH_TYPE and graph_types[1] == constants.TF_GRAPH_TYPE:
                return True
        return False

    def _detect_ambiguous(self, nodes, match_idx):
        """Detect if the pair is mapped ambiguously."""
        similarity = self._sim_matrix[match_idx, match_idx]
        threshold = similarity - self._SIM_EPSILON
        match_idx_min = nodes[0].partition[0][0].match_idx
        part_size = len(nodes[0].partition[0])
        for i in (0, 1):
            if i == 0:
                sims = self._sim_matrix[match_idx, match_idx_min:match_idx_min + part_size]
            else:
                sims = self._sim_matrix[match_idx_min:match_idx_min + part_size, match_idx]
            # compare the other similarities in the partition with the threshold
            # don't wanna make an array copy here, therefore, backup and restore the original value
            offset = match_idx - match_idx_min
            sim_bak = sims[offset]
            sims[offset] = -1
            max_sim = sims.max()
            sims[offset] = sim_bak
            if max_sim >= threshold:
                return True
        return False

    def _gather_results(self):
        """Gather results from process()."""
        node_pairs = [[None, None] for _ in range(self._fp_mat_dim)]

        self._matched_node_infos = ([None] * self._fp_mat_dim, [None] * self._fp_mat_dim)
        for i in (0, 1):
            for node in self._graphs[i].nodes:
                node_pairs[node.match_idx][i] = node
                # node.label may be changed by other node mapper later, we keep it in node info
                self._matched_node_infos[i][node.match_idx] = (node, node.label)

        self._node_id_map_recs = [dict(), dict()]
        self._mapped_node_pairs = []
        for match_idx, pair in enumerate(node_pairs):

            if all(pair):
                match_idx = match_idx
                similarity = self._sim_matrix[match_idx, match_idx]
                ambiguous = self._detect_ambiguous(pair, match_idx)
                homogeneous = pair[0].label == pair[1].label
                pair_idx = len(self._mapped_node_pairs)
                self._mapped_node_pairs.append(NodePair(pair[0], pair[1],
                                                        match_idx,
                                                        similarity,
                                                        homogeneous,
                                                        ambiguous))
            else:
                pair_idx = -1

            for i in (0, 1):
                if pair[i] is not None:
                    self._node_id_map_recs[i][pair[i].internal_id] = (match_idx, pair_idx)

    def _cleanup(self):
        """Release memory for process()."""
        self._reset_nodes()
        self._mixed_part = None
        self._homo_parts = None
        self._fp_mat0 = None
        self._feat_mat0_t = None
        self._feat_mat1 = None

    def get_match_idx(self, graph_idx, internal_id):
        """
        Get the match index of a node. Match index is the row(or column if graph_idx is 1) index in
        the similarity matrix.

        Args:
            graph_idx (int): Graph index, either 0 or 1.
            internal_id (str): Node id.

        Returns:
            int, the match index.
        """
        if graph_idx not in (0, 1):
            raise ValueError(f'Invalid graph_idx: {graph_idx}, must be 0 or 1.')

        if not self.processed:
            raise RuntimeError('Function process() not yet invoked.')

        map_rec = self._node_id_map_recs[graph_idx].get(internal_id, None)
        if map_rec is None:
            raise ValueError(f"Node:{internal_id} does not exists in graph#{graph_idx}'s record.")

        return map_rec[0]

    def map_strict(self, graph_idx, internal_id, ret_info='pair'):
        """
        Get the one to one mapped node that is homogeneous and non-ambiguous.

        Notes:
            Equivalent to map() with homogeneous=True and non_ambiguous=True.

        Args:
            graph_idx (int): Query graph index, either 0 or 1.
            internal_id (int): Query node id.
            ret_info (str): Information to be returned, options:
                'name' - Node name (str).
                'node' - Node object.
                'pair' - MappedNodePair object.

        Returns:
            union[str, Node, MappedNodePair, None], The mapped node information.
        """
        return self.map(graph_idx=graph_idx,
                        internal_id=internal_id,
                        homogeneous=True,
                        non_ambiguous=True,
                        ret_info=ret_info)

    def map(self, graph_idx, internal_id, homogeneous=False, non_ambiguous=False, ret_info='pair'):
        """
        Get the one to one mapped node.

        Notes:
            Returns None if no such mapping.

        Args:
            graph_idx (int): Query graph index, either 0 or 1.
            internal_id (int): Query node id.
            homogeneous (bool): Returns None if the mapped node is not homogeneous.
            non_ambiguous (bool): Returns None if the mapped node is ambiguous.
            ret_info (str): Information to be returned, options:
                'name' - Node name (str).
                'node' - Node object.
                'pair' - MappedNodePair object.

        Returns:
            union[str, Node, MappedNodePair, None], The mapped node information.
        """
        if graph_idx not in (0, 1):
            raise ValueError(f'Invalid graph_idx: {graph_idx}, must be 0 or 1.')

        if not self.processed:
            raise RuntimeError('Function process() not yet invoked.')

        map_rec = self._node_id_map_recs[graph_idx].get(internal_id, None)
        if map_rec is None:
            raise ValueError(f"Node:{internal_id} does not exists in graph#{graph_idx}'s record.")

        pair_idx = map_rec[1]
        if pair_idx < 0:
            # no 1-1 mapping
            return None

        pair = self._mapped_node_pairs[pair_idx]

        if (homogeneous and not pair.homogeneous) or (non_ambiguous and pair.ambiguous):
            return None

        if ret_info == 'pair':
            return copy.copy(pair)

        resp_graph_idx = 1 - graph_idx
        if ret_info == 'name':
            return pair.nodes[resp_graph_idx].name
        if ret_info == 'node':
            return pair.nodes[resp_graph_idx]

        raise ValueError(f'Unrecognized ret_info: {ret_info}, must be one of "pair", "name" or "node".')

    def top_k(self, graph_idx, internal_id, k, homogeneous=False, ret_info='pair'):
        """
        Get the top-k most topologically similar nodes.

        Notes:
            The length of returned list is less then or equals to k.

        Args:
            graph_idx (int): Query graph index, either 0 or 1.
            internal_id (int): Query node id.
            k (int): Positive integer k.
            homogeneous (bool): Returns homogeneous nodes only.
            ret_info (str): Information of the top-k similar nodes to be returned, options:
                'name' - Node name (str).
                'node' - Node object.
                'pair' - MappedNodePair object.

        Returns:
            list[union[str, Node, MappedNodePair]], Top-k similar nodes.
        """
        if graph_idx not in (0, 1):
            raise ValueError(f'Invalid graph_idx: {graph_idx}, must be 0 or 1.')
        if k <= 0:
            raise ValueError(f'Invalid k: {k}, must be greater then zero.')

        if not self.processed:
            raise RuntimeError('Function process() not yet invoked.')

        map_rec = self._node_id_map_recs[graph_idx].get(internal_id, None)
        if map_rec is None:
            raise ValueError(f"Node:{internal_id} does not exists in graph#{graph_idx}'s records.")

        match_idx = map_rec[0]
        node_info = self._matched_node_infos[graph_idx][match_idx]
        if node_info is None:
            raise ValueError(f"Unexpected node_info is None.")

        if graph_idx == 0:
            similarities = self._sim_matrix[match_idx, :]
        else:
            similarities = self._sim_matrix[:, match_idx]

        cmp_match_list = list(enumerate(similarities))
        cmp_match_list.sort(key=lambda x: x[1], reverse=True)

        cmp_graph_idx = 1 - graph_idx
        top_k_list = []
        for match in cmp_match_list:
            cmp_match_idx = match[0]
            similarity = match[1]
            if similarity <= 0:
                break
            cmp_node_info = self._matched_node_infos[cmp_graph_idx][cmp_match_idx]
            if cmp_node_info is None:
                continue

            if homogeneous:
                # check node label again
                if node_info[1] != cmp_node_info[1]:
                    continue

            sim_node = cmp_node_info[0]

            if ret_info == 'name':
                top_k_list.append(sim_node.name)
            elif ret_info == 'node':
                top_k_list.append(sim_node)
            elif ret_info == 'pair':
                pair = [None, None]
                pair[graph_idx] = node_info[0]
                pair[cmp_graph_idx] = sim_node
                node_pair = NodePair(pair[0], pair[1],
                                     cmp_match_idx,
                                     similarity,
                                     node_info[1] == cmp_node_info[1],
                                     None)
                top_k_list.append(node_pair)
            else:
                raise ValueError(f'Unrecognized ret_info: {ret_info}, must be one of "pair", "name" or "node".')

            if len(top_k_list) >= k:
                break

        return top_k_list

    def _reset_nodes(self):
        """Reset nodes' temporary variables."""
        for graph in self._graphs:
            for node in graph.nodes:
                node.partition = None
                node.footprints = None
                node.match_sim = 0
                node.tmp_match_sim = 0
                node.tmp_matched_node = None

    def _create_parts(self):
        """Create nodes partitions."""
        mixed_part = [[], []]
        d_label_parts = dict()

        self._create_d_label_parts(d_label_parts)

        self._process_d_label_parts(d_label_parts, mixed_part)

        max_len = max(len(mixed_part[0]), len(mixed_part[1]))
        for i in (0, 1):
            dummy_count = max_len - len(mixed_part[i])
            for _ in range(dummy_count):
                mixed_part[i].append(_DummyNode())
            np.random.shuffle(mixed_part[i])

        self._homo_parts = list(d_label_parts.values())
        self._mixed_part = mixed_part
        self._all_parts = list(self._homo_parts)
        self._all_parts.append(self._mixed_part)

        for i in (0, 1):
            match_idx = 0
            for part in self._all_parts:
                for node in part[i]:
                    node.match_idx = match_idx
                    node.partition = part
                    match_idx += 1

        self._fp_mat_dim = 0
        for part in self._all_parts:
            self._fp_mat_dim += len(part[0])

    def _process_d_label_parts(self, d_label_parts, mixed_part):
        """Process d label parts."""
        labels = list(d_label_parts.keys())
        for label in labels:
            part = d_label_parts[label]
            if not part[0] or not part[1]:
                for i in (0, 1):
                    for node in part[i]:
                        mixed_part[i].append(node)
                del d_label_parts[label]

        for part in d_label_parts.values():
            max_len = max(len(part[0]), len(part[1]))
            for i in (0, 1):
                part[i].sort(key=lambda x: x.homo_depth)
                dummy_count = max_len - len(part[i])
                for _ in range(dummy_count):
                    part[i].append(_DummyNode())

    def _create_d_label_parts(self, d_label_parts):
        """Create d label parts."""
        for i in (0, 1):
            for node in self._graphs[i].nodes:
                if not node.label:
                    raise ValueError(
                        f"The graph{i}'s node:{node.name} is not labeled.")
                if node.homo_depth < 0:
                    raise ValueError(
                        f"The graph{i}'s node:{node.name} has no homo-depth.")
                part = d_label_parts.get(node.label, None)
                if part is None:
                    part = [[], []]
                    d_label_parts[node.label] = part
                part[i].append(node)

    def _create_fp_mats(self):
        """Create footprint matrices."""
        fp_mats = [None, None]
        fp_mats[0] = np.zeros((self._fp_mat_dim, self._fp_mat_dim), dtype=self._dtype)
        fp_mats[1] = np.zeros((self._fp_mat_dim, self._fp_mat_dim), dtype=self._dtype)

        for graph, fp_mat in zip(self._graphs, fp_mats):
            for src_node in graph.nodes:
                src_node.footprints = dict()
                for to_node in src_node.to_nodes:
                    self._depth_first_walk(src_node, to_node, 1, fp_mat)
        self._fp_mat0 = fp_mats[0]
        self._feat_mat0_t = np.transpose(self._fp_to_feat(fp_mats[0]))
        self._feat_mat1 = self._fp_to_feat(fp_mats[1])

    def _depth_first_walk(self, src_node, visit_node, dist, fp_mat):
        """Depth first walk for filling in footprint matrix."""
        step_fp = 1 / dist
        fp = fp_mat[visit_node.match_idx, src_node.match_idx]

        if step_fp <= fp:
            return

        fp_mat[visit_node.match_idx, src_node.match_idx] = step_fp
        src_node.footprints[visit_node] = step_fp

        if dist < self._max_walk_dist:
            next_dist = dist + 1
            for to_node in visit_node.to_nodes:
                self._depth_first_walk(src_node, to_node, next_dist, fp_mat)

    def _update_match_sim(self):
        """Update nodes' match similarity and compute objective function."""
        # dot operation can be speedup with CUDA backed numpy like library
        self._sim_matrix = self._feat_mat0_t.dot(self._feat_mat1)

        objective = 0
        total_sim = 0
        for node0 in self._graphs[0].nodes:
            node0.match_sim = self._sim_matrix[node0.match_idx, node0.match_idx]
            total_sim += node0.match_sim
            clipped = np.clip(node0.match_sim, self._CLIPPED_SIM_MIN, self._CLIPPED_SIM_MAX)
            objective -= clipped/np.log(clipped)

        return objective, total_sim

    def _priority_match(self, mixed_only=False):
        """Prioritized node matching."""
        if mixed_only:
            mapping = [-1] * len(self._mixed_part[0])
        else:
            mapping = [-1] * self._fp_mat_dim

        map_pos = 0
        changed = 0
        if not mixed_only:
            for part in self._homo_parts:
                pair_changed, map_pos = self._priority_match_part(part, mapping, map_pos)
                changed += pair_changed

        pair_changed, _ = self._priority_match_part(self._mixed_part, mapping, map_pos)
        changed += pair_changed

        for idx in mapping:
            if idx < 0:
                raise RuntimeError("Invalid mapping index.")

        return mapping, changed

    def _priority_match_part(self, part, mapping, map_pos):
        """Prioritized node matching of a partition."""
        part_size = len(part[0])
        if part_size == 0:
            return 0, map_pos

        for nodes in part:
            for node in nodes:
                node.tmp_matched_node = None
                node.tmp_match_sim = -1

        priority_nodes0 = sorted(part[0],
                                 key=lambda x: x.match_sim,
                                 reverse=False)

        part_min_match_idx = part[1][0].match_idx

        # first pass, overwriting previous matched node is enabled
        for node0 in priority_nodes0:
            if isinstance(node0, _DummyNode):
                continue
            part_sim = self._sim_matrix[node0.match_idx, part_min_match_idx:part_min_match_idx + part_size]
            node1_idx_in_part = np.argmax(part_sim)
            node1_match_idx = node1_idx_in_part + part_min_match_idx
            node1 = part[1][node1_idx_in_part]
            match_sim = self._sim_matrix[node0.match_idx, node1_match_idx]

            if node1.tmp_matched_node is not None:
                if match_sim <= node1.tmp_match_sim:
                    continue
                # overwrite
                node1.tmp_matched_node.tmp_matched_node = None
            node1.tmp_matched_node = node0
            node1.tmp_match_sim = match_sim
            node0.tmp_matched_node = node1
            node0.tmp_match_sim = match_sim

        # second pass, match the unmatched
        priority_nodes0 = reversed(priority_nodes0)
        for node0 in priority_nodes0:
            if node0.tmp_matched_node is not None:
                continue

            max_sim = -1
            max_sim_node1 = None

            for node1 in part[1]:
                if node1.tmp_matched_node is not None:
                    continue

                if isinstance(node0, _DummyNode):
                    max_sim = 0
                    max_sim_node1 = node1
                    break

                sim = self._sim_matrix[node0.match_idx, node1.match_idx]
                if sim > max_sim:
                    max_sim = sim
                    max_sim_node1 = node1

            if max_sim_node1 is None:
                raise RuntimeError("Variable max_sim_node1 is None.")

            max_sim_node1.tmp_matched_node = node0
            max_sim_node1.tmp_match_sim = max_sim
            node0.tmp_matched_node = max_sim_node1
            node0.tmp_match_sim = max_sim

        changed = 0
        for n, node0 in enumerate(part[0]):
            new_match_idx = node0.tmp_matched_node.match_idx
            mapping[map_pos + n] = new_match_idx
            if not isinstance(node0, _DummyNode) and new_match_idx != node0.match_idx:
                changed += 1

        return changed, map_pos + part_size

    def _update_fp_mat(self, mapping, mixed_only=False):
        """Update graph0's footprint matrix with new node mappings."""
        map_pos = 0
        if not mixed_only:
            for part in self._homo_parts:
                map_pos = self._update_part(mapping, map_pos, part)

        self._update_part(mapping, map_pos, self._mixed_part)

        # reconstruct footprint matrix 0
        self._fp_mat0.fill(0)
        for src_node in self._graphs[0].nodes:
            for visited_node, footprint in src_node.footprints.items():
                self._fp_mat0[visited_node.match_idx, src_node.match_idx] = footprint

        self._feat_mat0_t = np.transpose(self._fp_to_feat(self._fp_mat0))

    @staticmethod
    def _update_part(mapping, map_pos, part):
        """Update graph0's footprint matrix of a partition with new node mappings."""
        part_size = len(part[0])
        if part_size == 0:
            return map_pos

        new_part0 = [None] * part_size
        part_min_match_idx = part[1][0].match_idx

        for n, node0 in enumerate(part[0]):
            new_match_idx = mapping[map_pos + n]
            node0.match_idx = new_match_idx
            new_part0[new_match_idx - part_min_match_idx] = node0

        for node0 in new_part0:
            if node0 is None:
                raise RuntimeError("Variable new_part0 has None element.")

        part[0] = new_part0
        return map_pos + part_size

    @classmethod
    def _fp_to_feat(cls, fp_mat):
        """Convert footprint matrix to feature matrix."""
        feat_mat = np.empty((2 * fp_mat.shape[0], fp_mat.shape[1]), dtype=fp_mat.dtype)
        feat_mat[:fp_mat.shape[0], :] = fp_mat
        feat_mat[fp_mat.shape[0]:, :] = np.transpose(fp_mat)

        # normalize non-zero columns
        for col in range(fp_mat.shape[1]):
            norm = np.linalg.norm(feat_mat[:, col])
            if norm > 0:
                feat_mat[:, col] /= norm
        return feat_mat
