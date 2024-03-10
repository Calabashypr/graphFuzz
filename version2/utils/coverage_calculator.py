import json

from version2.graph import Graph, Node
from version2.model_gen.OperatorSet import operator_set


class CoverageCalculator:
    def __init__(self):
        self.all_op_config = {}

    def op_type_cover(self, sub_g: Graph):  # 算子类型覆盖率(OTC)
        sub_s = set()
        for node in sub_g.nodes.values():
            sub_s.add(node.str_op)
        return len(sub_s) / len(operator_set)

    def op_num_cover(self, sub_g: Graph, g: Graph):  # 算子个数覆盖率（即点覆盖率）
        return len(sub_g) / len(g)

    def input_degree_cover(self, sub_g: Graph, g: Graph):  # (IDC)
        set_g, set_s = dict(), dict()
        op_set = set()
        for node in sub_g.nodes.values():
            op_type = node.str_op
            if op_type not in set_s:
                set_s[op_type] = set()
            set_s[op_type].add(len(node.from_nodes))

        for node in g.nodes.values():
            op_type = node.str_op
            op_set.add(op_type)
            if op_type not in set_g:
                set_g[op_type] = set()
            set_g[op_type].add(len(node.from_nodes))

        IDC_op = 0
        for op_type in set_g:
            if op_type not in set_s:
                continue
            IDC_op += len(set_s[op_type]) / len(set_g[op_type])
        return IDC_op / len(op_set)

    def output_degree_cover(self, sub_g: Graph, g: Graph):  # (ODC)
        set_g, set_s = dict(), dict()
        op_set = set()
        for node in sub_g.nodes.values():
            op_type = node.str_op
            if op_type not in set_s:
                set_s[op_type] = set()
            set_s[op_type].add(len(node.to_nodes))

        for node in g.nodes.values():
            op_type = node.str_op
            op_set.add(op_type)
            if op_type not in set_g:
                set_g[op_type] = set()
            set_g[op_type].add(len(node.to_nodes))

        ODC_op = 0
        for op_type in set_g:
            if op_type not in set_s:
                continue
            ODC_op += len(set_s[op_type]) / len(set_g[op_type])
        return ODC_op / len(op_set)

    def single_edge_cover(self, sub_g: Graph, g: Graph):  # (SEC)
        set_g, set_s = dict(), dict()
        op_set = set()
        for node in sub_g.nodes.values():
            op_type = node.str_op
            if op_type not in set_s:
                set_s[op_type] = set()
            for to_node_id in node.to_nodes:
                to_node = sub_g.nodes[to_node_id]
                if to_node.str_op == op_type:
                    continue
                set_s[op_type].add((node.id, to_node_id))

        for node in g.nodes.values():
            op_type = node.str_op
            op_set.add(op_type)
            if op_type not in set_g:
                set_g[op_type] = set()
            for to_node_id in node.to_nodes:
                to_node = g.nodes[to_node_id]
                if to_node.str_op == op_type:
                    continue
                set_g[op_type].add((node.id, to_node_id))

        SEC = 0
        for op_type in set_g:
            if op_type not in set_s or len(set_g[op_type]) == 0:
                continue
            SEC += len(set_s[op_type]) / len(set_g[op_type])
        return SEC / len(op_set)

    def shape_param_cover(self, sub_g: Graph):  # (SPC)
        op_set = set()
        sub_op_info = {}
        for node in sub_g.nodes.values():
            op_type = node.str_op
            op_set.add(op_type)
            if op_type not in self.all_op_config:
                self.all_op_config[op_type] = set()
            if op_type not in sub_op_info:
                sub_op_info[op_type] = set()
            shape_param = str(node.input_shape) + str(node.params)
            if shape_param not in self.all_op_config[op_type]:
                self.all_op_config[op_type].add(shape_param)
            if shape_param not in sub_op_info[op_type]:
                sub_op_info[op_type].add(shape_param)

        SPC = 0
        for op_type in op_set:
            SPC += len(sub_op_info[op_type]) / len(self.all_op_config[op_type])

        return SPC / len(op_set)

    def edge_cover(self, sub_g: Graph, g: Graph):  # 边覆盖率
        sub_edges = 0
        edges = 0
        for node in sub_g.nodes.values():
            sub_edges += len(node.to_nodes)
        for node in g.nodes.values():
            edges += len(node.to_nodes)
        return sub_edges / edges

    def op_level_cover(self, sub_g: Graph, g: Graph, w1, w2, w3, w4, w5):
        return (w1 * self.op_type_cover(sub_g) + w2 * self.input_degree_cover(sub_g, g) + w3 * self.output_degree_cover(
            sub_g, g) + w4 * self.single_edge_cover(sub_g, g) + w5 * self.shape_param_cover(sub_g))

    def get_cover(self, sub_g: Graph, g: Graph):  # 取平均值
        return (self.op_type_cover(sub_g=sub_g) + self.op_num_cover(sub_g=sub_g, g=g) + self.edge_cover(sub_g=sub_g,
                                                                                                        g=g)) / 3
