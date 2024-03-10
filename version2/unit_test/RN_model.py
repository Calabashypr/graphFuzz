import math
import random
from version2.graph import Graph, Node
import json
from version2.utils.shape_calculator import *


# s = input().split()
# nx = int(s[0])
# px = float(s[1])
# kx = int(s[2])

class GraphRN:
    def __init__(self, n, p, k):
        self.n = n
        self.p = p
        self.k = k
        self.fro = [[] for _ in range(2*n)]
        self.to = [[] for _ in range(2*n)]
        self.op = [dict() for _ in range(2*n)]


    def json_to_graph(self, model_info):
        output_data = model_info
        g = output_data['network']
        graph = Graph([i for i in range(len(g))])
        for node_ in g:
            node = Node(node_['id'])
            node.from_nodes = node_['from']
            node.to_nodes = node_['to']
            node.str_op = node_['name']
            node.params = node_['params']
            node.state = node_['state']
            node.in_degree = len(node.from_nodes)
            node.out_degree = len(node.to_nodes)
            if node.state == 'des':
                node.out_degree = 1
            graph.nodes[node.id] = node
        return graph

    def generate_graph(self):
        n, p, k = self.n, self.p, self.k
        for i in range(1, n):
            for time in range(k):
                for j in range(i + 1, n):
                    if len(self.fro[j]) == k or len(self.to[i]) == k: continue
                    p_now = random.uniform(0, 1)
                    if p_now < p and (j not in self.to[i]):
                        self.fro[j].append(i)
                        self.to[i].append(j)
        des_node = []
        for i in range(1, n):
            if len(self.fro[i]) == 0:
                self.fro[i].append(0)
                self.to[0].append(i)
            if len((self.to[i])) == 0:
                des_node.append(i)
        if len(des_node) == 1:
            return
        add_op = -1

        while len(des_node):
            self.n += 1
            node_left = add_op if add_op!=-1 else des_node.pop()
            node_right = des_node.pop()
            new_add_op = self.n-1
            self.to[node_left].append(new_add_op)
            self.to[node_right].append(new_add_op)
            self.fro[new_add_op] = [node_left, node_right]
            add_op = new_add_op

    def generate_Conv2d_param(self):
        in_channels = 2 ** random.randint(2, 10)
        out_channels = 2 ** random.randint(int(math.log2(in_channels)), 10)
        kernel_size = random.randint(1, 3)
        if kernel_size % 2 == 0: kernel_size -= 1
        stride = random.randint(1, 2)
        padding = random.randint(0, 1)
        params = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding
        }
        # return params
        return params

    def generate_MaxPool2d_param(self):
        kernel_size = random.randint(1, 11)
        if kernel_size % 2 == 0: kernel_size -= 1
        stride = random.randint(1, 2)
        padding = random.randint(0, 1)
        dilation = random.choice([1, 2, 3])
        ceil_mode = False
        params = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "ceil_mode": ceil_mode
        }
        # return params
        return params

    def generate_ZeroPadding2D_param(self):
        padding = random.randint(0, 1),
        data_format = random.choice(['channels_first', 'channels_last'])
        params = {
            "padding": padding,
            "data_format": data_format
        }
        return params

    def generate_linear_param(self):
        params = {}
        return params

    def generate_Add_params(self):
        params = {}
        return params

    def generate_RELU_param(self):
        params = {}
        return params

    def generate_op(self, op_id):
        # op_set = ["Conv2d", "MaxPool2d", "ZeroPadding2D", "linear", "Add", "RELU"]
        op_set = ["Conv2d", "MaxPool2d"]
        op_type = op_set[random.randint(0, 0)]
        params = {}
        if op_type == "Conv2d":
            params = self.generate_Conv2d_param()
        elif op_type == "MaxPool2d":
            params = self.generate_MaxPool2d_param()
        elif op_type == "ZeroPadding2D":
            params = self.generate_ZeroPadding2D_param()
        elif op_type == "linear":
            params = self.generate_linear_param()
        elif op_type == "Add":
            params = self.generate_Add_params()
        elif op_type == 'RELU':
            params = self.generate_RELU_param()
        state = "none"
        from_node = self.fro[op_id]
        to_node = self.to[op_id]
        if len(from_node) == 0:
            state = "src"
        elif len(to_node) == 0:
            state = "des"
        # rande = random.randint(0, 1)
        # if rande == 1:
        #     params = {}
        if len(from_node) > 1:
            op_type="Add"
            params = self.generate_Add_params()
        op_info = {
            "id": op_id,
            "name": op_type,
            "params": params,
            "state": state,
            "from": from_node,
            "to": to_node
        }
        return op_info

    def generate_model(self):
        self.generate_graph()
        model_info = {
            "name": "RN_model",
            "input_shape": [],
            "network": []
        }
        n = self.n
        network = []
        for i in range(n):
            network.append(self.generate_op(i))
        model_info["network"] = network
        return model_info


input_shape = [1, 3, 32, 32]
n, p, k = 10, 0.3, 2
# while True:
#     has_exception = False
#     try:
#         g = GraphX(n, p, k)
#         model_info = g.generate_model()
#         graph = g.json_to_graph(model_info)
#         shape_calculator = ShapeCalculator()
#         shape_calculator.set_shape(graph, input_shape=input_shape)
#     except Exception as e:
#         continue
#     for node_id in range(n):
#         try:
#             graph.fix_shape(node_id)
#         except Exception as e:
#             print(f"Exception caught for node_id {node_id}: {e}")
#             has_exception = True
#             break
#
#     if not has_exception:
#         break
# g = GraphRN(15, 0.7, 2)
# model_info = g.generate_model()
# print(model_info)
# print(g.n)
# file_path = r"C:\Users\Lenovo\Desktop\model_info.json"
# with open(file_path, 'w') as json_file:
#     json.dump(model_info, json_file, indent=4)
