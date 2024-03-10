import math
import random
from version2.graph import Graph, Node
import json
from version2.utils.shape_calculator import *
from collections import deque


# s = input().split()
# nx = int(s[0])
# px = float(s[1])
# kx = int(s[2])

class GraphWS:
    def __init__(self):
        self.fro = [[] for _ in range(4 * 20)]
        self.to = [[] for _ in range(4 * 20)]
        self.op = [dict() for _ in range(4 * 20)]
        self.node_num = 0
        self.edge = []
        for i in range(0, 20):
            self.edge.append((i, (i + 1) % 20))
            if i == 0 or i == 6 or i == 10:
                continue
            self.edge.append((i, (i + 2) % 20))
        self.edge.append((0, 5))
        self.edge.append((6, 11))
        self.edge.append((5, 13))

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

    def check_topo(self):
        in_degree, vis = [[0 for i in range(20)] for j in range(2)]
        que = deque()
        for i in range(20):
            in_degree[i] = len(self.fro[i])
            if len(self.fro[i]) == 0:
                que.append(i)
                vis[i] = 1
        while len(que):
            cur_node = que.popleft()
            for nxt_node in self.to[cur_node]:
                in_degree[nxt_node] -= 1
                if in_degree[nxt_node] == 0 and vis[nxt_node] == 0:
                    que.append(nxt_node)
                    vis[nxt_node] = 1
        return sum(in_degree) == 0

    def generate_graph(self):
        while True:
            self.fro = [[] for _ in range(4 * 20)]
            self.to = [[] for _ in range(4 * 20)]
            for (from_node, to_node) in self.edge:
                direct = random.randint(0, 1)
                if direct == 0:
                    self.to[from_node].append(to_node)
                    self.fro[to_node].append(from_node)
                else:
                    self.to[to_node].append(from_node)
                    self.fro[from_node].append(to_node)
            if self.check_topo():
                self.fix_graph()
                return

    def fix_graph(self):
        cur_node = 19
        for i in range(20):
            if len(self.fro[i]) > 2:
                pre_node = list(self.fro[i])
                self.fro[i] = []
                add_op = -1
                while len(pre_node):
                    cur_node += 1
                    if add_op != -1:
                        node_left = add_op
                    else:
                        node_left = int(pre_node.pop())
                        for j in range(len(self.to[node_left])):
                            if self.to[node_left][j] == i:
                                self.to[node_left].pop(j)
                                break

                    node_right = int(pre_node.pop())
                    new_add_op = cur_node
                    self.to[node_left].append(new_add_op)
                    self.to[node_right].append(new_add_op)
                    self.fro[new_add_op] = [node_left, node_right]
                    add_op = new_add_op
                self.fro[i].append(add_op)
                self.to[add_op].append(i)

        des_node = []
        cur_node += 1
        src_node_id = cur_node
        for i in range(cur_node):
            if len(self.fro[i]) == 0:
                self.fro[i].append(src_node_id)
                self.to[src_node_id].append(i)
            if len((self.to[i])) == 0:
                des_node.append(i)
        if len(des_node) == 1:
            self.node_num = cur_node + 1
            return

        add_op = -1
        while len(des_node):
            cur_node += 1
            node_left = add_op if add_op != -1 else des_node.pop()
            node_right = des_node.pop()
            new_add_op = cur_node
            self.to[node_left].append(new_add_op)
            self.to[node_right].append(new_add_op)
            self.fro[new_add_op] = [node_left, node_right]
            add_op = new_add_op

        self.node_num = cur_node + 1

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
        kernel_size = random.randint(1, 3)
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
            op_type = "Add"
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
            "name": "WS_model",
            "input_shape": [],
            "network": []
        }
        n = self.node_num
        network = []
        for i in range(n):
            network.append(self.generate_op(i))
        model_info["network"] = network
        return model_info


g = GraphWS()
model_info = g.generate_model()
v = [0] * g.node_num
for i in range(g.node_num):
    v[i] = 1
    for j in g.to[i]:
        print(i, j)
print(g.node_num)
print(sum(v))
print(model_info)
