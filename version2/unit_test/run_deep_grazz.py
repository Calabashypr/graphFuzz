import copy

from version2.MCTS.MonteCarloTree import *
from version2.utils.shape_calculator import ShapeCalculator
from version2.utils.coverage_calculator import CoverageCalculator
from version2.Results.result_analyser import ResultAnalyser
from version2.data.mnist_load import MnistDataLoader
from version2.data.rand_data_load import RandomDataLoader
from version2.data.cifar10_load import Cifar10DataLoader
from version2.unit_test.RN_model import GraphRN
from version2.model_gen.TorchModelGenerator import TorchModel
from version2.model_gen.TFModelGenerator import TensorflowModel
from version2.model_gen.MindsporeModelGenerator import MindSporeModel
from concurrent.futures import ThreadPoolExecutor
from version2.Mutate.Mutate import MutationSelector
from version2.MCTS.random_graph_gen import *
from version2.unit_test.WS_model import GraphWS
import pickle
import time
import os


def model_run(model, frame_work, res_dict: dict, data):
    r"""

    :param model:
    :param frame_work:
    :param res_dict:
    :param data:
    :return:
    """
    res_dict[frame_work]['flag_run'] = True
    try:
        print('-' * 10, f'{frame_work}_res', '-' * 10)
        res = model.compute_dag(data)
        res_dict[frame_work]['final_res'] = res[0]
        res_dict[frame_work]['layer_res'] = res[1]
    except BaseException as e:
        res_dict[frame_work]['flag_run'] = False
        res_dict[frame_work]['exception_info'] = e
        print(f'{frame_work} compute failure:{e}')


if __name__ == '__main__':
    shape_calculator = ShapeCalculator()
    cover_cal = CoverageCalculator()
    result_analyser = ResultAnalyser()
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'model_graph')
    model_json = 'ResNet50'
    # model_json = 'vgg16'
    has_mutate = True
    has_MCTS = True
    terminate_condition = 100
    data_set = 'cifar10'
    if has_mutate:
        mutate_flag = 'mutation'
    else:
        mutate_flag = 'noMutation'
    if has_MCTS:
        search_flag = 'mcts'
    else:
        search_flag = 'rand'
    # log_file = f'test_{model_json.lower()}_{mutate_flag}_{data_set.lower()}_0{terminate_condition}.txt'
    log_file = 'result_0025.txt'
    json_dir = os.path.join('..', 'config', model_json, f'{model_json}.json')
    res_dir = os.path.join('..', 'Results', log_file)

    # graph = json_to_graph(json_dir)
    # graph.display()

    print('-' * 50)
    config_dict = {'log_dir': res_dir, 'model_cnt': 0}

    data_set = data_set.lower()
    batch_size = 1
    input_shape = [batch_size, 3, 32, 32]
    data_load = RandomDataLoader(input_shape)
    if data_set == 'mnist':
        data_load = MnistDataLoader()
        input_shape = [batch_size, 1, 28, 28]
    elif data_set == 'cifar10':
        data_load = Cifar10DataLoader()
        input_shape = [batch_size, 3, 32, 32]
    data_gen = data_load.data_gen()

    graph = None
    cnt = 0
    model_info = None
    n_, p_, k_ = 10, 0.8, 2
    seed = "WS"
    g = None

    while True:
        has_exception = False
        try:
            if seed == "RN":
                g = GraphRN(n_, p_, k_)
            elif seed == "WS":
                g = GraphWS()
            model_info = g.generate_model()
            # with open(r"C:\Users\Lenovo\Desktop\RN.json", 'w') as json_file:
            #     json.dump(model_info, json_file, indent=4)
            # print(model_info)
            # print("num:",g.n)
            graph = g.json_to_graph(model_info)
            shape_calculator = ShapeCalculator()
            shape_calculator.set_shape(graph, input_shape=input_shape)
        except Exception as e:
            print("error")
            continue
        if seed == "RN":
            node_num = g.n
        else:
            node_num = g.node_num
        for node_id in range(1, node_num):
            try:
                graph.fix_shape(node_id)
            except Exception as e:
                print(f"Exception caught for node_id {node_id}: {e}")
                has_exception = True
                break
        # d = np.random.randn(1, 3, 32, 32)
        # tf_model = TensorflowModel(graph)
        # tf_model.compute_dag(d)
        try:
            torch_model = TorchModel(graph)
            d = np.random.randn(1, 3, 32, 32)
            torch_model.compute_dag(d)
            print("torch_suc!")
            ms_model = MindSporeModel(graph)
            ms_model.compute_dag(d)
            print("ms_suc!")
            tf_model=TensorflowModel(graph)
            tf_model.compute_dag(d)
            print("tf_suc!")

        except Exception as e:
            continue
        if not has_exception:
            break

    # json_dir_test = r"C:\Users\Lenovo\Desktop\RN.json"
    # graph = json_to_graph(json_dir_test)
    # print(len(graph.nodes))
    # shape_calculator.set_shape(graph, input_shape=input_shape)
    # with open(r"C:\Users\Lenovo\Desktop\RN.json", 'w') as json_file:
    #     json.dump(model_info, json_file, indent=4)

    cover_set = set()
    model_cnt = 0
    crash_model = 0
    model_num = 0
    start_time = time.time()
    lst_g = None
    for i in range(100):
        if model_cnt == 100:
            break
        data = next(data_gen)
        print("iter", i)

        if i == 10 or i == 50 or i == 100:
            result_analyser.statics_bugs(config_dict=config_dict)
        if has_MCTS:
            sub_g = block_chooser(graph, 10)
        else:
            sub_g = random_graph(graph, i % 10 + 1)
        shape_calculator.set_shape(sub_g, input_shape=input_shape)
        # try:
        #     shape_calculator.set_shape(sub_g, input_shape=input_shape)
        # except BaseException as e:
        #     print(f'calculate shape failure')
        #     continue

        nodes_path = [node.id for node in sub_g.nodes.values() if node.id in graph.nodes.keys()]

        print('sub graph:')
        sub_g.display()
        try:
            if has_mutate:
                mutation_sets = [randint(0, 100) for _ in range(min(i, 8))]
                mutation_selector = MutationSelector(mutation_sets, r=1)
                mutation_selector.mutate(sub_g)
                print('after mutate')
                sub_g.display()
        except BaseException as e:
            print('mutation failure')
            continue

        print(f'generate model and compute')
        coverage = cover_cal.op_level_cover(sub_g, graph, .2, .2, .2, .2, .2)

        # torch_model = TorchModel(sub_g)
        # tf_model = TensorflowModel(sub_g)
        # mindspore_model = MindSporeModel(sub_g)
        model_dict = {}
        try:
            model_dict = {'torch': TorchModel(sub_g), 'tensorflow': TensorflowModel(sub_g),
                          'mindspore': MindSporeModel(sub_g)}
        except BaseException as e:
            print(f'model generate failure in iter:{i}')
            continue

        print(f'model_cnt:{model_cnt}')
        if round(coverage, 3) in cover_set:
            continue
        cover_set.add(round(coverage, 3))
        torch_res = {}
        tf_res = {}
        ms_res = {}
        res_dict = {'tensorflow': {}, 'torch': {}, 'mindspore': {}}
        config_dict['iter'] = i
        torch_exception = ''
        tf_exception = ''
        ms_exception = ''

        with ThreadPoolExecutor(max_workers=1) as pool:
            for frame in model_dict.keys():
                model_task = pool.submit(model_run,
                                         model=model_dict[frame], frame_work=frame, res_dict=res_dict,
                                         data=data)
                try:
                    model_task.result(timeout=3)
                except BaseException as e:
                    # model_num+=1
                    # with open(file_path + str(model_num) + '.pkl', 'wb') as file:
                    #     pickle.dump(lst_g, file)
                    res_dict[frame]['exception_info'] = e
                    print(f'{frame} compute failure:{e}')

        print(f'frame work compute complete, starting compare')

        flag_tf = res_dict['tensorflow']['flag_run']
        flag_torch = res_dict['torch']['flag_run']
        flag_ms = res_dict['mindspore']['flag_run']

        if flag_tf == flag_ms and flag_ms == flag_torch and not flag_torch:
            '''all have crash'''
            crash_model += 1
            continue
        lst_g = copy.deepcopy(sub_g)
        model_cnt += 1
        config_dict['model_cnt'] = model_cnt
        model_num += 1

        with open(file_path + str(model_num) + '.pkl', 'wb') as file:
            pickle.dump(sub_g, file)
        sub_g_des = sub_g.get_des()
        sub_g_src = sub_g.get_src()

        if not (flag_tf == flag_ms and flag_ms == flag_torch):
            tf_track = model_dict['tensorflow'].exception_track
            torch_track = model_dict['torch'].exception_track
            ms_track = model_dict['mindspore'].exception_track
            res_dict['tensorflow']['track'] = tf_track
            res_dict['torch']['track'] = torch_track
            res_dict['mindspore']['track'] = ms_track

            result_analyser.analyse_exception(config_dict=config_dict, res_dict=res_dict)
            back_propagation(graph, nodes_path)
            continue
            # break

        print(f'compare done, write into log')
        r'''res[0]:output
            res[1]:layer_result {id:name,output,output_size,from,to}
        '''

        if not (np.allclose(res_dict['torch']['final_res'], res_dict['tensorflow']['final_res'], 1e-3)
                and np.allclose(res_dict['torch']['final_res'], res_dict['mindspore']['final_res'], 1e-3)
                and np.allclose(res_dict['tensorflow']['final_res'], res_dict['mindspore']['final_res'], 1e-3)):
            if not (np.isnan(res_dict['torch']['final_res']).all() and np.isnan(
                    res_dict['tensorflow']['final_res']).all() and np.isnan(res_dict['mindspore']['final_res']).all()):
                has_bug = result_analyser.analyse_arrays(graph=sub_g, config_dict=config_dict, res_dict=res_dict)

                back_propagation(graph, nodes_path)

                # if has_bug:
                #     break
    print("crash:", crash_model)
    result_analyser.statics_bugs(config_dict=config_dict)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"time: {execution_time}s")
    print('done')
