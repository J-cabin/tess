import dgl
import torch
from dgl.data.dgl_dataset import DGLDataset
import dgl.backend as F
from scipy.sparse import load_npz
from collections import Counter
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
import numpy as np
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import lzma
from scipy.sparse import csr_matrix, save_npz


class SATDataset(DGLDataset):
    def __init__(self, name, args=None):
        self.graphs = []
        self.labels = []
        self.gfeatures = []
        self.times = []
        self.benchmark_ids = []
        self.censor_type = []       # 0-uncensor; 1-censor

        # global num
        # self.N = self.get_benchmark_num(name)  # total graphs number
        self.num_benchmarks = 0
        self.num_solver = 0
        # self.gclasses = 0
        self.dim_nfeats = 0
        self.dim_gfeats = 0
        self.onehot_feats = 'typ'  # ont-hot编码的内容：degree-节点度；typ-节点类型；不能为None
        # self.append_feats = ['max_clause_length', 'avg_degree']   # 直接拼接的内容，modularity-图的模块度，avg_degree-图的平均度
        self.append_feats = []

        # 命令行参数
        self.gtype = args.graph
        self.time_out = args.time_out
        self.label_scale = args.label_scale

        # 原始数据
        if name in ['SAT2022', 'SAT2022_small', 'SAT2022_smallest']:
            self.input_path = Path.cwd().parent.joinpath('2022', 'anni-benchmarks')
        elif name in ['aig_dataset', 'INDU', 'UUF250_1065_100', 'Flat200-479', 'flat_all']:
            self.input_path = Path.cwd().parent.joinpath('prework', 'generate_dataset', 'dataset', name)

        # 特征和标签
        self.label_path = Path.cwd().parent.joinpath('data', 'label', name+'.csv')
        self.feat_path = Path.cwd().parent.joinpath('data', 'feature', name+'.csv')
        self.prop_path = Path.cwd().parent.joinpath('data', 'graph_properties', name+'.csv')

        # 生成的图
        self.struc_path = Path.cwd().joinpath('data', f'{name}_{args.graph}_bin')
        self.graph_path = Path.cwd().joinpath('data', f'{name}_{args.graph}_bin_attributed')

        super().__init__(name=name)

    # def get_benchmark_num(self, name):
    #     benchmark_file = Path.cwd().joinpath('data', name, 'benchmark.txt')
    #     benchmark_data = pd.read_csv(benchmark_file, sep=',')
    #     return benchmark_data.shape[0]

    def get_graph_features(self, ):
        from sklearn import preprocessing
        def col_operation(df):
            df[np.isinf(df)] = np.nan
            df = df.fillna(df.mean())
            if df.min() != df.max():
                df = (df - df.min()) / (df.max() - df.min())
            return df

        print('generate feature ...')
        df = pd.read_csv(self.feat_path, sep='\t')
        df = df.apply(col_operation)
        X = df.values
        X = preprocessing.normalize(X, axis=0)
        X = torch.FloatTensor(X)
        return X

    def process(self):
        print('Start Processing')

        # load graph labels (runtime), times, uncensor/censor idx
        interaction = pd.read_csv(self.label_path)
        self.num_benchmarks, self.num_solver = interaction.shape
        self.benchmark_ids = np.array([i for i in range(self.num_benchmarks)])
        self.times = torch.FloatTensor(np.clip(interaction.values, 0, self.time_out))  # 截取[0, time_out] 做求解器选择才用
        self.labels = interaction['Kissat_MAB-HyWalk'].values  # 第五个solver(Hywalk)
        self.uncensored_ids = np.argwhere(self.labels < self.time_out)
        self.censored_ids = np.argwhere(self.labels >= self.time_out)
        self.censor_type = (self.labels >= self.time_out).astype(int)
        if self.label_scale == 'log':
            ground_truth = np.log1p(self.labels)
        # elif self.label_scale == 'norm':
        #     ground_truth = (self.labels - np.min(self.labels)) / (np.max(self.labels) - np.min(self.labels))
        elif self.label_scale == 'none':
            ground_truth = self.labels
        else:
            raise ValueError(f'Undefined scaling type.')
        self.ground_truth = torch.FloatTensor(ground_truth)  # [2945]

        # load graph structures and features
        if not self.struc_path.exists() or not any([True for _ in os.scandir(self.struc_path)]):
            self.generate_structure_graph(self.struc_path, self.gtype)
        if not self.graph_path.exists() or not any([True for _ in os.scandir(self.graph_path)]):
            self.generate_attributed_graph(self.struc_path)

        self.gfeatures = self.get_graph_features()
        self.dim_gfeats = self.gfeatures.shape[1]

        # For each graph ID...
        for graph_id in tqdm(range(self.num_benchmarks)):
            # npz version
            # file_name = Path.cwd().joinpath('data', self._name, f'{graph_id}.npz')
            # data = load_npz(file_name)
            # g = dgl.from_scipy(data)

            # bin version
            # file_name = Path.cwd().joinpath('data', self._name + '_attr', f'{graph_id}.bin')
            file_name = self.graph_path.joinpath(f'{graph_id}.bin')
            g, _ = load_graphs(str(file_name))
            assert (len(g) == 1)
            g = g[0]
            self.graphs.append(g)
        self.dim_nfeats = len(self.graphs[0].ndata['attr'][0])
        # print(self.graphs)


    def generate_structure_graph(self, structure_path, gtype):
        def neg_transform(x, negvar_start_id):
            if x > 0:
                return x - 1
            else:
                return negvar_start_id - x - 1

        def skip_comment(lines):
            for idx, line in enumerate(lines):
                if not line.startswith('c'):
                    return idx

        def generate_lcg(f):
            if self._name in ['SAT2022', 'SAT2022_small', 'SAT2022_smallest']:
                lines = f.readlines()
                _, _, nbvar, nbclause = lines[0].decode().strip().split(' ')
            elif self._name == 'aig_dataset':
                lines = f.readlines()[1:]
                _, _, nbvar, nbclause = lines[0].strip().split(' ')
            elif self._name in ['INDU', 'UUF250_1065_100', 'Flat200-479', 'flat_all']:
                lines = f.readlines()
                start_idx = skip_comment(lines)
                lines = lines[start_idx:]
                _, _, nbvar, nbclause = lines[0].strip().split()
            else:
                raise ValueError('Undefined dataset.')
            num_nodes = int(nbvar) * 2 + int(nbclause)      # LCG
            # properties_data['num_nodes'].append(num_nodes)
            negvar_start_id = int(nbvar)
            clause_start_id = int(nbvar) * 2
            src = []
            dst = []
            # 添加L和C之间的边
            for clause_id, line in enumerate(lines[1:]):
                # print(line.decode())
                if self._name in ['SAT2022', 'SAT2022_small', 'SAT2022_smallest']:
                    vars = line.decode().strip('0\n').split()
                elif self._name in ['aig_dataset', 'INDU', 'UUF250_1065_100', 'Flat200-479', 'flat_all']:
                    vars = line.strip('0\n').split()
                else:
                    raise ValueError('Undefined dataset.')
                vars = list(map(eval, vars))  # 字符转数字
                # print(vars)
                for x in vars:
                    src.append(neg_transform(x, negvar_start_id))
                # src += [int(x) for x in vars]
                # print(src)
                dst += [clause_start_id + clause_id] * len(vars)
                # dst = np.ones_like(src) * (clause_start_id+clause_id)
                # for j in vars:
                # edge_data['graph_id'].append(graph_id)
                # edge_data['src'].append(j)
                # edge_data['dst'].append(clause_start_id+clause_id)
            src = np.array(src)
            dst = np.array(dst)
            data = np.ones_like(src)
            csr_data = csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
            g = dgl.from_scipy(csr_data)
            typ_list = [0] * int(nbvar) + [1] * int(nbvar) + [2] * int(nbclause)
            g.ndata['type'] = torch.tensor(typ_list)
            return g

        def generate_lig(f, weighting=False):
            from itertools import product
            if self._name in ['SAT2022', 'SAT2022_small', 'SAT2022_smallest']:
                lines = f.readlines()
                _, _, nbvar, nbclause = lines[0].decode().strip().split(' ')
            elif self._name == 'aig_dataset':
                lines = f.readlines()[1:]
                _, _, nbvar, nbclause = lines[0].strip().split(' ')
            elif self._name in ['INDU', 'UUF250_1065_100', 'Flat200-479', 'flat_all']:
                lines = f.readlines()
                start_idx = skip_comment(lines)
                lines = lines[start_idx:]
                _, _, nbvar, nbclause = lines[0].strip().split()
            else:
                raise ValueError('Undefined dataset.')
            num_nodes = int(nbvar) * 2  # LIG
            negvar_start_id = int(nbvar)
            # data = np.zeros((num_nodes, num_nodes))
            dic = dict()
            for clause_id, line in enumerate(lines[1:]):
                if self._name in ['SAT2022', 'SAT2022_small', 'SAT2022_smallest']:
                    vars = line.decode().strip('0\n').split()
                elif self._name in ['aig_dataset', 'INDU', 'UUF250_1065_100', 'Flat200-479', 'flat_all']:
                    vars = line.strip('0\n').split()
                else:
                    raise ValueError('Undefined dataset.')
                vars = list(map(eval, vars))  # 字符转数字
                for idx, x in enumerate(vars):
                    vars[idx] = neg_transform(x, negvar_start_id)
                pair = product(vars, vars)  # 有self-loop，有bi-direction
                for x in pair:
                    if weighting:   # 加权
                        if (x[0], x[1]) in dic.keys():
                            dic[(x[0], x[1])] += 1
                        else:
                            dic[(x[0], x[1])] = 1
                    else:
                        dic[(x[0], x[1])] = 1
            dic_key = dic.keys()
            src, dst = map(list, zip(*dic_key))
            data = list(dic.values())
            csr_data = csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
            g = dgl.from_scipy(csr_data)
            typ_list = [0] * int(nbvar) + [1] * int(nbvar)
            g.ndata['type'] = torch.tensor(typ_list)
            return g

        def generate_vig(f, weighting=False):
            from itertools import product
            if self._name in ['SAT2022', 'SAT2022_small', 'SAT2022_smallest']:
                lines = f.readlines()
                _, _, nbvar, nbclause = lines[0].decode().strip().split(' ')
            elif self._name == 'aig_dataset':
                lines = f.readlines()[1:]
                _, _, nbvar, nbclause = lines[0].strip().split(' ')
            elif self._name in ['INDU', 'UUF250_1065_100', 'Flat200-479', 'flat_all']:
                lines = f.readlines()
                start_idx = skip_comment(lines)
                lines = lines[start_idx:]
                _, _, nbvar, nbclause = lines[0].strip().split()
            else:
                raise ValueError('Undefined dataset.')
            num_nodes = int(nbvar)  # VIG
            # negvar_start_id = int(nbvar)
            dic = dict()
            for clause_id, line in enumerate(lines[1:]):
                if self._name in ['SAT2022', 'SAT2022_small', 'SAT2022_smallest']:
                    vars = line.decode().strip('0\n').split()
                elif self._name in ['aig_dataset', 'INDU', 'UUF250_1065_100', 'Flat200-479', 'flat_all']:
                    vars = line.strip('0\n').split()
                else:
                    raise ValueError('Undefined dataset.')
                vars = list(map(eval, vars))  # 字符转数字
                vars = list(map(abs, vars))   # 负转正
                vars = list(map(lambda var: var-1, vars))   # -1
                pair = product(vars, vars)
                for x in pair:
                    if weighting:  # 加权
                        if (x[0], x[1]) in dic.keys():
                            dic[(x[0], x[1])] += 1
                        else:
                            dic[(x[0], x[1])] = 1
                    else:
                        dic[(x[0], x[1])] = 1
            dic_key = dic.keys()
            src, dst = map(list, zip(*dic_key))
            data = list(dic.values())
            csr_data = csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
            g = dgl.from_scipy(csr_data)
            typ_list = [0] * int(nbvar)
            g.ndata['type'] = torch.tensor(typ_list)
            return g

        # STEP 1: 存结构（并且记录节点类型），一次性存所有图
        print('Generating structure graphs...')
        benchmark_data = pd.read_csv(self.prop_path, sep=',')
        for idx, file in enumerate(tqdm(benchmark_data['benchmark_name'])):
            if self._name in ['SAT2022', 'SAT2022_small', 'SAT2022_smallest']:
                f = lzma.open(self.input_path.joinpath(file))
            elif self._name in ['aig_dataset', 'INDU', 'UUF250_1065_100', 'Flat200-479', 'flat_all']:
                f = open(self.input_path.joinpath(file))
            else:
                raise ValueError(f'Undefined dataset.')
            graph_id = benchmark_data['benchmark_id'][idx]
            if gtype == 'lcg':
                g = generate_lcg(f)
            elif gtype == 'lig':
                g = generate_lig(f, weighting=False)
            elif gtype == 'wlig':
                g = generate_lig(f, weighting=True)
            elif gtype == 'vig':
                g = generate_vig(f, weighting=False)
            elif gtype == 'wvig':
                g = generate_vig(f, weighting=True)
            else:
                raise ValueError(f'Undefined Graph Type.')
            output_file = structure_path.joinpath(f'{graph_id}.bin')
            save_graphs(str(output_file), g)
            f.close()


    def generate_attributed_graph(self, structure_path):
        print('Generating attributed graphs...')
        graphs = []
        for graph_id in tqdm(range(self.num_benchmarks)):
            g, _ = load_graphs(str(structure_path.joinpath(f'{graph_id}.bin')))
            assert (len(g) == 1)
            g = g[0]
            graphs.append(g)
        print(len(graphs))

        print('Label Encoding...')
        if self.onehot_feats == 'degree':
            for g in graphs:
                g.ndata['label'] = g.in_degrees()

            # in case the labels/degrees are not continuous number
            # 映射到0~k
            nlabel_set = set([])
            for g in tqdm(graphs):
                nlabel_set = nlabel_set.union(
                    set([F.as_scalar(nl) for nl in g.ndata['label']]))
            nlabel_set = list(nlabel_set)
            label2idx = {
                nlabel_set[i]: i
                for i in range(len(nlabel_set))
            }
        elif self.onehot_feats == 'typ':
            for g in tqdm(graphs):
                # 根据图中每个节点的类型而定
                g.ndata['label'] = g.ndata['type']
                label2idx = {0: 0, 1: 1, 2: 2}  # LCG graph，0-positive variable，1-negative variable，2-clause

        print("Generate node attr by node label...")
        for gid, g in enumerate(tqdm(graphs)):
            attr = np.zeros((
                g.number_of_nodes(), len(label2idx) + len(self.append_feats)))
            attr[range(g.number_of_nodes()), [label2idx[nl]
                                              for nl in F.asnumpy(g.ndata['label']).tolist()]] = 1
            if len(self.append_feats) > 0:
                properties_file = Path.parent.joinpath('prework', 'graph_properties',
                                                       'graph_properties.csv')  # 和上面那个properties文件不同
                df = pd.read_csv(properties_file, usecols=self.append_feats)
                attr[range(g.number_of_nodes()), -len(self.append_feats)] = np.ones(g.number_of_nodes()) * df.loc[
                    gid]  # 可能会报错
            g.ndata['attr'] = F.tensor(attr, F.float32)

            output_file = self.graph_path.joinpath(f'{gid}.bin')
            save_graphs(str(output_file), g)

    def __getitem__(self, i):
        return self.graphs[i], self.gfeatures[i], self.labels[i], self.ground_truth[i], self.benchmark_ids[i], \
            self.censor_type[i]

    def __len__(self):
        return len(self.graphs)
