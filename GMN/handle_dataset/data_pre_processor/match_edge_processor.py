from cmath import nan
import random
from typing_extensions import Self
import math

from tqdm import tqdm
import numpy as np
import ast
from typing import Dict, List
import torch
from GMN.handle_dataset.data_pre_processor.BaseProcessor import BaseDataPreProcessor
from torch_geometric.data import Data
from GMN.position_encoding.data import GraphDataset


class DataPreProcessor(BaseDataPreProcessor):
    def read_data(self, data):
        pairs, labels = [], []

        for i in tqdm(range(len(data))):
            try:
                p_sql_rel = data.loc[i]['P_rel']
                g_sql_rel = data.loc[i]['G_rel']
            except:
                print(i)

            try:
                label = data.loc[i]['new_labels']
            except:
                label = data.loc[i]['label']
            db_id = data.loc[i]['db_id']
            if "[" and "]" in db_id:
                db_id = ast.literal_eval(db_id)
                db_id = db_id[0]
            
            if int(label) == 0:
                label = -1
            elif int(label) == 1:
                label = 1

            ground_truth = self.sql_to_graph(g_sql_rel, db_id)
            prediction = self.sql_to_graph(p_sql_rel, db_id)
            match_edge = self.add_match_edge(ground_truth, prediction)
            pairs.append((ground_truth, prediction, match_edge))
            labels.append(label)
        return pairs, labels

    def read_data_cl(self, data, aug, aug_ratio):
        pairs, labels = [], []

        for i in tqdm(range(len(data))):
            try:
                p_sql_rel = data.loc[i]['P_rel']
                g_sql_rel = data.loc[i]['G_rel']
            except:
                print(i)

            try:
                label = data.loc[i]['new_labels']
            except:
                label = data.loc[i]['label']
            db_id = data.loc[i]['db_id']
            if "[" and "]" in db_id:
                db_id = ast.literal_eval(db_id)
                db_id = db_id[0]
            if math.isnan(label):
                label = 0
            elif int(label) == 0:
                    label = -1
            elif int(label) == 1:
                    label = 1
    
            try:
                ground_truth = self.sql_to_graph(g_sql_rel, db_id)
                prediction = self.sql_to_graph(p_sql_rel, db_id)
                if aug == 'dropN':
                    ground_truth = self.drop_nodes(ground_truth, aug_ratio)
                    prediction = self.drop_nodes(prediction, aug_ratio)
                elif aug == 'permE':
                    ground_truth = self.permute_edges(ground_truth, aug_ratio)
                    prediction = self.permute_edges(prediction, aug_ratio)
                elif aug == 'subgraph':
                    ground_truth = self.subgraph(ground_truth, aug_ratio)
                    prediction = self.subgraph(prediction, aug_ratio)
                elif aug == 'maskN':
                    ground_truth = self.mask_nodes(ground_truth, aug_ratio)
                    prediction = self.mask_nodes(prediction, aug_ratio)

                match_edge = self.add_match_edge(ground_truth, prediction)
                pairs.append((ground_truth, prediction, match_edge))
                labels.append(label)
            except:
                print('wrong data')

        return pairs, labels

    def update_edges(self, edge, remove_node, edge_type):
        remove_node_list = set(remove_node)
        edge_new = []
        edge_type_new = []

        for i in range(len(edge)):
            if edge[i][0] in remove_node_list or edge[i][1] in remove_node_list:
                continue
            else:
                count1 = sum(1 for num in remove_node if edge[i][0] > num)
                count2 = sum(1 for num in remove_node if edge[i][1] > num)
                edge_new.append([edge[i][0] - count1, edge[i][1] - count2])
                edge_type_new.append(edge_type[i])

        return edge_new, edge_type_new


    def drop_nodes(self, data, aug_ratio):

        node_num = len(data['node_type'])
        drop_num = int(node_num  * aug_ratio)
        idx_perm = np.random.permutation(node_num)

        idx_drop = idx_perm[:drop_num]
        idx_nondrop = idx_perm[drop_num:]
        idx_nondrop.sort()

        edge, edge_type = self.update_edges(data['edge'], idx_drop.tolist(), data['edge_type'])
        if len(edge) == 0:
            data = data
        else:
            data['edge'] = edge
            data['edge_type'] = edge_type
            data['node_emb'] = [data['node_emb'][i] for i in idx_nondrop]
            data['node_type'] = [data['node_type'][i] for i in idx_nondrop]
            data['node_value'] = [data['node_value'][i] for i in idx_nondrop]
            data['mask_com'] = [data['mask_com'][i] for i in idx_nondrop]

        return data

    def permute_edges(self, data, aug_ratio):

        node_num = len(data['node_type'])
        edge_num = len(data['edge'])
        permute_num = int(edge_num * aug_ratio)

        idx_add = [[random.randint(0, node_num-1), random.randint(0, node_num-1)] for _ in range(permute_num)]

        e_set = [set(i) for i in data['edge']]
        count = 0
        for i in idx_add:
            if set(i) not in e_set:
                count += 1
                del data['edge'][-count]
                data['edge'].append(i)

        return data

    def subgraph(self, data, aug_ratio):

        node_num = len(data['node_type'])
        edge_num = len(data['edge'])

        sub_num = int(node_num * aug_ratio)

        idx_sub = [random.randint(0, node_num - 1)]
        idx_neigh = [sublist[0] if sublist[1] == idx_sub[-1] else sublist[1] for sublist in data['edge'] if idx_sub[-1] in sublist]

        count = 0
        while len(idx_sub) <= sub_num:
            count += 1
            if count > node_num:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = random.choice(idx_neigh)
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            idx_neigh += [sublist[0] if sublist[1] == idx_sub[-1] else sublist[1] for sublist in data['edge'] if idx_sub[-1] in sublist]
            idx_neigh = list(set(idx_neigh))

        idx_drop = [n for n in range(node_num) if not n in idx_sub]
        idx_nondrop = idx_sub
        idx_drop.sort()
        idx_nondrop.sort()

        edge, edge_type = self.update_edges(data['edge'], idx_drop, data['edge_type'])
        if len(edge) == 0:
            print('none_edge')
            data = data
        else:
            data['edge'] = edge
            data['edge_type'] = edge_type
            data['node_emb'] = [data['node_emb'][i] for i in idx_nondrop]
            data['node_type'] = [data['node_type'][i] for i in idx_nondrop]
            data['node_value'] = [data['node_value'][i] for i in idx_nondrop]
            data['mask_com'] = [data['mask_com'][i] for i in idx_nondrop]

        return data
    
    def mask_nodes(self, data, aug_ratio):

        node_num = len(data['node_type'])
        mask_num = int(node_num * aug_ratio)

        mean_array = np.mean(data['node_emb'], axis=0)
        mean_array = mean_array.astype(int)

        idx_mask = np.random.choice(node_num, mask_num, replace=False)
        for i in idx_mask:
            data['node_emb'][i] = mean_array

        return data


    def add_match_edge(self, ground_truth, prediction):
        match_edge = []
        ground_node_value = ground_truth['node_value']
        prediction_node_value = prediction['node_value']

        for g_i, g_value in enumerate(ground_node_value):
            if g_value is None or g_value not in prediction_node_value:
                continue

            for i, value in enumerate(prediction_node_value):

                if value == g_value:
                    match_edge.append([g_i, i])

        return match_edge

    def create_graph(self, rel, last, node_type, node_emb, node_value, edge, edge_type, mask_com, output_map):
        # {0 : {output : [2,3,4,6,7,8,3], input: []},
        #  1 : {output : [2,3,4,6,7,8,3], input: [0],
        #  2 : {output : [2,3,4,6,7,8,3], input: [0, 1]}}
        # Id, input
        #  input: [0, 1] -> [2,3,4,6,7,8,3] + [2,3,4,6,7,8,3]
        if isinstance(rel, Dict):
            for k, v in rel.items():

                # array = computing_one_hot_encoded[k]
                # array = array.astype(int).T.values
                # node_emb.append(array)

                array = self.computing_one_hot_encoded[k]
                array = array.astype(int).T.values
                node_embedding = np.zeros(64)
                node_embedding[: array.shape[0]] = array
                node_emb.append(np.concatenate((node_embedding, self.string_to_hash_vector(k))))

                node_type.append(k)
                mask_com.append(1)
                node_value.append(None)

                if k != "rels":
                    max_key = max(output_map, key=int)
                    include_id = output_map[max_key]["include_id"]
                    include_id.append(len(node_type) - 1)
                    output_map[max_key]["include_id"] = include_id

                if last != -1:
                    index = len(node_type) - 1
                    edge.append([last, index])
                    # edge_type.append("ast_edge")
                    edge_type.append(np.array([1, 0, 0]))

                index = len(node_type) - 1

                if k == "outputs":
                    self.handling_output_remain_number = len(v)
                if k in ["input", "group", "operands", "partition_by", "requiredColumns", "field"]:

                    max_output_key = str(max(map(int, output_map.keys())))
                    max_output_input = output_map[max_output_key]['input']
                    merge_outputs_list = []
                    for input_value in max_output_input:
                        merge_outputs_list.extend(output_map[str(input_value)]['output_id'])
                    if isinstance(v, int):
                        edge.append([index, merge_outputs_list[v]])
                        # edge_type.append("data_edge")
                        edge_type.append(np.array([0, 1, 0]))

                    elif self.isinstanceIntList(v):
                        for v_index in v:
                            edge.append([index, merge_outputs_list[v_index]])
                            # edge_type.append("data_edge")
                            edge_type.append(np.array([0, 1, 0]))
                    else:
                        self.create_graph(v, index, node_type, node_emb, node_value, edge, edge_type, mask_com, output_map)
                else:
                   self.create_graph(v, index, node_type, node_emb, node_value, edge, edge_type, mask_com, output_map)

        elif isinstance(rel, List) and len(rel) == 0:

            # array = computing_one_hot_encoded['content_node']
            # array = array.astype(int).T.values
            # node_emb.append(array)

            rel = str(rel)
            ascii_values = [ord(char) for char in rel]
            node_embedding = np.zeros(64)
            node_embedding[: np.array(ascii_values).shape[0]] = np.array(ascii_values)
            node_emb.append(np.concatenate((node_embedding, self.string_to_hash_vector(rel))))

            node_type.append('[]')
            mask_com.append(0)
            node_value.append('[]')

            index = len(node_type) - 1
            edge.append([last, index])
            # edge_type.append("ast_edge")
            edge_type.append(np.array([1, 0, 0]))

            max_key = max(output_map, key=int)
            include_id = output_map[max_key]["include_id"]
            include_id.append(index)
            output_map[max_key]["include_id"] = include_id


        elif isinstance(rel, List):
            for v in rel:

                if 'operator' in v and isinstance(v, dict):
    
                    # array = computing_one_hot_encoded[v['operator']]
                    # array = array.astype(int).T.values
                    # node_emb.append(array)

                    array = self.computing_one_hot_encoded[v['operator']]
                    array = array.astype(int).T.values
                    node_embedding = np.zeros(64)
                    node_embedding[: array.shape[0]] = array
                    node_emb.append(np.concatenate((node_embedding, self.string_to_hash_vector(v['operator']))))

                    node_type.append(v['operator'])
                    mask_com.append(1)
                    node_value.append(None)

                    edge.append([last, len(node_type) - 1])
                    # edge_type.append("ast_edge")
                    edge_type.append(np.array([1, 0, 0]))

                    del v['operator']
                    index = len(node_type) - 1
                    self.init_output_map(output_map, v['inputs'], index)
                    for input_id in v['inputs']:
                        logic_node = output_map[input_id]['operator_id']
                        edge.append([len(node_type) - 1, logic_node])
                        # edge_type.append("logic_edge")
                        edge_type.append(np.array([0, 1, 0]))

                    del v['inputs']
                    self.create_graph(v, index, node_type, node_emb, node_value, edge, edge_type, mask_com, output_map)

                else:
                    self.create_graph(v, last, node_type, node_emb, node_value, edge, edge_type, mask_com, output_map)

        elif type(rel) in [float, bool, int, str]:
            rel = str(rel)

            # array = computing_one_hot_encoded['content_node']
            # array = array.astype(int).T.values
            # node_emb.append(array)

            rel = str(rel)
            if len(rel) > 64:
                rel = rel[:64]

            ascii_values = [ord(char) for char in str(rel) if 0 <= ord(char) <= 127]
            node_embedding = np.zeros(64)
            node_embedding[: np.array(ascii_values).shape[0]] = np.array(ascii_values)
            node_emb.append(np.concatenate((node_embedding, self.string_to_hash_vector(rel))))

            node_type.append(rel)
            mask_com.append(0)
            node_value.append(rel)

            index = len(node_type) - 1
            edge.append([last, index])
            # edge_type.append("ast_edge")
            edge_type.append(np.array([1, 0, 0]))

            max_key = max(output_map, key=int)
            include_id = output_map[max_key]["include_id"]
            include_id.append(index)
            output_map[max_key]["include_id"] = include_id

            if self.handling_output_remain_number != 0:
                max_output_key = str(max(map(int, output_map.keys())))
                output_map[max_output_key]['output_id'].append(index)
                output_map[max_output_key]['output_name'].append(rel)
                self.handling_output_remain_number = self.handling_output_remain_number - 1

                input_output_id = []
                intput_output_name = []
                for input_id in output_map[max_output_key]['input']:
                    input_output_id += output_map[input_id]['output_id']
                    intput_output_name += output_map[input_id]['output_name']

                indexes = [index for index, value in enumerate(intput_output_name) if value == rel]
                for i in indexes:
                    edge.append([index, input_output_id[i]])
                    # edge_type.append("new_data_edge")
                    edge_type.append(np.array([0, 1, 0]))

    def pack_batch(self, graphs):
        from_idx = []
        to_idx = []
        graph_idx = []
        node_features = []
        edge_features = []
        mask_com = []
        n_total_nodes = 0
        n_total_edges = 0

        for i, pair_graph in enumerate(graphs):
            ground_truth = pair_graph[0]
            prediction = pair_graph[1]
            match_edge = pair_graph[2]

            g_node_fea = ground_truth['node_emb']
            p_node_fea = prediction['node_emb']

            g_n_nodes = len(g_node_fea)
            p_n_nodes = len(p_node_fea)

            g_edges = torch.tensor(ground_truth['edge'], dtype=torch.int32)
            p_edges = torch.tensor(prediction['edge'], dtype=torch.int32)
            m_edges = torch.tensor(match_edge, dtype=torch.int32)

            from_idx.append(g_edges[:, 0] + n_total_nodes)
            to_idx.append(g_edges[:, 1] + n_total_nodes)

            # edge_features += ground_truth['edge_type']

            if len(match_edge) == 0:
                n_total_nodes += g_n_nodes
            else:
                # from_idx.append(m_edges[:, 0] + n_total_nodes)
                n_total_nodes += g_n_nodes
                # to_idx.append(m_edges[:, 1] + n_total_nodes)

            # edge_features += [np.array([0, 0, 1]) for i in range(len(match_edge))]

            from_idx.append(p_edges[:, 0] + n_total_nodes)
            to_idx.append(p_edges[:, 1] + n_total_nodes)
            n_total_nodes += p_n_nodes

            # edge_features += prediction['edge_type']

            g_idx = i * 2
            p_idx = (i * 2) + 1

            graph_idx.append(torch.ones(g_n_nodes, dtype=torch.int32) * g_idx)
            graph_idx.append(torch.ones(p_n_nodes, dtype=torch.int32) * p_idx)

            node_features += (g_node_fea + p_node_fea)

            mask_com.append(np.array(ground_truth['mask_com']))
            mask_com.append(np.array(prediction['mask_com']))

            n_total_edges += len(ground_truth['edge']) + len(prediction['edge'])

        edge_features = [np.array([1]) for i in range(n_total_edges)]
        from_idx = torch.cat(from_idx).long()
        to_idx = torch.cat(to_idx).long()
        edge_features = torch.from_numpy(np.array(edge_features, dtype=np.float32))
        edge_tuple = torch.cat((from_idx.unsqueeze(1), to_idx.unsqueeze(1), edge_features), dim=1)

        graph_idx = torch.cat(graph_idx).long()
        node_features = torch.from_numpy(np.array(node_features, dtype=np.float32))
        mask_com = torch.from_numpy(np.concatenate(mask_com))

        node_tuple = torch.cat((graph_idx.unsqueeze(1), mask_com.unsqueeze(1), node_features), dim=1)

        n_graphs = len(graphs) * 2
        return edge_tuple, node_tuple, n_graphs

    def shuffle(self, pair_list, labels):
        data_with_label = [(pair, labels[idx]) for idx, pair in enumerate(pair_list)]
        random.shuffle(data_with_label)
        pair_list = [pair[0] for pair in data_with_label]
        labels = [pair[1] for pair in data_with_label]
        return pair_list, labels

    def pairs_spider(self, batch_size, pair_list, labels):
        pair_list, labels = self.shuffle(pair_list, labels) # For pretraining, please comment out this line.
        batch_data_list = []
        ptr = 0
        while ptr < len(pair_list):
            if ptr + batch_size > len(pair_list):
                next_ptr = len(pair_list)
            else:
                next_ptr = ptr + batch_size
            batch_graphs = pair_list[ptr: next_ptr]
            edge_tuple, node_tuple, n_graphs = self.pack_batch(batch_graphs)

            batch_data_list.append(
                [edge_tuple, node_tuple, n_graphs, torch.tensor(labels[ptr: ptr + batch_size], dtype=torch.int32)])
            ptr = next_ptr

        return batch_data_list

    def read_pair(self, g_sql_rel: str, p_sql_rel: str, db_id: str):

        ground_truth = self.sql_to_graph(g_sql_rel, db_id)
        prediction = self.sql_to_graph(p_sql_rel, db_id)
        match_edge = self.add_match_edge(ground_truth, prediction)
        return [(ground_truth, prediction, match_edge)]
