import collections
import json
import os
import pickle
from typing import List, Tuple

import numpy as np
from dataset.constants import get_basics, all_dicts, tpch_input_functions


###############################################################################
#       Parsing data from csv files that contain json output of queries       #
###############################################################################

class PSQLTPCHDataSet:
    def __init__(self, parameters):
        """
            Initialize the dataset by parsing the data files.
            Perform train test split and normalize each feature using mean and max of the train dataset.

            self.dataset is the train dataset
        """

        self.scale_factor = 1
        self.train_test_split = 0.8
        self.training_samples_per_query = int(500 * self.train_test_split)
        self.batch_size = parameters.batch_size
        self.number_of_queries = 22
        self.input_functions = collections.defaultdict(lambda: get_basics, tpch_input_functions)

        # Reading and sorting file names that each contain the query plans
        file_names = [fname for fname in os.listdir(parameters.data_dir)]
        file_names = sorted(file_names, key=lambda fname: int(fname.split('_plans')[0][6:]))

        all_query_plans = []
        all_queries_by_group, all_test_queries_by_group = [], []

        self.group_indexes = []
        self.num_groups = [0] * self.number_of_queries

        # Collect query plans and groups from file
        for file_index, file_name in enumerate(file_names):
            query_plans = self.load_query_from_file(parameters.data_dir + '/' + file_name)
            group_index_per_query, number_of_groups = self.group_by_plan_structure(query_plans)

            # For all the different groups (e.g. query types) in the plan, do actual grouping of queries
            queries_by_group = [[] for _ in range(number_of_groups)]
            for j, grp_idx in enumerate(group_index_per_query):
                queries_by_group[grp_idx].append(query_plans[j])
            all_queries_by_group += queries_by_group

            # Training data
            self.group_indexes += group_index_per_query[:self.training_samples_per_query]
            self.num_groups[file_index] = number_of_groups
            all_query_plans += query_plans[:self.training_samples_per_query]

            # Testing data
            test_queries_by_group = [[] for _ in range(number_of_groups)]
            for j, grp_idx in enumerate(group_index_per_query[self.training_samples_per_query:]):
                test_queries_by_group[grp_idx].append(query_plans[self.training_samples_per_query + j])
            all_test_queries_by_group += test_queries_by_group

        self.query_plans = all_query_plans
        self.num_query_plans = len(self.query_plans)

        self.init_feature_stats(parameters)

        self.test_dataset = [self.vectorize_plans(grp) for grp in all_test_queries_by_group]
        self.train_dataset = [self.vectorize_plans(grp) for grp in all_queries_by_group]

    def init_feature_stats(self, parameters):
        if not parameters.test_time:
            self.feature_statistics = self.get_feature_statistics()
            with open('mean_range_dict.pickle', 'wb') as f:
                pickle.dump(self.feature_statistics, f)
        else:
            with open(parameters.feature_statistics, 'rb') as f:
                self.feature_statistics = pickle.load(f)

    def get_feature_statistics(self):  # compute the mean and std vec of each operator
        """
            For each operator, normalize each input feature to have a mean of 0 and maximum of 1

            Returns:
            - mean_range_dict: a dictionary where the keys are the Operator Names and the values are 2-tuples (mean_vec, max_vec):
                -- mean_vec : a vector of mean values for input features of this operator
                -- max_vec  : a vector of max values for input features of this operator
        """
        feat_vec_col = {operator: [] for operator in all_dicts}

        def parse_input(data):
            feat_vec = [self.input_functions[data[0]["Node Type"]](jss) for jss in data]
            if 'Plans' in data[0]:
                for i in range(len(data[0]['Plans'])):
                    parse_input([jss['Plans'][i] for jss in data])
            feat_vec_col[data[0]["Node Type"]].append(np.array(feat_vec).astype(np.float32))

        for i in range(self.num_query_plans // self.training_samples_per_query):
            try:
                if self.num_groups[i] == 1:
                    parse_input(
                        self.query_plans[i * self.training_samples_per_query:(i + 1) * self.training_samples_per_query])
                else:
                    groups = [[] for j in range(self.num_groups[i])]
                    offset = i * self.training_samples_per_query
                    for j, plan_dict in enumerate(self.query_plans[offset:offset + self.training_samples_per_query]):
                        groups[self.group_indexes[offset + j]].append(plan_dict)
                    for grp in groups:
                        parse_input(grp)
            except:
                print('i: {}'.format(i))

        def cmp_mean_range(feat_vec_lst):
            if len(feat_vec_lst) == 0:
                return 0, 1
            else:
                total_vec = np.concatenate(feat_vec_lst)
                return (np.mean(total_vec, axis=0),
                        np.max(total_vec, axis=0) + np.finfo(np.float32).eps)

        mean_range_dict = {operator: cmp_mean_range(feat_vec_col[operator]) \
                           for operator in all_dicts}
        return mean_range_dict

    def load_query_from_file(self, file_name: str) -> List[dict]:
        """
            Parse from data file

            Args:
            - file_name: the name of data file to be parsed

            Returns:
            - jss: a sanitized list of dictionary, one per query, parsed from the input data file
        """
        # jsonstrs = []
        # curr = ""
        # prev = None
        # prevprev = None
        jss = []
        with open(file_name, 'r') as f:
            for row in f.readlines():
                # if not ('[' in row or '{' in row or ']' in row or '}' in row \
                #         or ':' in row):
                #     continue
                # newrow = row.replace('+', "").replace("(1 row)\n", "").strip('\n').strip(' ')
                # if 'CREATE' not in newrow and 'DROP' not in newrow and 'Tim' != newrow[:3]:
                #     curr += newrow
                # if prevprev is not None and 'Execution Time' in prevprev:
                #     jsonstrs.append(curr.strip(' ').strip('QUERY PLAN').strip('-'))
                #     curr = ""
                # prevprev = prev
                # prev = newrow
                jss.append(json.loads(row)[0]['Plan'])

        # strings = [s for s in jsonstrs if s[-1] == ']']
        # print(len(strings))

        # for idx in range(len(strings)):
        #     s = strings[idx]
        #     print(idx)
        #     print(len(s))
        #     json.loads(s)

        #     break
        # jss = [json.loads(s)[0]['Plan'] for s in strings]
        # jss is a list of json-transformed dicts, one for each query
        return jss

    def group_by_plan_structure(self, data: List[dict]) -> Tuple[List, int]:
        """
            Groups the queries by their query plan structure

            Args:
            - data: a list of dictionaries, each being a query from the dataset

            Returns:
            - enum    : a list of same length as data, containing the group indexes for each query in data
            - counter : number of distinct groups/templates
        """

        def hash(plan_dict):
            res = plan_dict['Node Type']
            if 'Plans' in plan_dict:
                for chld in plan_dict['Plans']:
                    res += hash(chld)
            return res

        counter = 0
        string_hash = []
        enum = []
        for plan_dict in data:
            string = hash(plan_dict)
            # print(string)
            try:
                idx = string_hash.index(string)
                enum.append(idx)
            except:
                idx = counter
                counter += 1
                enum.append(idx)
                string_hash.append(string)
        # print(f"{counter} distinct templates identified")
        # print(f"Operators: {string_hash}")
        assert (counter > 0)
        return enum, counter

    def vectorize_plans(self, data: List[dict]) -> dict:  # Helper for sample_data
        """
            Vectorize the input of a list of queries that have the same plan structure (of the same template/group)

            Args:
            - data: a list of plan_dict, each plan_dict correspond to a query plan in the dataset;
                    requires that all plan_dicts is of the same query template/group

            Returns:
            - samp_dict: a dictionary, where each level has the following attribute:
                -- node_type     : name of the operator
                -- subbatch_size : number of queries in data
                -- feat_vec      : a numpy array of shape (batch_size x feat_dim) that's
                                   the vectorized inputs for all queries in data
                -- children_plan : list of dictionaries with each being an output of
                                   a recursive call to get_input on a child of current node
                -- total_time    : a vector of prediction target for each query in data
                -- is_subplan    : if the queries are subplans
        """
        samp_dict = {"node_type": data[0]["Node Type"],
                     "subbatch_size": len(data)}

        feature_vector = np.array([self.input_functions[jss["Node Type"]](jss) for jss in data])

        # normalize feat_vec
        feature_vector = (feature_vector -
                          self.feature_statistics[samp_dict["node_type"]][0]) \
                         / self.feature_statistics[samp_dict["node_type"]][1]

        total_time = [jss['Actual Total Time'] for jss in data]

        child_plan_lst = []
        if 'Plans' in data[0]:
            for i in range(len(data[0]['Plans'])):
                child_plan_dict = self.vectorize_plans([jss['Plans'][i] for jss in data])
                child_plan_lst.append(child_plan_dict)

        samp_dict["feat_vec"] = np.array(feature_vector).astype(np.float32)
        samp_dict["children_plan"] = child_plan_lst
        samp_dict["total_time"] = np.array(total_time).astype(np.float32) / self.scale_factor

        if 'Subplan Name' in data[0]:
            samp_dict['is_subplan'] = True
        else:
            samp_dict['is_subplan'] = False
        return samp_dict

    ###############################################################################
    #       Sampling subbatch data from the dataset; total size is batch_size     #
    ###############################################################################
    def sample_new_batch(self) -> List[dict]:
        """
            Randomly sample a batch of data points from the train dataset

            Returns:
            - parsed_input: a list of dictionaries with inputs vectorized by get_input,
                            each dictionary contains all samples in the batch that comes from this group
        """
        sample_indexes = np.random.choice(np.arange(self.num_query_plans), self.batch_size, replace=False)

        grouped_samples = [[[] for j in range(self.num_groups[i])] for i in range(self.number_of_queries)]

        for sample_index in sample_indexes:
            group_index = self.group_indexes[sample_index]
            grouped_samples[sample_index // self.training_samples_per_query][group_index].append(self.query_plans[sample_index])

        batch = []
        for i, temp in enumerate(grouped_samples):
            for grp in temp:
                if len(grp) != 0:
                    batch.append(self.vectorize_plans(grp))

        return batch
