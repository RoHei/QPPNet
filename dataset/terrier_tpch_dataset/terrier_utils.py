import collections
import pickle
import json
from typing import List, Tuple

import numpy as np
from collections import Counter

from dataset.constants import terrier_pname_group_dict, terrier_dimensions
from dataset.postgres_tpch_dataset.tpch_utils import PSQLTPCHDataSet
import dataset.terrier_tpch_dataset.terrier_query_info_1G as tqi_1
import dataset.terrier_tpch_dataset.terrier_query_info_0p1G as tqi_0p1
import dataset.terrier_tpch_dataset.terrier_query_info_10G as tqi_10


def get_input_for_all(SF):
    if SF == 1:
        tqi = tqi_1
    elif SF == 0.1:
        tqi = tqi_0p1
    else:
        tqi = tqi_10

    MEM_ADJUST_MAP = getattr(tqi, "MEM_ADJUST_MAP")

    def get_input(plan_dict):
        id_name = plan_dict["Node Type"].strip("tpch").upper()
        lst = getattr(tqi, id_name)
        feat_vec = []
        for op, feat in lst:
            feat_vec += feat
        if plan_dict["Node Type"] in MEM_ADJUST_MAP:
            feat_vec += [MEM_ADJUST_MAP[plan_dict["Node Type"]]]
        return feat_vec

    return get_input


def get_input_func(data_dir):
    SF = data_dir.strip('.csv').split('execution_')[1]
    if '0p1' in SF:
        num = 0.1
    elif '10' in SF:
        num = 10
    else:
        num = 1
    TR_GET_INPUT = collections.defaultdict(lambda: get_input_for_all(num))
    return TR_GET_INPUT


###############################################################################
#       Parsing data from csv files that contain json output of queries       #
###############################################################################

class TerrierTPCHDataSet(PSQLTPCHDataSet):
    def __init__(self, opt):
        self.batch_size = opt.batch_size
        self.num_q = 1
        self.scale = 10000
        self.eps = 0.001
        self.train_test_split = 0.8
        self.input_func: collections.defaultdict = get_input_func(opt.data_dir)

        all_plans = self.get_all_plans(opt.data_dir)
        # print(" \n".join([str(ent) for ent in all_data[:10]]))
        enum, num_grp = self.group_by_plan_structure(all_plans)

        count = Counter(enum)
        all_samp_num = count[min(count, key=lambda k: count[k])]

        all_groups = [[] for _ in range(num_grp)]
        for j, grp_idx in enumerate(enum):
            all_groups[grp_idx].append(all_plans[j])

        self.training_samples_per_query = int(all_samp_num * self.train_test_split)
        self.group_indexes = []

        train_data = []
        train_groups = [[] for _ in range(num_grp)]
        test_groups = [[] for _ in range(num_grp)]

        print(f"# of samples per query used: {all_samp_num}",
              f"# of training samples per query used: {self.training_samples_per_query}")

        counter = 0
        for idx, grp in enumerate(all_groups):
            train_data += grp[:self.training_samples_per_query]
            train_groups[idx] += grp[:self.training_samples_per_query]
            test_groups[idx] += grp[self.training_samples_per_query: all_samp_num]
            self.group_indexes += [idx] * self.training_samples_per_query
            counter += len(grp)

        self.num_groups = [num_grp]
        print("Number of samples per train groups: ", [len(grp) for grp in train_groups])
        self.dataset = train_data

        if not opt.test_time:
            self.mean_range_dict = self.normalize_operators(train_groups)
            with open('mean_range_dict.pickle', 'wb') as f:
                pickle.dump(self.mean_range_dict, f)
        else:
            with open(opt.mean_range_dict, 'rb') as f:
                self.mean_range_dict = pickle.load(f)

        self.test_dataset = [self.get_input(grp) for grp in test_groups]
        self.all_dataset = [self.get_input(grp) for grp in all_groups]

    def get_input(self, operator_data: List[dict]):  # Helper for sample_data
        """
            Vectorize the input of a list of plan_dicts (=elements of training data),
            that have the same query plan structure (of the same template/group)

            Args:
            - operator_data: a list of plan_dict, each plan_dict correspond to a query plan in the dataset;
                    requires that all plan_dicts is of the same query template/group

            Returns:
            - new_samp_dict: a dictionary, with the following keys:
                -- node_type      : name of the operator that the pipeline corresponds to
                -- real_node_type : the pipeline name
                -- subbatch_size  : number of queries in data
                -- feat_vec       : a numpy array of shape (batch_size x feat_dim) that's the vectorized inputs for all queries in data
                -- children_plan  : list of dictionaries with each being an output of  a recursive call to get_input on a child of current node
                -- total_time     : a vector of prediction target for each query in data
                -- is_subplan     : if the queries are subplans
        """
        samp_dict = {"node_type": operator_data[0]["Operator Type"],
                     "real_node_type": operator_data[0]["Node Type"],
                     "subbatch_size": len(operator_data)}

        # Reading out feature vector according to input function
        feature_vector = np.array([self.input_func[jss["Node Type"]](jss) for jss in operator_data])
        feature_vector = (feature_vector + self.eps) / (self.mean_range_dict[samp_dict["node_type"]][0] + self.eps)

        # Adding random weights? Special case for "lineitem" nodes
        if 'lineitem' in samp_dict["real_node_type"]:
            feature_vector += np.random.default_rng().normal(loc=0, scale=1, size=feature_vector.shape)
        else:
            feature_vector += np.random.default_rng().normal(loc=0, scale=0.1, size=feature_vector.shape)

        operator_runtimes = [jss['Actual Total Time'] for jss in operator_data]

        # Recursive pass over children
        child_plan_lst = []
        if 'Plans' in operator_data[0]:
            for i in range(len(operator_data[0]['Plans'])):
                child_plan_dict = self.get_input([jss['Plans'][i] for jss in operator_data])
                child_plan_dict['is_subplan'] = False
                child_plan_lst.append(child_plan_dict)

        samp_dict["feat_vec"] = np.array(feature_vector).astype(np.float32)
        samp_dict["children_plan"] = child_plan_lst
        samp_dict["total_time"] = np.array(operator_runtimes).astype(np.float32) / self.scale
        return samp_dict

    def normalize_operators(self, train_groups):  # compute the mean and std vec of each operator
        feat_vec_col = {operator: [] for operator in terrier_dimensions}

        def parse_input(data):
            feat_vec = [self.input_func[data[0]["Operator Type"]](jss) for jss in data]
            # print(feat_vec)
            if 'Plans' in data[0]:
                for i in range(len(data[0]['Plans'])):
                    parse_input([jss['Plans'][i] for jss in data])
            feat_vec_col[data[0]["Operator Type"]].append(np.array(feat_vec).astype(np.float32))

        for grp in train_groups:
            parse_input(grp)

        def cmp_mean_range(feat_vec_lst):
            if len(feat_vec_lst) == 0:
                return 0, 1
            else:
                total_vec = np.concatenate(feat_vec_lst)
                return (np.mean(total_vec, axis=0),
                        np.max(total_vec, axis=0))

        mean_range_dict = {operator: cmp_mean_range(feat_vec_col[operator]) \
                           for operator in terrier_dimensions}
        return mean_range_dict

    def get_all_plans(self, file_name: str):
        """
        Returns a list of dicts each with the key:
        Actual Total Time, Node Type             ,  Operator Type, Plans
        123142423,         tpc_h_scan_lineitem_p1, operator_16., [more dicts]
        """
        output: List[dict] = []
        current_operator_tree: dict = {"Actual Total Time": 0}
        previous_query = None

        f = open(file_name, 'r')
        lines = f.readlines()[1:]

        for line in lines:
            tokens = line.strip('\n').split(",")
            node_type = tokens[0]

            if len(node_type.split('_')) < 2:
                raise ValueError(f"Node Type is invalid {node_type}")

            # The query is extracted from the column "feature"
            current_query = "_".join(node_type.split('_')[1:-1])

            if previous_query is not None:
                if previous_query != current_query:
                    output.append(current_operator_tree)
                    current_operator_tree = {"Actual Total Time": 0}
                elif previous_query == current_query:
                    # Create a subplan. This assumes that the operators are all in the right order in the file!?
                    new_tree = {"Plans": [current_operator_tree],
                                'Actual Total Time': current_operator_tree['Actual Total Time']}
                    current_operator_tree = new_tree

            current_operator_tree['Node Type'] = node_type.strip(" ")
            current_operator_tree['Operator Type'] = "operator_" + str(terrier_pname_group_dict[node_type])
            current_operator_tree['Actual Total Time'] += float(tokens[-1])
            previous_query = current_query
        # jss is a list of json-transformed dicts, one for each query
        return output

    def group_by_plan_structure(self, data: List[dict]) -> Tuple[List, int]:
        counter = 0
        enum = []
        unique = []
        for plan_dict in data:
            grp_num = "_".join(plan_dict['Node Type'].split("_")[1:-1])
            if grp_num in unique:
                enum.append(unique.index(grp_num))
            else:
                enum.append(counter)
                unique.append(grp_num)
                counter += 1
        print(f"{counter} distinct templates identified")
        print(f"Operators: {unique}")
        return enum, counter

    def sample_new_batch(self):
        """
        Reads out a random batch from the dataset and applies a group-wise parsing
        """

        # Create random indexes from the dataset (num = batch size)
        sample_indexes = np.random.choice(np.arange(len(self.dataset)), self.batch_size, replace=False)

        # Group them according to the given groups
        grouped_samples = [[] for _ in range(self.num_groups[0])]
        for sample_index in sample_indexes:
            # Look up the group of each datapoint
            group_index = self.group_indexes[sample_index]
            # Collect the actual datapoints by their group
            grouped_samples[group_index].append(self.dataset[sample_index])

        # Parse each group individually and append the parsed results to the output
        batch = []
        for group in grouped_samples:
            if len(group) != 0:
                batch.append(self.get_input(group))
        return batch
