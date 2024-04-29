import pickle

terrier_pname_group_dict = {
    "tpch_q1_p1": 0,
    "tpch_q1_p2": 1,
    "tpch_q4_p3": 1,
    "tpch_q5_p7": 1,
    "tpch_q7_p6": 1,
    "tpch_q1_p3": 2,
    "tpch_q4_p4": 2,
    "tpch_q5_p8": 2,
    "tpch_q7_p7": 2,
    "tpch_q11_p6": 2,
    "tpch_q4_p1": 3,
    "tpch_q5_p1": 4,
    "tpch_q11_p1": 4,
    "tpch_q4_p2": 5,
    "tpch_q5_p2": 6,
    "tpch_q5_p3": 6,
    "tpch_q7_p2": 6,
    "tpch_q7_p3": 6,
    "tpch_q11_p2": 6,
    "tpch_q5_p4": 7,
    "tpch_q5_p5": 8,
    "tpch_q7_p4": 8,
    "tpch_q5_p6": 9,
    "tpch_q7_p5": 10,
    "tpch_q6_p1": 11,
    "tpch_q7_p1": 12,
    "tpch_q11_p3": 13,
    "tpch_q11_p4": 14,
    "tpch_q11_p5": 15,
    "tpch_scan_lineitem_p1": 16,
    "tpch_scan_orders_p1": 16
}

terrier_dimensions = {
    "operator_0": 25,
    "operator_1": 41,
    "operator_2": 40,
    "operator_3": 13,
    "operator_4": 13,
    "operator_5": 53,
    "operator_6": 45,
    "operator_7": 49,
    "operator_8": 41,
    "operator_9": 57,
    "operator_10": 61,
    "operator_11": 24,
    "operator_12": 21,
    "operator_13": 48,
    "operator_14": 53,
    "operator_15": 49,
    "operator_16": 8
}

# all operators used in tpc-h
all_dicts = [
    "Aggregate",
    "Gather Merge",
    "Sort",
    "Seq Scan",
    "Index Scan",
    "Index Only Scan",
    "Bitmap Heap Scan",
    "Bitmap Index Scan",
    "Limit",
    "Hash Join",
    "Hash",
    "Nested Loop",
    "Materialize",
    "Merge Join",
    "Subquery Scan",
    "Gather",
]

join_types = ["semi", "inner", "anti", "full", "right"]

parent_rel_types = [
    "inner",
    "outer",
    "subquery"
]

sort_algos = [
    "quicksort",
    "top-n heapsort"
]

aggreg_strats = [
    "plain",
    "sorted",
    "hashed"
]

rel_names = [
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
]

index_names = [
    "customer_pkey",
    "idx_customer_nationkey",
    "part_pkey",
    "supplier_pkey",
    "idx_supplier_nation_key",
    "partsupp_pkey",
    "idx_partsupp_suppkey",
    "idx_partsupp_partkey",
    "partsupp_ps_partkey_fkey",
    "partsupp_ps_suppkey_fkey",
    "orders_pkey",
    "idx_orders_custkey",
    "idx_orders_orderdate",
    "lineitem_pkey",
    "idx_lineitem_orderkey",
    "idx_lineitem_part_supp",
    "idx_lineitem_shipdate",
    "lineitem_l_orderkey_fkey",
    "lineitem_l_partkey_l_suppkey_fkey",
    "nation_pkey",
    "idx_nation_regionkey",
    "region_pkey",
]

rel_attr_list_dict = {
    "customer": [
        "c_custkey",
        "c_name",
        "c_address",
        "c_nationkey",
        "c_phone",
        "c_acctbal",
        "c_mktsegment",
        "c_comment",
    ],
    "lineitem": [
        "l_orderkey",
        "l_partkey",
        "l_suppkey",
        "l_linenumber",
        "l_quantity",
        "l_extendedprice",
        "l_discount",
        "l_tax",
        "l_returnflag",
        "l_linestatus",
        "l_shipdate",
        "l_commitdate",
        "l_receiptdate",
        "l_shipinstruct",
        "l_shipmode",
        "l_comment",
    ],
    "nation": ["n_nationkey", "n_name", "n_regionkey", "n_comment"],
    "orders": [
        "o_orderkey",
        "o_custkey",
        "o_orderstatus",
        "o_totalprice",
        "o_orderdate",
        "o_orderpriority",
        "o_clerk",
        "o_shippriority",
        "o_comment",
    ],
    "part": [
        "p_partkey",
        "p_name",
        "p_mfgr",
        "p_brand",
        "p_type",
        "p_size",
        "p_container",
        "p_retailprice",
        "p_comment",
    ],
    "partsupp": [
        "ps_partkey",
        "ps_suppkey",
        "ps_availqty",
        "ps_supplycost",
        "ps_comment",
    ],
    "region": ["r_regionkey", "r_name", "r_comment"],
    "supplier": [
        "s_suppkey",
        "s_name",
        "s_address",
        "s_nationkey",
        "s_phone",
        "s_acctbal",
        "s_comment",
    ],
}

postgres_tpch_attr_val_dict = {
    'med': {'customer': [7500.0, 0, 0, 12.0, 0, 4404.87, 0, 0],
            'lineitem': [300486.0, 10003.0, 501.0, 3.0, 26.0, 34461.75, 0.05, 0.04, 0, 0, 0, 0, 0, 0, 0, 0],
            'nation': [12.0, 0, 2.0, 0],
            'orders': [300000.0, 7484.0, 0, 136058.42, 0, 0, 0, 0.0, 0],
            'part': [10000.0, 0, 0, 0, 0, 25.0, 0, 1409.49, 0],
            'partsupp': [10000.0, 500.0, 4995.0, 498.72, 0],
            'region': [2.0, 0, 0],
            'supplier': [500.0, 0, 0, 12.0, 0, 4422.77, 0]},

    'min': {'customer': [1.0, 0, 0, 0.0, 0, -999.95, 0, 0],
            'lineitem': [1.0, 1.0, 1.0, 1.0, 1.0, 901.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0],
            'nation': [0.0, 0, 0.0, 0],
            'orders': [1.0, 1.0, 0, 833.4, 0, 0, 0, 0.0, 0],
            'part': [1.0, 0, 0, 0, 0, 1.0, 0, 901.0, 0],
            'partsupp': [1.0, 1.0, 1.0, 1.01, 0],
            'region': [0.0, 0, 0],
            'supplier': [1.0, 0, 0, 0.0, 0, -966.2, 0]},

    'max': {'customer': [15000.0, 0, 0, 24.0, 0, 9999.72, 0, 0],
            'lineitem': [600000.0, 20000.0, 1000.0, 7.0, 50.0, 95949.5, 0.1, 0.08, 0, 0, 0, 0, 0, 0, 0, 0],
            'nation': [24.0, 0, 4.0, 0],
            'orders': [600000.0, 14999.0, 0, 479129.21, 0, 0, 0, 0.0, 0],
            'part': [20000.0, 0, 0, 0, 0, 50.0, 0, 1918.99, 0],
            'partsupp': [20000.0, 1000.0, 9999.0, 999.99, 0],
            'region': [4.0, 0, 0],
            'supplier': [1000.0, 0, 0, 24.0, 0, 9993.46, 0]}}

num_rel = 8
max_num_attr = 16
num_index = 22

tpch_dimensions = {'Seq Scan': num_rel + max_num_attr * 3 + 3,
                   'Index Scan': num_index + num_rel + max_num_attr * 3 + 3 + 1,
                   'Index Only Scan': num_index + num_rel + max_num_attr * 3 + 3 + 1,
                   'Bitmap Heap Scan': num_rel + max_num_attr * 3 + 3 + 32,
                   'Bitmap Index Scan': num_index + 3,
                   'Sort': 128 + 5 + 32,
                   'Hash': 4 + 32,
                   'Hash Join': 11 + 32 * 2,
                   'Merge Join': 11 + 32 * 2,
                   'Aggregate': 7 + 32,
                   'Nested Loop': 32 * 2 + 3,
                   'Limit': 32 + 3,
                   'Subquery Scan': 32 + 3,
                   'Materialize': 32 + 3,
                   'Gather Merge': 32 + 3,
                   'Gather': 32 + 3}


# need to normalize Plan Width, Plan Rows, Total Cost, Hash Bucket
def get_basics(plan_dict):
    return [plan_dict['Plan Width'], plan_dict['Plan Rows'], plan_dict['Total Cost']]


def get_rel_one_hot(rel_name):
    arr = [0] * num_rel
    arr[rel_names.index(rel_name)] = 1
    return arr


def get_index_one_hot(index_name):
    arr = [0] * num_index
    arr[index_names.index(index_name)] = 1
    return arr


def get_rel_attr_one_hot(rel_name, filter_line):
    attr_list = rel_attr_list_dict[rel_name]

    med_vec, min_vec, max_vec = [0] * max_num_attr, [0] * max_num_attr, \
                                [0] * max_num_attr

    for idx, attr in enumerate(attr_list):
        if attr in filter_line:
            med_vec[idx] = postgres_tpch_attr_val_dict['med'][rel_name][idx]
            min_vec[idx] = postgres_tpch_attr_val_dict['min'][rel_name][idx]
            max_vec[idx] = postgres_tpch_attr_val_dict['max'][rel_name][idx]
    return min_vec + med_vec + max_vec


def get_scan_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Seq Scan'
    rel_vec = get_rel_one_hot(plan_dict['Relation Name'])
    try:
        rel_attr_vec = get_rel_attr_one_hot(plan_dict['Relation Name'],
                                            plan_dict['Filter'])
    except:
        try:
            rel_attr_vec = get_rel_attr_one_hot(plan_dict['Relation Name'],
                                                plan_dict['Recheck Cond'])
        except:
            if 'Filter' in plan_dict:
                print('************************* default *************************')
                print(plan_dict)
            rel_attr_vec = [0] * max_num_attr * 3

    return get_basics(plan_dict) + rel_vec + rel_attr_vec


def get_index_scan_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Index Scan'

    rel_vec = get_rel_one_hot(plan_dict['Relation Name'])
    index_vec = get_index_one_hot(plan_dict['Index Name'])

    try:
        rel_attr_vec = get_rel_attr_one_hot(plan_dict['Relation Name'],
                                            plan_dict['Index Cond'])
    except:
        if 'Index Cond' in plan_dict:
            print('********************* default rel_attr_vec *********************')
            print(plan_dict)
        rel_attr_vec = [0] * max_num_attr * 3

    res = get_basics(plan_dict) + rel_vec + rel_attr_vec + index_vec \
          + [1 if plan_dict['Scan Direction'] == 'Forward' else 0]

    return res


def get_bitmap_index_scan_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Bitmap Index Scan'
    index_vec = get_index_one_hot(plan_dict['Index Name'])

    return get_basics(plan_dict) + index_vec


def get_hash_input(plan_dict):
    return get_basics(plan_dict) + [plan_dict['Hash Buckets']]


def get_join_input(plan_dict):
    type_vec = [0] * len(join_types)
    type_vec[join_types.index(plan_dict['Join Type'].lower())] = 1
    par_rel_vec = [0] * len(parent_rel_types)
    if 'Parent Relationship' in plan_dict:
        par_rel_vec[parent_rel_types.index(plan_dict['Parent Relationship'].lower())] = 1
    return get_basics(plan_dict) + type_vec + par_rel_vec


def get_sort_key_input(plan_dict):
    kys = plan_dict['Sort Key']
    one_hot = [0] * (num_rel * max_num_attr)
    for key in kys:
        key = key.replace('(', ' ').replace(')', ' ')
        for subkey in key.split(" "):
            if subkey != ' ' and '.' in subkey:
                rel_name, attr_name = subkey.split(' ')[0].split('.')
                if rel_name in rel_names:
                    one_hot[rel_names.index(rel_name) * max_num_attr
                            + rel_attr_list_dict[rel_name].index(attr_name.lower())] = 1

    return one_hot


def get_sort_input(plan_dict):
    sort_meth = [0] * len(sort_algos)
    if 'Sort Method' in plan_dict:
        if "external" not in plan_dict['Sort Method'].lower():
            sort_meth[sort_algos.index(plan_dict['Sort Method'].lower())] = 1

    return get_basics(plan_dict) + get_sort_key_input(plan_dict) + sort_meth


def get_aggreg_input(plan_dict):
    strat_vec = [0] * len(aggreg_strats)
    strat_vec[aggreg_strats.index(plan_dict['Strategy'].lower())] = 1
    partial_mode_vec = [0] if plan_dict['Parallel Aware'] == 'false' else [1]
    return get_basics(plan_dict) + strat_vec + partial_mode_vec


tpch_input_functions = \
    {
        "Hash Join": get_join_input,
        "Merge Join": get_join_input,
        "Seq Scan": get_scan_input,
        "Index Scan": get_index_scan_input,
        "Index Only Scan": get_index_scan_input,
        "Bitmap Heap Scan": get_scan_input,
        "Bitmap Index Scan": get_bitmap_index_scan_input,
        "Sort": get_sort_input,
        "Hash": get_hash_input,
        "Aggregate": get_aggreg_input
    }


tpcc_dimensions = {
    "4.0;7.0;8.0": 18,
    "0.0;4.0;7.0;11.0": 24,
    "6.0": 6,
    "0.0;3.0;4.0;7.0;11.0": 30,
    "0.0;3.0;4.0;7.0": 24,
    "0.0;4.0;7.0": 18,
    "4.0;7.0;7.0;12.0": 24,
    "2.0;8.0": 44,
    "4.0;7.0;12.0": 18,
    "1.0;2.0;8.0": 50,
    "4.0;7.0;10.0": 18,
    "3.0;4.0;7.0;9.0": 24,
    "5.0;8.0": 44,
    "4.0;4.0;7.0;9.0;11.0": 30
}
