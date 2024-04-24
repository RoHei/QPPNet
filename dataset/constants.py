
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
                 'Hash Join': 11 + 32 * 2, 'Merge Join': 11 + 32 * 2,
                 'Aggregate': 7 + 32, 'Nested Loop': 32 * 2 + 3, 'Limit': 32 + 3,
                 'Subquery Scan': 32 + 3,
                 'Materialize': 32 + 3, 'Gather Merge': 32 + 3, 'Gather': 32 + 3}

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