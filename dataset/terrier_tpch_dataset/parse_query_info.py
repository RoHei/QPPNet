

import terrier_query_info as tqi
import json

from dataset.constants import terrier_pname_group_dict

input_len_dict = dict()
mem_map = getattr(tqi, "MEM_ADJUST_MAP")
for pname in terrier_pname_group_dict:
    plst = getattr(tqi, pname.split("tpch")[1].upper())
    input_len_dict[pname] = len(plst) * 4 + (1 if pname in mem_map else 0)
    if int(pname.split("_")[-1][1:]) > 1:
        input_len_dict[pname] += 32

with open("input_dim_dict.json", "w+") as f:
    json.dump(input_len_dict, f, indent=4)
