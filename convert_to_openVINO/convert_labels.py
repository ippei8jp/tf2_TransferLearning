# openVINO用ラベルマップのコンバート

import sys
import os
import object_detection.utils.label_map_util as label_util

INPUT_LABELS_FILE  = sys.argv[1]
OUTPUT_LABELS_FILE = sys.argv[2]

category_index = label_util.create_category_index_from_labelmap(INPUT_LABELS_FILE)

with open(OUTPUT_LABELS_FILE, mode='w') as f:
    for i in range(len(category_index)+1) :
        try:
            name = category_index[i]["name"]
        except:
            name = str(i)
        
        # print(name)
        f.write(name + '\n')


