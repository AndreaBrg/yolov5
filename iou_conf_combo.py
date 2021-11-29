import numpy as np

import val
import time
import datetime
import json

from utils.general import check_yaml, check_dataset

import os

def get_best_combo(data):
    name_ds = os.path.split(data[my_options.task])[1]
    
    path_res = f"runs/val/save_results/{name_ds}"
    max_recall_all = 0.0
    max_recall_price = 0.0
    max_recall_name = 0.0
    max_recall_tag = 0.0
    best_combo_all = {}
    best_combo_price = {}
    best_combo_name = {}
    best_combo_tag = {}
    
    print("Looking for the best combinations of thresholds ...\n")
    # Navigate through all the directories
    for child in os.listdir(path_res):
        if not child == "best_results.json":
            for c in os.listdir(f"{path_res}/{child}"):
                items = c.split('.')
                if(items[len(items)-1] == 'json' ):
                    with open(f"{path_res}/{child}/{c}") as json_file:
                        data_dict = json.load(json_file)
                        for p in data_dict["results"]:
                            if p["class"] == "all":
                                if(float(p["recall"]) > max_recall_all):
                                    max_recall_all = float(p["recall"])
                                    best_combo_all["class"] = "all"
                                    best_combo_all["iou"] = data_dict["iou"]
                                    best_combo_all["conf"] = data_dict["conf"]
                                    best_combo_all["recall"] = max_recall_all
                            elif p["class"] == "price":
                                if(float(p["recall"]) > max_recall_price):
                                    max_recall_price = float(p["recall"])
                                    best_combo_price["class"] = "price"
                                    best_combo_price["iou"] = data_dict["iou"]
                                    best_combo_price["conf"] = data_dict["conf"]
                                    best_combo_price["recall"] = max_recall_price
                            elif p["class"] == "name":
                                if(float(p["recall"]) > max_recall_name):
                                    max_recall_name = float(p["recall"])
                                    best_combo_name["class"] = "name"
                                    best_combo_name["iou"] = data_dict["iou"]
                                    best_combo_name["conf"] = data_dict["conf"]
                                    best_combo_name["recall"] = max_recall_name
                            else:
                                if(float(p["recall"]) > max_recall_tag):
                                    max_recall_tag = float(p["recall"])
                                    best_combo_tag["class"] = "tag"
                                    best_combo_tag["iou"] = data_dict["iou"]
                                    best_combo_tag["conf"] = data_dict["conf"]
                                    best_combo_tag["recall"] = max_recall_tag
                      
    print(f"\t\tBest combinations to maximize RECALL:\n\t\t{best_combo_all}\n\t\t{best_combo_price}\n\t\t{best_combo_name}\n\t\t{best_combo_tag}\n\n")
    
    # Save best results in JSON file
    if not os.path.exists(f"{path_res}/best_results.json"):
        open(f"{path_res}/best_results.json", "x")
        
    # Create dict with all best results
    root_dict = {"best_results": []}
    root_dict["best_results"].append(best_combo_all)
    root_dict["best_results"].append(best_combo_price)
    root_dict["best_results"].append(best_combo_name)
    root_dict["best_results"].append(best_combo_tag)
    
    # Create json object to be written in the file
    json_obj = json.dumps(root_dict)
    
    with open(f"{path_res}/best_results.json", "w") as json_file:
        json_file.write(json_obj)
        
    print(f"\t\tData SAVED in '{path_res}/best_results.json'\n\n")

if __name__ == "__main__":
    '''
    my_conf_thr = 0.0
    my_iou_thr = 0.0
    
    # Get arguments
    my_options = val.parse_opt()
    # Force task to "test"
    my_options.task = "test"
    my_options.save_results = True
    
    # Save DS name
    data = check_yaml(my_options.data)
    data = check_dataset(data)
    
    start_time = time.time()
    
    i = 0
    # conf iou in [0.0, 1.0] -> step 0.1 --> 121 run (11 x 11)
    for i_conf in np.arange(0.0,1.1,0.1):
        my_options.conf_thres = round(i_conf)
        for i_iou in np.arange(0.0,1.1,0.1):
            print(f"\n\n\t***CONF: {i_conf} - IOU: {i_iou}***\n\n")
            my_options.iou_thres = round(i_iou,1)
            # Edit directory name
            my_options.name = f"conf_{i_conf}-iou_{i_iou}"
            # Run on the current combination of thresholds
            val.main(my_options)
            i+=1
    
    print(f"\n\n\t**** Total time of execution save date of {i} combinations of conf e iou: %s. ****\n\n" %
          str(datetime.timedelta(seconds=(time.time() - start_time))))
    '''    
    
    # Get arguments
    my_options = val.parse_opt()
    # Force task to "test"
    my_options.task = "test"
    my_options.save_results = True
    
    # Save DS name
    data = check_yaml(my_options.data)
    data = check_dataset(data)
    
    get_best_combo(data)