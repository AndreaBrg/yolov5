from subprocess import IDLE_PRIORITY_CLASS
import numpy as np

import val


if __name__ == "__main__":
    my_conf_thr = 0.0
    my_iou_thr = 0.0
    
    # Get arguments
    my_options = val.parse_opt()
    # Force task to "test"
    my_options.task = "test"
    my_options.save_results = True

    
    
    i = 1
    # conf iou in [0.0, 1.0] -> step 0.1 --> 121 run
    for i_conf in np.arange(0.0,0.2,0.1):
        my_options.conf_thres = i_conf
        for i_iou in np.arange(0.0,0.2,0.1):
            my_options.iou_thres = i_iou
            
            # Edit directory name
            my_options.name = f"conf_{i_conf}-iou_{i_iou}"
            # Run on the current combination of thresholds
            val.main(my_options)
