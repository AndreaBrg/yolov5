import numpy as np

import val
import time
import datetime


if __name__ == "__main__":
    my_conf_thr = 0.0
    my_iou_thr = 0.0
    
    # Get arguments
    my_options = val.parse_opt()
    # Force task to "test"
    my_options.task = "test"
    print(f"MY_ {my_options}")
    my_options.save_results = True
    print(f"\nMY_ {my_options}")
    
    start_time = time.time()
    
    i = 0
    # conf iou in [0.0, 1.0] -> step 0.1 --> 121 run (11 x 11)
    for i_conf in np.arange(0.0,1.1,0.1):
        my_options.conf_thres = np.round(i_conf,1)
        for i_iou in np.arange(0.0,1.1,0.1):
            print(f"\n\n\t***CONF: {i_conf} - IOU: {i_iou}***\n\n")
            my_options.iou_thres = np.round(i_iou,1)
            # Edit directory name
            my_options.name = f"conf_{i_conf}-iou_{i_iou}"
            # Run on the current combination of thresholds
            val.main(my_options)
            i+=1
    
    print(f"\n\nt**** Total time of execution save date of {i} combinations of conf e iou: %s. ****\n\n" %
          str(datetime.timedelta(seconds=(time.time() - start_time))))