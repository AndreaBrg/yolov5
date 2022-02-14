# Crop function by Gianluca Dalmasso, GPL-3.0 license
'''
Crop images around their labelled items (it uses a YoLo labels format)

Usage:
    $ python crop_generator.py --images path/to/images/ --labels path/to/labels/ --output path/to/output/
    $ python crop_generator.py --images D:\\Documenti\\UNITO\\MAGISTRALE\\____TIROCINIO\\IMG\\IMG_REALI\\TEST_REAL_IMG\\7_eink\\images\\ --labels .\runs\detect\5_synth\5_synth_OBJ-DET\labels\ --output .\runs\detect\5_synth\5_synth_OBJ-DET\my_crops\ --xoffset 5 --yoffset 5
'''

from PIL import Image
import pandas as pd
import glob
import os
import argparse
import tqdm
import cv2
import numpy as np
#


# This will read all the data we need to work on and keep only the images that have their label file
def prepare_data_to_crop(data_path, labels_path, result_path):
    # If data paths do not exist we manage the exception outside
    if not (os.path.exists(data_path) and os.path.exists(labels_path)):
        return None
    # Read directories files
    images_paths = (glob.glob(data_path + "*.*"))
    labels_paths = (glob.glob(labels_path + "*.*"))
    # Create output directory if not exists
    if not os.path.exists(result_path):
        os.makedirs(result_path + "images/")
    # Keep only the images having their labels file
    images_ok = []
    for image_path in images_paths:
        print(image_path)
        image_path_name = image_path.split("\\")[-1].split(".")[0]
        print(image_path_name)
        image_entry = None
        for label_path in labels_paths:
            label_path_name = label_path.split("\\")[-1].split(".")[0]
            if image_path_name == label_path_name:
                image_entry = (image_path_name, image_path, label_path)
                break
        #
        if image_entry != None:
            images_ok.append(image_entry)
    print("Found", str(len(images_ok)), "images.")
    return images_ok

# This just read the label text and save into a pandas dataframe (YoLo format uses spaces in the csv)
def read_labels(labels_path):
    colnames=['Labels_ID', 'X_CENTER_NORM', 'Y_CENTER_NORM', 'WIDTH_NORM', 'HEIGHT_NORM'] 
    return  pd.read_csv(labels_path, names=colnames, header=None, delim_whitespace=True)

# This create a crop around every labelled item and save it as a new file
def generate_image_crops(image_path, image_labels_path, output_prefix_name, offsets, skip_w_rateo=2.0, skip_h_rateo=0.25):
    x_offset, y_offset = offsets
    # mi leggo le labels
    labels_data = read_labels(image_labels_path).to_numpy()
    #print(labels_data)
    crops = []
    skipped = 0
    with Image.open(image_path) as image_obj:
        img_converted = image_obj.convert('RGB')
        image_width, image_height = img_converted.size
        for label_data, cont in zip(labels_data, range(0, len(labels_data))):
            # Convert YoLo normalized data into item coordinates
            label_id, x_center_norm, y_center_norm, width_norm, height_norm = label_data
            label_width = width_norm * image_width
            label_height = height_norm * image_height
            xmin = (x_center_norm * image_width) - (label_width / 2)
            ymin = (y_center_norm * image_height) - (label_height / 2)
            # Apply an offset around the item
            x1 = xmin - x_offset
            x2 = xmin + label_width + x_offset
            y1 = ymin - y_offset
            y2 = ymin + label_height + y_offset
            # Padded crop dimensions
            label_padded_width = x2 - x1
            label_padded_height = y2 - y1
            ## Check padded width/height rateo, if less than skip_w_rateo it is a "bad image" so we skip it (squared, probably cut in half (horizontaly) but might be a miss)
            #if skip_w_rateo != -1 and label_padded_width / label_padded_height < skip_w_rateo:
            #    skipped += 1
            #    continue
            ## Check padded height/width rateo, if less than skip_h_rateo it is a "bad image" so we skip it (bad labelling, probably cut in half (vertically) but might be a miss)
            #if skip_h_rateo != -1 and label_padded_height / label_padded_width < skip_h_rateo:
            #    skipped += 1
            #    continue
            # Else save the crop
            coords = (x1, y1, x2, y2) #x1, y1, x2, y2
            crop_img = img_converted.crop(coords)
            crops.append((output_prefix_name + "-" + str(cont) + ".jpg", crop_img, label_id))
    return crops, skipped


# Wrapper function, read all the labelled data and then crops every image one at a time
def generate_crops(images_path, labels_path, result_path, offsets=(20, 20), crop_max_count=-1, crop_class_max_count=-1, skip_w_rateo=2.0, skip_h_rateo=0.25):
    print("Reading data...")
    data = prepare_data_to_crop(images_path, labels_path, result_path)
    if data == None:
        print("Missing input directories paths!")
        return
    # Else
    print("Cropping images...")
    cont = 0
    skipped = 0
    early_exit = False
    class_count = {}
    #print(data)
    with open(result_path + "labels.txt", "w") as labels_results:
        for (file_name_prefix, image_path, label_path), i in zip (data, tqdm.tqdm(range(len(data)))):
            print(file_name_prefix, image_path, label_path)
            crops, sub_skipped = generate_image_crops(image_path, label_path, file_name_prefix, offsets, skip_w_rateo=skip_w_rateo, skip_h_rateo=skip_h_rateo)
            skipped += sub_skipped
            for crop_name, crop_img, crop_label in crops:
                #print(f"\ncrop name: {crop_name}")
                # If the label is already present in the dictionary
                if crop_label in class_count:
                    # Skip if we reached the max count (and if there is a max count per class)
                    if crop_class_max_count != -1 and class_count[crop_label] == crop_class_max_count:
                        continue
                # Otherwise insert the label entry with 0 count
                else:
                    class_count[crop_label] = 0
                crop_path = result_path + "images/" + crop_name
                # Save crop image
                crop_img.save(crop_path)
                # Save crop label
                labels_results.write(crop_name + " " + str(crop_label) + "\r")
                # Increase current class number of crops
                class_count[crop_label] += 1
                cont += 1
                # If we specified a maximum number of crops, and we have already made enough crops then we can stop
                if crop_max_count != -1 and cont >= crop_max_count:
                    early_exit=True
                    break
            # We reached max number of needed crops, we can exit
            if early_exit:
                break
    print("Created", cont, "cropped images.")
    print("Skipped (bad rateo)", skipped, "crops")
    print("Results saved in:", result_path)


def parse_main_args():
    parser = argparse.ArgumentParser(description='Crops every images in their labelled elements (images without labels will not be considered)')
    parser.add_argument('--images', type=str, required=True, help='Path to images directory')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels directory')
    parser.add_argument('--output', type=str, required=True, help='Path for output results directory')
    parser.add_argument('--xoffset', type=int, default=20, required=False, help='Crop X offset(px)')
    parser.add_argument('--yoffset', type=int, default=20, required=False, help='Crop Y offset(px)')
    parser.add_argument('--skip-wrateo', type=float, default=2.0, required=False, help="Don't save crop if its padded rateo (width/height) is less than skip_wrateo (-1 if no use)")
    parser.add_argument('--skip-hrateo', type=float, default=0.25, required=False, help="Don't save crop if its padded rateo (height/width) is less than skip_hrateo (-1 if no use)")
    parser.add_argument('--crop-cap', type=int, default=-1, required=False, help='Maximum number of crops generated (-1 if no cap)')
    parser.add_argument('--class-cap', type=int, default=-1, required=False, help='Maximum number of crops generated per class (-1 if no cap)')
    return parser.parse_args()

def main(args):
    images_path, labels_path, result_path, offsetX, offsetY, crop_cap, crop_class_cap, skip_wrateo, skip_hrateo = args.images, args.labels, args.output, args.xoffset, args.yoffset, args.crop_cap, args.class_cap, args.skip_wrateo, args.skip_hrateo
    generate_crops(images_path, labels_path, result_path, offsets=(offsetX, offsetY), crop_max_count=crop_cap, crop_class_max_count=crop_class_cap, skip_w_rateo=skip_wrateo, skip_h_rateo=skip_hrateo)

if __name__ == "__main__":
    #args = parse_main_args()
    #main(args)
    im_name = "_01.png"
    img = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)
  
    # Specify the kernel size.
    # The greater the size, the more the motion.
    kernel_size = 8
    
    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))
    
    # Create a copy of the same for creating the horizontal kernel.
    kernel_h = np.copy(kernel_v)
    
    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    
    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size
    
    # Apply the vertical kernel.
    vertical_mb = cv2.filter2D(img, -1, kernel_v)
    
    # Apply the horizontal kernel.
    horizonal_mb = cv2.filter2D(img, -1, kernel_h)
    
    # Save the outputs.
    cv2.imwrite(f'{im_name.split(".")[0]}_vertical{kernel_size}.png', vertical_mb)
    print("Verical saved")
    cv2.imwrite(f'{im_name.split(".")[0]}_horizontal{kernel_size}.png', horizonal_mb)
    print("\nHorizontal saved")