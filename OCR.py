"""
OCR + OBJ-DETECTION detect and combination

Usage:
    $ python path/to/OCR.py --nnparams path/to/nnparams/yaml --save-txt --save-crop --name det
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import pathlib
import sys
from pathlib import Path, WindowsPath

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, is_ascii, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh, check_yaml
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

# My new import
import glob
import yaml
import os
import pprint as pp
import random
import string

from PIL import Image, ImageDraw, ImageFont

save_dir = 'D:\\Documenti\\UNITO\\MAGISTRALE\\____TIROCINIO\\IMG\\IMG_WITH_BOXES\\OCR_OBJ_DET\\combo'
my_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(string.ascii_uppercase))]

cls_OCR = []
cls_OBJ_DET = []

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnparams', type=str, help='config yaml with data from OCR and obj-det NN')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--local-test', action='store_true', default=False, help='run local test (no inference, fixed paths with imgs already detected')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main(opt): #,nnparams):
    check_requirements(exclude=('tensorboard', 'thop'))
    
    opt.nnparams = check_yaml(opt.nnparams)
    assert len(opt.nnparams), '--nnparams must be specified (yaml file)'
    
    # Save NN parameters and path into dict
    with open(opt.nnparams) as f:
        nnparams = yaml.safe_load(f)  # load params dict
        
    cls_OCR = nnparams["ocr_classes"]
    cls_OBJ_DET = nnparams["obj_det_classes"]
    source_imgs = nnparams["source"]
    res_OCR = nnparams["ocr_res"]
    res_OBJ_DET = nnparams["obj_det_res"]
    
    opt.source = source_imgs
    
    #print(f"\nPath imgs: {nnparams['path_img']}")
    #print(f"OCR weights: {nnparams['ocr_w']}")
    #print(f"OBJ-DET weights: {nnparams['obj_det_w']}")
    
    paths = []
    
    i = 0 # 0 = OCR, 1 = OBJ-DET
    name_radix = opt.name
    for w in [nnparams['ocr_w'],nnparams['obj_det_w']]:
        opt.weights = w
        opt.project = f"runs/detect/{name_radix}"
        opt.imgsz = [res_OCR] if i == 0 else [res_OBJ_DET] # [res_OCR,res_OCR] if i == 0 else [res_OBJ_DET, res_OBJ_DET]
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        opt.name = name_radix + "_OCR" if i == 0 else name_radix + "_OBJ-DET"
        prj = Path(opt.project)
        (prj/'OCR_images').mkdir(parents=True, exist_ok=True)
        i += 1
        print(f"\nDETECTING with '{w}' CHECKPOINTS/WEIGHTS")
        paths.append(run(**vars(opt)))
    
    return paths, cls_OCR,cls_OBJ_DET
    
@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        save_txt=False,  # save results to *.txt
        save_crop=False,  # save cropped prediction boxes
        local_test=False,  # run local test, no inference
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_conf=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference,
        nnparams="", # yaml file with params to combine OCR and OBJ-DET
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'images' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / 'images' / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            if len(det):
                #det[..., 0] *= im0[1]  # x
                #det[..., 1] *= im0[0]  # y
                #det[..., 2] *= im0[1]  # w
                #det[..., 3] *= im0[0]
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            print(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
        
    return save_dir # Return path where all detected data is stored

# box = [center-x, center-y, width, height, str_class]
path_img = "D:\\Documenti\\UNITO\\MAGISTRALE\\____TIROCINIO\\IMG\\IMG_REALI\\TEST_REAL_IMG\\lbl_combo\\images\\0a807922-P_20210709_172746_vHDR_Auto.jpg"
def check_rect_inside(big_rect, small_rect,i, area, img):
    # Conditions to check if small is inside big rectangle
    # https://math.stackexchange.com/questions/3086589/determine-if-a-rectangle-is-inside-overlaps-doesnt-overlaps-another-rectangle
    #img = Image.open(path_img)
    
    if  img.size[0] * (small_rect[0] + small_rect[2]/2) <= img.size[0] * (big_rect[0] + big_rect[2]/2) and\
        img.size[0] * (small_rect[0] - small_rect[2]/2) >= img.size[0] * (big_rect[0] - big_rect[2]/2) and\
        img.size[1] * (small_rect[1] + small_rect[3]/2) <= img.size[1] * (big_rect[1] + big_rect[3]/2) and\
        img.size[1] * (small_rect[1] - small_rect[3]/2) >= img.size[1] * (big_rect[1] - big_rect[3]/2):
        #print(f"\t\tCHAR box INSIDE NAME box\n")
        
        #if area == "NAME":
        
        #draw_box_on_img(img,small_rect)
        #draw_box_on_img(img,big_rect)
        #img.save(f"{save_dir}/inside/{area}/{str(i).zfill(3)}.png", "PNG")
        #print(f"IMG '{str(i).zfill(3)}.png' saved in {save_dir}\inside\{area}\n")
        
        
        return True
    
    #if area == "NAME":

    #draw_box_on_img(img,small_rect)
    #draw_box_on_img(img,big_rect)
    #img.save(f"{save_dir}/no_inside/{area}/{str(i).zfill(3)}.png", "PNG")
    #print(f"IMG '{str(i).zfill(3)}.png' saved in {save_dir}\\no_inside\{area}\n")

    return False

# Draw box in a image
def draw_box_on_img(img, box):
    print("Drawing box on the image...")
    box = [
        img.size[0] * (box[0] - box[2] / 2),
        img.size[1] * (box[1] - box[3] / 2),
        img.size[0] * (box[0] + box[2] / 2),
        img.size[1] * (box[1] + box[3] / 2)
    ]
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, outline=tuple(my_colors[random.randint(0,23)]), width=2)
    
# Compose string from ordered character based on x position
def compose_string(lst_name, lst_price):
    names = []
    prices = []
    # Name of the product
    for boxes_chars in lst_name:
        prod_name = ""
        for b in boxes_chars["boxes"]:
            prod_name += b["char"]
        names.append({"name": prod_name, "img":boxes_chars["img"]})
    # Price of the product
    for boxes_chars in lst_price:
        price = ""
        for b in boxes_chars["boxes"]:
            price += b["char"]
        prices.append({"price": price, "img":boxes_chars["img"]})
    
    return names, prices

if __name__ == '__main__':
    opt = parse_opt()
    if not (opt.local_test):
        print("\nRUN detection\n")
        # Run detect
        paths, cls_OCR, cls_OBJ_DET = main(opt) #,nnparams)
        root_dir = os.path.join(os.getcwd(), os.path.split(str(paths[0]))[:-1][0])
    else:
        # LOCAL TEST
        print("LOCAL TEST: using image already detected\n")
        opt.nnparams = check_yaml(opt.nnparams)
        assert len(opt.nnparams), '--nnparams must be specified (yaml file)'

        # Save NN parameters and path into dict
        with open(opt.nnparams) as f:
            nnparams = yaml.safe_load(f)  # load params dict

        cls_OCR = nnparams["ocr_classes"]
        cls_OBJ_DET = nnparams["obj_det_classes"]
        p1 = WindowsPath("runs/detect/5_synth/5_synth_OCR")
        p2 = WindowsPath("runs/detect/5_synth/5_synth_OBJ-DET")
        paths = [p1,p2]

        s = str(p1)
        root_dir = os.path.join(os.getcwd(), os.path.split(s)[:-1][0])
    # Loop on each dir
    i = 0 # 0 = OCR , 1 = OBJ-DET
    boxes_OCR = []
    boxes_OBJ_DET = []
    for dir in paths:
        path = os.path.join(os.getcwd(), str(dir), "labels", "*")
        idx = -1
        print(path)
        for label in glob.glob(str(path)):
            idx +=1
            
            with open(label, "r") as lbl_file:
                boxes_in_file = []
                for row in lbl_file:
                    curr_box = row.strip(" \n").split(" ")[1:]
                    curr_box = [float(b) for b in curr_box]
                    cl = int(row.strip(" \n").split(" ")[0])
                    curr_box.append(cl)
                    
                    if not(i == 1 and cl == 2): # OBJ-DET and CL tag (not useful for our task)
                        boxes_in_file.append(curr_box)
            # Save boxes
            if i == 0: # OCR
                boxes_OCR.append({"boxes":boxes_in_file})
            else:
                boxes_OBJ_DET.append({"boxes":boxes_in_file})
        
        path = os.path.join(os.getcwd(), str(dir),"images", "*")
        curr_img = 0
        for img in glob.glob(str(path)):
            if i == 0:
                boxes_OCR[curr_img]["img"] = img
            else:
                boxes_OBJ_DET[curr_img]["img"] = img
            curr_img += 1
        
        i += 1
    
    print()
    #pp.pprint(f"BOX OCR({len(boxes_OCR)}):\n{boxes_OCR}")
    print(f"LEN IMGS WITH BOXES OBJ-DET: {len(boxes_OBJ_DET)}")
    pp.pprint(boxes_OBJ_DET)
    print(f"LEN IMGS WITH BOXES BOX OCR {len(boxes_OCR)}")
    #pp.pprint(boxes_OCR)
    
    # Save original dict of obj_det
    original_boxes_OBJ_DET = boxes_OBJ_DET.copy()
    
    i=0
    j=0
    i_img = 0
    i_curr_img = 0
    
    chars_in_name = [] # lst of boxes inside NAME area
    chars_in_price = [] # lst of boxes inside PRICE area
    for item_OBJ_DET in boxes_OBJ_DET: # loop over all dict (images with their AREA [name,price] boxes)
        # TEST WITH IMG WITH NO BOXES
        path_img =  os.path.join(opt.source,item_OBJ_DET["img"].split('\\')[-1])
        curr_img = Image.open(path_img)
        # path_img = item_OBJ_DET["img"] # img with already box and classes from detection
        boxes_OBJ_DET = item_OBJ_DET["boxes"]
        lst_chars_name = []
        lst_chars_price = []
        for curr_box_OBJ_DET in boxes_OBJ_DET: # loop over single box in OBJ_DET (name and price)
            img_clear = curr_img.copy()
            inside = 0
            curr_boxes_OCR = boxes_OCR[i_curr_img]["boxes"] # lst of chars boxes of current img              
            curr_area = "PRICE" if curr_box_OBJ_DET[-1] == 0 else "NAME"
            for char_box in curr_boxes_OCR: # loop over all chars boxes of current imgs
                if not (char_box in chars_in_name) and not (char_box in chars_in_price): # check if they're already saved
                    # Set padding
                    padding_w = 0.005 # if curr_area == "NAME" else 0.001
                    padding_h = 0.03 if curr_area == "NAME" else 0.005
                    curr_box_OBJ_DET[2] += curr_box_OBJ_DET[2] * padding_w # Width
                    curr_box_OBJ_DET[3] += curr_box_OBJ_DET[3] * padding_h # Height
                    # Check if char box is inside the current obj_det area
                    box_inside = check_rect_inside(curr_box_OBJ_DET, char_box, i_img,curr_area, img_clear)
                    i_img += 1
                    inside += 1 if  box_inside else 0                    
                    if box_inside: # char inside AREA
                        if curr_area == "NAME": # NAME AREA
                            lst_chars_name.append({"box":char_box, "char": cls_OCR[char_box[-1]]}) # Save box of the char and the char itself
                            #chars_in_name.append({"boxes_inside": char_box})
                        elif curr_area == "PRICE": # PRICE AREA
                            lst_chars_price.append({"box":char_box, "char": cls_OCR[char_box[-1]]}) # Save box of the char and the char itself
                            #chars_in_price.append({"boxes_inside": char_box})

            
            #if i_curr_img == 0 and curr_area == "NAME":
            #    draw_box_on_img(curr_img,curr_box_OBJ_DET)
            #    draw = ImageDraw.Draw(curr_img)
            #    x = curr_img.size[0] * (curr_box_OBJ_DET[0] + curr_box_OBJ_DET[2]/2)
            #    y = curr_img.size[1] * (curr_box_OBJ_DET[1] - curr_box_OBJ_DET[3])
            #    h = curr_box_OBJ_DET[3]/2
            #    fnt = ImageFont.truetype("C:\Windows\Fonts\ARIAL.TTF", 40)
            #    draw.text((x,y), "NOME PROD", fill=tuple(my_colors[random.randint(0,23)]), font=fnt)
            #    curr_img.save(f"D:/Documenti/UNITO/MAGISTRALE/____TIROCINIO/IMG/IMG_WITH_BOXES/OCR_OBJ_DET/combo/{str(i_curr_img).zfill(3)}.png", "PNG")
            print(f"{inside} out of {len(boxes_OCR[i_curr_img]['boxes'])} are inside {curr_area} AREA")
                
            # Draw NAME or PRICE big BOX on the image
            #draw_box_on_img(img_clear,curr_box_OBJ_DET)
            #img_clear.save(f"{save_dir}/inside/{curr_area}/{str(i_curr_img).zfill(3)}.png", "PNG")
            #print(f"IMG '{str(i).zfill(3)}.png' saved in {save_dir}\inside\{curr_area}\n")
            
        
        chars_in_price.append({"img":path_img, "boxes": lst_chars_price})
        chars_in_name.append({"img": path_img, "boxes": lst_chars_name})

        #print(f"\nCHARS IN PRICE")
        #pp.pprint(chars_in_price)
        #print(f"\nCHARS IN NAME")
        #pp.pprint(chars_in_name)
        
        i_curr_img += 1    
    
    #print(f"\nNAMES BEFORE")
    #pp.pprint(chars_in_name)
    # Order boxes for each element {img, boxes:{box[],char}} for x position of each box
    for boxes_char in chars_in_name:
        boxes_char["boxes"].sort(key=lambda x: x["box"][0], reverse=False)
    #print(f"\nNAMES AFTER")
    #pp.pprint(chars_in_name)
    
    #print(f"\nPRICES BEFORE")
    #pp.pprint(chars_in_price)
    for boxes_char in chars_in_price:
        boxes_char["boxes"].sort(key=lambda x: x["box"][0], reverse=False)
    #print(f"\nPRICES AFTER")
    #pp.pprint(chars_in_price)
   
    lst_names, lst_prices = compose_string(chars_in_name, chars_in_price)
    
    print("\nPRODUCT NAMES\n")
    pp.pprint(lst_names)
    print("\nPRICES\n")
    pp.pprint(lst_prices)
    
    # Save images with name and price composed by code
    k = 0
    for item in original_boxes_OBJ_DET:
        im_name = str(os.path.split(item["img"])[-1])
        image = Image.open(os.path.join(opt.source,im_name))
        draw = ImageDraw.Draw(image)
        print(str(os.path.join(opt.source,im_name)))
        for box in item["boxes"]:
            # Draw box
            draw_box_on_img(image,box)
            # Draw text on the top-right corner
            x = image.size[0] * (box[0] + box[2]/2)
            y = image.size[1] * (box[1] - box[3]/2)
            #h = box[3]/2
            fnt = ImageFont.truetype("C:\\Windows\\Fonts\\ARIAL.TTF", 70)
            txt = lst_names[k]["name"] if box[-1] == 1 else lst_prices[k]["price"]
            print(f"Writing text on the image...")
            draw.text((x,y), txt, fill=(100,100,100), font=fnt)
            # TO DO - DRAW RECT FILLED AROUND TEXT
            #w_font, h_font = fnt.getsize(txt)
            #print(w_font, h_font)
            #w_font /= 2
            #h_font /= 2
            #b_font = [
            #    box[0],
            #    box[1] - h_font,
            #    box[0] + w_font,
            #    box[1] + h_font
            #    #image.size[0] * (box[0] + w_font / 2),
            #    #image.size[1] * (box[1] + h_font / 2)
            #]
            ##b_font = # [box[0], box[1] - h_font, box[0] + w_font + 1, box[1] + 1]#[x/image.size[0], y/image.size[1],w_font/2,h_font/2]
            #draw_box_on_img(image, b_font)
            #draw.rectangle([box[0], box[1] - h_font, box[0] + w_font + 1, box[1] + 1], fill=tuple(my_colors[random.randint(0,23)]))
        k += 1
        image.save(f"{root_dir}/OCR_images/{im_name.split('.')[0]}.png", "PNG")
    
    # Save data of names and prices
    with open(os.path.join(root_dir,"names.txt"),"w") as f:
        for n in lst_names:
            f.write(f"{n}\n")
        print(f"\nNames saved in {f.name}")
            
    with open(os.path.join(root_dir,"prices.txt"),"w") as f:
        for n in lst_prices:
            f.write(f"{n}\n")
        print(f"Prices saved in {f.name}")