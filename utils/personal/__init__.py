import os
import random
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw
from skimage import img_as_float

import val
from utils.datasets import create_dataloader
from utils.general import colorstr
from utils.general import xywh2xyxy
from utils.plots import Colors


def get_extra_loader(data_dict, imgsz, batch_size, WORLD_SIZE, gs, single_cls, hyp, noval, opt, workers):
    return create_dataloader(data_dict['val'], imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                             augment=True, hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
                             workers=workers, pad=0.5, prefix=colorstr('val: '))


def log_extra_val(data_dict, batch_size, WORLD_SIZE, imgsz, ema, single_cls, dataloader, save_dir, is_coco, final_epoch,
                  nc, plots, callbacks, compute_loss, tb, wandb, epoch, conf_thres, iou_thres):
    print("Extra log")
    keys = ['val/precision', 'val/recall', 'val/mAP_0.5', 'val/mAP_0.5:0.95',
            'val/box_loss', 'val/obj_loss', 'val/cls_loss']
    results, _, _ = val.run(data_dict,
                            batch_size=batch_size // WORLD_SIZE * 2,
                            imgsz=imgsz,
                            model=ema.ema,
                            single_cls=single_cls,
                            dataloader=dataloader,
                            save_dir=save_dir,
                            save_json=is_coco and final_epoch,
                            verbose=nc < 50 and final_epoch,
                            plots=plots and final_epoch,
                            callbacks=callbacks,
                            compute_loss=compute_loss,
                            conf_thres=conf_thres,
                            iou_thres=iou_thres)
    
    x = {k: v for k, v in zip(keys, results)}
    if tb:
        for k, v in x.items():
            tb.add_scalar(k, v, epoch)
    
    if wandb:
        wandb.log(x)


def plot_extra_labels(labels, dataset, names=(), save_dir=Path('')):
    colors = Colors()
    
    # plot dataset labels
    print('Plotting labels... ')
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])
    
    # seaborn correlogram
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / f'{dataset}_labels_correlogram.jpg', dpi=200)
    plt.close()
    
    # matplotlib labels
    matplotlib.use('svg')  # faster
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    # [y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)]  # update colors bug #3195
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)
    
    # rectangles
    labels[:, 1:3] = 0.5  # center
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis('off')
    
    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)
    
    plt.savefig(save_dir / f'{dataset}_labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()


def overlay_transparent(background_img, img_to_overlay, x=0, y=0):
    composite_img = background_img.copy()
    
    # Random flips 50% independent chance
    if random.random() < 0.5:
        composite_img = cv2.flip(composite_img, 0)
    if random.random() < 0.5:
        composite_img = cv2.flip(composite_img, 1)
    
    # Extract the alpha mask of the RGBA image, convert to RGB
    b, g, r, a = cv2.split(img_to_overlay)
    overlay_color = cv2.merge((b, g, r))
    
    # Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a, 5)
    
    h, w, _ = overlay_color.shape
    roi = composite_img[y:y + h, x:x + w]
    
    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    
    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)
    
    # Update the original image with our new ROI
    composite_img[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)
    
    return composite_img


def add_background(image, background):
    image, background = match_hist(image, background)
    background = cv2.resize(background.copy(), (image.shape[1], image.shape[0]))
    
    return overlay_transparent(background, image)


def match_hist(image, background):
    # Save original alpha layer
    image_alpha = image[:, :, -1]
    
    # Cut alpha layer
    image = np.asarray(image)[:, :, :3]
    
    # BGR -> HSV
    image = img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    background = img_as_float(cv2.cvtColor(background, cv2.COLOR_BGR2HSV))
    
    # Histogram matching on V channel
    image[:, :, 2] = np.where(image_alpha == 0, np.full(image[:, :, 2].shape, -1), image[:, :, 2])
    image[:, :, 2] = match_masked_histogram(image[:, :, 2], background[:, :, 2])
    
    # HSV -> BGR
    image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    background = cv2.cvtColor((background * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Re-add the previously memorized alpha layer
    image = np.concatenate((image, np.expand_dims(image_alpha, axis=2)), axis=2)
    
    return image, background


def match_masked_histogram(source, template):
    start = perf_counter()
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    start = perf_counter()
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)
    
    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts[1:]) / source[source != -1].size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size
    
    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    interp_a_values = np.insert(interp_a_values, 0, 0)
    matched = interp_a_values[src_unique_indices]
    
    return matched.reshape(source.shape)


def match_cumulative_cdf(source, template):
    _, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                  return_inverse=True,
                                                  return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)
    
    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size
    
    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)


def show_image(name, image):
    plt.imshow(image)
    plt.title(name)
    plt.show()


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


if __name__ == '__main__':
    matplotlib.use("module://backend_interagg")
    image = cv2.imread("C:/Users/Andrea/Desktop/000000000086.jpg", -1)
    image2 = cv2.imread("C:/Users/Andrea/Desktop/000000000086.jpg")
    print()
