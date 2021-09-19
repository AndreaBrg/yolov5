from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from PIL import Image, ImageDraw

import val
from utils.datasets import create_dataloader
from utils.general import colorstr
from utils.general import xywh2xyxy
from utils.plots import Colors


def get_test_loader(data_dict, imgsz, batch_size, WORLD_SIZE, gs, single_cls, hyp, noval, opt, workers):
    return create_dataloader(data_dict['synthetic_val'], imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                             hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
                             workers=workers, pad=0.5, prefix=colorstr('synthetic_val: '))[0]


def log_extra_val(data_dict, batch_size, WORLD_SIZE, imgsz, ema, single_cls, dataloader, save_dir, is_coco, final_epoch,
                  nc, plots, callbacks, compute_loss, tb, wandb, epoch):
    keys = ['synth/precision', 'synth/recall', 'synth/mAP_0.5', 'synth/mAP_0.5:0.95',
            'synth/box_loss', 'synth/obj_loss', 'synth/cls_loss']
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
                            compute_loss=compute_loss)
    
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
