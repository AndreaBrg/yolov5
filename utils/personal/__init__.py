import val
from utils.datasets import create_dataloader
from utils.general import colorstr


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
