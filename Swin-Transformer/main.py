import os
import time
import json
import random
import argparse
import datetime
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, AsymmetricLossSingleLabel
from timm.utils import accuracy, AverageMeter
from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message="Importing from timm.layers is deprecated")

warnings.filterwarnings("ignore", message="To use FusedLAMB or FusedAdam, please install apex.")

PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs='+',)
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, full: cache all data, part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained', help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, required=False,
                            help='(disabled) local rank for DistributedDataParallel')
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    print("[DEBUG] final NUM_CLASSES =", config.MODEL.NUM_CLASSES)
    return args, config



def main(config):
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn = build_loader(config)
    
    if config.LOSS.CLASS_WEIGHTS is None:
        logger.info("Computing class weights automatically from training dataset...")
        targets = np.array([label for _, label in dataset_train.samples])
        unique_classes = np.unique(targets)
        logger.info(f"Unique classes in training set: {unique_classes.tolist()}")
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=targets)
        config.defrost()
        config.LOSS.CLASS_WEIGHTS = class_weights.tolist()
        config.freeze()

    with open(os.path.join(config.OUTPUT, "class_to_idx.json"), "w") as f:
        json.dump(dataset_train.class_to_idx, f)
    
    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))


    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model, logger)
    if hasattr(model, 'head'):
        in_features = model.head.in_features
        logger.info(f"BEFORE REPLACE: config.MODEL.NUM_CLASSES = {config.MODEL.NUM_CLASSES}")
        model.head = torch.nn.Linear(in_features, 15).cuda()
        config.defrost()
        config.MODEL.NUM_CLASSES = 15
        config.freeze()
        logger.info(f"Final model.head replaced: {in_features} → {config.MODEL.NUM_CLASSES}")
    else:
        raise RuntimeError("model has no attribute 'head', cannot replace classifier head!")

    logger.info(f"model.head final shape: {model.head}")
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    logger.info(f"🔥 model.head = {model.head}")
    model.cuda()
    
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    
    class_weights_arr = np.array(config.LOSS.CLASS_WEIGHTS)
    weight_std = np.std(class_weights_arr)
    logger.info(f"Provided class weights: {config.LOSS.CLASS_WEIGHTS}, std: {weight_std:.5f}")
    if weight_std < 1e-3:
        logger.info("Dataset is balanced based on class weights. Using standard CrossEntropyLoss.")
        if config.AUG.MIXUP > 0.:
            criterion = SoftTargetCrossEntropy()
            logger.info("Using SoftTargetCrossEntropy as loss (for mixup).")
        elif config.MODEL.LABEL_SMOOTHING > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
            logger.info(f"Using LabelSmoothingCrossEntropy as loss with smoothing={config.MODEL.LABEL_SMOOTHING}.")
        else:
            criterion = torch.nn.CrossEntropyLoss()
            logger.info("Using torch.nn.CrossEntropyLoss as loss.")
    else:
        logger.info("Dataset is imbalanced based on class weights. Using AsymmetricLossSingleLabel.")
        criterion = AsymmetricLossSingleLabel(gamma_pos=0, gamma_neg=4, eps=0.1, reduction='mean')
        logger.info("Using AsymmetricLossSingleLabel as loss.")

    max_accuracy = 0.0
    epoch_metrics = []
    early_stop_patience = config.TRAIN.EARLY_STOP_PATIENCE if hasattr(config.TRAIN, 'EARLY_STOP_PATIENCE') else 10
    no_improve_epochs = 0
    
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, acc5, val_loss, macro_f1, macro_recall = validate(config, data_loader_val, model, criterion)
        logger.info(f"Validation Accuracy (from resumed checkpoint): {acc1:.1f}%")
        if config.EVAL_MODE:
            logger.info("Evaluating on test set...")
            acc1_test, acc5_test, test_loss, macro_f1_test, macro_recall_test = validate(config, data_loader_test, model, criterion)
            logger.info(f"[TEST SET] Accuracy@1: {acc1_test:.2f}%, Accuracy@5: {acc5_test:.2f}%, Loss: {test_loss:.4f}")
            logger.info(f"[TEST SET] Macro-F1: {macro_f1_test:.4f}, Macro-Recall: {macro_recall_test:.4f}")
            return
        epoch_metrics.append({
            "epoch": config.TRAIN.START_EPOCH - 1,
            "val_loss": val_loss,
            "acc1": acc1,
            "acc5": acc5,
            "macro_f1": macro_f1,
            "macro_recall": macro_recall,
        })

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        model.head = torch.nn.Linear(model.head.in_features, config.MODEL.NUM_CLASSES).cuda()
        acc1, acc5, val_loss, macro_f1, macro_recall = validate(config, data_loader_val, model, criterion)
        logger.info(f"Validation Accuracy (after loading pretrained): {acc1:.1f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler)
        acc1, acc5, val_loss, macro_f1, macro_recall = validate(config, data_loader_val, model, criterion)
        logger.info(f"Epoch {epoch}: Validation Accuracy: {acc1:.1f}% on {len(dataset_val)} images, Loss: {val_loss:.4f}, Macro-F1: {macro_f1:.4f}, Macro-Recall: {macro_recall:.4f}")
        epoch_metrics.append({
            "epoch": epoch,
            "val_loss": val_loss,
            "acc1": acc1,
            "acc5": acc5,
            "macro_f1": macro_f1,
            "macro_recall": macro_recall,
        })
        if acc1 > max_accuracy:
            max_accuracy = acc1
            logger.info(f"New best accuracy: {max_accuracy:.2f}%, saving checkpoint...")
            ckpt_dir = config.OUTPUT
            for file in os.listdir(ckpt_dir):
                if file.endswith('.pth'):
                    os.remove(os.path.join(ckpt_dir, file))
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    metrics_csv_path = os.path.join(config.OUTPUT, "epoch_metrics.csv")
    df = pd.DataFrame(epoch_metrics)
    df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Epoch metrics saved to {metrics_csv_path}")

    logger.info("Evaluating on test set...")
    acc1_test, acc5_test, test_loss, macro_f1_test, macro_recall_test = validate(config, data_loader_test, model, criterion)
    logger.info(f"[TEST SET] Accuracy@1: {acc1_test:.2f}%, Accuracy@5: {acc5_test:.2f}%, Loss: {test_loss:.4f}")
    logger.info(f"[TEST SET] Macro-F1: {macro_f1_test:.4f}, Macro-Recall: {macro_recall_test:.4f}")


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(
            loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=model.parameters(), create_graph=is_second_order,
            update_grad=((idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        )
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)

        loss_scale_value = loss_scaler.state_dict()["scale"]
        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB'
            )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, criterion):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    all_preds = []
    all_targets = []

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        
        preds = output.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(target.cpu().numpy())

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB'
            )

    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    print(f"🔍 [DEBUG] model output shape = {output.shape}")
    print(f"🔍 [DEBUG] target shape = {target.shape}, target unique values: {target.unique()}")

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    cm = confusion_matrix(all_targets, all_preds, labels=range(config.MODEL.NUM_CLASSES))
    cls_report = classification_report(all_targets, all_preds, labels=range(config.MODEL.NUM_CLASSES), output_dict=True)
    macro_f1 = cls_report['macro avg']['f1-score']
    macro_recall = cls_report['macro avg']['recall']


    cm_df = pd.DataFrame(cm, index=range(config.MODEL.NUM_CLASSES), columns=range(config.MODEL.NUM_CLASSES))
    cm_csv_path = os.path.join(config.OUTPUT, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)
    logger.info(f"Confusion matrix saved to {cm_csv_path}")


    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_png_path = os.path.join(config.OUTPUT, "confusion_matrix.png")
    plt.savefig(cm_png_path)
    plt.close(fig)
    logger.info(f"Confusion matrix figure saved to {cm_png_path}")

    logger.info("Classification Report (macro avg): " +
                f"Precision: {cls_report['macro avg']['precision']:.4f}, " +
                f"Recall: {macro_recall:.4f}, " +
                f"F1-score: {macro_f1:.4f}")

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg, macro_f1, macro_recall


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()
    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))
    main(config)
    import sys; sys.exit(0)
