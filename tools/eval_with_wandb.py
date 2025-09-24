import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

# ---- optional wandb ----
try:
    import wandb
except Exception:
    wandb = None


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/hgsfusion/hgsfusion_vod.yaml',
                        help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for eval dataloader')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')

    parser.add_argument('--ckpt', type=str,
                        default='./output/tools/cfgs/hgsfusion/hgsfusion_vod/default/ckpt/checkpoint_epoch_25.pth',
                        help='checkpoint to evaluate (ignored if --eval_all)')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model (optional)')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distributed')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes (eval_all polling)')
    parser.add_argument('--start_epoch', type=int, default=16, help='eval_all: start epoch')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='evaluate all checkpoints in dir')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='checkpoint dir if --eval_all')
    parser.add_argument('--save_to_file', action='store_true', default=True, help='store txt/json to disk')
    parser.add_argument('--infer_time', action='store_true', default=True, help='calculate inference latency')

    # ---- W&B options ----
    parser.add_argument('--use_wandb', action='store_true', help='log eval metrics to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='pcdet', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (team/org)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--wandb_group', type=str, default=None, help='W&B group')
    parser.add_argument('--wandb_tags', type=str, default='', help='comma-separated tags')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'], help='W&B mode')
    parser.add_argument('--wandb_notes', type=str, default=None, help='W&B notes')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    np.random.seed(1024)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


# ==== NEW: reuse train run id (so eval logs attach to the same W&B run) ====
def _find_last_wandb_id(output_dir: Path):
    env_id = os.environ.get('WANDB_RUN_ID')
    if env_id:
        return env_id
    run_id_file = output_dir / 'wandb_run_id.txt'
    if run_id_file.exists():
        rid = run_id_file.read_text().strip()
        if rid:
            return rid
    try:
        import json, glob as g
        runs = sorted(g.glob(str(output_dir / 'wandb' / 'run-*')), key=os.path.getmtime)
        if runs:
            meta = Path(runs[-1]) / 'files' / 'wandb-metadata.json'
            if meta.exists():
                return json.loads(meta.read_text()).get('id', None)
    except Exception:
        pass
    return None
# ==========================================================================


def wandb_init_if_needed(args, eval_output_dir):
    if cfg.LOCAL_RANK != 0 or not args.use_wandb:
        return False

    if wandb is None:
        print('[W&B] wandb is not installed; skipping W&B logging.')
        return False

    run_name = args.wandb_run_name or f"eval-{cfg.TAG}-{args.extra_tag}"
    tags = [t.strip() for t in args.wandb_tags.split(',')] if args.wandb_tags else None

    # attach to the same output_dir used by train
    train_output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    prev_id = _find_last_wandb_id(train_output_dir)

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        group=args.wandb_group,
        notes=args.wandb_notes,
        mode=args.wandb_mode,
        dir=str(eval_output_dir),
        id=prev_id,        # ★ attach to the same run if found
        resume="allow"     # ★ merge logs
    )
    # minimal config snapshot
    try:
        wandb.config.update({
            'cfg_file': str(cfg.TAG),
            'exp_group_path': str(cfg.EXP_GROUP_PATH),
            'extra_tag': str(args.extra_tag),
            'eval_all': bool(args.eval_all),
            'start_epoch': int(args.start_epoch),
        }, allow_val_change=True)
    except Exception:
        pass
    if tags:
        try:
            wandb.run.tags = list(set((wandb.run.tags or []) + tags))
        except Exception:
            pass

    # ★ Define eval axes = epoch
    try:
        wandb.define_metric("epoch")
        wandb.define_metric("eval/*", step_metric="epoch")
        wandb.define_metric("val/*",  step_metric="epoch")
    except Exception:
        pass
    return True


def wandb_log_metrics(epoch, metrics: dict):
    """Log only epoch-based eval metrics (no global_step to avoid step conflicts)."""
    if cfg.LOCAL_RANK != 0 or wandb is None or wandb.run is None:
        return
    payload = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.floating)):
            # 선택적 리네이밍: mAP → "val mAP"
            if k.lower() in {"map", "mAP", "mean_ap", "meanap"}:
                payload["eval/val mAP"] = float(v)
            else:
                payload[f'eval/{k}'] = float(v)
    # epoch 축만 기록
    try:
        payload['epoch'] = int(float(epoch))
    except Exception:
        payload['epoch'] = epoch
    if payload:
        wandb.log(payload)


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test,
                                pre_trained_path=args.pretrained_model)
    model.cuda()

    # start evaluation
    tb_dict = eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
    )

    # TB and W&B logging (rank 0)
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / (f'tensorboard_{cfg.DATA_CONFIG.DATA_SPLIT["test"]}')))
        for key, val in tb_dict.items():
            # TB는 epoch를 step처럼 써도 무방
            tb_log.add_scalar(key, val, int(float(epoch_id)) if str(epoch_id).replace('.', '', 1).isdigit() else 0)
        tb_log.close()
        # W&B: epoch 축만
        wandb_log_metrics(epoch=epoch_id, metrics=tb_dict)


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()] if os.path.exists(ckpt_record_file) else []

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard (rank 0)
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / (f'tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT["test"])))
    total_time = 0
    first_eval = True

    while True:
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, args, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir
        )

        if cfg.LOCAL_RANK == 0:
            step_epoch = int(float(cur_epoch_id))
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, step_epoch)
            # W&B: epoch 축만
            wandb_log_metrics(epoch=step_epoch, metrics=tb_dict)

        # mark evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)

    if cfg.LOCAL_RANK == 0:
        tb_log.close()


def main():
    args, cfg_ = parse_config()

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'
    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'
    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # logging
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    # W&B init (rank 0 only)
    _wb_on = wandb_init_if_needed(args, eval_output_dir)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    with torch.no_grad():
        if args.eval_all:
            repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:
            # derive epoch id for single ckpt path
            num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
            epoch_id = num_list[-1] if num_list.__len__() > 0 else '0'
            eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)

    if cfg.LOCAL_RANK == 0 and wandb is not None and wandb.run is not None:
        wandb.log({'meta/finished_eval': 1, 'epoch': int(float(epoch_id)) if str(epoch_id).replace('.', '', 1).isdigit() else 0})
        wandb.finish()


if __name__ == '__main__':
    main()
