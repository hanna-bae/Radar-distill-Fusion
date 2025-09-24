import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from eval_with_wandb import repeat_eval_ckpt
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils_new import train_model

# ---- optional wandb ----
try:
    import wandb
except Exception:
    wandb = None


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='./tools/cfgs/hgsfusion/hgsfusion_vod.yaml',
                        help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')

    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn (legacy)')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=10, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=True, help='')
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False,
                        help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')

    # gradient accumulation
    parser.add_argument('--accum_steps', type=int, default=1, help='gradient accumulation steps')

    # ---------- Norm switch ----------
    parser.add_argument('--bn', type=str, default='bn', choices=['bn', 'gn', 'syncbn'],
                        help='normalization: bn | gn | syncbn')
    parser.add_argument('--gn_groups', type=int, default=32, help='GroupNorm groups (when --bn gn)')

    # ---------- W&B ----------
    parser.add_argument('--use_wandb', action='store_true', help='log training metrics to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='pcdet', help='W&B project')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity/team')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--wandb_group', type=str, default=None, help='W&B group')
    parser.add_argument('--wandb_tags', type=str, default='', help='comma-separated tags')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_notes', type=str, default=None)
    parser.add_argument('--wandb_watch', type=str, default='none', choices=['none', 'gradients', 'all'],
                        help='disable by default to save VRAM')
    parser.add_argument('--wandb_ckpt', type=str, default='none', choices=['none', 'latest', 'epoch', 'all'],
                        help='upload checkpoints as artifacts')

    # ---------- Eval during training ----------
    parser.add_argument('--eval_during_train', action='store_true', help='run validation during training')
    parser.add_argument('--eval_interval_epochs', type=int, default=1, help='eval every N epochs')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def _convert_bn_to_gn(module: nn.Module, num_groups: int):
    """Recursively replace BatchNorm{1,2,3}d with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            C = child.num_features
            gn = nn.GroupNorm(num_groups=min(max(1, num_groups), C), num_channels=C, eps=child.eps, affine=True)
            setattr(module, name, gn)
        else:
            _convert_bn_to_gn(child, num_groups)


# ======================= NEW: W&B resume helpers =======================
def _find_last_wandb_id(output_dir: Path):
    """env -> run_id file -> last wandb run folder 순으로 run id를 찾아낸다."""
    env_id = os.environ.get('WANDB_RUN_ID')
    if env_id:
        return env_id
    run_id_file = output_dir / 'wandb_run_id.txt'
    if run_id_file.exists():
        rid = run_id_file.read_text().strip()
        if rid:
            return rid
    try:
        # fallback: 최근 run-*/files/wandb-metadata.json에서 id 추출
        import json, glob
        runs = sorted(glob.glob(str(output_dir / 'wandb' / 'run-*')), key=os.path.getmtime)
        if runs:
            meta = Path(runs[-1]) / 'files' / 'wandb-metadata.json'
            if meta.exists():
                return json.loads(meta.read_text()).get('id', None)
    except Exception:
        pass
    return None
# ======================================================================


def main():
    args, cfg_ = parse_config()

    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    logger.info("----------- Create dataloader & network & optimizer -----------")
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)

    # --- Norm switch ---
    if args.bn == 'syncbn' or args.sync_bn:
        if dist_train:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logger.info('Applied SyncBatchNorm conversion.')
        else:
            logger.warning('syncbn requested but not in distributed mode; keeping BatchNorm.')
    elif args.bn == 'gn':
        _convert_bn_to_gn(model, args.gn_groups)
        logger.info(f'Applied GroupNorm conversion: groups={args.gn_groups}')

    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                    )
                    last_epoch = start_epoch + 1
                    break
                except:
                    ckpt_list = ckpt_list[:-1]

    model.train()  # before wrap to DDP
    if dist_train:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], find_unused_parameters=True
        )
    logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # ---------- W&B init (rank 0) ----------
    if cfg.LOCAL_RANK == 0 and args.use_wandb and wandb is not None:
        run_name = args.wandb_run_name or f"train-{cfg.TAG}-{args.extra_tag}"
        tags = [t.strip() for t in args.wandb_tags.split(',')] if args.wandb_tags else None

        # 기존 run으로 이어붙이기(id + resume="allow") + run_id 파일 고정
        run_id_file = output_dir / 'wandb_run_id.txt'
        prev_id = _find_last_wandb_id(output_dir)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=args.wandb_group,
            notes=args.wandb_notes,
            mode=args.wandb_mode,
            dir=str(output_dir),
            id=prev_id,
            resume="allow"
        )
        try:
            wandb.config.update({
                'cfg_file': args.cfg_file,
                'exp_group_path': str(cfg.EXP_GROUP_PATH),
                'extra_tag': str(args.extra_tag),
                'batch_size_per_gpu': args.batch_size,
                'accum_steps': int(args.accum_steps),
                'use_amp': bool(args.use_amp),
                'norm': args.bn,
                'gn_groups': args.gn_groups if args.bn == 'gn' else None,
            }, allow_val_change=True)
        except Exception:
            pass
        if tags:
            try:
                wandb.run.tags = list(set((wandb.run.tags or []) + tags))
            except Exception:
                pass

        # run_id 파일이 없으면 현재 run id를 기록
        try:
            if (not run_id_file.exists()) and wandb.run is not None:
                run_id_file.write_text(wandb.run.id)
        except Exception:
            pass

        # step 축 정의(분리)
        try:
            wandb.define_metric("global_step")
            wandb.define_metric("epoch")
            # train → global_step, eval/val → epoch
            wandb.define_metric("train/*", step_metric="global_step")
            wandb.define_metric("val/*",   step_metric="epoch")
            wandb.define_metric("eval/*",  step_metric="epoch")
        except Exception:
            pass

    if args.use_wandb:
        os.environ['WANDB_CKPT_POLICY'] = args.wandb_ckpt

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # optional validation dataloader (for eval during training)
    val_loader = None
    if args.eval_during_train:
        _, val_loader, _ = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_train, workers=args.workers, logger=logger, training=False
        )

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger,
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record,
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg,
        # NEW:
        accum_steps=args.accum_steps,
        val_loader=val_loader,
        eval_during_train=args.eval_during_train,
        eval_interval_epochs=args.eval_interval_epochs,
        dist_eval=dist_train,
        eval_root_dir=(output_dir / 'eval')
    )

    # post-train evaluation (keep as-is)
    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
