import os
import glob
import time
from types import SimpleNamespace

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_

from pcdet.utils import common_utils, commu_utils
from eval_utils import eval_utils  # for eval during training

# ---- optional wandb ----
try:
    import wandb
except Exception:
    wandb = None


def zero_grad_safe(optimizer, set_to_none=True):
    """PyTorch 구버전 호환: set_to_none 미지원시 자동 폴백"""
    try:
        return optimizer.zero_grad(set_to_none=set_to_none)
    except TypeError:
        return optimizer.zero_grad()


# ========= W&B logging helpers (분리: step 기반 vs epoch 기반) =========
def _wb_log_step(step: int, payload: dict, rank: int):
    """
    train 전용: global_step(=accumulated_iter) 축으로 기록.
    """
    if rank == 0 and wandb is not None and wandb.run is not None and payload:
        # step 인자를 명시 → global_step 축으로 정렬
        wandb.log(payload, step=int(step))


def _wb_log_epoch(payload: dict, rank: int):
    """
    eval/val 전용: epoch 축으로만 기록.
    절대 step 인자를 넘기지 않는다(=경고 방지).
    payload 안에 반드시 'epoch' 키가 포함되어야 한다.
    """
    if rank == 0 and wandb is not None and wandb.run is not None and payload and ('epoch' in payload):
        # step 인자를 넘기지 않음 → define_metric("eval/*", step_metric="epoch")에 의해 epoch 축 사용
        wandb.log(payload)


def _wb_log_ckpt(path: str, alias: str, rank: int):
    if rank != 0 or wandb is None or wandb.run is None:
        return
    try:
        art = wandb.Artifact('model-ckpt', type='model')
        art.add_file(str(path))
        wandb.log_artifact(art, aliases=[alias])
    except Exception:
        pass
# =====================================================================


def train_one_epoch(
    model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False,
    use_logger_to_record=False, logger=None, logger_iter_interval=50, cur_epoch=None,
    total_epochs=None, ckpt_save_dir=None, ckpt_save_time_interval=300, show_gpu_stat=False,
    use_amp=False, accum_steps=1
):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    ckpt_save_cnt = 1
    start_it = accumulated_iter % total_it_each_epoch

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0 ** 16))

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()
        losses_m = common_utils.AverageMeter()

    end = time.time()
    zero_grad_safe(optimizer, set_to_none=True)

    for cur_it in range(start_it, total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter, cur_epoch)

        try:
            cur_lr = float(optimizer.lr)
        except Exception:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()

        # forward & backward
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, tb_dict, disp_dict = model_func(model, batch)
            loss_to_backprop = loss / max(1, accum_steps)

        scaler.scale(loss_to_backprop).backward()

        # step at accumulation boundary
        total_grad_norm = None
        do_step = ((cur_it - start_it + 1) % max(1, accum_steps) == 0)
        if do_step:
            scaler.unscale_(optimizer)
            total_grad_norm = clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            scaler.step(optimizer)
            scaler.update()
            zero_grad_safe(optimizer, set_to_none=True)

        accumulated_iter += 1

        cur_forward_time = time.time() - data_timer
        cur_batch_time = time.time() - end
        end = time.time()

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        if rank == 0:
            batch_size = batch.get('batch_size', None)

            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            losses_m.update(loss.item(), batch_size)

            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr,
                'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})',
                'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })

            if use_logger_to_record:
                if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch:
                    trained_time_past_all = tbar.format_dict['elapsed']
                    second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                    trained_time_each_epoch = pbar.format_dict['elapsed']
                    remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                    remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)

                    logger.info(
                        'Train: {:>4d}/{} ({:>3.0f}%) [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'LR: {lr:.3e}  '
                        f'Time cost: {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)} '
                        f'[{tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}]  '
                        'Acc_iter {acc_iter:<10d}  '
                        'Data time: {data_time.val:.2f}({data_time.avg:.2f})  '
                        'Forward time: {forward_time.val:.2f}({forward_time.avg:.2f})  '
                        'Batch time: {batch_time.val:.2f}({batch_time.avg:.2f})'.format(
                            cur_epoch + 1, total_epochs, 100. * (cur_epoch + 1) / total_epochs,
                            cur_it, total_it_each_epoch, 100. * cur_it / total_it_each_epoch,
                            loss=losses_m, lr=cur_lr, acc_iter=accumulated_iter,
                            data_time=data_time, forward_time=forward_time, batch_time=batch_time
                        )
                    )

                    if show_gpu_stat and accumulated_iter % (3 * logger_iter_interval) == 0:
                        gpu_info = os.popen('gpustat').read()
                        logger.info(gpu_info)
            else:
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)

            # TensorBoard
            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                if total_grad_norm is not None and float(total_grad_norm) == float(total_grad_norm):
                    tb_log.add_scalar('train/grad_norm', float(total_grad_norm), accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)

            # W&B (train: global_step)
            wb_payload = {
                'train/loss': float(loss.item()),
                'meta_data/learning_rate': float(cur_lr),
                'time/data_time': float(avg_data_time),
                'time/forward_time': float(avg_forward_time),
                'time/batch_time': float(avg_batch_time),
            }
            if total_grad_norm is not None and float(total_grad_norm) == float(total_grad_norm):
                wb_payload['train/grad_norm'] = float(total_grad_norm)
            for k, v in tb_dict.items():
                try:
                    wb_payload[f'train/{k}'] = float(v)
                except Exception:
                    pass
            try:
                wb_payload['sys/vram_alloc_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
                wb_payload['sys/vram_reserved_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
                wb_payload['sys/vram_max_alloc_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            except Exception:
                pass
            _wb_log_step(accumulated_iter, wb_payload, rank)

            # time-based latest ckpt
            time_past_this_epoch = pbar.format_dict['elapsed']
            if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
                ckpt_name = ckpt_save_dir / 'latest_model'
                save_checkpoint(
                    checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), filename=ckpt_name,
                )
                if os.environ.get('WANDB_CKPT_POLICY', 'none') in ('latest', 'all'):
                    _wb_log_ckpt(f'{ckpt_name}.pth', alias='latest', rank=rank)
                if logger is not None:
                    logger.info(f'Save latest model to {ckpt_name}')
                ckpt_save_cnt += 1

    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(
    model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
    start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
    lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
    merge_all_iters_to_one_epoch=False, use_amp=False,
    use_logger_to_record=False, logger=None, logger_iter_interval=None, ckpt_save_time_interval=None,
    show_gpu_stat=False, cfg=None,
    # NEW:
    accum_steps=1, val_loader=None, eval_during_train=False, eval_interval_epochs=1, dist_eval=False, eval_root_dir=None
):
    accumulated_iter = start_iter

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # scheduler pick
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                cur_epoch=cur_epoch, total_epochs=total_epochs,
                use_logger_to_record=use_logger_to_record,
                logger=logger, logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval,
                show_gpu_stat=show_gpu_stat,
                use_amp=use_amp,
                accum_steps=accum_steps
            )

            # epoch checkpoint
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )
                if os.environ.get('WANDB_CKPT_POLICY', 'none') in ('epoch', 'all'):
                    _wb_log_ckpt(f'{ckpt_name}.pth', alias=f'epoch-{trained_epoch}', rank=rank)

            # eval during training
            if eval_during_train and (trained_epoch % max(1, eval_interval_epochs) == 0):
                if val_loader is not None:
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        eval_model = model.module
                    else:
                        eval_model = model
                    was_training = eval_model.training
                    eval_model.eval()
                    try:
                        cur_result_dir = None
                        if eval_root_dir is not None and rank == 0:
                            cur_result_dir = (eval_root_dir / f'epoch_{trained_epoch}' / cfg.DATA_CONFIG.DATA_SPLIT['test'])
                            cur_result_dir.mkdir(parents=True, exist_ok=True)
                        with torch.no_grad():
                            tb_dict = eval_utils.eval_one_epoch(
                                cfg,
                                SimpleNamespace(save_to_file=True, infer_time=False),
                                eval_model, val_loader, trained_epoch, logger,
                                dist_test=dist_eval, result_dir=cur_result_dir
                            )
                        if rank == 0:
                            # TensorBoard: epoch를 step처럼 사용
                            for key, val in tb_dict.items():
                                if isinstance(val, (int, float)):
                                    if tb_log is not None:
                                        tb_log.add_scalar(f'eval/{key}', float(val), trained_epoch)

                            # W&B: epoch 축만 사용 (경고 방지)
                            eval_payload = {f'eval/{k}': float(v) for k, v in tb_dict.items()
                                            if isinstance(v, (int, float))}
                            # 선택: mAP 키 이름을 "val mAP"로 보이게 리네이밍
                            for cand in list(eval_payload.keys()):
                                base = cand.split('/', 1)[-1].lower()
                                if base in {'map', 'mean_ap', 'meanap', 'mAP'}:
                                    eval_payload.pop(cand, None)
                                    eval_payload['eval/val mAP'] = float(tb_dict[cand.split('/', 1)[-1]]) \
                                        if '/' in cand else float(tb_dict['mAP']) if 'mAP' in tb_dict else float(tb_dict.get('map', 0.0))
                                    break
                            eval_payload['epoch'] = trained_epoch
                            _wb_log_epoch(eval_payload, rank)
                    finally:
                        if was_training:
                            eval_model.train()
                        torch.cuda.empty_cache()


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except Exception:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        if torch.__version__ >= '1.4':
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save({'optimizer_state': optimizer_state})

    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)
