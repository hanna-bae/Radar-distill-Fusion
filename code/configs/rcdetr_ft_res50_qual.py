# ① LEARNED, SUPERVISED quality head as the memory-selection score (isolated).
# Root-cause fix for the parity ceiling: the previous cls+|Doppler|+MMR blend was computed at
# selection time but never supervised into the ranking (topk + .detach() => zero gradient to the
# score), so it could only match the cls baseline. Here a per-layer quality head is trained by an
# auxiliary objectness loss (Hungarian-matched query -> 1) and its sigmoid output IS the top-K=128
# memory-propagation score (cf. GFL joint cls-quality, Sparse4D-v3 Quality Estimation).
# No Doppler, no MMR here — isolates the effect of the learned supervised head alone.
# Recipe identical to ft_cls / ft_doppler / ft_dmmr_v03 (load public weight, 4-GPU, 6ep, lr1e-4).
_base_ = ['./rcdetr_90e_256×704_res50.py']

num_gpus = 4
batch_size = 4
num_iters_per_epoch = 28130 // (num_gpus * batch_size)   # 1758
num_epochs = 6

model = dict(
    pts_bbox_head=dict(
        query_select=dict(
            mode='quality',
            quality_weight=2.0,   # weight on the auxiliary objectness loss
        ),
    ),
)

load_from = 'ckpts/rctrans_r50_train.pth'

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    weight_decay=0.01,
)

runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
evaluation = dict(interval=num_iters_per_epoch * num_epochs)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=6)

data = dict(samples_per_gpu=batch_size, workers_per_gpu=6)
