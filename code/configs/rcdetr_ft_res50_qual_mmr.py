# ①+⑤ LEARNED supervised quality head + spatial diversity (MMR) — recommended first experiment.
# Combines the root-cause fix (a trained selection score, see rcdetr_ft_res50_qual.py) with the
# diversity term that recovered importance-only's loss in §9 and which counters the attention-collapse
# RCTrans itself warns about (queries collapsing to one region after position-embedding recompute).
# Selection score = sigmoid(quality_head) then greedy MMR redundancy suppression over BEV xy.
# No Doppler term (mode lacks 'doppler'); the head can learn velocity sensitivity from query features.
# Recipe identical to ft_cls / ft_dmmr_v03 (load public weight, 4-GPU, 6ep, lr1e-4).
_base_ = ['./rcdetr_90e_256×704_res50.py']

num_gpus = 4
batch_size = 4
num_iters_per_epoch = 28130 // (num_gpus * batch_size)   # 1758
num_epochs = 6

model = dict(
    pts_bbox_head=dict(
        query_select=dict(
            mode='quality_mmr',
            quality_weight=2.0,   # weight on the auxiliary objectness loss
            mmr_gamma=0.5,        # diversity strength (same as dmmr_v03)
            mmr_sigma=0.08,       # spatial kernel bandwidth (BEV, normalized)
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
