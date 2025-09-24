import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from .anchor_head_template import AnchorHeadTemplate
from ..utils import centernet_utils
from ..backbones_2d.resnet_modules.custom_resnet import CustomResNet

class AnchorHeadSingleDAL(AnchorHeadTemplate):
    def __init__(self,
                 model_cfg,
                 input_channels,
                 num_class,
                 class_names,
                 grid_size,
                 point_cloud_range,
                 img_feat_dim=128,
                 feat_bev_img_dim=32,
                 sparse_fuse_layers=2,
                 dense_fuse_layers=2,
                 **kwargs):
        super().__init__(
            model_cfg=model_cfg,
            num_class=num_class,
            class_names=class_names,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=True
        )

        self.hidden_channel = model_cfg.get('HIDDEN_CHANNEL', 128)
        self.img_feat_dim = img_feat_dim
        self.feat_bev_img_dim = feat_bev_img_dim
        self.num_proposals = model_cfg.get('NUM_PROPOSALS', 200)
        self.nms_kernel_size = model_cfg.get('NMS_KERNEL_SIZE', 3)

        # Dense stage backbone (CustomResNet)
        cfg = dict(
            type='CustomResNet',
            numC_input=self.hidden_channel + self.feat_bev_img_dim,
            num_layer=[dense_fuse_layers + 1],
            num_channels=[self.hidden_channel],
            stride=[1],
            backbone_output_ids=[0],
        )
        self.dense_heatmap_fuse_convs = custom_resnet.build_custom_resnet(cfg)

        # Sparse stage fuse convs
        fuse_convs = []
        c_in = self.img_feat_dim + self.hidden_channel + self.feat_bev_img_dim
        for i in range(sparse_fuse_layers - 1):
            fuse_convs.append(
                ConvModule(
                    c_in, c_in, kernel_size=1, stride=1, padding=0,
                    conv_cfg=dict(type='Conv1d'),
                    norm_cfg=dict(type="BN1d"))
            )
        fuse_convs.append(
            ConvModule(
                c_in, self.hidden_channel,
                kernel_size=1, stride=1, padding=0,
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type="BN1d"))
        )
        self.fuse_convs = nn.Sequential(*fuse_convs)

        # Shared lidar conv
        self.shared_conv = nn.Conv2d(input_channels, self.hidden_channel, kernel_size=3, padding=1)

        # Heatmap heads
        self.heatmap_head_dense = nn.Conv2d(self.hidden_channel, self.num_class, kernel_size=3, padding=1)
        self.class_encoding = nn.Conv1d(self.num_class, self.hidden_channel, 1)

        # Prediction heads (center, dim, rot, etc.)
        # 단순 예시: center/height/dim/rot 각각 1D conv
        self.pred_center = nn.Conv1d(self.hidden_channel, 2, 1)
        self.pred_height = nn.Conv1d(self.hidden_channel, 1, 1)
        self.pred_dim = nn.Conv1d(self.hidden_channel, 3, 1)
        self.pred_rot = nn.Conv1d(self.hidden_channel, 2, 1)
        self.pred_heatmap = nn.Conv1d(self.hidden_channel, self.num_class, 1)

        # Positional embedding for BEV
        x_size = self.grid_size[0] // model_cfg.get('FEATURE_MAP_STRIDE', 8)
        y_size = self.grid_size[1] // model_cfg.get('FEATURE_MAP_STRIDE', 8)
        self.bev_pos = self.create_2D_grid(x_size, y_size)

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def extract_proposal(self, heatmap):
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        heatmap = heatmap * (heatmap == local_max)
        B, C, H, W = heatmap.shape
        heatmap_flat = heatmap.view(B, C, -1)
        top_proposals = heatmap_flat.view(B, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // (H*W)
        top_proposals_index = top_proposals % (H*W)
        return top_proposals_class, top_proposals_index

    def extract_img_feat_from_3dpoints(self, points, img_inputs_list):
        # 실제 DALHead/TransFusionHead 코드 복사하기
        # 여기선 단순 placeholder
        B, P, _ = points.shape
        C = self.img_feat_dim
        return torch.zeros(B, C, P, device=points.device)  # [B,C,P]

    def extract_instance_img_feat(self, res_layer, img_inputs):
        center = res_layer['center']
        height = res_layer['height']
        center_x = center[:, 0:1, :] * 8 * 0.075 + self.point_cloud_range[0]
        center_y = center[:, 1:2, :] * 8 * 0.075 + self.point_cloud_range[1]
        ref_points = torch.cat([center_x, center_y, height], dim=1).permute(0, 2, 1)
        img_feat = self.extract_img_feat_from_3dpoints(ref_points, img_inputs)
        return img_feat

    def forward(self, data_dict):
        bev_feat_lidar = self.shared_conv(data_dict['spatial_features_2d'])   # [B,C,H,W]
        B, C, H, W = bev_feat_lidar.shape
        bev_feat_lidar_flatten = bev_feat_lidar.view(B, C, -1)

        bev_feat_img = data_dict['bev_feat_img']  # [B,C,H,W]
        dense_fuse_feat = torch.cat([bev_feat_lidar, bev_feat_img], dim=1)
        dense_fuse_feat = self.dense_heatmap_fuse_convs(dense_fuse_feat)[0]
        dense_heatmap = self.heatmap_head_dense(dense_fuse_feat)
        heatmap = dense_heatmap.detach().sigmoid()

        top_cls, top_idx = self.extract_proposal(heatmap)

        index = top_idx.expand(-1, C, -1)
        query_feat_lidar = bev_feat_lidar_flatten.gather(index=index, dim=-1)
        one_hot = F.one_hot(top_cls, num_classes=self.num_classes).permute(0, 2, 1)
        query_feat_lidar += self.class_encoding(one_hot.float())

        bev_pos = self.bev_pos.to(bev_feat_lidar.device).repeat(B, 1, 1)
        query_pos = bev_pos.gather(
            index=top_idx.permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1
        )

        # regression heads
        center = self.pred_center(query_feat_lidar) + query_pos.permute(0, 2, 1)
        height = self.pred_height(query_feat_lidar)
        dim = self.pred_dim(query_feat_lidar)
        rot = self.pred_rot(query_feat_lidar)

        res_layer = dict(center=center, height=height, dim=dim, rot=rot)
        query_feat_img = self.extract_instance_img_feat(res_layer, data_dict['img_inputs'])
        bev_feat_img_flat = bev_feat_img.view(B, bev_feat_img.shape[1], -1)
        query_feat_img_bev = bev_feat_img_flat.gather(
            index=top_idx.expand(-1, bev_feat_img.shape[1], -1), dim=-1
        )

        query_feat_fuse = torch.cat([query_feat_lidar, query_feat_img, query_feat_img_bev], dim=1)
        query_feat_fuse = self.fuse_convs(query_feat_fuse)
        heatmap_refined = self.pred_heatmap(query_feat_fuse)

        # 출력
        data_dict['cls_preds'] = heatmap_refined
        data_dict['box_preds'] = torch.cat([center, height, dim, rot], dim=1)
        return data_dict
