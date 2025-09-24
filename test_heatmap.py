import os
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmcv.cnn import ConvModule, build_conv_layer

def visualize_proposals_per_class(
    dense_heatmap: torch.Tensor,
    top_proposals_index: torch.Tensor,
    top_proposals_class: torch.Tensor,
    frame_id: str,
    save_dir: str = "./viz",
    class_names: list = None
):
    """
    카테고리별로 proposal을 dense heatmap 위에 시각화하여 저장

    Args:
        dense_heatmap (torch.Tensor): [1, C, H, W]
        top_proposals_index (torch.Tensor): [1, 1, K] flattened indices
        top_proposals_class (torch.Tensor): [1, K]
        frame_id (str): 저장 파일 이름 앞에 붙을 프레임 ID
        save_dir (str): 결과 저장 폴더
        class_names (list): 카테고리 이름 리스트 (e.g., ["Car", "Pedestrian", "Cyclist"])
    """
    os.makedirs(save_dir, exist_ok=True)

    heatmap = dense_heatmap.squeeze(0).detach().cpu().numpy()  # [C, H, W]
    num_classes, H, W = heatmap.shape

    topk_indices = top_proposals_index[0, 0]  # [K]
    topk_classes = top_proposals_class[0]     # [K]

    for cls in range(num_classes):
        mask = (topk_classes == cls)
        if mask.sum() == 0:
            continue

        cls_coords = topk_indices[mask]  # [N]
        x = (cls_coords % W).cpu().numpy()
        y = (cls_coords // W).cpu().numpy()

        base_heat = heatmap[cls]  # [H, W]
        plt.figure(figsize=(5, 5))
        plt.imshow(base_heat, cmap='hot')
        plt.scatter(x, y, c='blue', s=10, marker='x', label='proposals')
        plt.title(f"{frame_id} - Class {cls} ({class_names[cls] if class_names else ''})")
        plt.legend()
        plt.axis('off')
        save_path = os.path.join(save_dir, f"{frame_id}_class{cls}.png")
        plt.savefig(save_path)
        plt.close()

# def save_heatmap(dense_heatmap, frame_id, out_dir):
#     dense_heatmap = dense_heatmap.squeeze(0).detach().cpu().numpy()  # [3, H, W]
#     #num_classes = dense_heatmap.shape[0]
#     fig, axes = plt.subplots(1, 1, figsize=(5 * num_classes, 5))

#     class_names = ['Car', 'Pedestrian', 'Cyclist']  # 원하는 클래스 이름
#     # for i in range(num_classes):
#     #     ax = axes[i] if num_classes > 1 else axes
#     #     ax.imshow(dense_heatmap[i], cmap='hot', interpolation='nearest')
#     #     ax.set_title(f'{class_names[i]} Heatmap')
#     #     ax.axis('off')
#     #     plt.tight_layout()
#     ax = axes 
#     ax.imshow(dense_heatmap,)
#     save_path = os.path.join(out_dir, f'{frame_id}_heatmap.png')
#     plt.savefig(save_path)
#     plt.close()


class FFN(nn.Module):
    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 prefix='',
                 **kwargs):
        super(FFN, self).__init__()

        self.heads = heads
        self.init_bias = init_bias
        self.prefix = prefix
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg))
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(prefix+head, conv_layers)

    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(self.prefix+head)[-1].bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(self.prefix+head).modules():
                    if isinstance(m, nn.Conv2d):
                        init.kaiming_normal(m)

    def forward(self, x):
        """Forward function for SepHead.
        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].
        Returns:
            dict[str: torch.Tensor]: contains the following keys:
                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 1, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict

def extract_proposal(heatmap):
    batch_size = heatmap.shape[0]
    padding = 3 // 2
    local_max = torch.zeros_like(heatmap)
    # equals to nms radius = voxel_size * out_size_factor * kenel_size
    local_max_inner = F.max_pool2d(heatmap, stride=1, padding=0,
                                    kernel_size=3)
    local_max[:, :, padding:(-padding), padding:(-padding)] = \
        local_max_inner
    ## for Pedestrian & Traffic_cone in nuScenes
    # if self.test_cfg["dataset"] == "nuScenes":
    #     local_max[:, 8,] = F.max_pool2d(heatmap[:, 8], kernel_size=1,
    #                                     stride=1, padding=0)
    #     local_max[:, 9,] = F.max_pool2d(heatmap[:, 9], kernel_size=1,
    #                                     stride=1, padding=0)
    # elif self.test_cfg["dataset"] == "Waymo":
    #     # for Pedestrian & Cyclist in Waymo
    #     local_max[:, 1,] = F.max_pool2d(heatmap[:, 1], kernel_size=1,
    #                                     stride=1, padding=0)
    #     local_max[:, 2,] = F.max_pool2d(heatmap[:, 2], kernel_size=1,
    #                                     stride=1, padding=0)

    # View-of-Delft 
    #local_max[:, 0] = F.max_pool2d(heatmap[:, 1], kernel_size=1,stride=1, padding=0) # Car 사실 이러면 걍 다 하면 되긴하는데 일단해서 비교
    #local_max[:, 1] = F.max_pool2d(heatmap[:, 1], kernel_size=1,stride=1, padding=0)
    local_max[:, 2] = F.max_pool2d(heatmap[:, 2], kernel_size=1,stride=1, padding=0)
    heatmap = heatmap * (heatmap == local_max)
    heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

    # top #num_proposals among all classes
    top_proposals = heatmap.view(batch_size, -1)
    top_proposals = top_proposals.argsort(dim=-1, descending=True)
    top_proposals = top_proposals[..., :200]
    top_proposals_class = top_proposals // heatmap.shape[-1]
    top_proposals_index = top_proposals % heatmap.shape[-1]
    top_proposals_index = top_proposals_index.unsqueeze(1)
    return top_proposals_class, top_proposals_index

class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)

def create_2D_grid(x_size, y_size):
    meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
    # NOTE: modified
    batch_y, batch_x = torch.meshgrid(
        *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
    )
    batch_x = batch_x + 0.5
    batch_y = batch_y + 0.5
    coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
    coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
    return coord_base

x_size = 320 #grid_size[0] out_size_factor image H W와 동일하게
y_size = 320 #grid_size[1]
bev_pos = create_2D_grid(x_size, y_size) # [1, 32400, 2]
print(f"bev pos shape: {bev_pos.shape}")
class SimpleBEVConv(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1)  # 마지막 1채널 heatmap
        )
    def forward(self, x):
        return self.net(x)

class CustomResNet(nn.Module):
    def __init__(
        self, 
        numC_input=128 + 128,
        num_layer=[2+1, ],
        num_channels=[128, ],
        stride = [1, ],
        backbone_output_ids = [0, ], 
        norm_cfg = dict(type='BN'),
        with_cp = False, 
        block_type = 'Basic'
    ):
        super(CustomResNet, self).__init__()

        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        curr_numC = numC_input
        for i in range(len(num_layer)):
            layer = [
                BasicBlock(
                    curr_numC,
                    num_channels[i],
                    stride=stride[i],
                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                            stride[i], 1),
                    norm_cfg=norm_cfg)
            ]
            curr_numC = num_channels[i]
            layer.extend([
                BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                for _ in range(num_layer[i] - 1)
            ])
            layers.append(nn.Sequential(*layer))
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp    

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

# 폴더 경로
img_bev_dir = 'vod/img_bev'
radar_bev_dir = 'vod/radar_bev'
out_dir = 'vod/heatmap_vis'
os.makedirs(out_dir, exist_ok=True)

# frame_id 리스트
frame_ids = sorted([f.replace('.npy', '') for f in os.listdir(radar_bev_dir) if f.endswith('.npy')])

# 모델 초기화 (256 → 64 → 1)
conv_net = SimpleBEVConv(in_channels=256).eval()

# SE block
se_block = SE_Block(256).eval()

class BasicBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        """
        Initializes convolutional block
        Args:
            in_channels: int, Number of input channels
            out_channels: int, Number of output channels
            **kwargs: Dict, Extra arguments for nn.Conv2d
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        """
        Applies convolutional block
        Args:
            features: (B, C_in, H, W), Input features
        Returns:
            x: (B, C_out, H, W), Output features
        """
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x

# shared conv 
shared_conv = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)
layers = []
layers.append(BasicBlock2D(128,128, kernel_size=3,padding=1,bias='auto'))
layers.append(nn.Conv2d(in_channels=128,out_channels=3,kernel_size=3,padding=1)) #out_channels = # of class
heatmap_head = nn.Sequential(*layers)
for frame_id in frame_ids:
    img_path = os.path.join(img_bev_dir, f'{frame_id}.npy')
    radar_path = os.path.join(radar_bev_dir, f'{frame_id}.npy')

    img_feat = np.load(img_path)   # shape: (128, 320, 320) H, W
    #print(img_feat.shape, 'img_shape')
    radar_feat = np.load(radar_path)  # shape: (128, 320, 320) H, W
    #print(radar_feat.shape, 'radar_shape')
  
    img_feat = torch.tensor(img_feat).unsqueeze(0)  # [1, 128, H, W]
    radar_feat = torch.tensor(radar_feat).unsqueeze(0)  # [1, 128, H, W]
    radar_feat = shared_conv(radar_feat) #[1, 128, 320, 320]
    #print(radar_feat.shape, 'radar_feat_conv')
    # Resize if needed
    if img_feat.shape[-2:] != radar_feat.shape[-2:]:
        img_feat = F.interpolate(img_feat, size=radar_feat.shape[-2:], mode='bilinear')

    radar_feat_flatten = radar_feat.view(1, radar_feat.shape[1], -1) #[BS, C, HW] [1, 128, 102400]
    #print('radar_feat_flatten', radar_feat_flatten.shape)
    # bev_pos = torch.Tensor()
    bev_pos = bev_pos.repeat(1, 1, 1).to(radar_feat.device) #[1, 200, 2]
    print('bev_pos', bev_pos.shape)
    fuse_feat = torch.cat([img_feat, radar_feat], dim=1)  # [1, 256, H, W]
    #print('fuse-feat', fuse_feat.shape) 
    resnet = CustomResNet().eval()
    fuse_feat = resnet(fuse_feat)[0]
    #print('create fuse feat', fuse_feat.shape) #[1, 128, H, W]

    dense_heatmap = heatmap_head(fuse_feat)
    #print(f"dense_heatmap.shape: {dense_heatmap.shape}")
    heatmap = dense_heatmap.detach().sigmoid() # [1, 3, H, W] H=W=320
    #print(f"heatmap shape : {heatmap.shape}")



    # generate proposal 
    top_proposals_class, top_proposals_index = extract_proposal(heatmap)
    # print(f"top_proposals_class: {top_proposals_class}")
    # print(f"top_proposals_index: {top_proposals_index}")
    query_labels = top_proposals_class

    # prepare sparse radar feat of proposal 
    index = top_proposals_index.expand(-1, radar_feat_flatten.shape[1], -1)
    #print(f"index : {index}, radar_feat_flatten shape : {radar_feat_flatten.shape}")
    query_feat_radar = radar_feat_flatten.gather(index=index, dim=-1)
    # radar feat flatten shape [1, 128, 102400]
    # add category embedding 
    one_hot = F.one_hot(top_proposals_class, num_classes=3).permute(0,2,1)
    class_encoding = nn.Conv1d(3, 128, 1)
    query_cat_encoding = class_encoding(one_hot.float())
    query_feat_radar += query_cat_encoding 

    query_pos_index = top_proposals_index.permute(0, 2, 1)
    query_pos_index = query_pos_index.expand(-1, -1, bev_pos.shape[-1])
    print(query_pos_index.shape) #62636
    query_pos = bev_pos.gather(index=query_pos_index, dim=1)

    
    common_heads=dict(
            center=[2, 2],
            height=[1, 2],
            dim=[3, 2],
            rot=[2, 2],
            vel=[2, 2])
    # Prediction Head
    prediction_heads = nn.ModuleList()
    for i in range(1):
        heads = copy.deepcopy(common_heads)
        heads.update(dict(heatmap=(3, 2)))
        prediction_heads.append(
            FFN(
                128,
                heads,
                conv_cfg=dict(type="Conv1d"),
                norm_cfg=dict(type="BN1d"),
                bias='auto',
            )
        )
    res = dict()
    for task in ['height', 'center', 'dim', 'rot', 'vel']:
        res[task] = \
            prediction_heads[0].__getattr__(task)(query_feat_radar)
    res['center'] += query_pos.permute(0, 2, 1)

    # generate sparse fuse feat 

    #query_feat_img = extract_instance_img_feat(res, )
    bev_feat_img = img_feat.view(1, img_feat.shape[1], -1)
    index = top_proposals_index.expand(-1, bev_feat_img.shape[1], -1)
    query_feat_bev_img = bev_feat_img.gather(index=index, dim=-1)
    # fuse net for second stage sparse prediction
    fuse_convs = []
    c_in = 128+ 128 
    for i in range(2 - 1): #sparse_fues_layers
        fuse_convs.append(
            ConvModule(
                c_in,
                c_in,
                kernel_size=1,
                stride=1,
                padding=0,
                bias='auto',
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type="BN1d")))
    fuse_convs.append(
        ConvModule(
            c_in,
            128,
            kernel_size=1,
            stride=1,
            padding=0,
            bias='auto',
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type="BN1d")))
    fuse_convs = nn.Sequential(*fuse_convs)
    #원본은 이미지 + 이미지 bev + radar bev인데 일단 두 개만 concat
    query_feat_fuse = torch.cat([query_feat_radar, query_feat_bev_img], dim=1)
    query_feat_fuse = fuse_convs(query_feat_fuse)
    res['heatmap'] = prediction_heads[0].__getattr__('heatmap')(query_feat_fuse)

    heatmap = heatmap.view(1, heatmap.shape[1], -1)
    res['query_heatmap_score'] = heatmap.gather(index=top_proposals_index.expand(-1, 3, -1), dim=-1) #[bs, num_classes, num_proposals] [1, 3, 200]
    res['dense_heatmap'] = dense_heatmap # [1, 3, 320, 320]

    # 예시 class 이름
    class_names = ["Car", "Pedestrian", "Cyclist"]

    # 시각화 호출
    visualize_proposals_per_class(
        dense_heatmap=res['dense_heatmap'],
        top_proposals_index=top_proposals_index,
        top_proposals_class=top_proposals_class,
        frame_id=frame_id,
        save_dir="vod/heatmap_vis",
        class_names=class_names
    )

    #save_heatmap(res['dense_heatmap'], frame_id, out_dir)

    # print([res])
    
    # heatmap 생성
    # heatmap = conv_net(fuse_feat)  # [1, 1, H, W]
    #heatmap_np = heatmap[0, 0].detach().numpy()

    # 정규화
    # vmin, vmax = np.percentile(heatmap_np, 1), np.percentile(heatmap_np, 99)
    # plt.imshow(heatmap_np, cmap='jet', vmin=vmin, vmax=vmax)
    # plt.title(f"Heatmap {frame_id}")
    # plt.axis('off')
    # plt.colorbar()
    # plt.savefig(os.path.join(out_dir, f"{frame_id}_heatmap.png"), bbox_inches='tight')
    # plt.close()
# 정규화
    # vmin, vmax = np.percentile(heatmap_np, 1), np.percentile(heatmap_np, 99)
    # heatmap_norm = (heatmap_np - vmin) / (vmax - vmin + 1e-5)
    # heatmap_rgb = plt.cm.jet(heatmap_norm)[..., :3]  # (H, W, 3)

    # # 0~255로 변환하여 PIL로 저장
    # heatmap_uint8 = (heatmap_rgb * 255).astype(np.uint8)

    # from PIL import Image
    # Image.fromarray(heatmap_uint8).save(os.path.join(out_dir, f"{frame_id}.png"))