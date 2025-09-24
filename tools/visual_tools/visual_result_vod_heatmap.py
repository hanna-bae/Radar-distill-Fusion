import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from skimage import io
from skimage.transform import resize
from matplotlib.image import imread
from pathlib import Path
from pcdet.datasets.kitti import kitti_utils
from pcdet.utils import box_utils, calibration_kitti, common_utils

# 경로 설정
data_root = Path('./data/vod_radar_5frames/')
save_path = './visual_result/vod_heatmap/'
heatmap_png_dir = Path('./heatmap') 

os.makedirs(save_path, exist_ok=True)

infos = np.load(data_root / 'kitti_infos_val.pkl', allow_pickle=True)
det_result = np.load('./output/tools/cfgs/hgsfusion/hgsfusion_tj4d/default/eval/epoch_4/val/default/result.pkl', allow_pickle=True)
classes = ['Car', 'Pedestrian', 'Cyclist']

def get_image(idx):
    img_file = data_root / 'training/image_2' / ('%s.png' % idx)
    image = io.imread(img_file).astype(np.float32) / 255.0
    return image

def get_lidar(idx):
    lidar_file = data_root / 'training/velodyne' / ('%s.bin' % idx)
    return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 8)

def get_calib(idx):
    calib_file = data_root / 'training/calib' / ('%s.txt' % idx)
    return calibration_kitti.Calibration(calib_file)

def plot_gt_bev(gt_boxes, color=np.array([255, 0, 0]) / 256, facecolor=None):
    def convert_center_to_leftdown(x_cord, y_cord, angle, width, height):
        xp = x_cord - (np.sqrt(width**2 + height**2) / 2) * np.cos(np.arctan2(height, width) + angle / 180 * np.pi)
        yp = y_cord - (np.sqrt(width**2 + height**2) / 2) * np.sin(np.arctan2(height, width) + angle / 180 * np.pi)
        return xp, yp
    for gt_box in gt_boxes:
        x, y, w, h, angle = gt_box[0], gt_box[1], gt_box[3], gt_box[4], gt_box[-1] / np.pi * 180
        xp, yp = convert_center_to_leftdown(x, y, angle, w, h)
        plt.gca().add_patch(plt.Rectangle([xp, yp], w, h, fill=facecolor is not None,
                                          facecolor=facecolor, angle=angle,
                                          edgecolor=color, linewidth=1))

idxes = list(range(0, 2040, 10))

for idx in idxes:
    gt, dt = infos[idx], det_result[idx]
    frame_id = gt['image']['image_idx']
    img = get_image(frame_id)
    pts = get_lidar(frame_id)
    calib = get_calib(frame_id)

    # ==== 시각화용 GT box ====
    annos = common_utils.drop_info_with_name(gt['annos'], name='DontCare')
    mask = [name in classes for name in annos['name']]
    colors = [[0.494, 0.184, 0.556] for valid in mask if valid]

    loc, dims, rots = annos['location'][mask], annos['dimensions'][mask], annos['rotation_y'][mask]
    gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
    gt_boxes = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
    trans_lidar_to_cam, trans_cam_to_img = kitti_utils.calib_to_matricies(calib)

    # ==== 2D 이미지 + heatmap overlay ====
    heatmap_path = os.path.join(heatmap_png_dir, f'{frame_id}.png')
    #print(heatmap_path)
    if os.path.exists(heatmap_path):
        #print(heatmap_path)
        heatmap = imread(heatmap_path)[..., :3]
        heatmap_resized = resize(heatmap, img.shape[:2], anti_aliasing=True)
        overlay = 0.6 * img + 0.4 * heatmap_resized
        overlay = np.clip(overlay, 0, 1)

        plt.figure(dpi=300)
        plt.imshow(overlay)
        plt.axis('off')
        plt.title(f"{frame_id} Image + Heatmap")
        plt.savefig(os.path.join(save_path, f"{frame_id}_image_overlay.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

    # ==== BEV + heatmap + GT box ====
    plt.figure(dpi=500)
    ax = plt.gca()
    ax.set_xlim(0, 69.12)
    ax.set_ylim(-39.68, 39.68)

    # Heatmap PNG overlay
    if os.path.exists(heatmap_path):
        heatmap = imread(heatmap_path)[..., :3]
        plt.imshow(heatmap, origin='lower', extent=[0, 69.12, -39.68, 39.68], alpha=0.5)

    # LiDAR points
    for i in range(len(pts)):
        circle = plt.Circle(pts[i, :2], 0.2, facecolor=[92/255,156/255,255/255])
        ax.add_artist(circle)
        circle.set_path_effects([pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])

    # GT boxes
    mask_bbox = gt['annos']['num_points_in_gt'][mask] > 0
    plot_gt_bev(gt_boxes[mask_bbox], color=[0.494, 0.184, 0.556],
                facecolor=np.array([0.494, 0.184, 0.556, 0.3]))

    plt.axis('off')
    plt.savefig(os.path.join(save_path, f"{frame_id}_bev_overlay.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

