
import yaml, numpy as np
from easydict import EasyDict
from pathlib import Path
from pcdet.datasets.kitti.vod_dataset import VODDataset

# === 경로 설정: 아래 두 줄을 네 환경에 맞게 수정 ===
DATASET_CFG_PATH = 'tools/cfgs/dataset_configs/vod_fusion.yaml'  # 데이터셋 yaml
DATA_ROOT = Path('data/vod_radar_5frames')                         # 데이터 루트

# === cfg 로드 및 데이터셋 생성 ===
cfg = EasyDict(yaml.safe_load(open(DATASET_CFG_PATH)))
class_names = cfg.get('CLASS_NAMES', ['Car', 'Pedestrian', 'Cyclist'])
ds = VODDataset(dataset_cfg=cfg, class_names=class_names, training=False, root_path=DATA_ROOT)

def shape_of(x):
    try:
        return None if x is None else tuple(x.shape)
    except Exception:
        return None

max_n = min(5, len(ds.kitti_infos))
print(f"total samples: {len(ds.kitti_infos)}, showing first {max_n}")

for i in range(len(ds.kitti_infos)-max_n, len(ds.kitti_infos)):
    info = ds.kitti_infos[i]
    idx = info['point_cloud']['lidar_idx']

    print("\n" + "="*60)
    print(f"[{i}] sample_id={idx}")

    # 1) raw LiDAR points
    lidar = ds.get_lidar(idx)
    print("raw lidar points:", lidar.shape)

    # 2) virtual / gt-real (있으면)
    have_virtual = False
    try:
        virtual_points, gt_real_points = ds.get_virtual_point(idx)
        have_virtual = True
        print("virtual points:", virtual_points.shape)
        print("gt_real points:", gt_real_points.shape)

        
    except AssertionError:
        print("no virtual/gt_real points found (skip)")

    # 3) __getitem__ 로직의 merged 포인트 shape만 계산 (파일 생성 없이)
    if have_virtual:
        real_points = lidar
        if len(gt_real_points) == 0:
            merged = np.ones([real_points.shape[0], virtual_points.shape[1] + 2])
        else:
            merged = np.ones([
                virtual_points.shape[0] + real_points.shape[0] + gt_real_points.shape[0],
                virtual_points.shape[1] + 2
            ])
        print("merged points (pre-prepare):", merged.shape)
    # === merge 완료 후 ===
    print(merged.shape)

    # === test code start ===
    real_n  = real_points.shape[0]
    gt_n    = gt_real_points.shape[0]
    virt_n  = virtual_points.shape[0]
    featdim = merged.shape[1]

    # 기대 조건 확인
    assert real_points.shape[1] >= 7, f"real_points has {real_points.shape[1]} cols, need >=7"
    assert featdim == virtual_points.shape[1] + 2, \
        f"points featdim={featdim} != virtual_featdim+2 ({virtual_points.shape[1]}+2)"

    print(f"[MERGE] points.shape = {merged.shape} = ({real_n} real + {gt_n} gt_real + {virt_n} virtual) x {featdim}")
    print(f"       real_points.shape = {real_points.shape}")
    print(f"       gt_real_points.shape = {gt_real_points.shape}")
    print(f"       virtual_points.shape = {virtual_points.shape}")

    # 플래그(마지막 두 컬럼) 분포
    last2_last1 = merged[:, -2:].copy()
    from collections import Counter
    flags = Counter(map(lambda x: (int(x[0]), int(x[1])), last2_last1))
    print("       flag counts (last2,last1 -> count):", dict(flags))

    # 올바른 플래그 분류인지 확인
    real_flag_ok   = np.all(merged[:real_n, -2:] == [1, 1])
    gt_flag_ok     = np.all(merged[real_n:real_n+gt_n, -2:] == [0, 0])
    virt_flag_ok   = np.all(merged[-virt_n:, -2:] == [0, 1])
    print(f"       flags ok? real={real_flag_ok}, gt_real={gt_flag_ok}, virtual={virt_flag_ok}")

    # real 영역 중간 feature가 아직 안 채워져 있는지 확인
    unfilled = (merged[:real_n, 7:-2] != 0).sum()
    print(f"       nonzero count in real[: , 7:-2] = {int(unfilled)} (0이면 중간 특성 모두 비워짐)")

    # 샘플 행 출력
    def peek_row(tag, arr, n=2):
        n = min(n, arr.shape[0])
        print(f"       {tag} head rows (first {n}):")
        for i in range(n):
            print("         ", arr[i])

    print("       sample rows:")
    peek_row("real", merged[:real_n])
    peek_row("gt_real", merged[real_n:real_n+gt_n])
    peek_row("virtual", merged[-virt_n:])
    # === test code end ===
    calib = ds.get_calib(idx)
    pts_rect = calib.lidar_to_rect(merged[:, 0:3])
    img_hw = tuple(ds[i].get('image_shape')) if 'image_shape' in ds[i] else None
    fov_flag = ds.get_fov_flag(pts_rect, img_hw, calib)
    mergd = merged[fov_flag]
    print("fov_flag_add : ", mergd.shape)
    print(mergd[:, 0])
    # 4) __getitem__ 호출 후 최종 입력 포맷 확인
    data = ds[i]  # prepare_data 포함
    # 안전하게 존재 여부 확인하며 shape 출력
    pts_shape = shape_of(data.get('points'))
    img_shape_full = shape_of(data.get('images'))
    depth_shape = shape_of(data.get('depth_maps'))
    
    gt_names_len = len(data.get('gt_names')) if 'gt_names' in data and data['gt_names'] is not None else 0

    print("prepared points:", pts_shape)
    print("prepared images:", img_shape_full)
    print("prepared depth_maps:", depth_shape)
    print("image_shape(H,W):", img_hw)

print("\nDone.")

