import numpy as np
import matplotlib.pyplot as plt
from skimage import io

radar_bev = np.load('./radar_bev/370078.npy')  # shape: (128, H, W)

feature = radar_bev.mean(axis=0)      # 채널 평균
vmin, vmax = np.percentile(feature, 1), np.percentile(feature, 99)

frame_id = "370078"
img_path = f"./data/tj4d/testing/image_2/{frame_id}.png"
img = io.imread(img_path)

plt.figure(figsize=(10, 5))

# 원본 이미지
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title(f"Image {frame_id}")
plt.axis('off')

# Radar BEV
plt.subplot(1, 2, 2)
plt.imshow(feature, cmap='viridis', vmin=vmin, vmax=vmax)
plt.title("Radar BEV (mean)")
plt.axis('off')

plt.tight_layout()
plt.savefig(f"compare_{frame_id}.png")
plt.show()
