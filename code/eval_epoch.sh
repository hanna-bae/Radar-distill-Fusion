#!/bin/bash
# 자동 epoch readout: ckpt 저장 대기 → **평가 직전 free GPU 동적 선택**(외부 작업 회피) → 평가 → 기록.
# usage: bash eval_epoch.sh <iter> <epoch_label>
set -u
IT=$1; EP=$2
LOCAL=/home/hanna.bae/rctrans_local
CKPT="$LOCAL/RCTrans/work_dirs/fs_dmmr_v03/iter_${IT}.pth"
RES="$LOCAL/ep${EP}_eval.txt"   # host-writable (work_dirs는 root 소유)
NAME="rctrans_eval_ep${EP}"
NUSC=/mnt/nfs_shared_data/dataset/nuscenes_bae
PORT=$((29500 + EP))   # epoch별 포트 분리

# 학습용 0-3 제외, 4-7 중 used memory < 2000 MiB 인 GPU만 free 로 간주 (외부 작업 자동 회피)
pick_free() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2)} $1>=4 && ($2+0)<2000 {print $1}' \
    | head -3 | paste -sd,
}

until [ -f "$CKPT" ]; do sleep 300; done
sleep 60   # 쓰기 완료 여유

# free GPU 1장 확보될 때까지 대기 (단일 GPU = DDP/NCCL hang 회피)
ONE=$(pick_free | cut -d, -f1)
while [ -z "$ONE" ]; do echo "$(date +%H:%M) ep${EP}: free GPU 없음, 대기"; sleep 600; ONE=$(pick_free | cut -d, -f1); done
echo "ep${EP}: free GPU = $ONE (단일) 사용" | tee "$RES"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "\"device=$ONE\"" --shm-size=16g --ipc=host \
  -v "$LOCAL/RCTrans:/workspace/RCTrans" \
  -v "$NUSC:/workspace/RCTrans/data/nuscenes:ro" \
  -w /workspace/RCTrans rctrans:4090 sleep infinity >/dev/null
docker exec "$NAME" bash -c "git config --global --add safe.directory /workspace/RCTrans 2>/dev/null"
docker exec "$NAME" bash -c "cd /workspace/RCTrans && CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/configs/RCTrans/rcdetr_fs4_res50_dmmr_v03.py work_dirs/fs_dmmr_v03/iter_${IT}.pth --eval bbox 2>&1 | grep -aoE 'mAP: [0-9.]+|mAVE: [0-9.]+|NDS: [0-9.]+'" >> "$RES" 2>&1
echo "EP${EP}_DONE" >> "$RES"
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "ep${EP} (iter ${IT}) 평가 완료:"; cat "$RES"
