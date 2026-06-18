# Physics-guided Query Selection for Efficient Radar–Camera 3D Detection

> 한 달(4주) 스코프 · ACCV 2026 (마감 2026-07-05) / WACV 2027 타깃 · 4×RTX 4090 (24GB)

---

## 1. 한 줄 요약

RCTrans(AAAI 2025, 코드 공개)를 베이스라인으로, 기존의 **학습 전략적 query pruning**을
**radar 물리 신호(Doppler velocity · RCS · SNR) 기반 importance 점수 + 공간적 diversity(중복 제거)**
를 결합한 query 선택 모듈로 교체한다.

- **주 주장:** 같은 query 수에서 더 높은 NDS/mAP, 또는 더 적은 query·layer로 동등 성능 + 낮은 latency.
- **보조 주장:** dynamic object(높은 Doppler)·신뢰 높은 반사(높은 RCS)를 우선 보존 → 야간·우천에서 강건.

---

## 2. 연구 배경 / 본인 강점과의 정합

- **본인 자산 (고정):**
  1. importance + diversity 기반 선택/pruning (PC-MMR, RKHS importance+diversity)
  2. KV cache 압축, training-free calibration, sparse attention / token pruning
- **연결:** 위 (1)이 radar 물리 신호와 결합되는 지점이 정확히 비어 있음 → 본 주제의 핵심 delta.
- **악천후/야간 강건이라는 원래 연구계획서 논지**와도 정합(radar는 저렴·전천후, dynamic/고RCS 우선 보존).

---

## 3. 신규성 — "왜 비어 있나" (reviewer 방어선)

검색으로 확인한 인접 연구와의 경계:

| 인접 연구 | 무엇을 했나 | 본 주제와 다른 점 |
|---|---|---|
| **RaCFormer** (CVPR 2025) | query-based radar-camera detection, RCS embedding | RCS를 **depth 예측 보조**로만 사용, query는 grid 초기화. 선택/pruning 신호 아님 |
| **RCTrans** (AAAI 2025, 베이스라인) | Pruning Sequential Decoder + pruning training strategy | pruning이 **물리량 무관 학습 전략**. Doppler/RCS를 importance로 안 씀 |
| **DGRO** (odometry) | RCS intensity로 점 선택 가중, Doppler로 dynamic 점 제거 | task가 **SLAM**. Gaussian/query/detection 아님, diversity 없음 |
| **MVFAN** (4D radar det) | Doppler+RCS로 foreground/background reweighting | **reweighting만**, 선택/pruning·diversity 없음 |
| **RadarSplat** | RCS → Gaussian opacity 매핑 | **reconstruction/denoising**, detection·occupancy 아님 |
| **RadarOcc** (NeurIPS 2024) | Doppler power를 confidence 신호로 | 4D radar **tensor** 기반 occupancy, query selection 아님 |
| **GaussianFusionOcc** | modality-agnostic Gaussian, adaptive allocation 언급 | 물리량 기반 아님, **명시적 diversity pruning 아님** |

**비어 있는 정확한 교집합:**
`radar 물리 신호(Doppler/RCS/SNR) 기반 importance` **＋** `diversity(중복 제거, MMR/RKHS)` **＋** `query 선택/pruning for detection`
→ 세 축을 동시에 만족하는 연구는 핵심 기여로 발표된 바 없음. 특히 **importance에 diversity를 결합한 사례가 전무**.

---

## 4. 베이스라인 정리

### 4.1 채택 — RCTrans (1순위)

- **레포:** https://github.com/liyih/rctrans  (`liyih/RCTrans`)
- **논문:** RCTrans: Radar-Camera Transformer via Radar Densifier and Sequential Decoder for 3D Object Detection — AAAI 2025 · arXiv 2412.12799
- **데이터셋:** nuScenes (3D radar + 카메라)
- **성능:** nuScenes test NDS 64.7% / mAP 57.8% (SOTA급 radar-camera)
- **구조:**
  - **Token generator:** multi-modality token 추출 + position embedding
  - **Radar Dense Encoder (RDE):** sparse valid radar token을 densify (downsample→upsample), 빈 grid 보완 (radar 유효 occupancy ~10%)
  - **Pruning Sequential Decoder (PSD):** random init query를 layer마다 갱신, step-by-step fusion → 마지막 layer pruning 결과가 최종 예측
- **본인 delta가 붙는 위치:** PSD의 query 선택/pruning 단계 (Table 5의 pruning training strategy, Table 4의 training/inference layer vs latency 분석 자리)
- **장점:** pruning + query distinctiveness 유지가 **이미 골격에 존재** → 본인 importance+diversity로 교체 자연스러움. AAAI 정식 게재 + 코드·config·ckpts 경로 공개 → from-scratch 회피.
- **주의(리스크):** 의존성이 구버전 (Python 3.8 / **CUDA 11.1 / torch 1.9.0 / flash-attn 0.2.2 / mmcv-full 1.6.0 / mmdet 2.28.2 / mmdet3d / spconv-cu111 2.1.21**). 4090(최신 드라이버, 보통 CUDA 11.8+ 필요)과 빌드 충돌 가능 → 1주차 최우선 점검.

**설치 개요 (레포 README 기준):**
```
conda create -n RCTrans python=3.8
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
  -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
pip install flash-attn==0.2.2 --no-build-isolation
pip install mmdet==2.28.2 mmsegmentation==0.30.0
cd mmdetection3d && pip install -v -e . && cd ..
pip install spconv-cu111==2.1.21 yapf==0.40.0 setuptools==59.5.0 ccimport==0.3.7 pccm==0.3.4 timm fvcore
# 데이터 준비
python tools/create_data_nusc.py --root-path ./data/nuscenes --out-dir ./data \
  --extra-tag nuscenes_radar --version v1.0
```

### 4.2 보조 / plan B 후보

| 베이스라인 | 레포 | 데이터셋 | 용도 |
|---|---|---|---|
| **CVFusion** | github.com/zhzhzhzhzhz/CVFusion | VoD / TJ4DRadSet (4D radar) | nuScenes 물리량이 약할 때 plan B. radar-guided iterative BEV fusion 2-stage → proposal/query 선택에 모듈 삽입 가능 |
| **Doracamom** | github.com/TJRadarLab/Doracamom | VoD / TJ4DRadSet / OmniHD-Scenes | detection+occupancy 동시 4D radar 베이스라인 (TCSVT 2026, 코드 공개). 확장 시 |
| **RadarGaussianDet3D** | arXiv 2509.16119 | 4D radar | radar→Gaussian detection. Gaussian 방향 회귀 시 참고 |
| **RaCFormer** | CVPR 2025 | nuScenes / VoD | 가장 가까운 인접 연구. 비교 대상(baseline-of-comparison)으로 인용 |

### 4.3 데이터셋 메모

- **nuScenes:** RCTrans occupancy/detection 파이프라인 완비, 그러나 **3D radar라 점이 매우 sparse** (RCS·velocity는 제공). 물리 신호 importance 효과가 약하게 나올 위험.
- **VoD / TJ4DRadSet:** **4D imaging radar** → RCS·Doppler·SNR 풍부 (점 0.1k~2k/frame). 물리량 스토리 강함. occupancy GT·Gaussian 베이스라인 정비도는 nuScenes보다 낮음.
- TJ4DRadSet 포인트 속성: V_r(상대 radial velocity), Range, Power(SNR, dB), Alpha/Beta(수평·수직 각).

---

## 5. 4주 실행 계획

### 1주차 — 재현 & 자리 확보
- RCTrans 환경 구축 → **CUDA/torch/flash-attn/spconv 구버전 vs 4090 충돌 점검(최우선, 첫 이틀)**
- nuScenes radar pkl 생성, 공개 weight로 평가 재현(NDS ~64.7 확인)
- PSD의 query 선택/pruning 코드 위치 파악
- baseline의 query 수·layer별 latency를 **본인 측정값으로 표 확보** (모든 비교의 기준선)

### 2주차 — 물리 신호 importance 모듈
- radar token에 실린 Doppler·RCS·SNR → importance score 매핑 (학습형 MLP vs 규칙 기반, 둘 다 ablation용)
- decoder query 선택 단계에서 random/grid 대신 physics-importance 상위 query 우선
- 1차 학습 → baseline 대비 NDS/mAP 변화 확인

### 3주차 — diversity(중복 제거) 결합
- importance 단독 시 dynamic·고RCS 영역에 query 몰림 → PC-MMR/RKHS식 diversity 항으로 분산 (**본인 wheelhouse**)
- importance-only vs importance+diversity ablation
- **핵심 표:** query 수를 줄이며 NDS/mAP/latency 곡선 (baseline pruning vs 제안) → "X% query로 동등 NDS, Y배 빠름"

### 4주차 — 강건성 · ablation · 집필
- nuScenes **night/rain split** 강건성 표 (baseline 대비)
- Doppler vs RCS vs SNR 기여 ablation, diversity 가중치 민감도
- 그림: query 선택 시각화 (보존 query가 dynamic/foreground에 정렬)
- ACCV(7/5) 맞춰 4주차 말 초안

---

## 6. 시작 시 즉시 점검할 리스크 3가지

1. **구버전 의존성 vs 4090** — RCTrans 스택이 CUDA 11.1 기반. 4090(CUDA 11.8+ 통상)에서 빌드 막힐 수 있음. 첫 이틀 내 안 되면 컨테이너/드라이버 우회.
2. **nuScenes 3D radar 물리량 품질** — 점이 sparse해 importance 효과가 약할 수 있음 → VoD(4D radar) 보조 실험 plan B 사전 준비.
3. **delta 크기** — "선택 기준만 바꿈"으로 보이지 않도록, **diversity 결합 + 강건성(night/rain)**을 메인 기여로 명확히 세울 것.

---

## 7. 참고 링크

- RCTrans (베이스라인): https://github.com/liyih/rctrans · arXiv 2412.12799 (AAAI 2025)
- RaCFormer (인접/비교): CVPR 2025 (openaccess)
- CVFusion (plan B): https://github.com/zhzhzhzhzhz/CVFusion · arXiv 2507.04587
- Doracamom (확장): https://github.com/TJRadarLab/Doracamom · arXiv 2501.15394 (TCSVT 2026)
- RadarOcc (참고): https://github.com/toytiny/radarocc · arXiv 2405.14014 (NeurIPS 2024)
- RadarGaussianDet3D: arXiv 2509.16119
- RadarSplat: arXiv 2506.01379
- TJ4DRadSet: https://github.com/TJRadarLab/TJ4DRadSet
- Awesome 4D Radar Detection: https://github.com/liuzengyun/Awesome-3D-Detection-with-4D-Radar
- Awesome Radar Perception: https://github.com/ZHOUYI1023/awesome-radar-perception

---

## 8. 마감 일정

- **ACCV 2026** — 마감 2026-07-05, 오사카, 12월 개최. 지금(6월 초) 시작 시 가능.
- **WACV 2027** — 2027년 1월 개최, 본 마감 통상 7월경. 동시 타깃 가능.
