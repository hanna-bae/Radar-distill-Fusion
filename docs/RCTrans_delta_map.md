# RCTrans 코드 분석 — physics-importance+diversity delta 삽입 지도

> 레포: `radar/RCTrans` (liyih/RCTrans, AAAI 2025) · plugin: `projects/mmdet3d_plugin`
> config 기준: `projects/configs/RCTrans/rcdetr_90e_256×704_res50.py`

## 0. 베이스라인 동작 요약 (확인된 사실)

| 항목 | 값 / 위치 | 비고 |
|---|---|---|
| query 수 | `num_query=900` (config) | 모두 **random-init** reference points |
| query 초기화 | `rctrans_head.py:305,322` `nn.Embedding(900,3)`, `uniform_(0,1)` | 물리량·grid 무관, 순수 학습형 |
| decoder | `num_layers=6`, BEV+RV cross-attn 교차 (sequential) | `rctrans_transformer.py:79~113` |
| **layer pruning** | `test_breaking=2` (`transformer.py:52`), `prune_number=3` (`head.py:204`) | inference 시 3 layer만 → latency 절감. **이게 RCTrans의 "Pruning"** (query 아님!) |
| **query 선택** | `head.py:382` `torch.topk(rec_score, 128)` | cls confidence 상위 128개를 **temporal memory queue**로 전파 |
| 메모리 | `mem_query=128`, `memory_len=512`, `num_propagated=128` | StreamPETR식 시계열 memory |
| **radar 입력 dim** | `radar_use_dims=[0,1,2,8,9,18]` (load_dim=18) | x,y,z + **vx_comp,vy_comp(Doppler)** + sweep. **RCS(idx 5) 미사용!** |

### ⭐ 핵심 발견 — EDA와 베이스라인이 일치
RCTrans는 radar에서 **Doppler(vx_comp,vy_comp)는 쓰고 RCS는 아예 입력에서 뺐다.**
→ 우리 EDA 결론(RCS는 foreground 신호 아님, Doppler는 유효)을 SOTA 베이스라인이 **독립적으로 입증**.
→ 신규성 스토리 강화: "RCTrans는 Doppler를 *입력 feature*로만 쓰고, **query 선택은 순수 cls-score**.
  우리는 Doppler를 **명시적 importance 신호**로 + **공간 diversity**로 선택에 직접 사용." (정확히 비어있는 교집합)

## 1. delta 삽입 지점 (우선순위)

### (A) ★주 기여 — 메모리 query 선택을 importance+diversity로 — `head.py:367-387 post_update_memory`
```python
# 현재 (head.py:378-384): cls-score 단독 top-128
rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[...,0:1]
_, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)   # ← 여기
rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
```
- **교체:** `score = α·cls_score + β·physics_importance(Doppler/range) − γ·redundancy`
  → top-K 대신 **importance+diversity(MMR/RKHS) 선택**. (본인 wheelhouse가 정확히 여기 붙음)
- `rec_velo`(line 387)에 Doppler가 이미 실려 전파됨 → importance 계산에 바로 사용 가능.

### (B) query 초기화를 physics-seeded 로 — `head.py:305,322`
```python
self.reference_points = nn.Embedding(self.num_query, 3)   # random
nn.init.uniform_(self.reference_points.weight.data, 0, 1)
```
- **교체/보강:** radar 점 중 importance(Doppler) 상위 + diversity 분산으로 일부 query를 seeding,
  나머지는 learnable 유지(hybrid). random→physics-prior anchor.

### (C) 진짜 query-count pruning 추가 (RCTrans엔 없음) — `transformer.py:79~` decoder loop
- RCTrans는 layer만 줄임. **layer마다 저-importance query를 점진적으로 drop**하는
  query-count pruning을 삽입 → "X% query로 동등 NDS, Y배 빠름" 표 (제안서 3주차 핵심).
- 비교 baseline = 기존 `test_breaking` layer-pruning.

### (D) importance score 계산 모듈 — `models/backbones/radar_encoder.py` (RDE)
- radar token feature(Doppler 포함)에서 importance head(MLP) 또는 규칙기반(|Doppler|) 산출.
- RCS를 **재도입하되 learned 비선형 feature로만** (EDA: raw RCS는 misleading) → ablation 항목.

## 2. ablation/표 매핑 (제안서 5장 ↔ 코드)
- **Table(query수↓ vs NDS/latency):** (C) + (A), baseline=`test_breaking`/`prune_number` 스윕
- **importance-only vs importance+diversity:** (A)의 γ(diversity) on/off
- **Doppler vs RCS vs (pdh0):** (D) feature ablation — EDA가 예측: Doppler 주효, RCS 미미
- **night/rain 강건성:** nuScenes weather split(이미 보유) + Doppler 보존 효과

## 3. 환경 리스크 (미해결, GPU 필요 단계)
- 스택: CUDA 11.1 / torch 1.9 / mmcv-full 1.6 / mmdet 2.28 / spconv-cu111 / flash-attn 0.2.2
- 4090(CUDA 11.8+)와 빌드 충돌 가능 → conda env + 컨테이너 점검 필요 (GPU 여유 시 1순위).
- config의 ann_file = `nuscenes_radar_temporal_infos_{train,val}.pkl` → RCTrans 전용 temporal pkl
  생성 필요 (현재 보유 pkl은 RCBEVDet식). `tools/create_data_nusc.py` 재생성 대상.
