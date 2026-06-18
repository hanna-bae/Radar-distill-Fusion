# RCTrans — Physics-guided Memory-Query Selection (실험 정리)

> ⚠️ 이 브랜치(`rctrans-experiments`)는 **RCTrans (AAAI 2025, nuScenes radar-camera, DETR-style)** 기반 query-selection 실험 기록입니다.
> repo의 `main` 브랜치 코드(HGSFusion, OpenPCDet/VoD·TJ4D)와는 **별개 코드베이스**입니다. 여기에는 코드 전체가 아니라 **delta + 문서 + 설정/로그**만 담았습니다.

## 한 줄 요약

RCTrans의 메모리 query 선택(`torch.topk(cls_score, 128)`)을 **물리 기반 importance(Doppler) + 공간 diversity(MMR)**, 나아가 **학습형 supervised quality head**로 교체하는 일련의 실험. 모든 변종이 cls baseline과 **parity(NDS ≈ 0.58)**에 수렴 → **"cls-score top-K 메모리 선택은 이미 (near-)최적"이라는 구조적(intrinsic) 천장**을 세 갈래 독립 경로로 확증. 분석/negative-result 논문(Outcome B)으로 정리.

## 핵심 결과

| 모델 | 선택 방식 | NDS | mAP | 비고 |
|---|---|---|---|---|
| cls baseline | cls top-K (공개 weight, 학습X) | **0.5861** | 0.5091 | 재현 기준 (go/no-go) |
| cls-finetune | cls top-K | 0.5831 | 0.5077 | 통제군 (finetune 거의 무해, −0.003) |
| doppler v1.0 | cls + 1.0·\|velo\| | 0.5628 | 0.4756 | importance-only **해로움 −0.020** |
| doppler_mmr v1.0 | + 공간 diversity(MMR) | 0.5737 | 0.4877 | diversity가 +0.011 회복 |
| doppler_mmr v0.3 | gentle importance + diversity | **0.5842** | 0.5059 | 완전 회복 (≈cls-ft) *(finetune)* |
| fs dmmr v0.3 (90ep) | from-scratch, gentle imp+div | 0.5760 | 0.4944 | from-scratch도 parity, win 아님 |
| **qual_mmr** | **학습형 supervised quality head**(sigmoid)=선택점수 + diversity | 0.5822 | 0.5042 | root-cause fix(①+⑤). loss_quality 정상수렴(head 실제 학습)에도 **parity** → "선택점수 unsupervised라 parity" 가설 기각, 천장 intrinsic 확정 |

### 천장이 intrinsic이라는 3중 증거
1. **물리 importance** (doppler / doppler_mmr): gentle하게 줄이면 cls와 동률, 키우면 해로움 → 추가 신호가 정보를 더하지 못함.
2. **from-scratch 90ep**: finetune basin에 갇힌 게 아님 (처음부터 학습해도 parity).
3. **학습형 supervised quality head** (`qual_mmr`): "선택점수가 학습되지 않아서"라는 deep-research 1순위 진단을 정확히 구현(BCE objectness, Hungarian-matched label, sigmoid 출력=선택점수). `loss_quality` 0.07→0.05로 정상 수렴 = head는 실제로 학습됨. **그럼에도 parity** → 진단된 원인이 천장의 실제 원인이 아님이 반증됨.

→ 천장은 ⓐ학습신호 부재(기각)도 ⓑfinetune 한계(기각)도 아니라, **cls-score top-K 메모리 선택이 이미 near-optimal**이라는 구조적 사실.

## 레이아웃

```
docs/
  PROGRESS.md                     # 단일 진실 소스 (전체 실험 이력·결론·변경기록)
  radar_accv_plan.md              # 논문 계획 + ①+⑤ root-cause fix 실험 상세
  RCTrans_delta_map.md            # 코드 delta 지도 (어디를 왜 바꿨나)
  research_proposal_radar_camera.md
code/
  rctrans_head.py                 # quality / quality_mmr 모드 포함 (로컬 diverged 사본)
  rctrans_head.delta.patch        # upstream(liyih/RCTrans @47884e8) 대비 unified diff
  configs/
    rcdetr_ft_res50_qual.py       # ① 격리: 학습형 quality head 단독 (미실행 대조군)
    rcdetr_ft_res50_qual_mmr.py   # ①+⑤: quality head + MMR diversity (실행됨)
  eval_epoch.sh                   # 단일-GPU standalone eval 스크립트
results/
  eval_qual_mmr.txt               # qual_mmr 최종 ckpt eval (NDS 0.5822)
```

## 재현 메모

- 베이스라인: [liyih/RCTrans](https://github.com/liyih/RCTrans) (arXiv 2412.12799), nuScenes radar+cam, ResNet50 256×704.
- `code/rctrans_head.delta.patch`를 upstream `projects/mmdet3d_plugin/models/dense_heads/rctrans_head.py`에 적용하면 `quality`/`quality_mmr` 모드 활성화.
- config의 `pts_bbox_head.query_select.mode`로 선택 방식 전환 (`cls`=baseline 보존 / `doppler` / `doppler_mmr` / `quality` / `quality_mmr`).
- finetune 레시피: `load_from` 공개 weight, 4-GPU, 6ep, lr 1e-4, img_backbone lr_mult 0.1.
- 환경: CUDA 11.8 / torch 1.13.1 / mmcv-full 1.6.0 / mmdet 2.28.2 / mmdet3d rc6 (Docker `rctrans:4090`).
