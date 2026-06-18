# ACCV 2026 투고 설계안 — Physics-guided Query Selection for Radar-Camera 3D Detection

> 작성 2026-06-10. 기준 문서: `radar/PROGRESS.md` (§8–9 결과까지 반영).
> **마감: ACCV 7/5 (D-25). 과제 종료 8/31. fallback: WACV(통상 7월) → 최종보고서 모드.**

---

## 0. 한 줄 결론

**from-scratch 1런을 오늘~3일 내 착수하는 것이 critical path.** 결과에 따라 두 가지 논문 중 하나로 분기하되, **어느 쪽이어도 투고 가능한 분석-중심 골격**으로 설계한다. 나머지 실험(latency, budget 곡선, night/rain, λ sweep, 시각화)은 전부 저비용(추론 위주)이라 from-scratch가 도는 동안 병렬 수행.

| 분기 | 조건 | 논문 프레이밍 |
|---|---|---|
| **Outcome A (win)** | from-scratch dmmr가 cls 대비 NDS +0.005↑ 또는 동등 NDS에 budget/latency 우위 | "Doppler-guided Diverse Query Selection improves RC detection" — 방법 논문 |
| **Outcome B (parity)** | 천장 동일 수렴 | **"When Does Radar Physics Help Query Selection?"** — 분석 논문 (negative→recovery 메커니즘 + budget 강건성 + 악천후) |

Outcome B도 충분히 성립하는 이유: §9의 단조회복(0.563→0.574→0.584)은 그 자체로 검증된 발견이고, MLLM token pruning에서 같은 구조의 결과(importance-only 한계 → diversity 결합 우위; CDPruner, ToDRE)가 2025–26 최상위 학회에 연달아 통과됨. 같은 원리를 3D detection query 선택에서 최초로 보이는 것.

---

## 1. 논문 주장(Claims)과 증거 매핑

| # | 주장 | 증거 | 상태 |
|---|---|---|---|
| C1 | radar 물리신호는 **point 수준**에서 유효(Doppler AUC 0.61–0.62, RCS는 무효 0.40–0.45)하나, **query 선택에 naive하게 옮기면 해롭다**(−0.020 NDS) | EDA(nuScenes+VoD, 12k 프레임) + doppler v1.0 vs cls-ft 통제 비교 | ✅ 완료 |
| C2 | 손실 원인은 **저신뢰 고속 query의 temporal memory 오염 + 공간 과집중**이며, **공간 diversity(MMR)가 이를 회복**시킨다 | dmmr v1.0 +0.011 회복, mAVE/mAOE 분해, 선택 BEV 시각화 | ✅ 수치 완료 / 시각화 ⬜ |
| C3 | **gentle importance + diversity가 최적** — finetune에서 cls 완전회복(0.5842≈0.5831), from-scratch에서 (A: 초과 / B: 동등+부가이득) | velo sweep 완료 + **from-scratch 1런** | ⬜ **critical path** |
| C4 | 같은 선택 메커니즘이 **test-time budget 축소에 강건**(mem_query K↓ 시 cls 대비 완만한 열화) → 효율 이득 | budget vs NDS 곡선 + latency 표 | ⬜ 추론만, 저비용 |
| C5 (보조) | dynamic 우선 보존 → **night/rain 강건** | weather split(scene-filter, night 602f/rain 1088f) per-condition mAP, CRN Table9式 | 🔶 데이터·코드 완료, 평가 진행 / night 분산 한계 명시 |
| C6 (보조) | 발견의 **cross-dataset 일반성** — VoD(4D radar)에서도 동일한 offline 패턴 | selection recall 곡선 (이미 VoD 포함) | ✅ 완료 |

> **VoD in-model 실험은 ACCV에서 제외** (KITTI 포맷 이식 리스크가 25일 일정에 안 맞음). VoD는 C1/C6의 offline 증거로만 사용하고, in-model은 camera-ready 또는 WACV/저널 확장으로 명시. R4Det(CVPR'26)이 VoD/TJ4DRadSet를 주 무대로 쓴 만큼, limitation에 "4D radar in-model 확장" 한 줄로 선점 방어.

---

## 2. 실험 설계

> **포지셔닝(deep-research 2026-06-11 반영):** 본 논문은 **SOTA 주장 논문이 아님.** 같은 r50 backbone에서 RaCFormer val(61.3 NDS) > RCTrans/우리(58.6)이고, RCBEVDet++(72.7 val)·RaCFormer(70.2 test)가 헤드라인 SOTA. ∴ 모든 비교는 **RCTrans baseline 대비 동일 backbone·동일 recipe의 controlled delta**로 한정하고, "selection 메커니즘 교체의 효과/무손실 + 분석"으로 프레이밍(SOTA 표는 related-work 맥락용, 우리 행은 baseline-relative). E1~E6는 이 controlled 비교를 채움.

### 2.1 ★ E1: from-scratch 90ep (critical path — 즉시 착수)

- **목적:** finetune의 "cls-최적 basin 갇힘" 가설 검증 = C3의 본 테스트.
- **구성:**
  - **Run-1 (필수):** `doppler_mmr, velo_weight=0.3` from-scratch 90ep. config는 `rcdetr_90e_res50_doppler.py` 파생으로 dmmr_v03 설정 이식.
  - **Run-2 (조건부):** 같은 환경 cls from-scratch. **GPU 8장 확보 시에만** Run-1과 병렬. 미확보 시 공개 weight 재현값(NDS 0.5861, 본인 환경 재현 검증됨)을 cls 기준으로 사용하고 limitation에 명시. 리뷰 방어 논리: "공개 weight는 동일 config·동일 epoch의 저자 학습본이며, 본 환경 재현 평가로 충실성 검증."
- **자원:** 8-GPU 추정 1.5–2일 → **4-GPU면 3–4일**. 6/13까지 착수해야 6/17–18 1차 판독, 재시도 슬랙 1회 확보.
- **중간 판독:** ep30/60 ckpt를 즉시 평가해 cls 궤적(공개 weight의 중간값은 없으므로, ft 실험에서 본 plateau 패턴과 비교)으로 조기 추세 판단. ep60 시점 NDS가 0.55 미만이면 Outcome B 확정하고 GPU를 E4/E5에 재배치.
- **운영 메모:** `git config --global --add safe.directory /workspace/RCTrans` 선적용(auto-eval crash 방지), `max_keep_ckpts` 넉넉히(ep30/60/85–90 보존), 학습 데이터는 `nuscenes_bae`(데몬 가시) 경로 고정, sshfs 무관 경로인지 시작 전 확인.

### 2.2 E2: test-time budget vs NDS 곡선 (추론만, GPU 반나절)

- mem_query **K ∈ {32, 64, 96, 128}** × {cls baseline, cls-ft, dmmr_v03} 교차 평가. 추가로 test_breaking ∈ {1,2,3} 축.
- ⚠️ **기술 caveat(검증됨):** `pseudo_reference_points = nn.Embedding(num_propagated, 3)`가 **K-의존 학습 파라미터**(ckpt shape (128,3)). test에서 디코더 budget `num_propagated`를 줄이면 **ckpt 로드 충돌** → "추론만"이 아님. 해결: ① ckpt의 `pseudo_reference_points[:K]` 슬라이싱 로드, 또는 ② "budget"을 선택수 `topk_proposals`(자유 조절, 단 디코더 budget은 불변)로만 정의. **K의 의미를 못박을 것.**
- **기대 스토리:** EDA recall 곡선("Doppler가 극소 budget에서 dynamic 81–98% 보존")의 in-model 대응. **저K에서 dmmr의 열화가 cls보다 완만하면 C4 성립** — Outcome B일 때 이 곡선이 사실상 주 figure가 됨.
- 주의: 학습은 K=128로 했으므로 "train-test budget mismatch 하의 강건성"으로 프레이밍(robustness 주장이지 효율 주장 아님 — 효율 주장하려면 저K 학습 1런 필요). 가능하면 class별 분해(dynamic class에서 격차 확대 기대).

### 2.3 E3: latency 표 (추론만, 반나절 — 1·3주차 잔여분)

- 동일 GPU(4090 1장, batch 1)에서: K별·test_breaking별 ms/frame, MMR 오버헤드(기측정 23ms/call) 명시. "동등 NDS @ 더 적은 연산" 또는 "오버헤드 +x% @ 강건성"으로 정리.

### 2.4 E4: night/rain 강건성 (추론 1회 + 오프라인, 반나절) — **방법 확정(deep-research 2026-06-11)**

- ⚠️ **정정:** 옛 `rcbevdet_weather_pkl/`은 nas2_home 파일정리 때 **삭제됨(복구 불가)**. 변환 경로 폐기.
- **데이터(완료):** val pkl을 `scene.json` description으로 필터해 **이미 재생성** — `val_{night,rain,day,clear}.pkl` (night **602f/15sc**, rain **1088f/27sc**; 독립계산이 TransCAR(arXiv 2305.00397) 보고치와 정확히 일치 → 표준 split·재현 확인). 이게 RC detection의 de-facto 프로토콜.
- **평가 방법(확정):** nuScenes devkit은 split 이름으로만 GT 선택 → drop-in subset-eval 코드 공개된 것 없음(CRN 포함 자체구현). 구현 = **전체 val 추론 1회 → `tools/eval_weather_subset.py`가 GT·pred EvalBoxes를 subset sample_token으로 제한 후 per-condition 산출**. (작성 완료)
- **지표:** **CRN(ICCV'23 Table 9)식 per-condition mAP** 주 보고(+NDS 부수). CRN 참조치(cam+radar) Sunny/Rainy/Day/Night = 54.8/57.0/55.1/30.4 mAP, camera-only 대비 +14~18 mAP, **night가 ~25 mAP 최대 하락**.
- **기대:** dynamic 보존 우선 효과가 야간·우천(카메라 약화→radar 의존↑)에서 확대. 전체 NDS 동등이어도 weather split 우위면 C5로 독립 기여.
- ⚠️ **함정(리뷰어):** night subset 602f/15sc로 작아 **분산 큼** → limitation에 frame/scene 수 명시, 가능하면 night+rain 합산 보조표.

### 2.5 E5: λ(velo_weight) 미세 sweep — 0.1 / 0.2 finetune (각 ~1일, 4-GPU 6ep)

- 목적: §9 단조 곡선의 **봉우리 위치 확정**(0.3이 최적인지, 0.1–0.2에서 cls-ft 초과가 나오는지). 초과가 나오면 Outcome B에서도 "finetune에서조차 소폭 우위" 문장 확보.
- 우선순위는 E1 다음. GPU 경합 시 0.2 한 점만.

### 2.6 E6: 선택 시각화 (GPU-free, 1일)

- 기존 `selection_dryrun_viz.py` 확장: **같은 프레임에서 cls vs doppler vs dmmr가 고른 128개 memory query의 BEV 분포 + GT 박스 오버레이**. 클러스터링(cls: confident 정적 위주 / doppler: 동적 과집중 / dmmr: 동적+공간 커버) 3패널 = **Figure 1 후보**.
- night 프레임 1개 포함하면 C5 정성 보강.

### 2.7 명시적으로 빼는 것 (시간 방어)

- ❌ query-init(§4-B), query-count pruning(§4-C), learned-MLP importance(mode='mlp') 학습 — future work 한 단락으로.
- ❌ VoD in-model, TJ4DRadSet — 확장 계획으로만 언급.
- ❌ RCS learned-feature ablation 학습 — EDA의 AUC 수치(RCS 무효)로 갈음.

---

## 3. 논문 골격 (8p + ref 기준)

1. **Intro** — radar 물리신호를 query 선택에 쓴다는 직관 → 그러나 naive 적용은 해롭다는 반전 → diversity와의 결합이 해법. 기여 3개: (i) point-level vs query-level 신호 유효성의 체계적 분석(EDA+통제실험), (ii) importance+diversity 선택 메커니즘(최초), (iii) budget 강건성·악천후 분석.
2. **Related Work** (deep-research 2026-06-11, high-conf 검증)
   - **RC detection SOTA(표, split 라벨 필수):** RCBEVDet++ 72.73 NDS/67.34 mAP(ViT-L, **val**), **RaCFormer 70.2/64.9(test, CVPR'25 — 검증된 최고 test CR)**, SpaRC 67.1/60.0(test, arXiv'24-25), CRT-Fusion +1.7 NDS(NeurIPS'24, 상대값). 베이스라인 RCTrans r50 **val 0.586/0.509**(본인 재현). ※RCTrans test 64.7/57.8은 워크플로우 검증실패 → 원표 재확인 후 인용.
   - **각 방법의 radar-physics 사용처(전부 feature-prior, 선택 아님):** RCTrans(pruning=layer early-exit 6→3, 전파 128 query는 StreamPETR cls-score top-K 상속=우리 교체 지점), RaCFormer(RCS는 depth feature 보강, query는 polar/circular init·전량 유지), RCBEVDet++(RCS-aware BEV scattering 인코더), SpaRC(LSA top-K는 **공간 k-NN**, Doppler는 velocity regression — 가장 가까운 query-based 경쟁군이나 physics-선택 아님), CRT-Fusion(dense-BEV/CenterPoint, query 단계 없음·velocity는 feature warp).
   - **★ 신규성(high-conf):** 위 6개 어느 것도 Doppler/RCS를 **per-query importance score로 선택/pruning**에 쓰지 않고, **diversity(MMR/DPP/FPS)와 결합한 사례 없음** → "physics-importance + diversity for query selection" 교집합 empty. **단 closed-world 아님:** camera-only/LiDAR/일반 DETR의 token-pruning(CDPruner/ToDRE 등 VLM)은 importance+diversity 자체 선례 보유 → **novel 축을 "radar-physics importance + 3D-detection-query 적용"으로 정밀 방어**(diversity-선택 일반론이 아니라).
   - **R4Det(CVPR'26):** 4D-imaging-radar dense-BEV(PDF/DGTF/IGDR, CNN호환), **TJ4DRadSet+VoD만, nuScenes 없음, query 메커니즘 없음** → **직교**(경쟁 아님), 4D-radar 추세의 adjacent로만 인용.
   - **Diversity-aware token reduction:** CDPruner, ToDRE, PC-MMR/RKHS — "VLM token pruning에서 입증된 importance+diversity 원리의 3D-detection query 이식"으로 포지셔닝(메커니즘 선례 인정 + radar-physics 결합이 신규).
3. **Preliminary Analysis (EDA)** — AUC 표(RCS 0.40–0.45 vs Doppler 0.61–0.62, 두 데이터셋), selection recall 곡선, "RCTrans도 RCS를 입력에서 제외" 코드 증거.
4. **Method** — importance `cls + λ·minmax(|rec_velo|)`, MMR 선택(γ, σ), RCTrans 삽입 지점(head:382 topk 교체) 도식.
5. **Experiments** — §1의 C1–C6 순서대로. 핵심 표 = §9 단조회복 표(+from-scratch 행), 핵심 그림 = budget 곡선 + 선택 시각화.
6. **Analysis & Discussion** — 왜 강한 importance가 해로운가(저신뢰 고속 query → temporal memory 오염, mAVE 악화 증거), point-level 신호와 query-level 결정의 간극.
7. **Limitation/Future** — from-scratch 비용, VoD/4D in-model, learned importance.

**Figure/Table 목록(목표 6개):** F1 선택 BEV 3패널, F2 budget vs NDS 곡선, F3 EDA(AUC+recall), T1 main(단조회복+from-scratch), T2 weather split, T3 latency/오버헤드. (+supp: λ sweep, per-class)

---

## 4. 일정 (D-day = 7/5)

| 날짜 | 작업 | 게이트 |
|---|---|---|
| **6/10–12** | E1 config 작성·스모크 → **from-scratch 착수**. 병렬: E6 시각화, E4 pkl 변환 스크립트 | 6/13까지 E1 미착수면 ACCV 포기 검토 |
| 6/13–17 | E2 budget 곡선, E3 latency (E1과 GPU 분리: 추론은 1장이면 됨), E4 weather 평가(기존 ckpt로 먼저), 논문 §1–4 초고 | — |
| **6/17–18** | E1 ep30/60 중간 판독 | **Gate-1: Outcome A/B 분기 결정** |
| 6/18–24 | (A면) E1 완주 대기 + E5 λ sweep / (B면) E1 자원 일부 회수 → E5 + weather에 집중. 논문 §5–6 작성 | — |
| 6/25–28 | E1 최종 평가 → T1 확정. 전체 표·그림 동결 | **Gate-2: 6/28 초안 완성도 판정** |
| 6/29–7/4 | 전문 퇴고, supp, 사사 문구 삽입 | 미달 시 → WACV(7월)로 전환, 손실 없음 |
| 7/5 | **ACCV 제출** | — |

**사사(투고 시 필수):** "This research was supported by Basic Science Research Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Education (grant number)" — **과제관리번호 IRIS에서 이번 주 내 확보** (글쓰기와 무관하게 지금 처리 가능한 행정 항목).

---

## 5. 리스크 & fallback

| 리스크 | 확률 | 대응 |
|---|---|---|
| GPU 미확보로 E1 착수 지연 | 중 | Gate(6/13) 넘기면 **finetune-only 분석 논문으로 WACV 직행** 결정. ACCV는 깨끗이 포기 |
| from-scratch도 parity | 중 | Outcome B 골격 그대로 — T1에 "from-scratch에서도 동등(=선택 메커니즘 교체가 무손실), budget·weather에서 차별화" 서사 |
| from-scratch가 오히려 미달 | 저 | C1–C2 중심의 순수 분석 논문(negative result 정직 보고)로 축소, ACCV 난도 상승 → WACV |
| weather pkl 변환 실패 | 중 | C5 드랍, C4(budget) 승격. 논문 성립에 지장 없음 |
| 학습 인프라(sshfs 끊김, auto-eval crash) | 중 | 기지 이슈 — 로컬 경로·safe.directory 체크리스트(§2.1) 선적용, ckpt 직접평가 루틴 유지 |
| ACCV·WACV 모두 불발 | 저 | 규정상 문제 없음: 최종보고서만 제출(종료 후 60일), 성과는 5년 내 IRIS 등록. **투고 시도 자체가 손해 없는 구조** |

---

## 6. 즉시 실행 체크리스트 (오늘)

- [ ] `rcdetr_90e_res50_dmmr_v03.py` 작성 (90ep 스케줄 + dmmr_v03 선택 설정 + ckpt 보존 정책)
- [ ] 1-iter 스모크 → 4-GPU(가용 시 8) from-scratch 발사
- [ ] IRIS 과제관리번호 확인
- [ ] E4용 weather pkl 포맷 diff 떠보기 (GPU-free)
- [ ] E6 시각화 스크립트에 cls/doppler/dmmr 3패널 모드 추가 (GPU-free)

---

## 10. 실측 결과 추가 (2026-06-11, 자율 실행)

### E4 weather per-condition (scene-filter, `tools/eval_weather_subset.py`, finetune ckpt)
| 조건 | cls-ft NDS/mAP | dmmr_v03 NDS/mAP | dmmr−cls |
|---|---|---|---|
| night (602f) | 0.3868 / 0.3171 | 0.3699 / 0.2987 | **−0.017 / −0.018** |
| rain (1088f) | 0.6026 / 0.5281 | 0.6067 / 0.5260 | +0.004 / −0.002 |
| day (5417f) | 0.5854 / 0.5120 | 0.5882 / 0.5116 | ~tie |
| clear (4931f) | 0.5816 / 0.5073 | 0.5828 / 0.5048 | ~tie |
| val (6019f) | 0.5832 / 0.5085 | 0.5856 / 0.5074 | ~tie |

⚠️ **C5(악천후 강건) 미지지:** night에서 dmmr가 오히려 −0.017(가설 반대), 나머지 tie. night 602f라 분산 가능성 있으나 우위 신호 없음. → weather 표는 "parity everywhere"로 **Outcome B(메커니즘 분석)를 강화**. C5는 보조→삭제/한계 강등 검토. (eval 코드·split pkl 완비 → from-scratch ckpt 나오면 재실행만.)

### E3 latency (benchmark.py, 1×4090, batch1, flash-shim)
- dmmr_v03 (MMR 포함 end-to-end): **16.5 fps ≈ 60.6 ms/frame**. cls는 MMR 없어 미미하게 빠름(선택부 차이 ~수ms). MMR 마이크로벤치 23ms/call(B=4 학습), 추론 batch1은 더 작음.
- ⚠️ flash-attn shim이라 절대 latency 비관적 — 실 flash-attn(1.0.x) 교체 시 단축. 논문엔 "오버헤드 비율"로 보고 권장(절대값 아님).

### E1 from-scratch (critical path) — 진행
- dmmr_v0.3 from-scratch 90ep, **8-GPU**, config `rcdetr_fs_res50_dmmr_v03.py`(load_from=None, ep10마다 ckpt 보존). cls 앵커=공개 0.5861(env 재현 검증). 2026-06-11 발사. ETA ~1.4일. ep30/60 중간판독 예정. **win이면 cls-fs(env-matched) 추가 확인.**

### E1 운영 교훈 (2026-06-11 디버깅)
- ⚠️ **학습 전 `nvidia-smi`로 타 사용자 GPU 점유 확인 필수.** 8-GPU 시도가 GPU7의 외부 작업(`nmr` conda env, 6.8GB)과 충돌 → rank7 CUDA OOM. → **0–3 전용**으로 회피.
- ⚠️ **torch 1.13은 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` 미지원**(torch2.0+). 쓰면 `Unrecognized CachingAllocator option`로 즉사. 유효값은 `max_split_size_mb:128`.
- 8-GPU per-iter 2.888s vs 4-GPU 1.5s → **per-epoch wall-clock 거의 동일(8-GPU 속도이득 없음, I/O bound)**. ∴ 4-GPU(0–3) 채택, 나머지는 타작업/cls-fs 여지. ETA ~2.7일.
- 최종 from-scratch config = `rcdetr_fs4_res50_dmmr_v03.py` (4-GPU, lr2e-4, ep10마다 보존).

### E6 선택 시각화 (Figure 1 후보, GPU-free 오프라인) — 완료
- 한 forward에서 선택직전 텐서(rec_score/velo/ref) 덤프(`SELDUMP` env 가드, `tools/test.py`) → `tools/viz_selection.py`가 동일 입력에 cls/doppler/dmmr 3 규칙 적용 → BEV 3패널 + GT(초록=dynamic) 오버레이. 8프레임 `work_dirs/viz/select_*.png`.
- **정량 (8프레임 평균 BEV 12×12 격자 coverage): cls 15% / doppler 17% / dmmr 52%.** → diversity가 memory query를 장면 전반에 ~3배 넓게 분산(=C2 시각·정량 증거). doppler는 두 동적군집 과집중(저coverage)이 육안으로 확인.
- ⚠️ ref는 metric BEV(-51~51m), N=1028(900 query+128 memory). GT는 CAM_FRONT 파일명으로 val pkl 매칭(test metas에 sample_idx 없음→filename 사용).

### E1 from-scratch 중간 궤적 (단일-GPU readout, 2026-06-13)
| epoch | NDS | mAP | mAVE |
|---|---|---|---|
| ep10 | 0.279 | 0.191 | — |
| ep30 | 0.432 | 0.337 | 0.311 |
| ep60 | 0.546 | 0.456 | 0.216 |
- 정상 수렴 곡선, ep60(2/3)서 0.546, lr 4.9e-5(바닥) → **ep90 ~0.57–0.59 예상 = cls(0.586)·ft-dmmr(0.584)와 parity 수렴 중.** win 신호 없음 → **Outcome B(분석 논문) 확정 방향.** 최종 ep90 대기(~21h).
- 운영: ep readout은 단일-GPU test.py로 안정화(3-GPU dist_test는 학습 DDP와 NCCL hang). 결과파일은 호스트 쓰기경로(work_dirs는 root 소유). free-GPU 동적 선택(외부 isaac-sim GPU7 회피).

### E1 from-scratch 최종 (2026-06-15, 완주)
- dmmr_v0.3 90ep(158,220 iter) 2026-06-14 05:25 KST 정상 완료. standalone raw eval **NDS 0.5760 / mAP 0.4944 / mAVE 0.2017** (in-training val 0.5767과 일치). ckpt `work_dirs/fs_dmmr_v03/iter_158220.pth`(=latest), eval `ep90_eval.txt`.
- **판정:** from-scratch도 cls baseline(0.5861)·ft-dmmr(0.5842)와 **parity(−0.010), 명확한 win 아님.** finetune이 cls-basin에 갇힌 게 아니라 **방법 자체가 cls 선택과 동급** = 정량 win 부재 확정. → Outcome B(분석 논문) 최종 확정.

---

## 11. parity 천장 돌파 — 개선안 우선순위 (deep-research, 2026-06-15)

> **핵심 진단(high-conf, 4개 독립소스 수렴 — GFL/CLQ/Rank-DETR/UN-DETR):** parity 원인 = **선택 점수가 학습되지 않음.** `cls + λ|Doppler| + MMR`은 **선택 시점에만 계산**될 뿐 ranking objective로 역전파 안 됨 = train/inference inconsistency → 손 blend는 cls를 **따라잡기만** 하고 못 넘음(§9 결과 0.576~0.586 수렴과 정합). **win엔 "손 blend → 학습형 supervised head" 전환 필요.** RCTrans는 6레이어 deep supervision(focal cls+L1) 이미 보유 → aux loss 추가 비용 ≈ 0.
> ※ 전제 확인: 리서치가 "RCTrans는 cls top-K 미사용(PREMISE ERROR)"이라 경고했으나 이는 **논문 기준**. **우리 코드는 `rctrans_head.py:382 torch.topk(cls_score,128)`이 temporal memory-queue 전파(512큐→128) 선택부**이고 우리가 교체한 지점이 정확히 거기 → 전제 유효, 모든 개선안은 이 memory-propagation 선택에 적용.

### Tier 1 — 천장 돌파 가능성 최상 (근본원인 수정)
- **① 학습형 importance/quality head로 교체** ⭐ 최우선·저비용·고upside. `compute_importance()`의 손 blend → 작은 MLP head(query feat + Doppler feat → quality logit), **이 출력이 곧 top-K 선택 점수.** detection quality(IoU/centerness 또는 GT-velocity-aware) 타깃으로 레이어별 aux loss 공동학습. 출처: GFL(NeurIPS'20, quality를 cls에 병합) / **Sparse4D-v3 Quality Estimation head → +3.0 mAP/+2.2 NDS**(nuScenes R50). 비용 낮음~중, finetune 친화.

### Tier 2 — 선택을 매칭에서 직접 감독
- **② Rank-DETR high-order matching cost** `p̂[c]·IoU^α`(α=4) + GIoU-aware cls 타깃 `t=(GIoU+1)/2`, 학습 중반부터. head `match_costs` 수정. 기전: "고conf **AND** 고IoU"만 positive → assignment가 우리 query 직접 보상(GIoU-aware loss만으로 AP75 52.9→54.1). 비용 중. ⚠️2D COCO 결과, 3D 전이 미검증. 출처: Rank-DETR(NeurIPS'23).

### Tier 3 — 학습시간 전용 add-on (추론 무변경)
- **③ CDN/DN-style query denoising(point-based)** GT anchor center-shift 노이즈 복제, 추론시 제거(=CMT PQD). **CMT +4.3 NDS/+5.7 mAP, DINO CDN 12ep서 +6 AP**(단기 스케줄 유리). 비용 중. ⚠️CMT는 from-scratch·multimodal이라 1:1 전이 아님. 출처: CMT(2301.01283)/DINO(2203.03605).
- **④ EMA-teacher distillation** 공개 cls weight = frozen teacher, query-init + query↔GT 매칭을 student에 distill(OD-DETR, IJCAI'24). 새 selection head 안정화 — **돌파보단 안정화 기여, ①과 결합 시너지.** 비용 중.

### Tier 4 — MMR을 "선택"이 아니라 "loss"로
- **⑤ diversity를 per-layer distinctiveness loss로** 현 선택시점 MMR → query feature 구별성 손실 추가. RCTrans 논문이 직접 경고하는 "position embedding 재계산→query 같은 영역 몰림(attention collapse)"이 바로 diversity 타깃. 비용 낮음, E6 시각화(cls15% vs dmmr52% coverage)와 스토리 정합.

### 낮은 우선순위
- **⑥ DINO mixed query selection**(positional/content init 분리, 현재 content=0/positions=learnable) 중간. **⑦ RaCFormer polar adaptive query init**(거리별 밀도) 낮음~중. **⑧ QAF2D 2D→3D anchor lifting** 높음(2D detector 필요, finetune 부적합) 후순위.

### ⚠️ 쫓지 말 것 (적대적 검증 기각)
- QAF2D +2.3 NDS 전이(1-2 기각) / UN-DETR UQS +7.3 U-AP 수치(1-2) / RaCFormer "Doppler implicit dynamic catcher" 프레이밍(0-3) / RaCFormer가 PETRv2 denoising 사용(1-2).

### ①구현 완료 (2026-06-15, 스모크 통과)
- **진짜 원인 코드 확인:** `compute_importance`의 `'mlp'`용 `importance_head`는 `select_memory_indices`가 `topk`+`topk_gather(...).detach()`라 **gradient가 0 = 학습 안 됨**(dead weight). 진단 정확히 일치.
- **구현(`rctrans_head.py`, mode `'quality'`/`'quality_mmr'`):** 레이어별 supervised `quality_branches`(embed→LN→ReLU→1) 추가, 출력 sigmoid가 곧 top-K 선택 점수. aux objectness loss(Hungarian-매칭 query→1, 기존 `labels` 재사용=추가 매칭비용 0) 6레이어 deep-supervision. `loss_single`이 3-tuple 반환, `loss`에 `loss_quality` 추가. `_select_layer=prune_number-1=2`(추론 시 마지막=메모리 전파 레이어와 일치).
- **config:** `rcdetr_ft_res50_qual.py`(head 단독, Doppler·MMR 없음=①격리), `rcdetr_ft_res50_qual_mmr.py`(①+⑤, quality_weight=2.0, mmr_gamma0.5/sigma0.08). 둘 다 load 공개weight·4GPU·6ep·lr1e-4(기존 ft 레시피 동일).
- **검증:** py_compile OK, 모델 빌드 OK(quality_branches=6), **2-iter 학습 스모크 통과**(`loss_quality` 6레이어 등장 1.2~1.7, loss 23.5→22.0, grad_norm 정상). ⚠️코드는 로컬복제본만(`rctrans_local`). ⚠️컨테이너 env `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` 잔존 → 실행 시 `-e PYTORCH_CUDA_ALLOC_CONF=`로 비워야 함(torch1.13 미지원).

### ①+⑤ 실험 결과 (2026-06-15) — 기각, 천장 intrinsic 확정
- **`qual_mmr` 6ep finetune 완주**(GPU0–3, ~4.5h, iter_10548.pth). 학습 정상: `loss_quality` 0.07→0.05 수렴 = **진단대로 기존 head는 zero-grad였으나 새 supervised head는 실제 학습됨**(원인 진단 자체는 정확했음을 코드+학습으로 이중 확인).
- **raw eval(GPU4 단일): NDS 0.5822 / mAP 0.5042 / mAVE 0.2082.** baseline 0.5861 −0.004, cls-finetune 통제군 0.5831과 동률, dmmr_v0.3 0.5842 −0.002. → **여전히 parity, 천장 못 깸.**
- **결정적 함의:** literature 1순위 root-cause fix(선택점수를 supervised로)를 정확히 구현·학습시켰는데 parity → **"선택점수 unsupervised라 parity"라는 진단된 원인이 천장의 실제 원인이 아님이 반증.** from-scratch parity(학습 불충분 기각)와 합쳐, 천장은 **cls-score top-K 메모리 선택이 이미 near-최적**이라는 구조적 사실로 확정. 분석 논문(Outcome B)에 강한 negative-result 증거.
- 잔여 Tier 1 후보(②Rank-DETR cost, ③DN, ④EMA-distill)는 같은 천장에 부딪힐 가능성 높아 ROI 낮음 — 분석 논문 진행엔 불필요. eval `rctrans_local/eval_qual_mmr.txt`.

### ★추천 첫 실험 (실행 완료 — 위 결과 참조)
- **①(학습형 supervised importance head) + ⑤(diversity loss)를 한 config에 묶어 6ep finetune 1회.** ②(Rank-DETR cost)는 다음 독립 변인. ①이 단일 실험 중 가장 방어 가능·저비용·돌파 가능성 최상.
- 미해결(open Q): Doppler를 어떻게 감독? (GT velocity 회귀 타깃 vs GFL식 joint cls-quality vs Rank-DETR cost reweight — 직접 테스트한 소스 없음). MMR은 3D/BEV/feature 중 어디서 계산? temporal 선택 query의 프레임간 consistency loss 가능 여부.
- 소스: GFL 2006.04388, CLQ 2309.13269, UN-DETR 2412.10176, Rank-DETR(NeurIPS'23), Sparse4D-v3 2311.11722, CMT 2301.01283, DINO 2203.03605, OD-DETR 2406.05791, RaCFormer 2412.12725, QAF2D 2403.06093.
