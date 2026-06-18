# Physics-guided Query Selection for Radar-Camera 3D Detection — 진행 문서

> **이 파일이 단일 source of truth.** 진행될 때마다 여기 갱신.
> 최종 갱신: 2026-06-02

---

## 0. 과제·전략 맥락 (규정 확인 완료)

- **과제:** NRF 석사과정생 연구장려금. **연구 종료 2026-08-31경.** 원 주제: 카메라-레이더 융합 3D 인지(악천후·야간 강건).
- **규정:** 연구기간 12개월(연장은 임신·육아만). **정량 논문 편수 의무 없음.** 성과는 종료 후 **5년 내 IRIS 등록** 가능 → 종료 시 논문 없어도 협약 위반 아님. 최종보고서(종료 후 60일 내)만 제출하면 정상 종료.
- **사사 필수문구(투고 시):** "This research was supported by Basic Science Research Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Education (grant number)". → **과제관리번호(grant number)를 IRIS에서 미리 확보.**
- **전략:** 본 RCTrans 주제를 4주 돌려 의미 있으면 **ACCV(마감 7/5)/WACV(통상 7월)** 투고+사사, 아니면 최종보고서만 내고 종료. 투고는 과제기간 중이라 일정·규정 OK(게재가 종료 후여도 5년 내 등록 처리).
- **★ go/no-go 게이트 = 1주차 내 RCTrans baseline 재현 성공 여부.** 막히면 보고서 모드 전환.
- 사용자는 별개로 **순수 3D reconstruction 연구로 이동**(이 과제와 무관). 카메라-레이더는 이 과제용으로만 유지. 순수 RGB 재구성엔 사사 안 함(접점 없음).
- **환경:** 4×RTX 4090(24GB). (현재 8×4090 중 전부 사용 중 → GPU-free 작업부터 진행.)

---

## 1. 주제·신규성 (검증 완료)

**한 줄:** RCTrans의 *물리량 무관* 학습 전략적 pruning을, **radar 물리신호(Doppler 중심) importance + 공간 diversity(PC-MMR/RKHS) 결합 query 선택**으로 교체.

- **베이스라인:** RCTrans (AAAI 2025, `github.com/liyih/rctrans`, arXiv 2412.12799). nuScenes radar+cam, NDS 64.7/mAP 57.8. PSD(Pruning Sequential Decoder) 골격 보유.
- **주 주장:** 같은 query 수 → 더 높은 NDS/mAP, 또는 더 적은 query·layer로 동등 성능+낮은 latency.
- **보조:** dynamic(고Doppler) 우선 보존 → night/rain 강건.
- **신규성 방어선:** RaCFormer(RCS는 depth 보조·query grid init), RCTrans(pruning이 물리량 무관), DGRO(SLAM), MVFAN(reweighting만, diversity 없음), RadarSplat(reconstruction). → **importance+diversity를 detection query 선택에 쓴 사례 없음.**
- **사용자 강점 정합:** importance+diversity 선택/pruning(PC-MMR, RKHS), sparse attention/token pruning.

---

## 2. ⚠️ EDA로 수정된 핵심 가설 (중요 — 제안서 대비 변경점)

GPU-free 데이터 분석(nuScenes 6,019 / VoD 6,435 프레임)이 제안서 가설을 **부분 수정**:

| 발견 | 근거 | 방향 수정 |
|---|---|---|
| **RCS는 foreground 신호 아님** | AUC(RCS→fg): nuScenes 0.45 / VoD 0.40 (둘 다 <0.5, 배경 clutter가 RCS 더 높음) | RCS를 raw importance로 쓰지 말 것. 학습형 비선형 feature로만(ablation) |
| **Doppler만 유효** | AUC(\|v_comp\|→fg): nuScenes 0.62 / VoD 0.61 | importance = **Doppler 중심** |
| **RCTrans도 RCS 미사용** | config `radar_use_dims=[0,1,2,8,9,18]` = x,y,z+vx_comp,vy_comp(Doppler), RCS(idx5) 제외 | EDA를 SOTA가 독립 입증. 신규성 스토리 강화 |
| **SNR 없음** | nuScenes/VoD에 Power/SNR 필드 부재 (nuScenes는 pdh0, AUC 0.44로 무효) | "Doppler·RCS·SNR" → **"Doppler 중심+RCS는 학습 feature"**로 재정의. SNR 빼기 |
| **diversity 필수** | radar 점의 90%+가 background clutter, high-RCS 정적·공간중복 | importance-only는 clutter 몰림 → **MMR이 구조적 필수**(사용자 강점이 핵심 기여) |
| **VoD를 주 데이터셋** | GT 박스 중 radar점 0개: nuScenes 65% vs VoD 32% | 주=VoD(4D, 물리량 풍부), 보조=nuScenes(night/rain split·RCTrans 호환) |

> **수정된 한 줄 주장:** "RCTrans는 Doppler를 *입력 feature*로만 쓰고 query 선택은 순수 cls-score. 우리는 Doppler를 *명시적 importance*로 + *공간 diversity*로 선택에 직접 사용." (제안서의 '비어있는 교집합'이 코드 레벨에서 확인됨)

### 2.1 선택 budget vs recall — 방법 사전 증명 (GPU-free, 각 1500프레임)
`selection_recall.py`, `selection_dryrun_viz.py`. 점수 top-K%만 남길 때 GT 객체 보존율.

- **★ DYNAMIC 객체:** Doppler-importance로 **10%서 81~87%, 15%서 94~98% 보존** (VoD/nuScenes). random 25~38%, RCS 3~16%(random 이하). → 안전 핵심 객체를 극소 budget에 보존.
- **tradeoff:** Doppler 단독은 소수 동적군집 **과집중→static 객체 포기**(5%서 0.02~0.04). **Doppler+MMR(+diversity)가 static 회복**(중·고 budget서 Doppler 단독 초과), 전체 recall 최고.
- **검증:** importance=동적 보존, diversity=static 회복, **결합이 최선** = 제안서 논지 정량 확인.
- **정직한 한계:** 공간 MMR이 raw object-recall에서 random을 압도하진 않음. diversity 본 효과는 in-model(NDS/mAP)에서 기대. 오프라인 proxy는 동기 검증까지. 상세: `eda/FINDINGS.md §4`.

---

## 3. 데이터셋·환경 자산 (확인됨)

- **nuScenes(작업 소스 = `/mnt/nfs_shared_data/dataset/nuscenes_bae`):** ⚠️ `nas2_home/nuScenes1/samples`는 **CAM_BACK·CAM_BACK_LEFT 누락**(6캠 중 4캠만)이라 RCTrans 평가 불가. nfs4의 `nuscenes_bae`(사용자 본인 것, 6캠 완전·sweeps·v1.0-trainval·RADAR 완비, **데몬 가시**)를 `data/nuscenes`로 mount. (`/mnt/.../nuScenes`도 CAM_BACK 비어있음 주의.) night/rain split pkl은 `nas2_home/.../rcbevdet_weather_pkl/` 보유(RCBEVDet식). RCTrans **사전생성 val pkl**(Drive, 6,019 infos) 사용.
- **VoD(주):** `/mnt/nfs_shared_data/dataset/VoD/view_of_delft_PUBLIC/` — KITTI 포맷, radar velodyne 8,682, label 6,435, infos pkl+사전학습 ckpt 보유. radar 포맷 7feat `[x,y,z,RCS,v_r,v_r_comp,time]`.
- **TJ4DRadSet:** `/mnt/nfs_shared_data/dataset/TJ4DRadSet_Public1/` — **파생물만(depth_viz 등), 원본 포인트·라벨 없음 → 사용 불가**(필요시 공식 다운로드).
- **RCTrans 코드:** `radar/RCTrans/` (clone 완료).

---

## 4. delta 삽입 지점 (RCTrans 코드 분석 — `radar/RCTrans_delta_map.md` 상세)

config: `projects/configs/RCTrans/rcdetr_90e_256×704_res50.py` — num_query=900, mem_query(topk)=128, decoder 6층, prune_number=3, test_breaking=2.

- **(A) 주 기여 — 메모리 query 선택** `rctrans_head.py:382` `torch.topk(cls_score,128)` → **importance(Doppler)+diversity(MMR) 선택으로 교체**. (`rec_velo`에 Doppler 이미 전파됨) → ✅ **구현 완료(2026-06-05):** `select_memory_indices()`/`compute_importance()`/`_mmr_select()` 추가, config `query_select=dict(mode='cls'|'doppler'|'doppler_mmr'|'mlp'|'mlp_mmr', velo_weight, mmr_gamma, mmr_sigma)`로 전환(default 'cls'=baseline 보존). importance=`cls + λ·minmax(|예측속도 rec_velo|)`. 단위검증(형태·고속보존50/50·MMR 분산↑)+스모크(doppler 추론 13.9 task/s 무에러) 통과. 파생 config `rcdetr_90e_res50_doppler.py`. **학습 투입 준비됨**(실제 NDS 개선은 재학습 필요).
- **(B) query 초기화** `rctrans_head.py:305,322` random `nn.Embedding(900,3)` → physics-seeded hybrid.
- **(C) query-count pruning 신설** `rctrans_transformer.py:79~` decoder loop (RCTrans는 layer만 줄임 = `test_breaking`). layer마다 저-importance query drop 추가.
- **(D) importance 모듈** `radar_encoder.py`(RDE) — Doppler 기반 score(MLP/규칙). RCS는 learned feature로 재도입(ablation).
- ※ 정정: RCTrans의 "pruning"은 **query가 아니라 layer early-exit**였음.

---

## 5. 4주 계획 & 상태

| 주차 | 내용 | 상태 |
|---|---|---|
| **사전(GPU-free)** | radar 물리량 EDA, RCTrans 코드 분석, 방향 확정 | ✅ **완료** (§2,§4) |
| 1주차 | 환경구축(4090 docker), val pkl, weight 재현, baseline latency 표 | ✅ **재현 PASS** (NDS .586/mAP .509, go/no-go 통과). latency 표만 남음 |
| 2주차 | Doppler→importance 모듈(MLP vs 규칙), query 선택 적용, 1차 학습 | ✅ 구현+검증+**1차 finetune 완료**. ⚠️결과 baseline 미달(아래) |
| 3주차 | diversity(MMR/RKHS) 결합, importance-only vs +diversity, query수↓ vs NDS/latency 곡선 | ✅ **ablation 완료**(importance-only<+diversity<gentle+diversity 단조회복 입증, §8 2026-06-08). query수↓곡선·latency 표는 남음 |
| 4주차 | night/rain 강건성 표, Doppler/RCS ablation, query 선택 시각화, ACCV 초안 | ⬜ |

---

## 6. 다음 할 일 (GPU 여유 시 / GPU-free 추가)

**환경 (Docker — conda 대신):**
- [x] **Docker 이미지 구축 완료** (`rctrans:4090`, 15.1GB) — 4090(sm_89) 호환 스택: CUDA 11.8 / torch 1.13.1+cu117 / **mmcv-full 1.6.0 소스빌드(arch `8.0;8.6+PTX`)** / mmdet 2.28.2 / mmseg 0.30.0 / spconv-cu118 / mmdet3d rc6. flash-attn 0.2.2 → **순수 torch shim**(소스 수정 0, 가중치 호환, 수식 검증 완료). **CPU import sanity 전체 통과.**
  - 핵심 교훈: ① 4090은 CUDA 11.1 불가 → 11.8. ② torch 1.13 확장빌더는 sm_89(8.9) 미지원 → `8.6+PTX`로 빌드, 드라이버가 sm_89로 JIT. ③ mmcv-full 1.6.0 prebuilt 부재 → 소스 빌드(~7.5분).
  - 사용: `bash docker/build.sh` → `GPUS='"device=N"' bash docker/run.sh` → 컨테이너 내부 1회 `bash docker/setup_inside.sh`. 상세 `docker/README.md`.

**✅ 차단 이슈 해결됨 — Docker 데이터 mount (2026-06-05):**
- 원인 확정: `/home/hanna.bae/nas2_home`이 **`allow_other` 없는 `fuse.sshfs`** → docker 데몬(root)이 그 안을 못 봄(`mkdir nas2_home: file exists`). VoD(`/mnt/nfs_shared_data`)=**nfs4는 데몬 가시(검증됨)**, 로컬 ext4도 가시.
- **해결책 = 별도 allow_other sshfs 마운트** (복사 0, 기존 마운트 안 건드림). `/etc/fuse.conf`에 `user_allow_other` 이미 설정. SSH 키 인증이 비대화형 거부라 마운트 명령은 사용자가 비밀번호로 1회 실행:
  `sshfs hanna.bae@siit-n24-2.synology.me:home /home/hanna.bae/nas_ao -o allow_other,reconnect,ServerAliveInterval=15`
  → `docker run -v /home/hanna.bae/nas_ao/nuScenes1:...` 데몬 가시 **검증 완료**.
- RCTrans 코드(sshfs 위, 28M)는 **로컬 `/home/hanna.bae/rctrans_local/RCTrans`로 복사**해 mount(데몬 가시·쓰기·빌드 영속). 컨테이너 `rctrans`(device=1) 영속 운영: code+eda+nuScenes(nas_ao)+VoD mount.

**GPU sanity (1주차, go/no-go) — ✅ 대부분 완료:**
- [x] 컨테이너 GPU + sm_89(8,9) + 8.6+PTX op **실제 JIT 실행** 검증: matmul·`iou3d_nms.boxes_iou_bev`/`nms_gpu` 정답 출력 ★해결
- [x] `setup_inside.sh`: iou3d 빌드 + 전체 import sanity(mmcv1.6/mmdet2.28/mmseg0.30/mmdet3d rc6/spconv2.3.6/flash-shim) + flash shim 수치 통과
- [x] temporal pkl: RCTrans **사전생성 val pkl**(6,019 infos, Drive) 사용 → `data/nuscenes_radar_temporal_infos_val.pkl`. (train pkl·create_data 재생성은 학습 시)
- [x] ckpts 배치: `rctrans_r50_train.pth`(1.0GB, Drive) + cascade nuim 백본(308MB, openmmlab)
- [x] **공개 weight로 baseline 재현 (= go/no-go ✅ PASS, 2026-06-05):** res50 256×704 config + 저자 `ResNet50-train` weight + 사전생성 val pkl(6,019) → **NDS 0.5861 / mAP 0.5091** (mATE .537 mASE .269 mAOE .493 mAVE .202 mAAE .184). per-class 정상(car AP .753, ped .575). single-GPU(device=1) ~수십분. ※ 논문 헤드라인 64.7은 **test·최대 config**; 이 값은 **r50 val** 저자 보고 범위와 일치 → 환경 충실 재현 확인(특히 flash-shim 정확도 손실 없음 입증). 로그 `rctrans_local/eval_res50.log`.
- [ ] baseline query수·layer별 latency 본인 측정 표 (다음)

**GPU-free (완료):**
- [x] (a) 선택 budget vs recall 곡선 — ✅ (§2.1). Doppler로 10~15%서 동적객체 81~98% 보존, RCS는 random 이하.
- [x] (b) importance+diversity(MMR) dry-run + BEV 시각화 — ✅. Doppler-only 과집중 vs Doppler+MMR static 회복 (`out/dryrun_*.png`).
- [x] Docker 이미지 빌드 + CPU import sanity — ✅ (`rctrans:4090`).

---

## 7. 산출물 위치
- 진행 문서: `radar/PROGRESS.md` (이 파일)
- EDA: `radar/eda/FINDINGS.md`, `vod_radar_eda.py`, `nusc_radar_eda.py`, `out/*.png|json`
- 코드 분석: `radar/RCTrans_delta_map.md`
- 원 제안서: `radar/research_proposal_radar_camera.md`
- 베이스라인: `radar/RCTrans/`

## 8. 변경 이력
- 2026-06-15: **★천장 돌파 1순위(①+⑤) 실행 → 기각, 천장 intrinsic 확정.** deep-research가 지목한 root-cause(선택점수 unsupervised=train/inference inconsistency) 수정안 `qual_mmr`(학습형 supervised quality head sigmoid=선택점수 + aux objectness loss Hungarian-matched + diversity) 구현·검증·6ep finetune 완주(GPU0–3, ~4.5h, iter_10548.pth). loss_quality 0.07→0.05 정상수렴(head 실제 학습 확인) **그럼에도 raw eval NDS 0.5822/mAP 0.5042/mAVE 0.2082 = parity**(baseline 0.5861 −0.004, cls-ft 0.5831과 동률). → **진단된 원인(unsupervised score)이 천장의 실제 원인 아님이 반증됨**; from-scratch parity와 합쳐 **천장은 cls top-K 선택이 이미 near-최적이라는 구조적 사실**로 확정(§9 결론5). 분석 논문(Outcome B) 더욱 견고. eval `rctrans_local/eval_qual_mmr.txt`.
- 2026-06-15: **★E1 from-scratch 완주 — critical path 종결, Outcome B 확정.** dmmr_v0.3 90ep(158,220 iter) 2026-06-14 05:25 KST 정상 완료(크래시 없음, GPU idle은 학습 종료 때문). standalone raw eval **NDS 0.5760/mAP 0.4944/mAVE 0.2017**(in-training val 0.5767과 일치). → from-scratch도 **baseline(0.5861)·ft-dmmr(0.5842)와 parity, 명확한 win 아님.** finetune이 cls-basin에 갇힌 게 아니라 **방법 자체가 cls 선택과 동급** = 정량 win 부재 확정. 논문은 분석 기여(강한 importance 해로움→diversity 회복→gentle+div 최적 + 선택 coverage cls 15% vs dmmr 52% 시각화)로 진행. 궤적 ep10 .279→ep30 .432→ep60 .546→ep90 .576(정상 수렴). ckpt `work_dirs/fs_dmmr_v03/`(iter별 9개 보존).
- 2026-06-11: **ACCV 준비 자율 실행.** deep-research(악천후 평가=CRN Table9 표준, subset-EvalBoxes 직접구현), weather split pkl 재생성(night602/rain1088, TransCAR와 일치), `tools/eval_weather_subset.py` 작성. **E4 weather: C5 미지지**(night dmmr −0.017, 나머지 tie=parity). E3 latency ~60ms/frame(shim). **E1 from-scratch dmmr_v0.3 발사**(4-GPU 0–3, ETA ~2.9일; 8-GPU는 GPU7 외부작업 OOM·expandable_segments 미지원·max_split bloat 겪고 클린 4-GPU 정착). 상세·표 `radar_accv_plan.md §10`. nas2_home/nuScenes1·rcbevdet_weather_pkl는 사용자 파일정리로 삭제(작업은 nuscenes_bae라 무관).
- 2026-06-08: **★velo_weight sweep — 가설 확증, 천장 도달.** doppler_mmr **velo0.3** finetune(6ep 완주) → **NDS 0.5842/mAP 0.5059**(raw eval; auto-eval은 git dubious-ownership로 crash했으나 ckpt 정상저장→직접평가. `git config --global --add safe.directory /workspace/RCTrans`로 영구수정). **단조 트렌드 확정:** doppler v1.0 0.5628 → dmmr v1.0 0.5737 → **dmmr v0.3 0.5842** (≈ cls-ft 0.5831 ≈ baseline 0.5861). 즉 "강한 importance 해로움→diversity 회복→gentle+diversity 완전회복"=EDA·제안서 논지 입증. **단 finetune 변종 전부 ~0.583–0.586 천장 수렴=명확한 win 아님**. 논문 win엔 **from-scratch(90ep) 필요**(finetune은 cls-최적 basin에 갇힘). 인프라 메모: nas2_home sshfs 세션 중 끊김→재마운트(학습데이터는 rctrans_local·nuscenes_bae라 무관). ckpt `work_dirs/ft_dmmr_v03/`(ep1-6 전부).
- 2026-06-02: 문서 생성. GPU-free 사전작업(EDA+코드분석) 완료, 핵심 가설 수정(§2), delta 지점 확정(§4).
- 2026-06-02: (a) 선택 recall 곡선 + (b) MMR dry-run 시각화 완료(§2.1). Doppler가 동적객체 10%서 81~98% 보존, importance+diversity tradeoff 정량 검증.
- 2026-06-02: Docker 환경 구축 완료(`rctrans:4090`). flash-attn→shim, mmcv 소스빌드, arch 8.6+PTX. CPU import sanity 통과. GPU 단계(NDS 재현)만 대기.
- 2026-06-05: GPU 가용(1–7). 컨테이너 GPU 실행 시도 → **데이터 mount 차단 발견**: nas2_home이 fuse.sshfs라 docker 데몬이 못 봄. mount 우회가 다음 선결과제(§6). 사용자 요청으로 중단.
- 2026-06-07: **doppler_mmr(+diversity) 결과 — diversity 효과 입증, 단 아직 baseline 미달.** 동일 레시피 mode='doppler_mmr'(velo1.0,gamma0.5,sigma0.08) → **NDS 0.5737/mAP 0.4877**(EMA). 4자: cls-base 0.5861 / cls-ft 0.5831 / doppler 0.5628 / **doppler_mmr 0.5737**. → **diversity가 importance-only 손실의 +0.011 회복**(EDA "importance-only 과집중, diversity 회복" 예측을 in-model NDS로 확증=논지 핵심증거). 그러나 cls-ft 대비 여전히 −0.009. mAOE 0.482(baseline 0.515보다 개선)·mAAE 최저 등 일부지표는 우위. MMR 학습 오버헤드 +5%(1.68 vs 1.6 s/it), 마이크로벤치 23ms/call. **가설: velo_weight=1.0 과함 → gentle importance(velo↓)+diversity가 cls 넘을 후보.** ckpt `work_dirs/ft_doppler_mmr/`.
- 2026-06-05(6): **★대조군 결정적 판정 — naive doppler 선택은 해롭다.** 동일 레시피(load_from cls weight, 4-GPU 6ep lr1e-4) **cls-finetune 대조군 NDS 0.5831/mAP 0.5077**(EMA). 3자: cls baseline 0.5861 / cls-ft 0.5831(−0.003, 거의 무해) / doppler-ft 0.5628(−0.020 vs cls-ft). → 하락은 finetune 교란 아님, **doppler 선택 자체가 −0.020 NDS**. 해석: velo_weight=1.0이 cls-confidence를 동등가중으로 덮어 신뢰도낮은 고속 query가 memory 오염, mAVE도 악화(0.20→0.22). EDA의 Doppler 유효성은 raw-point 수준이지 query-선택 수준이 아님. 함의: ①additive blend 부적절(velo_weight↓) ②diversity(doppler_mmr) 결합이 실제 기여일 수 있음(importance-only는 몰림=EDA 예측) ③doppler를 query-init(§4-B)/learned-MLP로 재배치. ckpt `work_dirs/ft_cls/`(epoch1-6 전부 보존).
- 2026-06-05(5): **doppler NDS 궤적(raw, standalone).** ep4 0.5549 / ep5 0.5629 / ep6 0.5622 (ep1–3 ckpt는 max_keep_ckpts=3로 삭제됨). 단조하락 아님 → **상승 후 ~0.563 정체(수렴)**. epoch만 늘려도 회복 X(plateau). epoch6 raw 0.5622 ≈ EMA 0.5628(교차검증). 해석: cls 최적점(0.586)에서 lr1e-4 finetune이 더 나쁜 basin에 수렴. **결정적 다음=cls-finetune 대조군**(동일 레시피) 또는 저-lr/from-scratch.
- 2026-06-05(4): **doppler 1차 finetune 결과(⚠️ baseline 미달).** load_from=cls weight, mode='doppler' velo_weight=1.0, 4-GPU 6ep lr1e-4 → **NDS 0.5628 / mAP 0.4756** (cls baseline 0.5861/0.5091 대비 −0.023/−0.034). 전 클래스 균일 하락 + mAVE 0.202→0.222 악화 → "doppler가 해롭다"기보다 **짧은 re-finetune이 90ep 수렴점을 흩뜬 것**으로 의심. **결정적 대조군 = 동일레시피 mode='cls' finetune**(미실행). 후속: ①cls-finetune 대조군 ②velo_weight↓/epoch↑ ③중간 epoch NDS 궤적. 로그 `train_doppler.log`, ckpt `work_dirs/ft_doppler/`.
- 2026-06-05(3): **2주차 핵심 구현.** `rctrans_head.py`에 physics-guided query 선택 delta(§4-A) 구현: cls top-K → `cls+λ|Doppler|` importance(+MMR diversity 스캐폴드), config 플래그로 전환·baseline 보존. 단위검증+스모크 통과, 학습 투입 준비. 다음: 재학습으로 NDS 비교(cls vs doppler vs doppler_mmr) + baseline latency 표.
- 2026-06-05(2): **mount 해결 + go/no-go PASS.** ① 코드 로컬복사+allow_other sshfs로 데몬가시 확보 ② GPU 스택 end-to-end 검증(sm_89 iou3d 커널 실행, flash-shim 수치) ③ weight/cascade백본/val pkl 배치 ④ **데이터 함정 발견**: nas2_home·`/mnt/.../nuScenes` samples에 CAM_BACK 2개 누락 → nfs4 완전본 `nuscenes_bae`로 교체 ⑤ **재현 평가 NDS 0.5861/mAP 0.5091**(r50 val, go/no-go ✅). 다음: baseline latency 표 → 2주차 Doppler importance 모듈.

---

## 9. 실험 결과 요약 (current bottom-line, 2026-06-08 정리)

**RCTrans r50 256×704, nuScenes val(6,019). 전부 동일 finetune 레시피(load_from=공개 cls weight, 4-GPU, 6ep, lr1e-4) — cls-finetune이 통제군.**

| 모델 | 선택 방식 | NDS | mAP | 비고 |
|---|---|---:|---:|---|
| cls baseline | cls top-K (공개 weight, 학습X) | **0.5861** | 0.5091 | 재현 기준 (go/no-go) |
| cls-finetune | cls top-K | 0.5831 | 0.5077 | 통제군(finetune 거의 무해, −0.003) |
| doppler v1.0 | cls + 1.0·\|velo\| | 0.5628 | 0.4756 | importance-only **해로움 −0.020** |
| doppler_mmr v1.0 | + 공간 diversity(MMR) | 0.5737 | 0.4877 | diversity가 +0.011 회복 |
| **doppler_mmr v0.3** | gentle importance + diversity | **0.5842** | 0.5059 | **완전 회복(≈cls-ft)** *(raw eval, finetune)* |
| **fs dmmr v0.3 (90ep)** | from-scratch, gentle imp+div | 0.5760 | 0.4944 | **★critical-path 완주.** mAVE 0.2017. baseline parity(−0.010), 명확한 win 아님 *(raw eval = in-training val 0.5767 일치)* |
| **qual_mmr** | **학습형 supervised quality head**(sigmoid)=선택점수 + diversity | 0.5822 | 0.5042 | mAVE 0.2082. **deep-research 1순위 root-cause fix(①+⑤).** loss_quality 0.07→0.05 정상수렴(head 실제 학습됨) **그럼에도 parity**(−0.004 vs base, cls-ft와 동률) → **"선택점수 unsupervised라 parity" 가설 기각, 천장 intrinsic 확정** *(raw eval, finetune)* |

**결론(논문 분석 기여로 valid):**
1. query-선택 수준에서 **강한 Doppler importance는 해롭다**(신뢰도 낮은 고속 query가 temporal memory 오염). EDA의 Doppler 유효성은 raw-point 수준이지 검출-query 수준이 아님.
2. **공간 diversity(MMR)가 그 손실을 회복**시킨다 — EDA "importance-only 과집중→diversity 필수" 예측을 in-model NDS로 확증.
3. **gentle importance + diversity가 최적**(단조 트렌드 0.563→0.574→0.584).
4. **한계:** finetune 변종 전부 사전학습 천장(~0.583–0.586)에 수렴 → 아직 baseline 대비 명확한 win 아님.
5. **천장의 본질 = intrinsic(2026-06-15 확증):** literature가 지목한 root-cause fix(①학습형 supervised quality head를 선택점수로, GFL/Sparse4D-v3식)를 정확히 구현·정상 학습(loss_quality 수렴)시켰는데도 qual_mmr 0.5822 = parity. 따라서 천장은 ⓐ선택점수의 학습신호 부재(기각됨)도 ⓑfinetune basin 갇힘(from-scratch도 parity로 기각됨)도 아니라, **cls-score top-K 메모리 선택이 이미 (near-)최적이고 물리·학습형 대체 점수는 매칭만 가능**하다는 구조적 사실.

**미해결 / 다음 후보(우선순위):**
- [x] **from-scratch 90ep 학습** — ✅ 완료(2026-06-14 05:25 KST, 4-GPU 0–3, ~2.9일). **NDS 0.5760/mAP 0.4944 = baseline parity, win 아님** → finetune 천장이 아니라 방법 자체가 cls와 동급. **Outcome B(분석 논문) 최종 확정.** ckpt `work_dirs/fs_dmmr_v03/iter_158220.pth`(=latest), eval `ep90_eval.txt`.
- [x] **★parity 천장 돌파안 1순위(①+⑤) — ✅ 실행·기각(2026-06-15):** deep-research 진단=선택점수 학습 안 됨(train/inference inconsistency). **①학습형 supervised quality head(sigmoid=선택점수, aux objectness loss, Hungarian-matched) + ⑤diversity** `qual_mmr` config로 6ep finetune. loss_quality 0.07→0.05 정상수렴(진단대로 기존 head는 zero-grad였으나 이제 학습됨) **그러나 NDS 0.5822 = parity** → **진단(unsupervised score)이 천장의 원인이 아님이 입증**, 천장 intrinsic 확정(§9 결론5). ckpt `work_dirs/ft_qual_mmr/iter_10548.pth`, eval `eval_qual_mmr.txt`. 코드: `rctrans_head.py` quality_branches + loss_quality(LOCAL 사본만), config `rcdetr_ft_res50_qual{,_mmr}.py`.
- [ ] (선택) 잔여 후보 — 천장 intrinsic 결론이 굳어졌으므로 ROI 낮음: ②Rank-DETR high-order cost, ③DN denoising, ④EMA-teacher distill. 분석 논문 진행엔 불필요.
- [ ] velo_weight 0.1/0.2 미세조정(finetune 천장 안 소폭 우위)
- [ ] baseline query수↓ vs NDS/latency 곡선, latency 표(1·3주차 잔여)
- [ ] query-init(§4-B) / learned-MLP importance 대안 배치

**자산 위치(컨테이너 `rctrans`, 코드 `/home/hanna.bae/rctrans_local/RCTrans`):**
- config: `projects/configs/RCTrans/rcdetr_ft_res50_{doppler,cls,doppler_mmr,dmmr_v03}.py` (+ `rcdetr_90e_res50_doppler.py`)
- ckpt: `work_dirs/ft_{doppler,cls,doppler_mmr,dmmr_v03}/` (각 epoch별 보존)
- 로그: `/home/hanna.bae/rctrans_local/{train_*,eval_*}.log`
- ⚠️ 코드는 sshfs가 아닌 **로컬 복제본**(`rctrans_local`). 원본(`nas2_home/radar/RCTrans`)과 분기됨 — delta는 로컬 복제본에만 있음.
- ⚠️ git: 컨테이너 내 `git config --global --add safe.directory /workspace/RCTrans` 설정해야 auto-eval이 안 죽음.
