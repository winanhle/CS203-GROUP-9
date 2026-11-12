# ITERATION 2 — DEEP RESEARCH & TOPIC SELECTION
## Smartphone No-Reference Image Quality Assessment (NR-IQA): chọn paper Q1, thẩm định SPAQ, thu hẹp câu hỏi khoa học

**Ngày nghiên cứu:** 08/11/2026  
**Mục đích:** thẩm định lại báo cáo `Văn bản đã dán (1)(20261108-165726).txt`, không chấp nhận các khẳng định chưa xác minh trong báo cáo đó là bằng chứng thực thi.  
**Giới hạn tính toán:** Kaggle miễn phí; tối đa 8,5 giờ cho *một phiên* hoàn chỉnh, không tính 8,5 giờ như thời gian được Kaggle bảo đảm.  
**Trạng thái:** research design / source inspection; **chưa** tải dataset đầy đủ, chạy code huấn luyện hay đo runtime trên Kaggle.

---

# EXECUTIVE DECISION — ĐỀ TÀI ĐƯỢC ĐỀ XUẤT

**Proposed Title (chốt ở vòng thiết kế):**  
**When Does EXIF Metadata Help Smartphone Image Quality Assessment? A Controlled Study of Fusion, Missingness, and Capture-Condition Shift**

**Diễn giải:** Khi dự đoán *Mean Opinion Score (MOS)* của ảnh smartphone, thông số chụp EXIF (ISO, exposure time, aperture, v.v.) có cải thiện kết quả so với thông tin ảnh đơn thuần không? Liệu lợi ích đó còn tồn tại khi EXIF thiếu, hoặc khi gặp điều kiện chụp khác phân phối huấn luyện? Cơ chế fusion phi tuyến có thực sự hơn additive fusion hay chỉ tăng độ phức tạp?

**Q1 anchor:** Chen et al., **TOPIQ**, *IEEE Transactions on Image Processing* (**TIP**, 2024; SJR/JCR Q1 theo dữ liệu xếp hạng 2024–2025). DOI: https://doi.org/10.1109/TIP.2024.3378466. Code/weights: https://github.com/chaofengc/IQA-PyTorch; API `topiq_nr` trong https://iqa-pytorch.readthedocs.io/en/latest/ModelCard.html.  
**Dataset/method precedent:** Fang et al., **SPAQ**, *CVPR* 2020 (hội nghị peer-reviewed, **không gọi là journal Q1**). Paper: https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf; repo https://github.com/h4nwei/SPAQ.  
**Related Q1:** Hu et al., *Toward a No-Reference Quality Metric for Camera-Captured Images*, *IEEE Transactions on Cybernetics* 53(6), 2023, DOI https://doi.org/10.1109/TCYB.2021.3128023.  
**Recent Q1 comparator:** Xiang et al., *Learning Decoupled Features With Perceptual Distillation for Blind Image Quality Assessment*, *IEEE TIP* 2026, DOI https://doi.org/10.1109/TIP.2025.3648584, repo https://github.com/JianjunXiang/DFT (có train scripts, checkpoint directory; **chưa thực thi**).  
**Secondary/preprint:** Zhai et al., *PrISM-IQA*, arXiv:2606.31626 (30/06/2026), https://arxiv.org/html/2606.31626, repo https://github.com/Multimedia-Analytics-Laboratory/PrISM-IQA. Không coi là paper journal Q1. Không đưa 34 nhãn của nó vào kế hoạch bắt buộc nếu CSV chưa tải được.

**Research strategy:** reproduction kiểm soát của hai baseline SPAQ (`BL`, `MT-E`) trong môi trường mới ở mức phù hợp tài nguyên; TOPIQ (Q1) dưới vai trò benchmark image-only được kiểm định bằng **weights chưa học SPAQ**; extension chính tập trung vào fusion/robustness, không tuyên bố đã reproduce *toàn bộ quá trình training TOPIQ gốc*.

**Không chọn:** “SPAQ + 53 ISP issues + 4 NR-IQA priors + FiLM + 7 cấu hình × 5 splits” làm kế hoạch bắt buộc: lệ thuộc dữ liệu chưa public trong repo, khó tách yếu tố, compute chưa benchmark, dễ bị data leakage từ pretrained IQA models và khó xây dựng claim nhân quả.

---

# PART A — KIỂM TOÁN BÁO CÁO ĐƯỢC CUNG CẤP

| Claim trong bản trước | Đối chiếu nguồn sơ cấp | Kết luận |
|---|---|---|
| SPAQ có 11.125 ảnh, 66 smartphone, MOS, 5 attributes, 9 scene categories, 7 EXIF tags | Paper và repo Fang et al. 2020 | **VERIFIED-DOCUMENT** |
| SPAQ nguyên bản rút cạnh ngắn về 512, lưu PNG | Phần 3.1 paper SPAQ | **VERIFIED-DOCUMENT** |
| Có tập `spaq.tar` 2,63 GB trên Hugging Face | `q-future/q-align-datasets` hiển thị file 2,63 GB | **VERIFIED-LISTING**, chưa tải/giải nén; phải kiểm tra tương đương ảnh gốc |
| PrISM tạo 53 nhãn ISP cho cùng 11.125 ảnh | Paper arXiv và README repo | **VERIFIED-DOCUMENT** |
| 53 nhãn trên là đánh giá trực tiếp bởi chuyên gia | Paper nêu **GPT-5 tái gán nhãn** cho SPAQ; bộ expert-annotated riêng có 2.983 ảnh, không public được xác minh | **INCORRECT nếu gọi là expert gold labels** |
| PrISM có `merged_labels_renamed.csv` công khai, merge chắc chắn 100% | README chỉ yêu cầu `--master-csv /path/to/merged_labels_renamed.csv`; repo kiểm tra không có master CSV đó | **UNVERIFIED/CRITICAL BLOCKER** |
| “34 nhãn global 100% observed” | 19/53 vấn đề ít hơn 100 mẫu không dùng đánh giá; local issues có N/A; phần còn lại không suy ra không thiếu | **OVERSTATED** |
| Các model cần 9 phút/run trên Kaggle | Không có benchmark; PrISM gốc báo **A100 80 GB, ~3 giờ/50 epochs/run** | **UNVERIFIED**, không thể dùng số từ báo cáo cũ như số đo |
| “SPAQ + EXIF là research gap mới” | SPAQ CVPR 2020 đã triển khai MT-E và phân tích EXIF | **KHÔNG PHẢI NOVELTY**; phải nhấn mạnh robustness/capture shift và kiểm tra tương tác |
| GPU T4 x2 luôn có và là 32 GB VRAM chung cho một model | Kaggle liệt kê T4 x2 và P100, nhưng lựa chọn/availability thay đổi; VRAM T4 không tự hợp nhất | **CONDITIONAL**; ưu tiên cấu hình một GPU |
| 20 GB `/kaggle/working` là toàn bộ disk session | Kaggle ghi 20 GB **auto-saved working** và scratch disk ngoài working; không mặc nhiên dung lượng `/tmp` là vô hạn | **CORRECTED**; kiểm tra `df -h` |
| Q1 có thể áp dụng cho CVPR 2020 | Q1 là journal quartile theo năm/hệ thống; CVPR là conference | **KHÔNG** |

**Đặc biệt:** PrISM-IQA dùng GPT-5 tái chú giải lỗi từ SPAQ (arXiv HTML, Introduction và Appendix B.1); đây là *pseudo-/model-generated annotation*, không phải nguồn human labels tương đương MOS. Đem 34 nhãn ấy làm “bằng chứng thực” không phù hợp nếu không đánh giá độ tin cậy độc lập.  
**Nguồn:** https://arxiv.org/html/2606.31626 ; https://github.com/Multimedia-Analytics-Laboratory/PrISM-IQA.

---

# PART B — BÀI BÁO Q1: THẨM ĐỊNH VÀ CHỌN PAPER

## B1. TOPIQ (Chen et al., IEEE TIP 2024) — **Q1 anchor tốt nhất cho project**

- Full title: *TOPIQ: A Top-Down Approach From Semantics to Distortions for Image Quality Assessment*.
- Authors: Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, Weisi Lin.
- Journal/year: *IEEE Transactions on Image Processing*, 33:2404–2418, 2024.
- DOI https://doi.org/10.1109/TIP.2024.3378466 ; free preprint https://arxiv.org/abs/2308.03060 .
- Q1: IEEE TIP SJR Q1 2024/2025 và JCR Q1 2024/2025 trên các trang tổng hợp chỉ số theo năm (https://www.akaturk.com/journals/25534?lang=en ; https://researchcloud.net/journal/1941-0042). **Chỉ rõ hệ thống/năm khi trích.**
- Method: feature semantic mức cao hướng các tầng đặc trưng thấp qua cross-scale attention; dùng ResNet-50 backbone, áp dụng FR/NR-IQA.
- Code: `chaofengc/IQA-PyTorch`; có inference/training framework, checkpoint configs `topiq_nr` (mặc định pretrained KonIQ) và `topiq_nr-spaq`. **Tuyệt đối không dùng `topiq_nr-spaq` làm zero-shot benchmark trên SPAQ** vì model đã học trên cùng dataset; không biết sample overlap với test split mình chọn.
- Khả thi: kiểm thử inference pretrained weights trên subset SPAQ trước; **fine-tune trọn TOPIQ là tùy chọn**, không cam kết trong một phiên 8,5h cho toàn bộ experiments.
- Phân biệt: sử dụng pretrained TOPIQ để inference ≠ reproduce training và reported metrics paper.

## B2. Hu et al. (IEEE Transactions on Cybernetics, 2023) — **Q1 về cơ chế chất lượng ảnh camera**

- Full title: *Toward a No-Reference Quality Metric for Camera-Captured Images*.
- DOI https://doi.org/10.1109/TCYB.2021.3128023 ; *IEEE Transactions on Cybernetics*, 53(6):3651–3664, 2023.
- Journal JCR Q1 2023–2025 (https://www.akaturk.com/journals/21100274221?lang=en).
- Method: tổ hợp low-level properties (brightness, saturation, contrast, noiseness, sharpness, naturalness) và high-level semantic cues → Support Vector Regression.
- Giá trị: là tài liệu cơ sở cho lập luận **image content × capture artifacts**; tương đối nhẹ compute nếu trích feature.
- Hạn chế: liên kết code trong abstract chỉ đến trang hồ sơ `github.com/YT2015?tab=repositories`, **chưa xác nhận repo cụ thể có script có thể chạy**; không chọn làm core reproduction.

## B3. Xiang et al. (IEEE TIP, 2026) — **Q1 mới hơn, phức tạp hơn**

- Full title: *Learning Decoupled Features With Perceptual Distillation for Blind Image Quality Assessment*.
- Authors: Jianjun Xiang, Yuanjie Dang, Peng Chen, Ronghua Liang, Weisi Lin.
- DOI https://doi.org/10.1109/TIP.2025.3648584 ; *IEEE TIP*, 35:418–433; published 12 Jan 2026.
- Code https://github.com/JianjunXiang/DFT có `train_on_koniq.py`, `train_on_koniq_712.py`, `test_on_koniq_ood.py`, `model/`, `checkpoint/`. **Code exists** nhưng chưa chạy, chưa chắc checkpoints truy cập được và đầy đủ cho reproduction trên SPAQ.
- Contribution: tách global content vs local distortion features bằng knowledge distillation, JND và local distortion attention.
- Xếp sau TOPIQ: compute/training stack phức tạp, lợi ích gia tăng cho *câu hỏi EXIF reliability* không đủ để đánh đổi tính khả thi.

## B4. Related nhưng không phải Q1 journal

- Fang et al., *Perceptual Quality Assessment of Smartphone Photography*, CVPR 2020; dataset và BL/MT-E/MT-A/MT-S. Có `BL_demo.py`, `MT-E_demo.py` và weights; **repo gốc không thấy training entrypoint**. Giữ làm methodological baseline đúng bài toán smartphone.
- Zhai et al., *PrISM-IQA*, arXiv:2606.31626 (2026), **preprint**. Báo cáo 53 issue labels và học ordinal structured; code có `train.py` nhưng repo thiếu label master CSV. Nên coi là literature tương lai, không đưa vào main ablations.
- Hosu et al., *KonIQ-10k: An Ecologically Valid Database for Deep Learning of Blind Image Quality Assessment*, *IEEE TIP*, 2020; DOI https://doi.org/10.1109/TIP.2020.2967829; data cross-dataset đáng tin cậy nhưng **không row-join** với SPAQ.

## B5. Cân bằng giữa Q1 và reproducibility

Nghiên cứu không mạnh hơn chỉ vì paper mới hơn. Quyết định hợp lý là **paper journal Q1 TOPIQ cho anchor/counterpart**, **paper SPAQ CVPR 2020 cho phương pháp và dữ liệu smartphone**, và **khóa chặt việc tái lập đến đúng phần có thể xác minh**. Việc “reproduce Q1 TOPIQ end-to-end” không được ghi là thành công trừ khi cùng dữ liệu/protocol/training/eval hoặc nêu rõ deviations.

---

# PART C — DATASET AUDIT & JOIN DESIGN

| Asset | Link nguồn | Điều đã xác minh | Điều phải thực thi để xác minh |
|---|---|---|---|
| SPAQ primary images + MOS, EXIF, attributes, scene | https://github.com/h4nwei/SPAQ | 11.125 ảnh, MOS [0,100], 7 EXIF, 5 human attributes, 9 multi-label scenes; official Google Drive/Baidu link | tải images+labels, in filenames/schemas, soát join, kiểm tra device identifiers/near-duplicates |
| SPAQ alternate compressed archive | https://huggingface.co/datasets/q-future/q-align-datasets/tree/main | file `spaq.tar`, 2,63 GB list | thử download đúng revision, unpack, image dimensions/file checksums/quality vs official |
| TOPIQ implementation + weights | https://github.com/chaofengc/IQA-PyTorch | có API `topiq_nr`, `topiq_nr-spaq`, pretrained KonIQ default | xác minh load checkpoint, output range, single-image inference latency, license |
| PrISM schema & proposed labels | https://github.com/Multimedia-Analytics-Laboratory/PrISM-IQA | code `label_schema.py`, `generate_cv_splits.py`, 53 labels định nghĩa; 34 eval; **không có master CSV xác minh** | liên hệ nhóm nghiên cứu/tìm official release; không giả lập nhãn |
| KonIQ-10k | https://doi.org/10.1109/TIP.2020.2967829 và HF q-align-datasets | 10.073 ảnh, MOS; `koniq.tar` 5,62 GB listed | kiểm tra metadata, official split, license, nguyên tắc không dùng test để tune |

## Chính sách join nghiêm ngặt

Không khẳng định `sample_id` thuộc [1..11125] là khóa nối chắc chắn chỉ vì ví dụ tên ảnh trông như vậy. Quy trình:
1. Lấy **tên ảnh thực tế** từ mọi nguồn (original PNG, tar variant, MOS table, EXIF table). Xuất `files_manifest.csv` kèm SHA-256, kích cỡ và source URL/revision.
2. Xây dựng hàm canonicalize image key bằng mapping được soát thực tế. Giữ `raw_filename`, `normalized_id`, `source_dataset`; kiểm tra uniqueness, orphan keys, coverage và 10 ảnh spot-check visually.
3. Chỉ inner join khi one-to-one; báo cáo số lượng before/after, missing values từng cột và các ảnh lỗi; **không âm thầm drop records**.
4. EXIF không có giá trị hoặc BrightnessValue được tác giả ước lượng phải đánh dấu provenance (`exif_observed`, `exif_estimated` nếu suy được), không giả định tất cả EXIF là raw observed.
5. Scene labels (9) và human perceptual attributes (5) là **ground-truth supervision chỉ dùng lúc training**. Không được đưa trực tiếp vào đầu vào inference trừ khi có một mô hình dự đoán từ ảnh độc lập và báo cáo overhead/leakage đầy đủ.
6. Phone model/manufacturer: paper ghi 66 phones nhưng **chưa chứng minh từng ảnh có `device_id` dùng được**. Nếu không có key đáng tin cậy, không được tự đặt tên thí nghiệm là unseen-device evaluation.
7. SHA-256 hoặc phash/embedding gần trùng lặp; chú ý khoảng 1.000 ảnh cùng các scene chụp bằng các thiết bị khác nhau (có thể gây shared scene content qua splits).

### Proposed schema — *minh họa*, chưa là cột đã quan sát

```text
sample_id, raw_image_filename, image_path, image_sha256,
mos, device_id?, manufacturer?,
exif_iso, exif_exposure_time, exif_fnumber, exif_focal_length,
exif_brightness, exif_flash, exif_hour,
exif_is_missing_*, exif_is_estimated_*,
attr_brightness, attr_colorfulness, attr_contrast,
attr_noisiness, attr_sharpness,
scene_animal, scene_cityscape, scene_human, scene_indoor,
scene_landscape, scene_night, scene_plant, scene_still_life,
scene_others,
source_revision, split_id
```

**Preprocessing:** ISO và exposure_time nên xem xét `log1p` hoặc log with epsilon hợp lý, circular encode giờ ngày (sin/cos) chỉ nếu thời gian trong nhãn hợp lệ; flash binary; numerical imputation từ TRAIN median; missingness indicators; fit scaling ONLY TRAIN mỗi split. Nếu EXIF field dự đoán bằng công thức từ các field khác, báo cáo redundancy/collinearity.

### Cross-dataset (KonIQ-10k) là một **external test**, không phải row merge

SPAQ và KonIQ-10k khác cảnh, thiết bị, phân phối MOS và quy trình chấm. Khi chạy external zero-shot chỉ nên dùng model image-only hoặc shared EXIF fields thực sự tồn tại trong KonIQ; SRCC ưu tiên hơn absolute MAE trừ khi hiệu chỉnh thang MOS bằng **training/validation nguồn** (không dùng target test labels). Nếu pretrained TOPIQ học KonIQ thì KonIQ không còn là external independent test cho TOPIQ; phải công bố điều đó.

### Dataset licensing

Việc một Hugging Face repository hiển thị `mit` **không chứng minh** ảnh SPAQ/KonIQ bên trong có license MIT tương đương hoặc mọi reuse đều được phép. Cần xem điều kiện nguồn gốc/attribution/redistribution, và không upload lại ảnh sang kho public nếu chưa được phép.

---

# PART D — ITERATION CHỦ ĐỀ: TỪ RỘNG ĐẾN HẸP

## Iteration 0 (báo cáo cũ)
**Reproducing and Extending Smartphone Perceptual Image Quality Assessment via EXIF-Conditioned Multi-Task Learning and ISP Artifact Supervision**.

- Điểm mạnh: nhiều nguồn tính chất EXIF–image–MOS; đáng báo cáo.
- Điểm yếu: 53 PrISM labels chưa có CSV; GPT-5 annotations không phải human ground truth; không đủ compute evidence; 7 cấu hình × 5 splits chưa chắc chạy; thêm priors có thể nhiễm tập SPAQ; nhiều RQ làm khó kết luận.
- **Quyết định:** không chọn nguyên dạng.

## Iteration 1
**EXIF-Conditioned Multimodal Fusion for Smartphone Image Quality Assessment**.

- So sánh image-only vs image+EXIF (additive vs FiLM); dùng SPAQ MOS, pretrained features. Giảm rủi ro dữ liệu.
- Điểm yếu: SPAQ 2020 đã dùng EXIF; chỉ “FiLM cao hơn” thiếu câu hỏi mới và dễ bị xem là architecture tweaking.
- **Quyết định:** giữ làm technical base, tiếp tục thu hẹp.

## Iteration 2 — FINAL PROPOSAL
**When Does EXIF Metadata Help Smartphone Image Quality Assessment? A Controlled Study of Fusion, Missingness, and Capture-Condition Shift**.

- Độc lập khỏi PrISM annotations không có CSV.
- Nghiên cứu khả năng bổ sung **thông tin** (EXIF) khác với việc thay đổi **phương pháp kết hợp** (additive vs gated), và khác với việc bổ sung **điều kiện vận hành** (metadata missingness & shift).
- Có negative outcomes có giá trị: EXIF có thể tăng SRCC nội miền nhưng giảm khi metadata thiếu, hoặc FiLM không vượt simple additive.
- Có trục phân tích explainable: ISO cao, exposure dài, night scene, uncertainty khi missing.
- Có Q1 literature chính và code khả dụng.
- **Novelty claim hạn chế:** là một controlled robustness/ablation study trên dataset và kiến trúc đã có, không khẳng định chưa ai từng nghiên cứu EXIF hoặc metadata dropout.

---

# PART E — RESEARCH PROPOSAL (GỬI GIẢNG VIÊN)

## 1. Proposed Title
**When Does EXIF Metadata Help Smartphone Image Quality Assessment? A Controlled Study of Fusion, Missingness, and Capture-Condition Shift**

## 2. Problem Statement
Với ảnh smartphone RGB `x`, EXIF vector `m`, dự đoán MOS `y∈[0,100]`. Ở hệ thống thực tế, EXIF có thể thiếu/không đồng nhất giữa thiết bị; điều kiện chụp khác train. Phương pháp image-only không dùng metadata; phương pháp image+metadata có thể hữu ích nhưng tăng rủi ro shortcut. Cần nghiên cứu **khi nào** metadata là tín hiệu bổ trợ thực sự.

## 3. Motivation
Thông số ISO/phơi sáng/khẩu độ có liên quan đến noise, blur, exposure; nhưng chúng cũng tương quan với loại điện thoại và loại cảnh. Cần phân biệt predictive association với causal improvement. Missing metadata là một failure mode thực tế và phải có test cố định.

## 4. Related Work
SPAQ CVPR 2020 cho MOS + metadata và các baseline. TOPIQ IEEE TIP 2024 phát triển biểu diễn semantic-distortion NR-IQA. Hu et al. IEEE TCYB 2023 nhấn mạnh low-/high-level cues camera images. Xiang et al. IEEE TIP 2026 phân rã content/distortion. PrISM 2026 preprint chuyển sang ordinal defect diagnosis nhưng label availability/provenance chưa đạt gate để dùng chính.

## 5. Research Gap — *inferred, không khẳng định độc quyền*
Trong khung bài toán SPAQ, chưa có bằng chứng đã xác minh trong nguồn đang đọc về một factorial study **giữ cố định visual encoder và split**, tách tác động (a) EXIF presence, (b) fusion mechanism, (c) metadata missingness robustness, (d) capture-condition holdout. Đó là lỗ hổng *của bộ nguồn đã khảo sát*, không phải lời khẳng định “không tồn tại công trình nào”.

## 6. Research Questions
- **RQ1 (Value):** EXIF có cải thiện SRCC của dự đoán MOS trên test in-distribution khi chỉ thay đổi input metadata, encoder/training protocol cố định không?
- **RQ2 (Mechanism):** Gated interaction có cải thiện hơn additive bias khi giữ cùng metadata, số bước train và độ phức tạp phù hợp không?
- **RQ3 (Reliability):** Khi 25/50/100% EXIF bị thiếu hoặc test tập trung vào điều kiện chụp ít gặp ở train, mô hình nào suy giảm ít hơn?

## 7. Hypotheses
- **H0₁:** ΔSRCC(EXIF-additive, image-only) = 0 trong test phân chia ngẫu nhiên. **H1₁:** ΔSRCC ≠ 0.
- **H0₂:** ΔSRCC(gated, additive) = 0 khi có metadata đầy đủ. **H1₂:** ≠ 0.
- **H0₃:** dropout-style training không cải thiện robustness curve/AUC under missingness khi so với cùng kiến trúc không dropout. **H1₃:** ≠ 0.
- Không viết H1 theo kiểu chắc chắn có tăng; prior kỳ vọng phải được báo cáo như hypothesis, không phải kết quả.

## 8. Variables
- **Independent:** EXIF presence, fusion type, training-time metadata dropout, test-time missingness fraction, split regime.
- **Dependent:** test SRCC **primary**, PLCC và MAE **secondary**, ΔSRCC/retention khi missingness, subgroup residual errors.
- **Controlled:** same image data/resolution/normalization, visual encoder, training budget, train/val/test IDs, optimizer/schedule, seed, early-stop rule.
- **Confounders:** camera identity, scene content, source resolution, image duplication, labels derived from same MOS, distribution/scale shifts, pretrained models possibly previously trained on SPAQ.

## 9. Dataset
SPAQ main; KonIQ-10k optional external evaluation (image-only or subset of shared EXIF, separately protocolled). PrISM issue CSV not core dependency. Exact schema/coverage TBD after data audit.

## 10. Methods & Model Families

**Reference benchmark R-Q1:** `pyiqa.create_metric('topiq_nr')` pretrained on KonIQ (verify checkpoint provenance) evaluated directly on SPAQ test. Report separately as a **pretrained reference model, not matched-data baseline**. Do not use `topiq_nr-spaq` on arbitrary holdout.

**Matched-data core models:** start from ResNet-50 ImageNet `f(x)` with shared MOS head `g(f(x))`. Fast path: frozen encoder + trained lightweight heads; fidelity path: optionally fine-tune last ResNet block, benchmark first. Source SPAQ baseline/MT-E serves as **conceptual reproduction**, labeling adapted protocol; no claim of exact CVPR scores without same conditions.

- **E0 IMAGE-ONLY:** `ŷ = g(f(x))`.
- **E1 ADDITIVE EXIF:** `ŷ = g(f(x)) + b(m)`; same image encoder; `b` small linear/MLP. Simple control aligned SPAQ MT-E.
- **E2 GATED EXIF:** `h'=(1+tanh(γ(m)))⊙h+β(m); ŷ=g(h')`; compare to E1, keep parameter counts similar or present control with parameter-matched additive MLP.
- **E3 ADDITIVE + METADATA DROPOUT:** identical E1 except training stochastic missing EXIF with mask channels.
- **E4 GATED + METADATA DROPOUT:** identical E2 except stochastic missing EXIF; necessary pair E2/E4 to isolate dropout.
- **E5 IMAGE-ONLY + HUMAN ATTRIBUTE AUXILIARY (OPTIONAL):** use the original five human perceptual attributes as auxiliary **training labels only** to isolate multi-task supervision, not as test inputs. Implement only after E0–E4 and gate results.

**Avoid:** stacking model-generated BRISQUE/NIQE/CLIPIQA/PrISM labels into E0–E4; this introduces an extra research dimension and may inherit SPAQ training leakage. Add separate exploratory appendix only after core results.

## 11. Experimental Protocol

**Primary IID split:** 70% train / 10% val / 20% test, deterministic seed, 3 repeats if budget; keep splits locked for all E0–E4; shuffle ID only (never use test for tuning). These proportions are a new experimental protocol, **not necessarily exact official TOPIQ / SPAQ protocol**.

**Metadata missingness:** evaluate at 0%, 25%, 50%, 100% missing EXIF, mask generated independently with reproducible seeds; use train-derived median imputation + explicit missingness flags; 100% missing case must fall back to valid image prediction. Include targeted missingness of high-ISO rows as stress test, label as MNAR simulated, not observed missingness mechanism.

**Capture-condition shift (built from available EXIF):** set aside a pre-specified high-ISO or long-exposure group as domain-shift test, selecting cutoffs from TRAIN-only calibration or a *pre-registered* quantile before final evaluation; ensure nonempty train/val/test and report label distributions. **Caution:** holdout high-ISO shifts both metadata and scene composition; does not establish a causal effect.

**Optional true device shift:** ONLY if each sample has reliable device ID or manufacturer ID. Group split by camera model, ensure no group overlaps train/test; group-aware confidence intervals. If IDs unavailable, report this experiment as *not feasible* rather than simulate device ID.

**Leakage checks:** no original or near-duplicate images across test/train; no matching same scene acquisition groups if identifiable; exclude pretrained IQA checkpoints trained on SPAQ for the *test-as-unseen* claim; scaler trained on train only. Do not add human attribute labels/scene labels directly as inputs at test. Lock data revisions/manifest/hash.

## 12. Evaluation & Statistics
- **Primary:** ΔSRCC on paired test predictions between E0 vs E1, E1 vs E2, E1 vs E3, E2 vs E4.
- **Secondary:** PLCC without post-hoc calibration using test labels, MAE on original MOS scale (fit calibration on train/val only), per-ISO/night-scene performance; deterioration under missingness 0/25/50/100.
- **Uncertainty:** paired bootstrap resampling *same test image IDs* (1,000+ replicates) for ΔSRCC CI; if device clusters exist, also cluster-bootstrap by device. More seeds describe training variance. Paired sample-level error tests only if test appropriate; do **not** use paired t-test on five per-image SRCC values (SRCC is set-level).
- **No cherry-picking:** pre-register one primary hypothesis and two secondary comparisons; report all models including nonimprovements; do not claim p<0.05 in advance; Holm correction if multiple tests are formally declared.

## 13. Expected Contributions
1. Reproducible SPAQ–EXIF dataset audit, split manifests and strong code baselines.
2. Controlled quantitative evidence of EXIF's incremental predictive value vs image-only features.
3. Explicit comparison of additive and gated fusion, isolated from metadata-dropout training.
4. Robustness curves under absent/partial metadata and capture-condition distribution shift.
5. Failure-case plots illustrating high ISO, exposure, and scene combinations. **No claim of SOTA invention.**

## 14. Limitations & Risks
- Repository PrISM does not ship issue label CSV; not core.
- Need check original SPAQ annotations download and legal access; HF tar may be a resized/re-encoded variant.
- Camera manufacturer information may not be per image; no device-shift claim before audit.
- TOPIQ trained on KonIQ cannot be compared as same-trained baseline to ImageNet+SPAQ models; present separate pretrained-reference table.
- Output MOS is subjective; capture metadata correlates with causal exposure physics but observational results do not prove causality.
- Random image split may overestimate generalization due to shared scenes/similar images.
- Large test group correlations have confidence intervals; tiny subgroups may yield unstable SRCC.
- TOPIQ model FLOPs and Kaggle session runtime must be measured, not inferred from paper's relative FLOPs.

## 15. Computational Feasibility

**Kaggle verified docs:** https://www.kaggle.com/docs/notebooks (12h max GPU/CPU sessions, 20GB saved `/kaggle/working`, scratch disk; options P100/T4x2), https://www.kaggle.com/docs/efficient-gpu-usage (~30h/week sometimes variable; quota depends on account/resource). **Project limit remains 8,5h**, regardless of platform max. 2×T4 != one 32GB unified memory device.

**No measured runtime yet.** Proposed *allocation ceiling* (NOT measured time):

| Block | Time allocation ceiling | Gate |
|---|---:|---|
| Environment, inventory, version pinning | 0.50h | imports, GPU memory, dataset ready |
| Load/data audit & split generation | 0.70h | join, hashes, missing values report |
| Frozen encoder / reference TOPIQ feature inference & caching | 1.25h | benchmark on 300 images, extrapolate measured throughput |
| Train 5 core lightweight variants × planned repeats | 2.75h | only proceed after measured epoch time |
| IID + missingness + conditional shift evaluation | 0.95h | all fixed test protocols |
| Statistics, qualitative figures, packaging | 0.55h | final CSV + figures + reproducibility config |
| **Buffer** | **1.80h** | safe-stop/resume |
| **Total ceiling** | **8.50h** | **non-empirical planning budget** |

The real planned workload is **6.7h**, buffer **1.8h** (~21%). Before full experiments, compute:

`T_total_est = T_setup_measured + (N_images / images_per_s) + Σ_runs(epochs × steps_per_epoch × seconds_per_step + eval_overhead) + T_stats + T_buffer`.

If measured full projected time >6.7h exclusive of buffer, reduce `n_seeds` or freeze encoder and train lightweight heads; preserve matched conditions and **do not silently remove the principal contrasts**. If environment cannot download/train at all, stop at minimum viable E0/E1/E3 with one fixed split, explicitly marking exploratory/no strong statistical claims.

---

# PART F — DECISION GATES & ROADMAP

| Gate | Action | Pass condition | Fail/fallback |
|---|---|---|---|
| **G0: DATA** | Download primary SPAQ images, official MOS/EXIF tables and inspect tar | 11,125 target IDs justified by source; MOS label matching and enough EXIF coverage; license noted | Recheck official Google Drive/Baidu; no spoofed labels; if unavailable, do not start this project |
| **G1: SOURCE** | Run TOPIQ KonIQ-pretrained inference on 10 examples; run SPAQ BL demo with original checkpoint if downloadable | outputs finite, model provenance audited | use ResNet50 ImageNet base; Q1 paper stays related benchmark, no fake reproduction |
| **G2: SPLIT** | Build fixed train/val/test, duplicate scan, EXIF missingness and camera key audit | zero overlaps; shift regime explicitly justified | use IID+missingness only; drop device shift claim |
| **G3: SMOKE** | In Kaggle, sample 300 images; time loading, TOPIQ forward, ResNet50 extraction and 20 train steps; measure peak GPU memory | projected runtime ≤6.7h + 1.8h buffer | freeze encoder, reduce seeds/epochs, or switch to minimum viable design |
| **G4: BASELINE** | Train E0 and E1 with matched protocols, save predictions | SRCC/PLCC/MAE valid; absence of test tuning | debugging; do not present reference paper metrics as own results |
| **G5: CONTRASTS** | Train E2/E3/E4, missingness evaluation | key contrasts computed on shared IDs | stop after E0/E1/E3 and report restricted inference |
| **G6: ANALYSIS** | Paired bootstrap, error cases, 95% CI; external test conditional | full provenance and unambiguous figures | describe null/negative findings faithfully |

**Minimum viable research:** image-only E0, additive EXIF E1, additive + dropout E3 on the same SPAQ fixed split; test both 0% and 100% metadata availability, report paired bootstrap CI for ΔSRCC. This answers a constrained scientific question even when gating architecture and PrISM data are not ready.

**Strong target:** 5 core runs E0–E4 × 3 seeds (with frozen encoder and precomputed embeddings) + capture-condition subgroup analysis. **Not guaranteed** without Kaggle smoke test.

---

# PART G — SPECIFICATION FOR CODEX (sau khi data gate đạt)

## Repository proposal

```text
spaq-exif-robustness/
  README.md
  LICENSE_AND_DATA_PROVENANCE.md
  configs/{data.yaml,experiment.yaml,seeds.yaml}
  scripts/{fetch_manifest.py,audit_data.py,smoke_kaggle.py,run_ablation.py,analyze.py}
  src/{data.py,features.py,models.py,train.py,evaluate.py,statistics.py}
  tests/{test_join.py,test_split.py,test_leakage.py,test_masking.py,test_metrics.py}
  data/{raw_manifest.csv,canonical_mapping.csv,labels_master.parquet,splits.json}
  results/{run_manifest.json,smoke_test.json,metrics.csv,predictions.parquet,figures/}
```

## Non-negotiable rules for coding agent

1. Read paper + original repo and pin commit SHA before code; do not pretend model TOPIQ checkpoint has never seen SPAQ without checking its pretraining dataset.
2. Never invent a dataset URL, private file, camera-ID table, label column, license, training accuracy, epoch runtime or checkpoint path.
3. Do not create fake PrISM ordinal labels; no synthetic “ground truth” from `GPT-5` without labeling it model-generated and auditing.
4. Do not use test data for feature normalization, thresholds, hyperparameters, checkpoint selection or significance model selection.
5. Each experimental change has a corresponding control; preserve identical split IDs, seeds, preprocessing and training budgets.
6. Save image ID, MOS, prediction, model, split, seed, metadata mask, checkpoint hash and preprocessing config in outputs. Keep a `provenance.yaml` for each dataset/weight.
7. If code train scripts exist but dependencies/checkpoints fail, report `UNVERIFIED_BY_EXECUTION`, not “reproduction success”.
8. Generate at least five figures: MOS distribution + EXIF missingness; per-method SRCC with 95% CI; missingness degradation curves; ISO/night subgroup performance; image qualitative error pairs.
9. Do not push images or third-party checkpoints to a public repo until license requirements checked.
10. Process must emit a `research_readiness_report.md` after G0–G3 with PASS/FAIL, actual observed data counts and measured Kaggle timing before full training.

## Preflight pseudocode (not execution claim)

```python
# Run this in Kaggle after attaching legitimate data and checkpoint sources.
from pathlib import Path
import torch

print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM GB:', torch.cuda.get_device_properties(0).total_memory / 1e9)

# 1. read actual annotation headers, sample filenames, directory sizes
# 2. verify unique mapping + MOS and EXIF availability
# 3. enforce no duplicate/near-duplicate leakage across splits
# 4. load TOPIQ checkpoint trained on KonIQ, not SPAQ
# 5. benchmark 300-image inference + 20 training steps + VRAM
# 6. compute projected runtime and STOP if above 6.7h workload
```

---

# REFERENCE & EVIDENCE REGISTER

**Primary paper/dataset:**
1. Fang et al. 2020, *Perceptual Quality Assessment of Smartphone Photography*, CVPR. https://doi.org/10.1109/CVPR42600.2020.00373 ; https://github.com/h4nwei/SPAQ .
2. Chen et al. 2024, *TOPIQ: A Top-Down Approach From Semantics to Distortions for Image Quality Assessment*, IEEE TIP 33, 2404–2418. https://doi.org/10.1109/TIP.2024.3378466 ; https://github.com/chaofengc/IQA-PyTorch .
3. Hu et al. 2023, *Toward a No-Reference Quality Metric for Camera-Captured Images*, IEEE Trans Cybern 53(6), 3651–3664. https://doi.org/10.1109/TCYB.2021.3128023 .
4. Xiang et al. 2026, *Learning Decoupled Features With Perceptual Distillation for Blind Image Quality Assessment*, IEEE TIP 35, 418–433. https://doi.org/10.1109/TIP.2025.3648584 ; https://github.com/JianjunXiang/DFT .
5. Hosu et al. 2020, *KonIQ-10k: An Ecologically Valid Database for Deep Learning of Blind Image Quality Assessment*, IEEE TIP 29, 4041–4056. https://doi.org/10.1109/TIP.2020.2967829 .
6. Zhai et al. 2026, *PrISM-IQA: Image Quality Assessment Made Practical for Smartphone Photography*, arXiv preprint. https://arxiv.org/html/2606.31626 ; https://github.com/Multimedia-Analytics-Laboratory/PrISM-IQA .

**Repositories/dataset hosts:**
7. IQA-PyTorch model cards: https://iqa-pytorch.readthedocs.io/en/latest/ModelCard.html .
8. Q-Align dataset archive listings: https://huggingface.co/datasets/q-future/q-align-datasets/tree/main .

**Quartile evidence (secondary aggregators; year-specific; clarify not publishers):**
9. IEEE TIP JCR/SJR 2024–25: https://www.akaturk.com/journals/25534?lang=en .
10. IEEE TCYB JCR 2023–25: https://www.akaturk.com/journals/21100274221?lang=en .

**Kaggle platform first party:**
11. https://www.kaggle.com/docs/notebooks .
12. https://www.kaggle.com/docs/efficient-gpu-usage .

## Final research integrity statement

- **Verified documentary:** existence and venues of TOPIQ, DFT, Hu, SPAQ; official source/repo structures; paper-reported SPAQ schema, 11,125 size; PrISM preprint/GPT-5 reconstructed labels; public HF tar listing; Kaggle published limits.
- **Estimated:** proposed model/training budget, usefulness of FiLM, expected metadata robustness, confidence interval precision, dataset join coverage beyond published schema.
- **Unverified by execution:** images and labels fully downloaded, actual checksum, exact join keys, individual camera IDs, TOPIQ weights load in Kaggle, repo training scripts reproducibility, VRAM, epoch runtime, all performance metrics.

**Do not report numerical results before computing them.** If G0 data access fails or grouped device IDs are unavailable, revise the research question explicitly; do not forge device-domain tests.
