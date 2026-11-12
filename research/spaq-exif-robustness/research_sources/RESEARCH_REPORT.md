# RESEARCH REPORT
## Khi nào EXIF Metadata thực sự hữu ích trong đánh giá chất lượng ảnh chụp điện thoại?

**Ngày:** 09/11/2026  
**Proposed English title:** **When Does EXIF Metadata Help Smartphone Image Quality Assessment? A Controlled Study of Fusion, Missingness, and Capture-Condition Shift**  
**Lĩnh vực:** Computer Vision · Machine Learning · No-Reference Image Quality Assessment · Multimodal Learning  
**Mục đích tài liệu:** Mô tả và thuyết minh **ý tưởng đề tài nghiên cứu** để trao đổi với giảng viên và nhóm. Đây **không phải** technical specification, task list hoặc prompt giao Codex.  
**Trạng thái:** Đã nghiên cứu và đối chiếu nguồn học thuật/README; **chưa** chạy huấn luyện, kiểm định dữ liệu tải về, hay đo runtime trên Kaggle.

---

# TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Đề tài nghiên cứu bài toán **dự đoán chất lượng cảm nhận của con người đối với ảnh chụp điện thoại** mà không cần ảnh chuẩn để đối chiếu (**No-Reference/Blind Image Quality Assessment — NR-IQA/BIQA**). Mỗi ảnh được gán một **Mean Opinion Score (MOS)**; mô hình học cách dự đoán MOS từ ảnh RGB và, nếu có, các **thông số chụp EXIF** như ISO, exposure time, f-number, focal length.

Nền tảng trực tiếp của đề tài là **Fang et al. (CVPR 2020)**, *Perceptual Quality Assessment of Smartphone Photography*, công trình công bố **SPAQ** với 11.125 ảnh từ 66 điện thoại và các mô hình **BL / MT-E / MT-A / MT-S**. Tác giả đã chứng minh rằng kết hợp EXIF theo cơ chế **additive offset** giúp cải thiện SRCC từ **0,908 (BL)** lên **0,926 (MT-E)** trên SPAQ, trung bình 5 phiên đánh giá. Đây là **kết quả của paper gốc, không phải kết quả nhóm chúng ta** [1, Sec. 5.5, Table 4].

Để bảo đảm nền tảng **tạp chí Q1**, báo cáo đồng thời chọn **Chen et al. (IEEE Transactions on Image Processing, 2024)**, *TOPIQ: A Top-Down Approach From Semantics to Distortions for Image Quality Assessment*, làm **Q1 reference paper và benchmark image-only hiện đại**. TOPIQ sử dụng cơ chế top-down/cross-scale attention để kết nối ngữ nghĩa ảnh với các vùng biến dạng cục bộ. Trong thực nghiệm của TOPIQ, kết quả **in-dataset** trên SPAQ của TOPIQ (ResNet50) là **SRCC 0,921; PLCC 0,924** [5, Table VII]; đây là **giao thức và mô hình riêng**, không thể so trực tiếp như một ablation cùng điều kiện huấn luyện với MT-E.

**Đóng góp mới được đề xuất không phải là “lần đầu tiên dùng EXIF”.** Paper SPAQ đã làm điều đó từ năm 2020. Nghiên cứu này tập trung vào **độ tin cậy và điều kiện hiệu quả của EXIF**: (1) EXIF bổ sung bao nhiêu thông tin? (2) cơ chế fusion phi tuyến có thực sự vượt additive baseline? (3) mô hình suy giảm ra sao khi EXIF thiếu hoặc test data có điều kiện chụp khác train data? Phương pháp là **reproduction có điều chỉnh tài nguyên + controlled extension + ablation + robustness evaluation**. Các kết quả âm vẫn có ý nghĩa vì kiểm tra một giả thuyết cụ thể.

> **Quyết định định hướng:** Sử dụng **SPAQ/CVPR 2020** làm **paper gốc trực tiếp về dataset và EXIF**, **TOPIQ/IEEE TIP 2024** làm **paper Q1 nền tảng và benchmark liên quan**, và đề xuất một **nghiên cứu mở rộng độc lập** về giá trị và sự thiếu hụt EXIF. **Không mô tả việc sử dụng checkpoint TOPIQ là tái lập quá trình huấn luyện TOPIQ.**

---

# 1. BỐI CẢNH VẤN ĐỀ VÀ ĐỘNG LỰC NGHIÊN CỨU

## 1.1. Vấn đề thực tế

Ảnh từ điện thoại có thể bị nhiễu do ISO cao, thiếu sáng, motion blur khi thời gian phơi sáng dài, out-of-focus blur, over-/under-exposure hoặc suy giảm màu sắc do chuỗi xử lý ảnh. Các suy giảm thường xuất hiện **đồng thời**; do đó những mô hình chỉ học trên nhiễu tổng hợp chưa chắc phản ánh mức độ hài lòng thực tế của người xem. Chất lượng ảnh là một biến liên tục và có yếu tố chủ quan, không phải một nhãn phân loại đúng/sai đơn giản.

**Đầu vào dự kiến:** ảnh RGB `x` và một vector EXIF `m` có thể thiếu một phần.  
**Đầu ra:** điểm dự đoán chất lượng cảm nhận `ŷ` (theo thang MOS của dataset) và, ở phần phân tích, độ tin cậy/khả năng suy giảm khi metadata thiếu.  
**Ground truth:** MOS được tổng hợp từ người chấm trong SPAQ, không phải “độ đẹp ảnh” do AI tự đánh giá.

### Ví dụ giải thích vấn đề

Hai bức ảnh có thể trông tương tự ở độ phân giải thumbnail nhưng được chụp ở ISO/phơi sáng khác nhau; EXIF cung cấp tín hiệu về nguy cơ noise/blur, song **không quyết định chất lượng**. ISO cao trên cảnh tối có thể phản ánh bối cảnh khó thay vì ảnh kém. Từ đó nảy sinh câu hỏi: **mô hình có học thêm quy luật hữu ích, hay chỉ ghi nhớ shortcut tương quan giữa loại cảnh/thiết bị và MOS?**

## 1.2. Tại sao đề tài phù hợp với Scientific Method?

- Có **bài toán đo lường được** (MOS regression) và chỉ số đánh giá phù hợp (SRCC, PLCC, MAE).
- Có **paper gốc**, baseline rõ ràng, dataset có ảnh và metadata, có thể đặt thí nghiệm đối chứng.
- Có **biến độc lập** tách được: có/không EXIF; additive/gated fusion; có/không metadata dropout; điều kiện test.
- Có giả thuyết **bác bỏ được**: thêm EXIF **có thể không giúp**, hoặc giúp ở điều kiện IID nhưng gây giảm độ bền vững khi thiếu metadata.
- Có minh họa trực quan: ảnh nhiễu/nhòe, bảng EXIF, residual plots, đường cong SRCC theo tỷ lệ thiếu metadata.

---

# 2. PAPER GỐC TRỰC TIẾP: FANG ET AL. — SPAQ (CVPR 2020)

## 2.1. Thông tin thư mục **đầy đủ**

| Thuộc tính | Thông tin đã kiểm chứng |
|---|---|
| **Full title** | **Perceptual Quality Assessment of Smartphone Photography** |
| **Authors (đúng thứ tự)** | **Yuming Fang, Hanwei Zhu, Yan Zeng, Kede Ma, Zhou Wang** |
| **Equal contributions** | Yuming Fang và Hanwei Zhu (đánh dấu * trong bản chính thức) |
| **Corresponding author** | Kede Ma |
| **Cơ quan tác giả** | Jiangxi University of Finance and Economics; City University of Hong Kong; University of Waterloo |
| **Publication** | *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (**CVPR 2020**) |
| **Năm / địa điểm hội nghị** | 2020; Seattle, Hoa Kỳ (hội nghị trực tuyến), 13–19/06/2020 |
| **Trang theo CVF proceedings** | **3677–3686** (một số metadata của cơ sở dữ liệu đại học dùng dải trang khác, cần trích nhất quán theo bản CVF) |
| **DOI** | **10.1109/CVPR42600.2020.00373** |
| **Published / peer-reviewed?** | **Có**: paper hội nghị CVPR, đã công bố chính thức |
| **Q1?** | **Không dùng nhãn Q1 cho CVPR**. Q1/Q2 là phân hạng tạp chí; CVPR là hội nghị |
| **Official paper page** | https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.html |
| **Paper PDF** | https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf |
| **Supplementary PDF** | https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Fang_Perceptual_Quality_Assessment_CVPR_2020_supplemental.pdf (liên kết từ trang tác giả; khi kiểm tra trực tiếp máy chủ trả 403, chưa đọc được trong phiên này) |
| **Official source code / dataset portal** | https://github.com/h4nwei/SPAQ |
| **IEEE Xplore** | https://ieeexplore.ieee.org/document/9156490 |

*Nguồn đối chiếu:* CVF Open Access [1], IEEE Xplore [2], GitHub chính chủ [3].

## 2.2. Mục tiêu và câu hỏi nghiên cứu của paper

Paper không chỉ tạo một mô hình dự đoán MOS. Nó trả lời bốn nhóm vấn đề:

1. Làm thế nào xây dựng database lớn và đáng tin cậy cho **realistic smartphone distortions** thay vì chỉ có synthetic distortion?
2. Mức độ chất lượng ảnh có quan hệ như thế nào với **EXIF camera settings**, **perceptual attributes** và **scene semantics**?
3. Có thể xây dựng **blind IQA model** dự đoán MOS từ ảnh thật bằng deep learning không?
4. Học kết hợp thông tin phụ trợ bằng **multi-task learning** có cải thiện chất lượng dự đoán không?

## 2.3. Đóng góp học thuật do **tác giả** tuyên bố

- Giới thiệu **SPAQ (Smartphone Photography Attribute and Quality)** — cơ sở dữ liệu ảnh chụp điện thoại cùng nhiều loại annotation.
- Thiết lập khảo sát chất lượng chủ quan tại phòng thí nghiệm, thu điểm **MOS**, 5 image attributes và 9 scene labels.
- Phân tích tương quan giữa metadata/attributes/scenes và chất lượng cảm nhận, bao gồm so sánh nhiều camera.
- Đề xuất **BL, MT-E, MT-A, MT-S** để nghiên cứu tác dụng của dữ liệu phụ trợ lên dự đoán MOS [1, Secs. 1, 3–5].

## 2.4. SPAQ được xây dựng như thế nào?

**Quy mô gốc:** 11.125 ảnh từ **66 smartphone cameras**, thuộc **11 nhà sản xuất**. Một subset **1.000 ảnh** được chụp tại **50 cảnh bằng 20 camera khác nhau** nhằm hỗ trợ so sánh smartphone. Ảnh ban đầu độ phân giải cao, được downsample sao cho **cạnh ngắn bằng 512 pixels**, lưu **PNG**. Các suy giảm thực tế gồm sensor noise, motion/out-of-focus blur, contrast reduction, under/over-exposure, color shift và các tổ hợp [1, Secs. 3.1, 4.2].

**Cấu trúc nhãn được công bố:**

| Nhóm dữ liệu | Nội dung | Vai trò trong nghiên cứu |
|---|---|---|
| RGB image | Ảnh smartphone | Input visual `x` |
| MOS | Điểm chất lượng **0–100**, cao tốt hơn | **Target `y`** |
| 5 perceptual attributes | Brightness; Colorfulness; Contrast; Noisiness; Sharpness | Auxiliary training labels / phân tích |
| 9 scene labels | Animal; Cityscape; Human; Indoor scene; Landscape; Night scene; Plant; Still life; Others | Multi-label context / phân tích shift |
| 7 EXIF tags | Focal length; F-number; Exposure time; ISO; Brightness value; Flash; Capture time | Candidate metadata `m` |

**Nguồn gốc nhãn:** Hơn **600 người** tham gia subjective testing trong phòng thí nghiệm được kiểm soát; chấm chất lượng ảnh và thuộc tính liên quan theo thang 0–100. Scene labels có thể có **nhiều nhãn cho một ảnh**. Paper nêu chi tiết kiểm soát độ tin cậy và loại outlier trong **supplementary**; vì supplemental PDF chưa được truy xuất ở phiên này, không đưa những con số chi tiết về tổng lượt chấm/lượt loại mà chưa kiểm tra được [1, Sec. 3.2].

**Lưu ý chất lượng EXIF:** `Brightness value` không phải lúc nào cũng được nhà sản xuất lưu. Tác giả dùng **ước lượng từ exposure equation** khi giá trị bị thiếu [1, Sec. 3.1]. Khi làm đề tài mới cần phân biệt `observed` và `derived/imputed` nếu có provenance đủ tin cậy.

## 2.5. Kiến trúc và công thức của bốn mô hình gốc

### BL — Image-only baseline (Sec. 5.1, Eq. 1)

- Backbone: **ResNet-50** khởi tạo từ ImageNet pretrained weights.
- Bỏ softmax, đổi final fully connected thành **một scalar** dự đoán MOS.
- Tối ưu loss **L1**:

$$
\mathcal L_{BL}=\sum_{i=1}^{N}\left|y_i-\hat y_i\right|.
$$

- Không sử dụng những phép tiền xử lý làm thay đổi mạnh chất lượng cảm nhận, ví dụ global contrast normalization.

### MT-E — Multi-Task Learning from EXIF Tags (Sec. 5.2, Eqs. 2–3)

- Nhánh ảnh dự đoán **generic quality** $g(x)$.
- Nhánh EXIF nhận vector `m`, qua một tầng fully connected để dự đoán **scalar offset** $b(m)$.
- Điểm cuối:

$$
\hat y = g(x)+b(m).
$$

- Loss có hai thành phần: score generic và score sau EXIF, trọng số tác giả **0,5 / 0,5**:

$$
\mathcal L_{MT-E}=0.5\,\|y-g(x)\|_1+0.5\,\|y-[g(x)+b(m)]\|_1.
$$

- Paper xem khả năng quay lại `g(x)` khi không có EXIF là một lợi ích khái niệm. Tuy nhiên mức **robustness khi EXIF thiếu một phần hoặc mất có hệ thống chưa được chứng minh bằng stress-test trong Table 4**.

### MT-A — Multi-Task Learning from Image Attributes (Sec. 5.3, Eq. 4)

- ResNet-50 dùng shared features, output 6 scalar: 1 MOS + 5 attribute scores.
- Loss là tổng có trọng số của L1 quality và L1 cho 5 attributes; biến thiên trọng số giữa các task.
- **Quan trọng:** điểm thuộc tính của người chấm là **training supervision**, không phải các giá trị ground-truth có sẵn lúc người dùng đưa một ảnh mới vào inference.

### MT-S — Multi-Task Learning from Scene Labels (Sec. 5.4, Eqs. 5–8)

- Chia shared backbone thành hai nhánh quality regression / scene recognition.
- Scene task nhận các nhãn multiple-category; mạng dùng softmax/cross entropy theo cách định nghĩa trong paper.
- Hai loss có scale khác nhau, tác giả dùng phương pháp **task-dependent uncertainty weighting** với các biến $\sigma_1,\sigma_2$ để học trọng số.
- Tác giả phân tích vị trí tách nhánh (**Conv1, Conv10, Conv22, Conv40, Conv49**), không chỉ thử một kiến trúc duy nhất.

## 2.6. Giao thức thực nghiệm paper gốc (Sec. 5.5)

| Thiết lập | Giá trị từ paper |
|---|---|
| Image split | **80% train / 20% test**, chia ngẫu nhiên |
| Repetitions | **5** lần huấn luyện/đánh giá, lấy trung bình |
| Backbone | ResNet-50 pretrained ImageNet |
| Epochs | **30** |
| Batch size | **16** |
| Optimizer | **Adam** |
| Initial learning rate | **1×10⁻³** |
| LR schedule | Giảm hệ số **0,1 mỗi 10 epochs** |
| Training crop | Random **224×224×3** |
| Training phases | **10 epochs đầu freeze backbone/train FC; 20 epochs sau fine-tune toàn mạng** |
| Test-time | Sliding patches **224×224**, stride **112**, trung bình các dự đoán |
| Main metrics | **SRCC**, **PLCC** |
| Các model đối chứng | QAC, DIIVINE, CORNIA, ILNIQE, BRISQUE, FRIQUEE, DB-CNN |

**Phân biệt reproduction:** Nếu nhóm dùng frozen encoder/feature caching hoặc chia 70/10/20 thì đây là **adapted replication**, không thể gọi là reproduction đầy đủ 30 epochs theo nguyên protocol tác giả. Nếu tái hiện kết quả 80/20 × 5 lần, cũng cần kiểm tra liệu seed/split có khớp chính xác (paper không phát hành sẵn một official fixed split trong nguồn đã kiểm tra).

## 2.7. Kết quả gốc **đúng theo Table 4**

Bảng sau **chép lại các giá trị metric công bố trong paper SPAQ**, không phải do nhóm tính toán:

| Mô hình | SRCC ↑ | PLCC ↑ | Vai trò |
|---|---:|---:|---|
| QAC | 0,092 | 0,497 | Traditional baseline |
| DIIVINE | 0,599 | 0,600 | Traditional baseline |
| CORNIA | 0,709 | 0,725 | Codebook baseline |
| ILNIQE | 0,713 | 0,721 | Traditional baseline |
| BRISQUE | 0,809 | 0,817 | NSS baseline |
| FRIQUEE | 0,819 | 0,830 | NSS baseline |
| DB-CNN | 0,911 | 0,915 | Strong deep-learning baseline |
| **BL (RGB only)** | **0,908** | **0,909** | **Paper's ResNet-50 baseline** |
| **MT-E (RGB + EXIF)** | **0,926** | **0,932** | **Paper's best listed SRCC/PLCC** |
| MT-A (attributes) | 0,916 | 0,916 | Multi-task attributes |
| MT-S (scene semantics) | 0,917 | 0,921 | Multi-task scenes |

**Hiệu quả EXIF trong paper:** $\Delta\mathrm{SRCC}=0,018$; $\Delta\mathrm{PLCC}=0,023$ khi so **MT-E** với **BL** [1, Table 4]. Đây là hiệu quả trên thiết lập gốc; **không** chứng minh rằng EXIF luôn giúp ở tập dữ liệu khác hoặc khi missingness xảy ra.

**Kết quả phân tích liên quan:** Table 2 báo cáo SRCC giữa MOS và điểm attribute từ người chấm lần lượt là Brightness **0,784**; Colorfulness **0,844**; Contrast **0,874**; Noisiness **0,893**; Sharpness **0,958**. Các tương quan lớn là **association**, không phải quan hệ nhân quả trực tiếp. Table 6 cho thấy hiệu quả scene branch thay đổi theo vị trí split; Conv40 có SRCC **0,917**, PLCC **0,921**, nhưng accuracy phân loại scene **0,673** [1, Tables 2, 6].

## 2.8. Điểm mạnh và giới hạn của SPAQ paper

**Điểm mạnh:** Dataset chuyên biệt smartphone với MOS, nhiều nhóm annotation, EXIF; baseline tương đối đơn giản; kết quả EXIF đã được kiểm chứng bằng các thí nghiệm trong paper; thiết lập đánh giá được mô tả rõ.

**Giới hạn liên quan trực tiếp đề tài mới:**

- **Random image-level split** có thể lạc quan nếu ảnh chụp cùng cảnh/thiết bị nằm ở cả train/test; paper có một subset chụp cùng cảnh từ nhiều camera.
- MT-E dùng **EXIF → scalar additive offset**, chưa trực tiếp kiểm tra tương tác nonlinear giữa đặc trưng ảnh và cấu hình camera.
- Các thí nghiệm báo cáo chính **không có đường cong robustness theo EXIF missingness 0–100%** hoặc đánh giá capture-condition holdout theo một protocol định trước.
- Repository **có demo và checkpoint**, nhưng trong danh sách file đã kiểm tra **không thấy training script đầy đủ cho tất cả các cấu hình**.
- Việc thêm EXIF **đã được paper chứng minh**: không thể xem “dùng EXIF” tự nó là research novelty của chúng ta.

---

# 3. PAPER TẠP CHÍ Q1: CHEN ET AL. — TOPIQ (IEEE TIP 2024)

## 3.1. Thông tin bibliographic

| Thuộc tính | Thông tin |
|---|---|
| **Title** | **TOPIQ: A Top-Down Approach From Semantics to Distortions for Image Quality Assessment** |
| **Authors** | **Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, Weisi Lin** |
| **Journal** | **IEEE Transactions on Image Processing (TIP)** |
| **Year / volume / pages** | **2024**, vol. **33**, pp. **2404–2418** |
| **DOI** | **10.1109/TIP.2024.3378466** |
| **Academic status** | Peer-reviewed **journal article**, không phải preprint đơn thuần |
| **Quartile** | **Q1 theo JCR và SJR năm 2024–2025** được hiển thị trong các nguồn tổng hợp số liệu xếp hạng [7]. Nên kiểm lại trên quyền truy cập JCR/SCImago của trường nếu cần chứng thư xếp hạng chính thức. |
| **Publisher / bibliographic** | https://pubmed.ncbi.nlm.nih.gov/38517711/ |
| **Accessible preprint** | https://arxiv.org/abs/2308.03060 ; https://arxiv.org/html/2308.03060v1 |
| **Official code** | https://github.com/chaofengc/IQA-PyTorch |
| **Model card** | https://iqa-pytorch.readthedocs.io/en/latest/ModelCard.html |

## 3.2. TOPIQ giải quyết vấn đề gì?

Mô hình IQA thông thường dùng biểu diễn đa tỉ lệ nhưng thường hợp nhất đặc trưng một cách tuyến tính/bottom-up. TOPIQ đặt giả thuyết rằng con người trước hết nhận biết **ngữ nghĩa tổng thể** (đâu là đối tượng quan trọng), sau đó tập trung vào **biến dạng cục bộ tại các vùng quan trọng**. Mô hình đề xuất **Coarse-to-Fine Attention Network (CFANet)** với **Cross-Scale Attention (CSA)** nhằm truyền thông tin semantic từ high-level sang low-level features; **Gated Local Pooling (GLP)** giảm chi phí tính toán. TOPIQ hỗ trợ cả Full-Reference lẫn No-Reference IQA [5, Secs. I–III].

**Nói dễ hiểu:** Một bức ảnh có nền hơi nhiễu nhưng khuôn mặt/người/vật thể chính rõ nét có thể vẫn được đánh giá cao. Một mô hình chỉ cộng trung bình nhiễu trên toàn ảnh dễ hiểu sai việc con người ưu tiên nội dung chính.

## 3.3. Những kết quả liên quan đề tài (từ paper TOPIQ)

| Benchmark / thiết lập paper | SRCC ↑ | PLCC ↑ | Chú thích |
|---|---:|---:|---|
| TOPIQ (ResNet50) trên SPAQ — within-dataset | **0,921** | **0,924** | [5, Table VII] |
| TOPIQ (ResNet50) trên KonIQ-10k — official split | **0,928** | **0,941** | [5, Table VIII] |
| TOPIQ trained on KonIQ → tested on SPAQ | **0,8758** | **0,8791** | Cross-dataset; [5, Table VI] |

**Rất quan trọng:** Ba dòng là **ba thiết lập khác nhau**. TOPIQ trained-on-KonIQ tested-on-SPAQ mới là benchmark hợp lý hơn cho một **reference zero-shot/cross-dataset trên SPAQ**. Checkpoint `topiq_nr-spaq` đã học SPAQ **không** được dùng làm “zero-shot test” trên SPAQ. Kết quả TOPIQ within-dataset 0,921/0,924 cũng không phải câu trả lời liệu EXIF có ích khi kiểm soát backbone.

## 3.4. TOPIQ được sử dụng như thế nào trong đề tài?

- Là **literature Q1** củng cố tính cập nhật của nghiên cứu NR-IQA.
- Là **image-only pretrained reference** để chứng minh độ khó của bài toán và đặt các kết quả reproduction vào bối cảnh mô hình IQA hiện đại.
- **Không** trộn TOPIQ vào E0/E1/E2 như thể mọi mô hình cùng architecture/training data; bảng kết quả phải ghi rõ nguồn pretraining/checkpoint.
- Nếu tài nguyên cho phép có thể thêm một exploratory experiment với image embedding từ TOPIQ, nhưng đó là **bước tùy chọn** và phải kiểm soát contamination, không phải điều kiện hoàn thành nghiên cứu chính.

---

# 4. SOURCE CODE, CHECKPOINT, DATASET VÀ GIỚI HẠN TÁI LẬP

| Nguồn | Link | Có gì trong nguồn / có thể dùng làm gì | Những gì CHƯA được xác minh |
|---|---|---|---|
| **Official SPAQ repo** | https://github.com/h4nwei/SPAQ | `BL_demo.py`, `MT-E_demo.py`, `MT-A_demo.py`, `MT-S_demo.py`, `Prepare_exif.py`, `Prepare_image.py`; link tới checkpoint của từng model và bộ dữ liệu [3] | Training loop đầy đủ không nằm trong danh sách root; checkpoint/data Drive có thể yêu cầu quyền; chưa chạy |
| **SPAQ original data portal** | Từ README [3], mục Database → Baidu/Google Drive | Ảnh, MOS, attributes, scene, EXIF | Google Drive folder URL trả 404 trong trình duyệt kiểm tra; cần thử tải trực tiếp/khảo sát quyền và provenance |
| **SPAQ alternate archive** | https://huggingface.co/datasets/q-future/q-align-datasets/tree/main | Có listing `spaq.tar` đã được báo cáo ở iteration trước; có thể là nguồn thực dụng cho Kaggle | Chưa tải và xác minh ảnh/độ phân giải/nhãn tương đương bản gốc; license nội dung |
| **TOPIQ official implementation** | https://github.com/chaofengc/IQA-PyTorch | `pyiqa` inference, train/eval framework, model `topiq_nr`, `topiq_nr-flive`, `topiq_nr-spaq` [6] | Chưa chạy trên GPU Kaggle, chưa kiểm tra trọng số và tốc độ |
| **TOPIQ model card** | https://iqa-pytorch.readthedocs.io/en/latest/ModelCard.html | Xác nhận `topiq_nr` mặc định KonIQ và có bản SPAQ-specific [6] | Cần log model/version/revision và exact training provenance khi chạy |
| **Third-party SPAQ trainer** | https://github.com/lllllllllllll-llll/SPAQ | Có `BL_Trainer.py`, `train_test.py` (bên thứ ba) | Không mặc định chính xác/trung thực với paper; phải audit mã trước khi dùng |
| **KonIQ-10k external dataset** | https://github.com/subpic/koniq | 10.073 ảnh tự nhiên, nhãn chất lượng, official split | Khác SPAQ: không merge theo image_id; model TOPIQ đã train KonIQ không thể xem KonIQ là external unseen test |
| **PrISM-IQA (optional)** | https://github.com/Multimedia-Analytics-Laboratory/PrISM-IQA | Related work về chẩn đoán lỗi ISP, có training code và label schema | **Chưa xác minh file CSV nhãn tái gán cho SPAQ được tải công khai**; không làm dependency chính |

**Tính hiện đại của source:** Official SPAQ README công bố môi trường demo cũ **Python 3.5, PyTorch 1.1.1, torchvision 0.3.0**. Đây là thông tin tương thích cũ, không phải cấu hình khuyến nghị Kaggle 2026. Việc port lên PyTorch 2.x cần test trước khi tuyên bố reproduction [3].

**Vì sao dùng code của tác giả vẫn cần kiểm thử?** Có repository không có nghĩa đầy đủ training scripts, có checkpoint không có nghĩa checkpoint tải được, và có link dataset không chứng minh license hoặc data provenance phù hợp.

---

# 5. Ý TƯỞNG NGHIÊN CỨU CỦA NHÓM — KHÁC PAPER GỐC Ở ĐÂU?

## 5.1. Proposed title & phát biểu bài toán

**When Does EXIF Metadata Help Smartphone Image Quality Assessment? A Controlled Study of Fusion, Missingness, and Capture-Condition Shift**

Cho một ảnh smartphone $x$, vector metadata EXIF $m$ và MOS $y$, học một bộ dự đoán $\hat y=f(x,m)$ có khả năng dùng $m$ khi hữu ích, nhưng không suy giảm nghiêm trọng khi $m$ thiếu hoặc khi điều kiện chụp của test khác train. Đo **giá trị tăng thêm của EXIF** trong một experimental protocol được kiểm soát, thay vì chỉ tăng capacity/stacking features để cải thiện score.

## 5.2. Sự khác nhau giữa SPAQ gốc và đề tài mới

| Nội dung | SPAQ paper (2020) | Đề tài nhóm **dự kiến** |
|---|---|---|
| Đối tượng | Ảnh smartphone + MOS | Cùng bài toán, cùng nhãn |
| Image-only | BL / ResNet50 | Rebuild BL hoặc resource-adapted baseline |
| EXIF | MT-E, **scalar additive offset** | So additive với **gated feature fusion** |
| Missing EXIF | Mô tả đường fallback | **Ablation có kiểm soát** theo 0/25/50/100% missing |
| Metadata availability | Chưa stress-test có hệ thống trong Table 4 | Missingness indicators + metadata dropout |
| Capture-condition shift | Phân tích ngữ cảnh/EXIF | Subgroup holdout theo ISO/exposure nếu đủ mẫu; protocol định trước |
| Modern benchmark | DB-CNN, BRISQUE, v.v. | TOPIQ image-only Q1, bảng kết quả riêng |
| Đóng góp mới | Dataset + original multitask methods | **Controlled evidence về WHEN / HOW ROBUST EXIF IS** |

## 5.3. Research Questions (RQ)

**RQ1 — Incremental information:** Khi encoder ảnh, data split, training budget giống nhau, **EXIF có cải thiện SRCC của MOS prediction** so với image-only không? Những trường nào đóng góp thực sự?

**RQ2 — Mechanism:** Nếu dùng cùng RGB + EXIF, **feature-wise gated fusion** có tạo thêm cải thiện so với additive fusion đơn giản không? Hay chỉ tăng tham số mà không có bằng chứng tốt hơn?

**RQ3 — Robustness:** Khi EXIF thiếu một phần/toàn bộ hoặc dữ liệu nằm ở các nhóm chụp khác phân phối huấn luyện, lợi ích EXIF có còn không? **Metadata dropout** có giúp duy trì hiệu quả hơn không?

## 5.4. Hypotheses

- **H0₁:** $\Delta\mathrm{SRCC}(\mathrm{EXIF\ additive},\mathrm{image\ only})=0$; **H1₁:** khác 0.
- **H0₂:** $\Delta\mathrm{SRCC}(\mathrm{gated},\mathrm{additive})=0$; **H1₂:** khác 0.
- **H0₃:** Metadata dropout không cải thiện robustness/missingness performance so với cùng kiến trúc không dropout; **H1₃:** có khác biệt.

Không giả định mọi H1 phải cho kết quả tăng; H0 không bị bác bỏ cũng là kết quả quan trọng. Đối với so sánh nhỏ, cần báo khoảng tin cậy, không chỉ dựa một seed.

## 5.5. Mô hình khái niệm và các cấu hình đối chứng

**Visual encoder dùng chung:** $h=F_\theta(x)$, khởi tạo ResNet-50 ImageNet. Ở thí nghiệm chính, cùng một nguồn pretraining và chính sách freeze/fine-tune. Nếu frozen backbone là cần thiết để tiết kiệm tài nguyên, phải gọi **resource-adapted replication**.

- **E0 — Image-only:** $\hat y=g(h)$. Không dùng EXIF.
- **E1 — Additive EXIF:** $\hat y=g(h)+b(m)$. Cơ chế cùng loại với SPAQ MT-E.
- **E2 — Gated EXIF:** học $\gamma(m),\beta(m)$, điều biến $h'=[1+\tanh\gamma(m)]\odot h+\beta(m)$, sau đó $\hat y=g(h')$. Định nghĩa cụ thể là **đề xuất mới của nhóm**, không phải công thức từ SPAQ/TOPIQ.
- **E3 — Additive + metadata dropout:** giữ E1, chỉ đổi training-time missingness augmentation.
- **E4 — Gated + metadata dropout:** giữ E2, chỉ đổi training-time missingness augmentation.
- **E5 (không bắt buộc) — human attributes auxiliary:** dùng 5 attributes của SPAQ làm supervision lúc train để xem multitask có bổ sung cho các RQ chính không. Không tự động đưa ground-truth attributes vào test input.

**Lưu ý công bằng:** E1/E2 có parameter-count khác nhau; cần ghi nhận, hoặc thêm **parameter-matched additive MLP** để loại bớt nghi vấn tăng kết quả chỉ do nhiều tham số.

## 5.6. Dữ liệu và phương thức bổ sung

**Dữ liệu mới thực sự trong core proposal:** không cần ghép hai bộ ảnh khác nhau. Tài sản chính là **EXIF đi cùng đúng ảnh trong SPAQ**. Nếu dùng PrISM issue labels, phải có file nhãn công khai và kiểm tra provenance riêng. KonIQ-10k là **external benchmark**, không phải bảng để `JOIN` theo ảnh với SPAQ.

**Join key:** tên ảnh hoặc ID ảnh được kiểm tra thực tế qua các archive, bảng MOS, EXIF; chỉ chuẩn hóa sau khi audit. Không giả định mọi bản tải đều giữ nguyên tên/đuôi ảnh. Cần xuất tỷ lệ `join_coverage`, duplicate keys, missingness của từng trường. Nếu thiếu dữ liệu EXIF trầm trọng, giảm scope về các trường đáng tin cậy và ghi rõ coverage, **không điền giả dữ liệu**.

---

# 6. THIẾT KẾ THỰC NGHIỆM VÀ PHÂN TÍCH KẾT QUẢ

## 6.1. Giao thức được đề xuất (khác giao thức paper gốc)

- **Primary study:** 70% train / 10% validation / 20% test trên SPAQ, một số random seeds nếu ngân sách cho phép. Đây là **protocol nhóm đề xuất** để có validation riêng; **không phải** 80/20 của SPAQ 2020.
- **Holdout:** một lần cố định và giữ nguyên image IDs cho mọi E0–E4; tránh tuning trên test.
- **Data leakage:** kiểm tra exact/near-duplicate; nếu nhận diện được nhóm ảnh cùng cảnh hoặc cùng camera, xem xét group-aware split; nếu không có `device_id`, không được tuyên bố unseen-device test.
- **Metadata missingness stress test:** 0%, 25%, 50%, 100% các trường hoặc các record thiếu theo một cách sinh mask định trước. Phân biệt *missing entire EXIF record* với *missing individual fields*. Tỷ lệ và random seed chỉ được sinh từ luật kiểm tra, không dựa trên MOS test.
- **Capture-condition shift:** phân tích nhóm high ISO, long exposure, night scene; nếu tạo holdout theo ngưỡng ISO phải khóa ngưỡng dựa trên **train distribution**, bảo đảm test size đủ để metric ổn định.
- **Scale/encoding:** khai báo ISO/exposure log-transform, encoding dạng categorical theo protocol gốc hoặc ablation tương ứng; numeric scaler/imputation fit trên train only.

## 6.2. Metrics

| Metric | Chức năng | Vai trò |
|---|---|---|
| **SRCC — Spearman Rank Correlation** | Đánh giá thứ tự dự đoán so với MOS | **Primary** |
| **PLCC — Pearson Linear Correlation** | Đánh giá mức liên hệ tuyến tính giữa score/MOS | Secondary |
| **MAE trên thang MOS** | Sai số tuyệt đối, có đơn vị 0–100 | Secondary; calibration phải học từ train/val |
| **ΔSRCC** | Chênh lệch cùng test IDs giữa hai cấu hình | Effect size |
| **Missingness degradation / AUC** | Phần hiệu năng giữ lại khi tỷ lệ EXIF thiếu tăng | Reliability |
| **Error by ISO/exposure/scene** | Phân tích failure modes và confounders | Interpretation |

**Thống kê:** paired bootstrap trên cùng image IDs cho $\Delta\mathrm{SRCC}$, confidence intervals (ví dụ 95%); báo thêm variance giữa seeds. Không áp dụng paired t-test trực tiếp lên từng ảnh với “SRCC từng ảnh” vì SRCC chỉ tính trên **một tập mẫu**. Tăng chỉnh multiple-comparison nếu test nhiều hypothesis.

## 6.3. Figures đề xuất cho báo cáo cuối

1. **SPAQ data overview**: phân phối MOS, ISO, exposure, tỷ lệ EXIF missing.
2. **Original vs reproduction metrics**: bảng/kết hợp error bars, ghi rõ paper's reported metrics và our measured metrics.
3. **Ablation chart**: SRCC/PLCC E0–E4 trên cùng test.
4. **Missingness curve**: trục X = tỷ lệ EXIF thiếu; trục Y = SRCC/MAE.
5. **Group analysis**: low/high ISO, short/long exposure, night/non-night; báo size nhóm.
6. **Qualitative cases**: ảnh đại diện và các model predictions, EXIF, MOS, absolute error, lý do phân tích.

---

# 7. KHẢ THI, RỦI RO VÀ RANH GIỚI NGHIÊN CỨU

## 7.1. Ngân sách thực nghiệm

Mục tiêu là **hoàn thành một phiên chạy trong tối đa 8,5 giờ** trên Kaggle khi GPU miễn phí có sẵn. Việc lựa chọn T4/P100, GPU quota và thời gian phiên chịu thay đổi từ phía nền tảng; **không** được xem ngân sách này là tài nguyên luôn được bảo đảm. Tài liệu trong iteration trước chỉ là **allocation ceiling**, chưa có training throughput thật.

**Kế hoạch đủ linh hoạt:** chạy dataset audit → inference vài trăm ảnh → benchmark preprocess/training steps → dự báo thời gian cho E0–E4 → lựa chọn freeze toàn bộ/partially tune dựa trên đo đạc. Reproduction gần paper gốc được ưu tiên nếu budget đủ; nếu không phải công khai adaptation.

## 7.2. Những rủi ro lớn

| Rủi ro | Mức ảnh hưởng | Phương án kiểm soát |
|---|---|---|
| Không tải được archive hoặc bảng EXIF | **Critical** | Kiểm tra official/HF sources, integrity, key joins trước khi cam kết |
| EXIF thiếu hoặc suy ra từ công thức | Cao | Audit từng cột, thêm provenance/missingness flags |
| Cùng cảnh/near-duplicate qua train/test | Cao | Exact hash + perceptual hash và group-aware holdout khi có ID |
| Model TOPIQ đã train trên SPAQ | Cao | Chỉ dùng model có nguồn pretrained độc lập để benchmark SPAQ; tránh `topiq_nr-spaq` trong zero-shot claim |
| Gated fusion chỉ tốt hơn do nhiều tham số | Cao | Parameter-matched control; same backbone/steps/split |
| EXIF shortcut theo scene/device | Cao | Subgroup analysis, missingness stress, holdout điều kiện chụp; không kết luận causal |
| GPU/runtime quá ngân sách | Cao | Bắt buộc smoke benchmark; frozen encoder / giảm epochs / chia phase có lưu artifacts |
| Nhãn PrISM không public | Trung bình | Loại khỏi core experiments; chỉ là related work |
| License dữ liệu không rõ | Cao | Kiểm tra license từ tổ chức phát hành; không tự ý rehost public |

## 7.3. Minimum viable research (MVR)

Dù không chạy được mô hình đầy đủ hoặc không đủ thời gian cho nhiều seeds, phiên bản tối thiểu vẫn có giá trị là:

**SPAQ + EXIF validated → E0 image-only → E1 additive EXIF → E2 gated EXIF → missingness stress test trên fixed test set → SRCC/PLCC/MAE + paired ΔSRCC bootstrap + failure analysis.**

Nếu không kiểm chứng được EXIF có join key và độ phủ hợp lý, dự án **không còn đủ cơ sở để giữ tên đề tài hiện tại**; phải điều chỉnh research question trước khi training, thay vì chế tạo EXIF.

---

# 8. KẾT LUẬN VÀ ĐỀ XUẤT ĐỂ THẢO LUẬN VỚI GIẢNG VIÊN

**Đề tài được đề xuất:** *When Does EXIF Metadata Help Smartphone Image Quality Assessment? A Controlled Study of Fusion, Missingness, and Capture-Condition Shift*.

**Tại sao đáng làm?** SPAQ đã chứng minh EXIF cải thiện MOS prediction trong một giao thức nhất định. Điều cần làm tiếp là hiểu **hiệu quả đó có đáng tin cậy hay không**, bao gồm sự lựa chọn fusion và khả năng chịu thiếu metadata. Đây là bài toán thực tế, có dữ liệu và baseline công bố, có khả năng thực hiện thí nghiệm kiểm soát; rủi ro khoa học chủ yếu là shortcut, contamination và khan hiếm metadata có thể dùng.

**Chúng ta học từ paper nào?**

- **SPAQ (Fang et al., CVPR 2020):** cách định nghĩa bài toán, dataset, MOS, EXIF, mô hình BL/MT-E, giao thức training và kết quả gốc.
- **TOPIQ (Chen et al., IEEE TIP 2024, Q1):** tiến bộ NR-IQA dựa trên semantic-distortion reasoning, làm Q1 academic anchor và benchmark hiện đại.
- **Related work khác:** KonIQ-10k cho generalization; PrISM-IQA cho hướng ISP issue diagnosis tương lai nhưng **không phải dữ liệu bắt buộc**.

**Phần nhóm dự định làm là gì?** Tái lập baseline SPAQ phù hợp tài nguyên, nghiên cứu fusion EXIF additive so với gated, thêm metadata-dropout training, đánh giá resilience và capture-condition shift, phân tích cả trường hợp không cải thiện.

**Giới hạn cam kết:** Trong phiên nghiên cứu tài liệu này **chưa có measured results, chưa có tập dữ liệu đã merge, chưa có verified Kaggle runtime**. Bất kỳ đề xuất “đã reproduce thành công” nào chỉ được đưa vào báo cáo sau khi có code, artifact và log thực thi.

**Ba câu hỏi nên hỏi giảng viên:** (i) Môn học chấp nhận **paper Q1 làm anchor** và reproduce **SPAQ CVPR 2020** hay bắt buộc reproduce chính xác phương pháp trong **TOPIQ Q1**? (ii) Ưu tiên giá trị khoa học ở **metadata robustness** hay ở model novelty? (iii) Yêu cầu số lượng seeds/nhóm kiểm thử và thời gian thực nghiệm tối thiểu là gì?

---

# REFERENCES & VERIFIED RESOURCE LINKS

**[1]** Fang, Y., Zhu, H., Zeng, Y., Ma, K., & Wang, Z. (2020). **Perceptual Quality Assessment of Smartphone Photography**. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 3677–3686. DOI: https://doi.org/10.1109/CVPR42600.2020.00373. Official PDF: https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf.

**[2]** IEEE Xplore publication record for Fang et al. (2020). https://ieeexplore.ieee.org/document/9156490.

**[3]** Fang et al. official **SPAQ repository**: https://github.com/h4nwei/SPAQ — demos/checkpoints, EXIF preprocessing, dataset access links; training completeness not established.

**[4]** Chen, C., Mo, J., Hou, J., Wu, H., Liao, L., Sun, W., Yan, Q., & Lin, W. (2024). **TOPIQ: A Top-Down Approach From Semantics to Distortions for Image Quality Assessment**. *IEEE Transactions on Image Processing*, **33**, 2404–2418. DOI: https://doi.org/10.1109/TIP.2024.3378466. Bibliographic record: https://pubmed.ncbi.nlm.nih.gov/38517711/.

**[5]** Accessible TOPIQ arXiv article with Tables V–VIII and methods: https://arxiv.org/html/2308.03060v1. Đây là preprint của **công trình đã được xuất bản chính thức ở IEEE TIP 2024**, không phải một bài khác.

**[6]** Official **IQA-PyTorch / TOPIQ implementation**: https://github.com/chaofengc/IQA-PyTorch; model cards: https://iqa-pytorch.readthedocs.io/en/latest/ModelCard.html.

**[7]** Quartile historical summary for *IEEE Transactions on Image Processing*: https://www.akaturk.com/journals/25534?lang=en (SJR/JCR Q1 2024–2025); independent index: https://researchcloud.net/journal/1941-0042. **Secondary sources** for quartile, not IEEE/Clarivate primary certificate.

**[8]** Hosu, V., Lin, H., Sziranyi, T., & Saupe, D. (2020). **KonIQ-10k: An Ecologically Valid Database for Deep Learning of Blind Image Quality Assessment**. *IEEE Transactions on Image Processing*, 29, 4041–4056. DOI: https://doi.org/10.1109/TIP.2020.2967829. Repository: https://github.com/subpic/koniq.

**[9]** Hu et al. (2023). **Toward a No-Reference Quality Metric for Camera-Captured Images**. *IEEE Transactions on Cybernetics*. DOI: https://doi.org/10.1109/TCYB.2021.3128023. Related Q1 on camera-captured quality assessment.

**[10]** Zhai et al. (2026). **PrISM-IQA: Image Quality Assessment Made Practical for Smartphone Photography**. Preprint: https://arxiv.org/html/2606.31626. Repository: https://github.com/Multimedia-Analytics-Laboratory/PrISM-IQA. Không dùng label table làm dependency khi chưa có verified public download.

**[11]** Kaggle Notebooks documentation: https://www.kaggle.com/docs/notebooks ; https://www.kaggle.com/docs/efficient-gpu-usage . GPU và quota thay đổi; không dùng như measured benchmark.

---

## GHI CHÚ VỀ MỨC ĐỘ XÁC MINH

- **VERIFIED FROM PAPER/README:** thông tin thư mục, phương pháp, số lượng ảnh, các chỉ số trên Tables 2/4/6 của SPAQ và Tables VI/VII/VIII của TOPIQ; danh sách file trong GitHub repos.
- **DESIGN PROPOSAL / INFERENCE:** research questions về when EXIF helps, gated fusion, missingness/capture-condition holdout, 70/10/20 validation split, proposed hypothesis/figures.
- **NOT YET EXECUTED:** dataset download/merge, license confirmation for every archive, Kaggle benchmark, model training, checkpoint test, empirical scores of our own work.

**Lần cập nhật:** 09/11/2026. Báo cáo tổng hợp từ các nguồn được dẫn, có bổ sung đánh giá và thiết kế nghiên cứu; không thay thế paper gốc.
