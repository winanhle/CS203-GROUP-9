# START HERE — Codex handoff for SPAQ/EXIF research

## Mục đích

Bộ hồ sơ để giao cho Codex triển khai **end-to-end** đề tài nghiên cứu:

> **When Does EXIF Metadata Help Smartphone Image Quality Assessment? A Controlled Study of Fusion, Missingness, and Capture-Condition Shift**

Codex chịu trách nhiệm khởi tạo repository, kiểm chứng nguồn, lấy dữ liệu có quyền truy cập, xây baseline, triển khai experiments, chạy Kaggle GPU nếu được xác thực, xuất artifacts, cập nhật docs và lưu trạng thái để khôi phục sau khi hết context/token.

**Đây là bộ yêu cầu và tài liệu tham khảo, không chứa code đã train, dataset hoặc kết quả nghiên cứu.** Không được nhận nhầm tài liệu nguồn thành bằng chứng đã thực thi.

## Cần gửi cho Codex

Cách dễ nhất: **giải nén toàn bộ ZIP** vào thư mục làm việc của Codex. Giữ nguyên cấu trúc. Dán nội dung `CODEX_MASTER_PROMPT.md` làm lời yêu cầu đầu tiên (hoặc bảo Codex mở và thực thi chính file đó). File prompt có thể dùng độc lập, nhưng cung cấp đủ hai nguồn nghiên cứu dưới `research_sources/` sẽ chính xác hơn.

### Các file

| File | Vai trò | Bắt buộc? |
|---|---|---|
| `CODEX_MASTER_PROMPT.md` | Mệnh lệnh triển khai end-to-end và các gate | **Có** |
| `research_sources/RESEARCH_REPORT.md` | Report ý tưởng và paper gốc đầy đủ | **Có** |
| `research_sources/ITERATION_2_REVIEW.md` | Kiểm toán claim, scope Q1, dataset, Kaggle | **Có** |
| `AGENTS.md` | Quy tắc làm việc để Codex copy vào repo mới | Nên dùng |
| `ACCEPTANCE_CRITERIA.md` | Bộ điều kiện nghiệm thu từng phase | Nên dùng |
| `KAGGLE_PLAYBOOK.md` | Kaggle CLI, auth, smoke test, recovery | Nên dùng |
| `CODEX_RESUME_PROMPT.md` | Prompt tiếp tục ở phiên khác | **Có cho resume** |
| `templates/` | Mẫu state/checkpoint/evidence | Nên dùng |

Không cần gửi `Scientific_Method_CV_Deep_Research_2026-11-08.md` về EuroSAT vì đã đổi hướng. Nếu gửi báo cáo nghiên cứu cũ, hãy coi là historical reference, **không** lấy chủ đề EuroSAT làm yêu cầu hiện tại.

## Bước chuẩn bị phía người dùng

1. Tạo/thư mục làm việc cho Codex và đưa các file vào; bật quyền truy cập workspace/các lệnh thực thi trong môi trường an toàn mà mày kiểm soát.
2. GitHub: Codex kiểm tra đăng nhập `gh auth status`. Nếu chưa đăng nhập, người dùng thực hiện OAuth/CLI auth. Mặc định tạo **private repository**; không push dataset, secrets, checkpoint có license hạn chế.
3. Kaggle: Codex kiểm tra `kaggle --version` và đăng nhập bằng `kaggle auth login` hoặc credential hợp lệ. **Không gửi API token, cookie, password vào chat/prompt, Git hay notebook**.
4. Nếu nền tảng Codex không có truy cập ngoài, không có credential hay không thể chạy GPU, Codex phải tạo notebook + CLI script chạy được, báo blocker có bằng chứng, và tiếp tục các nhiệm vụ offline phù hợp. Không tuyên bố đã train.
5. Hạn mức mục tiêu cho **một full experimental run**: **8,5 giờ**, phải đo smoke test và điều chỉnh quy mô; không tự giả định có T4 x2 hay hạn ngạch cố định.

## Một lệnh mở đầu ngắn nhất

```text
Read CODEX_MASTER_PROMPT.md, research_sources/RESEARCH_REPORT.md,
research_sources/ITERATION_2_REVIEW.md, AGENTS.md, ACCEPTANCE_CRITERIA.md
and KAGGLE_PLAYBOOK.md. Execute the full project as a staged, evidence-gated
research engineering workflow. Create the repository and progress ledger
before any major work. Do not stop at planning. Begin Phase 0 immediately.
```

## Khi hết token/context hoặc tạo session mới

Giữ nguyên repository. Dán `CODEX_RESUME_PROMPT.md`, yêu cầu Codex đọc `PROJECT_STATUS.md`, `STATE.json`, `TASKS.md`, `SESSION_HANDOFF.md`, run registry và recent git commits trước khi làm tiếp. Không yêu cầu sinh lại dự án từ đầu.

## Những gì không được kỳ vọng trước khi chạy thực tế

- Chưa có measured SRCC/PLCC/MAE hoặc pretrained model được xác minh bằng execution.
- Chưa có dataset đã tải và kiểm chứng checksum/EXIF schema.
- Việc hứa hẹn 8,5 giờ chỉ là budget, không phải measured throughput.
- TOPIQ Q1 checkpoint evaluation **không đồng nghĩa** reproduce training TOPIQ.
