# Chạy CM4IR trên bộ LSUN Bedroom đầy đủ

Runner dùng thuật toán và checkpoint Bedroom đã kiểm tra ở [báo cáo tái lập](CM4IR_REPRODUCTION_REPORT.md). Mặc định dùng Appendix B của paper, nhiễu quan sát 0,025 và 4 lượt gọi UNet. Đổi `--preset repo` để dùng tham số script tác giả, hoặc `--noise 0.05` cho mức nhiễu còn lại.

## Đầu vào

- Thư mục riêng chứa 300 ảnh gốc LSUN Bedroom validation 256×256 để chương trình tạo ảnh mờ Gaussian trước khi phục hồi. Runner chấp nhận thư mục phẳng hoặc có thư mục con, sắp xếp theo tên và kiểm tra số lượng, kích thước, khả năng giải mã và trùng byte trước khi chạy.
- Dùng đúng split của tác giả khi đối chiếu benchmark. Số lượng 300 chỉ kiểm tra tính đầy đủ về số ảnh. `manifest.json` ghi tên và SHA-256 để đối chiếu danh sách khi data được bàn giao. [Dữ liệu được repo tác giả dẫn tới](https://github.com/tirer-lab/CM4IR#full-datasets)
- Checkpoint `cd_bedroom256_lpips.pt` và môi trường Python 3.11+ với PyTorch/torchvision chạy CUDA. Xem [setup](DEBLUR.md), sau đó cài `python -m pip install -r requirements-deblur.txt`. Nếu dùng `--lpips` trên máy offline, chuẩn bị cache VGG16 trước.

## Lệnh chạy

```sh
python scripts/deblur_full.py --dataset /data/lsun-bedroom-val --checkpoint /models/cd_bedroom256_lpips.pt --output exp/repro_runs/bedroom-paper-025 --workers 4 --lpips
```

Lệnh dùng một GPU được cấp, nạp checkpoint một lần và ghi PNG phục hồi, PSNR trước/sau, VGG-LPIPS và hash theo từng ảnh. `summary.json` được tạo khi toàn bộ ảnh hoàn tất. PSNR của runner full lấy từ tensor đã chặn về [0,1] trước lượng tử hóa PNG. PSNR hoàn hảo được biểu diễn bằng `null`. Kết quả LPIPS dùng VGG như repo tác giả.

Chạy lại đúng lệnh và thư mục `--output` để tiếp tục. Ảnh đã hoàn thành được xác nhận bằng hash PNG. File mất hoặc bị thay đổi sẽ được tạo lại. Seed mỗi ảnh là seed gốc cộng chỉ số ảnh trong manifest, để chạy lại một phần dùng đúng chuỗi nhiễu của ảnh đó. Seed theo ảnh khác cách tiêu thụ RNG liên tục của runner tái lập. Thay đổi split, checkpoint, phiên bản code, môi trường hoặc cấu hình thì dùng thư mục output mới. Khi chạy hai preset đối chiếu, dùng cùng split, seed và mức nhiễu với hai output riêng.

Trước khi có data full, dùng `--dataset exp/datasets/lsun_bedroom --expected-count 8` để chạy bộ mẫu có sẵn. Lệnh full dành cho ảnh gốc sạch. Nếu data bàn giao là ảnh đã mờ, dùng đường chạy `--observed` trong [DEBLUR.md](DEBLUR.md) cho từng ảnh.

## Kiểm tra trên bộ mẫu

Đã chạy 8/8 ảnh với checkpoint chính thức, RTX 4050, 2 workers và VGG-LPIPS. Với preset paper và nhiễu 0,025, PSNR trung bình từ 23,24 dB lên 29,47 dB, VGG-LPIPS 0,2105. Khi bỏ một PNG ra khỏi thư mục đầu ra, lần chạy tiếp chỉ xử lý lại ảnh đó và tạo lại cùng hash PNG và nội dung tổng hợp. Lần chạy 300 ảnh chờ split đầy đủ.
