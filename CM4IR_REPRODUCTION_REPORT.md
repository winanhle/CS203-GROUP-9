# Báo cáo xác nhận tái lập CM4IR gửi Quỳnh Anh

Ngày kiểm tra: 14/11/2026. Nhánh gốc của nhóm: `codex/anh-core` (`3d09c11`). Code đã chạy: `8599c71`. Hạng mục: Gaussian deblurring trên 8 ảnh LSUN Bedroom 256×256 có trong repo nhóm.

## Thiết lập và kết quả

Code dùng lại UNet, toán tử Gaussian SVD và bộ lấy mẫu 4 bước của tác giả. Cấu hình `repo` lấy từ lệnh công bố trong README, còn `paper` lấy từ Appendix B, Table 5. Chạy trên RTX 4050 Laptop, seed 1234, nhiễu quan sát 0,025, checkpoint chính thức `cd_bedroom256_lpips.pt`.

| Đầu ra trên cùng 8 ảnh | PSNR trung bình (dB) | VGG-LPIPS trung bình |
| --- | ---: | ---: |
| Ảnh mờ đầu vào | 23,24 | — |
| CM4IR theo script tác giả | 29,48 | 0,2200 |
| CM4IR theo tham số trong paper | 29,51 | 0,2104 |

Đã xác nhận 8/8 PNG phục hồi hợp lệ mỗi cấu hình. Đã đưa một PNG mờ đã lưu trở lại qua `--observed` và thu được PNG phục hồi 256×256. Trường hợp này dùng ảnh mờ nhân tạo theo kernel đã biết.

SHA-256 của checkpoint: `61238b8de0ba4976ec35a60fd4786a02a862df0960aa8a2554467e9486e71252`.

Ảnh kiểm tra: `bedroom_0000000.png`, `bedroom_0000001.png`, `bedroom_0000006.png`, `bedroom_0000009.png`, `bedroom_0000016.png`, `bedroom_0000263.png`, `bedroom_0000285.png`, `bedroom_0000288.png`.

## Cách kiểm tra lại

```sh
python scripts/deblur.py --preset repo --noise 0.025 --limit 8 --lpips
python scripts/deblur.py --preset paper --noise 0.025 --limit 8 --lpips
```

Mỗi lần chạy lưu ảnh trước/sau, ảnh gốc, log và `run.json` dưới `exp/image_samples/`. Xem [hướng dẫn setup](DEBLUR.md). Phạm vi xác nhận là 8 ảnh có sẵn. Benchmark Bedroom trong paper dùng 300 ảnh và cần chạy trên đúng split trước khi đối chiếu Table 2.

Nguồn: [paper CM4IR](https://arxiv.org/html/2412.20596) · [repo tác giả](https://github.com/tirer-lab/CM4IR).
