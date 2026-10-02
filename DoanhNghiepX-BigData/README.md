# Ứng dụng dữ liệu lớn cho doanh nghiệp nhỏ: dự báo nhu cầu, thiết kế combo và tối ưu giá trị đơn hàng

Mã nguồn kèm tiểu luận cuối kỳ môn **Nghiên cứu Dữ liệu lớn và Ứng dụng trong Kinh doanh** (TS. Nguyễn Thôn Dã),
Chương trình Thạc sĩ Thương mại điện tử, Trường Đại học Kinh tế – Luật, ĐHQG TP.HCM.

Học viên: Nguyễn Thu Thảo – MSSV C25611324

- Ứng dụng: (https://doanhnghiepx-monbigdata-gz8xhjnf3rkaxmtcwjkrue.streamlit.app/)
- Trường hợp nghiên cứu: Doanh nghiệp X (đã ẩn danh), cơ sở bán đồ thể thao tự sản xuất theo đơn.
- Dữ liệu: 17.642 dòng bán, 8.460 đơn hàng, 06/2025 – 07/2026 (không công khai trong kho này).

## Ba phần phân tích

| Phần | Phương pháp | Kết quả chính |
|---|---|---|
| Dự báo nhu cầu | LightGBM học chung (kiểu M5) so với 7 phương pháp thống kê; kiểm định lùi 12 tuần; kiểm định Diebold–Mariano | LightGBM có RMSE thấp nhất (10,64 so với 11,47 của naive), khác biệt chưa có ý nghĩa thống kê |
| Kế hoạch cắt | Dự báo cấp dòng sản phẩm, phân bổ xuống màu/cỡ theo tỷ trọng 8 tuần và phương pháp phần dư lớn nhất | 84 tổ hợp màu–cỡ, 160 sản phẩm cho tuần 03/08/2026 |
| Combo | FP-Growth (Spark MLlib trong notebook; mlxtend trong app) | 6 cặp sản phẩm có độ nâng từ 2,1 đến 4,5 |
| Giá trị đơn | Thống kê mô tả, mô phỏng combo và ngưỡng miễn phí vận chuyển, phân tích độ nhạy | Đơn nhiều món gấp 2,3 lần đơn một món; tiềm năng tăng khoảng 4% doanh thu |
| Khả năng mở rộng | Nhân dữ liệu 1–50 lần trên PySpark | 882.100 dòng: FP-Growth 9,8 giây, tập luật không đổi |

## Cấu trúc kho mã

```
├── app.py                 # Giao diện Streamlit (5 thẻ)
├── core.py                # Toàn bộ phần tính toán, không phụ thuộc giao diện
├── requirements.txt       # Thư viện Python
├── notebooks/
│   ├── DoanhNghiepX_DayDu_Demo.ipynb   # Notebook Colab: 3 module + 4 demo, tái tạo mọi số liệu Chương 4
│   └── DuBao_SKU_PySpark_vong1.ipynb   # Vòng thử đầu tiên: GBTRegressor trên PySpark
└── ket_qua/               # Kết quả tổng hợp (không chứa thông tin khách hàng)
```

## Chạy ứng dụng trên máy

```bash
pip install -r requirements.txt
streamlit run app.py
```

Khi chưa tải tệp, ứng dụng hiển thị các kết quả đã lưu trong `ket_qua/`. Khi tải tệp đơn hàng (.xlsx có các cột
STT, Ngày tạo đơn, Giá trị đơn hàng sau giảm giá, Sản phẩm), toàn bộ quy trình được tính lại trên dữ liệu mới.

## Chạy notebook

Mở `notebooks/DoanhNghiepX_DayDu_Demo.ipynb` trên Google Colab, chạy lần lượt từng ô và tải tệp đơn hàng lên ở Ô 1.

## Bảo mật dữ liệu

Tệp đơn hàng gốc chứa thông tin khách hàng nên không được đưa lên kho mã (xem `.gitignore`).
Ứng dụng chỉ xử lý dữ liệu trong phiên làm việc và không lưu lại.
