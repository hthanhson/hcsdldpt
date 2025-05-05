# Hệ thống Tìm kiếm Khuôn mặt Dựa trên Thuộc tính

Dự án này triển khai một hệ thống tìm kiếm khuôn mặt dựa trên các thuộc tính khuôn mặt, tập trung vào màu da, nhóm tuổi và cảm xúc. Hệ thống không nhận diện danh tính cá nhân mà chỉ nhận diện các thuộc tính chung và hiển thị các điểm giống nhau giữa ảnh đầu vào và các ảnh được tìm thấy.

## Tổng quan

Hệ thống phân tích ảnh để xác định các thuộc tính sau:

- **Màu da**: light (sáng), medium (trung bình), dark (tối)
- **Nhóm tuổi**: child (trẻ em), young_adult (thanh niên), adult (người trưởng thành), senior (người cao tuổi)
- **Cảm xúc**: smile (cười), neutral (trung tính), serious (nghiêm túc)

Khi tìm kiếm, hệ thống sẽ phát hiện khuôn mặt trong ảnh đầu vào, phân tích các thuộc tính, tìm kiếm các ảnh tương tự trong cơ sở dữ liệu, và hiển thị các điểm giống nhau.

## Cấu trúc dự án

- `main.py`: Chương trình chính, xử lý các lệnh từ dòng lệnh
- `face_detection.py`: Module phát hiện khuôn mặt
- `feature_extraction.py`: Module trích xuất đặc trưng và phân loại thuộc tính
- `database.py`: Module quản lý cơ sở dữ liệu khuôn mặt
- `search_engine.py`: Module tìm kiếm khuôn mặt tương tự
- `utils.py`: Các hàm tiện ích
- `data_test/`: Thư mục chứa dữ liệu ảnh
- `database/`: Thư mục lưu trữ cơ sở dữ liệu
- `results/`: Thư mục lưu trữ kết quả tìm kiếm

## Cách hoạt động

1. **Phát hiện khuôn mặt**: Sử dụng mô hình MTCNN để phát hiện khuôn mặt trong ảnh.
2. **Trích xuất đặc trưng**:
   - Trích xuất vector đặc trưng 128 chiều từ mô hình deep learning (InceptionResNetV2).
   - Phân tích màu da dựa trên giá trị độ sáng trong không gian màu YCrCb.
   - Ước tính nhóm tuổi dựa trên tỷ lệ khuôn mặt.
   - Phát hiện cảm xúc dựa trên hình dạng miệng.
3. **Xây dựng cơ sở dữ liệu**: Xử lý tất cả ảnh trong thư mục dữ liệu, lưu vector đặc trưng và các thuộc tính.
4. **Tìm kiếm**:
   - Tính toán độ tương đồng cosine giữa vector đặc trưng của ảnh đầu vào và các ảnh trong cơ sở dữ liệu.
   - Lọc kết quả dựa trên các thuộc tính yêu cầu (nếu có).
   - Hiển thị các điểm giống nhau giữa ảnh đầu vào và các ảnh được tìm thấy.

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

2. Chuẩn bị thư mục dữ liệu chứa các ảnh khuôn mặt.

## Sử dụng

### Xây dựng cơ sở dữ liệu

```bash
python main.py --mode build_database --data_dir data_test
```

### Tìm kiếm

Tìm kiếm thông thường:
```bash
python main.py --mode search --image path/to/image.jpg
```

Tìm kiếm theo thuộc tính cụ thể:
```bash
python main.py --mode search --image path/to/image.jpg --skin_tone light --emotion smile
```

### Chế độ demo

```bash
python main.py --mode demo --demo_type all
```

## Kết quả

Hệ thống trả về ba ảnh khuôn mặt có các thuộc tính tương tự với ảnh đầu vào, sắp xếp theo điểm tương đồng. Mỗi kết quả hiển thị:

1. Điểm tương đồng (từ 0 đến 1)
2. Thuộc tính của ảnh tìm được (màu da, nhóm tuổi, cảm xúc)
3. Các thuộc tính giống nhau với ảnh đầu vào
4. Lý do tương đồng

Kết quả được lưu trong thư mục `results/` bao gồm:
- Ảnh đầu vào
- Các ảnh tương tự
- File thông tin chi tiết
- Ảnh tổng hợp kết quả 