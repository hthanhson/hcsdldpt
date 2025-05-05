import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path):
    """
    Đọc ảnh từ đường dẫn và chuyển đổi sang định dạng RGB
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc file ảnh: {image_path}")
    
    # Chuyển đổi từ BGR sang RGB (OpenCV đọc ảnh theo BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def resize_image(image, target_size=(224, 224)):
    """
    Thay đổi kích thước ảnh
    """
    return cv2.resize(image, target_size)

def normalize_image(image):
    """
    Chuẩn hóa ảnh cho mô hình deep learning
    """
    # Chuẩn hóa về khoảng [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Tiền xử lý ảnh: đọc, thay đổi kích thước và chuẩn hóa
    """
    image = load_image(image_path)
    image = resize_image(image, target_size)
    image = normalize_image(image)
    
    return image

def extract_person_id(filename):
    """
    Trích xuất ID người từ tên file ảnh
    Ví dụ: 16_1_0_20170109214419099.jpg -> 16_1_0
    """
    parts = filename.split('_')
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    return None

def display_images(images, titles=None, figsize=(15, 10)):
    """
    Hiển thị nhiều ảnh trong một figure
    """
    n = len(images)
    if titles is None:
        titles = [f"Image {i}" for i in range(n)]
    
    plt.figure(figsize=figsize)
    
    for i in range(n):
        plt.subplot(1, n, i+1)
        if len(images[i].shape) == 2:  # grayscale
            plt.imshow(images[i], cmap='gray')
        else:  # RGB
            plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_results(image_path, query_face, similar_faces, similar_filenames, 
                similarity_scores, query_attributes, similar_attributes, 
                similarity_reasons, output_dir='results', required_attrs=None):
    """
    Lưu kết quả tìm kiếm vào thư mục
    
    Parameters:
    - image_path: Đường dẫn đến ảnh đầu vào
    - query_face: Ảnh khuôn mặt đầu vào
    - similar_faces: Danh sách các ảnh khuôn mặt tương tự
    - similar_filenames: Danh sách các đường dẫn file ảnh tương tự
    - similarity_scores: Điểm tương đồng tương ứng
    - query_attributes: Các thuộc tính của ảnh đầu vào
    - similar_attributes: Danh sách các thuộc tính của ảnh tương tự
    - similarity_reasons: Lý do tương đồng cho mỗi kết quả
    - output_dir: Thư mục đầu ra
    - required_attrs: Các thuộc tính yêu cầu khi tìm kiếm theo thuộc tính
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Tạo thư mục kết quả cụ thể cho ảnh này
    timestamp = os.path.basename(image_path).split('.')[0]
    result_dir = os.path.join(output_dir, f'result_{timestamp}')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # Lưu ảnh đầu vào
    input_rgb = cv2.cvtColor(query_face, cv2.COLOR_BGR2RGB)
    input_pil = Image.fromarray(input_rgb)
    input_pil.save(os.path.join(result_dir, 'input.jpg'))
    
    # Lưu các ảnh tương tự
    for i, (img, score) in enumerate(zip(similar_faces, similarity_scores)):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.save(os.path.join(result_dir, f'similar_{i+1}_{score:.4f}.jpg'))
    
    # Lưu thông tin vào file text
    with open(os.path.join(result_dir, 'results_info.txt'), 'w', encoding='utf-8') as f:
        f.write("=== KẾT QUẢ TÌM KIẾM ===\n\n")
        
        # Thông tin ảnh đầu vào
        f.write("THÔNG TIN ẢNH ĐẦU VÀO:\n")
        f.write(f"- Tệp ảnh: {os.path.basename(image_path)}\n")
        if query_attributes:
            f.write(f"- Màu da: {query_attributes['skin_tone']}\n")
            f.write(f"- Nhóm tuổi: {query_attributes['age_group']}\n")
            f.write(f"- Cảm xúc: {query_attributes['emotion']}\n\n")
            
        # Thông tin tìm kiếm
        if required_attrs:
            f.write("THÔNG TIN TÌM KIẾM:\n")
            f.write("- Tìm kiếm theo thuộc tính:\n")
            for attr, value in required_attrs.items():
                f.write(f"  + {attr}: {value}\n")
            f.write("\n")
        
        # Thông tin các ảnh tương tự
        f.write("THÔNG TIN CÁC ẢNH TƯƠNG TỰ:\n")
        for i, (filename, score, attrs, reason) in enumerate(zip(similar_filenames, similarity_scores, similar_attributes, similarity_reasons)):
            f.write(f"\n{i+1}. {os.path.basename(filename)} - Điểm tương đồng: {score:.4f}\n")
            
            if attrs:
                f.write(f"   - Màu da: {attrs['skin_tone']}\n")
                f.write(f"   - Nhóm tuổi: {attrs['age_group']}\n")
                f.write(f"   - Cảm xúc: {attrs['emotion']}\n")
            
            # Phân tích thuộc tính giống nhau
            shared_attributes = []
            if query_attributes and attrs:
                if query_attributes['skin_tone'] == attrs['skin_tone']:
                    shared_attributes.append(f"Màu da ({attrs['skin_tone']})")
                if query_attributes['age_group'] == attrs['age_group']:
                    shared_attributes.append(f"Nhóm tuổi ({attrs['age_group']})")
                if query_attributes['emotion'] == attrs['emotion']:
                    shared_attributes.append(f"Cảm xúc ({attrs['emotion']})")
            
            if shared_attributes:
                f.write(f"   - Thuộc tính giống nhau: {', '.join(shared_attributes)}\n")
            
            if reason:
                f.write(f"   - Lý do tương đồng: {reason}\n")
    
    # Tạo ảnh kết quả tổng hợp
    plt.figure(figsize=(15, 8))
    
    # Hiển thị ảnh đầu vào
    plt.subplot(1, len(similar_faces) + 1, 1)
    plt.imshow(input_rgb)
    title = 'Ảnh đầu vào'
    if query_attributes:
        title += f"\n{query_attributes['skin_tone']}, {query_attributes['age_group']}, {query_attributes['emotion']}"
    plt.title(title, fontsize=9)
    plt.axis('off')
    
    # Hiển thị các ảnh tương tự
    for i, (img, filename, score, attrs, reason) in enumerate(zip(similar_faces, similar_filenames, similarity_scores, similar_attributes, similarity_reasons)):
        plt.subplot(1, len(similar_faces) + 1, i+2)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        
        # Tạo tiêu đề với thông tin thuộc tính
        title = f"#{i+1}: {score:.4f}"
        if attrs:
            title += f"\n{attrs['skin_tone']}, {attrs['age_group']}, {attrs['emotion']}"
        title += f"\n{reason}"
        
        plt.title(title, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'summary.jpg'))
    plt.close()
    
    print(f"\nĐã lưu kết quả tìm kiếm vào thư mục: {result_dir}") 