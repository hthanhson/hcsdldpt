import cv2
import numpy as np
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from utils import resize_image

class FaceDetector:
    def __init__(self, min_face_size=20, scale_factor=0.709):
        """
        Khởi tạo bộ phát hiện khuôn mặt sử dụng MTCNN
        
        Parameters:
        - min_face_size: Kích thước mặt nhỏ nhất có thể phát hiện
        - scale_factor: Hệ số tỉ lệ cho việc phát hiện khuôn mặt ở nhiều tỉ lệ
        """
        self.detector = MTCNN(min_face_size=min_face_size, scale_factor=scale_factor)
    
    def detect_faces(self, image):
        """
        Phát hiện tất cả khuôn mặt trong ảnh
        
        Parameters:
        - image: Ảnh đầu vào (RGB)
        
        Returns:
        - faces: Danh sách các khuôn mặt được phát hiện
          Mỗi khuôn mặt là một từ điển có các khóa: 'box', 'confidence', 'keypoints'
        """
        # Đảm bảo ảnh là RGB
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Phát hiện khuôn mặt
        faces = self.detector.detect_faces(image)
        
        return faces
    
    def get_largest_face(self, image):
        """
        Phát hiện khuôn mặt lớn nhất trong ảnh
        
        Parameters:
        - image: Ảnh đầu vào (RGB)
        
        Returns:
        - face: Thông tin về khuôn mặt lớn nhất, hoặc None nếu không tìm thấy
        """
        faces = self.detect_faces(image)
        
        if not faces:
            return None
        
        # Tìm khuôn mặt lớn nhất dựa trên diện tích của bbox
        largest_face = max(faces, key=lambda face: face['box'][2] * face['box'][3])
        
        return largest_face
    
    def extract_face(self, image, face=None, margin=20, target_size=(224, 224)):
        """
        Trích xuất vùng khuôn mặt từ ảnh
        
        Parameters:
        - image: Ảnh đầu vào (RGB)
        - face: Thông tin về khuôn mặt (nếu đã có). Nếu None, sẽ phát hiện khuôn mặt lớn nhất
        - margin: Lề bổ sung xung quanh khuôn mặt (pixels)
        - target_size: Kích thước đầu ra của ảnh khuôn mặt
        
        Returns:
        - face_image: Ảnh khuôn mặt đã được cắt và thay đổi kích thước
          hoặc None nếu không tìm thấy khuôn mặt
        """
        if face is None:
            face = self.get_largest_face(image)
            if face is None:
                return None
        
        # Lấy tọa độ bbox
        x, y, width, height = face['box']
        
        # Thêm margin
        x_min = max(0, x - margin)
        y_min = max(0, y - margin)
        x_max = min(image.shape[1], x + width + margin)
        y_max = min(image.shape[0], y + height + margin)
        
        # Cắt ảnh khuôn mặt
        face_image = image[y_min:y_max, x_min:x_max]
        
        # Thay đổi kích thước ảnh khuôn mặt
        face_image = resize_image(face_image, target_size)
        
        return face_image
    
    def detect_and_extract_all_faces(self, image, margin=20, target_size=(224, 224)):
        """
        Phát hiện và trích xuất tất cả khuôn mặt từ ảnh
        
        Parameters:
        - image: Ảnh đầu vào (RGB)
        - margin: Lề bổ sung xung quanh khuôn mặt (pixels)
        - target_size: Kích thước đầu ra của ảnh khuôn mặt
        
        Returns:
        - face_images: Danh sách các ảnh khuôn mặt đã được cắt và thay đổi kích thước
        """
        faces = self.detect_faces(image)
        face_images = []
        
        for face in faces:
            face_image = self.extract_face(image, face, margin, target_size)
            if face_image is not None:
                face_images.append(face_image)
        
        return face_images
    
    def draw_faces(self, image, faces=None):
        """
        Vẽ các khuôn mặt được phát hiện lên ảnh
        
        Parameters:
        - image: Ảnh đầu vào (RGB)
        - faces: Danh sách các khuôn mặt. Nếu None, sẽ phát hiện khuôn mặt
        
        Returns:
        - result_image: Ảnh với các khuôn mặt được đánh dấu
        """
        if faces is None:
            faces = self.detect_faces(image)
        
        result_image = image.copy()
        
        for face in faces:
            # Vẽ bbox
            x, y, width, height = face['box']
            cv2.rectangle(result_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Vẽ các điểm mốc (landmarks)
            keypoints = face['keypoints']
            for point in keypoints.values():
                cv2.circle(result_image, point, 2, (0, 0, 255), 2)
            
            # Hiển thị độ tin cậy
            confidence = face['confidence']
            cv2.putText(result_image, f"{confidence:.2f}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_image

# Hàm demo sử dụng bộ phát hiện khuôn mặt
def demo_face_detection(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Tạo bộ phát hiện khuôn mặt
    detector = FaceDetector()
    
    # Phát hiện khuôn mặt
    faces = detector.detect_faces(image)
    print(f"Số khuôn mặt phát hiện được: {len(faces)}")
    
    # Vẽ khuôn mặt lên ảnh
    result_image = detector.draw_faces(image, faces)
    
    # Hiển thị kết quả
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Ảnh gốc")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(result_image)
    plt.title("Khuôn mặt được phát hiện")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Trích xuất khuôn mặt lớn nhất
    face_image = detector.extract_face(image)
    if face_image is not None:
        plt.figure(figsize=(5, 5))
        plt.imshow(face_image)
        plt.title("Khuôn mặt được trích xuất")
        plt.axis('off')
        plt.show()
    
if __name__ == "__main__":
    # Demo với một ảnh mẫu
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "data_test/16_1_0_20170109214419099.jpg"  # Ảnh mặc định
    
    demo_face_detection(image_path) 