import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from face_detection import FaceDetector
from feature_extraction import FaceFeatureExtractor
from database import FaceDatabase
from utils import save_results

class FaceSearchEngine:
    """
    Lớp triển khai công cụ tìm kiếm khuôn mặt
    """
    def __init__(self, data_dir='data_test'):
        """
        Khởi tạo công cụ tìm kiếm khuôn mặt
        """
        self.database = FaceDatabase()
        self.database.load_database()
        
        if self.database.is_empty():
            print("Cơ sở dữ liệu trống, đang xây dựng...")
            self.database.build_database(data_dir)
        
        self.face_detector = FaceDetector()
        self.feature_extractor = FaceFeatureExtractor()
    
    def search(self, image_path, top_k=3):
        """
        Tìm kiếm khuôn mặt tương tự trong cơ sở dữ liệu
        """
        print(f"Đang tìm kiếm khuôn mặt tương tự cho ảnh: {image_path}")
        
        # Đọc ảnh đầu vào
        image = cv2.imread(image_path)
        if image is None:
            print(f"Không thể đọc file ảnh: {image_path}")
            return [], [], [], None, None, None
        
        # Phát hiện khuôn mặt trong ảnh
        detected_faces = self.face_detector.detect_faces(image)
        if len(detected_faces) == 0:
            print("Không phát hiện khuôn mặt nào trong ảnh đầu vào")
            return [], [], [], None, None, None
        
        # Lấy khuôn mặt rõ nhất (có điểm tin cậy cao nhất)
        main_face = max(detected_faces, key=lambda x: x['confidence'])
        face_image = main_face['face']
        
        # Trích xuất đặc trưng
        face_vector = self.feature_extractor.extract_features(face_image)
        query_attributes = self.feature_extractor.extract_additional_features(face_image, main_face['landmarks'])
        
        # Tìm kiếm khuôn mặt tương tự
        similar_faces, similar_filenames, similarity_scores, similarity_reasons = \
            self.database.search_similar_faces_by_image(face_vector, image_path, query_attributes, top_k)
        
        # Hiển thị kết quả
        if similar_filenames:
            self.visualize_results(image, face_image, similar_faces, similar_filenames, 
                                similarity_scores, query_attributes, similarity_reasons)
            
            # Lưu kết quả
            similar_attributes = [self.database.get_attributes_by_filename(f) for f in similar_filenames]
            save_results(image_path, face_image, similar_faces, similar_filenames, 
                        similarity_scores, query_attributes, similar_attributes, similarity_reasons)
        
        return similar_faces, similar_filenames, similarity_scores, face_image, query_attributes, similarity_reasons
    
    def search_with_attributes(self, image_path, required_attrs, top_k=3):
        """
        Tìm kiếm khuôn mặt tương tự dựa trên các thuộc tính cụ thể
        
        Args:
            image_path: Đường dẫn đến ảnh đầu vào
            required_attrs: Dict các thuộc tính cần tìm kiếm
                           (ví dụ: {'skin_tone': 'light', 'emotion': 'smile'})
            top_k: Số lượng kết quả trả về
            
        Returns:
            similar_faces: Danh sách các ảnh khuôn mặt tương tự
            similar_filenames: Danh sách tên file ảnh tương tự
            similarity_scores: Danh sách điểm tương đồng
            face_image: Ảnh khuôn mặt đầu vào
            query_attributes: Thuộc tính của khuôn mặt đầu vào
            similarity_reasons: Lý do tương đồng
        """
        print(f"Đang tìm kiếm khuôn mặt với các thuộc tính: {required_attrs}")
        
        # Đọc ảnh đầu vào
        image = cv2.imread(image_path)
        if image is None:
            print(f"Không thể đọc file ảnh: {image_path}")
            return [], [], [], None, None, None
        
        # Phát hiện khuôn mặt trong ảnh
        detected_faces = self.face_detector.detect_faces(image)
        if len(detected_faces) == 0:
            print("Không phát hiện khuôn mặt nào trong ảnh đầu vào")
            return [], [], [], None, None, None
        
        # Lấy khuôn mặt rõ nhất (có điểm tin cậy cao nhất)
        main_face = max(detected_faces, key=lambda x: x['confidence'])
        face_image = main_face['face']
        
        # Trích xuất đặc trưng
        face_vector = self.feature_extractor.extract_features(face_image)
        query_attributes = self.feature_extractor.extract_additional_features(face_image, main_face['landmarks'])
        
        # Kiểm tra xem khuôn mặt đầu vào có phù hợp với các thuộc tính yêu cầu không
        for attr, value in required_attrs.items():
            if query_attributes.get(attr) != value:
                print(f"Khuôn mặt đầu vào có thuộc tính {attr}={query_attributes.get(attr)}, không phù hợp với yêu cầu {value}")
        
        # Tìm kiếm khuôn mặt tương tự với các thuộc tính chỉ định
        similar_faces, similar_filenames, similarity_scores, similarity_reasons = \
            self.database.search_similar_faces_with_attributes(face_vector, image_path, 
                                                        query_attributes, required_attrs, top_k)
        
        # Hiển thị kết quả
        if similar_filenames:
            self.visualize_results(image, face_image, similar_faces, similar_filenames, 
                                similarity_scores, query_attributes, similarity_reasons)
            
            # Lưu kết quả
            similar_attributes = [self.database.get_attributes_by_filename(f) for f in similar_filenames]
            save_results(image_path, face_image, similar_faces, similar_filenames, 
                        similarity_scores, query_attributes, similar_attributes, similarity_reasons, 
                        required_attrs=required_attrs)
        
        return similar_faces, similar_filenames, similarity_scores, face_image, query_attributes, similarity_reasons
    
    def visualize_results(self, input_image, input_face, similar_faces, similar_filenames, 
                        similarity_scores, query_attributes, similarity_reasons):
        """
        Hiển thị kết quả tìm kiếm
        """
        # Chuẩn bị hình ảnh
        plt.figure(figsize=(15, 8))
        
        # Hiển thị ảnh đầu vào
        plt.subplot(1, len(similar_faces) + 1, 1)
        input_rgb = cv2.cvtColor(input_face, cv2.COLOR_BGR2RGB)
        plt.imshow(input_rgb)
        plt.title(f"Ảnh đầu vào\n{query_attributes['skin_tone']}, {query_attributes['age_group']}, {query_attributes['emotion']}", 
                fontsize=9)
        plt.axis('off')
        
        # Hiển thị các ảnh tương tự
        for i, (face, filename, score, reason) in enumerate(zip(similar_faces, similar_filenames, 
                                                            similarity_scores, similarity_reasons)):
            plt.subplot(1, len(similar_faces) + 1, i + 2)
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            plt.imshow(face_rgb)
            
            # Lấy thuộc tính
            attributes = self.database.get_attributes_by_filename(filename)
            attr_text = f"{attributes['skin_tone']}, {attributes['age_group']}, {attributes['emotion']}"
            
            plt.title(f"Tương tự #{i+1}\nĐiểm: {score:.4f}\n{attr_text}\n{reason}", fontsize=8)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def print_search_results(self, similar_filenames, similarity_scores, 
                           query_attributes, similar_attributes, similarity_reasons):
        """
        In kết quả tìm kiếm
        """
        print("\n--- Kết quả tìm kiếm ---")
        print(f"\nẢnh đầu vào có các thuộc tính:")
        print(f"  - Màu da: {query_attributes['skin_tone']}")
        print(f"  - Nhóm tuổi: {query_attributes['age_group']}")
        print(f"  - Cảm xúc: {query_attributes['emotion']}")
        
        print("\nCác khuôn mặt tương tự:")
        for i, (filename, score, attrs, reason) in enumerate(zip(similar_filenames, 
                                                            similarity_scores, 
                                                            similar_attributes,
                                                            similarity_reasons)):
            print(f"\n{i+1}. {os.path.basename(filename)} - Điểm tương đồng: {score:.4f}")
            print(f"   Thuộc tính: {attrs['skin_tone']}, {attrs['age_group']}, {attrs['emotion']}")
            print(f"   Lý do: {reason}")

def get_similarity_description(similarity_score):
    """
    Chuyển đổi điểm tương đồng thành mô tả
    """
    if similarity_score >= 0.9:
        return "Gần như giống hệt"
    elif similarity_score >= 0.8:
        return "Rất giống nhau"
    elif similarity_score >= 0.7:
        return "Giống nhau đáng kể"
    elif similarity_score >= 0.6:
        return "Tương đối giống nhau"
    elif similarity_score >= 0.5:
        return "Hơi giống nhau"
    else:
        return "Không giống nhau nhiều" 