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

        Args:
            data_dir (str): Thư mục chứa dữ liệu ảnh để xây dựng cơ sở dữ liệu
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

        Args:
            image_path (str): Đường dẫn đến ảnh đầu vào
            top_k (int): Số lượng kết quả tương tự trả về

        Returns:
            tuple: (similar_faces, similar_filenames, similarity_scores, face_image, 
                    query_attributes, similarity_reasons)
        """
        print(f"Đang tìm kiếm khuôn mặt tương tự cho ảnh: {image_path}")
        
        # Đọc và kiểm tra ảnh đầu vào
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            print(f"Không thể đọc hoặc ảnh không hợp lệ: {image_path}")
            return [], [], [], None, None, None
        
        # Phát hiện khuôn mặt trong ảnh
        detected_faces = self.face_detector.detect_faces(image)
        if len(detected_faces) == 0:
            print("Không phát hiện khuôn mặt nào trong ảnh đầu vào")
            return [], [], [], None, None, None
        
        # Lấy khuôn mặt rõ nhất (có điểm tin cậy cao nhất)
        try:
            main_face = max(detected_faces, key=lambda x: x['confidence'])
        except KeyError:
            print("Lỗi: Dữ liệu khuôn mặt không chứa key 'confidence'")
            return [], [], [], None, None, None
        
        # Kiểm tra key 'face' trong main_face
        if 'face' not in main_face:
            print("Lỗi: Dữ liệu khuôn mặt không chứa key 'face'")
            return [], [], [], None, None, None
        
        face_image = main_face['face']
        
        # Kiểm tra key 'landmarks' cho feature extraction
        if 'landmarks' not in main_face:
            print("Lỗi: Dữ liệu khuôn mặt không chứa key 'landmarks'")
            return [], [], [], None, None, None
        
        # Trích xuất đặc trưng
        face_vector = self.feature_extractor.extract_features(face_image)
        query_attributes = self.feature_extractor.extract_additional_features(
            face_image, main_face['landmarks'])
        
        # Tìm kiếm khuôn mặt tương tự
        similar_faces, similar_filenames, similarity_scores, similarity_reasons = \
            self.database.search_similar_faces_by_image(
                face_vector, image_path, query_attributes, top_k)
        
        # Hiển thị và lưu kết quả nếu tìm thấy khuôn mặt tương tự
        if similar_filenames:
            self.visualize_results(
                image, face_image, similar_faces, similar_filenames, 
                similarity_scores, query_attributes, similarity_reasons)
            
            # Lưu kết quả
            similar_attributes = [self.database.get_attributes_by_filename(f) 
                               for f in similar_filenames]
            save_results(
                image_path, face_image, similar_faces, similar_filenames, 
                similarity_scores, query_attributes, similar_attributes, 
                similarity_reasons)
        
        return (similar_faces, similar_filenames, similarity_scores, 
                face_image, query_attributes, similarity_reasons)
    
    def search_with_attributes(self, image_path, required_attrs, top_k=3):
        """
        Tìm kiếm khuôn mặt tương tự dựa trên các thuộc tính cụ thể

        Args:
            image_path (str): Đường dẫn đến ảnh đầu vào
            required_attrs (dict): Thuộc tính cần tìm kiếm (e.g., {'skin_tone': 'light', 'emotion': 'smile'})
            top_k (int): Số lượng kết quả trả về

        Returns:
            tuple: (similar_faces, similar_filenames, similarity_scores, face_image, 
                    query_attributes, similarity_reasons)
        """
        print(f"Đang tìm kiếm khuôn mặt với các thuộc tính: {required_attrs}")
        
        # Đọc và kiểm tra ảnh đầu vào
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            print(f"Không thể đọc hoặc ảnh không hợp lệ: {image_path}")
            return [], [], [], None, None, None
        
        # Phát hiện khuôn mặt trong ảnh
        detected_faces = self.face_detector.detect_faces(image)
        if len(detected_faces) == 0:
            print("Không phát hiện khuôn mặt nào trong ảnh đầu vào")
            return [], [], [], None, None, None
        
        # Lấy khuôn mặt rõ nhất (có điểm tin cậy cao nhất)
        try:
            main_face = max(detected_faces, key=lambda x: x['confidence'])
        except KeyError:
            print("Lỗi: Dữ liệu khuôn mặt không chứa key 'confidence'")
            return [], [], [], None, None, None
        
        # Kiểm tra key 'face' trong main_face
        if 'face' not in main_face:
            print("Lỗi: Dữ liệu khuôn mặt không chứa key 'face'")
            return [], [], [], None, None, None
        
        face_image = main_face['face']
        
        # Kiểm tra key 'landmarks' cho feature extraction
        if 'landmarks' not in main_face:
            print("Lỗi: Dữ liệu khuôn mặt không chứa key 'landmarks'")
            return [], [], [], None, None, None
        
        # Trích xuất đặc trưng
        face_vector = self.feature_extractor.extract_features(face_image)
        query_attributes = self.feature_extractor.extract_additional_features(
            face_image, main_face['landmarks'])
        
        # Kiểm tra xem khuôn mặt đầu vào có phù hợp với các thuộc tính yêu cầu không
        for attr, value in required_attrs.items():
            if query_attributes.get(attr) != value:
                print(f"Khuôn mặt đầu vào có thuộc tính {attr}={query_attributes.get(attr)}, "
                      f"không phù hợp với yêu cầu {value}")
        
        # Tìm kiếm khuôn mặt tương tự với các thuộc tính chỉ định
        similar_faces, similar_filenames, similarity_scores, similarity_reasons = \
            self.database.search_similar_faces_with_attributes(
                face_vector, image_path, query_attributes, required_attrs, top_k)
        
        # Hiển thị và lưu kết quả nếu tìm thấy khuôn mặt tương tự
        if similar_filenames:
            self.visualize_results(
                image, face_image, similar_faces, similar_filenames, 
                similarity_scores, query_attributes, similarity_reasons)
            
            # Lưu kết quả
            similar_attributes = [self.database.get_attributes_by_filename(f) 
                               for f in similar_filenames]
            save_results(
                image_path, face_image, similar_faces, similar_filenames, 
                similarity_scores, query_attributes, similar_attributes, 
                similarity_reasons, required_attrs=required_attrs)
        
        return (similar_faces, similar_filenames, similarity_scores, 
                face_image, query_attributes, similarity_reasons)
    
    def visualize_results(self, input_image, input_face, similar_faces, 
                        similar_filenames, similarity_scores, query_attributes, 
                        similarity_reasons):
        """
        Hiển thị kết quả tìm kiếm bằng matplotlib

        Args:
            input_image: Ảnh gốc đầu vào
            input_face: Ảnh khuôn mặt được trích xuất
            similar_faces: Danh sách ảnh khuôn mặt tương tự
            similar_filenames: Danh sách tên file ảnh tương tự
            similarity_scores: Danh sách điểm tương đồng
            query_attributes: Thuộc tính của khuôn mặt đầu vào
            similarity_reasons: Lý do tương đồng
        """
        plt.figure(figsize=(15, 8))
        
        # Hiển thị ảnh đầu vào
        plt.subplot(1, len(similar_faces) + 1, 1)
        input_rgb = cv2.cvtColor(input_face, cv2.COLOR_BGR2RGB)
        plt.imshow(input_rgb)
        plt.title(f"Ảnh đầu vào\n{query_attributes.get('skin_tone', 'N/A')}, "
                 f"{query_attributes.get('age_group', 'N/A')}, "
                 f"{query_attributes.get('emotion', 'N/A')}", fontsize=9)
        plt.axis('off')
        
        # Hiển thị các ảnh tương tự
        for i, (face, filename, score, reason) in enumerate(
                zip(similar_faces, similar_filenames, similarity_scores, similarity_reasons)):
            plt.subplot(1, len(similar_faces) + 1, i + 2)
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            plt.imshow(face_rgb)
            
            # Lấy thuộc tính
            attributes = self.database.get_attributes_by_filename(filename)
            attr_text = (f"{attributes.get('skin_tone', 'N/A')}, "
                        f"{attributes.get('age_group', 'N/A')}, "
                        f"{attributes.get('emotion', 'N/A')}")
            
            plt.title(f"Tương tự #{i+1}\nĐiểm: {score:.4f}\n{attr_text}\n{reason}", 
                     fontsize=8)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def print_search_results(self, similar_filenames, similarity_scores, 
                           query_attributes, similar_attributes, similarity_reasons):
        """
        In kết quả tìm kiếm ra console

        Args:
            similar_filenames: Danh sách tên file ảnh tương tự
            similarity_scores: Danh sách điểm tương đồng
            query_attributes: Thuộc tính của khuôn mặt đầu vào
            similar_attributes: Thuộc tính của các khuôn mặt tương tự
            similarity_reasons: Lý do tương đồng
        """
        print("\n--- Kết quả tìm kiếm ---")
        print(f"\nẢnh đầu vào có các thuộc tính:")
        print(f"  - Màu da: {query_attributes.get('skin_tone', 'N/A')}")
        print(f"  - Nhóm tuổi: {query_attributes.get('age_group', 'N/A')}")
        print(f"  - Cảm xúc: {query_attributes.get('emotion', 'N/A')}")
        
        print("\nCác khuôn mặt tương tự:")
        for i, (filename, score, attrs, reason) in enumerate(
                zip(similar_filenames, similarity_scores, similar_attributes, similarity_reasons)):
            print(f"\n{i+1}. {os.path.basename(filename)} - Điểm tương đồng: {score:.4f}")
            print(f"   Thuộc tính: {attrs.get('skin_tone', 'N/A')}, "
                  f"{attrs.get('age_group', 'N/A')}, {attrs.get('emotion', 'N/A')}")
            print(f"   Lý do: {reason}")

def get_similarity_description(similarity_score):
    """
    Chuyển đổi điểm tương đồng thành mô tả văn bản

    Args:
        similarity_score (float): Điểm tương đồng từ 0 đến 1

    Returns:
        str: Mô tả mức độ tương đồng
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