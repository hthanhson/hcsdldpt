import os
import json
import numpy as np
import h5py
from tqdm import tqdm
import shutil
from utils import extract_person_id, load_image
from face_detection import FaceDetector
from feature_extraction import FaceFeatureExtractor

class FaceDatabase:
    def __init__(self, database_dir='database'):
        """
        Khởi tạo cơ sở dữ liệu khuôn mặt
        
        Parameters:
        - database_dir: Thư mục lưu trữ cơ sở dữ liệu
        """
        self.database_dir = database_dir
        self.features_file = os.path.join(database_dir, 'face_features.h5')
        self.metadata_file = os.path.join(database_dir, 'metadata.json')
        self.feature_extractor = FaceFeatureExtractor()
        
        # Tạo thư mục database nếu chưa tồn tại
        if not os.path.exists(database_dir):
            os.makedirs(database_dir)
        
        # Khởi tạo cấu trúc cơ sở dữ liệu
        self.features = None  # Ma trận chứa các vector đặc trưng
        self.metadata = {
            'filenames': [],
            'attributes': []  # Chứa các thuộc tính: màu da, tuổi, cảm xúc
        }
    
    def build_database(self, data_dir, force_rebuild=False):
        """
        Xây dựng cơ sở dữ liệu từ thư mục dữ liệu
        
        Parameters:
        - data_dir: Thư mục chứa dữ liệu ảnh
        - force_rebuild: Xây dựng lại cơ sở dữ liệu nếu đã tồn tại
        
        Returns:
        - success: True nếu xây dựng thành công, False nếu thất bại
        """
        # Kiểm tra xem cơ sở dữ liệu đã tồn tại chưa
        if os.path.exists(self.features_file) and os.path.exists(self.metadata_file) and not force_rebuild:
            print(f"Cơ sở dữ liệu đã tồn tại tại {self.database_dir}")
            print("Sử dụng force_rebuild=True để xây dựng lại cơ sở dữ liệu")
            return self.load_database()
        
        # Tìm tất cả các file ảnh trong thư mục dữ liệu
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print(f"Không tìm thấy ảnh trong thư mục {data_dir}")
            return False
        
        print(f"Tìm thấy {len(image_files)} ảnh để xử lý")
        
        # Trích xuất đặc trưng từ mỗi ảnh
        all_features = []
        filenames = []
        attributes_list = []
        
        for image_file in tqdm(image_files, desc="Trích xuất đặc trưng"):
            try:
                # Trích xuất đặc trưng từ ảnh
                features_data = self.feature_extractor.extract_all_features(image_file)
                
                if features_data is None:
                    continue
                
                # Lấy tên file
                filename = os.path.basename(image_file)
                
                # Trích xuất ảnh và phân loại thuộc tính
                image = load_image(image_file)
                face_detector = FaceDetector()
                face_image = face_detector.extract_face(image)
                
                if face_image is not None:
                    attributes = self.feature_extractor.classify_attributes(face_image)
                    
                    # Thêm các giá trị cụ thể để hỗ trợ so sánh
                    additional_features = features_data['additional_features']
                    attributes['skin_value'] = additional_features['skin_value'] 
                    attributes['age_value'] = additional_features['age_value']
                    attributes['emotion_value'] = additional_features['emotion_value']
                else:
                    attributes = None
                
                # Lưu thông tin vào danh sách
                all_features.append(features_data['embedding'])
                filenames.append(filename)
                attributes_list.append(attributes)
                
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {image_file}: {str(e)}")
        
        if not all_features:
            print("Không thể trích xuất đặc trưng từ bất kỳ ảnh nào")
            return False
        
        # Chuyển đổi danh sách đặc trưng thành mảng numpy
        self.features = np.array(all_features)
        
        # Cập nhật metadata
        self.metadata = {
            'filenames': filenames,
            'attributes': attributes_list
        }
        
        # Lưu cơ sở dữ liệu
        self._save_database()
        
        print(f"Đã xây dựng cơ sở dữ liệu với {len(filenames)} khuôn mặt")
        return True
    
    def _save_database(self):
        """
        Lưu cơ sở dữ liệu vào tệp
        """
        # Lưu vector đặc trưng vào file HDF5
        with h5py.File(self.features_file, 'w') as f:
            f.create_dataset('features', data=self.features)
        
        # Lưu metadata vào file JSON
        with open(self.metadata_file, 'w') as f:
            # Chuyển đổi các đối tượng numpy thành danh sách Python
            metadata_json = self.metadata.copy()
            
            # Xử lý các đối tượng NumPy trong attributes
            for i, attrs in enumerate(metadata_json['attributes']):
                if attrs is not None:
                    for key, value in attrs.items():
                        if isinstance(value, (np.ndarray, np.number)):
                            attrs[key] = value.tolist()
            
            json.dump(metadata_json, f, indent=2)
    
    def load_database(self):
        """
        Đọc cơ sở dữ liệu từ tệp
        
        Returns:
        - success: True nếu đọc thành công, False nếu thất bại
        """
        if not os.path.exists(self.features_file) or not os.path.exists(self.metadata_file):
            print(f"Không tìm thấy cơ sở dữ liệu tại {self.database_dir}")
            return False
        
        # Đọc vector đặc trưng từ file HDF5
        with h5py.File(self.features_file, 'r') as f:
            self.features = f['features'][:]
        
        # Đọc metadata từ file JSON
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Đã đọc cơ sở dữ liệu với {len(self.metadata['filenames'])} khuôn mặt")
        return True
    
    def search_similar_faces(self, query_features, top_k=3):
        """
        Tìm kiếm khuôn mặt tương tự dựa trên vector đặc trưng
        
        Parameters:
        - query_features: Vector đặc trưng của khuôn mặt cần tìm
        - top_k: Số lượng kết quả trả về
        
        Returns:
        - similar_indices: Chỉ số của các khuôn mặt tương tự nhất
        - similarity_scores: Điểm tương đồng tương ứng
        """
        if self.features is None:
            print("Cơ sở dữ liệu chưa được tải")
            return [], []
        
        # Tính toán độ tương đồng cosine
        # sim = dot(a, b) / (||a|| * ||b||)
        # Giả sử vector đặc trưng đã được chuẩn hóa (có norm = 1)
        similarity_scores = np.dot(self.features, query_features)
        
        # Sắp xếp theo thứ tự giảm dần của độ tương đồng
        similar_indices = np.argsort(similarity_scores)[::-1][:top_k]
        similarity_scores = similarity_scores[similar_indices]
        
        return similar_indices, similarity_scores
    
    def search_similar_faces_by_image(self, query_features, image_path, query_attributes, top_k=3):
        """
        Tìm kiếm khuôn mặt tương tự dựa trên vector đặc trưng và ảnh đầu vào
        
        Parameters:
        - query_features: Vector đặc trưng của khuôn mặt cần tìm
        - image_path: Đường dẫn đến ảnh đầu vào
        - query_attributes: Thuộc tính của khuôn mặt đầu vào
        - top_k: Số lượng kết quả trả về
        
        Returns:
        - similar_faces: Danh sách ảnh khuôn mặt tương tự
        - similar_filenames: Danh sách tên file ảnh tương tự 
        - similarity_scores: Điểm tương đồng tương ứng
        - similarity_reasons: Lý do tương đồng cho mỗi kết quả
        """
        if self.features is None:
            print("Cơ sở dữ liệu chưa được tải")
            return [], [], [], []
        
        # Tính toán độ tương đồng cosine
        # sim = dot(a, b) / (||a|| * ||b||)
        # Giả sử vector đặc trưng đã được chuẩn hóa (có norm = 1)
        similarity_scores = np.dot(self.features, query_features)
        
        # Sắp xếp theo thứ tự giảm dần của độ tương đồng
        similar_indices = np.argsort(similarity_scores)[::-1]
        
        # Lọc ra các file khác với file đầu vào (nếu có trong DB)
        image_filename = os.path.basename(image_path)
        filtered_indices = []
        for idx in similar_indices:
            if self.metadata['filenames'][idx] != image_filename:
                filtered_indices.append(idx)
            if len(filtered_indices) >= top_k:
                break
        
        if not filtered_indices:
            print("Không tìm thấy khuôn mặt tương tự")
            return [], [], [], []
        
        # Lấy tên file, điểm tương đồng và ảnh khuôn mặt
        similar_filenames = []
        result_scores = []
        similar_faces = []
        
        for idx in filtered_indices:
            filename = self.metadata['filenames'][idx]
            score = similarity_scores[idx]
            
            # Tìm đường dẫn đầy đủ của file
            full_path = None
            for root, _, files in os.walk(os.path.dirname(image_path)):
                if filename in files:
                    full_path = os.path.join(root, filename)
                    break
            
            if full_path:
                # Đọc ảnh gốc
                image = load_image(full_path)
                face_detector = FaceDetector()
                face_image = face_detector.extract_face(image)
                
                if face_image is not None:
                    similar_faces.append(face_image)
                    similar_filenames.append(full_path)
                    result_scores.append(score)
        
        # Tính toán lý do tương đồng
        similarity_reasons = []
        for filename in similar_filenames:
            # Lấy thuộc tính của ảnh tương tự
            similar_attrs = self.get_attributes_by_filename(os.path.basename(filename))
            
            # Tạo lý do tương đồng
            reasons = []
            
            # So sánh màu da
            if similar_attrs['skin_tone'] == query_attributes['skin_tone']:
                reasons.append(f"cùng màu da ({similar_attrs['skin_tone']})")
            
            # So sánh nhóm tuổi
            if similar_attrs['age_group'] == query_attributes['age_group']:
                reasons.append(f"cùng nhóm tuổi ({similar_attrs['age_group']})")
            
            # So sánh cảm xúc
            if similar_attrs['emotion'] == query_attributes['emotion']:
                reasons.append(f"cùng cảm xúc ({similar_attrs['emotion']})")
            
            if not reasons:
                reasons.append("đặc trưng khuôn mặt tương tự")
            
            similarity_reasons.append(", ".join(reasons))
        
        return similar_faces, similar_filenames, result_scores, similarity_reasons
    
    def search_similar_faces_with_attributes(self, query_features, image_path, query_attributes, required_attrs, top_k=10):
        """
        Tìm kiếm khuôn mặt tương tự dựa trên vector đặc trưng và các thuộc tính yêu cầu
        
        Parameters:
        - query_features: Vector đặc trưng của khuôn mặt cần tìm
        - image_path: Đường dẫn đến ảnh đầu vào
        - query_attributes: Thuộc tính của khuôn mặt đầu vào
        - required_attrs: Từ điển chứa các thuộc tính yêu cầu, ví dụ {'skin_tone': 'light'}
        - top_k: Số lượng kết quả trả về
        
        Returns:
        - similar_faces: Danh sách ảnh khuôn mặt tương tự
        - similar_filenames: Danh sách tên file ảnh tương tự
        - similarity_scores: Điểm tương đồng tương ứng
        - similarity_reasons: Lý do tương đồng cho mỗi kết quả
        """
        if self.features is None:
            print("Cơ sở dữ liệu chưa được tải")
            return [], [], [], []
        
        # Tính toán độ tương đồng cosine
        similarity_scores = np.dot(self.features, query_features)
        
        # Lọc các khuôn mặt có thuộc tính phù hợp
        filtered_indices = []
        filtered_scores = []
        
        # Lấy danh sách chỉ mục theo thứ tự giảm dần của điểm tương đồng
        sorted_indices = np.argsort(similarity_scores)[::-1]
        
        # Lọc theo thuộc tính yêu cầu
        image_filename = os.path.basename(image_path)
        
        for idx in sorted_indices:
            # Bỏ qua nếu là ảnh đầu vào
            if self.metadata['filenames'][idx] == image_filename:
                continue
            
            # Lấy thuộc tính của ảnh
            attrs = self.metadata['attributes'][idx]
            if attrs is None:
                continue
            
            # Kiểm tra xem có thỏa mãn tất cả các thuộc tính yêu cầu không
            match = True
            for attr_name, attr_value in required_attrs.items():
                if attr_name not in attrs or attrs[attr_name] != attr_value:
                    match = False
                    break
            
            if match:
                filtered_indices.append(idx)
                filtered_scores.append(similarity_scores[idx])
            
            # Lấy đủ top_k kết quả
            if len(filtered_indices) >= top_k:
                break
        
        if not filtered_indices:
            print(f"Không tìm thấy khuôn mặt nào thỏa mãn các thuộc tính: {required_attrs}")
            return [], [], [], []
        
        # Lấy tên file, điểm tương đồng và ảnh khuôn mặt
        similar_filenames = []
        similar_faces = []
        result_scores = []
        
        for idx, score in zip(filtered_indices, filtered_scores):
            filename = self.metadata['filenames'][idx]
            
            # Tìm đường dẫn đầy đủ của file
            full_path = None
            for root, _, files in os.walk(os.path.dirname(image_path)):
                if filename in files:
                    full_path = os.path.join(root, filename)
                    break
            
            if full_path:
                # Đọc ảnh gốc
                image = load_image(full_path)
                face_detector = FaceDetector()
                face_image = face_detector.extract_face(image)
                
                if face_image is not None:
                    similar_faces.append(face_image)
                    similar_filenames.append(full_path)
                    result_scores.append(score)
        
        # Tính toán lý do tương đồng
        similarity_reasons = []
        for filename in similar_filenames:
            # Lấy thuộc tính của ảnh tương tự
            similar_attrs = self.get_attributes_by_filename(os.path.basename(filename))
            
            # Tạo lý do tương đồng
            reasons = []
            
            # So sánh màu da
            if 'skin_tone' in required_attrs:
                reasons.append(f"cùng màu da ({required_attrs['skin_tone']})")
            elif similar_attrs['skin_tone'] == query_attributes['skin_tone']:
                reasons.append(f"cùng màu da ({similar_attrs['skin_tone']})")
            
            # So sánh nhóm tuổi
            if 'age_group' in required_attrs:
                reasons.append(f"cùng nhóm tuổi ({required_attrs['age_group']})")
            elif similar_attrs['age_group'] == query_attributes['age_group']:
                reasons.append(f"cùng nhóm tuổi ({similar_attrs['age_group']})")
            
            # So sánh cảm xúc
            if 'emotion' in required_attrs:
                reasons.append(f"cùng cảm xúc ({required_attrs['emotion']})")
            elif similar_attrs['emotion'] == query_attributes['emotion']:
                reasons.append(f"cùng cảm xúc ({similar_attrs['emotion']})")
            
            if not reasons:
                reasons.append("đặc trưng khuôn mặt tương tự")
            
            similarity_reasons.append(", ".join(reasons))
        
        return similar_faces, similar_filenames, result_scores, similarity_reasons
    
    def get_face_image_by_filename(self, filename, data_dir):
        """
        Lấy ảnh khuôn mặt từ tên file
        
        Parameters:
        - filename: Tên file ảnh
        - data_dir: Thư mục chứa dữ liệu ảnh
        
        Returns:
        - face_image: Ảnh khuôn mặt đã được cắt
        """
        # Tìm đường dẫn đầy đủ của file
        image_path = None
        for root, _, files in os.walk(data_dir):
            if filename in files:
                image_path = os.path.join(root, filename)
                break
        
        if image_path is None:
            print(f"Không tìm thấy file {filename} trong thư mục {data_dir}")
            return None
        
        # Đọc ảnh và trích xuất khuôn mặt
        image = load_image(image_path)
        face_detector = FaceDetector()
        face_image = face_detector.extract_face(image)
        
        return face_image
    
    def get_attributes_by_filename(self, filename):
        """
        Lấy thuộc tính khuôn mặt từ tên file
        
        Parameters:
        - filename: Tên file ảnh
        
        Returns:
        - attributes: Thuộc tính khuôn mặt
        """
        try:
            index = self.metadata['filenames'].index(filename)
            return self.metadata['attributes'][index]
        except (ValueError, IndexError):
            return None
    
    def backup_database(self, backup_dir=None):
        """
        Sao lưu cơ sở dữ liệu
        
        Parameters:
        - backup_dir: Thư mục sao lưu. Nếu None, sẽ sử dụng database_dir + '_backup'
        
        Returns:
        - success: True nếu sao lưu thành công, False nếu thất bại
        """
        if backup_dir is None:
            backup_dir = f"{self.database_dir}_backup"
        
        if not os.path.exists(self.features_file) or not os.path.exists(self.metadata_file):
            print(f"Không tìm thấy cơ sở dữ liệu tại {self.database_dir}")
            return False
        
        # Tạo thư mục sao lưu nếu chưa tồn tại
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        # Sao chép các file cơ sở dữ liệu
        backup_features_file = os.path.join(backup_dir, os.path.basename(self.features_file))
        backup_metadata_file = os.path.join(backup_dir, os.path.basename(self.metadata_file))
        
        shutil.copy2(self.features_file, backup_features_file)
        shutil.copy2(self.metadata_file, backup_metadata_file)
        
        print(f"Đã sao lưu cơ sở dữ liệu tại {backup_dir}")
        return True
    
    def get_database_statistics(self):
        """
        Lấy thống kê về cơ sở dữ liệu
        
        Returns:
        - stats: Từ điển chứa thống kê về cơ sở dữ liệu
        """
        if self.features is None or not self.metadata:
            print("Cơ sở dữ liệu chưa được tải")
            return {}
        
        # Số lượng khuôn mặt
        num_faces = len(self.metadata['filenames'])
        
        # Thống kê các thuộc tính
        skin_tone_stats = {}
        age_group_stats = {}
        emotion_stats = {}
        
        for attrs in self.metadata['attributes']:
            if attrs is None:
                continue
                
            # Thống kê màu da
            skin_tone = attrs.get('skin_tone')
            if skin_tone:
                skin_tone_stats[skin_tone] = skin_tone_stats.get(skin_tone, 0) + 1
            
            # Thống kê nhóm tuổi
            age_group = attrs.get('age_group')
            if age_group:
                age_group_stats[age_group] = age_group_stats.get(age_group, 0) + 1
            
            # Thống kê cảm xúc
            emotion = attrs.get('emotion')
            if emotion:
                emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1
        
        # Tạo từ điển thống kê
        stats = {
            'num_faces': num_faces,
            'skin_tone_stats': skin_tone_stats,
            'age_group_stats': age_group_stats,
            'emotion_stats': emotion_stats
        }
        
        return stats
    
    def print_database_statistics(self):
        """
        In thống kê về cơ sở dữ liệu
        """
        stats = self.get_database_statistics()
        
        if not stats:
            return
        
        print(f"Thống kê cơ sở dữ liệu:")
        print(f"- Số lượng khuôn mặt: {stats['num_faces']}")
        
        print("\nThống kê màu da:")
        for tone, count in stats['skin_tone_stats'].items():
            print(f"- {tone}: {count} khuôn mặt")
        
        print("\nThống kê nhóm tuổi:")
        for age, count in stats['age_group_stats'].items():
            print(f"- {age}: {count} khuôn mặt")
        
        print("\nThống kê cảm xúc:")
        for emotion, count in stats['emotion_stats'].items():
            print(f"- {emotion}: {count} khuôn mặt")

# Demo sử dụng cơ sở dữ liệu
def demo_database(data_dir='data_test', rebuild=False):
    # Khởi tạo cơ sở dữ liệu
    db = FaceDatabase()
    
    # Xây dựng cơ sở dữ liệu
    if rebuild or not os.path.exists(db.features_file) or not os.path.exists(db.metadata_file):
        print("Đang xây dựng cơ sở dữ liệu...")
        db.build_database(data_dir, force_rebuild=rebuild)
    else:
        print("Đang đọc cơ sở dữ liệu...")
        db.load_database()
    
    # In thống kê về cơ sở dữ liệu
    db.print_database_statistics()

if __name__ == "__main__":
    import sys
    rebuild = len(sys.argv) > 1 and sys.argv[1].lower() == 'rebuild'
    demo_database(rebuild=rebuild) 