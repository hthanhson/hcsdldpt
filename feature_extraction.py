import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, InceptionResNetV2
from face_detection import FaceDetector
from utils import load_image, normalize_image

class FaceFeatureExtractor:
    def __init__(self, model_type='inception_resnet'):
        """
        Khởi tạo bộ trích xuất đặc trưng khuôn mặt
        
        Parameters:
        - model_type: Loại mô hình để trích xuất đặc trưng ('vgg16', 'inception_resnet')
        """
        self.model_type = model_type
        self.face_detector = FaceDetector()
        self.model = self._build_model()
        
        # Định nghĩa các ngưỡng cho các đặc trưng
        # Ngưỡng màu da (dựa trên giá trị màu trong không gian YCrCb)
        self.skin_tone_thresholds = {
            'light': (140, 170),   # Ngưỡng cho da sáng
            'medium': (110, 140),  # Ngưỡng cho da trung bình
            'dark': (80, 110)      # Ngưỡng cho da tối
        }
        
        # Ngưỡng tuổi (dựa trên tỉ lệ khuôn mặt và vị trí các điểm mốc)
        self.age_thresholds = {
            'child': 0.5,        # Tỉ lệ khuôn mặt cho trẻ em
            'young_adult': 0.55, # Tỉ lệ khuôn mặt cho thanh niên
            'adult': 0.6,        # Tỉ lệ khuôn mặt cho người trưởng thành
            'senior': 0.65       # Tỉ lệ khuôn mặt cho người cao tuổi
        }
        
        # Ngưỡng cảm xúc (dựa trên vị trí tương đối của miệng và mắt)
        self.emotion_thresholds = {
            'smile': 1.5,       # Tỉ lệ miệng rộng so với khoảng cách mắt
            'neutral': 1.0      # Tỉ lệ miệng tiêu chuẩn
        }
        
    def _build_model(self):
        """
        Xây dựng mô hình deep learning để trích xuất đặc trưng
        """
        if self.model_type == 'vgg16':
            # Sử dụng VGG16 pretrained trên ImageNet
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
        elif self.model_type == 'inception_resnet':
            # Sử dụng InceptionResNetV2 pretrained trên ImageNet
            base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv_7b_ac').output)
        else:
            raise ValueError(f"Model type '{self.model_type}' is not supported")
        
        return model
    
    def extract_features(self, face_image):
        """
        Trích xuất vector đặc trưng từ ảnh khuôn mặt đã cắt
        
        Parameters:
        - face_image: Ảnh khuôn mặt đã được cắt và thay đổi kích thước (224x224x3)
        
        Returns:
        - features: Vector đặc trưng của khuôn mặt
        """
        # Đảm bảo kích thước ảnh đúng
        if face_image.shape[:2] != (224, 224):
            face_image = cv2.resize(face_image, (224, 224))
        
        # Chuẩn hóa ảnh
        if np.max(face_image) > 1.0:
            face_image = normalize_image(face_image)
        
        # Mở rộng chiều để phù hợp với đầu vào batch của model
        face_image = np.expand_dims(face_image, axis=0)
        
        # Trích xuất đặc trưng
        features = self.model.predict(face_image, verbose=0)
        
        # Làm phẳng vector đặc trưng
        features = features.flatten()
        
        # Chuẩn hóa vector đặc trưng
        features = features / np.linalg.norm(features)
        
        return features
    
    def extract_face_and_features(self, image_path):
        """
        Phát hiện khuôn mặt và trích xuất đặc trưng từ ảnh
        
        Parameters:
        - image_path: Đường dẫn đến ảnh
        
        Returns:
        - face_image: Ảnh khuôn mặt đã được cắt
        - features: Vector đặc trưng của khuôn mặt
        Trả về (None, None) nếu không tìm thấy khuôn mặt
        """
        # Đọc ảnh
        image = load_image(image_path)
        
        # Phát hiện và trích xuất khuôn mặt lớn nhất
        face_image = self.face_detector.extract_face(image)
        
        if face_image is None:
            print(f"Không tìm thấy khuôn mặt trong ảnh: {image_path}")
            return None, None
        
        # Trích xuất đặc trưng
        features = self.extract_features(face_image)
        
        return face_image, features

    def detect_skin_tone(self, face_image):
        """
        Phát hiện màu da
        
        Parameters:
        - face_image: Ảnh khuôn mặt đã được cắt
        
        Returns:
        - skin_tone: Màu da ('light', 'medium', 'dark')
        - skin_value: Giá trị số đại diện cho màu da
        """
        # Chuyển đổi ảnh sang không gian màu YCrCb (tốt cho việc phân tích màu da)
        ycrcb_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2YCrCb)
        
        # Lấy kênh Y (độ sáng)
        y_channel = ycrcb_image[:, :, 0]
        
        # Tạo mặt nạ cho vùng khuôn mặt
        # Giả sử vùng giữa khuôn mặt là vùng da
        height, width = face_image.shape[:2]
        center_x, center_y = width // 2, height // 2
        roi_size = min(width, height) // 3
        roi = y_channel[center_y-roi_size:center_y+roi_size, center_x-roi_size:center_x+roi_size]
        
        # Tính giá trị trung bình của độ sáng trong vùng da
        skin_brightness = np.mean(roi)
        
        # Phân loại màu da dựa trên ngưỡng
        if skin_brightness >= self.skin_tone_thresholds['light'][0]:
            skin_tone = 'light'
        elif skin_brightness >= self.skin_tone_thresholds['medium'][0]:
            skin_tone = 'medium'
        else:
            skin_tone = 'dark'
        
        return skin_tone, skin_brightness
    
    def estimate_age(self, face_image, landmarks):
        """
        Ước lượng độ tuổi dựa trên tỉ lệ khuôn mặt và các điểm mốc
        
        Parameters:
        - face_image: Ảnh khuôn mặt đã được cắt
        - landmarks: Các điểm mốc khuôn mặt
        
        Returns:
        - age_group: Nhóm tuổi ('child', 'young_adult', 'adult', 'senior')
        - age_value: Giá trị số đại diện cho tuổi
        """
        # Trích xuất các điểm mốc quan trọng
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        nose = np.array(landmarks['nose'])
        mouth_left = np.array(landmarks['mouth_left'])
        mouth_right = np.array(landmarks['mouth_right'])
        
        # Tính khoảng cách giữa các điểm mốc
        eye_distance = np.linalg.norm(right_eye - left_eye)
        face_height = face_image.shape[0]
        face_width = face_image.shape[1]
        
        # Tính tỉ lệ khuôn mặt (mắt đến cằm so với chiều cao khuôn mặt)
        eyes_y = (left_eye[1] + right_eye[1]) / 2
        mouth_y = (mouth_left[1] + mouth_right[1]) / 2
        forehead_ratio = eyes_y / face_height
        lower_face_ratio = (mouth_y - eyes_y) / face_height
        
        # Sử dụng tỉ lệ này để ước lượng tuổi
        # Trẻ em có trán rộng hơn và phần dưới mặt ngắn hơn
        # Người cao tuổi có phần dưới mặt dài hơn
        
        # Tính điểm tuổi dựa trên tỉ lệ khuôn mặt
        age_ratio = lower_face_ratio / forehead_ratio if forehead_ratio > 0 else 0
        
        # Phân loại nhóm tuổi
        if age_ratio < self.age_thresholds['child']:
            age_group = 'child'  # 0-12 tuổi
        elif age_ratio < self.age_thresholds['young_adult']:
            age_group = 'young_adult'  # 13-30 tuổi
        elif age_ratio < self.age_thresholds['adult']:
            age_group = 'adult'  # 31-60 tuổi
        else:
            age_group = 'senior'  # 60+ tuổi
        
        return age_group, age_ratio
    
    def detect_emotion(self, face_image, landmarks):
        """
        Phát hiện cảm xúc dựa trên hình dạng miệng và vị trí các điểm mốc
        
        Parameters:
        - face_image: Ảnh khuôn mặt đã được cắt
        - landmarks: Các điểm mốc khuôn mặt
        
        Returns:
        - emotion: Cảm xúc ('smile', 'neutral', 'serious')
        - emotion_value: Giá trị số đại diện cho cảm xúc
        """
        # Trích xuất các điểm mốc quan trọng
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        mouth_left = np.array(landmarks['mouth_left'])
        mouth_right = np.array(landmarks['mouth_right'])
        
        # Tính khoảng cách giữa các điểm mốc
        eye_distance = np.linalg.norm(right_eye - left_eye)
        mouth_width = np.linalg.norm(mouth_right - mouth_left)
        
        # Tính tỉ lệ miệng so với khoảng cách mắt
        mouth_eye_ratio = mouth_width / eye_distance if eye_distance > 0 else 0
        
        # Phân loại cảm xúc dựa trên tỉ lệ miệng
        if mouth_eye_ratio > self.emotion_thresholds['smile']:
            emotion = 'smile'  # Cười
        elif mouth_eye_ratio > self.emotion_thresholds['neutral']:
            emotion = 'neutral'  # Bình thường
        else:
            emotion = 'serious'  # Nghiêm túc
        
        return emotion, mouth_eye_ratio

    def extract_additional_features(self, face_image):
        """
        Trích xuất các đặc trưng bổ sung: màu da, tuổi, cảm xúc
        
        Parameters:
        - face_image: Ảnh khuôn mặt đã được cắt
        
        Returns:
        - additional_features: Từ điển chứa các đặc trưng bổ sung
        """
        # Phát hiện khuôn mặt để lấy các điểm mốc
        faces = self.face_detector.detect_faces(face_image)
        
        if not faces:
            return None
        
        face = faces[0]  # Lấy khuôn mặt đầu tiên
        
        # Lấy các điểm mốc khuôn mặt
        landmarks = face['keypoints']
        
        # Chuẩn hóa tọa độ điểm mốc theo kích thước ảnh
        normalized_landmarks = {}
        for key, point in landmarks.items():
            normalized_landmarks[key] = (
                point[0] / face_image.shape[1],  # x / width
                point[1] / face_image.shape[0]   # y / height
            )
        
        # Phát hiện màu da
        skin_tone, skin_value = self.detect_skin_tone(face_image)
        
        # Ước lượng tuổi
        age_group, age_value = self.estimate_age(face_image, landmarks)
        
        # Phát hiện cảm xúc
        emotion, emotion_value = self.detect_emotion(face_image, landmarks)
        
        # Tạo từ điển đặc trưng bổ sung
        additional_features = {
            'landmarks': normalized_landmarks,
            'skin_tone': skin_tone,
            'skin_value': float(skin_value),
            'age_group': age_group,
            'age_value': float(age_value),
            'emotion': emotion,
            'emotion_value': float(emotion_value),
            'confidence': face['confidence']
        }
        
        return additional_features

    def extract_all_features(self, image_path):
        """
        Trích xuất tất cả các đặc trưng từ ảnh: vector embedding và đặc trưng bổ sung
        
        Parameters:
        - image_path: Đường dẫn đến ảnh
        
        Returns:
        - features_data: Từ điển chứa tất cả thông tin và đặc trưng của khuôn mặt
          hoặc None nếu không tìm thấy khuôn mặt
        """
        # Trích xuất khuôn mặt và vector đặc trưng chính
        face_image, embedding = self.extract_face_and_features(image_path)
        
        if face_image is None or embedding is None:
            return None
        
        # Trích xuất các đặc trưng bổ sung
        additional_features = self.extract_additional_features(face_image)
        
        if additional_features is None:
            return None
        
        # Tạo từ điển chứa tất cả thông tin
        features_data = {
            'filename': os.path.basename(image_path),
            'embedding': embedding,
            'additional_features': additional_features
        }
        
        return features_data
    
    def classify_attributes(self, face_image):
        """
        Phân loại ảnh theo các đặc trưng: màu da, tuổi, cảm xúc
        
        Parameters:
        - face_image: Ảnh khuôn mặt đã được cắt
        
        Returns:
        - attributes: Từ điển chứa các đặc trưng được phân loại
        """
        additional_features = self.extract_additional_features(face_image)
        
        if additional_features is None:
            return None
        
        # Tạo từ điển chứa các thuộc tính đã phân loại
        attributes = {
            'skin_tone': additional_features['skin_tone'],
            'age_group': additional_features['age_group'],
            'emotion': additional_features['emotion']
        }
        
        return attributes

# Demo trích xuất đặc trưng
def demo_feature_extraction(image_path):
    # Khởi tạo bộ trích xuất đặc trưng
    extractor = FaceFeatureExtractor()
    
    # Trích xuất khuôn mặt và đặc trưng
    face_image, features = extractor.extract_face_and_features(image_path)
    
    if face_image is None:
        print("Không tìm thấy khuôn mặt trong ảnh")
        return
    
    print(f"Kích thước vector đặc trưng: {features.shape}")
    
    # Trích xuất các đặc trưng bổ sung
    additional_features = extractor.extract_additional_features(face_image)
    
    if additional_features is not None:
        print("\nCác đặc trưng được phát hiện:")
        print(f"- Màu da: {additional_features['skin_tone']} (giá trị: {additional_features['skin_value']:.2f})")
        print(f"- Nhóm tuổi: {additional_features['age_group']} (giá trị: {additional_features['age_value']:.2f})")
        print(f"- Cảm xúc: {additional_features['emotion']} (giá trị: {additional_features['emotion_value']:.2f})")
    
    # Phân loại các thuộc tính
    attributes = extractor.classify_attributes(face_image)
    
    if attributes is not None:
        print("\nCác thuộc tính được phân loại:")
        for key, value in attributes.items():
            print(f"- {key}: {value}")
    
    # Hiển thị ảnh khuôn mặt
    import matplotlib.pyplot as plt
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
    
    demo_feature_extraction(image_path) 