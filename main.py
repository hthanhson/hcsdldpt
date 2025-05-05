import os
import argparse
import matplotlib.pyplot as plt
from database import FaceDatabase
from search_engine import FaceSearchEngine
from face_detection import FaceDetector, demo_face_detection
from feature_extraction import FaceFeatureExtractor, demo_feature_extraction

def parse_arguments():
    """
    Phân tích các đối số dòng lệnh
    """
    parser = argparse.ArgumentParser(description='Hệ thống tìm kiếm ảnh khuôn mặt')
    
    # Chế độ hoạt động
    parser.add_argument('--mode', type=str, default='search',
                        choices=['search', 'build_database', 'demo'],
                        help='Chế độ hoạt động (search, build_database, demo)')
    
    # Các tham số cho chế độ tìm kiếm
    parser.add_argument('--image', type=str, default=r'E:\hcsdldpt\data\10_0_0_20161220222308131.jpg',
                        help='Đường dẫn đến ảnh đầu vào cho chế độ tìm kiếm')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Số lượng kết quả trả về trong chế độ tìm kiếm')
    
    # Các tham số cho chế độ xây dựng cơ sở dữ liệu
    parser.add_argument('--data_dir', type=str, default='data_test',
                        help='Đường dẫn đến thư mục dữ liệu ảnh')
    parser.add_argument('--force_rebuild', action='store_true',
                        help='Xây dựng lại cơ sở dữ liệu nếu đã tồn tại')
    
    # Các tham số cho chế độ demo
    parser.add_argument('--demo_type', type=str, default='all',
                        choices=['all', 'detection', 'feature', 'search'],
                        help='Loại demo (detection, feature, search, all)')
    
    # Các tham số cho tìm kiếm theo thuộc tính
    parser.add_argument('--skin_tone', type=str, default=None,
                        choices=['light', 'medium', 'dark'],
                        help='Tìm kiếm theo màu da')
    parser.add_argument('--age_group', type=str, default=None,
                        choices=['child', 'young_adult', 'adult', 'senior'],
                        help='Tìm kiếm theo nhóm tuổi')
    parser.add_argument('--emotion', type=str, default=None,
                        choices=['smile', 'neutral', 'serious'],
                        help='Tìm kiếm theo cảm xúc')
    
    return parser.parse_args()

def main():
    """
    Hàm chính của chương trình
    """
    # Phân tích đối số dòng lệnh
    args = parse_arguments()
    
    # Kiểm tra thư mục dữ liệu
    if not os.path.exists(args.data_dir):
        print(f"Thư mục dữ liệu {args.data_dir} không tồn tại")
        return
    
    if args.mode == 'search':
        # Chế độ tìm kiếm
        if args.image is None:
            print("Vui lòng cung cấp đường dẫn đến ảnh đầu vào với tham số --image")
            return
        
        if not os.path.exists(args.image):
            print(f"Không tìm thấy file ảnh {args.image}")
            return
        
        # Khởi tạo động cơ tìm kiếm
        search_engine = FaceSearchEngine(data_dir=args.data_dir)
        
        # Kiểm tra xem có tìm kiếm theo thuộc tính không
        required_attrs = {}
        if args.skin_tone:
            required_attrs['skin_tone'] = args.skin_tone
        if args.age_group:
            required_attrs['age_group'] = args.age_group
        if args.emotion:
            required_attrs['emotion'] = args.emotion
        
        if required_attrs:
            # Tìm kiếm theo thuộc tính
            print(f"Tìm kiếm ảnh với các thuộc tính: {required_attrs}")
            similar_images, similar_filenames, similarity_scores, query_face_image, query_attributes, similarity_reasons = \
                search_engine.search_with_attributes(args.image, required_attrs, top_k=10)
        else:
            # Tìm kiếm thông thường
            similar_images, similar_filenames, similarity_scores, query_face_image, query_attributes, similarity_reasons = \
                search_engine.search(args.image, top_k=args.top_k)
        
        # In kết quả tìm kiếm
        if similar_filenames:
            similar_attributes = [search_engine.database.get_attributes_by_filename(f) for f in similar_filenames]
            search_engine.print_search_results(similar_filenames, similarity_scores, 
                                            query_attributes, similar_attributes, similarity_reasons)
    
    elif args.mode == 'build_database':
        # Chế độ xây dựng cơ sở dữ liệu
        db = FaceDatabase()
        print(f"Đang xây dựng cơ sở dữ liệu từ thư mục {args.data_dir}...")
        db.build_database(args.data_dir, force_rebuild=args.force_rebuild)
        
        # In thống kê về cơ sở dữ liệu
        db.print_database_statistics()
    
    elif args.mode == 'demo':
        # Chế độ demo
        if args.demo_type == 'detection' or args.demo_type == 'all':
            # Demo phát hiện khuôn mặt
            print("\n--- Demo phát hiện khuôn mặt ---")
            image_files = []
            for root, _, files in os.walk(args.data_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_files.append(os.path.join(root, file))
            
            if image_files:
                demo_image = image_files[0]  # Chọn ảnh đầu tiên
                print(f"Ảnh demo: {demo_image}")
                demo_face_detection(demo_image)
            else:
                print(f"Không tìm thấy ảnh trong thư mục {args.data_dir}")
        
        if args.demo_type == 'feature' or args.demo_type == 'all':
            # Demo trích xuất đặc trưng
            print("\n--- Demo trích xuất đặc trưng khuôn mặt ---")
            image_files = []
            for root, _, files in os.walk(args.data_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_files.append(os.path.join(root, file))
            
            if image_files:
                demo_image = image_files[0]  # Chọn ảnh đầu tiên
                print(f"Ảnh demo: {demo_image}")
                demo_feature_extraction(demo_image)
            else:
                print(f"Không tìm thấy ảnh trong thư mục {args.data_dir}")
        
        if args.demo_type == 'search' or args.demo_type == 'all':
            # Demo tìm kiếm
            print("\n--- Demo tìm kiếm khuôn mặt ---")
            if args.image:
                demo_image = args.image
            else:
                # Chọn ảnh ngẫu nhiên
                import random
                image_files = []
                for root, _, files in os.walk(args.data_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_files.append(os.path.join(root, file))
                
                if not image_files:
                    print(f"Không tìm thấy ảnh trong thư mục {args.data_dir}")
                    return
                
                demo_image = random.choice(image_files)
            
            print(f"Ảnh demo: {demo_image}")
            search_engine = FaceSearchEngine(data_dir=args.data_dir)
            similar_images, similar_filenames, similarity_scores, query_face_image, query_attributes, similarity_reasons = \
                search_engine.search(demo_image, top_k=args.top_k)
            
            # Demo tìm kiếm theo thuộc tính
            if query_attributes:
                print("\n--- Demo tìm kiếm theo thuộc tính ---")
                
                # Demo tìm kiếm theo màu da
                print(f"\nTìm kiếm theo màu da: {query_attributes['skin_tone']}")
                search_engine.search_with_attributes(
                    demo_image, 
                    required_attrs={'skin_tone': query_attributes['skin_tone']}
                )
                
                # Demo tìm kiếm theo nhóm tuổi
                print(f"\nTìm kiếm theo nhóm tuổi: {query_attributes['age_group']}")
                search_engine.search_with_attributes(
                    demo_image, 
                    required_attrs={'age_group': query_attributes['age_group']}
                )
                
                # Demo tìm kiếm theo cảm xúc
                print(f"\nTìm kiếm theo cảm xúc: {query_attributes['emotion']}")
                search_engine.search_with_attributes(
                    demo_image, 
                    required_attrs={'emotion': query_attributes['emotion']}
                )
    
    else:
        print(f"Chế độ không hợp lệ: {args.mode}")

def visualize_face_attributes():
    """
    Hiển thị các đặc trưng khuôn mặt từ cơ sở dữ liệu
    """
    db = FaceDatabase()
    if not db.load_database():
        print("Không thể tải cơ sở dữ liệu. Vui lòng xây dựng cơ sở dữ liệu trước.")
        return
    
    stats = db.get_database_statistics()
    
    # Hiển thị thống kê các thuộc tính
    plt.figure(figsize=(15, 12))
    
    # Vẽ thống kê màu da
    plt.subplot(3, 1, 1)
    skin_tones = list(stats['skin_tone_stats'].keys())
    skin_counts = list(stats['skin_tone_stats'].values())
    plt.bar(skin_tones, skin_counts)
    plt.title('Phân bố màu da')
    plt.ylabel('Số lượng')
    
    # Vẽ thống kê nhóm tuổi
    plt.subplot(3, 1, 2)
    age_groups = list(stats['age_group_stats'].keys())
    age_counts = list(stats['age_group_stats'].values())
    plt.bar(age_groups, age_counts)
    plt.title('Phân bố nhóm tuổi')
    plt.ylabel('Số lượng')
    
    # Vẽ thống kê cảm xúc
    plt.subplot(3, 1, 3)
    emotions = list(stats['emotion_stats'].keys())
    emotion_counts = list(stats['emotion_stats'].values())
    plt.bar(emotions, emotion_counts)
    plt.title('Phân bố cảm xúc')
    plt.ylabel('Số lượng')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 