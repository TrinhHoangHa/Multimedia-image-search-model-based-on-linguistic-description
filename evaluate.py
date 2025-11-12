# Tên file: evaluate.py
import streamlit as st
import os
import torch
from PIL import Image
import clip
import pandas as pd
from pathlib import Path

# --- CẤU HÌNH (Giống hệt file app.py) ---
IMAGE_DIR = "D:/DoAn/images" 
MODEL_PATH = "checkpoints/clip_best.pt"

# --- TẢI MODEL VÀ DỮ LIỆU (Tương tự file app.py) ---
@st.cache_resource
def load_model_and_index_images():
    # ... (Sao chép y hệt hàm load_model_and_index_images từ file app.py của bạn) ...
    st.info("Bắt đầu tải model và lập chỉ mục cho kho ảnh...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        st.error(f"Lỗi khi đọc file checkpoint: {e}")
        return None, None, None, None, None
    image_paths = []
    ground_truths = {}
    for root, _, files in os.walk(IMAGE_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, file)
                image_paths.append(path)
                # Lấy tên thư mục cha làm nhãn (ground truth)
                ground_truths[path] = Path(path).parent.name
    
    if not image_paths:
        st.error(f"Không tìm thấy file ảnh nào trong thư mục: {IMAGE_DIR}")
        return None, None, None, None, None

    all_image_features = []
    # ... (Phần xử lý ảnh giữ nguyên như trong app.py) ...
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            preprocessed_img = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(preprocessed_img)
                feat /= feat.norm(dim=-1, keepdim=True)
            all_image_features.append(feat)
        except Exception:
            continue
    image_features_tensor = torch.cat(all_image_features, dim=0)
    st.success(f"Đã tải model và lập chỉ mục thành công {len(image_paths)} ảnh!")
    return model, device, image_features_tensor, image_paths, ground_truths

# --- HÀM TÍNH TOÁN ĐỘ CHÍNH XÁC ---
def calculate_top_k_accuracy(model, device, image_features, image_paths, ground_truths, k_value):
    # Lấy danh sách các nhãn duy nhất từ tên thư mục
    unique_labels = sorted(list(set(ground_truths.values())))
    
    hits = 0
    results_data = []

    progress_bar = st.progress(0, text=f"Đang đánh giá {len(unique_labels)} nhãn...")

    for i, label in enumerate(unique_labels):
        # Tạo câu truy vấn từ nhãn
        query = f"a photo of a {label.replace('_', ' ')}"
        
        # Tìm kiếm
        with torch.no_grad():
            text_input = clip.tokenize([query]).to(device)
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
            _, indices = similarity[0].topk(k_value)
        
        # Lấy K kết quả hàng đầu
        top_k_paths = [image_paths[idx] for idx in indices]
        
        # Kiểm tra xem có "hit" hay không
        is_hit = False
        for path in top_k_paths:
            if ground_truths.get(path) == label:
                is_hit = True
                hits += 1
                break # Nếu đã hit thì không cần kiểm tra nữa
        
        results_data.append({
            "Nhãn (Thư mục)": label,
            "Câu truy vấn": query,
            "Dự đoán đúng?": "✅ Đúng" if is_hit else "❌ Sai",
            "Top K kết quả trả về": [Path(p).name for p in top_k_paths]
        })
        
        progress_bar.progress((i + 1) / len(unique_labels), text=f"Đang đánh giá nhãn: {label}")
    
    progress_bar.empty()
    accuracy = (hits / len(unique_labels)) * 100
    return accuracy, pd.DataFrame(results_data)

# --- GIAO DIỆN WEB ---
st.set_page_config(page_title="Đánh giá mô hình", layout="wide")
st.title("📊 Báo cáo độ chính xác của mô hình tìm kiếm")

# Tải dữ liệu
model, device, image_features, image_paths, ground_truths = load_model_and_index_images()

if model:
    st.header("Thiết lập đánh giá")
    k_value = st.slider(
        "Chọn giá trị K (Top-K Accuracy)",
        min_value=1,
        max_value=10,
        value=5, # Mặc định là Top-5
        help="Mô hình sẽ được coi là 'đoán đúng' nếu kết quả chính xác nằm trong Top-K ảnh trả về."
    )

    if st.button(f"🚀 Bắt đầu tính toán Top-{k_value} Accuracy", type="primary"):
        with st.spinner("Đang thực hiện đánh giá, vui lòng chờ..."):
            accuracy, results_df = calculate_top_k_accuracy(
                model, device, image_features, image_paths, ground_truths, k_value
            )
        
        st.header("Kết quả đánh giá")
        st.metric(label=f"Độ chính xác Top-{k_value}", value=f"{accuracy:.2f} %")
        
        st.info(f"Trong tổng số {len(results_df)} nhãn, mô hình đã dự đoán đúng {int(accuracy/100*len(results_df))} nhãn.")
        
        st.header("Chi tiết từng truy vấn")
        st.dataframe(results_df)