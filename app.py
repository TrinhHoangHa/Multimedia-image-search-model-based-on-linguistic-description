# Tên file: app.py
# Sao chép và dán toàn bộ code này vào file app.py của bạn.

import streamlit as st
import os
import torch
from PIL import Image
import clip
import time

# --- CẤU HÌNH ---
IMAGE_DIR = "D:/DoAn/images" 
MODEL_PATH = "checkpoints/clip_best.pt"

# --- LOGIC BACKEND ---

@st.cache_resource(show_spinner="Đang tải model và lập chỉ mục cho kho ảnh...")
def load_model_and_index_images():
    """
    Tải model CLIP và xử lý ảnh.
    Hàm này được cache và KHÔNG chứa bất kỳ lệnh giao diện streamlit nào.
    Thông báo tải sẽ được xử lý bởi show_spinner.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Tải thành công trọng số model từ file checkpoint!")
    except Exception as e:
        raise RuntimeError(f"Lỗi khi tải model: {e}. Hãy chắc chắn file checkpoint tồn tại và hợp lệ.") from e

    image_paths = [os.path.join(root, file) for root, _, files in os.walk(IMAGE_DIR) for file in files if file.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    if not image_paths:
        raise FileNotFoundError(f"Không tìm thấy file ảnh nào trong thư mục: {IMAGE_DIR}")

    all_image_features = []
    valid_paths = []
    
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            preprocessed_img = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(preprocessed_img)
                feat /= feat.norm(dim=-1, keepdim=True)
            all_image_features.append(feat)
            valid_paths.append(path)
        except Exception as e:
            print(f"Bỏ qua ảnh lỗi {path}: {e}")
            
    if not all_image_features:
        raise ValueError("Không thể xử lý bất kỳ ảnh nào. Vui lòng kiểm tra định dạng ảnh.")

    image_features_tensor = torch.cat(all_image_features, dim=0)
    print(f"Đã lập chỉ mục thành công {len(valid_paths)} ảnh!")
    
    return model, device, image_features_tensor, valid_paths

def search_images(query, model, device, image_features, image_paths, top_k=5):
    with torch.no_grad():
        text_input = clip.tokenize([query]).to(device)
        text_features = model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(top_k)
    return [(image_paths[i], float(v)) for v, i in zip(values, indices)]

# --- GIAO DIỆN WEB (FRONTEND) ---

st.set_page_config(page_title="Tìm kiếm hình ảnh xe", page_icon="🚗", layout="wide")

if 'first_load_success' not in st.session_state:
    st.session_state.first_load_success = True

try:
    model, device, image_features, image_paths = load_model_and_index_images()

    if st.session_state.first_load_success:
        st.success(f"Đã lập chỉ mục thành công {len(image_paths)} ảnh! Hệ thống đã sẵn sàng.")
        time.sleep(2)
        st.session_state.first_load_success = False
        st.rerun()

    col1, col2, col3 = st.columns([2,3,2])
    with col2:
        
        
        st.markdown(
            "<h1 style='text-align: center; white-space: nowrap;'>Tìm Kiếm Hình Ảnh Xe Thông Minh</h1>", 
            unsafe_allow_html=True
        )
        

    # Phần Tùy chọn
    with st.expander("⚙️ Tùy chọn tìm kiếm"):
        top_k = st.slider(
            "Số lượng kết quả hiển thị", 
            min_value=1, 
            max_value=20, 
            value=6, 
            step=1
        )

    if 'query' not in st.session_state:
        st.session_state.query = ""

    with st.form(key='search_form'):
        query_input = st.text_input(
            "Mô tả xe bạn muốn tìm kiếm...",
            value=st.session_state.query,
            placeholder="ví dụ: xe SUV màu trắng, xe bán tải màu đen...",
            label_visibility="collapsed"
        )
        submit_button = st.form_submit_button(label='🔍 Tìm kiếm')

    if submit_button and query_input:
        st.session_state.query = query_input

    if st.session_state.query:
        st.write("---") 
        st.subheader(f"Kết quả tìm kiếm cho: '{st.session_state.query}'")
        
        results = search_images(st.session_state.query, model, device, image_features, image_paths, top_k)
        
        if not results:
            st.warning("Rất tiếc, không tìm thấy hình ảnh nào phù hợp với mô tả của bạn.")
        else:
            num_columns = 3 
            cols = st.columns(num_columns)
            for i, (img_path, score) in enumerate(results):
                with cols[i % num_columns]:
                    st.image(
                        img_path,
                        use_container_width=True,
                        caption=f"Độ khớp: {score*100:.2f}%"
                    )

except (RuntimeError, FileNotFoundError, ValueError) as e:
    st.error(f"**Đã xảy ra lỗi nghiêm trọng:**\n\n{e}\n\nVui lòng kiểm tra lại đường dẫn file và cấu hình, sau đó làm mới lại trang.")