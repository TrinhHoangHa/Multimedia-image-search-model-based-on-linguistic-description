import os
import torch
from PIL import Image
import clip

# Đường dẫn thư mục chứa ảnh (có thể có thư mục con: Toyota, Honda,...)
IMAGE_DIR = "D:/DoAn/images"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("clip_best.pt", device=device)

# --- Duyệt tất cả ảnh trong thư mục con ---
image_paths = []
for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, file))

print(f"🔎 Tìm thấy {len(image_paths)} ảnh trong {IMAGE_DIR}")

# --- Trích xuất đặc trưng ảnh ---
image_features = []
valid_image_paths = []

for path in image_paths:
    try:
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat /= feat.norm(dim=-1, keepdim=True)
        image_features.append(feat)
        valid_image_paths.append(path)
    except Exception as e:
        print(f"⚠️ Lỗi khi đọc ảnh {path}: {e}")

if len(image_features) == 0:
    raise RuntimeError("❌ Không tìm thấy ảnh hợp lệ trong thư mục IMAGE_DIR")

image_features = torch.cat(image_features, dim=0)
image_paths = valid_image_paths

# --- Hàm tìm kiếm ---
def search_images(query, top_k=5):
    with torch.no_grad():
        text = clip.tokenize([query]).to(device)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(top_k)

    # Trả về tuple (đường dẫn ảnh, điểm số)
    results = [(image_paths[i], float(values[j])) for j, i in enumerate(indices)]
    return results
