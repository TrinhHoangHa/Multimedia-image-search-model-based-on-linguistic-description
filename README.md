\<div align="center"\>

\<p align="center"\>  
  \<img src="https://www.google.com/search?q=https://upload.wikimedia.org/wikipedia/vi/thumb/d/db/Dai\_hoc\_Dai\_Nam.png/220px-Dai\_hoc\_Dai\_Nam.png" alt="DaiNam University Logo" width="150"/\>  
\</p\>

# **Mô hình tìm kiếm hình ảnh phương tiện 🚗🔍**

\<h3 align="center"\>Hệ thống tìm kiếm hình ảnh thông minh sử dụng mô hình CLIP và mô tả ngôn ngữ tự nhiên\</h3\>

\<p align="center"\>  
  \<a href="\#-tính-năng"\>Tính năng\</a\> •  
  \<a href="\#-công-nghệ-sử-dụng"\>Công nghệ\</a\> •  
  \<a href="\#-cài-đặt"\>Cài đặt\</a\> •  
  \<a href="\#-sử-dụng"\>Sử dụng\</a\> •  
  \<a href="\#-kết-quả-đánh-giá"\>Kết quả\</a\> •  
  \<a href="\#-cấu-trúc-dự-án"\>Cấu trúc\</a\>  
\</p\>  
\<p align="center"\>  
   
   
   
   
\</p\>  
\</div\>

Dự án này xây dựng một hệ thống tìm kiếm hình ảnh thông minh, cho phép người dùng truy vấn hình ảnh phương tiện (ô tô, xe máy, v.v.) bằng cách sử dụng các câu mô tả bằng ngôn ngữ tự nhiên, thay vì phụ thuộc vào các thẻ (tags) gán thủ công.

Hệ thống sử dụng mô hình **CLIP (Contrastive Language-Image Pre-Training)** của OpenAI, đã được tinh chỉnh (fine-tune) trên một bộ dữ liệu tùy chỉnh gồm 3346 hình ảnh phương tiện với các chú thích chi tiết bằng tiếng Việt (ví dụ: "Xe Audi Q2 màu xám, kiểu SUV").

## **✨ Tính năng**

* **Tìm kiếm ngữ nghĩa:** Hiểu và truy vấn hình ảnh bằng các câu mô tả tiếng Việt tự nhiên (ví dụ: "xe audi màu trắng", "xe bán tải màu đen").  
* **Mô hình tinh chỉnh:** Sử dụng mô hình ViT-B/32 đã được fine-tune trên bộ dữ liệu xe cộ để tăng độ chính xác.  
* **Giao diện tương tác:** Ứng dụng web demo được xây dựng bằng Streamlit, cho phép tìm kiếm và xem kết quả trực quan.  
* **Độ chính xác cao:** Đạt 100% độ chính xác Top-10 trên 7 nhãn hiệu xe được kiểm thử.

## **🚀 Công nghệ sử dụng**

* **Python 3.9+**  
* **PyTorch & CLIP:** Để xây dựng, huấn luyện và trích xuất đặc trưng từ mô hình.  
* **Streamlit:** Để xây dựng giao diện người dùng Web App.  
* **Pandas & Pillow (PIL):** Để xử lý dữ liệu và hình ảnh.  
* **Faiss-cpu:** (Tùy chọn, đã có trong requirements.txt) để tối ưu hóa tìm kiếm vector.

## **💾 Cài đặt**

1. Clone repository này về máy của bạn:  
   git clone \[https://github.com/ten-cua-ban/ten-repo-cua-ban.git\](https://github.com/ten-cua-ban/ten-repo-cua-ban.git)

2. Di chuyển vào thư mục dự án:  
   cd ten-repo-cua-ban

3. (Khuyến nghị) Tạo một môi trường ảo (virtual environment):  
   python \-m venv venv  
   source venv/bin/activate  \# Trên Windows: venv\\Scripts\\activate

4. Cài đặt các gói thư viện cần thiết:  
   pip install \-r requirements.txt

## **🛠️ Sử dụng**

### **1\. Chuẩn bị dữ liệu**

Để hệ thống hoạt động, bạn cần chuẩn bị dữ liệu theo cấu trúc sau:

* **Thư mục images/:** Chứa tất cả hình ảnh của bạn, nên được sắp xếp vào các thư mục con theo nhãn (ví dụ: images/Audi, images/Toyota\_Innova).  
* **Tệp metadata.csv:** Một tệp CSV nằm ở thư mục gốc, chứa 2 cột bắt buộc là filename (đường dẫn tương đối của ảnh, ví dụ: Audi/1.jpg) và caption (câu mô tả tiếng Việt cho ảnh đó).

### **2\. Huấn luyện (Tùy chọn)**

Nếu bạn có bộ dữ liệu metadata.csv của riêng mình và muốn huấn luyện lại hoặc tinh chỉnh mô hình, hãy chạy lệnh:

python train.py \--images\_dir ./images \--metadata metadata.csv \--epochs 5 \--batch\_size 16

Mô hình tốt nhất sẽ được lưu tại checkpoints/clip\_best.pt.

### **3\. Chạy ứng dụng Demo**

Để khởi chạy giao diện web demo (sử dụng tệp clip\_best.pt đã được huấn luyện):

1. Mở terminal và chạy lệnh:  
   streamlit run app.py

2. Mở trình duyệt của bạn và truy cập vào địa chỉ http://localhost:8501.  
3. Chờ thông báo "Đã lập chỉ mục thành công..." và bắt đầu tìm kiếm.

### **4\. Đánh giá mô hình (Tùy chọn)**

Để chạy giao diện web đánh giá độ chính xác Top-K của mô hình (dựa trên tên thư mục làm nhãn):

streamlit run evaluate.py

## **📊 Kết quả đánh giá**

Hệ thống được đánh giá bằng kịch bản evaluate.py trên 7 nhãn hiệu xe (Audi, Hyundai Creta, Mahindra Scorpio, Rolls Royce, Swift, Tata Safari, Toyota Innova).

* Độ chính xác Top-10: 100.00%  
  (Kết quả được coi là "Đúng" nếu ít nhất một trong 10 ảnh trả về thuộc đúng nhãn với câu truy vấn "a photo of a \[nhãn\]")

## **📁 Cấu trúc dự án**

.  
├── app.py              \# Ứng dụng Streamlit chính (Demo tìm kiếm)  
├── train.py            \# Kịch bản huấn luyện (tinh chỉnh) mô hình CLIP  
├── evaluate.py         \# Ứng dụng Streamlit để đánh giá Top-K Accuracy  
├── models.py           \# (Không dùng cho CLIP) Định nghĩa mô hình ResNet/BERT  
├── search.py           \# Kịch bản tìm kiếm cơ bản (không có giao diện)  
├── extract\_features.py \# (Không dùng cho CLIP) Trích xuất đặc trưng ResNet  
├── requirements.txt    \# Các gói thư viện Python cần thiết  
├── checkpoints/        \# Thư mục chứa các model đã huấn luyện  
│   └── clip\_best.pt    \# Model CLIP đã được tinh chỉnh  
├── images/             \# Thư mục chứa bộ dữ liệu ảnh  
│   ├── Audi/  
│   │   └── 1.jpg  
│   └── ...  
└── metadata.csv        \# Tệp CSV ánh xạ tên ảnh và mô tả

## **👨‍💻 Tác giả**

* Trịnh Hoàng Hà  
* Lê Ngọc Hưng

## **🚀 Hướng phát triển**

* **Tối ưu tốc độ:** Tích hợp thư viện faiss (đã có trong requirements.txt) để tăng tốc độ tìm kiếm trên các bộ dữ liệu lớn (hàng triệu ảnh).  
* **Mở rộng dữ liệu:** Huấn luyện thêm mô hình với bộ dữ liệu tiếng Việt đa dạng hơn để tăng khả năng hiểu ngữ nghĩa.  
* **Đa phương thức:** Mở rộng hệ thống để có thể tìm kiếm video hoặc âm thanh dựa trên mô tả.
