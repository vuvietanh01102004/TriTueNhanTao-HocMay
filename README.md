# Link dữ liệu: https://drive.google.com/drive/folders/1d7yRgh_ojzem_B_QqTZNc6TPWho5EenA?usp=drive_link

# Link video youtube: https://youtu.be/C2xsyBsjjIY

# Hướng dẫn cài đặt và chạy chương trình
## 1. Lấy dữ liệu Dataset
- Truy cập vào:  https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
<img width="1919" height="863" alt="image" src="https://github.com/user-attachments/assets/a273efbf-280e-48b8-a54d-5fd98be791b3" />

### Sẽ được 1 file có tên archive.zip:

<img width="329" height="41" alt="image" src="https://github.com/user-attachments/assets/c25fb0ff-97d2-4a43-ac01-16a2822896dc" />

<img width="1263" height="699" alt="image" src="https://github.com/user-attachments/assets/5f4ee2c2-248f-416c-9571-95f55d98e8c0" />

## 2. Huấn luyện trí tuệ nhân tạo
- Truy cập vào google colab: https://colab.research.google.com/
- Tìm Tệp -> Sổ tay mới trong Drive
<img width="1384" height="856" alt="image" src="https://github.com/user-attachments/assets/c4517e2c-7746-4069-9b11-a41106a37740" />
- Chọn Mã để nhập code:

```
from google.colab import files
files.upload()
```
<img width="343" height="82" alt="image" src="https://github.com/user-attachments/assets/d0fa0caa-7815-405b-b3d5-2344b397c93d" />

- ấn vào chọn tệp và chọn file: Garbage classification vừa giải nến từ file archive.zip
- Sau đó tiếp tục nhập code:
```
!unzip -o "Garbage classification.zip"
```
- Tiếp tục nhập code để import các thư viện cần thiết:
```
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
```
- Chuẩn bị dữ liệu và chia tập huấn luyện và kiểm tra:
```
data_dir = "Garbage classification"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

--- Train/Validation/Test split ---
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
seed = 42

assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9

Chia theo lớp (stratified) để các tập có phân bố lớp tương tự nhau
from collections import defaultdict
import random
from torch.utils.data import Subset

targets = getattr(dataset, "targets", None)
if targets is None:
    targets = [label for _, label in dataset.samples]

indices_by_class = defaultdict(list)
for idx, label in enumerate(targets):
    indices_by_class[int(label)].append(idx)

rng = random.Random(seed)
train_indices, val_indices, test_indices = [], [], []
for label, idxs in indices_by_class.items():
    rng.shuffle(idxs)
    n = len(idxs)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    train_indices.extend(idxs[:n_train])
    val_indices.extend(idxs[n_train : n_train + n_val])
    test_indices.extend(idxs[n_train + n_val :])

rng.shuffle(train_indices)
rng.shuffle(val_indices)
rng.shuffle(test_indices)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Classes:", dataset.classes)
print(f"Split sizes -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
```
- Tiếp tục tải và cấu hình mô hình ResNet18:
```
# Tải ResNet18 pretrained (ImageNet) - tương thích nhiều phiên bản torchvision
try:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
except Exception:
    model = models.resnet18(pretrained=True)

Số lớp đầu ra (mặc định 6 lớp rác)
num_classes = 6
try:
    num_classes = len(dataset.classes)
except Exception:
    pass

model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```
- Hàm đánh giá mô hình:
```
def evaluate(model, loader, criterion, device):
    """Tính loss và accuracy cho validation/test."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with ctx():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc
```
- Huấn luyện mô hình:
```
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

import copy
best_val_acc = -1.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(epochs):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum / max(1, train_total)
    train_acc = train_correct / max(1, train_total)

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train loss: {train_loss:.4f} | Train acc: {train_acc*100:.2f}% | "
        f"Val loss: {val_loss:.4f} | Val acc: {val_acc*100:.2f}%"
    )

Load lại mô hình tốt nhất theo Validation trước khi đánh giá Test / lưu model
model.load_state_dict(best_model_wts)
print(f"Best Val acc: {best_val_acc*100:.2f}%")

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc*100:.2f}%")
```
- Kết quả huấn luyện:
<img width="528" height="119" alt="image" src="https://github.com/user-attachments/assets/98223d8a-b832-48c6-94c0-de0ad048a96b" />

- Tải mô hình huấn luyện:
```
torch.save(model.state_dict(), "garbage_model.pth")
from google.colab import files
files.download("garbage_model.pth")
```
- Sau khi chạy xong sẽ có 1 file tên: garbage_model.pth để ta có thể tải về:
<img width="268" height="49" alt="image" src="https://github.com/user-attachments/assets/51b24c1c-48be-4e29-8b2a-c24024842220" />

## 3. Chạy code để xuất hiện giao diện Streamlit
- Tạo 2 file là app.py và model.py là 2 file code chính
<img width="630" height="630" alt="image" src="https://github.com/user-attachments/assets/5cccc329-e6c8-4192-a2ba-fcd60d3a48cf" />
- app.py
```
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import load_model
from pathlib import Path

st.set_page_config(
    page_title="Phân loại rác AI",
    page_icon="♻️",
    layout="wide",
)

# ===== CSS Custom =====
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #e0f7fa, #e8f5e9);
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #2e7d32;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ===== Load model =====
@st.cache_resource
def get_model():
    model_path = Path(__file__).with_name("garbage_model.pth")
    return load_model(str(model_path))

model = get_model()

classes_vi = ["Bìa carton", "Thủy tinh", "Kim loại", "Giấy", "Nhựa", "Rác khác"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===== Header =====
st.markdown('<div class="title">♻️ Hệ thống phân loại rác bằng AI</div>', unsafe_allow_html=True)
st.write("")

# ===== Sidebar =====
with st.sidebar:
    st.header("📘 Hướng dẫn")
    st.write("✔️ Tải ảnh JPG/PNG")
    st.write("✔️ Ảnh rõ vật thể")
    st.write("✔️ AI sẽ dự đoán loại rác")

# ===== Upload =====
uploaded_file = st.file_uploader("📤 Tải ảnh lên", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    # ===== Image =====
    with col1:
        st.subheader("🖼️ Ảnh đầu vào")
        st.image(image, use_container_width=True)

    # ===== Prediction =====
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        conf, predicted = torch.max(probs, dim=0)

    predicted_idx = int(predicted.item())
    predicted_vi = classes_vi[predicted_idx]
    confidence_pct = float(conf.item() * 100)

    with col2:
        st.subheader("🤖 Kết quả dự đoán")

        st.success(f"👉 {predicted_vi}")
        
        # Display main confidence
        st.write(f"**Độ tin cậy:** {confidence_pct:.2f}%")
        st.progress(int(confidence_pct))

        st.markdown("---")
        st.subheader("📊 Xác suất các loại")

        # Sort classes by probability and display both text and progress
        probs_list = []
        for i, label in enumerate(classes_vi):
            prob = float(probs[i].item() * 100)
            probs_list.append({"label": label, "prob": prob})
            
        probs_list.sort(key=lambda x: x["prob"], reverse=True)

        for item in probs_list:
            lbl = item["label"]
            p = item["prob"]
            col_lbl, col_pct = st.columns([3, 1])
            with col_lbl:
                st.write(f"**{lbl}**")
            with col_pct:
                st.write(f"{p:.2f}%")
            
            st.progress(int(p))

else:
    st.markdown("""
    <div style='text-align:center; padding:50px'>
        <h3>📷 Hãy tải ảnh để bắt đầu</h3>
    </div>
    """, unsafe_allow_html=True)
```

- model.py
```
import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 6

def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    return model
```
- Truy cập terminal rồi chạy lần lượt các code:
```
cd garbage_app
..\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
- Sẽ xuất hiện 1 giao diện Streamlit sau:
<img width="1919" height="859" alt="image" src="https://github.com/user-attachments/assets/b848b682-9a5c-48bc-a514-d5655df64d49" />
- Chọn Browse files -> tìm file Garbage classification vừa giải nén sẽ có 6 thư mục:
<img width="1153" height="720" alt="image" src="https://github.com/user-attachments/assets/24d112bb-ff66-4da4-836b-7955ac77f853" />

<img width="1919" height="1013" alt="image" src="https://github.com/user-attachments/assets/b9f17c38-fd70-4905-ba18-de45dcf72a2e" />




