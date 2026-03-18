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