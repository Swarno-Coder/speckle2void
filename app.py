import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import json
import os
import time
import pandas as pd
from scipy import ndimage

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Speckle2Void | L-Band SAR Denoising",
    page_icon="🛰️",
    layout="wide"
)

# --- ENHANCED CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
<style>
    /* Main Background */
    .main { background-color: #0e1117; }
    
    /* Title Styling */
    h1 { 
        color: #00e0ff; 
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        text-shadow: 0 0 10px rgba(0, 224, 255, 0.3);
    }
    
    h2, h3 { 
        color: #ffffff; 
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Button Styling */
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        background: linear-gradient(135deg, #00e0ff 0%, #00a8cc 100%);
        color: #0e1117; 
        font-weight: bold;
        border: none;
        padding: 12px 24px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 224, 255, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 224, 255, 0.5);
    }
    
    /* Metric Card Styling */
    .metric-card { 
        background: linear-gradient(145deg, #1a1f2e 0%, #262730 100%);
        padding: 20px; 
        border-radius: 12px; 
        border-left: 4px solid #00e0ff;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin: 10px 0;
    }
    
    .metric-card h4 {
        color: #00e0ff;
        margin-bottom: 15px;
        font-size: 18px;
    }
    
    .metric-card p {
        color: #e0e0e0;
        margin: 8px 0;
    }
    
    /* Metrics Table Styling */
    .metrics-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        background: linear-gradient(145deg, #1a1f2e 0%, #262730 100%);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin: 15px 0;
    }
    
    .metrics-table th {
        background: linear-gradient(135deg, #00e0ff 0%, #00a8cc 100%);
        color: #0e1117;
        padding: 14px 18px;
        text-align: left;
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metrics-table td {
        padding: 12px 18px;
        border-bottom: 1px solid #2d3748;
        color: #e0e0e0;
        font-size: 14px;
    }
    
    .metrics-table tr:last-child td {
        border-bottom: none;
    }
    
    .metrics-table tr:hover td {
        background-color: rgba(0, 224, 255, 0.05);
    }
    
    .metrics-table .metric-name {
        color: #a0aec0;
        font-weight: 500;
    }
    
    .metrics-table .metric-value {
        color: #00e0ff;
        font-weight: 600;
        font-size: 15px;
    }
    
    .metrics-table .metric-good {
        color: #48bb78;
    }
    
    .metrics-table .metric-neutral {
        color: #ecc94b;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .status-success {
        background-color: rgba(72, 187, 120, 0.2);
        color: #48bb78;
        border: 1px solid #48bb78;
    }
    
    .status-processing {
        background-color: rgba(0, 224, 255, 0.2);
        color: #00e0ff;
        border: 1px solid #00e0ff;
    }
    
    /* Image Container */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid #2d3748;
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00e0ff, transparent);
        margin: 30px 0;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(145deg, #1a2332 0%, #1e2738 100%);
        border-left: 4px solid #3182ce;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Streamlit Metrics Override */
    [data-testid="stMetricValue"] {
        color: #00e0ff !important;
        font-size: 24px !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0aec0 !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #3d4f6f;
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #00e0ff;
        background-color: rgba(0, 224, 255, 0.02);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #1a1f2e 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# --- TITLE & SIDEBAR ---
st.title("🛰️ Speckle2Void: Self-Supervised SAR Denoising")
st.markdown("**Defense Surveillance Showcase** | Self-Supervised Blind-Spot Network for Radar Imagery")

with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <p><b>🧠 Model:</b> Blind-Spot U-Net</p>
        <p><b>📦 Format:</b> Quantized INT8 ONNX</p>
        <p><b>⚡ Engine:</b> ONNX Runtime (CPU)</p>
        <p><b>📊 Training:</b> Official SSDD Dataset</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📡 About")
    st.write("This tool removes coherent speckle noise from radar imagery without requiring ground truth, utilizing a **Noise2Void** self-supervised architecture.")
    
    st.markdown("---")
    st.markdown("### 🎯 Supported Bands")
    st.markdown("""
    - **L-Band** (1-2 GHz) ✅
    - **P-Band** (0.3-1 GHz) ✅
    - **X-Band** (8-12 GHz) ✅
    - **C-Band** (4-8 GHz) ✅
    """)
    
    st.markdown("---")
    st.caption("Built with ❤️ for Defense Applications")

# --- 1. MODEL LOADER (Cached) ---
@st.cache_resource
def load_model():
    # Load the Quantized ONNX model
    try:
        session = ort.InferenceSession("speckle2void_quantized.onnx")
        return session
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

session = load_model()

# --- 2. IMAGE PROCESSING FUNCTIONS ---
def preprocess(image):
    """Resizes and normalizes image for the model."""
    img = image.convert('L') # Force Grayscale
    img = img.resize((256, 256))
    img_arr = np.array(img).astype(np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0) # Channel dim
    img_arr = np.expand_dims(img_arr, axis=0) # Batch dim
    return img_arr, img

def postprocess(output_tensor):
    """Converts model output back to displayable image."""
    out = np.squeeze(output_tensor)
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def calculate_psnr(original, denoised):
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = np.mean((original.astype(float) - denoised.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(original, denoised):
    """Calculate Structural Similarity Index (simplified version)."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    original = original.astype(float)
    denoised = denoised.astype(float)
    
    mu_x = np.mean(original)
    mu_y = np.mean(denoised)
    sigma_x = np.std(original)
    sigma_y = np.std(denoised)
    sigma_xy = np.mean((original - mu_x) * (denoised - mu_y))
    
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
    return ssim

def calculate_enl(image):
    """Calculate Equivalent Number of Looks."""
    mean_val = np.mean(image)
    std_val = np.std(image)
    if std_val == 0:
        return float('inf')
    return (mean_val / std_val) ** 2

def calculate_edge_preservation(original, denoised):
    """Calculate edge preservation index using Sobel operator."""
    # Compute edges using Sobel
    orig_edges_x = ndimage.sobel(original.astype(float), axis=0)
    orig_edges_y = ndimage.sobel(original.astype(float), axis=1)
    orig_edges = np.hypot(orig_edges_x, orig_edges_y)
    
    den_edges_x = ndimage.sobel(denoised.astype(float), axis=0)
    den_edges_y = ndimage.sobel(denoised.astype(float), axis=1)
    den_edges = np.hypot(den_edges_x, den_edges_y)
    
    # Correlation coefficient
    orig_flat = orig_edges.flatten()
    den_flat = den_edges.flatten()
    
    if np.std(orig_flat) == 0 or np.std(den_flat) == 0:
        return 1.0
    
    correlation = np.corrcoef(orig_flat, den_flat)[0, 1]
    return max(0, correlation)

# --- 3. MAIN INTERFACE ---
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("### 📤 Upload SAR Image for Denoising")
uploaded_file = st.file_uploader("Upload Raw L-Band SAR Image", type=["jpg", "png", "jpeg", "tif"], label_visibility="collapsed")

if uploaded_file is not None:
    # Read Image
    original_pil = Image.open(uploaded_file)
    input_tensor, resized_pil = preprocess(original_pil)

    col1, col2, col3 = st.columns([1, 0.15, 1])

    with col1:
        st.markdown("#### 📡 Raw Input (Noisy)")
        st.image(resized_pil)
        st.caption("🔴 Speckle noise obscures structural details")

    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        denoise_clicked = st.button("⚡ DENOISE")

    # Initialize session state for results
    if 'denoised_result' not in st.session_state:
        st.session_state.denoised_result = None
        st.session_state.metrics = None

    if denoise_clicked:
        with st.spinner("🔄 Running Blind-Spot Inference..."):
            start_time = time.time()
            
            # --- INFERENCE ---
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            result = session.run([output_name], {input_name: input_tensor})[0]
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            # --- RESULT ---
            denoised_pil = postprocess(result)
            
            # Convert to numpy for metrics
            original_arr = np.array(resized_pil)
            denoised_arr = np.array(denoised_pil)

            # Calculate comprehensive metrics
            noise_diff = np.mean(np.abs(original_arr - denoised_arr))
            noise_reduction_pct = (noise_diff / np.mean(original_arr)) * 100
            
            original_std = np.std(original_arr)
            denoised_std = np.std(denoised_arr)
            snr_improvement = 20 * np.log10(original_std / (denoised_std + 1e-10))
            
            psnr = calculate_psnr(original_arr, denoised_arr)
            ssim = calculate_ssim(original_arr, denoised_arr)
            
            enl_original = calculate_enl(original_arr)
            enl_denoised = calculate_enl(denoised_arr)
            enl_improvement = enl_denoised / (enl_original + 1e-10)
            
            edge_preservation = calculate_edge_preservation(original_arr, denoised_arr)
            
            # Store in session state
            st.session_state.denoised_result = denoised_pil
            st.session_state.metrics = {
                'noise_diff': noise_diff,
                'noise_reduction_pct': noise_reduction_pct,
                'snr_improvement': snr_improvement,
                'psnr': psnr,
                'ssim': ssim,
                'enl_original': enl_original,
                'enl_denoised': enl_denoised,
                'enl_improvement': enl_improvement,
                'edge_preservation': edge_preservation,
                'inference_time': inference_time,
                'original_mean': np.mean(original_arr),
                'denoised_mean': np.mean(denoised_arr),
                'original_std': original_std,
                'denoised_std': denoised_std
            }

    with col3:
        if st.session_state.denoised_result is not None:
            st.markdown("#### ✨ Speckle2Void Output")
            st.image(st.session_state.denoised_result)
            st.markdown('<span class="status-badge status-success">✓ Denoised</span>', unsafe_allow_html=True)
        else:
            st.markdown("#### ⏳ Awaiting Processing")
            st.info("Click **DENOISE** to process the image")

    # --- METRICS TABLE SECTION ---
    if st.session_state.metrics is not None:
        m = st.session_state.metrics
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Comprehensive Denoising Metrics")
        
        # Create three columns for different metric categories
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown("""
            <table class="metrics-table">
                <tr>
                    <th colspan="2">🎯 Quality Metrics</th>
                </tr>
                <tr>
                    <td class="metric-name">PSNR</td>
                    <td class="metric-value metric-good">{:.2f} dB</td>
                </tr>
                <tr>
                    <td class="metric-name">SSIM</td>
                    <td class="metric-value metric-good">{:.4f}</td>
                </tr>
                <tr>
                    <td class="metric-name">SNR Improvement</td>
                    <td class="metric-value">{:.2f} dB</td>
                </tr>
                <tr>
                    <td class="metric-name">Edge Preservation</td>
                    <td class="metric-value">{:.1%}</td>
                </tr>
            </table>
            """.format(m['psnr'], m['ssim'], m['snr_improvement'], m['edge_preservation']), 
            unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown("""
            <table class="metrics-table">
                <tr>
                    <th colspan="2">📉 Noise Analysis</th>
                </tr>
                <tr>
                    <td class="metric-name">Noise Reduction</td>
                    <td class="metric-value metric-good">{:.1f}%</td>
                </tr>
                <tr>
                    <td class="metric-name">Noise Difference</td>
                    <td class="metric-value">{:.2f}</td>
                </tr>
                <tr>
                    <td class="metric-name">Original Std Dev</td>
                    <td class="metric-value metric-neutral">{:.2f}</td>
                </tr>
                <tr>
                    <td class="metric-name">Denoised Std Dev</td>
                    <td class="metric-value metric-good">{:.2f}</td>
                </tr>
            </table>
            """.format(m['noise_reduction_pct'], m['noise_diff'], m['original_std'], m['denoised_std']), 
            unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown("""
            <table class="metrics-table">
                <tr>
                    <th colspan="2">📈 ENL & Performance</th>
                </tr>
                <tr>
                    <td class="metric-name">ENL (Original)</td>
                    <td class="metric-value metric-neutral">{:.2f}</td>
                </tr>
                <tr>
                    <td class="metric-name">ENL (Denoised)</td>
                    <td class="metric-value metric-good">{:.2f}</td>
                </tr>
                <tr>
                    <td class="metric-name">ENL Improvement</td>
                    <td class="metric-value metric-good">{:.1f}x</td>
                </tr>
                <tr>
                    <td class="metric-name">Inference Time</td>
                    <td class="metric-value">{:.1f} ms</td>
                </tr>
            </table>
            """.format(m['enl_original'], m['enl_denoised'], m['enl_improvement'], m['inference_time']), 
            unsafe_allow_html=True)
        
        # Summary Statistics Table
        st.markdown("#### 📋 Full Statistics Summary")
        
        summary_data = {
            'Metric': [
                'Peak Signal-to-Noise Ratio (PSNR)',
                'Structural Similarity Index (SSIM)',
                'Signal-to-Noise Ratio Improvement',
                'Equivalent Number of Looks (Original)',
                'Equivalent Number of Looks (Denoised)',
                'ENL Improvement Factor',
                'Edge Preservation Index',
                'Noise Reduction Percentage',
                'Mean Intensity (Original)',
                'Mean Intensity (Denoised)',
                'Standard Deviation (Original)',
                'Standard Deviation (Denoised)',
                'Inference Time'
            ],
            'Value': [
                f"{m['psnr']:.2f} dB",
                f"{m['ssim']:.4f}",
                f"{m['snr_improvement']:.2f} dB",
                f"{m['enl_original']:.2f}",
                f"{m['enl_denoised']:.2f}",
                f"{m['enl_improvement']:.2f}x",
                f"{m['edge_preservation']:.2%}",
                f"{m['noise_reduction_pct']:.1f}%",
                f"{m['original_mean']:.2f}",
                f"{m['denoised_mean']:.2f}",
                f"{m['original_std']:.2f}",
                f"{m['denoised_std']:.2f}",
                f"{m['inference_time']:.1f} ms"
            ],
            'Status': [
                '✅ Excellent' if m['psnr'] > 30 else '⚠️ Good' if m['psnr'] > 20 else '❌ Low',
                '✅ Excellent' if m['ssim'] > 0.9 else '⚠️ Good' if m['ssim'] > 0.7 else '❌ Low',
                '✅ Improved' if m['snr_improvement'] > 0 else '❌ Degraded',
                '—',
                '✅ Higher' if m['enl_denoised'] > m['enl_original'] else '—',
                '✅ Excellent' if m['enl_improvement'] > 2 else '⚠️ Good',
                '✅ Excellent' if m['edge_preservation'] > 0.7 else '⚠️ Moderate',
                '✅ Good' if m['noise_reduction_pct'] > 10 else '⚠️ Minimal',
                '—',
                '—',
                '—',
                '✅ Reduced' if m['denoised_std'] < m['original_std'] else '—',
                '✅ Fast' if m['inference_time'] < 100 else '⚠️ Moderate'
            ]
        }
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df)

# --- 4. EXAMPLE SHOWCASE ---
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("## 🏆 Top 5 Best Denoising Examples")
st.markdown("These examples demonstrate the highest SNR improvement achieved by our model on the SSDD test dataset.")

# Load overall stats if available
examples_dir = "examples"
stats_path = os.path.join(examples_dir, "overall_stats.json")
if os.path.exists(stats_path):
    with open(stats_path, 'r') as f:
        overall_stats = json.load(f)
    
    st.subheader("📊 Model Performance Summary")
    stat_cols = st.columns(5)
    with stat_cols[0]:
        st.metric("Avg SNR", f"{overall_stats['avg_snr_improvement']:.2f} dB")
    with stat_cols[1]:
        st.metric("Avg PSNR", f"{overall_stats['avg_psnr']:.2f} dB")
    with stat_cols[2]:
        st.metric("Avg SSIM", f"{overall_stats['avg_ssim']:.4f}")
    with stat_cols[3]:
        st.metric("Avg ENL Gain", f"{overall_stats['avg_enl_improvement']:.1f}x")
    with stat_cols[4]:
        st.metric("Edge Preserve", f"{overall_stats['avg_edge_preservation']:.2%}")
    
    st.caption(f"Statistics computed across {overall_stats['total_images']} test images")
    st.write("")

# Load example metadata
if os.path.exists(os.path.join(examples_dir, "metadata.json")):
    with open(os.path.join(examples_dir, "metadata.json"), 'r') as f:
        example_metadata = json.load(f)
    
    # Display examples in a grid
    for example in example_metadata:
        ex_id = example['id']
        snr = example['snr_improvement']
        psnr = example.get('psnr', 'N/A')
        ssim = example.get('ssim', 'N/A')
        enl_orig = example.get('enl_original', 'N/A')
        enl_den = example.get('enl_denoised', 'N/A')
        enl_imp = example.get('enl_improvement', 'N/A')
        edge_pres = example.get('edge_preservation', 'N/A')
        noise_pct = example['noise_reduction_pct']
        
        st.markdown(f"### Example {ex_id} — SNR Improvement: **{snr} dB**")
        
        col_a, col_b, col_c = st.columns([1, 1, 1])
        
        input_path = os.path.join(examples_dir, f"example_{ex_id}_input.png")
        denoised_path = os.path.join(examples_dir, f"example_{ex_id}_denoised.png")
        
        with col_a:
            if os.path.exists(input_path):
                st.image(input_path, caption=f"Noisy Input", use_column_width=True)
        
        with col_b:
            if os.path.exists(denoised_path):
                st.image(denoised_path, caption=f"Denoised Output", use_column_width=True)
        
        with col_c:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📈 Metrics</h4>
                <p><b>SNR:</b> {snr} dB</p>
                <p><b>PSNR:</b> {psnr} dB</p>
                <p><b>SSIM:</b> {ssim}</p>
                <p><b>ENL:</b> {enl_orig} → {enl_den} ({enl_imp}x)</p>
                <p><b>Edge Preservation:</b> {edge_pres}</p>
                <p><b>Noise Reduction:</b> {noise_pct}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.caption(f"Original file: `{example['original_filename']}`")
        st.write("")
else:
    st.info("📁 Example images not found. Run `find_best_examples.py` to generate them.")

# --- 5. FOOTER ---
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #a0aec0; padding: 20px;">
    <p><b>Speckle2Void</b> — Self-Supervised SAR Denoising for Defense Surveillance</p>
    <p>Built with PyTorch • ONNX Runtime • Streamlit</p>
    <p>© 2024 | <a href="https://github.com/Swarno-Coder/speckle2void" style="color: #00e0ff;">GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)



