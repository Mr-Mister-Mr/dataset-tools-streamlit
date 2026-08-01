import os
import io
import re
import hashlib
import time
import shutil  
from collections import Counter, defaultdict
import logging

import numpy as np
from PIL import Image, ImageFilter
import imagehash
import streamlit as st


# ======================================================================
#  1. IMAGE ANALYZER CLASS
# ======================================================================
logger = logging.getLogger(__name__)

class ImageAnalyzer:
    def __init__(self):
        self.ideal_resolutions = [
            (1024, 1024), (832, 1216), (1216, 832),
            (768, 1024), (1024, 768),
            (768, 768), (640, 640),
            (576, 768), (768, 576),
            (512, 768), (768, 512),
            (512, 512), (1344, 768), (768, 1344),
            (1152, 896), (896, 1152)
        ]

    def compute_file_hash(self, filepath, algorithm='sha256'):
        hash_func = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def compute_perceptual_hash(self, img, hash_size=8):
        try:
            img_rgb = img.convert('RGB')
            phash = imagehash.phash(img_rgb, hash_size=hash_size)
            return str(phash)
        except Exception as e:
            logger.error(f"pHash error: {e}")
            return None

    def sharpness_metric(self, gray_np):
        try:
            import cv2
            laplacian = cv2.Laplacian(gray_np, cv2.CV_64F)
            variance = laplacian.var()
            return variance
        except ImportError:
            gray_float = gray_np.astype(np.float32)
            gradient_x = np.abs(np.diff(gray_float, axis=1))
            gradient_y = np.abs(np.diff(gray_float, axis=0))
            min_h = min(gradient_x.shape[0], gradient_y.shape[0])
            min_w = min(gradient_x.shape[1], gradient_y.shape[1])
            gradient_x = gradient_x[:min_h, :min_w]
            gradient_y = gradient_y[:min_h, :min_w]
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            return float(np.std(gradient_magnitude))
        except Exception as e:
            logger.error(f"Sharpness error: {e}")
            return 0.0

    def jpeg_artifact_metric(self, gray_np):
        try:
            import cv2
            smoothed = cv2.medianBlur(gray_np, 3)
            diff = np.abs(gray_np.astype(np.float32) - smoothed.astype(np.float32))
            h, w = diff.shape
            h = h - h % 8
            w = w - w % 8
            if h < 8 or w < 8:
                return 0.0
            diff = diff[:h, :w]
            block_mask = np.zeros((h, w), dtype=np.float32)
            block_mask[::8, :] = 1
            block_mask[:, ::8] = 1
            boundary_mean = np.mean(diff[block_mask == 1])
            inner_mean = np.mean(diff[block_mask == 0])
            if inner_mean == 0:
                return 0.0
            ratio = boundary_mean / (inner_mean + 1e-6)
            norm_ratio = min(ratio / 3.0, 1.0)
            return norm_ratio
        except ImportError:
            img_pil = Image.fromarray(gray_np)
            smoothed_pil = img_pil.filter(ImageFilter.MedianFilter(size=3))
            smoothed_np = np.array(smoothed_pil, dtype=np.float32)
            diff = np.abs(gray_np.astype(np.float32) - smoothed_np)
            h, w = diff.shape
            h = h - h % 8
            w = w - w % 8
            if h < 8 or w < 8:
                return 0.0
            diff = diff[:h, :w]
            block_mask = np.zeros((h, w), dtype=np.float32)
            block_mask[::8, :] = 1
            block_mask[:, ::8] = 1
            boundary_mean = np.mean(diff[block_mask == 1])
            inner_mean = np.mean(diff[block_mask == 0])
            if inner_mean == 0:
                return 0.0
            ratio = boundary_mean / (inner_mean + 1e-6)
            norm_ratio = min(ratio / 3.0, 1.0)
            return norm_ratio
        except Exception as e:
            logger.error(f"JPEG artifact error: {e}")
            return 0.0

    def noise_metric(self, gray_np):
        try:
            import cv2
            blurred = cv2.GaussianBlur(gray_np, (0, 0), sigmaX=2, sigmaY=2)
            residual = np.abs(gray_np.astype(np.float32) - blurred.astype(np.float32))
            grad_x = cv2.Sobel(gray_np, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_np, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            low_texture_mask = grad_mag < np.percentile(grad_mag, 30)
            if np.sum(low_texture_mask) == 0:
                noise_level = 0.0
            else:
                noise_level = np.mean(residual[low_texture_mask])
            norm_noise = min(noise_level / 50.0, 1.0)
            return norm_noise
        except ImportError:
            img_pil = Image.fromarray(gray_np)
            blurred_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=2))
            blurred_np = np.array(blurred_pil, dtype=np.float32)
            residual = np.abs(gray_np.astype(np.float32) - blurred_np)
            grad_x = np.abs(np.diff(gray_np, axis=1))
            grad_y = np.abs(np.diff(gray_np, axis=0))
            grad_x = np.pad(grad_x, ((0,0),(0,1)), mode='edge')
            grad_y = np.pad(grad_y, ((0,1),(0,0)), mode='edge')
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            low_texture_mask = grad_mag < np.percentile(grad_mag, 30)
            if np.sum(low_texture_mask) == 0:
                noise_level = 0.0
            else:
                noise_level = np.mean(residual[low_texture_mask])
            norm_noise = min(noise_level / 50.0, 1.0)
            return norm_noise
        except Exception as e:
            logger.error(f"Noise error: {e}")
            return 0.0

    def detect_watermark(self, img):
        # stub
        return False

    def resolution_score(self, width, height):
        area = width * height
        best_score = 0.0
        for (iw, ih) in self.ideal_resolutions:
            ideal_area = iw * ih
            aspect = width / height if height != 0 else 0
            ideal_aspect = iw / ih
            aspect_diff = abs(aspect - ideal_aspect) / max(aspect, ideal_aspect) if aspect != 0 else 1
            aspect_score = max(0, 1 - aspect_diff)
            if area <= ideal_area:
                area_score = area / ideal_area
            else:
                ratio = area / ideal_area
                area_score = 1.0 / (1.0 + (ratio - 1.0) / 2.0)
                area_score = max(0.0, area_score)
            score = aspect_score * 0.4 + area_score * 0.6
            if score > best_score:
                best_score = score
        return best_score

    def _get_aspect_label(self, ratio):
        targets = {
            '1:1': 1.0, '4:3': 4/3, '3:4': 3/4,
            '16:9': 16/9, '9:16': 9/16, '2:3': 2/3,
            '3:2': 3/2, '21:9': 21/9, '9:21': 9/21,
        }
        best = min(targets.items(), key=lambda item: abs(ratio - item[1]))
        return best[0]

    def analyze_image(self, filepath):
        if not os.path.isfile(filepath):
            return None
        file_hash = self.compute_file_hash(filepath)
        try:
            img = Image.open(filepath)
        except Exception as e:
            logger.error(f"Failed to open {filepath}: {e}")
            return None
        width, height = img.size
        aspect = width / height if height != 0 else 0
        aspect_label = self._get_aspect_label(aspect)
        multiple_32 = (width % 32 == 0 and height % 32 == 0)
        multiple_64 = (width % 64 == 0 and height % 64 == 0)
        resolution_score = self.resolution_score(width, height)
        perceptual_hash = self.compute_perceptual_hash(img)
        gray_img = img.convert('L')
        gray_np = np.array(gray_img, dtype=np.uint8)
        sharpness = self.sharpness_metric(gray_np)
        jpeg_artifacts = self.jpeg_artifact_metric(gray_np)
        noise_level = self.noise_metric(gray_np)
        has_watermark = self.detect_watermark(img)
        metrics = {
            'file_hash': file_hash,
            'perceptual_hash': perceptual_hash,
            'width': width, 'height': height,
            'aspect_ratio': aspect_label,
            'multiple_32': multiple_32, 'multiple_64': multiple_64,
            'resolution_score': resolution_score,
            'sharpness': sharpness,
            'jpeg_artifacts': jpeg_artifacts,
            'noise_level': noise_level,
            'has_watermark': has_watermark
        }
        metrics['overall_quality'] = self.overall_quality_score(metrics)
        return metrics

    def overall_quality_score(self, metrics):
        sharp_norm = min(metrics['sharpness'] / 30.0, 1.0)
        no_artifacts = 1.0 - metrics['jpeg_artifacts']
        no_noise = 1.0 - metrics['noise_level']
        mult_bonus = 0.04 if (metrics['multiple_32'] or metrics['multiple_64']) else 0.0
        score = (0.15 * sharp_norm + 0.08 * no_artifacts + 0.08 * no_noise +
                 0.65 * metrics['resolution_score'] + mult_bonus) * 100
        return max(0, min(100, score))



# ======================================================================
#  2. STREAMLIT CONFIG & CSS
# ======================================================================
st.set_page_config(page_title="Dataset Caption Editor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .block-container { padding-top: 3.2rem; padding-bottom: 0rem; }
        div[data-testid="stForm"] { border: 1px solid #464b5d; padding: 10px; border-radius: 8px; }
        .stTextArea textarea { font-size: 14px !important; }
        .stImage img { max-height: 60vh; object-fit: contain; }
        div[data-testid="stDialog"] .stImage img { max-height: none; object-fit: contain; }

        /* Global tiny toggle button */
        section[data-testid="stSidebar"] button:first-of-type {
            font-size: 1.5rem !important;
            padding: 0rem 0.3rem !important;
            min-height: 1.5rem !important;
            line-height: 1 !important;
            border: none !important;
            background: transparent !important;
        }

        /* Compact separators */
        section[data-testid="stSidebar"] hr {
            margin: 0.4rem 0 !important;
        }
        section[data-testid="stSidebar"] .stTextInput {
            margin-top: -0.5rem !important;
        }
        section[data-testid="stSidebar"] h3 {
            margin-bottom: 0.2rem !important;
        }
    </style>
""", unsafe_allow_html=True)


# ======================================================================
#  3. SESSION STATE INITIALISATION
# ======================================================================
if "dataset_dir" not in st.session_state:
    st.session_state.dataset_dir = ""
if "selected_img" not in st.session_state:
    st.session_state.selected_img = None
if "current_page" not in st.session_state:
    st.session_state.current_page = 0
if "search_phrase" not in st.session_state:
    st.session_state.search_phrase = ""
if "analytics_search" not in st.session_state:
    st.session_state.analytics_search = ""
if "zoom_image_path" not in st.session_state:
    st.session_state.zoom_image_path = None
if "ignored_duplicate_groups" not in st.session_state:
    st.session_state.ignored_duplicate_groups = set()
if "preview_quality_image" not in st.session_state:
    st.session_state.preview_quality_image = None
if "delete_confirm" not in st.session_state:
    st.session_state.delete_confirm = None
if "quality_page" not in st.session_state:
    st.session_state.quality_page = 0
if "global_comma_counter" not in st.session_state:
    st.session_state.global_comma_counter = Counter()
if "tag_library" not in st.session_state:
    st.session_state.tag_library = []
if "bias_prompt" not in st.session_state:
    st.session_state.bias_prompt = ""

ITEMS_PER_PAGE = 50
WORDS_PER_PAGE = 50

STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do',
    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having',
    'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it',
    'its', 'itself', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only',
    'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such',
    'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this',
    'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where',
    'which', 'while', 'who', 'whom', 'why', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves'
}


# ======================================================================
#  4. CACHED HELPERS + FAVORITES UTILITIES
# ======================================================================
import json

APP_DIR = os.path.dirname(os.path.abspath(__file__))
RECENT_PATHS_FILE = os.path.join(APP_DIR, "recent_paths.json")
FAVORITES_FILE = os.path.join(APP_DIR, "favorites.json")

def load_recent_paths():
    if os.path.exists(RECENT_PATHS_FILE):
        try:
            with open(RECENT_PATHS_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_recent_path(path):
    paths = load_recent_paths()
    if path in paths:
        paths.remove(path)
    paths.insert(0, path)
    paths = paths[:10]
    with open(RECENT_PATHS_FILE, "w") as f:
        json.dump(paths, f)

def load_favorites():
    if os.path.exists(FAVORITES_FILE):
        try:
            with open(FAVORITES_FILE, "r") as f:
                data = json.load(f)
            # Normalise: convert old string entries to dicts
            normalised = []
            for item in data:
                if isinstance(item, str):
                    normalised.append({"path": item, "name": os.path.basename(item)})
                elif isinstance(item, dict) and "path" in item:
                    # ensure name exists
                    if "name" not in item:
                        item["name"] = os.path.basename(item["path"])
                    normalised.append(item)
            # deduplicate by path
            seen = set()
            unique = []
            for d in normalised:
                if d["path"] not in seen:
                    seen.add(d["path"])
                    unique.append(d)
            if len(unique) != len(data):
                save_favorites(unique)
            return unique
        except:
            return []
    return []

def save_favorites(favs):
    with open(FAVORITES_FILE, "w") as f:
        json.dump(favs, f, indent=2)

def add_favorite(path, name=""):
    favs = load_favorites()
    # check if path already present
    if any(d["path"] == path for d in favs):
        return False   # already exists
    display = name.strip() if name.strip() else os.path.basename(path)
    favs.append({"path": path, "name": display})
    save_favorites(favs)
    return True

def remove_favorite(path):
    favs = load_favorites()
    new_favs = [d for d in favs if d["path"] != path]
    if len(new_favs) != len(favs):
        save_favorites(new_favs)
        return True
    return False

@st.cache_data(show_spinner=False)
def load_image_pil(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

@st.cache_data(show_spinner="Analyzing natural language...")
def parse_dataset_cached(folder_path):
    if not folder_path or not os.path.exists(folder_path):
        return [], Counter(), Counter()
    all_images = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    items = []
    phrase_counter = Counter()
    comma_counter = Counter()
    for img in all_images:
        base_name = os.path.splitext(img)[0]
        txt_file = os.path.join(folder_path, f"{base_name}.txt")
        caption_text = ""
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                caption_text = f.read().strip()
            sentences = re.split(r'(?<=[.!?])\s+', caption_text.lower())
            for sentence in sentences:
                raw_words = re.findall(r'\b\w+\b', sentence)
                filtered_words = [w for w in raw_words if w not in STOP_WORDS and not w.isdigit()]
                if not filtered_words:
                    continue
                phrase_counter.update(filtered_words)
                for i in range(len(filtered_words) - 1):
                    phrase_counter.update([f"{filtered_words[i]} {filtered_words[i+1]}"])
                for i in range(len(filtered_words) - 2):
                    phrase_counter.update([f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]}"])
                for i in range(len(filtered_words) - 3):
                    phrase_counter.update([f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]} {filtered_words[i+3]}"])
            comma_segments = [seg.strip() for seg in caption_text.lower().split(',') if seg.strip()]
            MIN_COMMA_WORDS = 4
            for seg in comma_segments:
                words = seg.split()
                if len(words) >= MIN_COMMA_WORDS:
                    comma_counter.update([seg])
        items.append({"img": img, "txt_path": txt_file, "caption": caption_text, "len": len(caption_text)})
    return items, phrase_counter, comma_counter

@st.cache_data(show_spinner="Computing image quality metrics...")
def compute_all_metrics(folder_path):
    analyzer = ImageAnalyzer()
    metrics_map = {}
    phash_map = defaultdict(list)
    png_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
    for fname in png_files:
        full_path = os.path.join(folder_path, fname)
        met = analyzer.analyze_image(full_path)
        if met is not None:
            metrics_map[fname] = met
            if met['perceptual_hash']:
                phash_map[met['perceptual_hash']].append(fname)
    duplicate_groups = {h: imgs for h, imgs in phash_map.items() if len(imgs) > 1}
    return metrics_map, duplicate_groups



# ======================================================================
#  5. SIDEBAR (with compact mode, favorites, sorting, tag library)
# ======================================================================
with st.sidebar:
    # ---- Initialisation ----
    if "sidebar_compact" not in st.session_state:
        st.session_state.sidebar_compact = False
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = None

    tab_options = [
        "🖥️ Soho Workspace & Browser",
        "📊 Dataset Word & Phrase Analytics",
        "📷 Image Quality Analytics",
        "📋 Dataset Statistics",
        "📝 Caption Tools",
        "🧠 AI Assistance",
        "💾 Captioner",
        "✂️ Image Cropper",
        "🔄 Image Converter",
        "📐 Smart Resize & Crop",
        "📝 Batch Rename",
        "🏷️ EXIF Viewer & Stripper",
        "ℹ️ Training Run Tracker",
        "🚀 Fine‑Tuning Launcher",
        "✨ Dataset Comparator",
        "📜 Prompt Generator",
        "📰 Report Generator",
        "🎨 PNG Info Viewer",
        "✍️ Notes",
        "📖 User Guide"
    ]


    # ========== FULL SIDEBAR MODE ==========
    if not st.session_state.sidebar_compact:
        # Header row with tiny toggle button
        col_tog, col_head = st.columns([0.08, 0.92])
        with col_tog:
            if st.button("◀", key="sidebar_toggle_full"):
                st.session_state.sidebar_compact = True
                st.rerun()
        with col_head:
            st.markdown("<h2 style='margin-top: 0; padding-top: 0;'>🧱 Navigation & Tools</h2>", unsafe_allow_html=True)

        # ---- Collapsible navigation (radio inside expander) ----
        with st.expander("Choose window", expanded=False):
            current_tab = st.radio(
                "Choose window:",
                tab_options,
                key="main_tab_radio"
            )

        st.markdown("---")
        st.subheader("⚙️ Settings")
        dataset_path = st.text_input("Path to dataset folder:", value=st.session_state.dataset_dir)

        # ---- Load dataset ----
        folder_valid = dataset_path and os.path.exists(dataset_path)
        if folder_valid:
            st.session_state.dataset_dir = dataset_path
            if "raw_items_cache" in st.session_state and st.session_state.get("raw_items_cache_folder") == dataset_path:
                raw_items = st.session_state.raw_items_cache
                global_phrase_counter = st.session_state.get("global_phrase_counter_cache", Counter())
                global_comma_counter = st.session_state.get("global_comma_counter_cache", Counter())
            else:
                raw_items, global_phrase_counter, global_comma_counter = parse_dataset_cached(dataset_path)
                st.session_state.raw_items_cache = raw_items
                st.session_state.raw_items_cache_folder = dataset_path
                st.session_state.global_phrase_counter_cache = global_phrase_counter
                st.session_state.global_comma_counter_cache = global_comma_counter
            save_recent_path(dataset_path)
        else:
            st.session_state.dataset_dir = dataset_path
            raw_items, global_phrase_counter, global_comma_counter = [], Counter(), Counter()
            if dataset_path:
                st.warning("Please enter a valid local folder path.")


        # ---- Favorites ----
        st.markdown("---")
        st.subheader("⭐ Favorites")
        favs = load_favorites()
        if favs:
            for d in favs:
                label = f"⭐ {d['name']}"
                if st.button(label, key=f"fav_btn_{d['path']}"):
                    st.session_state.dataset_dir = d["path"]
                    st.rerun()
        else:
            st.caption("No favorites yet.")

        # Add with optional nickname
        with st.expander("➕ Add current folder to favorites"):
            nickname = st.text_input("Nickname:", key="fav_nickname")
            if st.button("⭐ Add to favorites", key="add_fav_with_nick"):
                if folder_valid:
                    ok = add_favorite(dataset_path, nickname)
                    if ok:
                        st.success("Folder added to favorites!")
                        st.rerun()
                    else:
                        st.info("Already in favorites.")
                else:
                    st.warning("Enter a valid folder path first.")

        # Remove current folder
        if st.button("🗑️ Remove current folder from favorites", key="remove_fav_btn"):
            if dataset_path:
                if remove_favorite(dataset_path):
                    st.success("Folder removed from favorites.")
                    st.rerun()
                else:
                    st.info("Not in favorites.")
            else:
                st.warning("No folder selected.")


        # ---- Sorting & orientation filter ----
        if folder_valid and raw_items:
            col_label, col_select = st.columns([0.35, 0.65])
            with col_label:
                st.markdown(
                    '<p style="margin: 0; padding: 0; line-height: 2.4rem; font-weight: bold; font-size: 1.3rem;">🔀 Sort </p>',
                    unsafe_allow_html=True,
                )
            with col_select:
                sort_by = st.selectbox(
                    " ",
                    [
                        "File name",
                        "Caption length (ascending)",
                        "Caption length (descending)",
                        "Image quality (ascending)",
                        "Image quality (descending)"
                    ],
                    label_visibility="collapsed",
                )

            st.subheader("📐 Orientation")
            show_square = st.checkbox("Square (1:1)", value=True)
            show_portrait = st.checkbox("Portrait (taller than wide)", value=True)
            show_landscape = st.checkbox("Landscape (wider than tall)", value=True)

            if st.session_state.search_phrase:
                st.markdown(f"🔍 Filtered by: `'{st.session_state.search_phrase}'`")
                if st.button("❌ Remove filter", use_container_width=True):
                    st.session_state.search_phrase = ""
                    st.rerun()

            need_quality = ("Image quality" in sort_by) or (not show_square or not show_portrait or not show_landscape)
            if need_quality:
                with st.spinner("Loading quality metrics..."):
                    qual_map, _ = compute_all_metrics(dataset_path)
            else:
                qual_map = None

            @st.cache_data(show_spinner=False)
            def get_filtered_with_quality(items, s_by, phrase, qual_map, square, portrait, landscape):
                res = items.copy()
                if phrase:
                    res = [i for i in res if phrase in i["caption"].lower()]
                if qual_map:
                    filtered_res = []
                    for i in res:
                        if i["img"] in qual_map:
                            w = qual_map[i["img"]]["width"]
                            h = qual_map[i["img"]]["height"]
                            if w == h and not square:
                                continue
                            if h > w and not portrait:
                                continue
                            if w > h and not landscape:
                                continue
                        filtered_res.append(i)
                    res = filtered_res
                if s_by == "File name":
                    res.sort(key=lambda x: x["img"])
                elif s_by == "Caption length (ascending)":
                    res.sort(key=lambda x: x["len"])
                elif s_by == "Caption length (descending)":
                    res.sort(key=lambda x: x["len"], reverse=True)
                elif s_by == "Image quality (ascending)" and qual_map:
                    def quality_key(item):
                        return qual_map.get(item["img"], {}).get("overall_quality", -1)
                    res.sort(key=quality_key)
                elif s_by == "Image quality (descending)" and qual_map:
                    res.sort(key=lambda x: qual_map.get(x["img"], {}).get("overall_quality", -1), reverse=True)
                return res

            dataset_items = get_filtered_with_quality(
                raw_items, sort_by,
                st.session_state.search_phrase,
                qual_map,
                show_square, show_portrait, show_landscape
            )
            total_items = len(dataset_items)
            total_pages = (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
            start_idx = st.session_state.current_page * ITEMS_PER_PAGE
            end_idx = min(start_idx + ITEMS_PER_PAGE, total_items)
            page_items = dataset_items[start_idx:end_idx]
        else:
            page = 1
            page_items = []


        # ---- Batch Tag Library ----
        with st.expander("🏷️ Batch Tag Library", expanded=False):
            new_tag = st.text_input("Add a tag (comma or newline for multiple):", key="new_lib_tag")
            if st.button("➕ Add to library", key="add_lib_btn"):
                if new_tag:
                    tags = [t.strip() for t in new_tag.replace("\n", ",").split(",") if t.strip()]
                    for t in tags:
                        if t not in st.session_state.tag_library:
                            st.session_state.tag_library.append(t)
                    st.rerun()

            if not st.session_state.tag_library:
                st.info("No tags yet. Add some above.")

            selected_tags = []
            for i, tag in enumerate(st.session_state.tag_library):
                col_chk, col_txt, col_del = st.columns([0.1, 0.7, 0.2])
                with col_chk:
                    checked = st.checkbox("", key=f"lib_chk_{i}")
                with col_txt:
                    st.write(tag)
                with col_del:
                    if st.button("❌", key=f"lib_del_{i}"):
                        st.session_state.tag_library.pop(i)
                        st.rerun()
                if checked:
                    selected_tags.append(tag)

            action = st.radio("Action:", ["Add tags", "Remove tags", "Replace tag"],
                              horizontal=True, key="lib_action")
            target_tag = None
            if action == "Replace tag":
                target_tag = st.text_input("Tag to replace:", key="lib_replace_target")

            if st.button("🚀 Apply batch action", use_container_width=True, key="apply_lib_btn"):
                if folder_valid and raw_items:
                    modified_count = 0
                    for item in raw_items:
                        caption = item["caption"]
                        parts = [p.strip() for p in caption.split(",") if p.strip()]
                        new_cap = caption
                        if action == "Add tags":
                            for t in selected_tags:
                                if t not in parts:
                                    parts.append(t)
                            new_cap = ", ".join(parts)
                        elif action == "Remove tags":
                            parts = [p for p in parts if p not in selected_tags]
                            new_cap = ", ".join(parts)
                        elif action == "Replace tag" and target_tag:
                            replacement = selected_tags[0] if selected_tags else target_tag
                            parts = [replacement if p == target_tag else p for p in parts]
                            new_cap = ", ".join(parts)
                        if new_cap != caption:
                            item["caption"] = new_cap
                            with open(item["txt_path"], "w", encoding="utf-8") as f:
                                f.write(new_cap)
                            modified_count += 1
                    st.cache_data.clear()
                    st.success(f"Updated {modified_count} captions!")
                    st.rerun()
                else:
                    st.warning("Load a dataset first.")

        # ⭐ DATASET BACKUP (chunked ZIP creation)
        if folder_valid:
            st.markdown("---")
            if "backup_cursor" not in st.session_state:
                st.session_state.backup_cursor = None
            if "backup_total" not in st.session_state:
                st.session_state.backup_total = 0
            if "backup_zip_path" not in st.session_state:
                st.session_state.backup_zip_path = None

            # Only show the button when not already in progress
            if st.session_state.backup_cursor is None:
                if st.button("📦 Backup Dataset (.zip)", use_container_width=True):
                    # Gather all files in the dataset folder (recursively)
                    all_files = []
                    for root, dirs, files in os.walk(dataset_path):
                        for file in files:
                            all_files.append(os.path.join(root, file))
                    if not all_files:
                        st.warning("No files to backup.")
                    else:
                        st.session_state.backup_files = all_files
                        st.session_state.backup_cursor = 0
                        st.session_state.backup_total = len(all_files)
                        # Create a temporary zip file
                        import tempfile, zipfile
                        tmp_dir = tempfile.mkdtemp()
                        zip_name = f"dataset_backup_{time.strftime('%Y%m%d_%H%M%S')}.zip"
                        tmp_zip = os.path.join(tmp_dir, zip_name)
                        st.session_state.backup_zip_path = tmp_zip
                        st.session_state.backup_temp_dir = tmp_dir
                        st.rerun()

            # Process a chunk of files
            if st.session_state.backup_cursor is not None:
                files = st.session_state.backup_files
                total = len(files)
                cursor = st.session_state.backup_cursor
                chunk_size = 50
                end = min(cursor + chunk_size, total)
                chunk = files[cursor:end]

                # Progress bar
                progress_text = f"Backing up {cursor+1}–{end} of {total} files..."
                progress_bar = st.progress(cursor / total if total else 0, text=progress_text)

                # Append to zip (open in append mode)
                import zipfile
                with zipfile.ZipFile(st.session_state.backup_zip_path, 'a', zipfile.ZIP_DEFLATED) as zf:
                    for fpath in chunk:
                        arcname = os.path.relpath(fpath, dataset_path)
                        try:
                            zf.write(fpath, arcname)
                        except Exception as e:
                            pass  # skip files that can't be read

                # Advance cursor
                st.session_state.backup_cursor = end

                if end >= total:
                    # Finished – move the zip to the dataset folder
                    final_zip = os.path.join(
                        dataset_path,
                        os.path.basename(st.session_state.backup_zip_path)
                    )
                    try:
                        shutil.move(st.session_state.backup_zip_path, final_zip)
                        st.success(f"Backup saved: `{final_zip}`")
                    except Exception as e:
                        st.error(f"Failed to move backup: {e}")
                    # Clean up state and temp dir
                    if os.path.exists(st.session_state.backup_temp_dir):
                        shutil.rmtree(st.session_state.backup_temp_dir, ignore_errors=True)
                    st.session_state.backup_files = []
                    st.session_state.backup_cursor = None
                    st.session_state.backup_total = 0
                    st.session_state.backup_zip_path = None
                    st.session_state.backup_temp_dir = None
                else:
                    st.rerun()

    # ========== COMPACT SIDEBAR MODE ==========
    else:
        # Expand button (tiny, same size as icons)
        if st.button("▶", key="sidebar_toggle_compact"):
            st.session_state.sidebar_compact = False
            st.rerun()

        st.markdown(
            """
            <style>
                section[data-testid="stSidebar"] {
                    width: 4.5rem !important;
                    min-width: 4.5rem !important;
                    max-width: 4.5rem !important;
                }
                section[data-testid="stSidebar"] .block-container {
                    padding: 0.3rem !important;
                }
                section[data-testid="stSidebar"] button {
                    width: 100% !important;
                    padding: 0.5rem 0.2rem !important;
                    margin-bottom: 0.2rem !important;
                    font-size: 1.6rem !important;
                    border: none !important;
                    background: transparent !important;
                    display: flex;
                    justify-content: center;
                }
                section[data-testid="stSidebar"] button:hover {
                    background: rgba(255,255,255,0.08) !important;
                    border-radius: 8px;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        tab_icons = ["🖥️", "📊", "📷", "📋", "📝", "🧠", "💾", "✂️", "🔄", "📐", "📝", "🏷️","ℹ️", "🚀", "✨", "📜", "📰", "🎨", "✍️", "📖"]
        for i, icon in enumerate(tab_icons):
            if st.button(icon, key=f"icon_tab_{i}"):
                st.session_state.active_tab = tab_options[i]
                st.rerun()

        current_tab = st.session_state.active_tab if st.session_state.active_tab else tab_options[0]
        dataset_path = st.session_state.get("dataset_dir", "")

        # Minimal dataset loading for compact mode (so other tabs don't crash)
        folder_valid = dataset_path and os.path.exists(dataset_path)
        if folder_valid:
            if "raw_items_cache" in st.session_state and st.session_state.get("raw_items_cache_folder") == dataset_path:
                raw_items = st.session_state.raw_items_cache
            else:
                raw_items, _, _ = parse_dataset_cached(dataset_path)
                st.session_state.raw_items_cache = raw_items
                st.session_state.raw_items_cache_folder = dataset_path
        else:
            raw_items = []

        # Provide default page_items for the Soho Workspace
        if folder_valid and raw_items:
            page = st.session_state.get("current_page", 0) + 1
            total_pages = max(1, (len(raw_items) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
            page = max(1, min(page, total_pages))
            start_idx = (page - 1) * ITEMS_PER_PAGE
            end_idx = min(start_idx + ITEMS_PER_PAGE, len(raw_items))
            page_items = raw_items[start_idx:end_idx]
        else:
            page = 1
            page_items = []



# ======================================================================
#  TAB 1: SOHO WORKSPACE & BROWSER
# ======================================================================

if current_tab == "🖥️ Soho Workspace & Browser":
    if not folder_valid or not raw_items:
        st.info("📁 Please provide a valid dataset folder path in the sidebar.")
    else:
        # ---- Search box (full width, outside the card) ----
        search_query = st.text_input(
            "🔍 Search captions",
            value=st.session_state.search_phrase,
            key="soho_search",
            placeholder="e.g., woman, outdoor, blue dress"
        )
        if search_query != st.session_state.search_phrase:
            st.session_state.search_phrase = search_query
            st.rerun()

# ---- SINGLE BORDERED CARD for both center and right panels ----

        with st.container(border=True):
            main_canvas, right_browser = st.columns([3, 1])

# ======== RIGHT PANEL WITH PAGINATION (larger, cleaner) ========

            with right_browser:
                # Scrollable container for thumbnails
                with st.container(height=550):
                    for item in page_items:
                        img_path = os.path.join(st.session_state.dataset_dir, item["img"])
                        is_active = (st.session_state.selected_img and
                                     st.session_state.selected_img["img"] == item["img"])
                        border_color = "#38BDF8" if is_active else "#334155"
                        with st.container():
                            col_thumb, col_btn = st.columns([1, 3])
                            with col_thumb:
                                img_pil = load_image_pil(img_path)
                                if img_pil:
                                    st.image(img_pil, width=65)
                                else:
                                    st.caption("Error")
                            with col_btn:
                                btn_label = f"📁 {item['img']}\n({item['len']} chars)"
                                if st.button(btn_label, key=f"btn_{item['img']}", use_container_width=True):
                                    st.session_state.selected_img = item
                                    st.session_state.zoom_image_path = None
                                    st.rerun()
                            st.markdown(
                                f"<div style='border-bottom: 1px solid {border_color}; margin-bottom: 6px;'></div>",
                                unsafe_allow_html=True,
                            )

                # ---- Spacer (keep your existing spacer here) ----
                st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)

                # ---- Pagination & image count (centered) ----
                total_pages = max(1, (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)

                # Row 1: "Page X of Y" (centered)
                st.markdown(
                    f"<p style='font-size:1.65rem; font-weight:bold; text-align:center; margin:0 0 0.4rem 0;'>"
                    f"Page {st.session_state.current_page + 1} of {total_pages}</p>",
                    unsafe_allow_html=True,
                )

                # Row 2: previous button | page input | next button (unchanged)
                col_prev, col_input, col_next = st.columns([0.7, 1.5, 0.7])
                with col_prev:
                    with st.container():
                        if st.button("◀", key="prev_page_btn",
                                     disabled=st.session_state.current_page <= 0,
                                     use_container_width=True):
                            st.session_state.current_page -= 1
                            st.rerun()
                with col_input:
                    new_page = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=total_pages,
                        value=st.session_state.current_page + 1,
                        key="soho_page_input",
                        label_visibility="collapsed",
                    )
                    if new_page != st.session_state.current_page + 1:
                        st.session_state.current_page = new_page - 1
                        st.rerun()
                with col_next:
                    with st.container():
                        if st.button("▶", key="next_page_btn",
                                     disabled=st.session_state.current_page >= total_pages - 1,
                                     use_container_width=True):
                            st.session_state.current_page += 1
                            st.rerun()

                # Row 3: total image count (centered)
                st.markdown(
                    f"<p style='font-size:1rem; color:#888; text-align:center; margin-top:0.5rem;'>"
                    f"📷 {total_items} images</p>",
                    unsafe_allow_html=True,
                )

# Center panel: image + editor + export + suggestions

            with main_canvas:
                if st.session_state.selected_img is None and page_items:
                    st.session_state.selected_img = page_items[0]

                if st.session_state.selected_img:
                    current = st.session_state.selected_img
                    img_full_path = os.path.join(st.session_state.dataset_dir, current["img"])

# Image

                    img_pil = load_image_pil(img_full_path)
                    if img_pil:
                        st.image(img_pil, use_container_width=True)
                    else:
                        st.error(f"Could not load image: {current['img']}")
                    st.markdown(f"**Editing:** `{current['img']}` | **Characters:** {len(current['caption'])}")

# Caption form

                    with st.form(key="edit_form", clear_on_submit=False):
                        new_caption = st.text_area(
                            "Edit Caption Text:", value=current["caption"],
                            height=140, label_visibility="collapsed",
                        )
                        submit_col, next_col, prev_col, zoom_col = st.columns(4)
                        with submit_col:
                            save_btn = st.form_submit_button("💾 Save", use_container_width=True)
                        with next_col:
                            save_next_btn = st.form_submit_button("⏩ Save & Next", use_container_width=True)
                        with prev_col:
                            save_prev_btn = st.form_submit_button("⏪ Save & Prev", use_container_width=True)
                        with zoom_col:
                            zoom_clicked = st.form_submit_button("🔍 Zoom", use_container_width=True)

                        if save_btn or save_next_btn or save_prev_btn:
                            with open(current["txt_path"], "w", encoding="utf-8") as f:
                                f.write(new_caption.strip())
                            parse_dataset_cached.clear()
                            updated_items, _, _ = parse_dataset_cached(st.session_state.dataset_dir)
                            updated_item = next((i for i in updated_items if i["img"] == current["img"]), None)
                            if updated_item:
                                st.session_state.selected_img = updated_item
                            else:
                                current["caption"] = new_caption.strip()
                                current["len"] = len(new_caption.strip())
                                st.session_state.selected_img = current

                            if save_next_btn or save_prev_btn:
                                current_page_items = page_items
                                current_idx = None
                                for idx, item in enumerate(current_page_items):
                                    if item["img"] == current["img"]:
                                        current_idx = idx
                                        break
                                if current_idx is not None:
                                    if save_next_btn and current_idx < len(current_page_items) - 1:
                                        st.session_state.selected_img = current_page_items[current_idx + 1]
                                    elif save_prev_btn and current_idx > 0:
                                        st.session_state.selected_img = current_page_items[current_idx - 1]
                            st.rerun()

                        if zoom_clicked:
                            st.session_state.zoom_image_path = img_full_path
                            st.rerun()

# ---- 📦 Export ----

                    with st.expander("📦 Export Filtered Subset", expanded=False):
                        st.caption("Download the currently displayed images (after all filters) as a ZIP file.")
                        if st.button("📥 Download ZIP", key="export_zip_btn"):
                            import zipfile, tempfile
                            with tempfile.TemporaryDirectory() as tmpdir:
                                zip_path = os.path.join(tmpdir, "filtered_dataset.zip")
                                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                                    for item in dataset_items:
                                        img_full = os.path.join(st.session_state.dataset_dir, item["img"])
                                        zf.write(img_full, arcname=item["img"])
                                        if os.path.exists(item["txt_path"]):
                                            zf.write(item["txt_path"], arcname=os.path.basename(item["txt_path"]))
                                with open(zip_path, "rb") as f:
                                    st.download_button(
                                        "💾 Download ZIP", f,
                                        file_name="filtered_dataset.zip",
                                        mime="application/zip",
                                    )

# ---- 💡 Tag Suggestions ----

                    current_caption = st.session_state.selected_img.get("caption", "")
                    current_tags = set(tag.strip().lower() for tag in current_caption.split(",") if tag.strip())
                    if current_tags:
                        from collections import Counter
                        tag_counter = Counter()
                        for item in raw_items:
                            if item["img"] == st.session_state.selected_img["img"]:
                                continue
                            other_tags = set(
                                tag.strip().lower() for tag in item["caption"].split(",") if tag.strip()
                            )
                            if current_tags.intersection(other_tags):
                                for t in other_tags:
                                    tag_counter[t] += 1
                        for existing_tag in current_tags:
                            if existing_tag in tag_counter:
                                del tag_counter[existing_tag]
                        tag_counter = Counter(
                            {tag: count for tag, count in tag_counter.items() if len(tag) >= 2}
                        )
                        suggestions = tag_counter.most_common(10)
                        if suggestions:
                            with st.expander("💡 Tag Suggestions", expanded=False):
                                st.caption("Click any tag to add it to the caption.")
                                cols = st.columns(5)
                                for i, (tag, count) in enumerate(suggestions):
                                    with cols[i % 5]:
                                        if st.button(
                                            f"➕ {tag} ({count})",
                                            key=f"suggest_{tag}_{current['img']}",
                                        ):
                                            new_cap = current_caption.strip()
                                            if new_cap and not new_caption.endswith(","):
                                                new_cap += ", "
                                            new_cap += f"{tag}"
                                            st.session_state.selected_img["caption"] = new_cap
                                            with open(st.session_state.selected_img["txt_path"], "w", encoding="utf-8") as f:
                                                f.write(new_cap)
                                            st.cache_data.clear()
                                            st.rerun()

# Zoom dialog (outside columns)

    if st.session_state.zoom_image_path:
        @st.dialog("🔍 Zoomed Preview", width="large")
        def show_zoom():
            st.image(st.session_state.zoom_image_path, use_container_width=True)
            if st.button("Close"):
                st.session_state.zoom_image_path = None
                st.rerun()
        show_zoom()



# ======================================================================        
#  TAB 2: WORD & PHRASE ANALYTICS
# ======================================================================

elif current_tab == "📊 Dataset Word & Phrase Analytics":
    if not folder_valid or not raw_items:
        st.info("📁 Please load a valid dataset first.")
    else:
        st.header("📊 Browse & search all words and phrases")
        st.markdown("Click any phrase to filter images containing it.")
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            phrase_length = st.selectbox(
                "Show combinations of:",
                ["All lengths", "1 word only", "2 words (e.g. 'sandy terrain')",
                 "3 words (e.g. 'long dark brown')", "4 or more words",
                 "Comma-separated chunks"]
            )
        with filter_col2:
            search_query = st.text_input("🔍 Search vocabulary:",
                                         value=st.session_state.analytics_search).lower().strip()
            st.session_state.analytics_search = search_query

        # Choose data source
        if phrase_length == "Comma-separated chunks":
            all_phrases = st.session_state.get("global_comma_counter", Counter()).most_common()
        else:
            all_phrases = global_phrase_counter.most_common()
            if phrase_length == "1 word only":
                all_phrases = [(p, c) for p, c in all_phrases if len(p.split()) == 1]
            elif phrase_length == "2 words (e.g. 'sandy terrain')":
                all_phrases = [(p, c) for p, c in all_phrases if len(p.split()) == 2]
            elif phrase_length == "3 words (e.g. 'long dark brown')":
                all_phrases = [(p, c) for p, c in all_phrases if len(p.split()) == 3]
            elif phrase_length == "4 or more words":
                all_phrases = [(p, c) for p, c in all_phrases if len(p.split()) >= 4]

        if search_query:
            all_phrases = [(p, c) for p, c in all_phrases if search_query in p]

        if not all_phrases:
            st.info("No words or phrases match the current filters.")
        else:
            total_words = len(all_phrases)
            total_word_pages = (total_words + WORDS_PER_PAGE - 1) // WORDS_PER_PAGE
            page_col1, page_col2 = st.columns(2)
            with page_col1:
                w_page = st.number_input(f"Word page (1-{total_word_pages})", min_value=1,
                                         max_value=max(1, total_word_pages), key="word_page_input")
            with page_col2:
                st.markdown(
                    f"<p style='padding-top:25px; color:#888;'>Showing {(w_page-1)*WORDS_PER_PAGE + 1} - "
                    f"{min(w_page*WORDS_PER_PAGE, total_words)} of {total_words} phrases</p>",
                    unsafe_allow_html=True
                )
            w_start_idx = (w_page - 1) * WORDS_PER_PAGE
            w_end_idx = min(w_start_idx + WORDS_PER_PAGE, total_words)
            current_page_phrases = all_phrases[w_start_idx:w_end_idx]
            with st.container(height=550, border=True):
                grid_cols = st.columns(3)
                for index, (phrase, count) in enumerate(current_page_phrases):
                    col_target = grid_cols[index % 3]
                    with col_target:
                        if st.button(f"🔎 {phrase} ({count}×)", key=f"phrase_{phrase}_{index}",
                                     use_container_width=True):
                            st.session_state.search_phrase = phrase
                            st.session_state.current_page = 0
                            st.rerun()



# ======================================================================
# TAB 3: Image Quality Analytics (session‑state metrics)
# ======================================================================

elif current_tab == "📷 Image Quality Analytics":
    if not folder_valid or not raw_items:
        st.info("📁 Please load a valid dataset first to see image quality metrics.")
    else:
        st.header("📷 Image Quality & Duplicate Detection")
        st.markdown(
            "Quality scores, sharpness, noise, JPEG artifacts, watermark flag, resolution/aspect, and duplicate groups."
        )

        # ----- session state for this tab -----
        if "ignored_duplicate_groups" not in st.session_state:
            st.session_state.ignored_duplicate_groups = set()
        if "quality_page" not in st.session_state:
            st.session_state.quality_page = 0
        if "preview_quality_image" not in st.session_state:
            st.session_state.preview_quality_image = None
        if "delete_confirm" not in st.session_state:
            st.session_state.delete_confirm = None

        ITEMS_PER_PAGE_QUALITY = 50

        # ----- Load metrics (session‑state cache) -----
        if "metrics_map" not in st.session_state or "duplicate_groups" not in st.session_state:
            with st.spinner("Computing image quality metrics (this runs only once)..."):
                st.session_state.metrics_map, st.session_state.duplicate_groups = compute_all_metrics(dataset_path)

        metrics_map = st.session_state.metrics_map
        duplicate_groups = st.session_state.duplicate_groups

        if not metrics_map:
            st.warning("No metrics computed (possibly corrupt images).")
        else:
            # ----- summary -----
            total_images = len(metrics_map)
            avg_quality = np.mean([m['overall_quality'] for m in metrics_map.values()])
            col1, col2, col3 = st.columns(3)
            col1.metric("Total images", total_images)
            col2.metric("Average quality", f"{avg_quality:.1f}")
            col3.metric("Duplicate groups", len(duplicate_groups))

            # ----- build dataframe (clean) -----
            import pandas as pd
            rows = []
            for fname, met in metrics_map.items():
                rows.append({
                    "Image": fname,
                    "Quality": met['overall_quality'],
                    "Sharpness": met['sharpness'],
                    "Noise": met['noise_level'],
                    "JPEG Artifacts": met['jpeg_artifacts'],
                    "Watermark": "Yes" if met['has_watermark'] else "No",
                    "Resolution": f"{met['width']}x{met['height']}",
                    "Aspect": met['aspect_ratio'],
                    "32-mult": met['multiple_32'],
                    "64-mult": met['multiple_64'],
                })
            df = pd.DataFrame(rows)

            # filter by quality
            min_qual = st.slider("Minimum overall quality", 0, 100, 0, 5)
            df_filtered = df[df["Quality"] >= min_qual].reset_index(drop=True)

            # pagination calculations
            total_quality_items = len(df_filtered)
            total_quality_pages = max(1, (total_quality_items + ITEMS_PER_PAGE_QUALITY - 1) // ITEMS_PER_PAGE_QUALITY)

            # ---- Unified row: page input + info + re‑scan button ----
            col_page, col_info, col_rescan = st.columns([1.2, 2, 1])
            with col_page:
                new_page = st.number_input(
                    f"Page (1–{total_quality_pages})",
                    min_value=1,
                    max_value=total_quality_pages,
                    value=st.session_state.quality_page + 1,
                    label_visibility="visible",
                )
                st.session_state.quality_page = new_page - 1
            with col_info:
                st.markdown(
                    f"<p style='padding-top:1.8rem; font-size:1.4rem;'>"
                    f"Showing {st.session_state.quality_page*ITEMS_PER_PAGE_QUALITY + 1} – "
                    f"{min((st.session_state.quality_page+1)*ITEMS_PER_PAGE_QUALITY, total_quality_items)} "
                    f"of {total_quality_items}</p>",
                    unsafe_allow_html=True,
                )
            with col_rescan:
                if st.button("🔄 Re‑scan metrics", use_container_width=True):
                    if "metrics_map" in st.session_state:
                        del st.session_state.metrics_map
                    if "duplicate_groups" in st.session_state:
                        del st.session_state.duplicate_groups
                    st.cache_data.clear()
                    st.rerun()

            start_q = st.session_state.quality_page * ITEMS_PER_PAGE_QUALITY
            end_q = min(start_q + ITEMS_PER_PAGE_QUALITY, total_quality_items)
            page_df = df_filtered.iloc[start_q:end_q]

            # ---- clean table ----
            st.dataframe(page_df, use_container_width=True, hide_index=True)

            # ---- preview button (selectbox + button) ----
            page_names = page_df["Image"].tolist()
            if page_names:
                st.markdown("**🔍 Preview an image from this page:**")
                preview_choice = st.selectbox("Select image:", page_names, key="preview_select")
                if st.button("🔍 Preview selected"):
                    st.session_state.preview_quality_image = preview_choice
                    st.rerun()

            # ----- duplicate groups -----
            st.subheader("🔁 Duplicate / Near-Duplicate Groups")

            active_dups = {h: imgs for h, imgs in duplicate_groups.items()
                           if h not in st.session_state.ignored_duplicate_groups}

            if active_dups:
                st.write(f"Found **{len(active_dups)}** groups of similar images (perceptual hash).")
                for h, imgs in active_dups.items():
                    with st.expander(f"Group hash {h[:8]}... ({len(imgs)} images)"):
                        thumb_cols = st.columns(len(imgs))
                        for i, img_name in enumerate(imgs):
                            img_path = os.path.join(dataset_path, img_name)
                            img_pil = load_image_pil(img_path)
                            with thumb_cols[i]:
                                if img_pil:
                                    st.image(img_pil, width=240)
                                    met = metrics_map.get(img_name)
                                    if met:
                                        res_str = f"{met['width']}×{met['height']}"
                                        qual_str = f"Q:{met['overall_quality']:.0f}"
                                        st.caption(f"{img_name}\n{res_str} · {qual_str}")
                                    else:
                                        st.caption(img_name)
                                else:
                                    st.caption(f"Error: {img_name}")

                        col_ignore, col_delete = st.columns(2)
                        with col_ignore:
                            if st.button("🚫 Ignore this group", key=f"ignore_{h}"):
                                st.session_state.ignored_duplicate_groups.add(h)
                                st.session_state.preview_quality_image = None
                                st.rerun()
                        with col_delete:
                            img_to_delete = st.radio("Select image to delete:", imgs, key=f"radio_{h}")
                            if st.button("🗑️ Delete selected", key=f"del_{h}"):
                                st.session_state.delete_confirm = (h, img_to_delete)
                                st.rerun()

                        if st.session_state.delete_confirm and st.session_state.delete_confirm[0] == h:
                            h_del, img_del = st.session_state.delete_confirm
                            st.warning(f"Are you sure you want to delete **{img_del}** and its .txt file?")
                            cc1, cc2 = st.columns(2)
                            with cc1:
                                if st.button("✅ Yes, delete", key=f"confirm_del_{h}"):
                                    png_path = os.path.join(dataset_path, img_del)
                                    txt_path = os.path.splitext(png_path)[0] + ".txt"
                                    try:
                                        if os.path.exists(png_path):
                                            os.remove(png_path)
                                        if os.path.exists(txt_path):
                                            os.remove(txt_path)
                                    except Exception as e:
                                        st.error(f"Delete error: {e}")
                                    # Update session state instead of clearing cache
                                    if img_del in st.session_state.metrics_map:
                                        del st.session_state.metrics_map[img_del]
                                    new_dup = {}
                                    for hh, imgs in st.session_state.duplicate_groups.items():
                                        imgs = [img for img in imgs if img != img_del]
                                        if len(imgs) > 1:
                                            new_dup[hh] = imgs
                                    st.session_state.duplicate_groups = new_dup
                                    parse_dataset_cached.clear()
                                    st.session_state.preview_quality_image = None
                                    st.session_state.delete_confirm = None
                                    st.rerun()
                            with cc2:
                                if st.button("❌ Cancel", key=f"cancel_del_{h}"):
                                    st.session_state.delete_confirm = None
                                    st.rerun()
            else:
                st.success("No duplicate or near‑duplicate images detected (or all groups ignored).")

            if st.session_state.ignored_duplicate_groups:
                if st.button("🔄 Reset ignored groups"):
                    st.session_state.ignored_duplicate_groups = set()
                    st.session_state.preview_quality_image = None
                    st.rerun()

            # ----- Smart Duplicate Cleanup -----
            st.markdown("---")
            st.subheader("🧠 Smart Duplicate Cleanup")
            if duplicate_groups and metrics_map:
                for h, imgs in duplicate_groups.items():
                    if h in st.session_state.ignored_duplicate_groups:
                        continue
                    if len(imgs) < 2:
                        continue
                    met_a = metrics_map.get(imgs[0])
                    met_b = metrics_map.get(imgs[1])
                    if not met_a or not met_b:
                        continue

                    def score(m):
                        return (m['sharpness'] * 0.3 +
                                (100 - m['jpeg_artifacts'] * 100) * 0.3 +
                                (100 - m['noise_level'] * 100) * 0.2 +
                                m['overall_quality'] * 0.2)

                    score_a = score(met_a)
                    score_b = score(met_b)
                    keep = "A" if score_a >= score_b else "B"
                    img_keep = imgs[0] if keep == "A" else imgs[1]
                    img_delete = imgs[1] if keep == "A" else imgs[0]

                    with st.expander(f"🔁 {imgs[0]} vs {imgs[1]} — Recommended to keep: **{keep}**"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(os.path.join(dataset_path, imgs[0]), width=200)
                            st.caption(imgs[0])
                            st.write(f"Quality: {met_a['overall_quality']:.1f}, Sharpness: {met_a['sharpness']:.2f}")
                        with col2:
                            st.image(os.path.join(dataset_path, imgs[1]), width=200)
                            st.caption(imgs[1])
                            st.write(f"Quality: {met_b['overall_quality']:.1f}, Sharpness: {met_b['sharpness']:.2f}")

                        if st.button(f"🗑️ Delete {img_delete}", key=f"smart_del_{h}"):
                            png_path = os.path.join(dataset_path, img_delete)
                            txt_path = os.path.splitext(png_path)[0] + ".txt"
                            try:
                                if os.path.exists(png_path):
                                    os.remove(png_path)
                                if os.path.exists(txt_path):
                                    os.remove(txt_path)
                            except Exception as e:
                                st.error(f"Error deleting: {e}")
                            if img_delete in st.session_state.metrics_map:
                                del st.session_state.metrics_map[img_delete]
                            new_dup = {}
                            for hh, imgs in st.session_state.duplicate_groups.items():
                                imgs = [img for img in imgs if img != img_delete]
                                if len(imgs) > 1:
                                    new_dup[hh] = imgs
                            st.session_state.duplicate_groups = new_dup
                            parse_dataset_cached.clear()
                            st.session_state.preview_quality_image = None
                            st.rerun()
            else:
                st.info("No duplicate groups available for smart cleanup.")

    # ----- Zoom dialog for quality preview (always at the end) -----
    if st.session_state.preview_quality_image:
        img_path = os.path.join(dataset_path, st.session_state.preview_quality_image)
        if os.path.exists(img_path):
            @st.dialog("🔍 Preview: " + st.session_state.preview_quality_image, width="large")
            def show_quality_preview():
                st.image(img_path, use_container_width=True)
                if st.button("Close"):
                    st.session_state.preview_quality_image = None
                    st.rerun()
            show_quality_preview()
        else:
            st.session_state.preview_quality_image = None



# ======================================================================
# TAB 4: Dataset Statistics (with Bias Report Prompt)
# ======================================================================

elif current_tab == "📋 Dataset Statistics":
    if not folder_valid or not raw_items:
        st.info("📁 Please load a valid dataset first to view statistics.")
    else:
        st.header("📊 Dataset Statistics & Concept Charts")
        st.markdown(
            "Select a concept to see its distribution. Scroll down for caption length, word cloud, "
            "quality‑concept crossover, dataset completeness, and an AI bias analysis prompt."
        )

        # ----- CONCEPT KEYWORDS (same as before) -----
        CONCEPT_KEYWORDS = {
            "Gender": {
                "man":        [" man ", " male ", " boy ", " guy ", " gentleman "],
                "woman":      [" woman ", " female ", " girl ", " lady ", " gal "],
                "non-binary": ["non-binary", "nonbinary", "genderfluid", "genderqueer"],
                "unspecified":[]
            },
            "Camera Model / Type": {
                "smartphone":    ["smartphone", "iphone", "android", "mobile phone", "google pixel", "samsung galaxy"],
                "DSLR":          ["dslr", "digital slr", "canon eos", "nikon d", "sony alpha", "pentax k"],
                "mirrorless":    ["mirrorless", "fujifilm x", "sony a7", "canon r", "nikon z", "panasonic lumix s"],
                "compact":       ["compact camera", "point and shoot", "pocket camera", "canon powershot", "sony rx100"],
                "medium format": ["medium format", "hasselblad", "fujifilm gfx", "phase one"],
                "film":          ["film camera", "analog", "35mm", "medium format film", "instant film", "polaroid"],
                "action cam":    ["gopro", "action camera", "dji osmo", "insta360"],
                "drone":         ["drone", "aerial", "dji mavic", "dji air", "phantom"],
                "other":         []
            },
            "Photographic Style": {
                "natural":         ["natural", "realistic", "unprocessed"],
                "cinematic":       ["cinematic", "film look", "movie"],
                "vintage":         ["vintage", "retro", "old"],
                "high-contrast":   ["high contrast", "dramatic contrast"],
                "black & white":   ["black and white", "b&w", "monochrome"],
                "soft":            ["soft", "dreamy", "pastel"],
                "candid":          ["candid", "unposed", "spontaneous", "snapshot"],
            },
            "Lighting Type": {
                "natural light":  ["natural light", "daylight", "sunlight", "outdoor light"],
                "studio light":   ["studio light", "artificial light", "flash", "continuous light"],
                "golden hour":    ["golden hour", "sunset light", "warm light"],
                "harsh light":    ["harsh light", "direct sunlight", "hard shadows"],
                "soft light":     ["soft light", "diffuse light", "softbox"]
            },
            "Background Environment": {
                "outdoor": ["outdoor", "outside", "garden", "park", "street"],
                "indoor":  ["indoor", "inside", "room", "studio", "apartment"],
                "urban":   ["urban", "city", "street", "alley"],
                "nature":  ["nature", "forest", "woods", "mountain", "beach"],
                "plain":   ["plain background", "solid background", "white background", "black background"]
            },
            "Pose & Expression": {
                "standing":     ["standing", "stands", "stand"],
                "sitting":      ["sitting", "sits", "sit"],
                "walking":      ["walking", "walks", "walk"],
                "smiling":      ["smiling", "smile", "grin"],
                "serious":      ["serious", "neutral", "expressionless"],
                "looking away": ["looking away", "averted gaze", "looking off"]
            },
            "Clothing & Colors": {
                "dress":        [" dress ", "dresses"],
                "shirt":        [" shirt ", "blouse", "t-shirt", "top"],
                "jeans":        [" jeans ", "denim"],
                "skirt":        [" skirt "],
                "jacket":       [" jacket ", "coat", "blazer"],
                "sweater":      [" sweater ", "jumper", "cardigan"],
                "shorts":       [" shorts "],
                "swimwear":     [" swimsuit", "bikini", "trunks"],
                "underwear":    [" underwear ", "bra ", "panties", "boxers"],
                "black":        [" black ", " ebony"],
                "white":        [" white ", " ivory"],
                "red":          [" red ", "crimson", "ruby"],
                "blue":         [" blue ", "navy", "teal"],
                "green":        [" green ", "olive", "emerald"],
                "pink":         [" pink "],
                "yellow":       [" yellow "],
                "purple":       [" purple ", "violet", "lavender"],
                "brown":        [" brown "],
                "grey":         [" grey ", " gray "]
            },
            "Eye & Hair Colour": {
                "brown eyes":  ["brown eyes", "dark eyes"],
                "blue eyes":   ["blue eyes", "azure eyes"],
                "green eyes":  ["green eyes", "emerald eyes"],
                "black hair":  ["black hair", "dark hair"],
                "blonde hair": ["blonde hair", "blond hair", "fair hair"],
                "brown hair":  ["brown hair", "brunette", "chestnut hair"],
                "red hair":    ["red hair", "ginger", "auburn hair"],
                "grey hair":   ["grey hair", "gray hair", "silver hair"]
            }
        }

        # ----- CACHED CONCEPT COUNTER -----
        @st.cache_data(show_spinner="Counting concepts...")
        def count_concepts(folder_path, concept_dict):
            from collections import Counter
            items, _, _ = parse_dataset_cached(folder_path)
            results = {}
            for concept, categories in concept_dict.items():
                counter = Counter()
                for item in items:
                    caption = item["caption"].lower()
                    matched = False
                    for cat, keywords in categories.items():
                        if any(kw in caption for kw in keywords):
                            counter[cat] += 1
                            matched = True
                    if concept == "Gender" and not matched:
                        counter["unspecified"] += 1
                if concept == "Camera Model / Type":
                    total_camera_mentions = sum(counter.values())
                    counter["other"] = len(items) - total_camera_mentions
                results[concept] = dict(counter)
            return results

        # ----- CACHED PER‑IMAGE CONCEPT FLAGS -----
        @st.cache_data(show_spinner="Building per‑image concept flags...")
        def get_per_image_concepts(folder_path, concept_dict):
            items, _, _ = parse_dataset_cached(folder_path)
            data = []
            for item in items:
                caption = item["caption"].lower()
                row = {"image": item["img"]}
                for concept, categories in concept_dict.items():
                    matched_cat = None
                    for cat, keywords in categories.items():
                        if any(kw in caption for kw in keywords):
                            matched_cat = cat
                            break
                    if concept == "Gender":
                        if matched_cat is None:
                            matched_cat = "unspecified"
                    elif concept == "Camera Model / Type":
                        if matched_cat is None:
                            matched_cat = "other"
                    row[concept] = matched_cat if matched_cat else "not found"
                data.append(row)
            import pandas as pd
            return pd.DataFrame(data)

        # ----- COMPUTE ALL COUNTS AND PER‑IMAGE DATA -----
        counts = count_concepts(st.session_state.dataset_dir, CONCEPT_KEYWORDS)
        per_image_df = get_per_image_concepts(st.session_state.dataset_dir, CONCEPT_KEYWORDS)

        # ----- CONCEPT SELECTOR + PIE / BAR CHARTS (original feature) -----
        concept_choice = st.selectbox("Choose a concept to view:", list(CONCEPT_KEYWORDS.keys()))
        if concept_choice:
            data = counts[concept_choice]
            if data:
                import pandas as pd
                import plotly.express as px

                df = pd.DataFrame({
                    "Category": list(data.keys()),
                    "Count": list(data.values())
                }).sort_values("Count", ascending=False)

                fig_pie = px.pie(df, names="Category", values="Count",
                                 title=f"{concept_choice} Distribution",
                                 hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)

                fig_bar = px.bar(df, x="Category", y="Count", text="Count",
                                 title=f"{concept_choice} Counts")
                fig_bar.update_traces(textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True)

                with st.expander("📋 View raw counts"):
                    st.dataframe(df, hide_index=True)
            else:
                st.info(f"No data found for {concept_choice}.")

        # ----- CAPTION LENGTH HISTOGRAM -----
        st.markdown("---")
        st.markdown("### 📏 Caption Length Distribution")
        all_lengths = [item["len"] for item in raw_items]
        fig_len = px.histogram(x=all_lengths, nbins=30,
                               labels={"x": "Caption length (characters)", "y": "Number of images"},
                               title="How long are your captions?")
        st.plotly_chart(fig_len, use_container_width=True)

        # ----- WORD CLOUD (or bar chart fallback) -----
        st.markdown("---")
        st.markdown("### ☁️ Word Cloud (most frequent meaningful words)")
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt

            all_text = " ".join(item["caption"] for item in raw_items).lower()
            all_text = re.sub(r'[^a-z\s]', '', all_text)
            words = [w for w in all_text.split() if w not in STOP_WORDS and len(w) > 2]
            text_for_cloud = " ".join(words)

            wc = WordCloud(width=800, height=400, background_color='white',
                           colormap='viridis', max_words=100).generate(text_for_cloud)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        except ImportError:
            st.info("Install `wordcloud` to see a word cloud (pip install wordcloud). Showing top 20 words instead.")
            all_text = " ".join(item["caption"] for item in raw_items).lower()
            all_text = re.sub(r'[^a-z\s]', '', all_text)
            words = [w for w in all_text.split() if w not in STOP_WORDS and len(w) > 2]
            from collections import Counter
            top_words = Counter(words).most_common(20)
            df_words = pd.DataFrame(top_words, columns=["Word", "Count"])
            fig_words = px.bar(df_words, x="Word", y="Count", title="Top 20 Most Common Words")
            st.plotly_chart(fig_words, use_container_width=True)

        # ----- QUALITY‑CONCEPT CROSSOVER (unchanged) -----
        st.markdown("---")
        st.markdown("### 🔬 Quality‑Concept Crossover")
        qual_map, _ = compute_all_metrics(dataset_path)
        crossover_df = per_image_df.copy()
        crossover_df["sharpness"] = crossover_df["image"].map(lambda f: qual_map.get(f, {}).get("sharpness", None))
        crossover_df["noise_level"] = crossover_df["image"].map(lambda f: qual_map.get(f, {}).get("noise_level", None))
        crossover_df["jpeg_artifacts"] = crossover_df["image"].map(lambda f: qual_map.get(f, {}).get("jpeg_artifacts", None))
        crossover_df["overall_quality"] = crossover_df["image"].map(lambda f: qual_map.get(f, {}).get("overall_quality", None))
        crossover_df.dropna(subset=["sharpness", "noise_level", "jpeg_artifacts", "overall_quality"], inplace=True)

        col_qual, col_conc = st.columns(2)
        with col_qual:
            quality_metric = st.selectbox(
                "Choose a quality metric:",
                ["sharpness", "noise_level", "jpeg_artifacts", "overall_quality"],
                key="crossover_metric"
            )
        with col_conc:
            concept_columns = [c for c in per_image_df.columns if c != "image"]
            concept_for_cross = st.selectbox("Choose a concept:", concept_columns, key="crossover_concept")

        fig_box = px.box(crossover_df, x=concept_for_cross, y=quality_metric,
                         color=concept_for_cross,
                         title=f"{quality_metric} distribution across {concept_for_cross}",
                         points="all")
        st.plotly_chart(fig_box, use_container_width=True)

        if quality_metric == "overall_quality":
            fig_scatter = px.scatter(crossover_df, x=concept_for_cross, y="overall_quality",
                                     color=concept_for_cross,
                                     title=f"Scatter of overall quality by {concept_for_cross}",
                                     opacity=0.6)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ----- DATASET COMPLETENESS (unchanged) -----
        st.markdown("---")
        st.markdown("### ✅ Dataset Completeness")
        total_images = len(per_image_df)
        completeness = {}
        for concept, categories in CONCEPT_KEYWORDS.items():
            if concept == "Gender":
                count_present = per_image_df[per_image_df[concept] != "unspecified"].shape[0]
            elif concept == "Camera Model / Type":
                count_present = per_image_df[per_image_df[concept] != "other"].shape[0]
            else:
                if concept in per_image_df.columns:
                    count_present = per_image_df[per_image_df[concept] != "not found"].shape[0]
                else:
                    count_present = 0
            completeness[concept] = count_present / total_images if total_images > 0 else 0

        for concept, pct in completeness.items():
            st.write(f"**{concept}** – present in {pct*100:.1f}% of images")
            st.progress(pct)

        short_captions = sum(1 for item in raw_items if item["len"] < 100)
        st.write(f"**Short captions (< 100 chars):** {short_captions} ({short_captions/total_images*100:.1f}%)")
        missing_txt = 0
        for item in raw_items:
            txt_path = os.path.splitext(item["txt_path"])[0] + ".txt" if "txt_path" in item else ""
            if txt_path and not os.path.exists(txt_path):
                missing_txt += 1
        st.write(f"**Images missing .txt file:** {missing_txt} ({missing_txt/total_images*100:.1f}%)")

        # ----- NEW: AI Bias Analysis Prompt Generator -----
        st.markdown("---")
        st.subheader("📋 AI Bias Analysis Prompt")
        if st.button("Generate Bias Analysis Prompt"):
            total_images = len(raw_items)
            empty_captions = sum(1 for item in raw_items if not item["caption"].strip())
            all_tags = []
            for item in raw_items:
                tags = [t.strip().lower() for t in item["caption"].split(",") if t.strip()]
                all_tags.extend(tags)
            tag_counts = Counter(all_tags)
            top_tags = tag_counts.most_common(50)
            rare_tags = [(tag, count) for tag, count in tag_counts.items() if count == 1]

            contradictions_found = []
            for item in raw_items:
                cap = item["caption"].lower()
                for (a, b) in [
                    ("day", "night"), ("daytime", "night"), ("solo", "multiple girls"),
                    ("solo", "multiple boys"), ("indoors", "outdoors"), ("outside", "inside"),
                    ("1girl", "1boy"), ("monochrome", "colorful")
                ]:
                    if re.search(r'\b' + re.escape(a) + r'\b', cap) and re.search(r'\b' + re.escape(b) + r'\b', cap):
                        contradictions_found.append(f"{item['img']}: '{a}' and '{b}'")
            contradictions_str = "\n".join(contradictions_found[:20]) if contradictions_found else "None"

            samples = [f"{item['img']}: {item['caption'][:100]}..." for item in raw_items[:10]]

            prompt = f"""
You are a senior dataset curator for Stable Diffusion / Flux LoRA training.
Analyze the following dataset statistics and provide a structured report on potential biases,
over‑represented concepts, missing diversity, contradictions, and recommendations for improving the dataset.

**Dataset Summary:**
- Total images: {total_images}
- Empty captions: {empty_captions}
- Unique tags: {len(tag_counts)}

**Top 50 Tags:**
{', '.join([f"{tag} ({count})" for tag, count in top_tags])}

**Rare Tags (appear only once):**
{', '.join([tag for tag, _ in rare_tags[:30]]) if rare_tags else 'None'}

**Logical Contradictions Found (same caption):**
{contradictions_str}

**Caption Samples (first 10 images):**
{', '.join(samples)}

Please provide:
1. Quick diagnosis: what concepts dominate the dataset?
2. Probable biases (subject, pose, framing, style, lighting, clothing, etc.)
3. Risks for LoRA training (overfitting, trigger weakness, contradictory tags)
4. Priority corrections (which tags to merge, remove, or add)
5. Diversity recommendations (missing angles, expressions, environments)
"""
            st.session_state.bias_prompt = prompt
            st.rerun()

        if st.session_state.bias_prompt:
            st.text_area("Copy this prompt and paste it into an LLM:", st.session_state.bias_prompt, height=300)
            if st.button("Clear Prompt"):
                st.session_state.bias_prompt = ""
                st.rerun()


                
# ======================================================================
# TAB 5: Caption Tools (Batch Quality Ratings + Validation & Auto‑Fix)
# ======================================================================

elif current_tab == "📝 Caption Tools":
    if not folder_valid or not raw_items:
        st.info("📁 Please load a valid dataset first to use caption tools.")
    else:
        st.header("📝 Caption Tools")
        st.markdown("Batch‑append quality ratings, validate captions, and auto‑fix common issues.")

        # ------------------------------------------------------------------
        # 1. BATCH QUALITY RATING APPENDER (chunked, connection‑safe)
        # ------------------------------------------------------------------
        st.subheader("⭐ Append Image Quality Ratings to Captions")
        st.markdown(
            "Add a line like `Image quality 95-100.` to the end of every caption, based on the "
            "overall quality score already computed in the Quality Analytics tab."
        )

        qual_map, _ = compute_all_metrics(dataset_path)

        def quality_range(score):
            if score < 60:
                return "60-70"
            elif 60 <= score < 70:
                return "60-70"
            elif 70 <= score < 80:
                return "70-80"
            elif 80 <= score < 90:
                return "80-90"
            elif 90 <= score < 95:
                return "90-95"
            else:
                return "95-100"

        # Chunked processing for quality ratings
        if "quality_append_cursor" not in st.session_state:
            st.session_state.quality_append_cursor = 0
        if "quality_append_total" not in st.session_state:
            st.session_state.quality_append_total = 0

        if st.button("🔍 Preview Quality Rating Append (show 5 random examples)"):
            import random
            valid_items = [item for item in raw_items if item["img"] in qual_map]
            if not valid_items:
                st.warning("No quality scores available. Run the Quality Analytics tab first.")
            else:
                samples = random.sample(valid_items, min(5, len(valid_items)))
                for item in samples:
                    score = qual_map[item["img"]]["overall_quality"]
                    q_range = quality_range(score)
                    with st.expander(f"📄 {item['img']} – Quality: {score:.1f}"):
                        st.caption(f"**Original caption end:**  `{item['caption'][-150:]}`")
                        st.caption(f"**Would append:**  `Image quality {q_range}.`")

        # Only show the "Append" button when not already in progress
        if st.session_state.quality_append_cursor == 0:
            if st.button("✅ Append Quality Ratings to ALL Captions"):
                valid_items = [item for item in raw_items if item["img"] in qual_map]
                st.session_state.quality_append_cursor = 0
                st.session_state.quality_append_total = len(valid_items)
                st.rerun()

        # If a batch is in progress, process one chunk of 10 files
        if st.session_state.quality_append_cursor < st.session_state.quality_append_total:
            valid_items = [item for item in raw_items if item["img"] in qual_map]
            total = st.session_state.quality_append_total
            start = st.session_state.quality_append_cursor
            end = min(start + 10, total)
            chunk = valid_items[start:end]

            progress_text = f"Processing files {start+1}–{end} of {total}..."
            progress_bar = st.progress(start / total, text=progress_text)

            for item in chunk:
                score = qual_map[item["img"]]["overall_quality"]
                q_range = quality_range(score)
                new_line = f" Image quality {q_range}."
                txt_path = item["txt_path"]
                with open(txt_path, "r", encoding="utf-8") as f:
                    current_text = f.read().strip()
                if not current_text.endswith(new_line):
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(current_text + new_line)

            # Advance cursor and rerun to process the next chunk
            st.session_state.quality_append_cursor = end
            if end >= total:
                # Finished – clean up and show success
                st.cache_data.clear()
                st.success(f"✅ Successfully appended quality ratings to {total} captions!")
                st.session_state.quality_append_cursor = 0
                st.session_state.quality_append_total = 0
            st.rerun()

        st.markdown("---")

        # ------------------------------------------------------------------
        # 2. CAPTION VALIDATION & AUTO‑FIX (chunked, connection‑safe)
        # ------------------------------------------------------------------
        st.subheader("🔍 Caption Validation & Auto‑Fix")
        st.markdown("Detect common formatting issues and fix them automatically.")

        def find_issues(caption):
            issues = []
            if re.search(r',[^\s]', caption):
                issues.append("Missing space after comma")
            if ',, ' in caption or ',,' in caption:
                issues.append("Double commas")
            if caption and caption[-1] not in '.!?':
                issues.append("Missing final period")
            if caption != caption.strip():
                issues.append("Extra spaces around text")
            if re.search(r'\bi\b', caption):
                issues.append("Lowercase 'i' (should be 'I')")
            return issues

        def auto_fix(caption):
            fixed = caption.strip()
            fixed = re.sub(r',([^\s])', r', \1', fixed)
            fixed = re.sub(r',\s*,+', ',', fixed)
            if fixed and fixed[-1] not in '.!?':
                fixed += "."
            fixed = re.sub(r'\bi\b', 'I', fixed)
            return fixed

        # Scan all captions (always fresh on rerun)
        issues_found = {}
        for item in raw_items:
            issues = find_issues(item["caption"])
            if issues:
                issues_found[item["img"]] = (item["caption"], issues)

        if not issues_found:
            st.success("🎉 No common issues found in any caption!")
        else:
            st.warning(f"Issues found in **{len(issues_found)}** captions.")

            # Chunked mass fix
            if "mass_fix_cursor" not in st.session_state:
                st.session_state.mass_fix_cursor = 0
            if "mass_fix_file_list" not in st.session_state:
                st.session_state.mass_fix_file_list = []
            if "mass_fix_total" not in st.session_state:
                st.session_state.mass_fix_total = 0

            # Start mass fix (only if not already in progress)
            if st.session_state.mass_fix_cursor == 0 and st.button("🛠️ Auto‑Fix ALL captions"):
                # Store list of files to fix (ordered)
                file_list = list(issues_found.keys())
                st.session_state.mass_fix_file_list = file_list
                st.session_state.mass_fix_total = len(file_list)
                st.session_state.mass_fix_cursor = 0
                st.rerun()

            # Process a chunk
            if st.session_state.mass_fix_cursor < st.session_state.mass_fix_total:
                file_list = st.session_state.mass_fix_file_list
                total = st.session_state.mass_fix_total
                start = st.session_state.mass_fix_cursor
                end = min(start + 10, total)
                chunk_files = file_list[start:end]

                progress_text = f"Fixing files {start+1}–{end} of {total}..."
                progress_bar = st.progress(start / total, text=progress_text)

                for fname in chunk_files:
                    original, _ = issues_found[fname]
                    fixed = auto_fix(original)
                    txt_path = os.path.join(st.session_state.dataset_dir, f"{os.path.splitext(fname)[0]}.txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(fixed)

                st.session_state.mass_fix_cursor = end
                if end >= total:
                    # Finished
                    st.cache_data.clear()
                    st.success(f"✅ Successfully fixed {total} captions!")
                    st.session_state.mass_fix_cursor = 0
                    st.session_state.mass_fix_file_list = []
                    st.session_state.mass_fix_total = 0
                st.rerun()

            # Individual fixes (unchanged – instant, no chunking needed)
            st.markdown("**Or fix files individually:**")
            for fname, (original, issues) in issues_found.items():
                with st.expander(f"📄 {fname} – {len(issues)} issue(s)"):
                    st.write("**Issues:** " + ", ".join(issues))
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Original (end):")
                        st.code(original[-200:], language="text")
                    fixed = auto_fix(original)
                    with col2:
                        st.caption("Fixed (end):")
                        st.code(fixed[-200:], language="text")
                    if st.button(f"✅ Fix this file", key=f"fix_{fname}"):
                        txt_path = os.path.join(st.session_state.dataset_dir, f"{os.path.splitext(fname)[0]}.txt")
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(fixed)
                        st.cache_data.clear()
                        st.success(f"Fixed {fname}")
                        st.rerun()



# ======================================================================
# TAB 6: AI Assistance (CLIP with PyTorch – quick actions + progress)
# ======================================================================

elif current_tab == "🧠 AI Assistance":
    if not folder_valid or not raw_items:
        st.info("📁 Please load a valid dataset first.")
    else:
        st.header("🧠 AI‑Assisted Hallucination Check")
        st.markdown(
            "Use CLIP to compute a **similarity score** between each image and its caption. "
            "Low scores may indicate hallucinated or mismatched descriptions. "
            "You can preview flagged pairs and decide whether to edit them."
        )

        # ---- session state for this tab ----
        if "confirm_delete" not in st.session_state:
            st.session_state.confirm_delete = None

        # ------------------------------------------------------------------
        # CACHED CLIP MODEL LOADING (runs once)
        # ------------------------------------------------------------------
        @st.cache_resource
        def load_clip_model():
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            return model, preprocess, tokenizer

        # ------------------------------------------------------------------
        # CACHED SIMILARITY COMPUTATION (with progress bar)
        # ------------------------------------------------------------------
        @st.cache_data(show_spinner=False)
        def compute_clip_scores(folder_path, raw_items):
            import torch
            from PIL import Image

            model, preprocess, tokenizer = load_clip_model()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            scores = []
            total = len(raw_items)
            progress_bar = st.progress(0, text="Computing CLIP scores...")
            for idx, item in enumerate(raw_items):
                img_path = os.path.join(folder_path, item["img"])
                try:
                    image = Image.open(img_path).convert("RGB")
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    caption = item["caption"]
                    text_input = tokenizer([caption]).to(device)

                    with torch.no_grad():
                        image_features = model.encode_image(image_input)
                        text_features = model.encode_text(text_input)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        similarity = (image_features @ text_features.T).item()

                    scores.append({
                        "image": item["img"],
                        "caption": caption[:150] + "..." if len(caption) > 150 else caption,
                        "score": round(similarity, 4)
                    })
                except Exception as e:
                    scores.append({
                        "image": item["img"],
                        "caption": "ERROR",
                        "score": None
                    })
                progress_bar.progress((idx + 1) / total,
                                      text=f"Computing CLIP scores ({idx+1}/{total})")
            progress_bar.empty()
            return scores

        # ------------------------------------------------------------------
        # MAIN INTERFACE
        # ------------------------------------------------------------------
        if st.button("🚀 Compute Similarity Scores"):
            scores = compute_clip_scores(dataset_path, raw_items)
            st.session_state.clip_scores = scores
            st.success(f"Computed scores for {len(scores)} images.")
            st.rerun()

        if "clip_scores" not in st.session_state:
            st.session_state.clip_scores = None

        if st.session_state.clip_scores:
            scores = st.session_state.clip_scores
            valid_scores = [s for s in scores if s["score"] is not None]
            if not valid_scores:
                st.warning("No valid scores computed.")
            else:
                import pandas as pd
                df = pd.DataFrame(valid_scores)
                df = df.sort_values("score")

                threshold = st.slider(
                    "Flag pairs with similarity below:",
                    min_value=0.0, max_value=1.0, value=0.25, step=0.01,
                    help="Lower threshold → fewer flagged pairs. Typical good matches are > 0.28."
                )
                flagged = df[df["score"] < threshold]
                st.write(f"**{len(flagged)} of {len(df)} pairs flagged** (score < {threshold:.2f})")

                st.subheader("🚩 Flagged Pairs")
                if flagged.empty:
                    st.success("No hallucinations detected – all scores are above the threshold.")
                else:
                    for _, row in flagged.iterrows():
                        img_path = os.path.join(dataset_path, row["image"])
                        col_img, col_cap, col_score, col_actions = st.columns([1, 3, 1, 1.5])
                        with col_img:
                            img_pil = load_image_pil(img_path)
                            if img_pil:
                                st.image(img_pil, width=120)
                            else:
                                st.caption("Error")
                        with col_cap:
                            st.write(f"**{row['image']}**")
                            st.caption(row["caption"])
                        with col_score:
                            st.metric("Score", f"{row['score']:.3f}")
                        with col_actions:
                            if st.button("✏️ Edit", key=f"edit_{row['image']}"):
                                new_item = next((item for item in raw_items if item["img"] == row["image"]), None)
                                if new_item:
                                    st.session_state.selected_img = new_item
                                    st.session_state.active_tab = "🖥️ Soho Workspace & Browser"
                                    st.rerun()
                            if st.button("🚫 Ignore", key=f"ignore_{row['image']}"):
                                st.session_state.clip_scores = [s for s in st.session_state.clip_scores
                                                                 if s["image"] != row["image"]]
                                st.rerun()
                            if st.button("🗑️ Delete", key=f"delete_{row['image']}"):
                                if st.session_state.confirm_delete != row["image"]:
                                    st.session_state.confirm_delete = row["image"]
                                    st.warning("Click again to confirm deletion.")
                                    st.rerun()
                                else:
                                    png_path = os.path.join(dataset_path, row["image"])
                                    txt_path = os.path.splitext(png_path)[0] + ".txt"
                                    try:
                                        if os.path.exists(png_path):
                                            os.remove(png_path)
                                        if os.path.exists(txt_path):
                                            os.remove(txt_path)
                                        st.session_state.clip_scores = [s for s in st.session_state.clip_scores
                                                                         if s["image"] != row["image"]]
                                        st.cache_data.clear()
                                        st.session_state.confirm_delete = None
                                        st.success(f"Deleted {row['image']}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Delete failed: {e}")
                        st.markdown("---")

                with st.expander("📋 View all similarity scores"):
                    st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Click the button above to compute CLIP similarity scores for the current dataset.")



# ======================================================================
# TAB 7: Captioner (Multi‑Model + Comparison)
# ======================================================================
elif current_tab == "💾 Captioner":
    import datetime

    st.header("💾 Image Captioner (Multi‑Model)")
    st.markdown(
        "Upload one or more images to caption. "
        "Single image mode lets you edit and save a caption. "
        "Batch mode runs **multiple models** on each image, saves all results, "
        "and shows a comparison view so you can pick the best one."
    )

    # ---- Mode selection ----
    mode = st.radio("Mode:", ["Single Image", "Batch Processing (multi‑model)"], horizontal=True)

    # ------------------------------------------------------------------
    # MODEL REGISTRY
    # ------------------------------------------------------------------
    # Default models (you can add / remove / edit these)
    if "custom_models" not in st.session_state:
        st.session_state.custom_models = {
            "JoyCaption (default)": {
                "type": "local_hf",
                "model_id": "fancyfeast/llama-joycaption-beta-one-hf-llava",
                "prompt_template": "{system}\n\n{user}",   # will be filled later
                "trust_remote_code": True,
                "torch_dtype": "float16",
                "device_map": "auto",
            },
            # Example: Ollama model
            # "Llava 7B (Ollama)": {
            #     "type": "ollama",
            #     "model_id": "llava:7b",
            #     "api_url": "http://localhost:11434",
            #     "prompt_template": "{system}\n\n{user}",
            # },
        }

    # ---- Load / edit custom models (collapsible) ----
    with st.expander("⚙️ Model Settings (add / remove models)", expanded=False):
        st.markdown("**Current models:**")
        model_names = list(st.session_state.custom_models.keys())
        for name in model_names:
            st.write(f"- {name} ({st.session_state.custom_models[name]['type']})")

        st.markdown("---")
        new_name = st.text_input("New model name", key="new_model_name")
        new_type = st.selectbox("Model type", ["local_hf", "ollama", "openai_compat"], key="new_model_type")
        new_model_id = st.text_input("Model ID / HuggingFace repo", key="new_model_id")
        if st.button("➕ Add model"):
            if new_name and new_model_id:
                st.session_state.custom_models[new_name] = {
                    "type": new_type,
                    "model_id": new_model_id,
                    "api_url": "http://localhost:11434" if new_type == "ollama" else "",
                    "prompt_template": "{system}\n\n{user}",
                    "trust_remote_code": True,
                    "torch_dtype": "float16",
                    "device_map": "auto",
                }
                st.success(f"Model '{new_name}' added!")
                st.rerun()
            else:
                st.warning("Name and Model ID are required.")

        # Remove model
        if len(st.session_state.custom_models) > 1:
            remove_model = st.selectbox("Remove model", [""] + list(st.session_state.custom_models.keys()),
                                        key="remove_model_select")
            if remove_model and st.button("🗑️ Remove selected model"):
                del st.session_state.custom_models[remove_model]
                st.success(f"Model '{remove_model}' removed.")
                st.rerun()

    # ------------------------------------------------------------------
    # CACHE THE JOYCAPTION MODEL (the default local one)
    # ------------------------------------------------------------------
    @st.cache_resource(show_spinner="Loading JoyCaption model (this may take a few minutes)...")
    def load_joycaption_model():
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        import torch
        model_id = "fancyfeast/llama-joycaption-beta-one-hf-llava"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        return processor, model

    # ------------------------------------------------------------------
    # COMMON CAPTION GENERATION (single image, single model)
    # ------------------------------------------------------------------
    def generate_caption(image, model_name, progress_callback=None):
        """Generate a caption using the specified model. Returns the caption string."""
        cfg = st.session_state.custom_models[model_name]
        model_type = cfg["type"]

        # Build the prompt from session state (system + user)
        system_prompt = st.session_state.get("captioner_system_prompt", "")
        user_prompt = st.session_state.get("captioner_user_prompt", "")
        full_prompt = cfg.get("prompt_template", "{system}\n\n{user}").format(
            system=system_prompt, user=user_prompt
        )

        if model_type == "local_hf":
            # Local HuggingFace model – only JoyCaption is implemented here.
            # For other HF models you would add similar loading logic.
            if "joycaption" in model_name.lower() or "fancyfeast" in cfg.get("model_id", ""):
                processor, model = load_joycaption_model()
                import torch
                convo = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
                device = model.device
                inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to(device)
                with torch.no_grad():
                    generate_ids = model.generate(
                        **inputs,
                        max_new_tokens=st.session_state.captioner_max_tokens,
                        do_sample=True,
                        temperature=st.session_state.captioner_temperature,
                        top_p=0.9,
                        repetition_penalty=1.1
                    )
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = generate_ids[0][input_length:]
                caption = processor.decode(generated_tokens, skip_special_tokens=True).strip()
                # Clean watermark phrases
                clean_phrases = []
                for phrase in caption.split(','):
                    phrase_clean = phrase.strip()
                    if any(bad_word in phrase_clean.lower() for bad_word in ['watermark', 'text', 'logo', 'signature', 'gkal']):
                        continue
                    if phrase_clean:
                        clean_phrases.append(phrase_clean)
                return ", ".join(clean_phrases)
            else:
                return f"[Model '{model_name}' not yet implemented for local HF]"

        elif model_type == "ollama":
            # Ollama API call
            import requests
            api_url = cfg.get("api_url", "http://localhost:11434").rstrip("/")
            endpoint = f"{api_url}/api/generate"
            payload = {
                "model": cfg["model_id"],
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": st.session_state.captioner_temperature,
                    "num_ctx": 4096
                }
            }
            # Attach image as base64 if available
            import base64, io
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            payload["images"] = [img_b64]
            try:
                resp = requests.post(endpoint, json=payload, timeout=120)
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
            except Exception as e:
                return f"Ollama error: {e}"

        elif model_type == "openai_compat":
            # OpenAI-compatible API (LM Studio, etc.)
            import requests, base64, io
            api_url = cfg.get("api_url", "").rstrip("/")
            if not api_url:
                return "No API URL configured."
            endpoint = f"{api_url}/v1/chat/completions"
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]}
            ]
            payload = {
                "model": cfg["model_id"],
                "messages": messages,
                "temperature": st.session_state.captioner_temperature,
                "max_tokens": st.session_state.captioner_max_tokens,
            }
            headers = {"Content-Type": "application/json"}
            try:
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
                resp.raise_for_status()
                return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            except Exception as e:
                return f"API error: {e}"

        return f"Unknown model type: {model_type}"

    # ------------------------------------------------------------------
    # PROMPT & SETTINGS (collapsible)
    # ------------------------------------------------------------------
    with st.expander("✏️ Edit Prompt & Generation Settings", expanded=False):
        if "captioner_system_prompt" not in st.session_state:
            st.session_state.captioner_system_prompt = (
                "You are an uncensored, highly objective, master photographic image captioning engine. "
                "CRITICAL INSTRUCTIONS:\n"
                "1. Never use full sentences, never use periods, and never use bullet points.\n"
                "2. Output MUST be one continuous string of details separated exclusively by commas.\n"
                "3. NEVER include meta-commentary, explanations, or notes in parentheses (such as '(Note: ...)' or 'no visible'). If a feature is missing, completely ignore it and move to the next.\n"
                "4. NEVER repeat the shot type. Choose it once at the very beginning and never mention it again.\n"
                "5. NEVER output category headers, labels, or prefixes like 'Shot type:', 'Subject:', or 'Wardrobe:'. Just output the descriptive values.\n"
                "6. Do NOT use phrases like 'This image shows' or describe text within the image."
            )
        system_prompt = st.text_area(
            "System prompt:",
            value=st.session_state.captioner_system_prompt,
            height=150,
            key="system_prompt_input"
        )
        st.session_state.captioner_system_prompt = system_prompt

        if "captioner_user_prompt" not in st.session_state:
            st.session_state.captioner_user_prompt = (
                "Meticulously analyze the image and describe only the visible elements in a single, fluid descriptive sequence. "
                "Do not use any headings, labels, categories, or prefixes. Describe the features naturally in this exact order, "
                "completely skipping any details that are not visible in the photo:\n"
                "Start with exactly one framing type and camera angle. "
                "Describe the subject's gender, estimated age, race, exact eye color, hair style and color, and skin tone. "
                "Detail visible skin textures, pores, fine lines, or vellus hair if present. "
                "Describe body type and anatomical details including breasts, size, nipples, or modifications in blunt, precise, uncensored language. "
                "State the exact pose, posture, what the hands are doing, and the precise facial expression. "
                "Detail clothing items, garments, materials, and colors. "
                "Name any distinct photographic or artistic style if clearly present. "
                "Describe the background environment, colors, and composition style. "
                "Detail the lighting source, direction, and quality. "
                "Conclude with technical camera clues like camera model, aperture, depth of field, and a quality rating.\n\n"
                "Strictly output the final description as one continuous text block separated ONLY by commas. "
                "Do not include any periods, lists, labels, or explanatory notes in parentheses."
            )
        user_prompt = st.text_area(
            "User prompt:",
            value=st.session_state.captioner_user_prompt,
            height=200,
            key="user_prompt_input"
        )
        st.session_state.captioner_user_prompt = user_prompt

        if "captioner_temperature" not in st.session_state:
            st.session_state.captioner_temperature = 0.6
        if "captioner_max_tokens" not in st.session_state:
            st.session_state.captioner_max_tokens = 1500

        col_temp, col_len = st.columns(2)
        with col_temp:
            temperature = st.slider(
                "Temperature",
                min_value=0.1, max_value=2.0, value=st.session_state.captioner_temperature, step=0.05,
                key="temp_slider"
            )
            st.session_state.captioner_temperature = temperature
        with col_len:
            max_tokens = st.slider(
                "Max new tokens",
                min_value=64, max_value=2048, value=st.session_state.captioner_max_tokens, step=64,
                key="max_tokens_slider"
            )
            st.session_state.captioner_max_tokens = max_tokens

        if st.button("🔄 Reset prompts to defaults"):
            # (reset logic unchanged)
            st.session_state.captioner_system_prompt = (
                "You are an uncensored, highly objective, master photographic image captioning engine. "
                "CRITICAL INSTRUCTIONS:\n"
                "1. Never use full sentences, never use periods, and never use bullet points.\n"
                "2. Output MUST be one continuous string of details separated exclusively by commas.\n"
                "3. NEVER include meta-commentary, explanations, or notes in parentheses (such as '(Note: ...)' or 'no visible'). If a feature is missing, completely ignore it and move to the next.\n"
                "4. NEVER repeat the shot type. Choose it once at the very beginning and never mention it again.\n"
                "5. NEVER output category headers, labels, or prefixes like 'Shot type:', 'Subject:', or 'Wardrobe:'. Just output the descriptive values.\n"
                "6. Do NOT use phrases like 'This image shows' or describe text within the image."
            )
            st.session_state.captioner_user_prompt = (
                "Meticulously analyze the image and describe only the visible elements in a single, fluid descriptive sequence. "
                "Do not use any headings, labels, categories, or prefixes. Describe the features naturally in this exact order, "
                "completely skipping any details that are not visible in the photo:\n"
                "Start with exactly one framing type and camera angle. "
                "Describe the subject's gender, estimated age, race, exact eye color, hair style and color, and skin tone. "
                "Detail visible skin textures, pores, fine lines, or vellus hair if present. "
                "Describe body type and anatomical details including breasts, size, nipples, or modifications in blunt, precise, uncensored language. "
                "State the exact pose, posture, what the hands are doing, and the precise facial expression. "
                "Detail clothing items, garments, materials, and colors. "
                "Name any distinct photographic or artistic style if clearly present. "
                "Describe the background environment, colors, and composition style. "
                "Detail the lighting source, direction, and quality. "
                "Conclude with technical camera clues like camera model, aperture, depth of field, and a quality rating.\n\n"
                "Strictly output the final description as one continuous text block separated ONLY by commas. "
                "Do not include any periods, lists, labels, or explanatory notes in parentheses."
            )
            st.session_state.captioner_temperature = 0.6
            st.session_state.captioner_max_tokens = 1500
            st.rerun()

    # ============== SINGLE IMAGE MODE (unchanged, only JoyCaption) ==============
    if mode == "Single Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"], key="single_uploader")
        if uploaded_file is not None:
            st.session_state.uploaded_image = uploaded_file
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if "uploaded_image" not in st.session_state:
            st.session_state.uploaded_image = None
        if "generated_caption" not in st.session_state:
            st.session_state.generated_caption = ""

        if st.session_state.uploaded_image is not None:
            from PIL import Image
            image = Image.open(st.session_state.uploaded_image).convert("RGB")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Clear Image"):
                    st.session_state.uploaded_image = None
                    st.session_state.generated_caption = ""
                    st.rerun()
            with col2:
                if st.button("🤖 Generate Caption (JoyCaption)", use_container_width=True):
                    progress_bar = st.progress(0, text="Starting...")
                    with st.spinner("Generating caption..."):
                        caption = generate_caption(image, "JoyCaption (default)")
                        st.session_state.generated_caption = caption
                    progress_bar.empty()
                    st.success("Caption generated!")
                    st.rerun()

            if st.session_state.generated_caption:
                st.markdown("### ✏️ Edit Caption")
                edited_caption = st.text_area(
                    "Edit the caption below:",
                    value=st.session_state.generated_caption,
                    height=200,
                    key="caption_editor"
                )
                st.markdown("---")
                st.subheader("💾 Save to Dataset")
                import os
                existing_files = os.listdir(st.session_state.dataset_dir) if os.path.isdir(st.session_state.dataset_dir) else []
                png_files = [f for f in existing_files if f.lower().endswith('.png')]
                base_name = f"captioned_{len(png_files)+1:04d}"
                suggested_img = f"{base_name}.png"
                suggested_txt = f"{base_name}.txt"
                col_name1, col_name2 = st.columns(2)
                with col_name1:
                    img_name = st.text_input("Image filename:", value=suggested_img, key="img_name_input")
                with col_name2:
                    txt_name = st.text_input("Caption filename:", value=suggested_txt, key="txt_name_input")
                if st.button("💾 Save Image & Caption to Dataset", use_container_width=True):
                    if not os.path.isdir(st.session_state.dataset_dir):
                        st.error("Dataset folder does not exist.")
                    else:
                        img_path = os.path.join(st.session_state.dataset_dir, img_name)
                        image.save(img_path, "PNG")
                        txt_path = os.path.join(st.session_state.dataset_dir, txt_name)
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(edited_caption.strip())
                        st.cache_data.clear()
                        st.success(f"Saved `{img_name}` and `{txt_name}` to dataset!")
                        st.session_state.uploaded_image = None
                        st.session_state.generated_caption = ""
                        st.rerun()
        else:
            st.info("Upload an image to get started.")

    # ============== BATCH PROCESSING (multi‑model) ==============
    else:
        st.subheader("📚 Batch Multi‑Model Captioning")
        st.markdown("Select models, upload images, and run batch captioning. Results are saved and compared.")

        # ---- Model selection ----
        model_names = list(st.session_state.custom_models.keys())
        selected_models = st.multiselect(
            "Select models to use:",
            model_names,
            default=["JoyCaption (default)"],
            key="batch_model_select"
        )
        if not selected_models:
            st.warning("Select at least one model.")
            st.stop()

        # ---- Upload images ----
        uploaded_files = st.file_uploader(
            "Choose images (multiple allowed)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="batch_uploader"
        )
        if uploaded_files:
            st.session_state.batch_files = uploaded_files
            st.write(f"**{len(uploaded_files)} images loaded**")
        else:
            st.info("Upload images to begin.")
            st.stop()

        # Session state for batch processing
        if "batch_files" not in st.session_state:
            st.session_state.batch_files = []
        if "batch_cursor_img" not in st.session_state:
            st.session_state.batch_cursor_img = 0      # index of current image
        if "batch_cursor_model" not in st.session_state:
            st.session_state.batch_cursor_model = 0    # index of current model
        if "batch_models" not in st.session_state:
            st.session_state.batch_models = []
        if "batch_total_images" not in st.session_state:
            st.session_state.batch_total_images = 0
        if "batch_total_models" not in st.session_state:
            st.session_state.batch_total_models = 0
        if "batch_results" not in st.session_state:
            st.session_state.batch_results = {}   # { img_index: { model_name: caption } }
        if "batch_start_time" not in st.session_state:
            st.session_state.batch_start_time = None

        # Start batch button
        can_start = len(uploaded_files) > 0 and len(selected_models) > 0
        if st.button("🚀 Start Batch Captioning", use_container_width=True, disabled=not can_start):
            st.session_state.batch_files = uploaded_files
            st.session_state.batch_models = selected_models
            st.session_state.batch_cursor_img = 0
            st.session_state.batch_cursor_model = 0
            st.session_state.batch_total_images = len(uploaded_files)
            st.session_state.batch_total_models = len(selected_models)
            st.session_state.batch_results = {}
            st.session_state.batch_start_time = time.time()
            st.rerun()

        # Process ONE (image, model) pair per rerun
        total_imgs = st.session_state.batch_total_images
        total_mdls = st.session_state.batch_total_models
        cursor_img = st.session_state.batch_cursor_img
        cursor_mdl = st.session_state.batch_cursor_model

        if cursor_img < total_imgs and cursor_mdl < total_mdls:
            files = st.session_state.batch_files
            models = st.session_state.batch_models
            current_file = files[cursor_img]
            current_model = models[cursor_mdl]

            # Overall progress
            total_jobs = total_imgs * total_mdls
            done_jobs = cursor_img * total_mdls + cursor_mdl
            progress_text = f"Image {cursor_img+1}/{total_imgs} · Model {cursor_mdl+1}/{total_mdls} ({current_model})"
            progress_bar = st.progress(done_jobs / total_jobs if total_jobs else 0, text=progress_text)

            # ETA
            if done_jobs > 0 and st.session_state.batch_start_time:
                elapsed = time.time() - st.session_state.batch_start_time
                sec_per_job = elapsed / done_jobs
                remaining = (total_jobs - done_jobs) * sec_per_job
                eta_min = int(remaining // 60)
                eta_sec = int(remaining % 60)
                eta_str = f"ETA: {eta_min}m {eta_sec}s"
                if sec_per_job > 0:
                    eta_str += f" · ~{1.0/sec_per_job:.1f} it/s"
                progress_bar.progress(done_jobs / total_jobs, text=f"{progress_text} | {eta_str}")

            # Generate caption
            from PIL import Image
            img = Image.open(current_file).convert("RGB")
            caption = generate_caption(img, current_model)

            # Save to session state
            results = st.session_state.batch_results
            img_idx = cursor_img
            if img_idx not in results:
                results[img_idx] = {}
            results[img_idx][current_model] = caption
            st.session_state.batch_results = results

            # Save to disk (suffix)
            base_name = os.path.splitext(current_file.name)[0]
            suffix = current_model.lower().replace(" ", "_").replace("(", "").replace(")", "")
            txt_name = f"{base_name}_{suffix}.txt"
            txt_path = os.path.join(st.session_state.dataset_dir, txt_name)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)

            # Advance cursor
            next_mdl = cursor_mdl + 1
            if next_mdl >= total_mdls:
                # Next image
                st.session_state.batch_cursor_img = cursor_img + 1
                st.session_state.batch_cursor_model = 0
            else:
                st.session_state.batch_cursor_model = next_mdl

            # Check if finished
            if st.session_state.batch_cursor_img >= total_imgs:
                st.success(f"✅ Batch complete! {total_imgs} images × {total_mdls} models = {total_jobs} captions.")
            st.rerun()

        # ---- AFTER BATCH: Comparison view ----
        if cursor_img >= total_imgs and total_imgs > 0:
            st.success("All captions generated! Use the comparison below.")
            st.markdown("---")
            st.subheader("🔍 Compare Captions")
            # Select image
            img_names = [f.name for f in st.session_state.batch_files]
            selected_img_name = st.selectbox("Select image:", img_names, key="compare_img_select")
            if selected_img_name:
                img_idx = img_names.index(selected_img_name)
                from PIL import Image
                img = Image.open(st.session_state.batch_files[img_idx]).convert("RGB")
                st.image(img, width=400, caption=selected_img_name)

                # Show captions from each model
                results = st.session_state.batch_results.get(img_idx, {})
                if results:
                    for model_name in st.session_state.batch_models:
                        cap = results.get(model_name, "N/A")
                        with st.expander(f"🤖 {model_name}"):
                            st.write(cap)
                            if st.button(f"✅ Use this caption ({model_name})", key=f"use_{img_idx}_{model_name}"):
                                # Overwrite the main caption of the image (if in dataset) or just copy to clipboard
                                # For simplicity, we copy to a textbox so the user can manually save.
                                st.session_state.selected_comparison_caption = cap
                                st.rerun()

                    # If a caption was selected, show it in a text area
                    if "selected_comparison_caption" in st.session_state and st.session_state.selected_comparison_caption:
                        st.markdown("### ✏️ Selected Caption (edit and save manually)")
                        edited = st.text_area("Edit caption:", value=st.session_state.selected_comparison_caption, height=200)
                        if st.button("💾 Save to Dataset as main caption"):
                            # Determine the main .txt name (without suffix)
                            main_txt = os.path.splitext(selected_img_name)[0] + ".txt"
                            main_txt_path = os.path.join(st.session_state.dataset_dir, main_txt)
                            with open(main_txt_path, "w", encoding="utf-8") as f:
                                f.write(edited.strip())
                            st.success(f"Saved main caption for {selected_img_name}")
                            st.cache_data.clear()
                            del st.session_state.selected_comparison_caption
                            st.rerun()



# ======================================================================
# TAB 8: Image Cropper (double‑click to confirm, hint included)
# ======================================================================

elif current_tab == "✂️ Image Cropper":
    from streamlit_cropper import st_cropper

    st.header("✂️ Image Cropper")
    st.markdown(
        "Select or upload a single image. "
        "💡 Drag/resize the crop box, then **double‑click** inside the box to confirm. "
        "Your cropped image will be ready for download immediately."
    )

    # ---- Source selection (single image) ----
    source = st.radio("Image source:", ["From dataset", "Upload new"], horizontal=True)
    current_pil = None
    current_name = None

    if source == "From dataset":
        if not folder_valid or not raw_items:
            st.warning("Load a dataset first.")
            st.stop()
        img_names = [item["img"] for item in raw_items]
        selected_name = st.selectbox("Choose an image:", img_names)
        if selected_name:
            path = os.path.join(dataset_path, selected_name)
            current_pil = Image.open(path).convert("RGB")
            current_name = selected_name
    else:
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded_file:
            current_pil = Image.open(uploaded_file).convert("RGB")
            current_name = uploaded_file.name

    if current_pil is None:
        st.info("Select or upload an image to begin.")
        st.stop()

    # ---- Aspect ratio ----
    aspect_ratio = st.selectbox("Aspect ratio:", ["Free", "16:9", "9:16", "4:3", "3:2", "1:1"])
    aspect_dict = {
        "Free": None,
        "16:9": (16, 9),
        "9:16": (9, 16),
        "4:3": (4, 3),
        "3:2": (3, 2),
        "1:1": (1, 1),
    }
    ratio = aspect_dict[aspect_ratio]

    # ---- Center the cropper ----
    left, center, right = st.columns([1, 2, 1])
    with center:
        cropped_img = st_cropper(
            current_pil,
            realtime_update=False,
            box_color="#00e5ff",
            aspect_ratio=ratio,
            return_type="image",
        )

    # ---- After double‑click, the cropped image is available ----
    if cropped_img is not None:
        buf = io.BytesIO()
        cropped_img.save(buf, format="PNG")
        buf.seek(0)

        left2, center2, right2 = st.columns([1, 2, 1])
        with center2:
            st.download_button(
                label="📥 Download Cropped PNG",
                data=buf,
                file_name=f"cropped_{current_name or 'image'}.png",
                mime="image/png",
            )
    else:
        # Hint shown until the user double‑clicks
        st.caption("💡 Double‑click inside the crop box to confirm – download will appear.")



# ======================================================================
# TAB 9: Image Converter (same size, max‑quality PNG)
# ======================================================================

elif current_tab == "🔄 Image Converter":
    import io, zipfile, tempfile, time
    from PIL import Image

    st.header("🔄 Image Converter")
    st.markdown(
        "Convert images (AVIF, WEBP, JPG, PNG, BMP, TIFF, etc.) to **PNG** format. "
        "The original size is kept, and the output uses maximum quality (no compression). "
        "Single image or batch mode (ZIP for batch). \n\n"
        "💡AVIF conversion needs pillow-avif installed."
    )

    # ---- Mode selection ----
    mode = st.radio("Mode:", ["Single Image", "Batch Processing"], horizontal=True)

    # ---- Helper: convert any image to a high‑quality PNG (same dimensions) ----
    def convert_to_png(img: Image.Image) -> Image.Image:
        # Convert to RGB if the image has transparency or palette
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        return img

    # ========== SINGLE IMAGE MODE ==========
    if mode == "Single Image":
        uploaded_file = st.file_uploader(
            "Choose an image…",
            type=["avif", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
            key="single_convert_uploader"
        )
        if uploaded_file:
            if st.button("🔄 Convert to PNG", key="convert_single_btn"):
                try:
                    img = Image.open(uploaded_file)
                    converted = convert_to_png(img)
                    buf = io.BytesIO()
                    converted.save(buf, format="PNG")
                    buf.seek(0)
                    st.success("Conversion complete!")
                    st.download_button(
                        label="📥 Download PNG",
                        data=buf,
                        file_name=f"converted_{os.path.splitext(uploaded_file.name)[0]}.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error converting image: {e}")
            st.image(uploaded_file, caption="Original Image", use_container_width=True)

    # ========== BATCH PROCESSING MODE ==========
    else:
        st.subheader("📚 Batch Conversion")
        st.markdown("Upload multiple images. They will be converted to PNG (same size) and zipped.")

        # Session state for batch
        if "batch_convert_files" not in st.session_state:
            st.session_state.batch_convert_files = []
        if "batch_convert_cursor" not in st.session_state:
            st.session_state.batch_convert_cursor = 0
        if "batch_convert_total" not in st.session_state:
            st.session_state.batch_convert_total = 0
        if "batch_convert_tempdir" not in st.session_state:
            st.session_state.batch_convert_tempdir = None
        if "batch_convert_start_time" not in st.session_state:
            st.session_state.batch_convert_start_time = None

        uploaded_files = st.file_uploader(
            "Choose images (multiple allowed)",
            type=["avif", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
            accept_multiple_files=True,
            key="batch_convert_uploader"
        )
        if uploaded_files:
            st.session_state.batch_convert_files = uploaded_files
            st.write(f"**{len(uploaded_files)} images loaded**")

        # Start batch button
        if st.button("🚀 Start Batch Conversion", use_container_width=True,
                     disabled=not st.session_state.batch_convert_files):
            st.session_state.batch_convert_cursor = 0
            st.session_state.batch_convert_total = len(st.session_state.batch_convert_files)
            st.session_state.batch_convert_tempdir = tempfile.mkdtemp()
            st.session_state.batch_convert_start_time = time.time()
            st.rerun()

        # Process 1 image per rerun (smooth progress)
        if st.session_state.batch_convert_cursor < st.session_state.batch_convert_total:
            files = st.session_state.batch_convert_files
            total = st.session_state.batch_convert_total
            idx = st.session_state.batch_convert_cursor
            temp_dir = st.session_state.batch_convert_tempdir

            # Progress bar with ETA
            progress_text = f"Processing {idx+1} of {total}..."
            progress_bar = st.progress(idx / total if total else 0, text=progress_text)
            if idx > 0 and st.session_state.batch_convert_start_time:
                elapsed = time.time() - st.session_state.batch_convert_start_time
                sec_per_img = elapsed / idx
                remaining = (total - idx) * sec_per_img
                eta_min = int(remaining // 60)
                eta_sec = int(remaining % 60)
                eta_str = f"ETA: {eta_min}m {eta_sec}s"
                if sec_per_img > 0:
                    eta_str += f" · ~{1.0/sec_per_img:.1f} it/s"
                progress_bar.progress(idx / total, text=f"{progress_text} | {eta_str}")

            # Process the current file
            file_obj = files[idx]
            try:
                img = Image.open(file_obj)
                converted = convert_to_png(img)
                out_name = f"{os.path.splitext(file_obj.name)[0]}.png"
                converted.save(os.path.join(temp_dir, out_name), "PNG")
            except Exception as e:
                st.error(f"Error processing {file_obj.name}: {e}")

            # Advance cursor
            st.session_state.batch_convert_cursor = idx + 1

            if st.session_state.batch_convert_cursor >= total:
                # Finished – create ZIP
                zip_path = os.path.join(temp_dir, "converted_images.zip")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in os.listdir(temp_dir):
                        if f.endswith(".png"):
                            zf.write(os.path.join(temp_dir, f), f)
                with open(zip_path, "rb") as f:
                    st.success(f"✅ Batch complete! {total} images processed.")
                    st.download_button(
                        label="📥 Download ZIP",
                        data=f,
                        file_name="converted_images.zip",
                        mime="application/zip"
                    )
                # Clean up state
                st.session_state.batch_convert_files = []
                st.session_state.batch_convert_cursor = 0
                st.session_state.batch_convert_total = 0
                st.session_state.batch_convert_tempdir = None
                st.session_state.batch_convert_start_time = None
            else:
                st.rerun()

        elif st.session_state.batch_convert_total > 0:
            st.success("All images processed! Download the ZIP above.")



# ======================================================================
# TAB 10: Smart Resize & Crop for Training
# ======================================================================

elif current_tab == "📐 Smart Resize & Crop":
    import time, os, shutil

    st.header("📐 Smart Resize & Crop")
    st.markdown(
        "Downscale and center‑crop images to exact training resolutions:\n"
        "- **Square** (1024×1024)\n"
        "- **Portrait** (832×1248)\n"
        "- **Landscape** (1344×768)\n\n"
        "Images are first resized so the shorter side covers the target, then center‑cropped. "
        "Only PNG images are processed. The original files are never modified.\n\n"
        "💡Crop manually to remove unwanted watermarks / Upscale your images manually if needed before running this.  "
    )

    # ---- Source folder ----
    st.subheader("📁 Source folder")
    use_current_dataset = st.checkbox("Use current dataset folder", value=True)
    if use_current_dataset:
        input_dir = st.session_state.dataset_dir if folder_valid else ""
        if not input_dir:
            st.warning("No dataset loaded. Please load a dataset first or uncheck the box.")
            st.stop()
        st.info(f"Input folder: `{input_dir}`")
    else:
        input_dir = st.text_input("Enter the folder path containing PNG images:", value="")

    # ---- Output folder ----
    st.subheader("📤 Output folder")
    output_dir = st.text_input(
        "Output folder (will be created if it doesn't exist):",
        value=os.path.join(input_dir, "resized_cropped") if input_dir else ""
    )

    # ---- Target resolutions (fixed, but displayed for info) ----
    st.subheader("🎯 Target Resolutions")
    st.markdown(
        """
        | Orientation   | Resolution   |
        |---------------|--------------|
        | Square        | 1024 × 1024  |
        | Portrait      | 832 × 1248   |
        | Landscape     | 1344 × 768   |
        """
    )

    # ---- Session state for batch processing ----
    if "smart_resize_files" not in st.session_state:
        st.session_state.smart_resize_files = []
    if "smart_resize_cursor" not in st.session_state:
        st.session_state.smart_resize_cursor = 0
    if "smart_resize_total" not in st.session_state:
        st.session_state.smart_resize_total = 0
    if "smart_resize_start_time" not in st.session_state:
        st.session_state.smart_resize_start_time = None

    # ---- Start button ----
    can_start = bool(input_dir) and os.path.isdir(input_dir) and bool(output_dir)
    if st.button("🚀 Start Processing", use_container_width=True, disabled=not can_start):
        # Gather PNG files from the input folder
        all_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
        if not all_files:
            st.warning("No PNG images found in the input folder.")
            st.stop()
        st.session_state.smart_resize_files = all_files
        st.session_state.smart_resize_cursor = 0
        st.session_state.smart_resize_total = len(all_files)
        st.session_state.smart_resize_start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)
        st.rerun()

    # ---- Process one image per rerun ----
    if st.session_state.smart_resize_cursor < st.session_state.smart_resize_total:
        files = st.session_state.smart_resize_files
        total = st.session_state.smart_resize_total
        idx = st.session_state.smart_resize_cursor

        # Resolutions
        SQUARE = (1024, 1024)
        PORTRAIT = (832, 1248)
        LANDSCAPE = (1344, 768)

        def smart_resize_and_crop(image_path, target_res):
            with Image.open(image_path) as img:
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                orig_w, orig_h = img.size
                target_w, target_h = target_res
                scale_w = target_w / orig_w
                scale_h = target_h / orig_h
                scale = max(scale_w, scale_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                left = (new_w - target_w) // 2
                top = (new_h - target_h) // 2
                return resized.crop((left, top, left + target_w, top + target_h))

        # Progress bar with ETA
        progress_text = f"Processing {idx+1} of {total}..."
        progress_bar = st.progress(idx / total if total else 0, text=progress_text)
        if idx > 0 and st.session_state.smart_resize_start_time:
            elapsed = time.time() - st.session_state.smart_resize_start_time
            sec_per_img = elapsed / idx
            remaining = (total - idx) * sec_per_img
            eta_min = int(remaining // 60)
            eta_sec = int(remaining % 60)
            eta_str = f"ETA: {eta_min}m {eta_sec}s"
            if sec_per_img > 0:
                eta_str += f" · ~{1.0/sec_per_img:.1f} it/s"
            progress_bar.progress(idx / total, text=f"{progress_text} | {eta_str}")

        # Process the current file
        fname = files[idx]
        img_path = os.path.join(input_dir, fname)
        try:
            with Image.open(img_path) as img:
                w, h = img.size
            aspect = w / h
            if 0.98 <= aspect <= 1.02:
                target = SQUARE
            elif w > h:
                target = LANDSCAPE
            else:
                target = PORTRAIT
            final_img = smart_resize_and_crop(img_path, target)
            final_img.save(os.path.join(output_dir, fname), "PNG")
        except Exception as e:
            st.error(f"Error processing {fname}: {e}")

        # Advance cursor
        st.session_state.smart_resize_cursor = idx + 1

        if st.session_state.smart_resize_cursor >= total:
            duration = time.time() - st.session_state.smart_resize_start_time if st.session_state.smart_resize_start_time else 0
            st.success(f"✅ Processing complete! {total} images saved to `{output_dir}` in {duration:.1f}s.")
            # Reset state
            st.session_state.smart_resize_files = []
            st.session_state.smart_resize_cursor = 0
            st.session_state.smart_resize_total = 0
            st.session_state.smart_resize_start_time = None
        else:
            st.rerun()

    elif st.session_state.smart_resize_total > 0:
        st.success(f"All {st.session_state.smart_resize_total} images processed!")



# ======================================================================
# TAB 11: Batch Rename (rename in place or copy to new folder)
# ======================================================================

elif current_tab == "📝 Batch Rename":
    import shutil

    st.header("📝 Batch Rename")
    st.markdown(
        "Rename all PNG images (and their matching `.txt` files) using a sequential pattern. "
        "You can either rename the originals directly, or copy them to a new folder with the new names."
    )

    # ---- Source folder ----
    st.subheader("📁 Source folder")
    use_current_dataset = st.checkbox("Use current dataset folder", value=True)
    if use_current_dataset:
        folder = st.session_state.dataset_dir if folder_valid else ""
        if not folder:
            st.warning("No dataset loaded. Please load a dataset first or uncheck the box.")
            st.stop()
    else:
        folder = st.text_input("Enter the folder path:", value="")

    if not folder or not os.path.isdir(folder):
        if folder:
            st.warning("Folder not found.")
        st.stop()

    # ---- Rename settings ----
    st.subheader("⚙️ Rename Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        prefix = st.text_input("Prefix", value="image")
    with col2:
        start_num = st.number_input("Start number", min_value=1, value=1, step=1)
    with col3:
        padding = st.number_input("Zero padding", min_value=1, value=3, step=1)

    # ---- Mode selection ----
    st.subheader("📂 Output Mode")
    mode = st.radio(
        "Mode:",
        ["Rename in place", "Copy to new folder"],
        horizontal=True,
        help="Rename in place modifies the original files. Copy to new folder creates renamed copies."
    )

    dest_folder = None
    if mode == "Copy to new folder":
        default_dest = os.path.join(folder, "renamed")
        dest_folder = st.text_input("Destination folder (will be created if needed):", value=default_dest)
        if not dest_folder:
            st.stop()

    # ---- Scan folder ----
    png_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
    if not png_files:
        st.info("No PNG images found in this folder.")
        st.stop()

    # ---- Preview ----
    st.subheader("🔍 Preview (first 10 files)")
    preview_data = []
    for i, fname in enumerate(png_files[:10]):
        new_idx = start_num + i
        new_name = f"{prefix}_{new_idx:0{padding}d}.png"
        preview_data.append((fname, new_name))
    if preview_data:
        import pandas as pd
        df = pd.DataFrame(preview_data, columns=["Current Name", "New Name"])
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.write(f"**{len(png_files)} files** will be processed.")

    # ---- Apply rename / copy ----
    button_label = "🚀 Rename All Files" if mode == "Rename in place" else "🚀 Copy with New Names"
    if st.button(button_label, use_container_width=True):
        if len(png_files) == 0:
            st.warning("No files to process.")
        else:
            if mode == "Copy to new folder":
                os.makedirs(dest_folder, exist_ok=True)

            success = 0
            errors = []
            for i, fname in enumerate(png_files):
                new_idx = start_num + i
                new_name = f"{prefix}_{new_idx:0{padding}d}.png"
                src_img = os.path.join(folder, fname)
                dst_img = os.path.join(dest_folder if mode == "Copy to new folder" else folder, new_name)
                src_txt = os.path.splitext(src_img)[0] + ".txt"
                dst_txt = os.path.join(dest_folder if mode == "Copy to new folder" else folder,
                                       f"{prefix}_{new_idx:0{padding}d}.txt")

                try:
                    if mode == "Copy to new folder":
                        shutil.copy2(src_img, dst_img)
                        if os.path.exists(src_txt):
                            shutil.copy2(src_txt, dst_txt)
                    else:
                        os.rename(src_img, dst_img)
                        if os.path.exists(src_txt):
                            os.rename(src_txt, dst_txt)
                    success += 1
                except Exception as e:
                    errors.append((fname, str(e)))

            if errors:
                st.error(f"{len(errors)} errors occurred.")
                for fname, err in errors:
                    st.write(f"- {fname}: {err}")
            else:
                msg = "renamed" if mode == "Rename in place" else "copied"
                st.success(f"✅ Successfully {msg} {success} files!")
                st.cache_data.clear()
                st.rerun()



# ======================================================================
# TAB 12: EXIF Viewer & Stripper
# ======================================================================

elif current_tab == "🏷️ EXIF Viewer & Stripper":
    import piexif
    from PIL import Image, ExifTags

    st.header("🏷️ EXIF Viewer & Stripper")
    st.markdown(
        "View hidden camera metadata (EXIF) from your images, or strip EXIF data "
        "from a single image or an entire folder for privacy / consistency."
    )

    # ---- Mode selection ----
    mode = st.radio("Mode:", ["View EXIF", "Strip EXIF"], horizontal=True)

    # ========== VIEW EXIF MODE ==========
    if mode == "View EXIF":
        st.subheader("🔍 View EXIF Data")
        source = st.radio("Image source:", ["From dataset", "Upload new"], horizontal=True, key="exif_view_source")

        image_to_inspect = None
        image_name = None

        if source == "From dataset":
            if not folder_valid or not raw_items:
                st.warning("Load a dataset first.")
                st.stop()
            img_names = [item["img"] for item in raw_items]
            selected = st.selectbox("Choose an image:", img_names, key="exif_dataset_select")
            if selected:
                image_to_inspect = os.path.join(dataset_path, selected)
                image_name = selected
        else:
            uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "tiff"],
                                        key="exif_uploader")
            if uploaded:
                # Save to a temporary file so piexif can read it
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
                    tmp.write(uploaded.getvalue())
                    image_to_inspect = tmp.name
                image_name = uploaded.name

        if image_to_inspect:
            try:
                img = Image.open(image_to_inspect)
                exif_data = img.getexif()
                if exif_data:
                    # Convert numeric tags to readable names
                    readable = {}
                    for tag_id, value in exif_data.items():
                        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                        # Decode bytes if necessary
                        if isinstance(value, bytes):
                            try:
                                value = value.decode()
                            except:
                                value = value.hex()
                        readable[tag_name] = str(value)

                    # Also try to get GPS info
                    gps_data = {}
                    if hasattr(exif_data, "get_ifd"):
                        gps_ifd = exif_data.get_ifd(ExifTags.IFD.GPSInfo)
                        if gps_ifd:
                            for tag_id, value in gps_ifd.items():
                                tag_name = ExifTags.GPSTAGS.get(tag_id, tag_id)
                                if isinstance(value, bytes):
                                    try:
                                        value = value.decode()
                                    except:
                                        value = value.hex()
                                gps_data[tag_name] = str(value)

                    # Build a clean table
                    st.subheader(f"📷 {image_name}")
                    st.image(image_to_inspect, use_container_width=True, width=400)

                    st.write("**Camera & Settings**")
                    # Filter commonly interesting tags
                    interesting = ["Make", "Model", "DateTime", "ExposureTime", "FNumber",
                                   "ISOSpeedRatings", "FocalLength", "LensModel", "Software"]
                    rows = []
                    for tag in interesting:
                        if tag in readable:
                            rows.append({"Tag": tag, "Value": readable[tag]})
                    # Add any other tags not in interesting
                    for tag, val in readable.items():
                        if tag not in interesting and not tag.startswith("GPS"):
                            rows.append({"Tag": tag, "Value": val})

                    if rows:
                        import pandas as pd
                        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                    if gps_data:
                        st.write("**📍 GPS Information**")
                        st.dataframe(pd.DataFrame(
                            [{"Tag": k, "Value": v} for k, v in gps_data.items()]
                        ), hide_index=True, use_container_width=True)
                else:
                    st.info("No EXIF data found in this image.")
            except Exception as e:
                st.error(f"Error reading EXIF: {e}")
            finally:
                if source == "Upload new" and image_to_inspect and os.path.exists(image_to_inspect):
                    try:
                        os.unlink(image_to_inspect)
                    except:
                        pass

    # ========== STRIP EXIF MODE ==========
    else:
        st.subheader("🧹 Strip EXIF Data")
        st.markdown(
            "Remove EXIF metadata from images. This overwrites the original files! "
            "Make sure you have a backup if needed."
        )

        target_folder = None
        use_dataset = st.checkbox("Use current dataset folder", value=True, key="exif_strip_dataset")
        if use_dataset:
            if not folder_valid:
                st.warning("No dataset loaded.")
                st.stop()
            target_folder = dataset_path
            st.info(f"Processing folder: `{target_folder}`")
        else:
            target_folder = st.text_input("Enter folder path:", key="exif_strip_folder")
            if not target_folder:
                st.stop()

        if not os.path.isdir(target_folder):
            st.error("Invalid folder path.")
            st.stop()

        # Scan for supported images (JPEG, TIFF, WebP – PNG often has no EXIF, but we'll include anyway)
        exts = (".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif")
        all_files = sorted([f for f in os.listdir(target_folder) if f.lower().endswith(exts)])
        if not all_files:
            st.info("No supported images found in this folder.")
            st.stop()

        st.write(f"**{len(all_files)} images** will be processed.")

        # Session state for batch stripping
        if "strip_exif_files" not in st.session_state:
            st.session_state.strip_exif_files = []
        if "strip_exif_cursor" not in st.session_state:
            st.session_state.strip_exif_cursor = 0
        if "strip_exif_total" not in st.session_state:
            st.session_state.strip_exif_total = 0
        if "strip_exif_start_time" not in st.session_state:
            st.session_state.strip_exif_start_time = None

        if st.button("🧹 Strip EXIF from All Images", use_container_width=True):
            st.session_state.strip_exif_files = all_files
            st.session_state.strip_exif_cursor = 0
            st.session_state.strip_exif_total = len(all_files)
            st.session_state.strip_exif_start_time = time.time()
            st.rerun()

        # Process one image per rerun
        if st.session_state.strip_exif_cursor < st.session_state.strip_exif_total:
            files = st.session_state.strip_exif_files
            total = st.session_state.strip_exif_total
            idx = st.session_state.strip_exif_cursor

            progress_text = f"Processing {idx+1} of {total}..."
            progress_bar = st.progress(idx / total if total else 0, text=progress_text)
            if idx > 0 and st.session_state.strip_exif_start_time:
                elapsed = time.time() - st.session_state.strip_exif_start_time
                sec_per_img = elapsed / idx
                remaining = (total - idx) * sec_per_img
                eta_min = int(remaining // 60)
                eta_sec = int(remaining % 60)
                eta_str = f"ETA: {eta_min}m {eta_sec}s"
                if sec_per_img > 0:
                    eta_str += f" · ~{1.0/sec_per_img:.1f} it/s"
                progress_bar.progress(idx / total, text=f"{progress_text} | {eta_str}")

            fname = files[idx]
            fpath = os.path.join(target_folder, fname)
            try:
                # Use piexif to remove EXIF – works on JPEG/TIFF; for PNG we can just clear info
                ext = os.path.splitext(fname)[1].lower()
                if ext in (".jpg", ".jpeg", ".tiff", ".tif"):
                    piexif.remove(fpath)
                else:
                    # For PNG, WebP – Pillow can remove EXIF by re-saving without exif
                    img = Image.open(fpath)
                    data = list(img.getdata())
                    mode = img.mode
                    # Re-save without any info
                    img_no_exif = Image.new(mode, img.size)
                    img_no_exif.putdata(data)
                    img_no_exif.save(fpath, format=img.format)
            except Exception as e:
                st.error(f"Error stripping {fname}: {e}")

            st.session_state.strip_exif_cursor = idx + 1

            if st.session_state.strip_exif_cursor >= total:
                duration = time.time() - st.session_state.strip_exif_start_time if st.session_state.strip_exif_start_time else 0
                st.success(f"✅ EXIF stripped from {total} images in {duration:.1f}s.")
                st.cache_data.clear()
                # Reset state
                st.session_state.strip_exif_files = []
                st.session_state.strip_exif_cursor = 0
                st.session_state.strip_exif_total = 0
                st.session_state.strip_exif_start_time = None
            else:
                st.rerun()

        elif st.session_state.strip_exif_total > 0:
            st.success(f"All {st.session_state.strip_exif_total} images processed!")



# ======================================================================
# TAB 13: Training Run Tracker
# ======================================================================

elif current_tab == "ℹ️ Training Run Tracker":
    import json, datetime

    st.header("ℹ️ Training Run Tracker")
    st.markdown(
        "Log your LoRA / model training runs. Record the dataset used, hyperparameters, "
        "and your own notes. All entries are saved in `training_runs.json` inside the app folder."
    )

    # ---- File to store runs ----
    RUNS_FILE = os.path.join(APP_DIR, "training_runs.json")

    # ---- Load existing runs ----
    def load_runs():
        if os.path.exists(RUNS_FILE):
            try:
                with open(RUNS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_runs(runs):
        with open(RUNS_FILE, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2, ensure_ascii=False)

    # ---- Add a new run ----
    st.subheader("➕ Log a New Training Run")
    with st.form("new_run_form"):
        col1, col2 = st.columns(2)
        with col1:
            run_name = st.text_input("Run name / identifier *", placeholder="e.g., joycaption-v3-epoch20")
            dataset_path_logged = st.text_input("Dataset path *", value=st.session_state.dataset_dir,
                                                help="Path to the dataset folder used for this run.")
        with col2:
            model_name = st.text_input("Model / LoRA name", placeholder="e.g., joycaption-beta-one")
            date = st.date_input("Date", value=datetime.date.today())

        st.markdown("---")
        st.markdown("**Hyperparameters (optional)**")
        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        with col_h1:
            epochs = st.number_input("Epochs", min_value=1, value=10, step=1)
        with col_h2:
            batch_size = st.number_input("Batch size", min_value=1, value=4, step=1)
        with col_h3:
            learning_rate = st.text_input("Learning rate", value="1e-4", placeholder="e.g., 1e-4")
        with col_h4:
            resolution = st.text_input("Resolution", value="1024x1024", placeholder="e.g., 1024x1024")

        notes = st.text_area("Notes / observations", placeholder="How did the model perform? Any issues?")

        submitted = st.form_submit_button("💾 Save Run")
        if submitted:
            if not run_name or not dataset_path_logged:
                st.error("Run name and dataset path are required.")
            else:
                runs = load_runs()
                new_run = {
                    "run_name": run_name,
                    "dataset_path": dataset_path_logged,
                    "model_name": model_name,
                    "date": date.isoformat(),
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "resolution": resolution,
                    "notes": notes,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                runs.append(new_run)
                save_runs(runs)
                st.success(f"Run '{run_name}' saved!")
                st.rerun()

    # ---- View all runs ----
    st.markdown("---")
    st.subheader("📚 All Training Runs")
    runs = load_runs()
    if not runs:
        st.info("No training runs logged yet. Use the form above to add one.")
    else:
        # Show runs in reverse chronological order
        runs_sorted = sorted(runs, key=lambda r: r.get("timestamp", ""), reverse=True)
        for i, run in enumerate(runs_sorted):
            with st.expander(f"📌 {run.get('run_name', 'Unnamed')} — {run.get('date', '?')}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Dataset:** `{run.get('dataset_path', '')}`")
                    st.write(f"**Model:** {run.get('model_name', 'N/A')}")
                with col_b:
                    st.write(f"**Epochs:** {run.get('epochs', '')} | **Batch:** {run.get('batch_size', '')}")
                    st.write(f"**LR:** {run.get('learning_rate', '')} | **Res:** {run.get('resolution', '')}")
                if run.get("notes"):
                    st.write(f"**Notes:** {run['notes']}")

                # Delete button for each run
                if st.button("🗑️ Delete this run", key=f"delete_run_{i}"):
                    # Reload runs to be safe, then filter out this one
                    current_runs = load_runs()
                    current_runs = [r for r in current_runs if r.get("timestamp") != run.get("timestamp")]
                    save_runs(current_runs)
                    st.success("Run deleted.")
                    st.rerun()

        # Export all runs as JSON
        if st.button("📥 Export as JSON"):
            import io
            buf = io.BytesIO()
            buf.write(json.dumps(runs, indent=2, ensure_ascii=False).encode("utf-8"))
            buf.seek(0)
            st.download_button("Download training_runs.json", buf, file_name="training_runs.json", mime="application/json")



# ======================================================================            
# TAB 14: Fine‑Tuning Launcher
# ======================================================================

elif current_tab == "🚀 Fine‑Tuning Launcher":
    import shlex

    st.header("🚀 Fine‑Tuning Launcher")
    st.markdown(
        "Generate a ready‑to‑use training command for your favourite LoRA / Dreambooth trainer. "
        "The command can be copied and pasted into your terminal. "
        "Optionally log the run in the **Training Run Tracker**."
    )

    # ---- Dataset selection ----
    st.subheader("📁 Dataset")
    use_current_dataset = st.checkbox("Use current dataset folder", value=True)
    if use_current_dataset:
        if not folder_valid:
            st.warning("No dataset loaded. Please load a dataset first or uncheck the box.")
            st.stop()
        dataset_path_log = st.session_state.dataset_dir
    else:
        dataset_path_log = st.text_input("Enter dataset folder path:", value="")

    if not dataset_path_log or not os.path.isdir(dataset_path_log):
        st.info("Please provide a valid dataset folder.")
        st.stop()

    # ---- Base model ----
    st.subheader("🧠 Base Model")
    model_id = st.text_input(
        "Hugging Face model ID (or path to local model)",
        value="stabilityai/stable-diffusion-2-1",
        help="The pre‑trained model you want to fine‑tune."
    )

    # ---- Training parameters ----
    st.subheader("⚙️ Training Parameters")
    col1, col2 = st.columns(2)
    with col1:
        output_dir = st.text_input("Output directory", value=os.path.join(dataset_path_log, "lora_model"))
        lora_rank = st.number_input("LoRA rank", min_value=1, value=16, step=1)
        learning_rate = st.text_input("Learning rate", value="1e-4")
    with col2:
        num_epochs = st.number_input("Epochs", min_value=1, value=10, step=1)
        batch_size = st.number_input("Batch size", min_value=1, value=4, step=1)
        resolution = st.text_input("Resolution", value="1024", help="Training resolution (e.g., 1024 for square)")

    # ---- Trainer selection ----
    st.subheader("🛠️ Trainer")
    trainer = st.selectbox(
        "Select trainer / script type:",
        [
            "Diffusers (text-to-image LoRA)",
            "Kohya SD-Scripts (LoRA)",
            "Custom command",
        ]
    )

    # ---- Generate command ----
    if trainer == "Diffusers (text-to-image LoRA)":
        # Assumes the user has the Diffusers example script
        script = "train_text_to_image_lora.py"
        command = (
            f"accelerate launch {script} "
            f"--pretrained_model_name_or_path={shlex.quote(model_id)} "
            f"--dataset_name={shlex.quote(dataset_path_log)} "
            f"--output_dir={shlex.quote(output_dir)} "
            f"--resolution={resolution} "
            f"--train_batch_size={batch_size} "
            f"--num_train_epochs={num_epochs} "
            f"--learning_rate={learning_rate} "
            f"--rank={lora_rank}"
        )
    elif trainer == "Kohya SD-Scripts (LoRA)":
        command = (
            f"accelerate launch --num_cpu_threads_per_process=2 train_network.py "
            f"--pretrained_model_name_or_path={shlex.quote(model_id)} "
            f"--train_data_dir={shlex.quote(dataset_path_log)} "
            f"--output_dir={shlex.quote(output_dir)} "
            f"--resolution={resolution} "
            f"--train_batch_size={batch_size} "
            f"--max_train_epochs={num_epochs} "
            f"--learning_rate={learning_rate} "
            f"--network_module=networks.lora "
            f"--network_dim={lora_rank}"
        )
    else:
        # Custom command – let the user edit freely
        command = st.text_area(
            "Enter your training command:",
            value="accelerate launch train.py --help",
            height=100
        )

    # ---- Display command ----
    st.subheader("📋 Generated Command")
    st.code(command, language="bash")
    st.caption("Copy the command above and run it in your terminal. Make sure the required script is in your working directory.")

    # ---- Save as script file ----
    if st.button("💾 Save as .bat / .sh file"):
        script_dir = os.path.join(dataset_path_log, "training_scripts")
        os.makedirs(script_dir, exist_ok=True)
        script_name = f"train_{time.strftime('%Y%m%d_%H%M%S')}.bat" if os.name == "nt" else ".sh"
        script_path = os.path.join(script_dir, script_name)
        with open(script_path, "w") as f:
            if os.name == "nt":
                f.write("@echo off\n")
            else:
                f.write("#!/bin/bash\n")
            f.write(command + "\n")
        st.success(f"Script saved to `{script_path}`")

    # ---- Log to Training Run Tracker ----
    st.subheader("📋 Log this run")
    run_name = st.text_input("Run name (optional)", placeholder="e.g., lora-test-1")
    if st.button("📝 Log run in Training Run Tracker"):
        runs_file = os.path.join(APP_DIR, "training_runs.json")
        run = {
            "run_name": run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}",
            "dataset_path": dataset_path_log,
            "model_name": model_id,
            "date": datetime.date.today().isoformat(),
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "resolution": resolution,
            "notes": f"LoRA rank {lora_rank}. Trainer: {trainer}.",
            "timestamp": datetime.datetime.now().isoformat()
        }
        # Load existing runs, append, save
        runs = []
        if os.path.exists(runs_file):
            with open(runs_file, "r", encoding="utf-8") as f:
                runs = json.load(f)
        runs.append(run)
        with open(runs_file, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2, ensure_ascii=False)
        st.success(f"Run '{run['run_name']}' logged!")

    st.markdown("---")
    st.info(
        "💡 **Tip:** Install `accelerate` and download the trainer script from the official Hugging Face / Kohya repositories. "
        "The command assumes the script is in your current working directory."
    )



# ======================================================================
# TAB 15: Dataset Comparator
# ======================================================================

elif current_tab == "✨ Dataset Comparator":
    st.header("✨ Dataset Comparator")
    st.markdown(
        "Compare two dataset folders side‑by‑side. "
        "Find overlapping images, identical copies, caption differences, and quality statistics."
    )

    # ---- Folder selection ----
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📁 Dataset A")
        use_current_a = st.checkbox("Use current dataset", value=True, key="use_current_a")
        if use_current_a:
            if not folder_valid:
                st.warning("No dataset loaded.")
                st.stop()
            path_a = st.session_state.dataset_dir
        else:
            path_a = st.text_input("Folder A path", value="")
    with col2:
        st.subheader("📁 Dataset B")
        path_b = st.text_input("Folder B path", value="")

    if not path_a or not os.path.isdir(path_a) or not path_b or not os.path.isdir(path_b):
        st.info("Please provide two valid dataset folders.")
        st.stop()

    # ---- Scan folders ----
    @st.cache_data(show_spinner="Scanning folders…")
    def scan_folder(folder):
        files = {}
        for f in os.listdir(folder):
            if f.lower().endswith('.png'):
                full = os.path.join(folder, f)
                txt = os.path.splitext(full)[0] + ".txt"
                caption = ""
                if os.path.exists(txt):
                    with open(txt, "r", encoding="utf-8") as fc:
                        caption = fc.read().strip()
                files[f] = {"path": full, "caption": caption}
        return files

    dataset_a = scan_folder(path_a)
    dataset_b = scan_folder(path_b)

    names_a = set(dataset_a.keys())
    names_b = set(dataset_b.keys())
    common = sorted(names_a & names_b)
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)

    # ---- Summary ----
    st.subheader("📋 Overview")
    col_a, col_b, col_common = st.columns(3)
    col_a.metric("Only in A", len(only_a))
    col_b.metric("Only in B", len(only_b))
    col_common.metric("In both", len(common))

    # ---- Identical images (by perceptual hash) ----
    st.subheader("🔍 Identical Copies (same perceptual hash)")
    if st.button("🔎 Find identical images"):
        with st.spinner("Computing perceptual hashes…"):
            import imagehash
            from PIL import Image

            def get_phashes(files_dict, folder):
                hashes = {}
                for name, info in files_dict.items():
                    try:
                        img = Image.open(info["path"])
                        phash = str(imagehash.phash(img))
                        if phash not in hashes:
                            hashes[phash] = []
                        hashes[phash].append(name)
                    except:
                        pass
                return hashes

            hashes_a = get_phashes(dataset_a, path_a)
            hashes_b = get_phashes(dataset_b, path_b)

            identical = []
            for h, files in hashes_a.items():
                if h in hashes_b:
                    for fa in files:
                        for fb in hashes_b[h]:
                            identical.append((fa, fb))

            if identical:
                st.write(f"Found **{len(identical)}** identical pairs.")
                for fa, fb in identical[:20]:
                    st.write(f"- A: `{fa}`  ↔  B: `{fb}`")
                if len(identical) > 20:
                    st.caption(f"… and {len(identical)-20} more.")
            else:
                st.success("No identical copies found.")
            st.session_state.identical_pairs = identical
    else:
        identical = st.session_state.get("identical_pairs", [])

    # ---- Caption differences ----
    st.subheader("📝 Caption Differences")
    if common:
        diff_count = 0
        for name in common[:100]:   # limit to 100 for performance
            cap_a = dataset_a[name]["caption"]
            cap_b = dataset_b[name]["caption"]
            if cap_a != cap_b:
                diff_count += 1
                with st.expander(f"✏️ {name}"):
                    col_left, col_right = st.columns(2)
                    with col_left:
                        st.markdown("**A**")
                        st.caption(cap_a[:300])
                    with col_right:
                        st.markdown("**B**")
                        st.caption(cap_b[:300])
        if diff_count == 0:
            st.success("No caption differences found among overlapping images.")
        else:
            st.write(f"Found **{diff_count}** caption differences (showing up to 100).")
    else:
        st.info("No overlapping images to compare captions.")

    # ---- Quality comparison ----
    st.subheader("📈 Quality Metrics Comparison")
    if st.button("📊 Compare Quality Metrics"):
        with st.spinner("Computing quality metrics…"):
            metrics_a, _ = compute_all_metrics(path_a)
            metrics_b, _ = compute_all_metrics(path_b)

            # Average scores
            avg_a = np.mean([m['overall_quality'] for m in metrics_a.values()]) if metrics_a else 0
            avg_b = np.mean([m['overall_quality'] for m in metrics_b.values()]) if metrics_b else 0

            col1, col2 = st.columns(2)
            col1.metric("Avg Quality A", f"{avg_a:.1f}")
            col2.metric("Avg Quality B", f"{avg_b:.1f}")

            # Scatter plot of shared images
            shared = []
            for name in common:
                if name in metrics_a and name in metrics_b:
                    shared.append({
                        "image": name,
                        "quality_a": metrics_a[name]['overall_quality'],
                        "quality_b": metrics_b[name]['overall_quality'],
                        "sharpness_a": metrics_a[name]['sharpness'],
                        "sharpness_b": metrics_b[name]['sharpness'],
                    })
            if shared:
                import pandas as pd
                import plotly.express as px
                df = pd.DataFrame(shared)
                fig = px.scatter(
                    df, x="quality_a", y="quality_b",
                    hover_data=["image"],
                    title="Quality: A vs B (shared images)",
                    labels={"quality_a": "Quality A", "quality_b": "Quality B"}
                )
                fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(dash="dash", color="gray"))
                st.plotly_chart(fig, use_container_width=True)

                # Sharpness comparison
                fig2 = px.scatter(
                    df, x="sharpness_a", y="sharpness_b",
                    hover_data=["image"],
                    title="Sharpness: A vs B (shared images)",
                    labels={"sharpness_a": "Sharpness A", "sharpness_b": "Sharpness B"}
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No shared images with quality metrics.")
            st.session_state.comp_metrics_a = metrics_a
            st.session_state.comp_metrics_b = metrics_b
    else:
        metrics_a = st.session_state.get("comp_metrics_a", {})
        metrics_b = st.session_state.get("comp_metrics_b", {})



# ======================================================================
# TAB 16: Prompt Generator
# ======================================================================

elif current_tab == "📜 Prompt Generator":
    import random

    st.header("📜 Prompt Generator")
    st.markdown(
        "Create detailed, varied prompts using built‑in descriptions. "
        "No dataset needed – just pick your concepts and generate."
    )

    # ---- Concept keywords (your Gender list) ----
    CONCEPT_KEYWORDS = {
        "Gender": {
            "man":   ["man", "male", "guy"],
            "woman": ["woman", "female", "lady"],
        },
        "Clothing & Colors": {
            "dress":     ["dress", "dresses"],
            "shirt":     ["shirt", "blouse", "t-shirt", "top"],
            "jeans":     ["jeans", "denim"],
            "skirt":     ["skirt"],
            "jacket":    ["jacket", "coat", "blazer"],
            "sweater":   ["sweater", "jumper", "cardigan"],
            "shorts":    ["shorts"],
            "black":     ["black", "ebony"],
            "white":     ["white", "ivory"],
            "red":       ["red", "crimson", "ruby"],
            "blue":      ["blue", "navy", "teal"],
            "green":     ["green", "olive", "emerald"],
            "pink":      ["pink"],
            "yellow":    ["yellow"],
            "purple":    ["purple", "violet", "lavender"],
            "brown":     ["brown"],
            "grey":      ["grey", "gray"]
        },
        "Background Environment": {
            "outdoor": ["outdoor", "outside", "garden", "park", "street"],
            "indoor":  ["indoor", "inside", "room", "studio", "apartment"],
            "urban":   ["urban", "city", "alley", "downtown", "metropolis"],
            "nature":  ["nature", "forest", "woods", "mountain", "beach", "lake", "field"],
            "plain":   ["plain background", "solid background", "white background", "black background"]
        },
        "Lighting Type": {
            "natural light": ["natural light", "daylight", "sunlight", "outdoor light"],
            "studio light":  ["studio light", "artificial light", "flash", "continuous light"],
            "golden hour":   ["golden hour", "sunset light", "warm light", "magic hour"],
            "harsh light":   ["harsh light", "direct sunlight", "hard shadows"],
            "soft light":    ["soft light", "diffuse light", "softbox", "window light"]
        },
        "Photographic Style": {
            "natural":        ["natural", "realistic", "unprocessed", "documentary"],
            "cinematic":      ["cinematic", "film look", "movie", "anamorphic"],
            "vintage":        ["vintage", "retro", "old", "70s", "polaroid"],
            "high-contrast":  ["high contrast", "dramatic contrast", "chiaroscuro"],
            "black & white":  ["black and white", "b&w", "monochrome", "grayscale"],
            "soft":           ["soft", "dreamy", "pastel", "ethereal"],
            "candid":         ["candid", "unposed", "spontaneous", "snapshot", "street photography"],
        },
    }

    CONCEPT_MAP = {
        "subject":    "Gender",
        "clothing":   "Clothing & Colors",
        "background": "Background Environment",
        "lighting":   "Lighting Type",
        "style":      "Photographic Style",
    }

    def get_keywords(placeholder):
        concept = CONCEPT_MAP.get(placeholder)
        if not concept or concept not in CONCEPT_KEYWORDS:
            return ["none"]
        flat = []
        for kw_list in CONCEPT_KEYWORDS[concept].values():
            flat.extend(kw_list)
        return sorted(set(kw.strip() for kw in flat if kw.strip()))

    placeholder_keywords = {ph: get_keywords(ph) for ph in ["subject", "clothing", "background", "lighting", "style"]}

    # ---- Descriptive libraries ----
    HAIR_COLORS = ["black", "brown", "blonde", "red", "grey", "silver", "auburn", "chestnut", "raven", "honey", "copper"]
    HAIR_STYLES = ["long", "short", "wavy", "curly", "straight", "braided", "shoulder-length", "sleek", "voluminous", "silky"]
    EYE_COLORS  = ["brown", "blue", "green", "hazel", "grey", "amber", "deep brown", "piercing blue"]
    EXPRESSIONS = ["smiling warmly", "with a neutral expression", "looking thoughtful",
                   "grinning slightly", "with a calm gaze", "looking directly at the camera",
                   "with a slight smirk", "gazing into the distance", "with a gentle smile"]
    ADJECTIVES  = ["elegant", "stylish", "casual", "athletic", "graceful",
                   "confident", "mysterious", "radiant", "charming", "sophisticated", "carefree"]
    CLOTHING_COLORS = ["black", "white", "red", "blue", "green", "pink", "yellow", "purple", "brown", "grey",
                       "navy", "teal", "crimson", "olive", "beige", "ivory", "charcoal", "burgundy"]
    CLOTHING_STYLES = ["short", "long", "tight", "flowing", "casual", "formal", "elegant",
                       "sporty", "loose", "patterned", "vintage", "modern", "chic"]
    CLOTHING_ITEMS  = ["dress", "shirt", "blouse", "t-shirt", "top", "jeans", "skirt", "jacket", "coat",
                       "blazer", "sweater", "jumper", "cardigan", "shorts", "trousers", "suit"]
    # Items that are always plural – these never take "a/an"
    PLURAL_CLOTHING = {"jeans", "shorts", "trousers"}
    SCENE_DETAILS = ["cars passing by", "people walking in the distance", "trees swaying gently",
                     "soft shadows stretching across the ground", "a gentle breeze moving the hair",
                     "city lights twinkling", "leaves scattered on the ground"]
    WEATHERS = ["under a bright blue sky", "on a cloudy afternoon", "during golden hour",
                "in soft morning light", "at dusk with warm tones", "under a clear starry sky",
                "with fog rolling in", "on a rainy evening"]
    CAMERA_DETAILS = ["shot from a slight low angle", "close‑up portrait", "medium shot",
                      "wide angle establishing shot", "shot with a shallow depth of field",
                      "portrait orientation", "captured with a 50mm lens", "photographed with soft focus"]

    BACKGROUND_PHRASES = {
        "indoor": "an indoor setting",
        "inside": "an indoor setting",
        "outdoor": "an outdoor setting",
        "outside": "an outdoor setting",
        "garden": "a garden",
        "park": "a park",
        "street": "a street",
        "urban": "an urban environment",
        "city": "a city street",
        "alley": "an alley",
        "downtown": "a downtown area",
        "metropolis": "a bustling metropolis",
        "nature": "a natural landscape",
        "forest": "a forest",
        "woods": "the woods",
        "mountain": "the mountains",
        "beach": "a beach",
        "lake": "a lakeside",
        "field": "an open field",
        "room": "a room",
        "studio": "a studio",
        "apartment": "an apartment",
        "plain background": "a plain background",
        "solid background": "a solid background",
        "white background": "a white background",
        "black background": "a black background",
    }

    OPENINGS = [
        "A photo of {subject}, wearing {clothing}, {background}, {lighting}, {style}.",
        "A photograph of {subject}, wearing {clothing}, {background}, {lighting}, {style}.",
        "A portrait of {subject}, wearing {clothing}, {background}, {lighting}, {style}.",
        "A candid shot of {subject}, wearing {clothing}, {background}, {lighting}, {style}.",
        "{subject}, wearing {clothing}, {background}, {lighting}, {style}.",
    ]

    templates = {
        "Custom": "",
        "Photorealistic Portrait": "A photo of {subject}, wearing {clothing}, {background}, {lighting}, {style}.",
        "Artistic Scene": "Digital art of {subject}, {background}, {style}, dramatic lighting.",
        "Fashion Shot": "Fashion photograph of {subject}, wearing {clothing}, {background}, {lighting}, {style}.",
        "Cinematic Frame": "Cinematic frame: {subject}, wearing {clothing}, {background}, {lighting}, {style}.",
    }
    template_name = st.selectbox("Choose a template:", list(templates.keys()))
    template = st.text_area(
        "Edit template:", value=templates[template_name], height=80,
        help="Use {subject}, {clothing}, {background}, {lighting}, {style} as placeholders."
    )

    # ---- Placeholder selection (Random as default) ----
    selected_tags = {}
    cols = st.columns(5)
    placeholders = ["subject", "clothing", "background", "lighting", "style"]
    for i, ph in enumerate(placeholders):
        with cols[i]:
            options = ["🎲 Random"] + placeholder_keywords[ph]
            selected = st.selectbox(f"**{ph}**", options, index=0, key=f"prompt_simple_{ph}")
            if selected and selected != "🎲 Random":
                selected_tags[ph] = selected

    with st.expander("✏️ Custom overrides (type any text)"):
        for ph in placeholders:
            custom_val = st.text_input(f"Custom {ph}:", key=f"custom_simple_{ph}")
            if custom_val:
                selected_tags[ph] = custom_val.strip()

    def get_random_tag(ph):
        kw = placeholder_keywords.get(ph, [])
        return random.choice(kw) if kw else "none"

    # ---------- Grammar helpers ----------
    def article_for(word):
        """Return 'a' or 'an' based on the first sound of the word."""
        if word[0].lower() == 'u':
            return 'a'
        return "an" if word[0].lower() in "aeiou" else "a"

    def expand_subject(tag):
        gender = tag.lower()
        if gender in ("man", "male", "guy"):
            base = random.choice(["man", "gentleman", "guy"])
            age_prefix = random.choice(["young", "middle‑aged", ""])
            if age_prefix:
                base = f"{age_prefix} {base}"
        elif gender in ("woman", "female", "lady"):
            base = random.choice(["woman", "lady"])
            age_prefix = random.choice(["young", "middle‑aged", ""])
            if age_prefix:
                base = f"{age_prefix} {base}"
        else:
            base = "person"
            age_prefix = random.choice(["young", "middle‑aged", ""])
            if age_prefix:
                base = f"{age_prefix} {base}"

        adj = random.choice(ADJECTIVES)
        hair = f"{random.choice(HAIR_STYLES)} {random.choice(HAIR_COLORS)} hair"
        eyes = f"{random.choice(EYE_COLORS)} eyes"
        expr = random.choice(EXPRESSIONS)

        if age_prefix:
            full = f"{adj} {base}"
        else:
            full = f"{adj} {base}"

        art = article_for(full.split()[0])
        return f"{art} {full} with {hair}, {eyes}, {expr}"

    def expand_clothing(tag):
        """Return ONLY the descriptive phrase (no 'wearing') – the template adds 'wearing'."""
        tag_lower = tag.lower()
        garment = None
        color = None
        for category, items in CONCEPT_KEYWORDS["Clothing & Colors"].items():
            if tag_lower in items:
                if category in ("black","white","red","blue","green","pink","yellow","purple","brown","grey"):
                    color = category
                else:
                    garment = category
                break
        if not garment and not color:
            garment = tag_lower
            color = random.choice(CLOTHING_COLORS)
        elif garment and not color:
            color = random.choice(CLOTHING_COLORS)
        elif color and not garment:
            garment = random.choice(CLOTHING_ITEMS)

        style = random.choice(CLOTHING_STYLES)

        # Plural garments → no article, and style comes after color
        if garment in PLURAL_CLOTHING:
            return f"{style} {color} {garment}"
        else:
            art = article_for(style)
            return f"{art} {style} {color} {garment}"

    def expand_background(tag):
        tag_lower = tag.lower()
        if tag_lower in BACKGROUND_PHRASES:
            base = BACKGROUND_PHRASES[tag_lower]
        else:
            base = tag

        if "plain" in tag_lower or "solid" in tag_lower:
            return base

        detail = random.choice(SCENE_DETAILS)
        return f"{base}, {detail}"

    def expand_lighting(tag):
        weather = random.choice(WEATHERS)
        return f"{tag}, {weather}"

    def expand_style(tag):
        cam = random.choice(CAMERA_DETAILS)
        return f"{tag}, {cam}"

    EXPANDERS = {
        "subject":    expand_subject,
        "clothing":   expand_clothing,
        "background": expand_background,
        "lighting":   expand_lighting,
        "style":      expand_style,
    }

    # ---- Generate single prompt ----
    if st.button("🔄 Generate Prompt"):
        parts = {}
        for ph in placeholders:
            tag = selected_tags.get(ph)
            if not tag:
                tag = get_random_tag(ph)
            parts[ph] = EXPANDERS[ph](tag)

        if template_name == "Custom" and not template.strip():
            tpl = random.choice(OPENINGS)
        else:
            tpl = template

        prompt = tpl
        for ph in placeholders:
            prompt = prompt.replace(f"{{{ph}}}", parts[ph])
        import re
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        if prompt:
            prompt = prompt[0].upper() + prompt[1:]
        st.session_state.generated_prompt = prompt

    if "generated_prompt" not in st.session_state:
        st.session_state.generated_prompt = ""

    if st.session_state.generated_prompt:
        st.subheader("📋 Generated Prompt")
        st.code(st.session_state.generated_prompt, language="text")
        st.download_button(
            "📥 Download prompt", st.session_state.generated_prompt,
            file_name="prompt.txt", mime="text/plain"
        )

    # ---- Batch generation ----
    with st.expander("📚 Batch generate (multiple prompts)", expanded=False):
        st.markdown("Generate several unique prompts at once.")
        num_prompts = st.slider("Number of prompts", 1, 20, 5)
        if st.button("⚡ Generate batch"):
            batch = []
            for _ in range(num_prompts):
                if template_name == "Custom" and not template.strip():
                    tpl = random.choice(OPENINGS)
                else:
                    tpl = template
                parts = {}
                for ph in placeholders:
                    tag = selected_tags.get(ph)
                    if not tag:
                        tag = get_random_tag(ph)
                    parts[ph] = EXPANDERS[ph](tag)
                prompt = tpl
                for ph in placeholders:
                    prompt = prompt.replace(f"{{{ph}}}", parts[ph])
                prompt = re.sub(r'\s+', ' ', prompt).strip()
                if prompt:
                    prompt = prompt[0].upper() + prompt[1:]
                batch.append(prompt)
            st.session_state.batch_prompts = batch

        if "batch_prompts" in st.session_state and st.session_state.batch_prompts:
            st.write(f"Generated {len(st.session_state.batch_prompts)} prompts:")
            for i, p in enumerate(st.session_state.batch_prompts):
                st.code(f"{i+1}. {p}", language="text")
            st.download_button(
                "📥 Download all prompts", "\n".join(st.session_state.batch_prompts),
                file_name="batch_prompts.txt", mime="text/plain"
            )



# ======================================================================
# TAB 17: Report Generator
# ======================================================================

elif current_tab == "📰 Report Generator":
    st.header("📰 Report Generator")
    st.markdown("Generate a comprehensive HTML report of your dataset with statistics, charts, and sample images.")

    if not folder_valid or not raw_items:
        st.info("Load a dataset first to generate a report.")
        st.stop()

    # ---- Get or compute quality metrics ----
    if "metrics_map" not in st.session_state:
        with st.spinner("Computing image quality metrics (this runs once)…"):
            st.session_state.metrics_map, _ = compute_all_metrics(dataset_path)
    metrics_map = st.session_state.metrics_map

    # ---- Gather data ----
    from collections import Counter

    total_images = len(raw_items)
    avg_quality = np.mean([m['overall_quality'] for m in metrics_map.values()]) if metrics_map else 0
    empty_captions = sum(1 for item in raw_items if not item["caption"].strip())
    all_tags = []
    for item in raw_items:
        tags = [t.strip().lower() for t in item["caption"].split(",") if t.strip()]
        all_tags.extend(tags)
    tag_counts = Counter(all_tags)
    top_tags = tag_counts.most_common(20)

    # Caption length statistics
    caption_lengths = [len(item["caption"]) for item in raw_items]
    avg_length = np.mean(caption_lengths) if caption_lengths else 0
    max_length = max(caption_lengths) if caption_lengths else 0

    # Pick a few random sample images
    import random
    samples = random.sample(raw_items, min(5, len(raw_items)))

    # ---- Build HTML report ----
    import datetime
    html_report = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Dataset Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            .stat-box {{ background: #f4f4f4; padding: 10px; margin: 10px 0; border-radius: 6px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #2c3e50; color: white; }}
            img {{ max-width: 300px; margin: 5px; }}
        </style>
    </head>
    <body>
        <h1>📊 Dataset Report</h1>
        <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="stat-box">
            <h2>Overview</h2>
            <p><strong>Total images:</strong> {total_images}</p>
            <p><strong>Average quality:</strong> {avg_quality:.1f}</p>
            <p><strong>Empty captions:</strong> {empty_captions}</p>
            <p><strong>Average caption length:</strong> {avg_length:.0f} characters</p>
            <p><strong>Max caption length:</strong> {max_length} characters</p>
            <p><strong>Unique tags:</strong> {len(tag_counts)}</p>
        </div>

        <div class="stat-box">
            <h2>Top 20 Tags</h2>
            <table>
                <tr><th>Tag</th><th>Count</th></tr>
                {"".join(f"<tr><td>{tag}</td><td>{count}</td></tr>" for tag, count in top_tags)}
            </table>
        </div>

        <div class="stat-box">
            <h2>Sample Images</h2>
            {"".join(f'<div><img src="file://{os.path.join(dataset_path, s["img"])}" alt="{s["img"]}"><br><small>{s["caption"][:100]}…</small></div>' for s in samples)}
        </div>

        <p><em>Report created by Dataset Caption Editor.</em></p>
    </body>
    </html>
    """

    # ---- Download button ----
    st.download_button(
        label="📥 Download HTML Report",
        data=html_report,
        file_name="dataset_report.html",
        mime="text/html"
    )

    st.markdown("---")
    st.info("💡 The downloaded report includes overview stats, top tags, and sample images. You can open it in any browser.")



# ======================================================================
# TAB 18: PNG Info Viewer
# ======================================================================

elif current_tab == "🎨 PNG Info Viewer":
    st.header("🎨 PNG Info Viewer")
    st.markdown(
        "View the hidden generation parameters embedded in AI‑generated PNG files "
        "(ComfyUI, Forge Neo, Civitai, etc.). Select an image from your dataset or upload one."
    )

    # ---- Source selection ----
    source = st.radio("Image source:", ["From dataset", "Upload new"], horizontal=True)
    image_path = None
    image_name = None

    if source == "From dataset":
        if not folder_valid or not raw_items:
            st.warning("Load a dataset first.")
            st.stop()
        img_names = [item["img"] for item in raw_items]
        selected = st.selectbox("Choose an image:", img_names)
        if selected:
            image_path = os.path.join(dataset_path, selected)
            image_name = selected
    else:
        uploaded = st.file_uploader("Upload a PNG file", type=["png"])
        if uploaded:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded.getvalue())
                image_path = tmp.name
            image_name = uploaded.name

    if not image_path:
        st.info("Select or upload an image to inspect its metadata.")
        st.stop()

    # ---- Extract PNG info ----
    try:
        img = Image.open(image_path)
        st.image(img, width=400, caption=image_name)

        # Read the embedded metadata
        # ComfyUI / webui store parameters in the "parameters" text chunk
        png_info = img.text if hasattr(img, 'text') else {}

        # The raw text is usually under a key like 'parameters' or 'Description'
        raw_text = ""
        for key, val in png_info.items():
            if isinstance(val, str):
                raw_text += val + "\n"
            elif isinstance(val, bytes):
                raw_text += val.decode(errors='replace') + "\n"

        if not raw_text:
            # Some older formats may have info in img.info
            raw_text = img.info.get("parameters", "") or img.info.get("Description", "") or ""

        if not raw_text:
            st.warning("No embedded generation data found in this PNG.")
        else:
            # Parse the raw text
            # Typical format:
            # Prompt: ... \nSteps: 8, CFG scale: 1, Sampler: er_sde, Seed: 208151501184135, engine: ComfyUI, Model: krea2_turbo_fp8
            prompt = ""
            params = {}
            lines = raw_text.strip().split('\n')
            # First line is usually the prompt, or prefixed with "Prompt:"
            if lines and lines[0].lower().startswith("prompt:"):
                prompt = lines[0][len("prompt:"):].strip()
                lines = lines[1:]
            elif lines:
                prompt = lines[0]
                lines = lines[1:]

            # Remaining lines contain parameters as comma-separated key: value pairs
            param_str = " ".join(lines).strip()
            # Split by comma or newline, but key:value pairs may contain spaces
            # Better: parse known keys
            import re
            # Extract key:value patterns like "Steps: 8" or "Sampler: er_sde"
            for match in re.finditer(r'([A-Za-z ]+?):\s*([^,]+)', param_str):
                key = match.group(1).strip()
                value = match.group(2).strip()
                params[key] = value

            # Also try to capture "Seed" which can be a long number
            if "Seed" not in params:
                seed_match = re.search(r'Seed:\s*(\d+)', param_str)
                if seed_match:
                    params["Seed"] = seed_match.group(1)

            # Display
            if prompt:
                st.subheader("📝 Prompt")
                st.text_area("Prompt", prompt, height=150, key="png_prompt_display")
            if params:
                st.subheader("⚙️ Generation Parameters")
                import pandas as pd
                df = pd.DataFrame(
                    [{"Parameter": k, "Value": v} for k, v in params.items()]
                )
                st.dataframe(df, hide_index=True, use_container_width=True)

            # Also show raw metadata for advanced users
            with st.expander("🔧 View raw metadata"):
                st.code(raw_text, language="text")

    except Exception as e:
        st.error(f"Could not read image: {e}")
    finally:
        if source == "Upload new" and image_path and os.path.exists(image_path):
            try:
                os.unlink(image_path)
            except:
                pass



# ======================================================================
# TAB 19: Notes
# ======================================================================

elif current_tab == "✍️ Notes":
    import datetime

    st.header("✍️ Notes")
    st.markdown("A simple notepad. Your text is saved automatically and will be here when you return.")

    NOTES_FILE = os.path.join(APP_DIR, "notes.json")

    # Load existing notes
    def load_notes():
        if os.path.exists(NOTES_FILE):
            try:
                with open(NOTES_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_notes(notes_dict):
        with open(NOTES_FILE, "w", encoding="utf-8") as f:
            json.dump(notes_dict, f, indent=2, ensure_ascii=False)

    # We'll store notes by a key – here we just use "main" as the single note.
    # But you can extend it to multiple named notes if you like.
    notes_data = load_notes()
    main_note = notes_data.get("main", "")

    # Editable text area
    new_note = st.text_area("Write your notes below:", value=main_note, height=400, key="notes_area")

    # Save button
    if st.button("💾 Save Note", use_container_width=True):
        notes_data["main"] = new_note
        notes_data["last_modified"] = datetime.datetime.now().isoformat()
        save_notes(notes_data)
        st.success("Note saved!")
        # No need to rerun – we just saved the current text



# ======================================================================
# TAB 20: User Guide
# ======================================================================

elif current_tab == "📖 User Guide":
    st.header("📖 User Guide")
    st.markdown("Welcome to the Dataset Caption Editor! Here's a quick overview of every tool in the order they appear in the sidebar.")

    tabs_info = [
        ("🖥️ Soho Workspace & Browser",
         "Browse images, edit captions, search by caption text, sort by quality or orientation, export filtered subsets, and use the zoom panel. "
         "The Save & Next / Save & Prev buttons let you edit quickly without clicking thumbnails."),
        ("📊 Dataset Word & Phrase Analytics",
         "Analyse word frequencies, sentence‑level n‑grams, and comma‑separated chunks. Click any phrase to filter the workspace by that phrase."),
        ("📷 Image Quality Analytics",
         "View technical quality metrics (sharpness, noise, JPEG artifacts, resolution, aspect ratio). Detect duplicate images and use the Smart Duplicate Cleanup to keep the best copy."),
        ("📋 Dataset Statistics",
         "Interactive pie/bar charts for concepts like gender, camera, lighting, clothing, etc. Includes a caption length histogram, word cloud, quality‑concept crossover, dataset completeness, and a bias analysis prompt generator."),
        ("📝 Caption Tools",
         "Batch‑append quality ratings to captions, and run automatic caption validation/auto‑fix (missing commas, punctuation, etc.) with a progress bar."),
        ("🧠 AI Assistance",
         "CLIP‑based hallucination checker. Compute similarity scores between images and captions, then edit, ignore, or delete flagged pairs directly from the results."),
        ("💾 Captioner",
         "Single‑image or batch captioning using JoyCaption (default) plus other models you can add (Ollama, OpenAI‑compatible). Multi‑model batch mode compares outputs and lets you pick the best caption."),
        ("✂️ Image Cropper",
         "Interactive drag‑to‑crop with aspect‑ratio lock (Free, 16:9, 1:1, etc.). Double‑click the crop box to confirm and download immediately."),
        ("🔄 Image Converter",
         "Convert images (AVIF, WEBP, JPG, PNG, BMP, TIFF) to high‑quality PNG, preserving original size. Single‑image or batch mode with ZIP download."),
        ("📐 Smart Resize & Crop",
         "Downscale and center‑crop PNG images to exact training resolutions (1024×1024, 832×1248, 1344×768) using the same algorithm as your standalone script."),
        ("📝 Batch Rename",
         "Rename all PNG images (and their .txt files) sequentially, e.g., image_001.png. You can rename in place or copy to a new folder."),
        ("🏷️ EXIF Viewer & Stripper",
         "View hidden camera metadata (camera model, lens, aperture, ISO, GPS). Strip EXIF from single images or entire folders for privacy."),
        ("ℹ️ Training Run Tracker",
         "Log your LoRA / model training runs with hyperparameters, dataset paths, and notes. All entries are saved automatically."),
        ("🚀 Fine‑Tuning Launcher",
         "Generate ready‑to‑run training commands for Diffusers or Kohya scripts. Save commands as .bat/.sh files and optionally log the run in the Training Run Tracker."),
        ("✨ Dataset Comparator",
         "Compare two dataset folders side‑by‑side: see which images are unique, find identical copies (perceptual hash), compare captions, and analyse quality metrics."),
        ("📜 Prompt Generator",
         "Assemble prompts from your dataset's most common tags using customisable templates. Generate single prompts or a batch list."),
        ("📰 Report Generator",
         "Download a self‑contained HTML report with dataset overview, top tags, and sample images."),
        ("🎨 PNG Info Viewer",
         "Read embedded generation parameters from AI‑generated PNG files (ComfyUI, Forge Neo, Civitai). Shows prompt, steps, CFG scale, sampler, seed, and model."),
        ("✍️ Notes",
         "A simple notepad that saves your notes automatically. Handy for quick ideas or to‑do lists."),
        ("📖 User Guide",
         "This guide! 😊"),
    ]

    for name, desc in tabs_info:
        with st.expander(name):
            st.write(desc)

    st.markdown("---")
    st.markdown("### 🛠️ Tips & Shortcuts")
    st.markdown("""
    - Use the **compact sidebar** (◀ button) to save screen space.
    - Drag the sidebar border to resize it.
    - Favorites folders appear as clickable buttons – pin your most‑used datasets.
    - The **Batch Tag Library** in the sidebar lets you add/remove/replace tags across your entire dataset. (I have not tested this)
    """)