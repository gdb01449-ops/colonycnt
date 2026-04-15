import io
import csv
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


st.set_page_config(page_title="대장균 배지 자동 계수기", layout="wide")


# -----------------------------
# 기본 설정 / 프리셋
# -----------------------------
PRESETS = {
    "분원성대장균": {
        "보수적": {
            "min_area": 45,
            "max_area": 1600,
            "circularity": 0.45,
            "min_distance": 16,
            "edge_margin_ratio": 0.12,
            "min_radius": 5,
            "max_radius": 18,
            "h_min": 95,
            "h_max": 160,
            "s_min": 45,
            "v_limit": 155,
            "blackhat_thresh": 20,
        },
        "기본": {
            "min_area": 30,
            "max_area": 1800,
            "circularity": 0.35,
            "min_distance": 14,
            "edge_margin_ratio": 0.10,
            "min_radius": 4,
            "max_radius": 20,
            "h_min": 90,
            "h_max": 165,
            "s_min": 35,
            "v_limit": 170,
            "blackhat_thresh": 18,
        },
        "민감": {
            "min_area": 20,
            "max_area": 2200,
            "circularity": 0.25,
            "min_distance": 10,
            "edge_margin_ratio": 0.08,
            "min_radius": 3,
            "max_radius": 22,
            "h_min": 85,
            "h_max": 170,
            "s_min": 25,
            "v_limit": 185,
            "blackhat_thresh": 15,
        },
    },
    "총대장균": {
        "보수적": {
            "min_area": 18,
            "max_area": 900,
            "circularity": 0.28,
            "min_distance": 10,
            "edge_margin_ratio": 0.10,
            "min_radius": 3,
            "max_radius": 10,
            "yellow_h_min": 10,
            "yellow_h_max": 42,
            "yellow_s_min": 45,
            "yellow_v_min": 95,
            "dark_v_max": 90,
            "blackhat_thresh": 20,
        },
        "기본": {
            "min_area": 12,
            "max_area": 1200,
            "circularity": 0.18,
            "min_distance": 8,
            "edge_margin_ratio": 0.07,
            "min_radius": 2,
            "max_radius": 14,
            "yellow_h_min": 8,
            "yellow_h_max": 48,
            "yellow_s_min": 35,
            "yellow_v_min": 85,
            "dark_v_max": 110,
            "blackhat_thresh": 16,
        },
        "민감": {
            "min_area": 8,
            "max_area": 1500,
            "circularity": 0.10,
            "min_distance": 6,
            "edge_margin_ratio": 0.05,
            "min_radius": 2,
            "max_radius": 16,
            "yellow_h_min": 5,
            "yellow_h_max": 55,
            "yellow_s_min": 25,
            "yellow_v_min": 75,
            "dark_v_max": 125,
            "blackhat_thresh": 12,
        },
    },
}

TARGET_PLATE_DIAMETER = 900


# -----------------------------
# 유틸
# -----------------------------
def read_uploaded_image(uploaded_file) -> np.ndarray:
    pil_img = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def detect_plate_and_normalize(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(min(gray.shape[:2]) // 2, 50),
        param1=120,
        param2=30,
        minRadius=max(min(gray.shape[:2]) // 5, 20),
        maxRadius=max(min(gray.shape[:2]) // 2, 40),
    )

    if circles is None:
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_r = 0
        for cnt in contours:
            (x, y), r = cv2.minEnclosingCircle(cnt)
            if r > best_r:
                best_r = r
                best = (int(x), int(y), int(r))
        if best is None:
            raise ValueError("페트리디시를 찾지 못했습니다. 배지가 크게 보이도록 다시 촬영해 주세요.")
        x, y, r = best
    else:
        circles = np.round(circles[0]).astype(int)
        circles = sorted(circles, key=lambda c: c[2], reverse=True)
        x, y, r = circles[0]

    pad = int(r * 0.08)
    x1 = max(x - r - pad, 0)
    y1 = max(y - r - pad, 0)
    x2 = min(x + r + pad, img_bgr.shape[1])
    y2 = min(y + r + pad, img_bgr.shape[0])
    crop = img_bgr[y1:y2, x1:x2].copy()

    crop_h, crop_w = crop.shape[:2]
    local_r = max(min(crop_w, crop_h) // 2 - pad, 1)
    scale = TARGET_PLATE_DIAMETER / max(2 * local_r, 1)
    new_w = max(int(crop_w * scale), 1)
    new_h = max(int(crop_h * scale), 1)
    normalized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    mask = np.zeros((new_h, new_w), dtype=np.uint8)
    center = (new_w // 2, new_h // 2)
    new_r = min(TARGET_PLATE_DIAMETER // 2, min(new_w, new_h) // 2 - 2)
    cv2.circle(mask, center, new_r, 255, -1)

    preview = normalized.copy()
    cv2.circle(preview, center, new_r, (255, 255, 0), 2)
    debug = {"1_정규화된_배지": preview}
    return normalized, mask, (center[0], center[1], new_r), debug


def filter_candidates(plate_bgr, binary, plate_mask, plate_center_radius, params, debug_name):
    hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
    h_img, s_img, v_img = cv2.split(hsv)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = plate_bgr.copy()
    result[plate_mask == 0] = 0
    cv2.circle(result, (plate_center_radius[0], plate_center_radius[1]), plate_center_radius[2], (255, 255, 0), 2)

    colonies = []
    edge_margin = int(plate_center_radius[2] * params["edge_margin_ratio"])
    valid_radius_limit = plate_center_radius[2] - edge_margin
    candidate_debug = result.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params["min_area"] or area > params["max_area"]:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < params["circularity"]:
            continue

        (cx_f, cy_f), radius = cv2.minEnclosingCircle(cnt)
        cx, cy = int(cx_f), int(cy_f)
        if radius < params["min_radius"] or radius > params["max_radius"]:
            continue

        dx = cx - plate_center_radius[0]
        dy = cy - plate_center_radius[1]
        if dx * dx + dy * dy > valid_radius_limit * valid_radius_limit:
            continue

        mask_one = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(mask_one, [cnt], -1, 255, -1)
        mean_h = cv2.mean(h_img, mask=mask_one)[0]
        mean_s = cv2.mean(s_img, mask=mask_one)[0]
        mean_v = cv2.mean(v_img, mask=mask_one)[0]

        too_close = False
        for c in colonies:
            if (c["cx"] - cx) ** 2 + (c["cy"] - cy) ** 2 < params["min_distance"] ** 2:
                too_close = True
                if area > c["area"]:
                    c.update({
                        "cx": cx,
                        "cy": cy,
                        "area": float(area),
                        "circularity": float(circularity),
                        "contour": cnt,
                        "mean_h": float(mean_h),
                        "mean_s": float(mean_s),
                        "mean_v": float(mean_v),
                        "radius": float(radius),
                    })
                break
        if too_close:
            continue

        colonies.append({
            "cx": cx,
            "cy": cy,
            "area": float(area),
            "circularity": float(circularity),
            "contour": cnt,
            "mean_h": float(mean_h),
            "mean_s": float(mean_s),
            "mean_v": float(mean_v),
            "radius": float(radius),
        })

        cv2.drawContours(candidate_debug, [cnt], -1, (0, 255, 255), 2)
        cv2.putText(
            candidate_debug,
            f"H{int(mean_h)} S{int(mean_s)} V{int(mean_v)}",
            (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    colonies = sorted(colonies, key=lambda item: (item["cy"], item["cx"]))

    final_img = result.copy()
    cv2.circle(final_img, (plate_center_radius[0], plate_center_radius[1]), plate_center_radius[2], (255, 255, 0), 2)
    for idx, colony in enumerate(colonies, start=1):
        cv2.drawContours(final_img, [colony["contour"]], -1, (0, 255, 0), 2)
        cv2.circle(final_img, (colony["cx"], colony["cy"]), 4, (0, 0, 255), -1)
        cv2.putText(
            final_img,
            str(idx),
            (colony["cx"] + 6, colony["cy"] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 50, 50),
            1,
            cv2.LINE_AA,
        )

    debug = {
        f"5_{debug_name}_후보": candidate_debug,
        "9_최종결과": final_img,
    }
    return colonies, final_img, debug


def detect_fecal_coliform(plate_bgr, plate_mask, plate_center_radius, params):
    hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (7, 7), 0)
    blackhat = cv2.morphologyEx(
        blur_gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    )

    blue_mask = cv2.inRange(h, params["h_min"], params["h_max"])
    sat_mask = cv2.inRange(s, params["s_min"], 255)
    dark_mask = cv2.inRange(v, 0, params["v_limit"])
    bh_mask = cv2.inRange(blackhat, params["blackhat_thresh"], 255)

    binary = cv2.bitwise_and(blue_mask, sat_mask)
    binary = cv2.bitwise_and(binary, dark_mask)
    binary = cv2.bitwise_or(binary, cv2.bitwise_and(bh_mask, cv2.bitwise_and(sat_mask, dark_mask)))
    binary = cv2.bitwise_and(binary, binary, mask=plate_mask)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=2)

    debug = {
        "2_분원성_파란색마스크": cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR),
        "3_분원성_블랙햇": cv2.cvtColor(blackhat, cv2.COLOR_GRAY2BGR),
        "4_분원성_이진화": cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
    }
    colonies, final_img, debug2 = filter_candidates(plate_bgr, binary, plate_mask, plate_center_radius, params, "분원성")
    debug.update(debug2)
    return colonies, final_img, debug


def detect_total_coliform(plate_bgr, plate_mask, plate_center_radius, params):
    hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (7, 7), 0)
    blackhat = cv2.morphologyEx(
        blur_gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    )

    yellow_h = cv2.inRange(h, params["yellow_h_min"], params["yellow_h_max"])
    yellow_s = cv2.inRange(s, params["yellow_s_min"], 255)
    yellow_v = cv2.inRange(v, params["yellow_v_min"], 255)
    yellow_mask = cv2.bitwise_and(yellow_h, yellow_s)
    yellow_mask = cv2.bitwise_and(yellow_mask, yellow_v)

    dark_mask = cv2.inRange(v, 0, params["dark_v_max"])
    bh_mask = cv2.inRange(blackhat, params["blackhat_thresh"], 255)

    binary = cv2.bitwise_or(yellow_mask, cv2.bitwise_and(dark_mask, bh_mask))
    binary = cv2.bitwise_and(binary, binary, mask=plate_mask)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel3, iterations=1)

    debug = {
        "2_총대장균_노랑마스크": cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR),
        "3_총대장균_블랙햇": cv2.cvtColor(blackhat, cv2.COLOR_GRAY2BGR),
        "4_총대장균_이진화": cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
    }
    colonies, final_img, debug2 = filter_candidates(plate_bgr, binary, plate_mask, plate_center_radius, params, "총대장균")
    debug.update(debug2)
    return colonies, final_img, debug


def build_results_df(colonies):
    rows = []
    for idx, c in enumerate(colonies, start=1):
        rows.append({
            "번호": idx,
            "중심X": c["cx"],
            "중심Y": c["cy"],
            "면적": round(c["area"], 2),
            "원형도": round(c["circularity"], 4),
            "평균H": round(c["mean_h"], 2),
            "평균S": round(c["mean_s"], 2),
            "평균V": round(c["mean_v"], 2),
            "반지름": round(c["radius"], 2),
        })
    return pd.DataFrame(rows)


def df_to_csv_bytes(df: pd.DataFrame, analysis_target: str, level: str) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(df.columns.tolist())
    for row in df.itertuples(index=False):
        writer.writerow(list(row))
    writer.writerow([])
    writer.writerow(["총 집락 수", len(df)])
    writer.writerow(["분석 대상", analysis_target])
    writer.writerow(["검출 강도", level])
    return output.getvalue().encode("utf-8-sig")


# -----------------------------
# 화면
# -----------------------------
st.title("대장균 배지 자동 계수기")
st.caption("Streamlit용 버전 · 베이지 배지는 분원성대장균, 붉은/분홍 배지는 총대장균")

with st.sidebar:
    st.header("설정")
    analysis_target = st.selectbox("분석 대상", ["분원성대장균", "총대장균"])
    detection_level = st.selectbox("검출 강도", ["보수적", "기본", "민감"], index=1)
    show_debug = st.checkbox("디버그 보기", value=True)
    uploaded_file = st.file_uploader("배지 사진 업로드", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

    st.markdown(
        """
        **사용 가이드**

        - 베이지 배지 → 분원성대장균
        - 붉은/분홍 배지 → 총대장균
        - 보수적 → 과검출 감소
        - 기본 → 일반 추천
        - 민감 → 누락 감소
        """
    )

if uploaded_file is None:
    st.info("왼쪽 사이드바에서 배지 사진을 업로드해 주세요.")
    st.stop()

try:
    img_bgr = read_uploaded_image(uploaded_file)
    params = PRESETS[analysis_target][detection_level]

    plate_bgr, plate_mask, plate_center_radius, debug_images = detect_plate_and_normalize(img_bgr)

    if analysis_target == "분원성대장균":
        colonies, result_bgr, debug2 = detect_fecal_coliform(plate_bgr, plate_mask, plate_center_radius, params)
    else:
        colonies, result_bgr, debug2 = detect_total_coliform(plate_bgr, plate_mask, plate_center_radius, params)

    debug_images.update(debug2)
    result_df = build_results_df(colonies)

except Exception as e:
    st.error(f"분석 중 오류가 발생했습니다: {e}")
    st.stop()

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("원본 사진")
    st.image(bgr_to_rgb(img_bgr), use_container_width=True)
with col2:
    st.subheader("분석 결과")
    st.image(bgr_to_rgb(result_bgr), use_container_width=True)

st.metric("자동 검출 집락 수", len(colonies))

if not result_df.empty:
    st.subheader("검출 결과 표")
    st.dataframe(result_df, use_container_width=True)

    csv_bytes = df_to_csv_bytes(result_df, analysis_target, detection_level)
    st.download_button(
        label="결과 CSV 다운로드",
        data=csv_bytes,
        file_name="colony_result.csv",
        mime="text/csv",
    )
else:
    st.warning("집락이 검출되지 않았습니다. 분석 대상 또는 검출 강도를 바꿔 보세요.")

if show_debug:
    st.subheader("디버그 화면")
    keys = sorted(debug_images.keys())
    for i in range(0, len(keys), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(keys):
                key = keys[i + j]
                with cols[j]:
                    st.markdown(f"**{key}**")
                    st.image(bgr_to_rgb(debug_images[key]), use_container_width=True)
