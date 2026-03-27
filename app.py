import io
import math
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st


TARGET_PLATE_DIAMETER = 900


@dataclass
class Colony:
    cx: int
    cy: int
    area: float
    circularity: float
    mean_h: float
    mean_s: float
    mean_v: float
    radius: float
    manual: bool = False
    contour: Optional[np.ndarray] = None


DEFAULTS = {
    "fecal_coliform": {
        "min_area": 30,
        "max_area": 1800,
        "circularity": 0.35,
        "min_distance": 14,
        "edge_margin_ratio": 0.10,
        "min_radius": 4,
        "max_radius": 20,
        "fecal_h_min": 90,
        "fecal_h_max": 165,
        "fecal_s_min": 35,
        "fecal_v_max": 170,
        "fecal_blackhat_thresh": 18,
        "total_yellow_h_min": 8,
        "total_yellow_h_max": 48,
        "total_yellow_s_min": 35,
        "total_yellow_v_min": 85,
        "total_dark_v_max": 110,
        "total_blackhat_thresh": 16,
    },
    "total_coliform": {
        "min_area": 12,
        "max_area": 1200,
        "circularity": 0.18,
        "min_distance": 8,
        "edge_margin_ratio": 0.07,
        "min_radius": 2,
        "max_radius": 14,
        "fecal_h_min": 90,
        "fecal_h_max": 165,
        "fecal_s_min": 35,
        "fecal_v_max": 170,
        "fecal_blackhat_thresh": 18,
        "total_yellow_h_min": 8,
        "total_yellow_h_max": 48,
        "total_yellow_s_min": 35,
        "total_yellow_v_min": 85,
        "total_dark_v_max": 110,
        "total_blackhat_thresh": 16,
    },
}


COMMON_KEYS = [
    "min_area",
    "max_area",
    "circularity",
    "min_distance",
    "edge_margin_ratio",
    "min_radius",
    "max_radius",
]

FECAL_KEYS = [
    "fecal_h_min",
    "fecal_h_max",
    "fecal_s_min",
    "fecal_v_max",
    "fecal_blackhat_thresh",
]

TOTAL_KEYS = [
    "total_yellow_h_min",
    "total_yellow_h_max",
    "total_yellow_s_min",
    "total_yellow_v_min",
    "total_dark_v_max",
    "total_blackhat_thresh",
]


if "manual_points" not in st.session_state:
    st.session_state.manual_points = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "settings_seed" not in st.session_state:
    st.session_state.settings_seed = 0


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def read_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다.")
    return img


def detect_plate_and_normalize(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(gray.shape[:2]) // 2,
        param1=120,
        param2=30,
        minRadius=min(gray.shape[:2]) // 5,
        maxRadius=min(gray.shape[:2]) // 2,
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
            raise ValueError("페트리디시를 찾지 못했습니다.")
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
    local_r = min(crop_w, crop_h) // 2 - pad
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
    return normalized, mask, (x, y, r), (center[0], center[1], new_r), preview



def extract_candidates(plate_bgr: np.ndarray, binary: np.ndarray, plate_mask: np.ndarray, plate_center_radius, settings: Dict, mode_name: str):
    hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
    h_img, s_img, v_img = cv2.split(hsv)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = plate_bgr.copy()
    result[plate_mask == 0] = 0
    cv2.circle(result, (plate_center_radius[0], plate_center_radius[1]), plate_center_radius[2], (255, 255, 0), 2)

    colonies: List[Colony] = []
    edge_margin = int(plate_center_radius[2] * settings["edge_margin_ratio"])
    valid_radius_limit = plate_center_radius[2] - edge_margin
    candidate_debug = result.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < settings["min_area"] or area > settings["max_area"]:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < settings["circularity"]:
            continue
        (cx_f, cy_f), radius = cv2.minEnclosingCircle(cnt)
        cx, cy = int(cx_f), int(cy_f)
        if radius < settings["min_radius"] or radius > settings["max_radius"]:
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
            if (c.cx - cx) ** 2 + (c.cy - cy) ** 2 < settings["min_distance"] ** 2:
                too_close = True
                if area > c.area:
                    c.cx = cx
                    c.cy = cy
                    c.area = float(area)
                    c.circularity = float(circularity)
                    c.contour = cnt
                    c.mean_h = float(mean_h)
                    c.mean_s = float(mean_s)
                    c.mean_v = float(mean_v)
                    c.radius = float(radius)
                    c.manual = False
                break
        if too_close:
            continue

        colonies.append(
            Colony(
                cx=cx,
                cy=cy,
                area=float(area),
                circularity=float(circularity),
                contour=cnt,
                mean_h=float(mean_h),
                mean_s=float(mean_s),
                mean_v=float(mean_v),
                radius=float(radius),
            )
        )
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

    colonies = sorted(colonies, key=lambda item: (item.cy, item.cx))
    return colonies, candidate_debug



def draw_colonies(base_img: np.ndarray, colonies: List[Colony], plate_center_radius):
    img = base_img.copy()
    cv2.circle(img, (plate_center_radius[0], plate_center_radius[1]), plate_center_radius[2], (255, 255, 0), 2)
    for idx, colony in enumerate(colonies, start=1):
        if colony.contour is not None:
            cv2.drawContours(img, [colony.contour], -1, (0, 255, 0), 2)
        color = (0, 0, 255) if not colony.manual else (255, 0, 255)
        cv2.circle(img, (colony.cx, colony.cy), 4, color, -1)
        cv2.putText(img, str(idx), (colony.cx + 6, colony.cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 50, 50), 1, cv2.LINE_AA)
    return img



def detect_fecal_coliform(plate_bgr, plate_mask, plate_center_radius, settings):
    hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (7, 7), 0)
    blackhat = cv2.morphologyEx(
        blur_gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    )

    blue_mask = cv2.inRange(h, settings["fecal_h_min"], settings["fecal_h_max"])
    sat_mask = cv2.inRange(s, settings["fecal_s_min"], 255)
    dark_mask = cv2.inRange(v, 0, settings["fecal_v_max"])
    bh_mask = cv2.inRange(blackhat, settings["fecal_blackhat_thresh"], 255)

    binary = cv2.bitwise_and(blue_mask, sat_mask)
    binary = cv2.bitwise_and(binary, dark_mask)
    binary = cv2.bitwise_or(binary, cv2.bitwise_and(bh_mask, cv2.bitwise_and(sat_mask, dark_mask)))
    binary = cv2.bitwise_and(binary, binary, mask=plate_mask)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=2)

    colonies, candidate_debug = extract_candidates(plate_bgr, binary, plate_mask, plate_center_radius, settings, "fecal")
    debug = {
        "2_fecal_blue_mask": cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR),
        "3_fecal_blackhat": cv2.cvtColor(blackhat, cv2.COLOR_GRAY2BGR),
        "4_fecal_binary": cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
        "5_fecal_candidates": candidate_debug,
    }
    return colonies, binary, debug



def detect_total_coliform(plate_bgr, plate_mask, plate_center_radius, settings):
    hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (7, 7), 0)
    blackhat = cv2.morphologyEx(
        blur_gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    )

    yellow_h = cv2.inRange(h, settings["total_yellow_h_min"], settings["total_yellow_h_max"])
    yellow_s = cv2.inRange(s, settings["total_yellow_s_min"], 255)
    yellow_v = cv2.inRange(v, settings["total_yellow_v_min"], 255)
    yellow_mask = cv2.bitwise_and(yellow_h, yellow_s)
    yellow_mask = cv2.bitwise_and(yellow_mask, yellow_v)

    dark_mask = cv2.inRange(v, 0, settings["total_dark_v_max"])
    bh_mask = cv2.inRange(blackhat, settings["total_blackhat_thresh"], 255)

    binary = cv2.bitwise_or(yellow_mask, cv2.bitwise_and(dark_mask, bh_mask))
    binary = cv2.bitwise_and(binary, binary, mask=plate_mask)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel3, iterations=1)

    colonies, candidate_debug = extract_candidates(plate_bgr, binary, plate_mask, plate_center_radius, settings, "total")
    debug = {
        "2_total_yellow_mask": cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR),
        "3_total_blackhat": cv2.cvtColor(blackhat, cv2.COLOR_GRAY2BGR),
        "4_total_binary": cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
        "5_total_candidates": candidate_debug,
    }
    return colonies, binary, debug



def detect_general(plate_bgr, plate_mask, plate_center_radius, settings):
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    background = cv2.medianBlur(gray, 31)
    corrected = cv2.subtract(background, gray)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    corrected = cv2.bitwise_and(corrected, corrected, mask=plate_mask)
    binary = cv2.adaptiveThreshold(corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, -6)
    binary = cv2.bitwise_and(binary, binary, mask=plate_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    colonies, candidate_debug = extract_candidates(plate_bgr, binary, plate_mask, plate_center_radius, settings, "general")
    debug = {
        "2_general_corrected": cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR),
        "3_general_binary": cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
        "5_general_candidates": candidate_debug,
    }
    return colonies, binary, debug



def add_manual_points(colonies: List[Colony], manual_points: List[Tuple[int, int]]):
    merged = list(colonies)
    for x, y in manual_points:
        merged.append(Colony(cx=int(x), cy=int(y), area=0.0, circularity=1.0, mean_h=0.0, mean_s=0.0, mean_v=0.0, radius=0.0, manual=True))
    return sorted(merged, key=lambda item: (item.cy, item.cx))



def colonies_to_dataframe(colonies: List[Colony]) -> pd.DataFrame:
    rows = []
    for idx, c in enumerate(colonies, start=1):
        rows.append({
            "번호": idx,
            "중심X": c.cx,
            "중심Y": c.cy,
            "면적": round(c.area, 2),
            "원형도": round(c.circularity, 4),
            "수동추가여부": "Y" if c.manual else "N",
            "평균H": round(c.mean_h, 2),
            "평균S": round(c.mean_s, 2),
            "평균V": round(c.mean_v, 2),
            "반지름": round(c.radius, 2),
        })
    return pd.DataFrame(rows)



def dataframe_to_csv_bytes(df: pd.DataFrame, analysis_target: str, analysis_mode: str) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.write("\n")
    buffer.write(f"총 집락 수,{len(df)}\n")
    buffer.write(f"분석 대상,{analysis_target}\n")
    buffer.write(f"분석 방식,{analysis_mode}\n")
    return buffer.getvalue().encode("utf-8-sig")



def image_to_bytes(img_bgr: np.ndarray, ext: str = ".png") -> bytes:
    ok, encoded = cv2.imencode(ext, img_bgr)
    if not ok:
        raise ValueError("이미지 인코딩에 실패했습니다.")
    return encoded.tobytes()



def render_sidebar_settings(analysis_target: str):
    defaults = DEFAULTS[analysis_target]
    with st.sidebar:
        st.header("설정")
        analysis_mode = st.radio("분석 방식", ["targeted", "general"], horizontal=True)

        if st.button("프리셋 다시 적용"):
            st.session_state.settings_seed += 1

        st.caption("베이지 배지 = fecal_coliform, 붉은 배지 = total_coliform")
        settings = {}
        seed = st.session_state.settings_seed
        for key in COMMON_KEYS:
            settings[key] = defaults[key]
        for key in FECAL_KEYS:
            settings[key] = defaults[key]
        for key in TOTAL_KEYS:
            settings[key] = defaults[key]

        st.subheader("공통 파라미터")
        settings["min_area"] = st.number_input("최소 면적", min_value=0, value=int(defaults["min_area"]), step=1, key=f"min_area_{analysis_target}_{seed}")
        settings["max_area"] = st.number_input("최대 면적", min_value=1, value=int(defaults["max_area"]), step=1, key=f"max_area_{analysis_target}_{seed}")
        settings["circularity"] = st.number_input("원형도 최소", min_value=0.0, max_value=1.0, value=float(defaults["circularity"]), step=0.01, key=f"circularity_{analysis_target}_{seed}")
        settings["min_distance"] = st.number_input("최소 중심거리", min_value=0, value=int(defaults["min_distance"]), step=1, key=f"min_distance_{analysis_target}_{seed}")
        settings["edge_margin_ratio"] = st.number_input("가장자리 제외비율", min_value=0.0, max_value=0.5, value=float(defaults["edge_margin_ratio"]), step=0.01, key=f"edge_margin_{analysis_target}_{seed}")
        settings["min_radius"] = st.number_input("최소 반지름", min_value=0, value=int(defaults["min_radius"]), step=1, key=f"min_radius_{analysis_target}_{seed}")
        settings["max_radius"] = st.number_input("최대 반지름", min_value=1, value=int(defaults["max_radius"]), step=1, key=f"max_radius_{analysis_target}_{seed}")

        st.subheader("분원성대장균 전용")
        settings["fecal_h_min"] = st.number_input("파란색 H 최소", min_value=0, max_value=255, value=int(defaults["fecal_h_min"]), step=1, key=f"fecal_h_min_{seed}")
        settings["fecal_h_max"] = st.number_input("파란색 H 최대", min_value=0, max_value=255, value=int(defaults["fecal_h_max"]), step=1, key=f"fecal_h_max_{seed}")
        settings["fecal_s_min"] = st.number_input("파란색 S 최소", min_value=0, max_value=255, value=int(defaults["fecal_s_min"]), step=1, key=f"fecal_s_min_{seed}")
        settings["fecal_v_max"] = st.number_input("파란색 V 최대", min_value=0, max_value=255, value=int(defaults["fecal_v_max"]), step=1, key=f"fecal_v_max_{seed}")
        settings["fecal_blackhat_thresh"] = st.number_input("fecal blackhat 임계값", min_value=0, max_value=255, value=int(defaults["fecal_blackhat_thresh"]), step=1, key=f"fecal_bh_{seed}")

        st.subheader("총대장균 전용")
        settings["total_yellow_h_min"] = st.number_input("노랑 H 최소", min_value=0, max_value=255, value=int(defaults["total_yellow_h_min"]), step=1, key=f"total_h_min_{seed}")
        settings["total_yellow_h_max"] = st.number_input("노랑 H 최대", min_value=0, max_value=255, value=int(defaults["total_yellow_h_max"]), step=1, key=f"total_h_max_{seed}")
        settings["total_yellow_s_min"] = st.number_input("노랑 S 최소", min_value=0, max_value=255, value=int(defaults["total_yellow_s_min"]), step=1, key=f"total_s_min_{seed}")
        settings["total_yellow_v_min"] = st.number_input("노랑 V 최소", min_value=0, max_value=255, value=int(defaults["total_yellow_v_min"]), step=1, key=f"total_v_min_{seed}")
        settings["total_dark_v_max"] = st.number_input("어두운 V 최대", min_value=0, max_value=255, value=int(defaults["total_dark_v_max"]), step=1, key=f"total_dark_v_{seed}")
        settings["total_blackhat_thresh"] = st.number_input("total blackhat 임계값", min_value=0, max_value=255, value=int(defaults["total_blackhat_thresh"]), step=1, key=f"total_bh_{seed}")

    return analysis_mode, settings



def main():
    st.set_page_config(page_title="대장균 배지 자동 계수 웹앱", layout="wide")
    st.title("대장균 배지 자동 계수 웹앱")
    st.caption("기존 Tkinter V4 구조를 Streamlit 웹앱으로 옮긴 버전")

    top_left, top_right = st.columns([1, 2])
    with top_left:
        analysis_target = st.selectbox("분석 대상", ["fecal_coliform", "total_coliform"], format_func=lambda x: "분원성대장균" if x == "fecal_coliform" else "총대장균")
    analysis_mode, settings = render_sidebar_settings(analysis_target)

    uploaded_file = st.file_uploader("배지 사진 업로드", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

    manual_expander = st.expander("수동 보정")
    with manual_expander:
        st.write("웹앱에서는 마우스 직접 클릭 대신 좌표 입력 방식으로 수동 추가/삭제를 지원합니다.")
        col1, col2, col3 = st.columns(3)
        with col1:
            manual_x = st.number_input("추가 X", min_value=0, value=0, step=1)
        with col2:
            manual_y = st.number_input("추가 Y", min_value=0, value=0, step=1)
        with col3:
            if st.button("수동 점 추가"):
                st.session_state.manual_points.append((int(manual_x), int(manual_y)))
        if st.session_state.manual_points:
            st.write("현재 수동 추가 예정 좌표:", st.session_state.manual_points)
            remove_index = st.number_input("삭제할 수동점 번호(1부터)", min_value=1, max_value=len(st.session_state.manual_points), value=1, step=1)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("선택 수동점 삭제"):
                    del st.session_state.manual_points[int(remove_index) - 1]
            with c2:
                if st.button("수동점 전체 초기화"):
                    st.session_state.manual_points = []

    if uploaded_file is None:
        st.info("이미지를 업로드하면 분석 결과, 디버그 이미지, CSV 다운로드를 사용할 수 있습니다.")
        return

    try:
        image_bgr = read_uploaded_image(uploaded_file)
    except Exception as e:
        st.error(str(e))
        return

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("원본 이미지")
        st.image(bgr_to_rgb(image_bgr), use_container_width=True)
    with col_b:
        st.subheader("안내")
        st.markdown(
            f"- 업로드 파일: **{uploaded_file.name}**\n"
            f"- 분석 대상: **{'분원성대장균' if analysis_target == 'fecal_coliform' else '총대장균'}**\n"
            f"- 분석 방식: **{analysis_mode}**\n"
            f"- 수동 추가 예정 점 수: **{len(st.session_state.manual_points)}개**"
        )

    if not st.button("분석 실행", type="primary"):
        if st.session_state.last_result is not None:
            st.info("설정을 바꿨다면 분석 실행 버튼을 다시 눌러 반영하세요.")
        return

    try:
        normalized_plate, plate_mask, circle_info, center_radius, plate_preview = detect_plate_and_normalize(image_bgr)
        debug_images = {"1_plate_normalized": plate_preview}

        if analysis_mode == "general":
            colonies, binary, extra_debug = detect_general(normalized_plate, plate_mask, center_radius, settings)
        else:
            if analysis_target == "fecal_coliform":
                colonies, binary, extra_debug = detect_fecal_coliform(normalized_plate, plate_mask, center_radius, settings)
            else:
                colonies, binary, extra_debug = detect_total_coliform(normalized_plate, plate_mask, center_radius, settings)
        debug_images.update(extra_debug)

        colonies = add_manual_points(colonies, st.session_state.manual_points)
        result_img = draw_colonies(normalized_plate.copy(), colonies, center_radius)
        debug_images["9_final_result"] = result_img.copy()

        df = colonies_to_dataframe(colonies)
        st.session_state.last_result = {
            "result_img": result_img,
            "binary": binary,
            "debug_images": debug_images,
            "colonies": colonies,
            "df": df,
            "circle_info": circle_info,
            "center_radius": center_radius,
            "analysis_target": analysis_target,
            "analysis_mode": analysis_mode,
            "uploaded_name": uploaded_file.name,
        }
    except Exception as e:
        st.error(f"처리 오류: {e}")
        return

    result = st.session_state.last_result
    x, y, r = result["circle_info"]
    st.success(f"분석 완료: 총 {len(result['colonies'])}개 집락 검출")

    info1, info2, info3 = st.columns(3)
    info1.metric("자동+수동 총 집락 수", len(result["colonies"]))
    info2.metric("원본 배지 반지름(px)", r)
    info3.metric("정규화 지름(px)", TARGET_PLATE_DIAMETER)

    st.markdown(
        f"원본 배지 중심은 **({x}, {y})**, 반지름은 **{r}px** 입니다.  \
공통 필터는 면적 **{settings['min_area']}~{settings['max_area']}**, 원형도 **≥ {settings['circularity']:.2f}**, 최소거리 **≥ {settings['min_distance']}** 를 사용했습니다."
    )

    v1, v2 = st.columns(2)
    with v1:
        st.subheader("결과 이미지")
        st.image(bgr_to_rgb(result["result_img"]), use_container_width=True)
    with v2:
        st.subheader("이진화 이미지")
        st.image(result["binary"], use_container_width=True, clamp=True)

    st.subheader("집락 데이터")
    st.dataframe(result["df"], use_container_width=True, hide_index=True)

    csv_bytes = dataframe_to_csv_bytes(result["df"], result["analysis_target"], result["analysis_mode"])
    img_bytes = image_to_bytes(result["result_img"])
    name_root = result["uploaded_name"].rsplit(".", 1)[0]
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("CSV 다운로드", data=csv_bytes, file_name=f"{name_root}_colony_result.csv", mime="text/csv")
    with d2:
        st.download_button("결과 이미지 다운로드", data=img_bytes, file_name=f"{name_root}_result.png", mime="image/png")

    with st.expander("디버그 보기", expanded=False):
        items = sorted(result["debug_images"].items(), key=lambda x: x[0])
        for i in range(0, len(items), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(items):
                    name, img = items[i + j]
                    with col:
                        st.markdown(f"**{name}**")
                        st.image(bgr_to_rgb(img), use_container_width=True)

    st.warning("현재 웹앱 버전의 수동 수정은 좌표 입력 방식입니다. 원본 Tkinter처럼 이미지 클릭 기반 수동 편집까지 하려면 별도 클릭 컴포넌트를 붙이면 됩니다.")


if __name__ == "__main__":
    main()
