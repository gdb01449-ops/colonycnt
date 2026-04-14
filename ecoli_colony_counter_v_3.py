import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import csv


class ColonyCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("대장균 배지 자동 계수 프로그램 V4")
        self.root.geometry("1540x980")
        self.root.minsize(1100, 760)

        self.image_bgr = None
        self.original_path = None
        self.result_image = None
        self.current_plate_base = None
        self.current_plate_center_radius = None
        self.binary_image = None
        self.debug_images = {}
        self.photo_refs = {}
        self.colonies = []
        self.debug_window = None
        self.manual_add_mode = False
        self.manual_remove_mode = False

        self.canvas_image_offset = (0, 0)
        self.canvas_scale = 1.0
        self.target_plate_diameter = 900

        self.analysis_target = tk.StringVar(value="fecal_coliform")
        self.analysis_mode = tk.StringVar(value="targeted")
        self.detection_level = tk.StringVar(value="normal")

        # 공통 파라미터
        self.min_area_var = tk.IntVar(value=30)
        self.max_area_var = tk.IntVar(value=2500)
        self.circularity_var = tk.DoubleVar(value=0.35)
        self.min_distance_var = tk.IntVar(value=16)
        self.edge_margin_ratio_var = tk.DoubleVar(value=0.10)
        self.min_radius_var = tk.IntVar(value=4)
        self.max_radius_var = tk.IntVar(value=24)

        # 분원성대장균용(베이지 배지 + 파란 집락)
        self.fecal_h_min_var = tk.IntVar(value=90)
        self.fecal_h_max_var = tk.IntVar(value=165)
        self.fecal_s_min_var = tk.IntVar(value=35)
        self.fecal_v_max_var = tk.IntVar(value=170)
        self.fecal_blackhat_thresh_var = tk.IntVar(value=18)

        # 총대장균용(붉은 배지 + 노랑/갈색 집락)
        self.total_yellow_h_min_var = tk.IntVar(value=8)
        self.total_yellow_h_max_var = tk.IntVar(value=48)
        self.total_yellow_s_min_var = tk.IntVar(value=35)
        self.total_yellow_v_min_var = tk.IntVar(value=85)
        self.total_dark_v_max_var = tk.IntVar(value=110)
        self.total_blackhat_thresh_var = tk.IntVar(value=16)

        self._build_ui()
        self.apply_target_preset()

    def _build_ui(self):
        self.main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        self.left_panel = tk.Frame(self.main_paned, width=380)
        self.right_panel = tk.Frame(self.main_paned)
        self.main_paned.add(self.left_panel, minsize=320)
        self.main_paned.add(self.right_panel, minsize=700)

        self._build_left_panel()
        self._build_right_panel()

    def _build_left_panel(self):
        top_btn_frame = tk.Frame(self.left_panel)
        top_btn_frame.pack(fill=tk.X, padx=8, pady=8)

        btns = [
            ("사진 불러오기", self.load_image),
            ("분석 실행", self.process_image),
            ("디버그 보기", self.show_debug_window),
            ("자동 설정 적용", self.apply_target_preset),
            ("결과 이미지 저장", self.save_result_image),
            ("결과 CSV 저장", self.save_csv),
            ("수동 추가", self.enable_add_mode),
            ("수동 삭제", self.enable_remove_mode),
            ("수동 모드 해제", self.disable_manual_mode),
        ]
        for i, (text, cmd) in enumerate(btns):
            tk.Button(top_btn_frame, text=text, command=cmd, width=16).grid(row=i // 2, column=i % 2, padx=4, pady=4, sticky="ew")
        top_btn_frame.grid_columnconfigure(0, weight=1)
        top_btn_frame.grid_columnconfigure(1, weight=1)

        settings_outer = tk.LabelFrame(self.left_panel, text="간단 설정")
        settings_outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.settings_canvas = tk.Canvas(settings_outer, highlightthickness=0)
        self.settings_scrollbar = ttk.Scrollbar(settings_outer, orient="vertical", command=self.settings_canvas.yview)
        self.settings_inner = tk.Frame(self.settings_canvas)
        self.settings_window_id = self.settings_canvas.create_window((0, 0), window=self.settings_inner, anchor="nw")

        self.settings_inner.bind("<Configure>", lambda e: self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all")))
        self.settings_canvas.configure(yscrollcommand=self.settings_scrollbar.set)
        self.settings_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.settings_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.settings_canvas.bind("<Configure>", self._on_settings_canvas_resize)

        row = 0
        tk.Label(self.settings_inner, text="분석 대상", anchor="w").grid(row=row, column=0, sticky="ew", padx=8, pady=8)
        tk.OptionMenu(self.settings_inner, self.analysis_target, "fecal_coliform", "total_coliform", command=lambda _: self.apply_target_preset()).grid(row=row, column=1, sticky="ew", padx=8, pady=8)
        row += 1

        tk.Label(self.settings_inner, text="검출 강도", anchor="w").grid(row=row, column=0, sticky="ew", padx=8, pady=8)
        tk.OptionMenu(self.settings_inner, self.detection_level, "conservative", "normal", "sensitive", command=lambda _: self.apply_target_preset()).grid(row=row, column=1, sticky="ew", padx=8, pady=8)
        row += 1

        help_box = tk.LabelFrame(self.settings_inner, text="사용 가이드")
        help_box.grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=10)
        tk.Label(
            help_box,
            text=(
                "• 베이지 배지 → fecal_coliform
"
                "• 붉은/분홍 배지 → total_coliform
"
                "• conservative: 과검출을 줄임
"
                "• normal: 기본 추천
"
                "• sensitive: 누락을 줄임
"
                "• 세부값은 내부에서 자동 설정됩니다"
            ),
            justify="left",
            anchor="w",
            wraplength=300,
        ).pack(fill=tk.X, padx=8, pady=8)
        row += 1

        self.settings_inner.grid_columnconfigure(0, weight=1)
        self.settings_inner.grid_columnconfigure(1, weight=1)

        self.info_label = tk.Label(self.left_panel, text="사진을 불러오세요.", anchor="nw", justify="left", wraplength=340)
        self.info_label.pack(fill=tk.X, padx=10, pady=(0, 10))

    def _build_right_panel(self):
        viewer_frame = tk.Frame(self.right_panel)
        viewer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas = tk.Canvas(viewer_frame, bg="gray88")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", lambda e: self.refresh_main_view())

    def _on_settings_canvas_resize(self, event):
        self.settings_canvas.itemconfig(self.settings_window_id, width=event.width)

    def refresh_main_view(self):
        if self.result_image is not None:
            self.show_image(self.result_image)
        elif self.image_bgr is not None:
            self.show_image(self.image_bgr)

    def apply_target_preset(self):
        target = self.analysis_target.get()
        level = self.detection_level.get()

        if target == "fecal_coliform":
            if level == "conservative":
                self.min_area_var.set(45)
                self.max_area_var.set(1600)
                self.circularity_var.set(0.45)
                self.min_distance_var.set(16)
                self.edge_margin_ratio_var.set(0.12)
                self.min_radius_var.set(5)
                self.max_radius_var.set(18)
                self.fecal_h_min_var.set(95)
                self.fecal_h_max_var.set(160)
                self.fecal_s_min_var.set(45)
                self.fecal_v_max_var.set(155)
                self.fecal_blackhat_thresh_var.set(20)
            elif level == "sensitive":
                self.min_area_var.set(20)
                self.max_area_var.set(2200)
                self.circularity_var.set(0.25)
                self.min_distance_var.set(10)
                self.edge_margin_ratio_var.set(0.08)
                self.min_radius_var.set(3)
                self.max_radius_var.set(22)
                self.fecal_h_min_var.set(85)
                self.fecal_h_max_var.set(170)
                self.fecal_s_min_var.set(25)
                self.fecal_v_max_var.set(185)
                self.fecal_blackhat_thresh_var.set(15)
            else:
                self.min_area_var.set(30)
                self.max_area_var.set(1800)
                self.circularity_var.set(0.35)
                self.min_distance_var.set(14)
                self.edge_margin_ratio_var.set(0.10)
                self.min_radius_var.set(4)
                self.max_radius_var.set(20)
                self.fecal_h_min_var.set(90)
                self.fecal_h_max_var.set(165)
                self.fecal_s_min_var.set(35)
                self.fecal_v_max_var.set(170)
                self.fecal_blackhat_thresh_var.set(18)
        else:
            if level == "conservative":
                self.min_area_var.set(18)
                self.max_area_var.set(900)
                self.circularity_var.set(0.28)
                self.min_distance_var.set(10)
                self.edge_margin_ratio_var.set(0.10)
                self.min_radius_var.set(3)
                self.max_radius_var.set(10)
                self.total_yellow_h_min_var.set(10)
                self.total_yellow_h_max_var.set(42)
                self.total_yellow_s_min_var.set(45)
                self.total_yellow_v_min_var.set(95)
                self.total_dark_v_max_var.set(90)
                self.total_blackhat_thresh_var.set(20)
            elif level == "sensitive":
                self.min_area_var.set(8)
                self.max_area_var.set(1500)
                self.circularity_var.set(0.10)
                self.min_distance_var.set(6)
                self.edge_margin_ratio_var.set(0.05)
                self.min_radius_var.set(2)
                self.max_radius_var.set(16)
                self.total_yellow_h_min_var.set(5)
                self.total_yellow_h_max_var.set(55)
                self.total_yellow_s_min_var.set(25)
                self.total_yellow_v_min_var.set(75)
                self.total_dark_v_max_var.set(125)
                self.total_blackhat_thresh_var.set(12)
            else:
                self.min_area_var.set(12)
                self.max_area_var.set(1200)
                self.circularity_var.set(0.18)
                self.min_distance_var.set(8)
                self.edge_margin_ratio_var.set(0.07)
                self.min_radius_var.set(2)
                self.max_radius_var.set(14)
                self.total_yellow_h_min_var.set(8)
                self.total_yellow_h_max_var.set(48)
                self.total_yellow_s_min_var.set(35)
                self.total_yellow_v_min_var.set(85)
                self.total_dark_v_max_var.set(110)
                self.total_blackhat_thresh_var.set(16)

        target_label = "분원성대장균" if target == "fecal_coliform" else "총대장균"
        level_label = {"conservative": "보수적", "normal": "기본", "sensitive": "민감"}[level]
        self.info_label.config(
            text=(
                f"자동 설정 적용 완료
"
                f"- 분석 대상: {target_label}
"
                f"- 검출 강도: {level_label}
"
                f"- 베이지 배지=분원성, 붉은 배지=총대장균
"
                f"- 세부 HSV/면적/반지름 값은 내부 자동 적용"
            )
        )

    def load_image(self):
        path = filedialog.askopenfilename(
            title="배지 사진 선택",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not path:
            return
        file_bytes = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("오류", "이미지를 불러올 수 없습니다.")
            return
        self.image_bgr = img
        self.original_path = path
        self.result_image = None
        self.binary_image = None
        self.current_plate_base = None
        self.current_plate_center_radius = None
        self.colonies = []
        self.debug_images = {}
        self.show_image(self.image_bgr)
        self.info_label.config(text=f"불러온 파일: {os.path.basename(path)}\n분석 대상 선택 후 분석 실행하세요.")

    def show_image(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)
        scale = min(canvas_w / w, canvas_h / h)
        self.canvas_scale = scale
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        self.canvas_image_offset = (x_offset, y_offset)
        pil_img = Image.fromarray(resized)
        self.photo_refs["main"] = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(x_offset, y_offset, image=self.photo_refs["main"], anchor=tk.NW)

    def detect_plate_and_normalize(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(gray.shape[:2]) // 2,
            param1=120, param2=30,
            minRadius=min(gray.shape[:2]) // 5, maxRadius=min(gray.shape[:2]) // 2,
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
        scale = self.target_plate_diameter / max(2 * local_r, 1)
        new_w = max(int(crop_w * scale), 1)
        new_h = max(int(crop_h * scale), 1)
        normalized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        mask = np.zeros((new_h, new_w), dtype=np.uint8)
        center = (new_w // 2, new_h // 2)
        new_r = min(self.target_plate_diameter // 2, min(new_w, new_h) // 2 - 2)
        cv2.circle(mask, center, new_r, 255, -1)

        plate_preview = normalized.copy()
        cv2.circle(plate_preview, center, new_r, (255, 255, 0), 2)
        self.debug_images["1_plate_normalized"] = plate_preview
        return normalized, mask, (x, y, r), (center[0], center[1], new_r)

    def detect_fecal_coliform(self, plate_bgr, plate_mask, plate_center_radius):
        hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (7, 7), 0)
        blackhat = cv2.morphologyEx(blur_gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))

        blue_mask = cv2.inRange(h, self.fecal_h_min_var.get(), self.fecal_h_max_var.get())
        sat_mask = cv2.inRange(s, self.fecal_s_min_var.get(), 255)
        dark_mask = cv2.inRange(v, 0, self.fecal_v_max_var.get())
        bh_mask = cv2.inRange(blackhat, self.fecal_blackhat_thresh_var.get(), 255)

        binary = cv2.bitwise_and(blue_mask, sat_mask)
        binary = cv2.bitwise_and(binary, dark_mask)
        binary = cv2.bitwise_or(binary, cv2.bitwise_and(bh_mask, cv2.bitwise_and(sat_mask, dark_mask)))
        binary = cv2.bitwise_and(binary, binary, mask=plate_mask)

        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=2)

        self.debug_images["2_fecal_blue_mask"] = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
        self.debug_images["3_fecal_blackhat"] = cv2.cvtColor(blackhat, cv2.COLOR_GRAY2BGR)
        self.debug_images["4_fecal_binary"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return self.extract_candidates(plate_bgr, binary, plate_mask, plate_center_radius, mode_name="fecal")

    def detect_total_coliform(self, plate_bgr, plate_mask, plate_center_radius):
        hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (7, 7), 0)
        blackhat = cv2.morphologyEx(blur_gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

        yellow_h = cv2.inRange(h, self.total_yellow_h_min_var.get(), self.total_yellow_h_max_var.get())
        yellow_s = cv2.inRange(s, self.total_yellow_s_min_var.get(), 255)
        yellow_v = cv2.inRange(v, self.total_yellow_v_min_var.get(), 255)
        yellow_mask = cv2.bitwise_and(yellow_h, yellow_s)
        yellow_mask = cv2.bitwise_and(yellow_mask, yellow_v)

        dark_mask = cv2.inRange(v, 0, self.total_dark_v_max_var.get())
        bh_mask = cv2.inRange(blackhat, self.total_blackhat_thresh_var.get(), 255)

        binary = cv2.bitwise_or(yellow_mask, cv2.bitwise_and(dark_mask, bh_mask))
        binary = cv2.bitwise_and(binary, binary, mask=plate_mask)

        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel3, iterations=1)

        self.debug_images["2_total_yellow_mask"] = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
        self.debug_images["3_total_blackhat"] = cv2.cvtColor(blackhat, cv2.COLOR_GRAY2BGR)
        self.debug_images["4_total_binary"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return self.extract_candidates(plate_bgr, binary, plate_mask, plate_center_radius, mode_name="total")

    def extract_candidates(self, plate_bgr, binary, plate_mask, plate_center_radius, mode_name="target"):
        hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
        h_img, s_img, v_img = cv2.split(hsv)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = plate_bgr.copy()
        result[plate_mask == 0] = 0
        cv2.circle(result, (plate_center_radius[0], plate_center_radius[1]), plate_center_radius[2], (255, 255, 0), 2)

        colonies = []
        edge_margin = int(plate_center_radius[2] * self.edge_margin_ratio_var.get())
        valid_radius_limit = plate_center_radius[2] - edge_margin
        candidate_debug = result.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_var.get() or area > self.max_area_var.get():
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity < self.circularity_var.get():
                continue
            (cx_f, cy_f), radius = cv2.minEnclosingCircle(cnt)
            cx, cy = int(cx_f), int(cy_f)
            if radius < self.min_radius_var.get() or radius > self.max_radius_var.get():
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
                if (c["cx"] - cx) ** 2 + (c["cy"] - cy) ** 2 < self.min_distance_var.get() ** 2:
                    too_close = True
                    if area > c["area"]:
                        c.update({
                            "cx": cx, "cy": cy, "area": float(area), "circularity": float(circularity),
                            "contour": cnt, "mean_h": float(mean_h), "mean_s": float(mean_s),
                            "mean_v": float(mean_v), "radius": float(radius), "manual": False
                        })
                    break
            if too_close:
                continue

            colonies.append({
                "cx": cx, "cy": cy, "area": float(area), "circularity": float(circularity),
                "contour": cnt, "mean_h": float(mean_h), "mean_s": float(mean_s),
                "mean_v": float(mean_v), "radius": float(radius), "manual": False
            })
            cv2.drawContours(candidate_debug, [cnt], -1, (0, 255, 255), 2)
            cv2.putText(candidate_debug, f"H{int(mean_h)} S{int(mean_s)} V{int(mean_v)}", (cx + 4, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

        self.debug_images[f"5_{mode_name}_candidates"] = candidate_debug
        colonies = sorted(colonies, key=lambda item: (item["cy"], item["cx"]))
        self.draw_colonies(result, colonies, plate_center_radius)
        self.binary_image = binary
        return self.result_image, binary, colonies

    def detect_general(self, plate_bgr, plate_mask, plate_center_radius):
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
        self.debug_images["2_general_corrected"] = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
        self.debug_images["3_general_binary"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return self.extract_candidates(plate_bgr, binary, plate_mask, plate_center_radius, mode_name="general")

    def draw_colonies(self, base_img, colonies, plate_center_radius):
        img = base_img.copy()
        cv2.circle(img, (plate_center_radius[0], plate_center_radius[1]), plate_center_radius[2], (255, 255, 0), 2)
        for idx, colony in enumerate(colonies, start=1):
            if colony.get("contour") is not None:
                cv2.drawContours(img, [colony["contour"]], -1, (0, 255, 0), 2)
            color = (0, 0, 255) if not colony.get("manual", False) else (255, 0, 255)
            cv2.circle(img, (colony["cx"], colony["cy"]), 4, color, -1)
            cv2.putText(img, str(idx), (colony["cx"] + 6, colony["cy"] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 50, 50), 1, cv2.LINE_AA)
        self.result_image = img
        self.current_plate_base = base_img.copy()
        self.debug_images["9_final_result"] = img.copy()

    def process_image(self):
        if self.image_bgr is None:
            messagebox.showwarning("알림", "먼저 사진을 불러오세요.")
            return
        try:
            self.debug_images = {}
            normalized_plate, plate_mask, circle_info, center_radius = self.detect_plate_and_normalize(self.image_bgr)
            self.current_plate_center_radius = center_radius

            if self.analysis_mode.get() == "general":
                _, _, colonies = self.detect_general(normalized_plate, plate_mask, center_radius)
            else:
                if self.analysis_target.get() == "fecal_coliform":
                    _, _, colonies = self.detect_fecal_coliform(normalized_plate, plate_mask, center_radius)
                else:
                    _, _, colonies = self.detect_total_coliform(normalized_plate, plate_mask, center_radius)

            self.colonies = colonies
            self.show_image(self.result_image)

            x, y, r = circle_info
            target_name = "분원성대장균" if self.analysis_target.get() == "fecal_coliform" else "총대장균"
            sample_note = "베이지 배지=분원성, 붉은 배지=총대장균" if self.analysis_mode.get() == "targeted" else "일반 모드"
            self.info_label.config(
                text=(
                    f"V4 분석 완료\n"
                    f"- 분석 대상: {target_name}\n"
                    f"- 분석 방식: {self.analysis_mode.get()}\n"
                    f"- 원본 배지 중심: ({x}, {y}), 반지름: {r}px\n"
                    f"- 정규화 지름: {self.target_plate_diameter}px\n"
                    f"- 자동 검출 집락 수: {len(self.colonies)}개\n"
                    f"- 검출 강도: {self.detection_level.get()}
"
                    f"- 메모: {sample_note}
"
                    f"- 수동 추가: 보라색 점 / 자동 검출: 빨간 점"
                )
            )
            if self.debug_window and self.debug_window.winfo_exists():
                self.populate_debug_window()
        except Exception as e:
            messagebox.showerror("처리 오류", str(e))

    def enable_add_mode(self):
        self.manual_add_mode = True
        self.manual_remove_mode = False
        self.info_label.config(text=self.info_label.cget("text") + "\n현재 모드: 수동 추가")

    def enable_remove_mode(self):
        self.manual_add_mode = False
        self.manual_remove_mode = True
        self.info_label.config(text=self.info_label.cget("text") + "\n현재 모드: 수동 삭제")

    def disable_manual_mode(self):
        self.manual_add_mode = False
        self.manual_remove_mode = False

    def canvas_to_image_coords(self, event_x, event_y):
        x_offset, y_offset = self.canvas_image_offset
        x = int((event_x - x_offset) / self.canvas_scale)
        y = int((event_y - y_offset) / self.canvas_scale)
        return x, y

    def on_canvas_click(self, event):
        if self.result_image is None or self.current_plate_center_radius is None:
            return
        x, y = self.canvas_to_image_coords(event.x, event.y)
        h, w = self.result_image.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        cx0, cy0, r0 = self.current_plate_center_radius
        if (x - cx0) ** 2 + (y - cy0) ** 2 > r0 ** 2:
            return

        if self.manual_add_mode:
            self.colonies.append({
                "cx": x, "cy": y, "area": 0.0, "circularity": 1.0, "contour": None,
                "mean_h": 0.0, "mean_s": 0.0, "mean_v": 0.0, "radius": 0.0, "manual": True
            })
            self.redraw_after_manual_edit()
        elif self.manual_remove_mode and self.colonies:
            distances = [((c["cx"] - x) ** 2 + (c["cy"] - y) ** 2, idx) for idx, c in enumerate(self.colonies)]
            _, idx_min = min(distances, key=lambda t: t[0])
            del self.colonies[idx_min]
            self.redraw_after_manual_edit()

    def redraw_after_manual_edit(self):
        if self.current_plate_base is None or self.current_plate_center_radius is None:
            return
        self.draw_colonies(self.current_plate_base, self.colonies, self.current_plate_center_radius)
        self.show_image(self.result_image)
        self.info_label.config(text=f"수동 수정 후 집락 수: {len(self.colonies)}개")
        if self.debug_window and self.debug_window.winfo_exists():
            self.populate_debug_window()

    def show_debug_window(self):
        if not self.debug_window or not self.debug_window.winfo_exists():
            self.debug_window = tk.Toplevel(self.root)
            self.debug_window.title("디버그 화면")
            self.debug_window.geometry("1300x900")
            self.debug_window.minsize(900, 650)
            self.debug_canvas = tk.Canvas(self.debug_window)
            self.debug_scroll_y = ttk.Scrollbar(self.debug_window, orient="vertical", command=self.debug_canvas.yview)
            self.debug_scroll_x = ttk.Scrollbar(self.debug_window, orient="horizontal", command=self.debug_canvas.xview)
            self.debug_frame = tk.Frame(self.debug_canvas)
            self.debug_window_id = self.debug_canvas.create_window((0, 0), window=self.debug_frame, anchor="nw")
            self.debug_frame.bind("<Configure>", lambda e: self.debug_canvas.configure(scrollregion=self.debug_canvas.bbox("all")))
            self.debug_canvas.configure(yscrollcommand=self.debug_scroll_y.set, xscrollcommand=self.debug_scroll_x.set)
            self.debug_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.debug_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
            self.debug_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
            self.debug_canvas.bind("<Configure>", lambda e: self.debug_canvas.itemconfig(self.debug_window_id, width=e.width))
        self.populate_debug_window()

    def populate_debug_window(self):
        if not self.debug_window or not self.debug_window.winfo_exists():
            return
        for widget in self.debug_frame.winfo_children():
            widget.destroy()
        if not self.debug_images:
            tk.Label(self.debug_frame, text="먼저 분석을 실행하세요.").pack(padx=20, pady=20)
            return

        items = sorted(self.debug_images.items(), key=lambda x: x[0])
        cols = 2
        thumb_w = 520
        for idx, (name, img_bgr) in enumerate(items):
            row = idx // cols
            col = idx % cols
            panel = tk.LabelFrame(self.debug_frame, text=name)
            panel.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            scale = min(thumb_w / w, 320 / h)
            new_w = max(int(w * scale), 1)
            new_h = max(int(h * scale), 1)
            resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            pil_img = Image.fromarray(resized)
            key = f"debug_{idx}"
            self.photo_refs[key] = ImageTk.PhotoImage(pil_img)
            tk.Label(panel, image=self.photo_refs[key]).pack(padx=6, pady=6)
        for c in range(cols):
            self.debug_frame.grid_columnconfigure(c, weight=1)

    def save_result_image(self):
        if self.result_image is None:
            messagebox.showwarning("알림", "먼저 분석을 실행하세요.")
            return
        path = filedialog.asksaveasfilename(title="결과 이미지 저장", defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        ext = os.path.splitext(path)[1]
        success, encoded = cv2.imencode(ext, self.result_image)
        if not success:
            messagebox.showerror("오류", "이미지 저장에 실패했습니다.")
            return
        encoded.tofile(path)
        messagebox.showinfo("저장 완료", f"결과 이미지를 저장했습니다.\n{path}")

    def save_csv(self):
        if not self.colonies:
            messagebox.showwarning("알림", "저장할 집락 데이터가 없습니다.")
            return
        default_name = "colony_count_result.csv"
        if self.original_path:
            base = os.path.splitext(os.path.basename(self.original_path))[0]
            default_name = f"{base}_colony_result.csv"
        path = filedialog.asksaveasfilename(title="CSV 저장", defaultextension=".csv", initialfile=default_name, filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["번호", "중심X", "중심Y", "면적", "원형도", "수동추가여부", "평균H", "평균S", "평균V", "반지름"])
            for idx, c in enumerate(self.colonies, start=1):
                writer.writerow([
                    idx, c.get("cx", 0), c.get("cy", 0), round(c.get("area", 0.0), 2), round(c.get("circularity", 0.0), 4),
                    "Y" if c.get("manual") else "N", round(c.get("mean_h", 0.0), 2), round(c.get("mean_s", 0.0), 2),
                    round(c.get("mean_v", 0.0), 2), round(c.get("radius", 0.0), 2)
                ])
            writer.writerow([])
            writer.writerow(["총 집락 수", len(self.colonies)])
            writer.writerow(["분석 대상", self.analysis_target.get()])
            writer.writerow(["분석 방식", self.analysis_mode.get()])
        messagebox.showinfo("저장 완료", f"CSV를 저장했습니다.\n{path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ColonyCounterApp(root)
    root.mainloop()
