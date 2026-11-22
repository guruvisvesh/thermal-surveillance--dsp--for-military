import os
import cv2
import time
import serial
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import firwin, lfilter, convolve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------- USER SETTINGS ----------------
DATASET_PATH = r"C:\Users\gurun\Desktop\New folder"   # Path to images
OUTPUT_CSV = "thermal_object_results_combined.csv"
FEATURES_CSV = "thermal_features.csv"
ARDUINO_PORT = 'COM6'                                 # Arduino port, or set to None to skip
BAUD_RATE = 9600
NUM_SENSOR_READINGS = 5
PROCESS_LIMIT = None    # Set to integer to limit processed images (None = all)
# ------------------------------------------------

def collect_sensor_data(port, baud_rate, num_readings):
    if port is None:
        return pd.DataFrame(columns=["Timestamp", "Ambient_Temp", "Object_Temp"])
    data = []
    try:
        ser = serial.Serial(port, baud_rate, timeout=2)
        time.sleep(4)  # Wait longer for Arduino to reset and start output
        ser.reset_input_buffer()
        for _ in range(10):  # skip some initial junk lines
            ser.readline()
        for i in range(num_readings):
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"Raw from serial: '{line}'")  # Debug print
            if line:
                # Use regex to extract both numbers (including decimal)
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                print(f"Extracted numbers: {matches}")  # Debug print
                if len(matches) >= 2:
                    try:
                        ambient = float(matches[0])
                        obj = float(matches[1])
                        data.append([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ambient, obj])
                        print(f"Sensor read {i+1}: Ambient={ambient:.2f}, Object={obj:.2f}")
                    except ValueError as e:
                        print("ValueError parsing floats:", e)
                else:
                    print("Could not extract two temperatures from:", line)
            time.sleep(0.15)
        ser.close()
    except Exception as e:
        print("Sensor read error:", e)
    return pd.DataFrame(data, columns=["Timestamp", "Ambient_Temp", "Object_Temp"])

def smooth_sensor(df):
    if df.empty:
        return df
    window_size = 5
    if len(df) < window_size:
        df["Filtered_Object_Temp"] = df["Object_Temp"]
        df["Filtered_Ambient_Temp"] = df["Ambient_Temp"]
        return df
    coeff = firwin(window_size, 0.3)
    df["Filtered_Object_Temp"] = lfilter(coeff, 1.0, df["Object_Temp"])
    df["Filtered_Ambient_Temp"] = lfilter(coeff, 1.0, df["Ambient_Temp"])
    return df

def process_image(file_path, target_size=(240, 320)):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read {file_path}")
    img = cv2.resize(img, target_size)
    img_eq = cv2.equalizeHist(img)
    return img, img_eq

def extract_image_features(img_eq, file_name):
    # 2D FFT
    f = np.fft.fft2(img_eq)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    kernel_smooth = np.ones((5,5)) / 25
    img_smooth = convolve(img_eq, kernel_smooth, mode='same')
    kernel_edge = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]])
    img_edges = np.abs(convolve(img_eq, kernel_edge, mode='same'))
    mean_temp = np.mean(img_eq)
    max_temp = np.max(img_eq)
    std_temp = np.std(img_eq)
    hot_pixels = np.sum(img_eq > 200)
    feature_row = {
        "Image": file_name,
        "Mean": mean_temp,
        "Max": max_temp,
        "StdDev": std_temp,
        "HotPixels": hot_pixels,
    }
    return feature_row, magnitude_spectrum, img_smooth, img_edges

def contour_features(cnt, img_eq):
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    rect_area = max(w*h, 1)
    extent = float(area) / rect_area
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull) if len(hull) > 2 else 0
    solidity = float(area) / hull_area if hull_area > 0 else 0.0
    perimeter = cv2.arcLength(cnt, True)
    aspect_ratio = float(h) / w if w > 0 else 0.0
    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
            if MA >= ma and MA > 0:
                major = MA / 2.0
                minor = ma / 2.0
            else:
                major = ma / 2.0
                minor = MA / 2.0
            if major > 0:
                eccentricity = np.sqrt(max(0.0, 1 - (minor/major)**2))
        except:
            eccentricity = 0.0
    mask = np.zeros_like(img_eq, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    if area > 0:
        vals = img_eq[mask==255]
        avg_int = float(np.mean(vals))
        std_int = float(np.std(vals))
    else:
        avg_int, std_int = 0.0, 0.0
    x1, y1, x2, y2 = x, y, x+w, y+h
    roi = img_eq[y1:y2, x1:x2]
    if roi.size == 0:
        hotspots = 0
        vs_ratio = 0.0
    else:
        thresh_val = np.percentile(roi, 90)
        _, hot_mask = cv2.threshold(roi, int(thresh_val), 255, cv2.THRESH_BINARY)
        num_labels, labels = cv2.connectedComponents(hot_mask)
        hotspots = max(0, num_labels - 1)
        cols = np.any(hot_mask, axis=0)
        rows = np.any(hot_mask, axis=1)
        hot_w = np.sum(cols)
        hot_h = np.sum(rows)
        hot_w = max(hot_w, 1)
        hot_h = max(hot_h, 1)
        vs_ratio = hot_h / float(hot_w)
    features = {
        "x": x, "y": y, "w": w, "h": h,
        "area": area, "aspect_ratio": aspect_ratio,
        "extent": extent, "solidity": solidity,
        "perimeter": perimeter, "eccentricity": eccentricity,
        "avg_intensity": avg_int, "std_intensity": std_int,
        "hotspots": hotspots, "vertical_hot_ratio": vs_ratio
    }
    return features

def heuristic_classify(feat, sensor_temp=None):
    score_h = 0
    score_v = 0
    if feat["area"] < 300:
        return {"label":"Ignore", "score_h":0, "score_v":0, "conf":0.0}
    if feat["aspect_ratio"] > 1.35: 
        score_h += 2
    elif feat["aspect_ratio"] < 1.05:
        score_v += 2
    if feat["vertical_hot_ratio"] > 1.2:
        score_h += 1
    elif feat["vertical_hot_ratio"] < 0.9:
        score_v += 1
    if feat["hotspots"] >= 2:
        score_v += 2
    elif feat["hotspots"] == 1:
        score_h += 1
    if feat["extent"] < 0.55:
        score_h += 1
    else:
        score_v += 1
    if feat["solidity"] < 0.88:
        score_h += 1
    else:
        score_v += 1
    if feat["eccentricity"] > 0.6:
        score_h += 1
    elif feat["eccentricity"] < 0.4:
        score_v += 1
    if feat["std_intensity"] > 18:
        score_v += 1
    confirmed = False
    if sensor_temp is not None and isinstance(sensor_temp, (int, float)):
        if abs(feat["avg_intensity"] - sensor_temp) < 12:
            confirmed = True
    if score_h > score_v:
        label = "Human"
    elif score_v > score_h:
        label = "Vehicle"
    else:
        label = "Unknown"
    conf = float(abs(score_h - score_v))
    if confirmed:
        conf += 2.0
    return {"label": label, "score_h": score_h, "score_v": score_v, "conf": conf, "confirmed": confirmed}

def train_model_if_possible(dataset_path):
    for f in ["thermal_features.csv", "thermal_features.xlsx", "thermal_features.xls"]:
        p = os.path.join(dataset_path, f)
        if os.path.exists(p):
            print("Found labeled features file:", f, "- training classifier...")
            if f.endswith(".csv"):
                df = pd.read_csv(p)
            else:
                df = pd.read_excel(p)
            expected = ["area","aspect_ratio","extent","solidity","eccentricity",
                        "avg_intensity","std_intensity","hotspots","vertical_hot_ratio"]
            if not set(expected).issubset(df.columns) or "Label" not in df.columns:
                print("Labeled file present but missing required columns; falling back to heuristic.")
                return None
            X = df[expected].fillna(0).values
            y = df["Label"].apply(lambda s: 1 if str(s).lower().startswith("h") else 0).values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            print("Trained RF. Test accuracy:", clf.score(X_test, y_test))
            return clf
    return None

def detect_objects_in_image(img_eq, sensor_temp=None, clf=None, debug=False):
    thresh_val = np.percentile(img_eq, 85)
    _, mask = cv2.threshold(img_eq, int(thresh_val), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        feat = contour_features(cnt, img_eq)
        if feat["area"] < 300:
            continue
        if clf is not None:
            X = np.array([[feat[k] for k in ["area","aspect_ratio","extent","solidity","eccentricity",
                                            "avg_intensity","std_intensity","hotspots","vertical_hot_ratio"]]])
            pred = clf.predict(X)[0]
            label = "Human" if pred == 1 else "Vehicle"
            prob = clf.predict_proba(X).max() if hasattr(clf, "predict_proba") else 0.5
            detection = {**feat, "Label": label, "Model_Prob": float(prob)}
            detections.append(detection)
        else:
            res = heuristic_classify(feat, sensor_temp)
            feat.update({
                "Label": res["label"],
                "Score_H": res["score_h"],
                "Score_V": res["score_v"],
                "Confidence": res["conf"],
                "Confirmed": res["confirmed"]
            })
            detections.append(feat)
    vis = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2BGR)
    for d in detections:
        x, y, w, h = int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])
        label = d.get("Label", "Obj")
        conf = d.get("Confidence", d.get("Model_Prob", 0))
        text = f"{label} ({conf:.1f})"
        color = (0,255,0) if ("Human" in label) else (0,128,255)
        cv2.rectangle(vis, (x,y), (x+w, y+h), color, 2)
        cv2.putText(vis, text, (x, max(0,y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if debug and len(detections) > 0:
        plt.figure(figsize=(6,8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Detections (improved)")
        plt.axis('off')
        plt.show()
    return detections, vis

def run_pipeline():
    clf = train_model_if_possible(DATASET_PATH)
    sensor_df = collect_sensor_data(ARDUINO_PORT, BAUD_RATE, NUM_SENSOR_READINGS) if ARDUINO_PORT else pd.DataFrame()
    sensor_df = smooth_sensor(sensor_df)
    print(sensor_df.head())       # DEBUG
    print(sensor_df.columns)      # DEBUG
    if not sensor_df.empty:
        if "Filtered_Object_Temp" in sensor_df.columns and not pd.isnull(sensor_df["Filtered_Object_Temp"].iloc[-1]):
            sensor_temp = float(sensor_df["Filtered_Object_Temp"].iloc[-1])
        elif "Object_Temp" in sensor_df.columns and not pd.isnull(sensor_df["Object_Temp"].iloc[-1]):
            sensor_temp = float(sensor_df["Object_Temp"].iloc[-1])
        else:
            sensor_temp = "NoSensor"
    else:
        sensor_temp = "NoSensor"
    print(f"Using sensor_temp: {sensor_temp}")
    img_files = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if PROCESS_LIMIT:
        img_files = img_files[:PROCESS_LIMIT]
    results = []
    features_for_csv = []
    for idx, fname in enumerate(sorted(img_files)):
        fpath = os.path.join(DATASET_PATH, fname)
        try:
            img_orig, img_eq = process_image(fpath)
        except Exception as e:
            print("Skipping", fname, ":", e)
            continue
        # --- Feature Extraction & Save for CSV (as image.py) ---
        # Resize for this feature extraction to 128x128 for compatibility
        img_resized = cv2.resize(img_orig, (128, 128))
        img_eq_small = cv2.equalizeHist(img_resized)
        f = np.fft.fft2(img_eq_small)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        kernel_smooth = np.ones((5,5)) / 25
        img_smooth = convolve(img_eq_small, kernel_smooth, mode='same')
        kernel_edge = np.array([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]])
        img_edges = np.abs(convolve(img_eq_small, kernel_edge, mode='same'))
        mean_temp = np.mean(img_eq_small)
        max_temp = np.max(img_eq_small)
        std_temp = np.std(img_eq_small)
        hot_pixels = np.sum(img_eq_small > 200)
        features_for_csv.append({
            "Image": fname,
            "Mean": mean_temp,
            "Max": max_temp,
            "StdDev": std_temp,
            "HotPixels": hot_pixels
        })
        if idx == 0:
            plt.figure(figsize=(8,6))
            plt.subplot(2,2,1)
            plt.imshow(img_resized, cmap='hot')
            plt.title("Original Thermal Image")
            plt.subplot(2,2,2)
            plt.imshow(img_eq_small, cmap='hot')
            plt.title("Histogram Equalized")
            plt.subplot(2,2,3)
            plt.imshow(magnitude_spectrum, cmap='gray')
            plt.title("2D DFT Spectrum")
            plt.subplot(2,2,4)
            plt.imshow(img_smooth, cmap='hot')
            plt.title("Smoothed Image")
            plt.tight_layout()
            plt.show()
            plt.figure()
            plt.imshow(img_edges, cmap='gray')
            plt.title("Edge Detection (Linear Convolution)")
            plt.show()
        # --- End of Feature Extraction for CSV ---
        img_feat_row, _, _, _ = extract_image_features(img_eq, fname)
        detections, vis = detect_objects_in_image(img_eq, sensor_temp=sensor_temp if isinstance(sensor_temp, (int, float)) else None, clf=clf, debug=True)
        if len(detections) == 0:
            print(f"[{idx+1}/{len(img_files)}] {fname}: No detections")
            img_feat_row.update({
                "Detection_Label": None, "Detection_Confidence": None,
                "Detection_x": None, "Detection_y": None, "Detection_w": None, "Detection_h": None,
                "Sensor_Object_Temp": sensor_temp
            })
            results.append(img_feat_row)
            continue
        for d in detections:
            row = dict(img_feat_row) # copy
            row.update({
                "Detection_Label": d.get("Label", "Unknown"),
                "Detection_Confidence": d.get("Confidence", d.get("Model_Prob", None)),
                "Detection_x": int(d["x"]),
                "Detection_y": int(d["y"]),
                "Detection_w": int(d["w"]),
                "Detection_h": int(d["h"]),
                "Detection_Area": float(d["area"]),
                "Detection_Aspect_Ratio": float(d["aspect_ratio"]),
                "Detection_Extent": float(d["extent"]),
                "Detection_Solidity": float(d["solidity"]),
                "Detection_Eccentricity": float(d["eccentricity"]),
                "Detection_Avg_Intensity": float(d["avg_intensity"]),
                "Detection_Std_Intensity": float(d["std_intensity"]),
                "Detection_Hotspots": int(d["hotspots"]),
                "Detection_Vertical_Hot_Ratio": float(d["vertical_hot_ratio"]),
                "Sensor_Object_Temp": sensor_temp
            })
            results.append(row)
    # Save features as in image.py
    feature_df = pd.DataFrame(features_for_csv)
    print("\n=== Image Features DataFrame (image.py style) ===")
    print(feature_df)
    feature_csv_path = os.path.join(DATASET_PATH, FEATURES_CSV)
    feature_df.to_csv(feature_csv_path, index=False)
    print(f"Features saved to {feature_csv_path}")
    # Save detection results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print("\n=== Detection Results DataFrame ===")
    print(df)
    print("Saved results to", OUTPUT_CSV)
    if not sensor_df.empty:
        sensor_df.to_csv("arduino_mlx90614_data.csv", index=False)


run_pipeline()