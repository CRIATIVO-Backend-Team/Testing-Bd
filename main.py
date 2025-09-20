from fastapi import FastAPI, UploadFile, File,Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import imutils
import os, json, random, string, logging, threading, time
from typing import Dict, OrderedDict
from collections import OrderedDict
from PIL import Image, ImageOps
from typing import List, Dict, Tuple
from pymongo import MongoClient

# --- Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

mongo_url = "mongodb://localhost:27017/"
client = MongoClient(mongo_url)
db = client["omr_database"]
collection = db["omr_results"]

app = FastAPI(title="OMR Sheet Processor")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JSON_DIR = "omr-json"
UPLOAD_DIR = "uploads"
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Config ---
MIN_AREA = 250
ROW_Y_THRESHOLD = 20
HORIZONTAL_MARGIN = 200
FILL_THRESHOLD = 0.8




def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [-15, -15],
        [maxWidth + 10, -10],
        [maxWidth + 10, maxHeight + 10],
        [-10, maxHeight + 10]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def crop_sheet(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return four_point_transform(image, approx.reshape(4, 2))
    return image


def separate_circles_and_squares(cnts):
    circles, squares = [], []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        circularity = (4 * np.pi * area) / (peri ** 2 + 1e-6)
        if len(approx) == 4:
            squares.append(c)
        elif circularity >= 0.85 or len(approx) > 4:
            circles.append(c)
        else:
            squares.append(c)
    return circles, squares


def group_squares_into_rows(square_coords):
    if not square_coords:
        return []
    square_coords.sort(key=lambda s: s["cy"])
    rows = []
    current_row = [square_coords[0]]
    for sq in square_coords[1:]:
        if abs(sq["cy"] - current_row[-1]["cy"]) < ROW_Y_THRESHOLD:
            current_row.append(sq)
        else:
            current_row.sort(key=lambda s: s["cx"])
            rows.append(current_row)
            current_row = [sq]
    current_row.sort(key=lambda s: s["cx"])
    rows.append(current_row)
    return rows


def analyze_sheet(image: np.ndarray, debug_dir: str, input_id) -> Tuple[List[Dict], str]:

    """Analyze sheet and return results as flat list + labeled image path"""
    labeled_img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    circles, squares = separate_circles_and_squares(cnts)

    square_coords = []
    for s in squares:
        x, y, w, h = cv2.boundingRect(s)
        square_coords.append({"x": x, "y": y, "w": w, "h": h,
                              "cx": x + w // 2, "cy": y + h // 2})
        cv2.rectangle(labeled_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    rows = group_squares_into_rows(square_coords)

    circle_points = []
    for c in circles:
        (x, y), radius = cv2.minEnclosingCircle(c)
        cx, cy, r = int(x), int(y), int(radius)
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.circle(mask, (cx, cy), r, 255, -1)

        total_pixels = cv2.countNonZero(mask)
        white_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
        fill_ratio = white_pixels / (total_pixels + 1e-6)
        filled = fill_ratio >= FILL_THRESHOLD

        color = (0, 255, 0) if filled else (0, 0, 255)
        cv2.circle(labeled_img, (cx, cy), r, color, 2)

        circle_points.append({"x": cx, "y": cy, "r": r,
                              "fill_ratio": fill_ratio, "filled": filled})

    letters = list(string.ascii_uppercase)
    results = []

    for row_idx, row in enumerate(rows, start=1):
        row_circles = [c for c in circle_points if abs(c["y"] - row[0]["cy"]) < ROW_Y_THRESHOLD]
        for col_idx, sq in enumerate(row, start=1):
            left_circles = [c for c in row_circles if c["x"] < sq["cx"]
                            and abs(c["x"] - sq["cx"]) < (sq["w"] + HORIZONTAL_MARGIN)]
            left_circles.sort(key=lambda c: c["x"])

            new_label = f"{row_idx + (col_idx - 1) * 10}"
            cv2.putText(labeled_img, new_label, (sq["x"], sq["y"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            answers = []
            ratios = []
            for i, c in enumerate(left_circles):
                if i >= len(letters):
                    break
                cv2.putText(labeled_img, letters[i], (c["x"], c["y"]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                if c["filled"]:
                    answers.append(letters[i])
                    ratios.append(c["fill_ratio"])

            accuracy = round(sum(ratios) / len(ratios), 2) if ratios else 0.0
            results.append({
                "question": new_label,
                "answers": answers,
                "accuracy": accuracy
            })

    # Save annotated image
    labeled_path = os.path.join(debug_dir, f"{input_id}_annotated.png")
    cv2.imwrite(labeled_path, labeled_img)
    results.sort(key=lambda x: int(x["question"]))

    return results, labeled_path

def convert_mobile_to_pc(mobile_path, pc_out="pc_style.jpg"):
    # Open image
    image = Image.open(mobile_path)

    # Fix orientation (important for mobile EXIF)
    image = ImageOps.exif_transpose(image)

    # Resize to match PC style (720x1600 max)
    image.thumbnail((720, 1600), Image.Resampling.LANCZOS)

    # Strip metadata (make a clean copy)
    clean = Image.new(image.mode, image.size)
    clean.putdata(list(image.getdata()))

    # Save compressed JPEG (like PC export)
    clean.save(pc_out, "JPEG", quality=70, optimize=True)
    print(f"Converted: {pc_out}")
    return pc_out

    
@app.post("/process/{suc}/{examID}")
async def process_sheet(file: UploadFile = File(...), suc: str = "", examID: str = "") -> JSONResponse:
    temp_name = "input_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    input_path = os.path.join(UPLOAD_DIR, f"{temp_name}.jpg")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    convert_path = convert_mobile_to_pc(input_path, pc_out=input_path)

    image = cv2.imread(convert_path)
    if image is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    cropped = crop_sheet(image)
    result, labeled_path = analyze_sheet(cropped, UPLOAD_DIR,temp_name)

    # Save JSON result (optional, for audit/debugging only)
    json_path = os.path.join(JSON_DIR, f"{temp_name}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)

    logger.info(f"Processed: {temp_name}, JSON saved to {json_path}")
    logger.info("image_url:" f"/uploads/{os.path.basename(labeled_path)}")
    return JSONResponse(content={
        "image_url": f"/uploads/{os.path.basename(labeled_path)}",
        "suc": suc,
        "examID": examID,
        "result": result
    })
@app.post("/save_results")
async def save_results(payload: Dict = Body(...)) -> JSONResponse:
    try:
        collection.insert_one(payload)
        return JSONResponse(content={"message": "Results saved successfully"})
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to save results"})

@app.get("/")
def root():
    return "Backend is Running"
