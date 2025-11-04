import os
from pathlib import Path

import cv2
import numpy as np

# -------- CONFIG: tweak for your data -------- #

# Morphology kernel for connecting characters within a line
LINE_DILATE_KERNEL = (50, 3)   # (width, height) -> bigger width joins more horizontally

MIN_LINE_HEIGHT = 15           # ignore very small blobs
MIN_LINE_WIDTH = 50            # ignore vertical noise

LINE_PADDING = 5               # pixels of padding above/below (and a bit left/right)

# For splitting multi-line bands
MULTILINE_HEIGHT_FACTOR = 1.5  # if box_height > 1.5 * median_height => try to split
LOCAL_HIST_THRESH_RATIO = 0.15 # inside a band, row sum threshold for "text"

# -------- PREPROCESSING -------- #

def preprocess_image(image_bgr):
    """
    Convert to grayscale, denoise slightly, then adaptive binarization.
    Returns:
        gray: grayscale image
        binary: binary image with text as white (255) on black (0)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        gray_blur,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,  # text becomes white
        blockSize=31,
        C=15,
    )
    return gray, binary

# -------- LINE DETECTION BY MORPHOLOGY -------- #

def coarse_line_boxes(binary):
    """
    Use horizontal dilation + connected components to get coarse line bands.
    Returns list of (x1, y1, x2, y2) in image coordinates.
    """
    h, w = binary.shape

    # Dilate horizontally to join characters of each line
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, LINE_DILATE_KERNEL)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw >= MIN_LINE_WIDTH and bh >= MIN_LINE_HEIGHT:
            boxes.append((x, y, x + bw, y + bh))

    # Sort by top coordinate (top-to-bottom)
    boxes.sort(key=lambda b: b[1])
    return boxes

# -------- SPLIT TALL (MULTI-LINE) BANDS -------- #

def split_multiline_boxes(binary, boxes):
    """
    Given the binary image and a list of coarse boxes, split those that are
    suspiciously tall into multiple single-line boxes using local horizontal
    projection inside the band.

    Returns new list of (x1, y1, x2, y2).
    """
    if not boxes:
        return []

    heights = [y2 - y1 for (x1, y1, x2, y2) in boxes]
    median_height = np.median(heights)

    new_boxes = []

    for (x1, y1, x2, y2) in boxes:
        band_height = y2 - y1

        # If height is reasonable, keep as-is
        if band_height <= MULTILINE_HEIGHT_FACTOR * median_height:
            new_boxes.append((x1, y1, x2, y2))
            continue

        # Otherwise, try to split: local horizontal projection
        band = binary[y1:y2, x1:x2]  # ROI
        h_band, w_band = band.shape

        hist = np.sum(band // 255, axis=1)  # row-wise sum of white pixels
        if np.max(hist) == 0:
            # No text? just keep the original box
            new_boxes.append((x1, y1, x2, y2))
            continue

        thresh = LOCAL_HIST_THRESH_RATIO * np.max(hist)
        mask = hist > thresh  # True where row contains text

        # Find contiguous text segments within band
        in_segment = False
        seg_start = 0
        segments = []

        for i, val in enumerate(mask):
            if val and not in_segment:
                in_segment = True
                seg_start = i
            elif not val and in_segment:
                in_segment = False
                seg_end = i
                if seg_end - seg_start >= MIN_LINE_HEIGHT:
                    segments.append((seg_start, seg_end))

        if in_segment:
            seg_end = h_band
            if seg_end - seg_start >= MIN_LINE_HEIGHT:
                segments.append((seg_start, seg_end))

        # If we couldn't find clear segments, keep original box
        if len(segments) <= 1:
            new_boxes.append((x1, y1, x2, y2))
            continue

        # For each local segment, create sub-box
        for (sy, ey) in segments:
            sub_y1 = y1 + sy
            sub_y2 = y1 + ey
            new_boxes.append((x1, sub_y1, x2, sub_y2))

    # Sort final boxes top-to-bottom
    new_boxes.sort(key=lambda b: b[1])
    return new_boxes

# -------- MAIN PIPELINE: PAGE → SINGLE LINE IMAGES -------- #

def split_page_to_single_lines(
    image_path,
    output_dir="lines_strict",
    save_debug=True,
):
    image_path = Path(image_path)
    base_name = image_path.stem

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image {image_path}")

    gray, binary = preprocess_image(img_bgr)
    H, W = binary.shape

    # 1. Coarse bands via morphology
    coarse_boxes = coarse_line_boxes(binary)

    # 2. Split tall bands into single lines
    line_boxes = split_multiline_boxes(binary, coarse_boxes)

    print(f"Coarse bands: {len(coarse_boxes)}, final lines: {len(line_boxes)}")

    # Prepare output dir
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_img = img_bgr.copy() if save_debug else None

    saved = []
    for idx, (x1, y1, x2, y2) in enumerate(line_boxes):
        # Add padding
        x1p = max(x1 - LINE_PADDING, 0)
        x2p = min(x2 + LINE_PADDING, W - 1)
        y1p = max(y1 - LINE_PADDING, 0)
        y2p = min(y2 + LINE_PADDING, H - 1)

        line_img = img_bgr[y1p:y2p, x1p:x2p]

        out_path = out_dir / f"{base_name}_line_{idx:02d}.png"
        cv2.imwrite(str(out_path), line_img)
        saved.append(out_path)

        if save_debug:
            cv2.rectangle(
                debug_img,
                (x1p, y1p),
                (x2p, y2p),
                (0, 0, 255),
                2,
            )

    if save_debug:
        debug_path = image_path.with_name(f"{base_name}_debug_lines_strict.png")
        cv2.imwrite(str(debug_path), debug_img)
        print(f"Saved debug overlay: {debug_path}")

    print(f"Saved {len(saved)} single-line images to {out_dir}")
    return saved

if __name__ == "__main__":
    PAGE_PATH = "a4_handwriting.png"  # <-- your scanned page here
    split_page_to_single_lines(PAGE_PATH)