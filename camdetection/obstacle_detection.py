import time
import argparse
import cv2
import numpy as np


def sector_free_ratios(occ_mask, grid_rows=2, grid_cols=2):
    """Calculate free space ratio for each grid cell (supports 2x2, 3x1, etc.)"""
    h, w = occ_mask.shape
    ratios = []
    sh = h // grid_rows
    sw = w // grid_cols
    for row in range(grid_rows):
        for col in range(grid_cols):
            y1 = row * sh
            y2 = h if row == grid_rows - 1 else (row + 1) * sh
            x1 = col * sw
            x2 = w if col == grid_cols - 1 else (col + 1) * sw
            sector = occ_mask[y1:y2, x1:x2]
            occupied = np.count_nonzero(sector)
            total = sector.size
            free_ratio = 1.0 - (occupied / total)
            ratios.append((free_ratio, x1, y1, x2, y2, row, col))
    return ratios


def human_label(r):
    if r > 0.7:
        return "CLEAR"
    if r > 0.4:
        return "PARTIAL"
    return "BLOCKED"


def run_camera(camera_index=0, width=640, grid_rows=2, grid_cols=2, show=True):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    # Try to set resolution (not guaranteed on all cameras)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

    last_print = 0.0
    print_interval = 0.5  # seconds

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # resize to target width while keeping aspect
            h0, w0 = frame.shape[:2]
            if w0 != width:
                scale = width / float(w0)
                frame = cv2.resize(frame, (width, int(h0 * scale)))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)

            fg = backSub.apply(blur)

            # post-process mask to clean noise
            _, th = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

            # occupancy mask: 1 where occupied, 0 where free
            occ_mask = (clean > 0).astype(np.uint8)

            ratios = sector_free_ratios(occ_mask, grid_rows=grid_rows, grid_cols=grid_cols)

            # detect pink color regions in the frame and determine which sector they are in
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # HSV range for pink/magenta (may need tuning per camera/lighting)
            lower_pink = np.array([140, 30, 30])
            upper_pink = np.array([170, 255, 255])
            pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
            # clean up mask
            pm = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            pm = cv2.morphologyEx(pm, cv2.MORPH_CLOSE, kernel, iterations=1)

            pink_contours, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pink_sector = None
            pink_centroid = None
            # pick the largest pink contour (if any) above area threshold
            if pink_contours:
                largest = max(pink_contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 300:
                    M = cv2.moments(largest)
                    if M.get('m00', 0) != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        pink_centroid = (cx, cy)
                        # determine which sector contains the centroid
                        for r, x1, y1, x2, y2, row, col in ratios:
                            if cx >= x1 and cx < x2 and cy >= y1 and cy < y2:
                                pink_sector = (row, col)
                                break

            now = time.time()
            if now - last_print >= print_interval:
                status = []
                for r, x1, y1, x2, y2, row, col in ratios:
                    status.append(f"[{row},{col}]:{int(r*100)}% {human_label(r)}")
                print(" | ".join(status))
                last_print = now

            # draw overlay
            overlay = frame.copy()
            h, w = occ_mask.shape
            for r, x1, y1, x2, y2, row, col in ratios:
                label = f"{int(r*100)}% {human_label(r)}"
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.putText(overlay, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # overlay for pink detection: draw centroid and sector
            if pink_centroid is not None:
                cx, cy = pink_centroid
                cv2.drawContours(overlay, [largest], -1, (255, 0, 255), 2)
                cv2.circle(overlay, (cx, cy), 6, (255, 0, 255), -1)
                if pink_sector is not None:
                    srow, scol = pink_sector
                    txt = f"Pink -> [{srow},{scol}]"
                else:
                    txt = "Pink -> Unknown"
                cv2.putText(overlay, txt, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            else:
                # small legend to show pink detection state
                cv2.putText(overlay, "Pink: none", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # draw contours of detected obstacles
            contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 500:  # ignore small noise
                    continue
                x, y, wcb, hcb = cv2.boundingRect(cnt)
                cv2.rectangle(overlay, (x, y), (x + wcb, y + hcb), (0, 0, 255), 2)

            if show:
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                cv2.imshow('Obstacle Detector', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description='Obstacle detector with 2D grid sectors showing free space')
    p.add_argument('--cam', type=int, default=0, help='camera index (default: 0)')
    p.add_argument('--width', type=int, default=640, help='frame width to process')
    p.add_argument('--rows', type=int, default=2, help='grid rows (default: 2)')
    p.add_argument('--cols', type=int, default=2, help='grid columns (default: 2)')
    p.add_argument('--no-show', dest='show', action='store_false', help='do not open display window')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_camera(camera_index=args.cam, width=args.width, grid_rows=args.rows, grid_cols=args.cols, show=args.show)
