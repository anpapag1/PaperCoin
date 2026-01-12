import cv2
import numpy as np
from pyusbcameraindex import enumerate_usb_video_devices_windows
from collections import deque
import time

# Global variable for tracking circles between frames
previous_circles = []

# Global variable for storing radius history for each coin
coin_radius_history = {}

# Global variable for display mode (True = show value, False = show radius)
show_value = True

# Global variable for tracking A4 paper detection
previous_paper_rect = None

# Preset camera index (None to prompt user)
PRESET_CAM = 2

# A4 paper detection smoothing parameters
A4_MOVEMENT_THRESHOLD = 2
A4_SMOOTHING_FACTOR = 0.8

# Coin radius filtering parameter
RADIUS_FILTER_FRAMES = 30

# A4 paper detection parameters
A4_MIN_AREA = 10000
A4_ASPECT_RATIO = 1.414
A4_ASPECT_TOLERANCE = 0.3
A4_BLUR_KERNEL = 5
A4_CANNY_LOW = 50
A4_CANNY_HIGH = 150

# Coin detection parameters
COIN_MIN_RADIUS = 15
COIN_MAX_RADIUS = 45
COIN_MIN_DIST = 40
COIN_HOUGH_PARAM1 = 50
COIN_HOUGH_PARAM2 = 30
COIN_MEDIAN_BLUR = 7
COIN_MATCH_DISTANCE = 40

# COIN rmin, rmax, value mapping
COIN_VALUES = {
    (18, 24): 0.01,  # 1 cent
    (25, 27): 0.02,  # 2 cents
    (29, 30): 0.05,  # 5 cents
    (28, 28): 0.10,  # 10 cents
    (31, 31): 0.20,  # 20 cents
    (34, 35): 0.50,  # 50 cents
    (32, 33): 1.00,  # 1 euro
    (36, 40): 2.00   # 2 euros
}

def camera_select(preset = None):
    """
    Lists available USB cameras and prompts user to select one.
    If a preset index is provided, it is used directly.
    """
    if preset:
        print(f"Using preset camera index: {preset}")
        return preset
    else:
        # Enumerate USB video devices
        devices = enumerate_usb_video_devices_windows()
        if not devices:
            print("No cameras found")
            return None

        print("\nAvailable cameras:")
        for dev in devices:
            print(f"  {dev.index}: {dev.name}")

        while True:
            try:
                idx = int(input("\nSelect camera (-1 to cancel): "))
                if idx == -1:
                    return None
                if any(d.index == idx for d in devices):
                    return idx
                print("Invalid index")
            except ValueError:
                print("Enter a number")

def find_and_crop_a4(frame):
    """
    Finds an A4 paper in the frame and crops to show only the paper.
    Detects orientation (portrait/landscape) and crops appropriately.
    Uses smoothing to prevent flickering.
    If no paper is found, returns the original frame.
    """
    global previous_paper_rect
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (A4_BLUR_KERNEL, A4_BLUR_KERNEL), 0)
    edges = cv2.Canny(blurred, A4_CANNY_LOW, A4_CANNY_HIGH)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    paper_contour = None
    
    # Look for the largest rectangular contour with A4-like aspect ratio
    for contour in contours[:10]:  # Check top 10 largest contours
        area = cv2.contourArea(contour)
        
        if area < A4_MIN_AREA:
            continue
        
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        # print(f"Contour area: {area}, Approx vertices: {len(approx)}")
        # cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        
        if len(approx) == 4:
            # Get the bounding rectangle to check aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = max(w, h) / min(w, h)
            
            # Check if aspect ratio matches A4 paper
            if abs(aspect_ratio - A4_ASPECT_RATIO) < A4_ASPECT_TOLERANCE:
                paper_contour = approx
                break
    
    # Apply perspective transform if paper is found
    if paper_contour is not None:
        points = paper_contour.reshape(4, 2) # Converts paper_contour from (4,1,2) to (4,2)
        
        # Sort points to get consistent ordering
        rect = np.zeros((4, 2), dtype=np.float32)
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]  # Top-left
        rect[2] = points[np.argmax(s)]  # Bottom-right
        
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]  # Top-right
        rect[3] = points[np.argmax(diff)]  # Bottom-left
        
        # Apply smoothing with previous detection
        if previous_paper_rect is not None:
            # Calculate average distance between corners
            avg_distance = 0
            for i in range(4):
                avg_distance += np.linalg.norm(rect[i] - previous_paper_rect[i])
            avg_distance /= 4
            
            # Only update if significant movement
            if avg_distance <= A4_MOVEMENT_THRESHOLD:
                # Keep previous position for stability
                rect = previous_paper_rect
        
        # Store for next frame
        previous_paper_rect = rect.copy()
        
        # Calculate width and height of the detected paper
        width_top = np.linalg.norm(rect[1] - rect[0])
        width_bottom = np.linalg.norm(rect[2] - rect[3])
        height_left = np.linalg.norm(rect[3] - rect[0])
        height_right = np.linalg.norm(rect[2] - rect[1])
        
        avg_width = (width_top + width_bottom) / 2
        avg_height = (height_left + height_right) / 2
        
        # Detect orientation
        orientation = 'landscape' if avg_width > avg_height else 'portrait'
        
        # Set output dimensions based on orientation
        if orientation == 'landscape':
            output_width = 842
            output_height = 595
        else:
            output_width = 595
            output_height = 842
        
        # Destination points for perspective transform
        dst = np.array([
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply perspective warp to get the A4 paper view
        warped = cv2.warpPerspective(frame, matrix, (output_width, output_height))
        
        return warped, orientation
    else:
        # No paper found, return original frame
        return frame, None
    
def hough_circle_detection(coins, min_r, max_r):
    # turn original image to grayscale
    gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)

    # blur grayscale image
    blurred = cv2.medianBlur(gray, COIN_MEDIAN_BLUR)

    return cv2.HoughCircles(
        blurred,  # source image (blurred and grayscaled)
        cv2.HOUGH_GRADIENT,  # type of detection
        1,  # inverse ratio of accumulator res. to image res.
        COIN_MIN_DIST,  # minimum distance between the centers of circles
        param1=COIN_HOUGH_PARAM1,  # Gradient value passed to edge detection
        param2=COIN_HOUGH_PARAM2,  # accumulator threshold for the circle centers
        minRadius=min_r,  # min circle radius
        maxRadius=max_r,  # max circle radius
    )


def find_coins(frame):
    global coin_radius_history
    
    detected = hough_circle_detection(frame, COIN_MIN_RADIUS, COIN_MAX_RADIUS)
    
    output = frame.copy()
    coin_count = 0
    total_value = 0.0
    
    if detected is not None:
        detected = np.uint16(np.around(detected))
        current_coins = {}
        
        # Match detected coins with historical data based on position
        for (x, y, r) in detected[0, :]:
            # Convert to regular Python int to avoid overflow
            x, y, r = int(x), int(y), int(r)
            
            # Create a unique identifier based on position
            coin_id = None
            min_dist = float('inf')
            
            # Find closest existing coin in history
            for existing_id in list(coin_radius_history.keys()):
                # Parse position from ID (format: "x_y")
                ex, ey = map(int, existing_id.split('_'))
                dist = np.sqrt(float((x - ex)**2 + (y - ey)**2))
                
                # If within threshold pixels, consider it the same coin
                if dist < COIN_MATCH_DISTANCE and dist < min_dist:
                    min_dist = dist
                    coin_id = existing_id
            
            # If no match found, create new coin ID
            if coin_id is None:
                coin_id = f"{x}_{y}"
                coin_radius_history[coin_id] = deque(maxlen=RADIUS_FILTER_FRAMES)
            else:
                # Update coin ID to current position
                old_id = coin_id
                coin_id = f"{x}_{y}"
                if old_id != coin_id and old_id in coin_radius_history:
                    # Transfer history to new position
                    coin_radius_history[coin_id] = coin_radius_history[old_id]
                    del coin_radius_history[old_id]
            
            # Add current radius to history
            coin_radius_history[coin_id].append(r)
            
            # Calculate filtered radius as minimum of history
            filtered_r = int(np.mean(coin_radius_history[coin_id]))
            
            # Store for cleanup
            current_coins[coin_id] = True
            
            # Draw the circle in the output image
            cv2.circle(output, (x, y), filtered_r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
            
            # Determine coin value based on filtered radius
            coin_value = 0.0
            for (rmin, rmax), value in COIN_VALUES.items():
                if rmin <= filtered_r <= rmax:
                    coin_value = value
                    break
            
            # Display value or radius
            if show_value:
                text = f"{coin_value:.2f}"
                total_value += coin_value
            else:
                text = f"r:{filtered_r}"
            
            cv2.putText(output, text, (x - 20, y - filtered_r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            coin_count += 1
        
        # Clean up history for coins that are no longer detected
        coins_to_remove = []
        for coin_id in coin_radius_history.keys():
            if coin_id not in current_coins:
                coins_to_remove.append(coin_id)
        for coin_id in coins_to_remove:
            del coin_radius_history[coin_id]
        
        return output, coin_count, total_value

    return frame, 0, 0.0

def main():
    global show_value
    
    # Load camera
    cap = cv2.VideoCapture(camera_select(PRESET_CAM))
    
    # Display cam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # find and crop A4 paper
        frame,orientation = find_and_crop_a4(frame)
        if orientation is not None:

            # find coins
            frame, coin_count, total_value = find_coins(frame)
            if show_value:
                frame = cv2.putText(frame, f'Total Value: {total_value:.2f}',
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                frame = cv2.putText(frame, f'Coins Detected: {coin_count}',
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            coin_count = 0
            total_value = 0.0
        
        # show frame
        cv2.imshow('Camera Feed', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            show_value = not show_value
            print(f"Display mode: {'Value' if show_value else 'Radius'}")
    
# if main
if __name__ == "__main__":
    main()
