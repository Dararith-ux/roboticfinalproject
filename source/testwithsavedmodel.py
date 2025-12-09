import os
import cv2
import numpy as np
import tensorflow as tf
import time 

# --- CONFIGURATION & OPTIMIZATIONS ---
# 1. Optimizations for CPU/TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress most TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' # Enable Intel CPU optimization

# 2. Paths (Use your absolute paths as Raw Strings)
MODEL_PATH = r'C:\Users\Dararith\Desktop\Fall_2025\Robotic_submission\exported_model\saved_model' 
LABEL_MAP_PATH = r'C:\Users\Dararith\Desktop\Fall_2025\Robotic_submission\datasettfrecord\label_map.pbtxt'
MIN_SCORE_THRESH = 0.5

# 3. Optimized Resolution for Speed (Try 320x240 for max speed)
TARGET_CAM_WIDTH = 640
TARGET_CAM_HEIGHT = 480

# --- LOAD MODEL & LABELS ---
print("Loading model... please wait.")
try:
    detect_fn = tf.saved_model.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    detect_fn = None
    
def load_label_map(label_map_path):
    """Parses the label map file to create a class ID to display name dictionary."""
    category_index = {}
    try:
        with open(label_map_path, 'r') as f:
            # We assume the file contains the structure you provided (item { ... })
            lines = f.read().split('item {')
            
            for item in lines[1:]: # Skip the first empty split
                current_id = None
                display_name = None
                
                # Extract ID
                if 'id:' in item:
                    try:
                        current_id = int(item.split('id:')[-1].splitlines()[0].strip().replace(',', ''))
                    except ValueError:
                        continue 

                # Extract Display Name
                if 'display_name:' in item and current_id is not None:
                    name_line = item.split('display_name:')[-1].splitlines()[0]
                    # Clean up quotes, commas, and whitespace
                    display_name = name_line.strip().strip('"').strip("'").strip(',')
                
                if current_id and display_name:
                    category_index[current_id] = display_name
                    
    except FileNotFoundError:
        print(f"Warning: Label map not found at {label_map_path}. Labels will be shown as IDs.")
    return category_index

category_index = load_label_map(LABEL_MAP_PATH)
print(f"Loaded classes: {category_index}") # Confirm loaded labels

def detect_and_draw(image_np):
    """Handles TensorFlow inference and drawing on the image."""
    if detect_fn is None:
        return image_np

    # Convert BGR (OpenCV) to RGB (TensorFlow expects RGB)
    # The image is resized by the webcam capture, so we use the resulting size.
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # Convert image to tensor and add batch dimension
    input_tensor = tf.convert_to_tensor(image_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = detect_fn.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # Process outputs
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy() 
                   for key, value in output_dict.items()}
    
    classes = output_dict['detection_classes'].astype(np.int64)
    boxes = output_dict['detection_boxes']
    scores = output_dict['detection_scores']

    # Draw bounding boxes (on the original BGR image)
    height, width, _ = image_np.shape
    for i in range(len(scores)):
        if scores[i] > MIN_SCORE_THRESH:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            
            # Draw Box
            cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            
            # --- 4. Optimization: Use Display Name from Label Map ---
            # Lookup the display_name, or default to ID if not found
            class_name = category_index.get(classes[i], f'ID: {classes[i]}') 
            label = f"{class_name}: {int(scores[i]*100)}%"
            
            # Draw Label
            cv2.putText(image_np, label, (int(left), int(top)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image_np

def run_local_detection():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # --- 1. Optimization: Set lowest resolution for capture ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_CAM_HEIGHT)
    
    # FPS tracking variables
    prev_time = time.time()
    
    print("Starting local webcam feed. Press 'q' to exit.")
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Webcam stream ended.")
            break
            
        # 2. Run Detection and Drawing
        result_frame = detect_and_draw(frame)
        
        # 3. Calculate and Display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        fps_text = f'FPS: {fps:.2f}'
        cv2.putText(result_frame, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 4. Display the resulting frame
        cv2.imshow('Local Object Detection Speed Test', result_frame)
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Local speed test finished.")

if __name__ == '__main__':
    if detect_fn:
        run_local_detection()