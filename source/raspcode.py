#!/usr/bin/env python3
"""
Object-Controlled Robot with Raspberry Pi
Detects waste objects from camera and controls AUPPBot accordingly

Object Behaviors:
- Tablet: Turn left
- Expired Cosmetic: Turn right
- Syringe: Move forward (approach based on distance)
- Used Battery: Move backward
"""
import os, time, cv2
import numpy as np
from threading import Thread, Lock
from flask import Flask, Response, jsonify, make_response
import onnxruntime as ort

# Import the robot controller
from auppbot import AUPPBot

# -------- config --------
MODEL_PATH = "C:\Users\Dararith\Desktop\Fall_2025\Robotic_submission\onnxconversion\model.onnx"
CAM_INDEX  = 0
SCORE_THRESHOLD = 0.50  # Detection confidence threshold

# Robot serial port (adjust to your setup)
ROBOT_PORT = "/dev/ttyUSB0"

# Performance settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
SKIP_FRAMES = 3  # Run inference every 3 frames
JPEG_QUALITY = 70

# Robot behavior settings
TURN_SPEED = 20           # Speed for turning
FORWARD_FAST = 20         # Fast approach speed
FORWARD_SLOW = 25         # Slow approach speed
BACKWARD_SPEED = 20       # Backward speed

# Object tracking
LOST_LIMIT = 10           # Frames before entering search mode
SEARCH_SPEED = 20         # Speed when searching for objects

# Object debouncing (prevent rapid changes)
OBJECT_HOLD_TIME = 2.0    # Hold object detection for 2 seconds before changing behavior
OBJECT_COOLDOWN = 1.0     # Wait 1 second after action before detecting new object

# Object labels
LABELS = {
    1: "Expired_Cosmetic",
    2: "Syringe",
    3: "Tablet",
    4: "Used_Battery"
}

# Colors for each object (BGR format)
OBJECT_COLORS = {
    'Expired_Cosmetic': (147, 20, 255),   # Purple
    'Syringe':          (0, 0, 255),      # Red
    'Tablet':           (0, 165, 255),    # Orange
    'Used_Battery':     (0, 255, 0),      # Green
}

# -------- initialize robot --------
print("Connecting to robot...")
try:
    robot = AUPPBot(port=ROBOT_PORT, baud=115200, auto_safe=True)
    print(f"✓ Robot connected on {ROBOT_PORT}")
    ROBOT_ENABLED = True
except Exception as e:
    print(f"⚠ Robot connection failed: {e}")
    print("⚠ Running in simulation mode (no robot control)")
    robot = None
    ROBOT_ENABLED = False

# -------- robot control functions --------
def stop():
    if ROBOT_ENABLED and robot:
        robot.stop_all()

def forward(speed):
    if ROBOT_ENABLED and robot:
        robot.motor1.speed(speed)
        robot.motor2.speed(speed)
        robot.motor3.speed(speed)
        robot.motor4.speed(speed)

def backward(speed):
    if ROBOT_ENABLED and robot:
        robot.motor1.speed(-speed)
        robot.motor2.speed(-speed)
        robot.motor3.speed(-speed)
        robot.motor4.speed(-speed)

def turn_left(speed):
    if ROBOT_ENABLED and robot:
        robot.motor1.speed(-speed)
        robot.motor2.speed(-speed)
        robot.motor3.speed(speed)
        robot.motor4.speed(speed)

def turn_right(speed):
    if ROBOT_ENABLED and robot:
        robot.motor1.speed(speed)
        robot.motor2.speed(speed)
        robot.motor3.speed(-speed)
        robot.motor4.speed(-speed)

# -------- load model --------
print("Loading object detection model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print(f"✓ Model loaded: {MODEL_PATH}")

# -------- camera thread --------
class Camera:
    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.ok, self.frame = self.cap.read()
        self.lock = Lock()
        self.running = True
        self.t = Thread(target=self.update, daemon=True)
        self.t.start()

    def update(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.ok, self.frame = ok, f
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ok, None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()

cam = Camera(CAM_INDEX, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)

# -------- robot controller --------
class ObjectRobotController:
    def __init__(self):
        self.current_object = "None"
        self.current_action = "STOP"
        self.lost_frames = 0
        self.lock = Lock()
    
    def update_detection(self, detected_label, box, frame_width, frame_height):
        """Update robot behavior based on detected object"""
        with self.lock:
            if detected_label is None:
                self.lost_frames += 1
                if self.lost_frames > LOST_LIMIT:
                    turn_left(SEARCH_SPEED)
                    self.current_action = "SEARCH (rotating)"
                else:
                    stop()
                    self.current_action = "STOP (no object)"
                self.current_object = "None"
                return self.current_action
            
            # Object detected
            self.lost_frames = 0
            self.current_object = detected_label
            
            # Convert box to pixel coords
            y1, x1, y2, x2 = box
            x1 = int(x1 * frame_width)
            x2 = int(x2 * frame_width)
            y1 = int(y1 * frame_height)
            y2 = int(y2 * frame_height)
            
            cx = (x1 + x2) // 2
            frame_cx = frame_width // 2
            offset = cx - frame_cx
            box_width = x2 - x1  # distance proxy
            
            # -------- CLASS-BASED ACTIONS --------
            if detected_label == "Expired_Cosmetic":
                turn_left(TURN_SPEED)
                self.current_action = "TURN LEFT (Expired_Cosmetic)"
            
            elif detected_label == "Tablet":
                turn_right(TURN_SPEED)
                self.current_action = "TURN RIGHT (Tablet)"
            
            elif detected_label == "Used_Battery":
                forward(FORWARD_FAST)   # always forward
                self.current_action = "FORWARD (Used_Battery)"

            
            elif detected_label == "Syringe":
                backward(BACKWARD_SPEED)
                self.current_action = "BACKWARD (Syringe)"
            
            else:
                stop()
                self.current_action = "UNKNOWN → STOP"
            
            return self.current_action
    
    def get_status(self):
        with self.lock:
            return {
                "object": self.current_object,
                "action": self.current_action
            }
    
    def stop(self):
        """Stop the robot"""
        stop()

# Initialize robot controller
object_controller = ObjectRobotController()

# -------- flask --------
app = Flask(__name__)

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>🤖 Object-Controlled Robot (Waste Sorting)</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root { color-scheme: light dark; }
    body { margin:0; min-height:100vh; display:grid; place-items:center;
           background:#0b0c10; color:#eaf0f6; font-family:system-ui,Segoe UI,Roboto,sans-serif; }
    .card { width:min(96vw,900px); background:#111417; border-radius:16px; padding:14px;
            border:1px solid rgba(255,255,255,0.08); box-shadow:0 10px 40px rgba(0,0,0,.35); }
    h1 { margin:6px 0 10px; font-size:1.05rem; display:flex; align-items:center; gap:8px; }
    .emoji { font-size:1.4rem; }
    .row { display:flex; gap:10px; justify-content:space-between; align-items:center; }
    .btn { border:1px solid rgba(255,255,255,.12); background:#1b2229; color:#eaf0f6;
           padding:6px 12px; border-radius:10px; cursor:pointer; font-weight:600; }
    .btn:hover { background:#222b33; }
    .frame { width:100%; aspect-ratio:16/9; background:#0d1117; border-radius:12px; overflow:hidden;
             border:1px solid rgba(255,255,255,0.08); display:grid; place-items:center; }
    img { width:100%; height:100%; object-fit:contain; }
    small { opacity:.65; }
    .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(100px,1fr)); 
             gap:8px; margin-top:10px; }
    .stat { background:#1b2229; padding:8px; border-radius:8px; text-align:center;
            border:1px solid rgba(255,255,255,0.08); }
    .stat-label { font-size:0.7rem; opacity:0.6; }
    .stat-value { font-size:1.1rem; font-weight:700; margin-top:2px; }
    .tablet { color:#ffa500; }
    .cosmetic { color:#9314ff; }
    .syringe { color:#f87171; }
    .battery { color:#4ade80; }
    .behaviors { background:#1b2229; padding:12px; border-radius:10px; margin-top:10px;
                 border:1px solid rgba(255,255,255,0.08); }
    .behaviors h3 { margin:0 0 8px 0; font-size:0.9rem; opacity:0.8; }
    .behavior-list { display:flex; flex-direction:column; gap:6px; }
    .behavior-item { display:flex; align-items:center; gap:8px; font-size:0.85rem; }
    .behavior-emoji { font-size:1.2rem; }
  </style>
</head>
<body>
  <div class="card">
    <div class="row">
      <h1><span class="emoji">🤖</span> Object-Controlled Robot (Waste Sorting)</h1>
      <button class="btn" onclick="reloadStream()">Reload</button>
    </div>
    <div class="frame">
      <img id="stream" src="/stream" alt="Stream">
    </div>
    <div class="stats" id="stats">
      <div class="stat">
        <div class="stat-label">Status</div>
        <div class="stat-value" id="health">...</div>
      </div>
      <div class="stat">
        <div class="stat-label">Robot</div>
        <div class="stat-value" id="robot">""" + ("✓ ON" if ROBOT_ENABLED else "✗ OFF") + """</div>
      </div>
      <div class="stat">
        <div class="stat-label">Detected</div>
        <div class="stat-value" id="object">-</div>
      </div>
      <div class="stat">
        <div class="stat-label">Action</div>
        <div class="stat-value" id="action">-</div>
      </div>
    </div>
    <div class="behaviors">
      <h3>♻️ Robot Behaviors</h3>
      <div class="behavior-list">
        <div class="behavior-item"><span class="behavior-emoji">💊</span> <strong>Tablet:</strong> Turn Right</div>
        <div class="behavior-item"><span class="behavior-emoji">💄</span> <strong>Expired Cosmetic:</strong> Turn Left</div>
        <div class="behavior-item"><span class="behavior-emoji">💉</span> <strong>Syringe:</strong> Approach (distance-based)</div>
        <div class="behavior-item"><span class="behavior-emoji">🔋</span> <strong>Used Battery:</strong> Move backward</div>
      </div>
    </div>
    <div class="row" style="margin-top:8px;">
      <small>Real-time object detection & control</small>
      <small>URL: <code id="url"></code></small>
    </div>
  </div>
<script>
  async function checkHealth() {
    try {
      const r = await fetch('/health', {cache:'no-store'});
      const j = await r.json();
      document.getElementById('health').textContent = j.camera_ok ? '✓ OK' : '✗ Error';
      
      if (j.object && j.object !== 'None') {
        const obj = j.object.replace('_', ' ');
        document.getElementById('object').textContent = obj;
        const className = j.object.toLowerCase().includes('tablet') ? 'tablet' :
                         j.object.toLowerCase().includes('cosmetic') ? 'cosmetic' :
                         j.object.toLowerCase().includes('syringe') ? 'syringe' :
                         j.object.toLowerCase().includes('battery') ? 'battery' : '';
        document.getElementById('object').className = 'stat-value ' + className;
      } else {
        document.getElementById('object').textContent = 'None';
        document.getElementById('object').className = 'stat-value';
      }
      
      if (j.action) {
        document.getElementById('action').textContent = j.action;
      }
    } catch (e) {
      document.getElementById('health').textContent = '✗ Offline';
    }
  }
  
  function reloadStream() {
    document.getElementById('stream').src = '/stream?ts=' + Date.now();
  }
  
  document.getElementById('url').textContent = location.href;
  checkHealth(); 
  setInterval(checkHealth, 500);
</script>
</body></html>
"""

@app.route("/")
def index():
    return make_response(INDEX_HTML, 200)

@app.route("/health")
def health():
    ok, _ = cam.read()
    status = object_controller.get_status()
    return jsonify({
        "camera_ok": bool(ok),
        "robot_enabled": ROBOT_ENABLED,
        "object": status["object"],
        "action": status["action"]
    })

def gen_mjpeg():
    frame_count = 0

    # keep last detection
    last_label = None
    last_box = None
    last_score = 0.0
    last_seen_frame = -999999

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue

        frame = cv2.rotate(frame, cv2.ROTATE_180)
        h, w = frame.shape[:2]
        annotated = frame.copy()
        frame_count += 1

        # only update detection on inference frames
        if frame_count % SKIP_FRAMES == 0:
            small = cv2.resize(frame, (320, 320))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            input_tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)

            outputs = session.run(None, {input_name: input_tensor})

            boxes  = outputs[1][0]
            classes = outputs[2][0].astype(int)
            scores = outputs[4][0]
            num = int(outputs[5][0])

            # pick best (highest score), not just "first"
            best_i = -1
            best_s = 0.0
            for i in range(num):
                s = float(scores[i])
                if s > best_s:
                    best_s = s
                    best_i = i

            if best_i != -1 and best_s >= SCORE_THRESHOLD:
                cls_id = int(classes[best_i])
                last_label = LABELS.get(cls_id, None)
                last_box = boxes[best_i]
                last_score = best_s
                last_seen_frame = frame_count
            # if nothing good detected, DO NOT erase last_label immediately

        # consider it "lost" only after LOST_LIMIT frames
        if frame_count - last_seen_frame > LOST_LIMIT:
            use_label, use_box, use_score = None, None, 0.0
        else:
            use_label, use_box, use_score = last_label, last_box, last_score

        # now update robot using the persisted detection
        action = object_controller.update_detection(use_label, use_box, w, h)

        # draw detection
        if use_label and use_box is not None:
            y1, x1, y2, x2 = use_box
            x1 = int(x1 * w); x2 = int(x2 * w)
            y1 = int(y1 * h); y2 = int(y2 * h)

            color = OBJECT_COLORS.get(use_label, (255, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label_text = f"{use_label.replace('_',' ')} {use_score:.2f}"
            cv2.putText(annotated, label_text, (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(annotated, f"Action: {action}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue

        b = jpg.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + b + b"\r\n")


@app.route("/stream")
def stream():
    return Response(gen_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def api_status():
    """API endpoint to get current status"""
    status = object_controller.get_status()
    return jsonify({
        "object": status["object"],
        "action": status["action"],
        "robot_enabled": ROBOT_ENABLED
    })

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 Object-Controlled Robot System (Waste Sorting)")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Camera: {CAM_INDEX}")
    print(f"Robot: {'ENABLED' if ROBOT_ENABLED else 'DISABLED (simulation mode)'}")
    print(f"Port: {ROBOT_PORT if ROBOT_ENABLED else 'N/A'}")
    print(f"\n🌐 Web interface: http://raspberrypi.local:5050")
    print(f"📡 API endpoint: http://raspberrypi.local:5050/api/status")
    print("\nRobot Behaviors:")
    print("  💊 Tablet            → Turn left")
    print("  💄 Expired Cosmetic  → Turn right")
    print("  💉 Syringe           → Approach (distance-based)")
    print("  🔋 Used Battery      → Move backward")
    print(f"\nDetection threshold: {SCORE_THRESHOLD}")
    print("="*60 + "\n")
    
    try:
        app.run(host="0.0.0.0", port=5050, threaded=True)
    finally:
        object_controller.stop()
        cam.release()
        if ROBOT_ENABLED and robot:
            robot.close() 