import torch
import numpy as np
import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io
from torchvision import transforms
import os
from datetime import datetime

from model_architecture import AppleGrasper

# ------------------------------
# Model + Grasp Classes
# ------------------------------
MODEL_PATH = "model_fulldataset.pth" #model_0(seed=44_labelsmoothed))17epochs"
model = AppleGrasper()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

CLASS_NAMES = ["movein", "good", "moveout"]
NUM_SENSORS = 16
NUM_VALUES = 50
SAMPLING_RATE = 20
ADS_DATA_RATE = 475

# ------------------------------
# Sensor Setup
# ------------------------------
i2c = busio.I2C(board.SCL, board.SDA)

ads0 = ADS.ADS1115(i2c, address=0x48, gain=1, data_rate=ADS_DATA_RATE)
ads1 = ADS.ADS1115(i2c, address=0x49, gain=1, data_rate=ADS_DATA_RATE)
ads2 = ADS.ADS1115(i2c, address=0x4A, gain=1, data_rate=ADS_DATA_RATE)
ads3 = ADS.ADS1115(i2c, address=0x4B, gain=1, data_rate=ADS_DATA_RATE)

channels = [
    [AnalogIn(ads0, ADS.P0), AnalogIn(ads0, ADS.P1), AnalogIn(ads0, ADS.P2), AnalogIn(ads0, ADS.P3)],
    [AnalogIn(ads1, ADS.P0), AnalogIn(ads1, ADS.P1), AnalogIn(ads1, ADS.P2), AnalogIn(ads1, ADS.P3)],
    [AnalogIn(ads3, ADS.P0), AnalogIn(ads3, ADS.P1), AnalogIn(ads3, ADS.P2), AnalogIn(ads3, ADS.P3)],
    [AnalogIn(ads2, ADS.P0), AnalogIn(ads2, ADS.P1), AnalogIn(ads2, ADS.P2), AnalogIn(ads2, ADS.P3)],
]

# ------------------------------
# Sampling Function
# ------------------------------
def sample_fsr_data():
    data = np.zeros((NUM_SENSORS, NUM_VALUES))
    for i in range(NUM_VALUES):
        start_time = time.time()
        readings = []
        for ads_channels in channels:
            readings.extend([ch.voltage if ch.voltage > 0 else 0 for ch in ads_channels])
        data[:, i] = readings
        time.sleep(max(0, (1.0 / SAMPLING_RATE) - (time.time() - start_time)))
    return data
    

# ------------------------------
# Transform for Inference (Matches Training)
# ------------------------------
image_transform = transforms.Compose([
    transforms.Resize((79, 248), interpolation=Image.NEAREST),
    transforms.ToTensor()
])

# ------------------------------
# Plotting Function (No Borders)
# ------------------------------
def plot_heatmap_array(data):
    fig = plt.Figure(figsize=(2.48, 0.79), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(data, cmap='hot', interpolation='nearest', vmin=0, vmax=4.1)
    ax.set_axis_off()
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_jpg(buf)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img = img.crop((1,0,248,79))
    return img
    
# ------------------------------
# Inference Function
# ------------------------------
def predict_grasp(data):
    debug_img = plot_heatmap_array(data)

    # Overwrite last debug image
    debug_img.save("/home/surp_apple/last_grasp_debug(fulldataset2.jpg")

    tensor = image_transform(debug_img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().numpy()
        pred_index = np.argmax(probs)

        print(f"Saved debug image to: /home/surp_apple/last_grasp_debug.jpg")
        print(f"Class probabilities: movein={probs[0]:.3f}, good={probs[1]:.3f}, moveout={probs[2]:.3f}")
        return CLASS_NAMES[pred_index]

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    print("Sampling FSR data...")
    tactile_data = sample_fsr_data()

    if tactile_data.max() < 0.1:
        print("Assuming movein (pre-model rule)")
        result = "movein"
    else:
        print("Running inference...")
        result = predict_grasp(tactile_data)

    print(f"Final grasp classification: {result}")
    
    # Map decision to number
class_to_code = {"good": 0, "movein": 1, "moveout": 2}
decision_code = class_to_code[result]

# Write to text file
with open("/home/surp_apple/grasp_decision_code.txt", "w") as f:
    f.write(str(decision_code))
print(f"Wrote decision code '{decision_code}' to text file.")

# Send via TCP to Ubuntu computer
import socket
HOST = '172.20.10.131'  # REPLACE with IP of Ubuntu computer which is 172.20.10.131
PORT = 65432

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(str(decision_code).encode())
        print(f"Sent decision code '{decision_code}' over TCP.")
except Exception as e:
    print(f"Failed to send decision over TCP: {e}")

    
	


