import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import time
import numpy as np
import socket

# Set up I2C
i2c = busio.I2C(board.SCL, board.SDA)

# Increase ADS1115 speed for faster sampling
ADS_DATA_RATE = 475  # Fastest stable rate without excessive noise, default is 128 SPS

# Initialize ADS1115s with optimized speed
ads0 = ADS.ADS1115(i2c, address=0x48, gain=1, data_rate=ADS_DATA_RATE)  # Uses all 4 channels
ads1 = ADS.ADS1115(i2c, address=0x49, gain=1, data_rate=ADS_DATA_RATE)  # Uses all 4 channels
ads2 = ADS.ADS1115(i2c, address=0x4A, gain=1, data_rate=ADS_DATA_RATE)  # Uses all 4 channels
ads3 = ADS.ADS1115(i2c, address=0x4B, gain=1, data_rate=ADS_DATA_RATE)  # Uses A0, A1, A3 for now (until sensor arrives)

# Organize sensors per ADS1115 (Bottom → Middle1 → Middle2 → Top for each finger)
channels = [
    [AnalogIn(ads0, ADS.P0), AnalogIn(ads0, ADS.P1), AnalogIn(ads0, ADS.P2), AnalogIn(ads0, ADS.P3)],  # Finger 1
    [AnalogIn(ads1, ADS.P0), AnalogIn(ads1, ADS.P1), AnalogIn(ads1, ADS.P2), AnalogIn(ads1, ADS.P3)],  # Finger 2
    [AnalogIn(ads3, ADS.P0), AnalogIn(ads3, ADS.P1), AnalogIn(ads3, ADS.P2), AnalogIn(ads3, ADS.P3)],  # Finger 4 (Missing one sensor)
    [AnalogIn(ads2, ADS.P0), AnalogIn(ads2, ADS.P1), AnalogIn(ads2, ADS.P2), AnalogIn(ads2, ADS.P3)],  # Finger 3
]

# Set up socket communication
HOST = '192.168.0.200'
PORT = 65432

def send_data_to_host(data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(data.tobytes())

# Parameters
sampling_rate = 20  # Hz
num_readings = 50  # 50 time steps per grasp
num_sensors = 16  # Total force sensors (16 after update)
data = np.zeros((num_sensors, num_readings))

print('initialized')


# Data collection with batched reads
total_start = time.time()
for i in range(num_readings):
    start_time = time.time()

    # Batch read from all sensors at once
    readings = []
    for ads_index, ads_channels in enumerate(channels):
        row_readings = [(ch.voltage) if ch.voltage > 0 else 0 for ch in ads_channels]
        readings.extend(row_readings) 

    # Store readings in the dataset
    data[:, i] = readings
    # Print voltage value of A0 from the ADS at address 0x48 (ADDR = SDA)
    #print(f"Stored Voltage at ADS0 A2: {data[1, i]:.4f}V")
    print(f"Voltage at ADS0 A0: {channels[2][2].voltage:.4f}V")

    # Enforce sampling rate
    elapsed = time.time() - start_time
    sleep_time = max(0, (1.0 / sampling_rate) - elapsed)
    time.sleep(sleep_time)

    print(f"Time per sample: {elapsed:.6f} sec")

total_end = time.time()
print(f"Total collection time: {total_end - total_start:.6f} sec")

# Send data to host
print('done')
print(data)
send_data_to_host(data)
