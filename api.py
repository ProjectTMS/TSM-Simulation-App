import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
import os
from threading import Event

# Load the YOLO model
model = YOLO('best.pt')

# Define the class names
class_names = ['bicycle', 'bus', 'car', 'motorbike', 'person']

# Global variables for control
stop_event = Event()
pause_event = Event()

def draw_traffic_light(frame, state):
    height, width = frame.shape[:2]
    light_height = height // 3
    light_width = width // 10
    light = np.zeros((light_height, light_width, 3), dtype=np.uint8)

    if state == 'red':
        cv2.circle(light, (light_width//2, light_height//4), light_width//4, (255,0,0), -1)  # Red in BGR
    elif state == 'yellow':
        cv2.circle(light, (light_width//2, light_height//2), light_width//4, (255,255,0), -1)  # Yellow in BGR
    elif state == 'green':
        cv2.circle(light, (light_width//2, 3*light_height//4), light_width//4, (0,255,0), -1)  # Green in BGR

    frame[0:light_height, 0:light_width] = light
    return frame

def process_video(video_file, signal_state):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())
        temp_filename = tfile.name

    cap = cv2.VideoCapture(temp_filename)

    if not cap.isOpened():
        st.error(f"Error opening video file: {temp_filename}")
        return

    try:
        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning of video
                continue

            # Perform object detection
            results = model(frame)

            # Process results
            detections = results[0].boxes.data
            for detection in detections:
                x1, y1, x2, y2, score, class_id = detection
                if class_id < 4:  # Only count vehicles (not persons)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw traffic light
            frame = draw_traffic_light(frame, signal_state.value)

            # Convert colors from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield frame

    finally:
        cap.release()
        try:
            os.unlink(temp_filename)
        except PermissionError:
            pass  # If we can't delete the file now, it will be deleted later

def main():
    st.title("Traffic Signal Management System")

    # Video file uploaders
    video_files = []
    for i in range(5):
        video_file = st.file_uploader(f"Upload video {i+1}", type=['mp4', 'avi'])
        if video_file:
            video_files.append(video_file)

    if len(video_files) < 3:
        st.warning("Please upload at least 3 video files.")
        return

    # Red light duration slider
    red_light_duration = st.slider("Red Light Duration (seconds)", min_value=1, max_value=120, value=10)
    yellow_light_duration = st.slider("Yellow Light Duration (seconds)", min_value=1, max_value=5, value=2)

    # Initialize signal states
    class SignalState:
        def __init__(self):
            self.value = 'red'
            self.last_change = time.time()

    signal_states = [SignalState() for _ in video_files]

    # Control buttons
    col1, col2, col3 = st.columns(3)
    start_button = col1.button("Start Traffic Management")
    #pause_button = col2.button("Pause/Resume")
    stop_button = col3.button("Stop")

    # Create placeholders for video streams and counters in a grid layout
    video_placeholders = []
    counter_placeholders = []

    # Arrange video feeds in a grid with titles
    for i in range(0, len(video_files), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(video_files):
                with col:
                    st.subheader(f"Road {i + j + 1}")
                    video_placeholders.append(st.empty())
                    counter_placeholders.append(st.empty())

    if start_button:
        stop_event.clear()
        pause_event.clear()

        video_generators = [process_video(vf, state) for vf, state in zip(video_files, signal_states)]
        last_frames = [next(gen) for gen in video_generators]

        current_green = 0
        signal_states[current_green].value = 'green'
        signal_states[current_green].last_change = time.time()

        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.1)
                continue

            vehicle_counts = []

            for i, (gen, placeholder, counter, state) in enumerate(zip(video_generators, video_placeholders, counter_placeholders, signal_states)):
                try:
                    if state.value == 'green':
                        frame = next(gen)
                        last_frames[i] = frame

                        # Perform object detection
                        results = model(frame)

                        # Count vehicles
                        vehicle_count = sum(1 for det in results[0].boxes.data if det[5] < 4)
                        vehicle_counts.append(vehicle_count)

                        # Update counter
                        #counter.text(f"Road {i+1} - Vehicles: {vehicle_count}")
                    else:
                        vehicle_counts.append(0)
                        frame = last_frames[i]
                        #counter.text(f"Road {i+1} - Vehicles: {vehicle_counts[i]}")

                    # Always update the frame to show current light state
                    frame = draw_traffic_light(frame, state.value)
                    placeholder.image(frame)

                except StopIteration:
                    st.warning(f"Video {i+1} has ended. Restarting...")
                    video_generators[i] = process_video(video_files[i], signal_states[i])

            # Update signal states
            current_time = time.time()
            time_since_last_change = current_time - signal_states[current_green].last_change

            if time_since_last_change >= red_light_duration:
                # Change signals
                next_green = vehicle_counts.index(max(vehicle_counts))
                if next_green == current_green:
                    next_green = (current_green + 1) % len(video_files)

                # Set current green to yellow
                signal_states[current_green].value = 'yellow'
                signal_states[current_green].last_change = current_time

                # Update frames to show yellow
                for _ in range(int(yellow_light_duration * 10)):  # 10 frames per second
                    for i, (placeholder, state) in enumerate(zip(video_placeholders, signal_states)):
                        frame = last_frames[i]
                        frame = draw_traffic_light(frame, state.value)
                        placeholder.image(frame)
                    time.sleep(0.1)

                # Set previous green (now yellow) to red
                signal_states[current_green].value = 'red'
                signal_states[current_green].last_change = current_time

                # Set next green to yellow first
                signal_states[next_green].value = 'yellow'
                signal_states[next_green].last_change = current_time

                # Update frames to show yellow
                for _ in range(int(yellow_light_duration * 10)):  # 10 frames per second
                    for i, (placeholder, state) in enumerate(zip(video_placeholders, signal_states)):
                        frame = last_frames[i]
                        frame = draw_traffic_light(frame, state.value)
                        placeholder.image(frame)
                    time.sleep(0.1)

                # Set next green to green
                signal_states[next_green].value = 'green'
                signal_states[next_green].last_change = current_time
                current_green = next_green

            # Display signal states
            for i, state in enumerate(signal_states):
                st.sidebar.text(f"Road {i+1}: {state.value.upper()}")

            # Pause for a moment to allow for visual updates
            time.sleep(0.1)

    if pause_button:
        if pause_event.is_set():
            pause_event.clear()
        else:
            pause_event.set()

    if stop_button:
        stop_event.set()

if __name__ == "__main__":
    main()
