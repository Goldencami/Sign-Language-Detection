import os
import cv2
import time

DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

number_of_classes = 3
dataset_size = 100

# Find a working camera index
def get_working_camera(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"Found working camera at index {i}")
                return i
        cap.release()
    raise RuntimeError("No working camera found. Try checking permissions or unplugging Continuity Camera.")

camera_index = get_working_camera()
cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

# Give camera time to warm up
time.sleep(1)

if not cap.isOpened():
    raise RuntimeError("Failed to open the camera.")

# Data collection loop
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_dir, exist_ok=True)

    print(f"\nCollecting data for class {j}")
    print("Press 'q' when ready to start capturing...")

    # Wait for user to press 'q' to start
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠Frame not received, retrying...")
            continue
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    print("Starting capture...")

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Skipping empty frame.")
            continue
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Stopping early.")
            break

print("\nData collection completed!")
cap.release()
cv2.destroyAllWindows()