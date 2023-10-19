import cv2
import numpy as np
import serial
import time
import threading

# Define the minimum and maximum distances for object detection
min_distance = 50  # in pixels
max_distance = 150  # in pixels

# Initialize the serial connection to the Arduino
ser = serial.Serial('/dev/ttyACM0', 9600)

# Initialize the motor commands
propeller_command = 'start'
conveyer_command = 'stop'

# Define variables for autonomous garbage collection
garbage_collected = 0
max_garbage = 10  # Maximum number of garbages the boat can hold


def detect_garbage():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30) # set the fps of the video stream

    classes = []

    with open("model/obj.names") as file_object:
        for class_name in file_object.readlines():
            class_name = class_name.strip()
            classes.append(class_name)
            
    print("Object list")
    print(classes)

    net = cv2.dnn.readNet("model/custom-yolov4-tiny-detector_last.weights","model/custom-yolov4-tiny-detector.cfg")
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255)

    detected_objects = []

    while True:
        success, frame = cap.read()
        classIds, scores, boxes = model.detect(frame, confThreshold=0.3, nmsThreshold=0.5)
        
        if len(classIds) != 0:
            for i in range(len(classIds)):
                classId = int(classIds[i])
                confidence = scores[i]
                box = boxes[i]
                x, y, w, h = box
                className = classes[classId-1]
                cv2.putText(frame, className.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                detected_objects.append(box)
                
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                cv2.putText(frame, f"FPS: {fps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print("Class Ids", classId)
                print("Confidences", confidence)
                print("Boxes", box)
        
        cv2.imshow("Image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if len(detected_objects) > 0:
        return detected_objects
    else:
        return None

lock = threading.Lock()       
def collect_garbage():
    global garbage_collected
    global propeller_command
    global conveyer_command

    while True:
        # Detect garbage in the camera frame
        box = detect_garbage()

        # If garbage is detected, collect it
        if box is not None:
            x, y, w, h = box
            center_x = x + w // 2
            center_y = y + h // 2

            # Move forward until the garbage is collected
            propeller_command = 'start'
            conveyer_command = 'start'
            
            while abs(center_x - 320) > min_distance:
                if center_x < 320:
                    ser.write(b'boat_left\n')
                else:
                    ser.write(b'boat_right\n')
                time.sleep(0.1)
                box = detect_garbage()
                if box is None:
                    break
                x, y, w, h = box
                center_x = x + w // 2
                center_y = y + h // 2
            
            # Stop moving and wait for the garbage to be collected
            propeller_command = 'stop'
            while conveyer_command == 'start':
                time.sleep(0.1)

            # Move backward to release the garbage on the conveyor belt
            propeller_command = 'start'
            while abs(center_x - 320) > min_distance:
                if center_x < 320:
                    ser.write(b'boat_right\n')
                else:
                    ser.write(b'boat_left\n')
                time.sleep(0.1)

                box = detect_garbage()
                if box is not None:
                    break
            
            # Stop moving and update the number of garbages collected
            with lock:
                garbage_collected += 1

            # Check if the maximum number of garbages is collected
            if garbage_collected >= max_garbage:
                print("Boat has reached maximum garbage capacity.")
                propeller_command = 'stop'
                conveyer_command = 'stop'
                break

            # Stop the boat
            propeller_command = 'stop'
            time.sleep(1)

        # Sleep for 100ms
        time.sleep(0.1)

lock = threading.Lock()
def control_motors():
    global propeller_command
    global conveyer_command

    rudder_position = 90  # Centered position
    Kp_rudder = 1.0 # Define the proportional gain for the rudder control
    min_rudder_position = -90 # Define the minimum rudder position in degrees
    max_rudder_position = 90 # Define the maximum rudder position in degrees
    desired_direction = 90 # Assuming we want to go east
    theta = 0 # Assuming the boat is facing north at the start

    while True:
        # Acquire the lock before modifying the global variables
        lock.acquire()
        
        # Call collect_garbage function
        collect_garbage()
        
        # Control propeller
        if propeller_command == 'start':
            ser.write(b'propeller_start\n')
        elif propeller_command == 'stop':
            ser.write(b'propeller_stop\n')

        # Control conveyer belt
        if conveyer_command == 'start':
            ser.write(b'conveyer_start\n')
        elif conveyer_command == 'stop':
            ser.write(b'conveyer_stop\n')
        
        # Release the lock after modifying the global variables
        lock.release()

        # Read rudder position from Arduino
        try:
            arduino_data = ser.readline().decode().strip()
            if arduino_data.startswith("rudder_pos"):
                rudder_position = int(arduino_data.split()[1])
        except:
            pass

        # Control rudder
        rudder_error = desired_direction - theta
        rudder_position += int(Kp_rudder * rudder_error)
        rudder_position = min(max(rudder_position, min_rudder_position), max_rudder_position)
        ser.write(f"rudder_set_pos {rudder_position}\n".encode())

        # Move boat based on propeller and rudder positions
        if propeller_command == 'start':
            if rudder_position > 0:
                ser.write(b'boat_right\n')
            elif rudder_position < 0:
                ser.write(b'boat_left\n')
            else:
                ser.write(b'boat_forward\n')

        # Sleep for 100ms
        time.sleep(0.1)
        



def detect_and_navigate():
    global propeller_command
    global conveyer_command
    
    while True:
        # Detect garbage in the camera frame
        box = detect_garbage()

        # If garbage is detected, collect it
        if box is not None:
            collect_garbage(box)

        # Check if the maximum number of garbages is collected
        if garbage_collected >= max_garbage:
            propeller_command = 'stop'
            conveyer_command = 'stop'
            break

        # Sleep for 100ms
        time.sleep(0.1)


if __name__ == '__main__':
    # Create thread for motor control
    motor_thread = threading.Thread(target=detect_garbage)
    motor_thread.start()

    # Create thread for garbage collection
    garbage_thread = threading.Thread(target=control_motors)
    garbage_thread.start()

    # Create thread for object detection
    detection_thread = threading.Thread(target=collect_garbage)
    detection_thread.start()
    
    detection_and_navigate_thread = threading.Thread(target=detect_and_navigate)
    detection_and_navigate_thread.start()

    # Wait for all threads to finish
    motor_thread.join()
    garbage_thread.join()
    detection_thread.join()
    detection_and_navigate_thread.join()

    # Print the number of garbages collected
    print(f"Total Garbage Collected: {garbage_collected}")
