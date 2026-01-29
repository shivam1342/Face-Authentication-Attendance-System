"""
Face Authentication Attendance System
Main entry point of the system.
Initializes the camera feed, face detection, liveness verification, and attendance logic,
and orchestrates the overall execution flow.
"""

import cv2
import numpy as np
import time
from camera.camera import Camera
from face.detector import FaceDetector
from face.encoder import FaceEncoder
from face.matcher import FaceMatcher
from attendance.storage import FaceStorage, AttendanceLogger
from attendance.attendance import AttendanceManager
from spoof.liveness import LivenessDetector


def main():
    """Main application entry point."""
    print("Face Attendance System")
    print("Controls: r=Register | i=Punch-In | o=Punch-Out | s=Summary | l=List | q=Quit\n")
    
    # Initialize components
    detector = FaceDetector(min_detection_confidence=0.7)
    encoder = FaceEncoder()
    storage = FaceStorage()
    matcher = FaceMatcher(threshold=8.0)
    liveness = LivenessDetector()
    logger = AttendanceLogger()
    attendance_manager = AttendanceManager(logger)
    
    print(f"System ready | Registered: {storage.count()} face(s)\n")
    
    with Camera() as cam:
        mode = "idle"
        liveness_active = False
        last_recognition = None
        
        while True:
            success, frame = cam.read_frame()
            if not success:
                break
            
            faces = detector.detect(frame)
            display_frame = detector.draw_detections(frame.copy(), faces)
            
            # Idle mode - show recognition
            if mode == "idle":
                cv2.putText(display_frame, f"Registered: {storage.count()} | R:Register I:Punch-In O:Punch-Out", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if faces and storage.count() > 0:
                    # Recognize face
                    encoding = encoder.encode(frame, faces[0]['bbox'])
                    if encoding is not None:
                        result = matcher.match_face(encoding, storage.get_all_encodings(), storage.get_all_names())
                        if result and result['matched']:
                            status = attendance_manager.get_status_today(result['name'])
                            x, y, w, h = faces[0]['bbox']
                            status_text = f" [{status.upper()}]" if status else ""
                            cv2.putText(display_frame, f"{result['name']}{status_text}", 
                                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(display_frame, f"Conf: {result['confidence']:.2f}", 
                                       (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            last_recognition = result
                        else:
                            x, y, w, h = faces[0]['bbox']
                            cv2.putText(display_frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            last_recognition = None
            
            # Liveness check mode
            elif mode == "punch_in" and liveness_active:
                if faces:
                    liveness_result = liveness.verify_liveness(frame, faces[0]['bbox'])
                    
                    # Display liveness status
                    status_color = (0, 255, 255) if liveness_result['is_live'] is None else (0, 255, 0) if liveness_result['is_live'] else (0, 0, 255)
                    cv2.putText(display_frame, liveness_result['message'], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
                    # Liveness verification complete
                    if liveness_result['is_live'] is True and last_recognition and last_recognition['matched']:
                        punch_result = attendance_manager.punch_in(last_recognition['name'], last_recognition['index'])
                        print(punch_result['message'])
                        border_color = (0, 255, 0) if punch_result['success'] else (0, 0, 255)
                        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), border_color, 20)
                        liveness_active = False
                        mode = "idle"
                        time.sleep(1)
                    elif liveness_result['is_live'] is False:
                        print("Liveness check failed")
                        liveness_active = False
                        mode = "idle"
                        time.sleep(1)
                else:
                    cv2.putText(display_frame, "No face detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Face Attendance System', display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('i'):
                if faces and storage.count() > 0:
                    # Recognize face
                    encoding = encoder.encode(frame, faces[0]['bbox'])
                    if encoding is not None:
                        result = matcher.match_face(encoding, storage.get_all_encodings(), storage.get_all_names())
                        if result and result['matched']:
                            print(f"Recognized: {result['name']}")
                            liveness.reset()
                            liveness_active = True
                            mode = "punch_in"
                            last_recognition = result
                        else:
                            print("Face not recognized")
                    else:
                        print("Face encoding failed")
                else:
                    print("No face detected" if not faces else "No registered faces")
            
            elif key == ord('o'):
                if faces and storage.count() > 0:
                    # Recognize face
                    encoding = encoder.encode(frame, faces[0]['bbox'])
                    if encoding is not None:
                        result = matcher.match_face(encoding, storage.get_all_encodings(), storage.get_all_names())
                        if result and result['matched']:
                            punch_result = attendance_manager.punch_out(result['name'], result['index'])
                            print(punch_result['message'])
                        else:
                            print("Face not recognized")
                    else:
                        print("Face encoding failed")
                else:
                    print("No face detected" if not faces else "No registered faces")
            
            elif key == ord('s'):
                summary = attendance_manager.get_today_summary()
                if not summary:
                    print("\n📋 No attendance records today")
                else:
                    print("\n" + "="*70)
                    print("📋 TODAY'S ATTENDANCE SUMMARY")
                    print("="*70)
                    print(f"{'Name':<20} {'Status':<15} {'Punch In':<12} {'Punch Out':<12} {'Duration':<10}")
                    print("-"*70)
                    for record in summary:
                        name = record['name'] or '-'
                        status = record['status'] or '-'
                        punch_in = record.get('punch_in') or '-'
                        punch_out = record.get('punch_out') or '-'
                        duration = record.get('duration') or '-'
                        print(f"{name:<20} {status:<15} {punch_in:<12} {punch_out:<12} {duration:<10}")
                    print("="*70)
            
            elif key == ord('r'):
                if not faces:
                    print("No face detected")
                    continue
                
                cv2.destroyAllWindows()
                name = input("Enter name: ").strip()
                if not name:
                    cv2.namedWindow('Face Attendance System')
                    continue
                
                # Multi-sample registration
                print(f"Capturing 7 samples for {name}...")
                samples = []
                frame_count = 0
                num_samples = 7
                frame_interval = 8
                
                while len(samples) < num_samples:
                    success, sample_frame = cam.read_frame()
                    if not success:
                        print("Failed to read frame")
                        break
                    
                    sample_faces = detector.detect(sample_frame)
                    display_frame = detector.draw_detections(sample_frame.copy(), sample_faces)
                    
                    cv2.putText(display_frame, f"Capturing: {len(samples)}/{num_samples}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    if sample_faces and frame_count % frame_interval == 0:
                        encoding = encoder.encode(sample_frame, sample_faces[0]['bbox'])
                        if encoding is not None:
                            samples.append(encoding)
                            print(f"Sample {len(samples)}/{num_samples}")
                            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 255, 0), 10)
                    
                    cv2.imshow('Face Attendance System', display_frame)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break
                    
                    frame_count += 1
                
                # Complete registration
                if len(samples) >= num_samples:
                    avg_encoding = np.mean(samples, axis=0)
                    face_id = storage.register_face(name, avg_encoding)
                    print(f'Registered {name} (ID: {face_id})')
                else:
                    print(f'Registration incomplete - only captured {len(samples)}/{num_samples} samples')
                
                cv2.namedWindow('Face Attendance System')
            
            elif key == ord('l'):
                if storage.count() == 0:
                    print("No faces registered")
                else:
                    for face_data in storage.list_all():
                        print(f"ID: {face_data['id']} | {face_data['name']}")
        
        cv2.destroyAllWindows()
    
    return 0


if __name__ == "__main__":
    exit(main())
