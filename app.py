"""
Face Authentication Attendance System
Main application with:
- Multi-sample face registration (7 frames averaged)
- Face recognition with matching
- Liveness detection (blink verification)
- Punch-in/punch-out attendance tracking

Controls:
- 'r': Register new face
- 'i': Punch-In (recognition + liveness check)
- 'o': Punch-Out (recognition only)
- 's': Show today's attendance summary
- 'l': List registered faces
- 'q': Quit
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
    print("=" * 70)
    print("Face Authentication Attendance System")
    print("=" * 70)
    print("\nFeatures:")
    print("  ✓ Multi-sample registration (7 frames averaged)")
    print("  ✓ Lighting normalization")
    print("  ✓ Liveness detection (blink verification)")
    print("  ✓ Punch-in/Punch-out tracking")
    print("\nControls:")
    print("  'r' - Register new face")
    print("  'i' - Punch-In (with liveness check - currently auto-passes)")
    print("  'o' - Punch-Out")
    print("  's' - Show attendance summary")
    print("  'l' - List registered faces")
    print("  'q' - Quit")
    print("\nStarting camera...\n")
    
    # Initialize components
    try:
        detector = FaceDetector(min_detection_confidence=0.7)
        encoder = FaceEncoder()
        storage = FaceStorage()
        matcher = FaceMatcher(threshold=8.0)  # Tightened threshold
        attendance_logger = AttendanceLogger()
        attendance_manager = AttendanceManager(attendance_logger)
        liveness = LivenessDetector()
        
        print(f"✓ Face detector initialized")
        print(f"✓ Face encoder initialized (encoding dimension: {encoder.get_encoding_shape()})")
        print(f"✓ Face matcher initialized (threshold: {matcher.threshold})")
        print(f"✓ Liveness detector initialized")
        print(f"✓ Attendance system initialized")
        print(f"✓ Currently registered: {storage.count()} face(s)\n")
        
        with Camera() as cam:
            print("✓ Camera opened successfully")
            print("✓ System ready\n")
            
            # Application state
            mode = "idle"  # idle, register, punch_in, punch_out
            liveness_active = False
            last_recognition = None
            
            while True:
                # Capture frame
                success, frame = cam.read_frame()
                
                if not success:
                    print("✗ Failed to read frame")
                    break
                
                # Detect faces
                faces = detector.detect(frame)
                display_frame = detector.draw_detections(frame.copy(), faces)
                
                # Handle different modes
                if mode == "idle":
                    # Recognition mode - show names for recognized faces
                    cv2.putText(display_frame, f"Registered: {storage.count()} | R:Register I:Punch-In O:Punch-Out", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if faces and storage.count() > 0:
                        face = faces[0]
                        bbox = face['bbox']
                        x, y, w, h = bbox  # bbox is a tuple (x, y, w, h)
                        encoding = encoder.encode(frame, bbox)
                        
                        if encoding is not None:
                            # Match face
                            result = matcher.match_face(encoding, storage.get_all_encodings(), storage.get_all_names())
                            
                            if result['matched']:
                                name = result['name']
                                confidence = result['confidence']
                                status = attendance_manager.get_status(name)
                                status_text = f" [{status.upper()}]" if status else ""
                                
                                # Display name and status
                                cv2.putText(display_frame, f"{name}{status_text}", 
                                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                cv2.putText(display_frame, f"Confidence: {confidence:.2f}", 
                                           (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                
                                last_recognition = result
                            else:
                                cv2.putText(display_frame, "Unknown", 
                                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                last_recognition = None
                
                elif mode == "punch_in" and liveness_active:
                    # Liveness check mode
                    if faces:
                        face = faces[0]
                        bbox = face['bbox']
                        
                        result = liveness.verify_liveness(frame, bbox)
                        
                        # Display liveness status
                        status_color = (0, 255, 255) if result['is_live'] is None else \
                                      (0, 255, 0) if result['is_live'] else (0, 0, 255)
                        
                        cv2.putText(display_frame, result['message'], 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                        cv2.putText(display_frame, f"Time: {result['time_elapsed']:.1f}s", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Check if liveness verification complete
                        if result['is_live'] is True:
                            # Liveness passed, perform punch-in
                            if last_recognition and last_recognition['matched']:
                                name = last_recognition['name']
                                face_id = last_recognition['index']
                                
                                punch_result = attendance_manager.punch_in(name, face_id)
                                print(f"\n{punch_result['message']}")
                                
                                # Visual feedback
                                cv2.rectangle(display_frame, (0, 0), 
                                            (display_frame.shape[1], display_frame.shape[0]),
                                            (0, 255, 0) if punch_result['success'] else (0, 0, 255), 20)
                            
                            liveness_active = False
                            mode = "idle"
                            time.sleep(1)
                        
                        elif result['is_live'] is False:
                            print(f"\n✗ Liveness check failed: {result['message']}")
                            liveness_active = False
                            mode = "idle"
                            time.sleep(1)
                    else:
                        cv2.putText(display_frame, "No face detected", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display frame
                cv2.imshow('Face Attendance System', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n✓ Exiting...")
                    break
                
                elif key == ord('i'):
                    # Punch-In with liveness check
                    if not faces:
                        print("\n✗ No face detected! Please position your face in frame.")
                        continue
                    
                    if storage.count() == 0:
                        print("\n✗ No registered faces. Please register first (press 'r').")
                        continue
                    
                    # First recognize the person
                    face = faces[0]
                    bbox = face['bbox']
                    encoding = encoder.encode(frame, bbox)
                    
                    if encoding is not None:
                        result = matcher.match_face(encoding, storage.get_all_encodings(), storage.get_all_names())
                        
                        if result['matched']:
                            last_recognition = result
                            print(f"\n🔍 Recognized: {result['name']} (confidence: {result['confidence']:.2f})")
                            print("🔍 Starting liveness check - Please blink naturally...")
                            
                            liveness.reset()
                            liveness_active = True
                            mode = "punch_in"
                        else:
                            print("\n✗ Face not recognized. Please register first.")
                    else:
                        print("\n✗ Failed to encode face.")
                
                elif key == ord('o'):
                    # Punch-Out (no liveness check needed for exit)
                    if not faces:
                        print("\n✗ No face detected! Please position your face in frame.")
                        continue
                    
                    if storage.count() == 0:
                        print("\n✗ No registered faces. Please register first (press 'r').")
                        continue
                    
                    face = faces[0]
                    bbox = face['bbox']
                    encoding = encoder.encode(frame, bbox)
                    
                    if encoding is not None:
                        result = matcher.match_face(encoding, storage.get_all_encodings(), storage.get_all_names())
                        
                        if result['matched']:
                            name = result['name']
                            face_id = result['index']
                            
                            punch_result = attendance_manager.punch_out(name, face_id)
                            print(f"\n{punch_result['message']}")
                            
                            # Visual feedback
                            if punch_result['success']:
                                for _ in range(2):
                                    cv2.rectangle(display_frame, (0, 0), 
                                                (display_frame.shape[1], display_frame.shape[0]),
                                                (0, 255, 0), 20)
                                    cv2.imshow('Face Attendance System', display_frame)
                                    cv2.waitKey(200)
                        else:
                            print("\n✗ Face not recognized.")
                    else:
                        print("\n✗ Failed to encode face.")
                
                elif key == ord('s'):
                    # Show attendance summary
                    attendance_manager.display_summary()
                
                elif key == ord('r'):
                    # Start multi-sample registration
                    if not faces:
                        print("✗ No face detected! Please position your face in frame.")
                        continue
                    
                    if len(faces) > 1:
                        print("✗ Multiple faces detected! Please ensure only one person is in frame.")
                        continue
                    
                    print("\n" + "=" * 50)
                    print("MULTI-SAMPLE REGISTRATION")
                    print("=" * 50)
                    
                    # Get name first
                    cv2.destroyAllWindows()  # Close window temporarily for input
                    name = input("\nEnter person's name: ").strip()
                    
                    if not name:
                        print("✗ Name cannot be empty. Registration cancelled.")
                        cv2.namedWindow('Face Attendance System')
                        continue
                    
                    # Capture multiple samples
                    print(f"\nCapturing multiple samples for {name}...")
                    print("Please keep your face in frame and move slightly (different angles)")
                    print("Press 'q' to cancel\n")
                    
                    samples = []
                    num_samples = 7  # Capture 7 frames
                    frame_interval = 8  # Capture every 8th frame for variety
                    frame_count = 0
                    
                    while len(samples) < num_samples:
                        success, sample_frame = cam.read_frame()
                        if not success:
                            break
                        
                        # Detect faces in current frame
                        sample_faces = detector.detect(sample_frame)
                        
                        # Draw detections
                        display_frame = detector.draw_detections(sample_frame.copy(), sample_faces)
                        
                        # Display progress
                        progress_text = f"Capturing: {len(samples)}/{num_samples} samples"
                        cv2.putText(display_frame, progress_text, (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(display_frame, "Move your head slightly", (10, 70),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Capture sample if face detected and at interval
                        if sample_faces and frame_count % frame_interval == 0:
                            bbox = sample_faces[0]['bbox']
                            encoding = encoder.encode(sample_frame, bbox)
                            
                            if encoding is not None:
                                samples.append(encoding)
                                print(f"✓ Sample {len(samples)}/{num_samples} captured")
                                # Visual feedback - flash green
                                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]),
                                            (0, 255, 0), 10)
                        elif not sample_faces:
                            cv2.putText(display_frame, "No face detected!", (10, 110),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        cv2.imshow('Face Attendance System', display_frame)
                        frame_count += 1
                        
                        # Allow cancellation
                        if cv2.waitKey(30) & 0xFF == ord('q'):
                            print("\n✗ Registration cancelled by user")
                            break
                    
                    # Check if we got enough samples
                    if len(samples) >= num_samples:
                        # Average all encodings for robust representation
                        avg_encoding = np.mean(samples, axis=0)
                        
                        # Register with averaged encoding
                        face_id = storage.register_face(name, avg_encoding)
                        print(f"\n✓ Successfully registered {name} with ID {face_id}")
                        print(f"  Used {len(samples)} sample frames (averaged)")
                        print(f"  Total registered faces: {storage.count()}")
                    else:
                        print(f"\n✗ Registration incomplete - only captured {len(samples)}/{num_samples} samples")
                    
                    print("\nPress any key to continue...")
                    cv2.waitKey(2000)  # Show message for 2 seconds
                    print("=" * 50 + "\n")
                    
                    # Reopen window
                    cv2.namedWindow('Face Attendance System')
                    mode = "idle"
                
                elif key == ord('l'):
                    # List all registered faces
                    print("\n" + "=" * 50)
                    print("REGISTERED FACES")
                    print("=" * 50)
                    
                    if storage.count() == 0:
                        print("No faces registered yet.")
                    else:
                        for face_data in storage.list_all():
                            print(f"ID: {face_data['id']} | Name: {face_data['name']} | "
                                  f"Registered: {face_data['registered_at']}")
                    
                    print("=" * 50 + "\n")
            
            # Cleanup
            cv2.destroyAllWindows()
            detector.close()
            encoder.close()
            print("✓ Resources released")
            
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        return 1
    
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    except KeyboardInterrupt:
        print("\n✓ Interrupted by user")
        cv2.destroyAllWindows()
        return 0
    
    return 0


if __name__ == "__main__":
    exit(main())
