"""
Face Authentication Attendance System
Main entry point - Start here

Development Flow (Incremental Testing):
========================================

Step 1: Test Camera [CURRENT]
- Open webcam
- Display live feed
- Press 'q' to quit
→ TEST: python app.py

Step 2: Add Face Detection
- Detect faces in frame
- Draw rectangles around faces
→ TEST: python app.py

Step 3: Add Face Encoding
- Extract face embeddings
- Display encoding info
→ TEST: python app.py

Step 4: Add Registration Mode
- Press 'r' to register face
- Enter name via console
- Save face encoding
→ TEST: python app.py

Step 5: Add Recognition Mode
- Match detected faces
- Display names on screen
- Show confidence scores
→ TEST: python app.py

Step 6: Add Liveness Detection
- Add anti-spoofing
- Validate real face
→ TEST: python app.py

Step 7: Add Attendance Logging
- Log attendance on recognition
- Prevent duplicate entries
- Save to JSON
→ TEST: python app.py

Step 8: Add UI Polish
- Better display
- Status messages
- Instructions on screen
→ TEST: python app.py
"""

import cv2
from camera.camera import Camera
from face.detector import FaceDetector
from face.encoder import FaceEncoder
from attendance.storage import FaceStorage


def main():
    """Main application entry point."""
    print("=" * 50)
    print("Face Authentication Attendance System")
    print("=" * 50)
    print("\nStep 3: Face Registration")
    print("Instructions:")
    print("  - Press 'r' to REGISTER a new face")
    print("  - Press 'l' to LIST registered faces")
    print("  - Press 'q' to quit")
    print("\nStarting camera...\n")
    
    # Initialize components
    try:
        detector = FaceDetector(min_detection_confidence=0.7)
        encoder = FaceEncoder()
        storage = FaceStorage()
        
        print(f"✓ Face detector initialized")
        print(f"✓ Face encoder initialized (encoding dimension: {encoder.get_encoding_shape()})")
        print(f"✓ Face storage initialized")
        print(f"✓ Currently registered: {storage.count()} face(s)\n")
        
        with Camera() as cam:
            print("✓ Camera opened successfully")
            print("✓ Ready for registration...\n")
            
            registration_mode = False
            
            while True:
                # Capture frame
                success, frame = cam.read_frame()
                
                if not success:
                    print("✗ Failed to read frame")
                    break
                
                # Detect faces
                faces = detector.detect(frame)
                
                # Draw detections
                frame = detector.draw_detections(frame, faces)
                
                # Display status
                if registration_mode:
                    cv2.putText(frame, "REGISTRATION MODE - Position your face", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, f"Registered: {storage.count()} | Press 'r' to register", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if faces:
                    cv2.putText(frame, f"Face detected!", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Face Attendance System - Registration', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n✓ Exiting...")
                    break
                
                elif key == ord('r'):
                    # Start registration
                    if not faces:
                        print("✗ No face detected! Please position your face in frame.")
                        continue
                    
                    if len(faces) > 1:
                        print("✗ Multiple faces detected! Please ensure only one person is in frame.")
                        continue
                    
                    print("\n" + "=" * 50)
                    print("REGISTRATION")
                    print("=" * 50)
                    
                    # Get the face
                    face = faces[0]
                    bbox = face['bbox']
                    
                    # Generate encoding
                    print("Capturing face...")
                    encoding = encoder.encode(frame, bbox)
                    
                    if encoding is None:
                        print("✗ Failed to encode face. Please try again.")
                        continue
                    
                    print(f"✓ Face captured successfully")
                    print(f"  Encoding shape: {encoding.shape}")
                    
                    # Get name from user
                    cv2.destroyAllWindows()  # Close window temporarily for input
                    name = input("\nEnter person's name: ").strip()
                    
                    if not name:
                        print("✗ Name cannot be empty. Registration cancelled.")
                        cv2.namedWindow('Face Attendance System - Registration')
                        continue
                    
                    # Save to storage
                    face_id = storage.register_face(name, encoding)
                    
                    print(f"✓ Face registered successfully!")
                    print(f"  Name: {name}")
                    print(f"  ID: {face_id}")
                    print(f"  Total registered: {storage.count()}")
                    print("=" * 50 + "\n")
                    
                    # Reopen window
                    cv2.namedWindow('Face Attendance System - Registration')
                
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
