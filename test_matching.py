"""
Quick test for face matching - Step 4
"""
import cv2
from camera.camera import Camera
from face.detector import FaceDetector
from face.encoder import FaceEncoder
from face.matcher import FaceMatcher
from attendance.storage import FaceStorage

print("=" * 50)
print("Step 4: Face Matching Test")
print("=" * 50)
print("Testing MATCH vs NO MATCH")
print("Press 'q' to quit\n")

detector = FaceDetector()
encoder = FaceEncoder()
matcher = FaceMatcher(threshold=0.6)
storage = FaceStorage()

print(f"Registered faces: {storage.count()}")
print(f"Threshold: {matcher.get_threshold()}\n")

with Camera() as cam:
    while True:
        success, frame = cam.read_frame()
        if not success:
            break
        
        faces = detector.detect(frame)
        
        if faces and storage.count() > 0:
            for face in faces:
                bbox = face['bbox']
                x, y, w, h = bbox
                encoding = encoder.encode(frame, bbox)
                
                if encoding is not None:
                    result = matcher.match_face(
                        encoding,
                        storage.get_all_encodings(),
                        storage.get_all_names()
                    )
                    
                    if result['matched']:
                        color = (0, 255, 0)  # Green
                        label = f"MATCH: {result['name']}"
                        print(f"✓ MATCH: {result['name']} | Conf: {result['confidence']:.2f} | Dist: {result['distance']:.4f}")
                    else:
                        color = (0, 0, 255)  # Red
                        label = "NO MATCH"
                        print(f"✗ NO MATCH | Dist: {result['distance']:.4f}")
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow('Matching Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
print("\n✓ Test complete")
