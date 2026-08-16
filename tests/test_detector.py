# tests/test_detector.py
"""
Quick manual test: run `python tests/test_detector.py` from the project root.
Downloads yolo11n.pt on first run (small model, few MB) and runs it on your webcam
for a few frames, printing detections to console.
"""
import cv2

from ai_engine.detector.detector import ObjectDetector

def main():
    detector = ObjectDetector()
    detector.load()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check camera permissions/index.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        for d in detections:
            print(f"{d.label} ({d.confidence:.2f}) at {d.box}")
            x1, y1, x2, y2 = d.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, d.label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("VisionGuide Detector Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()