# tests/test_ocr.py
"""
Quick manual test: run `python tests/test_ocr.py` from the project root.
Opens your webcam and runs EasyOCR on each frame, drawing the detected text and boxes.
Press 'q' to quit.
"""
import cv2

from ai_engine.ocr.ocr_reader import OCRReader

def main():
    reader = OCRReader(languages=["en"])
    reader.load()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check camera permissions/index.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run OCR
        results = reader.read(frame)
        for r in results:
            print(f"Detected: '{r.text}' ({r.confidence:.2f})")
            
            # Draw bounding boxes (r.box contains 4 corner points: [(x0,y0), (x1,y1), (x2,y2), (x3,y3)])
            # We can draw it as a bounding rectangle from top-left (box[0]) to bottom-right (box[2])
            x1, y1 = r.box[0]
            x2, y2 = r.box[2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, r.text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("VisionGuide OCR Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()