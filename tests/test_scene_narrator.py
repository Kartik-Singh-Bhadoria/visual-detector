# tests/test_scene_narrator.py
"""
Quick manual test: run `python -m tests.test_scene_narrator` from the project root.
Opens your webcam, shows a live preview, and press SPACE to capture one frame
and get a scene description back from Gemini. Press 'q' to quit.
"""
import cv2

from ai_engine.scene.scene_narrator import SceneNarrator

def main():
    narrator = SceneNarrator()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Press SPACE to describe the current scene. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("VisionGuide Scene Narrator Test", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            print("Describing scene...")
            try:
                description = narrator.describe(frame)
                print(f"-> {description}")
            except Exception as e:
                print(f"Error: {e}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()