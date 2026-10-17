
import cv2
import mediapipe as mp

# MediaPipe vorbereiten
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Webcam öffnen
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Fehler: Keine Webcam gefunden!")
    raise SystemExit(1)

# Handerkennung konfigurieren
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        success, frame = camera.read()

        if not success:
            print("Fehler beim Lesen des Kamerabildes.")
            break

        # Bild spiegeln, damit es wie ein Spiegel wirkt
        frame = cv2.flip(frame, 1)

        # OpenCV verwendet BGR, MediaPipe erwartet RGB
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        # Hände erkennen
        results = hands.process(rgb_frame)

        # Erkannte Hände zeichnen
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 0),
                        thickness=2,
                        circle_radius=4
                    ),
                    mp_drawing.DrawingSpec(
                        color=(255, 0, 0),
                        thickness=2
                    )
                )

            # Anzahl erkannter Hände anzeigen
            count = len(results.multi_hand_landmarks)

            cv2.putText(
                frame,
                f"Haende erkannt: {count}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "Keine Hand erkannt",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Kamerabild anzeigen
        cv2.imshow("Python Hand Tracking", frame)

        # Mit Q beenden
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Ressourcen freigeben
camera.release()
cv2.destroyAllWindows()
