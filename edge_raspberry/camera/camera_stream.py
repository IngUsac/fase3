import cv2
import serial
import time
from ultralytics import YOLO

# =========================
# SERIAL ARDUINO
# =========================
arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)

# =========================
# MODELO CASCO
# =========================
model = YOLO("/home/arqui/Fase3/models/best.pt")

camera_active = False
cap = None

print("Sistema EcoSort listo")

while True:

    # =========================
    # LECTURA RFID
    # =========================
    if arduino.in_waiting > 0:
        msg = arduino.readline().decode().strip()
        print("Arduino:", msg)

        if msg == "ACCESO_OK":
            if not camera_active:
                cap = cv2.VideoCapture(0)
                camera_active = True
                print("Cámara activada")

        elif msg == "ACCESO_DENEGADO":
            camera_active = False
            if cap:
                cap.release()
                cap = None
            cv2.destroyAllWindows()
            print("Cámara desactivada")

    # =========================
    # SOLO SI CAMARA ACTIVA
    # =========================
    if camera_active and cap is not None:

        ret, frame = cap.read()

        if not ret:
            continue

        results = model(frame)

        casco = False

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])].lower()

                if "helmet" in label:
                    casco = True

        # =========================
        # VISUALIZACION
        # =========================
        if casco:
            estado = "CASCO_OK"
            color = (0, 255, 0)
        else:
            estado = "SIN CASCO"
            color = (0, 0, 255)

        cv2.putText(frame, estado, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("EcoSort Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
