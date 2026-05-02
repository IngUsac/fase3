from ultralytics import YOLO
import cv2

# =========================
# MODELO
# =========================
model = YOLO("/home/arqui/Fase3/models/best.pt")

# =========================
# CAMARA
# =========================
cap = cv2.VideoCapture(0)

print("EcoSort: Detección de casco iniciada")

while True:

    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame)

    casco = False

    # =========================
    # ANALISIS
    # =========================
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower()

            if "helmet" in label:
                casco = True

    # =========================
    # ESTADO
    # =========================
    if casco:
        estado = "CASCO_OK"
        color = (0, 255, 0)
    else:
        estado = "CASCO_MISSING"
        color = (0, 0, 255)

    print(estado)

    # =========================
    # VISUALIZACION
    # =========================
    cv2.putText(frame, estado, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("EcoSort Vision - Casco", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()# casco detection


