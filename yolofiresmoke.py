from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # NOTE: NOT FIRE TRAINED

results = model.predict("test.jpg")

for r in results:
    r.show()  # display result window

print(results[0].boxes)  # show detections, if any
