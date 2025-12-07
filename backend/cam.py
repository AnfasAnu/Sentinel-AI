# test_cam.py
import cv2
for idx in range(3):
    print(f"\nTrying camera index {idx} ...")
    # try different backends on Windows if available
    for backend_name, backend in [("default", None),
                                  ("CAP_DSHOW", cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else None),
                                  ("CAP_MSMF", cv2.CAP_MSMF if hasattr(cv2, "CAP_MSMF") else None),
                                  ("CAP_V4L2", cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else None)]:
        try:
            if backend is None:
                cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(idx, backend)
            ok = cap.isOpened()
            ret, frame = cap.read() if ok else (False, None)
            print(f"  backend={backend_name:8} opened={ok} read_ok={ret} frame_shape={None if frame is None else frame.shape}")
            cap.release()
        except Exception as e:
            print("  backend", backend_name, "exception:", e)
