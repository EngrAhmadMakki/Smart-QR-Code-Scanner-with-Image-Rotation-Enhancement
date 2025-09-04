import cv2
import numpy as np
from pyzbar.pyzbar import decode
import argparse
import os

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def process_images(image_paths):
    detected_qrcodes = set()
    saved_results = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not read image: {image_path}")
            continue

        for angle in range(0, 360, 10):
            rotated = rotate_image(image, angle)
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            sharpened = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 3), -0.5, 0)
            decoded_objects = decode(sharpened)
            new_qr_found = False

            for obj in decoded_objects:
                qr_text = obj.data.decode("utf-8")

                if qr_text not in detected_qrcodes:
                    detected_qrcodes.add(qr_text)
                    new_qr_found = True

                    points = obj.polygon
                    if len(points) > 4:
                        hull = cv2.convexHull(np.array([p for p in points], dtype=np.float32))
                        hull = list(map(tuple, np.squeeze(hull)))
                    else:
                        hull = points

                    for j in range(len(hull)):
                        cv2.line(rotated, hull[j], hull[(j + 1) % len(hull)], (0, 255, 0), 2)

                    x, y = obj.rect.left, obj.rect.top
                    cv2.putText(rotated, qr_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    print(f"✅ QR Code detected in '{image_path}' at {angle}°: {qr_text}")

            if new_qr_found:
                out_path = f"result_{os.path.basename(image_path).split('.')[0]}_{angle}.jpg"
                cv2.imwrite(out_path, rotated)
                saved_results.append(out_path)

    if not saved_results:
        print("❌ No QR codes detected in any image.")
    else:
        print("\n📁 Saved annotated results:")
        for path in saved_results:
            print(f" - {path}")

if __name__ == "__main__":
    import glob

    folder_path = "images"  # ✅ Your image folder

    # Scan for all JPEG, PNG images
    image_paths = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                  glob.glob(os.path.join(folder_path, "*.jpeg")) + \
                  glob.glob(os.path.join(folder_path, "*.png"))

    if not image_paths:
        print("❌ No images found in folder:", folder_path)
    else:
        print(f"🔍 Found {len(image_paths)} images in '{folder_path}'")
        process_images(image_paths)
