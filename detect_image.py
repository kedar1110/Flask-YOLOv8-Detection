import os
import cv2

def detect_image(model, image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image = cv2.imread(image_path)
    results = model(image)

    for r in results:
        annotated_image = r.plot()

    name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"pred_{name}.jpg")
    cv2.imwrite(output_path, annotated_image)

    return output_path
