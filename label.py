import torch
import os

# Load the YOLO model
model = torch.load('best1.pt', map_location=torch.device('cpu'))

def predict_and_save(model, images_folder, labels_folder, output_folder, conf=0.2, iou=0.5):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all label files
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]

    for label_file in label_files:
        label_path = os.path.join(labels_folder, label_file)
        image_file = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_folder, image_file)

        # Check if corresponding image file exists
        if not os.path.exists(image_path):
            print(f"Image file {image_file} for label {label_file} not found.")
            continue

        # Predict
        results = model.predict(source=image_path, save=False, conf=conf, iou=iou)
        results = results[0]

        # Prepare the output in the same format
        with open(os.path.join(output_folder, label_file), 'w') as out_file:
            for result in results.boxes:
                cls = int(result.cls.item())
                x, y, w, h = result.xywhn[0].tolist()
                out_file.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        print(f"Processed {image_file}")

# RUN predict_and_save
images_folder = "images"
labels_folder = "labels"
output_folder = "runs"
predict_and_save(model, images_folder, labels_folder, output_folder)
