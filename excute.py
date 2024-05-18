import torch
import os
import cv2
# model = YOLO("best.pt")
model = torch.load('best1.pt', map_location=torch.device('cpu'))

def predict_and_plot(model, folder_path, save_results=True, conf=0.2, iou=0.5):
    # Get a list of all files in the directory
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        # Predict
        results = model.predict(source=image_path, save=save_results, conf=conf, iou=iou)
        Results = results[0]

        # Plot image
        plot = Results.plot()
        plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)

        # Show image
        # Image.fromarray(plot).show()
        

# RUN predict_and_plot
folder_path = "images"
predict_and_plot(model, folder_path)
