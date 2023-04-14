import sys
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Load your trained model
model_path = 'Unet-Mobilenet.pt'
model = torch.load(model_path, map_location='cpu')
model.eval()

# Define pre-processing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_frame(frame):
    input_tensor = preprocess(Image.fromarray(frame))
    input_batch = input_tensor.unsqueeze(0)

    # Move the model and input tensor to the CPU
    model.to('cpu')
    input_batch = input_batch.to('cpu')

    with torch.no_grad():
        output = model(input_batch)

    # Find the class with the highest probability at each pixel
    class_map = torch.argmax(output.squeeze(), dim=0).numpy()
    confidence_scores = torch.softmax(output.squeeze(), dim=0).numpy()

    return class_map, confidence_scores

def display_result(frame, class_map, confidence_scores, confidence_threshold):
    person_class_id = 4 # Assuming "person" has a class ID of 1 in the model
    # get maximum value found in class_map
    # max_class_id = np.max(class_map)
    # print(max_class_id)
    person_color = (96, 255, 22)  # RGB color for "person"

    mask = (class_map == person_class_id)

    # Create a mask only if the confidence is high
    high_confidence_mask = confidence_scores[person_class_id] > confidence_threshold
    mask = np.logical_and(mask, high_confidence_mask)

    mask_overlay = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
    colored_mask = np.zeros_like(frame)
    colored_mask[mask_overlay == 1] = person_color

    masked_frame = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
    cv2.putText(masked_frame, f'Class: {person_class_id}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return masked_frame

if len(sys.argv) > 1:
    image_path = sys.argv[1]
    try:
        frame = cv2.imread(image_path)
        class_map, confidence_scores = process_frame(frame)
        result = display_result(frame, class_map, confidence_scores, confidence_threshold=0)
        cv2.imshow('Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error processing image: {e}")
else:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        class_map = process_frame(frame)
        masked_frame = display_result(frame, class_map)

        cv2.imshow('Frame', masked_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
