from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from PIL import Image
import requests

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

# Initialize the processor and model
# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Load model directly

processor = AutoImageProcessor.from_pretrained("jayanta/vit-base-patch16-224-in21k-face-recognition")
model = AutoModelForImageClassification.from_pretrained("jayanta/vit-base-patch16-224-in21k-face-recognition")

image_paths = ['./images/harryReid.png', './images/mitchmcconnell.png', './images/mitchmcconnell2.png']

output_hidden_states = []

for image_path in image_paths:
    # Load the image from the local file system

    image = Image.open(image_path).convert('RGB')

    # Process the image and prepare the inputs
    inputs = processor(images=image, return_tensors="pt")

    # Get the model outputs and compute the logits
    outputs = model(**inputs, output_hidden_states=True)
    logits = outputs.logits
    output_hidden_states.append(outputs.hidden_states[-1])
    print(outputs.hidden_states[-1].size()) # last hidden state, torch.Size([1, 197, 768])

    # Predict the class from the logits
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])


# Example hidden states
# Let's assume output_hidden_states[i] are tensors with shape (num_vectors, vector_dim)

# Calculating pairwise distances
dist_0_1 = torch.cdist(output_hidden_states[0], output_hidden_states[1], p=2)
average_dist_0_1 = dist_0_1.mean().item()  # Get the mean distance

dist_0_2 = torch.cdist(output_hidden_states[0], output_hidden_states[2], p=2)
average_dist_0_2 = dist_0_2.mean().item()  # Get the mean distance

dist_1_2 = torch.cdist(output_hidden_states[1], output_hidden_states[2], p=2)
average_dist_1_2 = dist_1_2.mean().item()  # Get the mean distance

print("Average distance between 0 and 1:", average_dist_0_1)
print("Average distance between 0 and 2:", average_dist_0_2)
print("Average distance between 1 and 2:", average_dist_1_2)
