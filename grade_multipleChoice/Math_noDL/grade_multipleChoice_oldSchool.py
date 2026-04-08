# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd

import cv2
import pytesseract
from pdf2image import convert_from_path

import matplotlib.pyplot as plt

# %%
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# %%
import multiple_choice_core as mg

import importlib;
importlib.reload(mg);

# %%

# %%
research_db = "/Users/hn/Documents/01_research_data/"
data_base = research_db + "multiple_choices/"
model_dir = data_base + "/models/"

test_directory = data_base + "test_1/"
question_count = 15

# %%

# %% [markdown]
#
# If PDF files are read via OpenCV,
# then the default order of chanels are BGR so we need COLOR_BGR2GRAY.
# If it is read via PIL() then we need COLOR_RGB2GRAY.
#
# ```
# PIL: from pdf2image import convert_from_path
# ```
#
# I read it via PIL so, here I stick to RGB2GRAY

# %%
tick_legend_FontSize = 4
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1,
          "axes.labelsize":  tick_legend_FontSize * 1,
          "axes.titlesize":  tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * .8,
          "ytick.labelsize": tick_legend_FontSize * .8,
          "axes.titlepad": 10,
          'legend.handlelength': 2,
          "axes.titleweight": 'bold',
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
          'axes.linewidth' : .05
}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
# Name region coordinates:
# y1=190
# y2=290
# x1=120
# x2=1400

y1=200
y2=280
x1=120
x2=1400

# %%

# %% [markdown]
# What BUBBLE_THRESHOLD is
#
# When we process the right side of the exam, each answer bubble (A–E) is converted to a black-and-white image (thresholded).
#
# Then we count the number of black pixels in each bubble using:
#
# ```cv2.countNonZero(bubble)```
#
# - A filled bubble will have many black pixels; an empty bubble will have few or zero black pixels.
#
# - ```BUBBLE_THRESHOLD``` is the minimum number of black pixels required to consider a bubble “filled”.
#
# - If the bubble's black pixel count is below this threshold, it’s treated as not filled.
#
# - If it’s above, it's considered the selected answer.

# %% [markdown]
# **Why it’s needed**
#
# Without this threshold, faint marks or scanning noise could be misread as answers. This helps ignore small smudges or background artifacts.
#
# **How to set it**
#
# - Start with 150 (works for most 300 dpi scans).
#
# - If your scans are very dark or light, adjust up or down.
#
# - You can even print the scores for each bubble to calibrate:
#
# ```print(scores)```
#
# Or, it can be done dynamically to be more robust. Done in ```read_answers_dynamicBubbleThres()```

# %%
PDF_FILE = "SCAN0003.pdf"
#--- Config:

TOTAL_QUESTIONS = 100  # template has 100 questions
# For each exam, you can specify the active number of questions
NUM_ACTIVE_QUESTIONS = 50  # adjust per PDF
LEFT_CROP_RATIO = 0.5  # left 30% for names
BUBBLE_THRESHOLD = 150 # threshold for filled bubble


# %%
pages = convert_from_path(test_directory + PDF_FILE, dpi=300)

# %%
# 

# answer_key = None

# for i, page in enumerate(pages):
#     img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

#     if i == 0:
#         answer_key = mg.read_answers(img, num_questions=question_count)
#     else:
#         name = read_name(img)
#         student_answers = mg.read_answers(img, num_questions=question_count)
#         score = grade(student_answers, answer_key)
#         print(i, name, score)

# %%
# results = []

# for page in pages[1:]:  # skip key
#     name = get_student_name(page)
#     answers = get_answers(page)
    
#     # Compare
#     score = sum([1 for a,b in zip(answers, key_answers) if a == b])
    
#     results.append({'Name': name, 'Score': score})

# %%
# First page is key
# key_answers = mg.get_answers(pages[0])

# %%
# %%time

# -------- MAIN --------
for i, page in enumerate(pages[1:], start=1):  # skip key page
    # There is no point in doing the following!
    # img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    name = mg.extract_name(np.array(page))
    print(f"Page {i}: {name}")

# %%
for i, page in enumerate(pages[1:], start=1):  # skip key page
    # There is no point in doing the following!
    # img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    name = mg.extract_name_removeVertical(np.array(page))
    print(f"Page {i}: {name}")

# %% [markdown]
# ```cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)``` and ```cv2.cvtColor(right, cv2.COLOR_RGB2BGR)```
# convert color chanels to one thing or another. sometimes you dont need to do it. 
# ChatGPT makes a mess

# %%
importlib.reload(mg);
fig, axes = plt.subplots(2, 2)

axes[0, 0].imshow(pages[1])
axes[0, 1].imshow(np.array(pages[1]))

img = cv2.cvtColor(np.array(pages[1]), cv2.COLOR_RGB2BGR)
axes[1, 0].imshow(img)

NR = mg.extract_name_region(img, y1=190, y2=290, x1=120, x2=1300)
axes[1, 1].imshow(NR)
plt.tight_layout()
plt.show()

# %%
importlib.reload(mg);

fig, axes = plt.subplots(2, 2)

axes[0, 0].imshow(pages[1])
axes[0, 1].imshow(np.array(pages[1]))

img = cv2.cvtColor(np.array(pages[1]), cv2.COLOR_RGB2BGR)
axes[1, 0].imshow(img)

NR = mg.extract_name_region(np.array(pages[1]), y1=190, y2=290, x1=120, x2=1400)
axes[1, 1].imshow(NR)
plt.tight_layout()
plt.show()

axes[0, 0].axis("off"); axes[1, 0].axis("off");
axes[0, 1].axis("off"); axes[1, 1].axis("off");

# %%

# %%
importlib.reload(mg);

fig, axes = plt.subplots(1, 2, figsize=(6, 1.2), dpi=600)
axes[0].imshow(cv2.cvtColor(np.array(pages[1]), cv2.COLOR_RGB2GRAY));
axes[1].imshow(cv2.cvtColor(np.array(pages[1]), cv2.COLOR_BGR2GRAY));

axes[0].axis("off"); axes[1].axis("off");

# %%
plt.imshow(img[y1:y2, x1:x2]) # , cmap="gray"
plt.axis("off")
plt.show()

# %%
name_region = NR

# %%
plt.imshow(NR, cmap="gray")
plt.axis("off")
plt.show()

# %%
cleaned = mg.remove_grid(NR)
mg.show(cleaned)

# %%
smooth = cv2.medianBlur(name_region, 3)
cleaned = mg.remove_grid(smooth)
mg.show(cleaned)

# %%
# smooth = cv2.GaussianBlur(name_region, (3,3), 0)
# cleaned = mg.remove_grid(smooth)
# mg.show(cleaned)

# %%
importlib.reload(mg);

#  gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 1)
cleaned = mg.remove_grid(smooth)
mg.show(cleaned)

# %%
importlib.reload(mg);

# gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 100)
# vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
cleaned = mg.remove_grid(smooth)
mg.show(cleaned)

# %%
importlib.reload(mg);

# gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 100)
# vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
cleaned = mg.remove_grid(smooth)
mg.show(cleaned)

# %%
plt.figure(figsize=(12, 3))
plt.imshow(boxes[0], cmap="gray")
plt.axis("off")
plt.show()

# %%
mg.ocr_boxes(boxes)

# %% [markdown]
# # DL for reading NAME

# %%
# Define a transformation (resizing, grayscale conversion, normalization)
transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),  # Resize to 28x28 (for EMNIST model)
            transforms.Grayscale(num_output_channels=1),  # Grayscale
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(
                (0.5,), (0.5,)
            ),  # Normalize based on EMNIST dataset stats
        ]
    )

# %%
importlib.reload(mg);

# %%
# Load pretrained weights
# DL_name_reading_model = mg.EMNIST_Model_CNN()
# DL_name_reading_model.load_state_dict(torch.load(model_dir+"emnist_letters_Aadam.pth", map_location="cpu"))
# DL_name_reading_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
DL_name_reading_model = mg.EMNIST_Model_CNN().to(device)
DL_name_reading_model.load_state_dict(torch.load(model_dir+"emnist_letters_Aadam.pth", map_location=device))
DL_name_reading_model.eval();

# %%
clean_image = mg.preprocess_nameRegion_4_cnn(name_region=name_region, transform_=transform)

# %%

# %%
# def find_characters_boxes(clean_image):
#     """Detect and extract bounding boxes of each character."""
#     contours, _ = cv2.findContours(clean_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Sort contours by x-coordinate (left-to-right)
#     boxes = []
#     for c in contours:
#         x, y, w, h = cv2.boundingRect(c)
        
#         # Add some conditions to filter out small artifacts (non-character contours)
#         if w > 10 and h > 10:  # Minimum size for characters
#             boxes.append((x, y, w, h))
    
#     boxes.sort(key=lambda b: b[0])  # Sort boxes left to right
    
#     return boxes

# %%
plt.imshow(name_region)

# %%
mg.classify_nameRegion(name_region=name_region, model=DL_name_reading_model, transform_=transform)

# %%
# Assuming you already have the model loaded
# model = torch.load('path_to_your_trained_model.pth')  # Load your trained model

# Define the necessary transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),  # Resize to 28x28 (adjust if needed)
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale (1 channel)
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Adjust based on your model's training
])

# Function to segment the name region into individual characters
def segment_name_into_characters(name_region):
    """Segment the name region into individual characters."""
    gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    
    char_images = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 20:  # Filter out small contours that are noise
            char_img = name_region[y:y+h, x:x+w]
            char_images.append(char_img)
    
    return char_images

# Function to preprocess each character for CNN
def preprocess_character_for_cnn(char_img, transform_):
    """Preprocess a single character image for the CNN."""
    if len(char_img.shape) == 3:  # Convert to grayscale if it's a 3-channel (RGB)
        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    else:  # Already grayscale (1-channel)
        gray = char_img
    
    # Apply transformation (resize, normalize, etc.)
    transformed_image = transform_(gray)
    
    # Add batch dimension (model expects a batch of images)
    transformed_image = transformed_image.unsqueeze(0)
    
    return transformed_image

# Function to classify each character using the CNN
def classify_character(char_img, model, transform_):
    """Classify a single character using the trained CNN."""
    transformed_image = preprocess_character_for_cnn(char_img, transform_)
    
    with torch.no_grad():  # Don't need gradients for inference
        output = model(transformed_image)
    
    # Get the predicted class (character)
    _, predicted_class = torch.max(output, 1)
    
    # Map index to character (EMNIST includes lowercase + uppercase letters)
    idx_to_char = [chr(i + 97) for i in range(26)] + [chr(i + 65) for i in range(26)]
    predicted_letter = idx_to_char[predicted_class.item()]
    
    return predicted_letter

# Function to extract the full name from the name region
def extract_full_name(name_region, model, transform_):
    """Extract and recognize the full name from the name region."""
    characters = segment_name_into_characters(name_region)  # Segment into individual characters
    full_name = ""
    
    for i, char_img in enumerate(characters):
        letter = classify_character(char_img, model, transform_)  # Classify each character
        full_name += letter  # Add the character to the full name
    
    return full_name


# %%
plt.imshow(NR);
plt.show()

# %%

# %%
# Main execution
if __name__ == "__main__":
    # Load your PDF page (assuming page is read with pdf2image and converted to a name region)
    # Example: pages = convert_from_path('path_to_pdf', dpi=300)
    # Let's assume pages[1] is the page you're working with

    name_region = NR  # This is the part of the page that contains the name box

    # Convert name_region to a format suitable for processing (e.g., numpy array, BGR)
    name_region = cv2.cvtColor(np.array(name_region), cv2.COLOR_RGB2BGR)

    # Now call the function to get the full name
    full_name = extract_full_name(name_region, DL_name_reading_model, transform)

    print("Extracted Full Name:", full_name)


# %%

# %% [markdown]
# # Update and play
#
# one change at a time. lets first play with ```segment_name_into_characters```

# %%
def segment_name_into_characters(name_region):
    """
    Segment the name region into individual characters.
    """
    try:
        gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)  # Convert BGR to grayscale
    except Exception as e:
        print(f"Error converting to grayscale: {e}")
        gray = name_region  # If already grayscale, continue with it
    
    # Apply threshold to obtain a binary image (invert for white text on black background)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours to isolate each character
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    char_images = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 20:  # Filter out noise (small contours)
            char_img = name_region[y:y+h, x:x+w]
            char_images.append(char_img)
    
    return char_images


# %%
def segment_name_into_characters(name_region):
    """Segment the name region into individual characters."""
    try:
        gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)  # Convert BGR to grayscale
    except Exception as e:
        print(f"Error converting to grayscale: {e}")
        gray = name_region  # If already grayscale, continue with it

    # Apply threshold to obtain a binary image (invert for white text on black background)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours to isolate each character
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    char_images = []
    
    # Create a copy of the original image to draw contours on
    contour_image = name_region.copy()

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 1 and h > 1:  # Filter out noise (small contours)
            char_img = name_region[y:y+h, x:x+w]
            char_images.append(char_img)
            # Draw the contour on the image for visualization
            cv2.rectangle(contour_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Use matplotlib to display the image inline
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off axis
    plt.show()
    
    print(f"Number of detected character boxes: {len(char_images)}")
    
    return char_images



# %%
# Main execution
if __name__ == "__main__":
    # Load your PDF page (assuming page is read with pdf2image and converted to a name region)
    # Example: pages = convert_from_path('path_to_pdf', dpi=300)
    # Let's assume pages[1] is the page you're working with

    name_region = NR  # This is the part of the page that contains the name box

    # Convert name_region to a format suitable for processing (e.g., numpy array, BGR)
    name_region = cv2.cvtColor(np.array(name_region), cv2.COLOR_RGB2BGR)

    # Now call the function to get the full name
    full_name = extract_full_name(name_region, DL_name_reading_model, transform)

    print("Extracted Full Name:", full_name)


# %%

# %%
def segment_name_into_characters(name_region):
    """Segment the name region into individual characters."""
    try:
        gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)  # Convert BGR to grayscale
    except Exception as e:
        print(f"Error converting to grayscale: {e}")
        gray = name_region  # If already grayscale, continue with it

    # Apply adaptive thresholding for better segmentation
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 2) # 15, 2

    # Apply some morphological operations to clean the image (dilation + erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours to isolate each character
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    char_images = []
    
    # Create a copy of the original image to draw contours on
    contour_image = name_region.copy()

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Adjust these thresholds if necessary to ensure all characters are detected
        if w > 1 and h > 1 and w < 50:  # Filter out noise and large contours
            char_img = name_region[y:y+h, x:x+w]
            char_images.append(char_img)
            # Draw the contour on the image for visualization
            cv2.rectangle(contour_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the contour image inline using matplotlib
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off axis
    plt.show()

    print(f"Number of detected character boxes: {len(char_images)}")
    
    return char_images



# %%
# Main execution
if __name__ == "__main__":
    # Load your PDF page (assuming page is read with pdf2image and converted to a name region)
    # Example: pages = convert_from_path('path_to_pdf', dpi=300)
    # Let's assume pages[1] is the page you're working with

    name_region = NR  # This is the part of the page that contains the name box

    # Convert name_region to a format suitable for processing (e.g., numpy array, BGR)
    name_region = cv2.cvtColor(np.array(name_region), cv2.COLOR_RGB2BGR)

    # Now call the function to get the full name
    full_name = extract_full_name(name_region, DL_name_reading_model, transform)

    print("Extracted Full Name:", full_name)


# %%

# %%
def segment_name_into_characters(name_region):
    """Segment the name region into individual characters."""
    try:
        gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)  # Convert BGR to grayscale
    except Exception as e:
        print(f"Error converting to grayscale: {e}")
        gray = name_region  # If already grayscale, continue with it

    # Apply adaptive thresholding for better segmentation
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 2)

    # Apply some morphological operations to clean the image (erosion followed by dilation)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Erosion + Dilation

    # Find contours to isolate each character
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    char_images = []
    
    # Create a copy of the original image to draw contours on
    contour_image = name_region.copy()

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # New filters: min size (area), aspect ratio, and height/width limits
        if w > 1 and h > 5 and w < 50 and 0.1 < float(h) / w < 3.0:  # Adjusting the aspect ratio and filtering size
            char_img = name_region[y:y+h, x:x+w]
            char_images.append(char_img)
            # Draw the contour on the image for visualization
            cv2.rectangle(contour_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the contour image inline using matplotlib
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off axis
    plt.show()

    print(f"Number of detected character boxes: {len(char_images)}")
    
    return char_images



# %%
# Main execution
if __name__ == "__main__":
    # Load your PDF page (assuming page is read with pdf2image and converted to a name region)
    # Example: pages = convert_from_path('path_to_pdf', dpi=300)
    # Let's assume pages[1] is the page you're working with

    name_region = NR  # This is the part of the page that contains the name box

    # Convert name_region to a format suitable for processing (e.g., numpy array, BGR)
    name_region = cv2.cvtColor(np.array(name_region), cv2.COLOR_RGB2BGR)

    # Now call the function to get the full name
    full_name = extract_full_name(name_region, DL_name_reading_model, transform)

    print("Extracted Full Name:", full_name)


# %%

# %%
def segment_name_into_characters(name_region):
    """Segment the name region into individual characters."""
    try:
        gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)  # Convert BGR to grayscale
    except Exception as e:
        print(f"Error converting to grayscale: {e}")
        gray = name_region  # If already grayscale, continue with it

    # Apply adaptive thresholding for better segmentation
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 2)

    # Apply some morphological operations to clean the image (erosion followed by dilation)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Erosion + Dilation

    # Find contours to isolate each character
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    char_images = []
    
    # Create a copy of the original image to draw contours on
    contour_image = name_region.copy()

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # New filters: min size (area), aspect ratio, and height/width limits
        if w > 1 and h > 5 and w < 50 and 0.1 < float(h) / w < 3.0:  # Adjusting the aspect ratio and filtering size
            char_img = name_region[y:y+h, x:x+w]
            char_images.append(char_img)
            # Draw the contour on the image for visualization
            cv2.rectangle(contour_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Debugging: Show each detected character in a separate plot
            plt.imshow(cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')  # Turn off axis
            plt.show()

    # Display the contour image inline using matplotlib
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off axis
    plt.show()

    print(f"Number of detected character boxes: {len(char_images)}")
    
    return char_images



# %%
# Main execution
if __name__ == "__main__":
    # Load your PDF page (assuming page is read with pdf2image and converted to a name region)
    # Example: pages = convert_from_path('path_to_pdf', dpi=300)
    # Let's assume pages[1] is the page you're working with

    name_region = NR  # This is the part of the page that contains the name box

    # Convert name_region to a format suitable for processing (e.g., numpy array, BGR)
    name_region = cv2.cvtColor(np.array(name_region), cv2.COLOR_RGB2BGR)

    # Now call the function to get the full name
    full_name = extract_full_name(name_region, DL_name_reading_model, transform)

    print("Extracted Full Name:", full_name)

# %%

# %%
importlib.reload(mg);

# Define a transformation (resizing, grayscale conversion, normalization)
transform_ = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),  # Resize to 28x28 (for EMNIST model)
            transforms.Grayscale(num_output_channels=1),  # Grayscale
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(
                (0.5,), (0.5,)
            ),  # Normalize based on EMNIST dataset stats
        ]
    )

# %%
# Suppose you have tight name_region (numpy array)

# %%
if __name__ == "__main__":
    # Load your PDF page (assuming page is read with pdf2image and converted to a name region)
    # Example: pages = convert_from_path('path_to_pdf', dpi=300)
    # Let's assume pages[1] is the page you're working with

    name_region = NR  # This is the part of the page that contains the name box

    # Convert name_region to a format suitable for processing (e.g., numpy array, BGR)
    name_region = cv2.cvtColor(np.array(name_region), cv2.COLOR_RGB2BGR)

    # Now call the function to get the full name
    full_name = mg.predict_name_from_region(name_region, model=DL_name_reading_model, device=device,
                                            transform=transform_, num_boxes=20)

    print("Extracted Full Name:", full_name)

# %%

# %%
plt.imshow(name_region)

# %%

# %% [markdown]
# # Let us Focus on answer part.

# %%
sy1, sy2 = 300, 2420
sx1, sx2 = 1500, 4800

# %%
image = np.array(pages[0])  # Convert the first page to a numpy array

# Crop the right half of the image (assuming the left part is for name/ID)
height, width, _ = image.shape
# cropped_image = image[:, width // 2:]  # right half of the image (answer area)
cropped_image = image[sy1:sy2, sx1:sx2]
plt.imshow(cropped_image)
plt.axis("off");

# %%

# %%
# Convert the cropped image to grayscale (since it's an RGB image)
gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)

# Apply thresholding to get a binary image (invert for filled bubbles)
_, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)

# %%
plt.imshow(binary_image)
plt.axis("off");

# %%
# Use HoughCircles to detect the bubbles (circles) in the binary image
circles = cv2.HoughCircles(binary_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                            param1=50, param2=30, minRadius=10, maxRadius=30)
# Convert the circles to integer values
circles = np.uint16(np.around(circles))

# %%
# Initialize a dictionary to store the student's answers
student_answers = {}

# For each circle (representing an answer bubble)
for idx, circle in enumerate(circles[0, :]):
    x, y, radius = circle[0], circle[1], circle[2]
    
    # Crop the region of interest around the circle
    roi = binary_image[y - radius:y + radius, x - radius:x + radius]
    
    # Count the number of filled pixels inside the bubble
    filled_pixels = np.count_nonzero(roi)
    threshold = 0.5  # threshold of 50% to consider the bubble filled
    is_filled = filled_pixels > (radius ** 2 * np.pi * threshold)  # Area of the circle * 50%
    
    # If filled, map it to the answer (A-D) based on its position
    if is_filled:
        # Map the circle's position (index) to the answer (A, B, C, D)
        answer = chr(65 + idx % 4)  # A=65, B=66, C=67, D=68 for first 4 circles

        # Record the student's answer (assuming question 1 is the first detected bubble)
        question_num = idx + 1
        student_answers[question_num] = answer

# Print the extracted answers (for debugging)
print("Student's Answer Sheet:")
for question, answer in student_answers.items():
    print(f"Question {question}: {answer}")

# Show the binary image for debugging
plt.imshow(binary_image, cmap='gray')
plt.axis('off')
plt.show()


# %% [markdown]
# # Use Contour method
#
# Maybe filling goes outside the circle, then circle detection fails.

# %%
# Find contours (bubbles) in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a dictionary to store the student's answers
student_answers = {}

# Loop through each contour (representing a bubble)
for idx, contour in enumerate(contours):
    # Get the bounding box of each contour (bubble)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Crop the region of interest around the bubble
    roi = binary_image[y:y+h, x:x+w]
    
    # Count the number of filled pixels inside the bubble (non-zero pixels)
    filled_pixels = np.count_nonzero(roi)
    
    # Define a threshold for filled bubbles (e.g., 50% of the bubble area)
    threshold = 0.5  # 50% filled
    total_pixels = w * h
    is_filled = filled_pixels > total_pixels * threshold
    
    # If the bubble is filled, map it to the answer (A, B, C, D)
    if is_filled:
        # Map the contour index to the answer (A, B, C, D)
        answer = chr(65 + idx % 4)  # A=65, B=66, C=67, D=68 for first 4 circles
        
        # Record the student's answer (assuming question 1 is the first detected bubble)
        question_num = idx + 1  # Assuming the first bubble corresponds to question 1
        student_answers[question_num] = answer

# Print the extracted answers (for debugging)
print("Student's Answer Sheet:")
for question, answer in student_answers.items():
    print(f"Question {question}: {answer}")

# Show the binary image for debugging (optional)
plt.imshow(binary_image, cmap='gray')
plt.axis('off')
plt.show()


# %%
plt.imshow(binary_image)
plt.axis("off");

# %%
# Find contours (bubbles) in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a dictionary to store the student's answers
student_answers = {}

# Define grid structure: 
questions_per_row = 5  # Assuming there are 5 bubbles (A-D) per row
question_offset = 1     # Start with question 1

# Loop through each contour (representing a bubble)
for idx, contour in enumerate(contours):
    # Get the bounding box of each contour (bubble)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Crop the region of interest around the bubble
    roi = binary_image[y:y+h, x:x+w]
    
    # Count the number of filled pixels inside the bubble (non-zero pixels)
    filled_pixels = np.count_nonzero(roi)
    
    # Define a threshold for filled bubbles (e.g., 50% of the bubble area)
    threshold = 0.5  # 50% filled
    total_pixels = w * h
    is_filled = filled_pixels > total_pixels * threshold
    
    # If the bubble is filled, map it to the answer (A, B, C, D)
    if is_filled:
        # Find the row and column position of the bubble
        row = y // 100  # Approximate row (assuming vertical spacing of bubbles)
        col = x // 100  # Approximate column (assuming horizontal spacing)
        
        # Calculate the question number based on the row and column
        question_num = row * questions_per_row + col + question_offset
        
        # Map the contour's position (column) to the answer (A, B, C, D)
        answer = chr(65 + col)  # A=65, B=66, C=67, D=68 for first 4 columns
        
        # Record the student's answer (question number: answer)
        student_answers[question_num] = answer

# Show the binary image for debugging (optional)
plt.imshow(binary_image, cmap='gray')
plt.axis('off')
plt.show()

# %%
print("Student's Answer Sheet:")
for question, answer in student_answers.items():
    print(f"Question {question}: {answer}")

# %%

# %%
# Find contours (bubbles) in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a dictionary to store the student's answers
student_answers = {}

# Define grid structure: 
questions_per_row = 5  # Assuming there are 5 bubbles (A-D) per row
question_offset = 1     # Start with question 1

# Define an area to avoid smaller contours (noise)
min_contour_area = 100

# Loop through each contour (representing a bubble)
for idx, contour in enumerate(contours):
    # Get the bounding box of each contour (bubble)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Skip small contours (likely noise)
    if cv2.contourArea(contour) < min_contour_area:
        continue
    
    # Crop the region of interest around the bubble
    roi = binary_image[y:y+h, x:x+w]
    
    # Count the number of filled pixels inside the bubble (non-zero pixels)
    filled_pixels = np.count_nonzero(roi)
    
    # Define a threshold for filled bubbles (e.g., 50% of the bubble area)
    threshold = 0.5  # 50% filled
    total_pixels = w * h
    is_filled = filled_pixels > total_pixels * threshold  # Check if the bubble is filled
    
    # If the bubble is filled, map it to the answer (A, B, C, D)
    if is_filled:
        # Find the row and column position of the bubble
        row = y // 100  # Approximate row (assuming vertical spacing of bubbles)
        col = x // 100  # Approximate column (assuming horizontal spacing)
        
        # Calculate the question number based on the row and column
        question_num = row * questions_per_row + col + question_offset
        
        # Map the contour's position (column) to the answer (A, B, C, D)
        answer = chr(65 + col)  # A=65, B=66, C=67, D=68 for first 4 columns
        
        # Record the student's answer (question number: answer)
        student_answers[question_num] = answer

# Show the binary image for debugging (optional)
plt.imshow(binary_image, cmap='gray')
plt.axis('off')
plt.show()

# %%
print (len(student_answers))

# %%

# %%
# Find contours (bubbles) in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a dictionary to store the student's answers
student_answers = {}

# Define grid structure: 
questions_per_row = 5  # Assuming there are 5 bubbles (A-D) per row
question_offset = 1     # Start with question 1

# Define an area to avoid smaller contours (noise)
min_contour_area = 100

# Loop through each contour (representing a bubble)
for idx, contour in enumerate(contours):
    # Stop if we reach the max number of questions
    if len(student_answers) >= question_count:
        break

    # Get the bounding box of each contour (bubble)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Skip small contours (likely noise)
    if cv2.contourArea(contour) < min_contour_area:
        continue
    
    # Crop the region of interest around the bubble
    roi = binary_image[y:y+h, x:x+w]
    
    # Count the number of filled pixels inside the bubble (non-zero pixels)
    filled_pixels = np.count_nonzero(roi)
    
    # Define a threshold for filled bubbles (e.g., 50% of the bubble area)
    threshold = 0.5  # 50% filled
    total_pixels = w * h
    is_filled = filled_pixels > total_pixels * threshold  # Check if the bubble is filled
    
    # If the bubble is filled, map it to the answer (A, B, C, D)
    if is_filled:
        # Find the row and column position of the bubble
        row = y // 100  # Approximate row (assuming vertical spacing of bubbles)
        col = x // 100  # Approximate column (assuming horizontal spacing)
        
        # Calculate the question number based on the row and column
        question_num = row * questions_per_row + col + question_offset
        
        # Ensure we don’t exceed the max number of questions
        if question_num > question_count:
            break
        
        # Map the contour's position (column) to the answer (A, B, C, D)
        answer = chr(65 + col)  # A=65, B=66, C=67, D=68 for first 4 columns
        
        # Record the student's answer (question number: answer)
        student_answers[question_num] = answer

# Print the extracted answers (for debugging)
print (len(student_answers))
print("Student's Answer Sheet:")
for question, answer in student_answers.items():
    print(f"Question {question}: {answer}")

# Show the binary image for debugging (optional)
plt.imshow(binary_image, cmap='gray')
plt.axis('off')
plt.show()

# %%

# %%
# Apply thresholding to get a binary image (invert for filled bubbles)
_, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)

# Show the binary image for debugging
plt.imshow(binary_image, cmap='gray')
plt.axis('off')
plt.show()

# %%
plt.imshow(binary_image[start_y:start_y+100, start_x:start_x+400])
plt.axis("off");

# %%
# Initialize a dictionary to store the student's answers
student_answers = {}

# Set the number of questions and how many bubbles (choices A-E) per row
questions_per_row = 5  # Adjust for number of columns with choices (A-E)
bubble_size = 30  # Estimated size of the bubble (adjust as needed)
gap_size = 10  # The gap between bubbles

# Starting coordinates (estimate based on the PDF structure)
start_x = 50  # Starting X position (where questions begin)
start_y = 50  # Starting Y position (row for first question)
question_height = 100  # Height of each row for questions
column_width = 400  # Width of each column for bubbles

# Loop through the grid and check if the bubble is filled
for row in range(0, question_count):  # Rows for each question (up to question_count)
    for col in range(0, questions_per_row):  # Columns for the choices A-E
        # Calculate the position of each bubble in the grid
        x = start_x + col * (bubble_size + gap_size)  # Horizontal position
        y = start_y + row * (question_height + gap_size)  # Vertical position
        
        # Define the bounding box for the bubble (adjusting position)
        roi = binary_image[y:y + bubble_size, x:x + bubble_size]
        
        # Count the number of filled pixels inside the bubble (non-zero pixels)
        filled_pixels = np.count_nonzero(roi)
        
        # Define a threshold for filled bubbles (e.g., 50% of the bubble area)
        threshold = 0.5  # 50% filled
        total_pixels = bubble_size * bubble_size
        is_filled = filled_pixels > total_pixels * threshold  # Check if the bubble is filled
        
        # If the bubble is filled, map it to the answer (A, B, C, D)
        if is_filled:
            # Map the bubble to the answer (A, B, C, D, E) based on column index
            answer = chr(65 + col)  # A=65, B=66, C=67, D=68, E=69
            
            # Record the student's answer (question number: answer)
            question_num = row * questions_per_row + col + 1
            student_answers[question_num] = answer

# Print the extracted answers (for debugging)
print("Student's Answer Sheet:")
for question, answer in student_answers.items():
    print(f"Question {question}: {answer}")


# %%
plt.imshow(binary_image[y:y+100, x:x+400])
plt.axis("off");

# %%
# Find contours (bubbles) in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a dictionary to store the student's answers
student_answers = {}

# Set the maximum number of questions (for limiting detected answers)
question_counter = 1

# Loop through each contour (representing a bubble)
for idx, contour in enumerate(contours):
    # Get the bounding box of each contour (bubble)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Skip small contours (likely noise or irrelevant parts of the sheet)
    if cv2.contourArea(contour) < 100:  # Adjust the threshold as needed
        continue
    
    # Crop the region of interest around the bubble
    roi = binary_image[y:y + h, x:x + w]
    # Count the number of filled pixels inside the bubble (non-zero pixels)
    filled_pixels = np.count_nonzero(roi)
    
    # Define a threshold for filled bubbles (e.g., 50% of the bubble area)
    threshold = 0.5  # 50% filled
    total_pixels = w * h
    is_filled = filled_pixels > total_pixels * threshold  # Check if the bubble is filled
    
    # If the bubble is filled, map it to the answer (A, B, C, D)
    if is_filled and question_counter <= question_count:
        # Map the contour's position (index) to the answer (A, B, C, D, E)
        answer = chr(65 + idx % 5)  # A=65, B=66, C=67, D=68, E=69 for first 5 columns
        
        # Record the student's answer (question number: answer)
        student_answers[question_counter] = answer
        question_counter += 1

print (len(student_answers))
# Show the binary image for debugging (optional)
plt.imshow(binary_image, cmap='gray')
plt.axis('off')
plt.show()

# %%
student_answers

# %%

# %%

# %%
