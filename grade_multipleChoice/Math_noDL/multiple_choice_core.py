import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import cv2
import pytesseract
from pdf2image import convert_from_path

import matplotlib.pyplot as plt

"""
If PDF files are read via OpenCV,
then the default order of chanels are BGR so we need COLOR_BGR2GRAY.
If it is read via PIL() then we need COLOR_RGB2GRAY.
PIL: from pdf2image import convert_from_path

I read it via PIL so, here I stick to RGB2GRAY
"""


def predict_name_from_region(name_region, model, device, transform, num_boxes=20):
    """
    Predicts full name from a tight name_region using EMNIST CNN.
    """
    boxes = split_name_region_into_boxes(name_region, num_boxes=num_boxes)
    full_name = ""

    for box in boxes:
        # Preprocess
        tensor = preprocess_box_for_cnn(box, transform)
        tensor = tensor.to(device)  # move to GPU if needed

        # Predict
        with torch.no_grad():
            output = model(tensor)
            pred_idx = output.argmax(dim=1).item()

        # EMNIST 'letters' dataset has labels 1-26 = A-Z
        # We trained with 26 lowercase + 26 uppercase mapping
        if pred_idx < 26:
            char = chr(pred_idx + ord("a"))  # lowercase
        else:
            char = chr(pred_idx - 26 + ord("A"))  # uppercase

        # Optionally, check if box is empty (sum of pixels very low)
        if np.mean(box) < 10:  # tweak threshold based on image
            char = " "  # empty box = space

        full_name += char

    return full_name


def split_name_region_into_boxes(name_region, num_boxes=20):
    """
    Splits the tight name_region into equal-width boxes.
    Args:
        name_region: numpy array (grayscale or BGR)
        num_boxes: total number of letter boxes (including empty spaces)
    Returns:
        list of numpy arrays (one per box)
    """
    h, w = name_region.shape[:2]
    box_width = w // num_boxes
    boxes = []

    for i in range(num_boxes):
        x1 = i * box_width
        x2 = x1 + box_width
        box = name_region[:, x1:x2]
        boxes.append(box)
    return boxes


def preprocess_box_for_cnn(box_image, transform):
    """
    Prepares a single letter box image for EMNIST CNN.
    Args:
        box_image: numpy array (grayscale or BGR)
        transform: torchvision transforms used during training
    Returns:
        tensor of shape (1, 1, 28, 28)
    """
    try:
        # Convert to grayscale if needed
        if len(box_image.shape) == 3:
            box_image = cv2.cvtColor(box_image, cv2.COLOR_RGB2GRAY)
    except:
        pass

    # Resize to 28x28 (EMNIST standard)
    resized = cv2.resize(box_image, (28, 28), interpolation=cv2.INTER_AREA)

    # Convert to tensor and normalize (same as training)
    tensor = transform(resized)
    tensor = tensor.unsqueeze(0)  # add batch dimension
    return tensor


def extract_full_name(name_region, model, transform_):
    """Extract and recognize the full name from the name region."""
    characters = segment_name_into_characters(name_region)
    full_name = ""
    for char_img in characters:
        letter = classify_character(char_img, model, transform_)
        full_name += letter
    return full_name


def classify_character(char_img, model, transform_):
    """Classify a single character using the trained CNN."""
    transformed_image = preprocess_character_for_cnn(char_img, transform_)
    with torch.no_grad():
        output = model(transformed_image)
    _, predicted_class = torch.max(output, 1)
    idx_to_char = [chr(i + 97) for i in range(26)] + [chr(i + 65) for i in range(26)]
    predicted_letter = idx_to_char[predicted_class.item()]
    return predicted_letter


def segment_name_into_characters(gray_name_region):
    """Segment the name region into individual characters."""
    # gray = cv2.cvtColor(name_region, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(
        gray_name_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    char_images = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 20:
            char_img = name_region[y : y + h, x : x + w]
            char_images.append(char_img)
    return char_images


def classify_nameRegion(name_region, model, transform_):
    """Classify the entire name region by processing character-by-character."""
    # Step 1: Preprocess the name region for the model
    transformed_image = preprocess_nameRegion_4_cnn(name_region, transform_)

    # Step 2: Get the prediction from the model
    with torch.no_grad():
        output = model(transformed_image)

    # Step 3: Get the predicted class (character)
    _, predicted_class = torch.max(output, 1)

    # Map the predicted class index to a character (EMNIST has 26 lowercase + 26 uppercase)
    idx_to_char = [chr(i + 97) for i in range(26)] + [
        chr(i + 65) for i in range(26)
    ]  # lowercase + uppercase
    predicted_letter = idx_to_char[predicted_class.item()]

    return predicted_letter


def preprocess_character_for_cnn(char_img, transform_):
    """Preprocess a single character image for the CNN."""
    if len(char_img.shape) == 3:
        gray = cv2.cvtColor(char_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = char_img
    transformed_image = transform_(gray)
    transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension
    return transformed_image


def preprocess_nameRegion_4_cnn(name_region, transform_):
    """Preprocess only the name region of the image for the model."""
    # Convert to grayscale if needed (in case the image isn't grayscale already)
    # gray_image = cv2.cvtColor(name_region, cv2.COLOR_RGB2GRAY)

    # Apply transformations (resize, normalize, etc.)
    # transformed_image = transform_(gray_image)
    if len(name_region.shape) == 3:
        gray = cv2.cvtColor(name_region, cv2.COLOR_RGB2GRAY)
    else:
        gray = name_region

    transformed_image = transform_(gray)

    # Add a batch dimension (model expects a batch)
    transformed_image = transformed_image.unsqueeze(0)

    return transformed_image


# Simple CNN for EMNIST
class EMNIST_Model_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            # the following is identical to nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 52),  # 26 lowercase + 26 uppercase
        )

    def forward(self, x):
        x = self.conv(x)
        # reshape, just like nn.Flatten():
        # And this could have gone into self.fc
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def split_boxes(cleaned, box_width):
    h, w = cleaned.shape
    boxes = []
    for x in range(0, w, box_width):
        box = cleaned[:, x : x + box_width]
        if box.shape[1] > box_width * 0.8:
            boxes.append(box)
    return boxes


def ocr_boxes(boxes):
    name = ""
    for box in boxes:
        # Resize to make letters bigger
        box_resized = cv2.resize(box, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        char = pytesseract.image_to_string(
            box_resized,
            config="--psm 10 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        ).strip()
        if len(char) == 1:
            name += char
        else:
            name += " "  # empty box
    return " ".join(name.split())


def split_boxes_v1(cleaned, box_width=60):
    h, w = cleaned.shape
    boxes = []

    for x in range(0, w, box_width):
        box = cleaned[:, x : x + box_width]
        if box.shape[1] < box_width * 0.8:
            continue
        boxes.append(box)

    return boxes


def ocr_boxes_v1(boxes):
    name = ""

    for box in boxes:
        box = cv2.resize(box, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        char = pytesseract.image_to_string(
            box,
            config="--psm 10 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        ).strip()

        if len(char) == 1:
            name += char
        else:
            name += " "  # empty square

    return name.strip()


def remove_grid(name_region):
    """
    no horizontal kernels here
    """
    # Invert for morphological ops
    bw = cv2.adaptiveThreshold(
        name_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )

    # Remove vertical lines (thin kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    vertical_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical_kernel)

    cleaned = cv2.subtract(bw, vertical_lines)

    # Optional: dilate lightly to reconnect faint strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    return cleaned


def remove_grid_v1(gray):
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    """
    2, 20 leaves some of the vertical lines in there.
    """
    # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 20))
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical_kernel)

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_kernel)

    grid = cv2.bitwise_or(vertical, horizontal)
    cleaned = cv2.subtract(bw, grid)

    return cleaned


def show(img, title=None):
    plt.figure(figsize=(12, 3))
    plt.imshow(img, cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def preprocess_right_side(image):
    # Assuming the right 2/3 of the image contains answers
    height, width = image.shape[:2]
    right = image[:, width // 3 :]  # crop right side

    gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    return thresh


def read_name(image):
    h, w = image.shape[:2]
    left = image[:, : int(w * LEFT_CROP_RATIO)]
    gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return pytesseract.image_to_string(thresh, config="--psm 7").strip()


def read_answers_dynamicBubbleThres(image, num_questions, min_fill_ratio=0.1):
    """
    Reads answers from the right side automatically.
    min_fill_ratio: minimum fraction of the row that must be filled to count as an answer
    """
    h, w = image.shape[:2]
    right = image[:, int(w * LEFT_CROP_RATIO) :]
    gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    step = h // TOTAL_QUESTIONS  # template: 100 questions
    answers = {}

    for q in range(num_questions):
        y1, y2 = q * step, (q + 1) * step
        row = thresh[y1:y2, :]
        bubble_w = row.shape[1] // 5

        scores = [
            cv2.countNonZero(row[:, i * bubble_w : (i + 1) * bubble_w])
            for i in range(5)
        ]
        max_score = max(scores)
        total_pixels = row.shape[0] * bubble_w

        # Automatically decide if any bubble is filled
        if max_score / total_pixels >= min_fill_ratio:
            answers[q + 1] = "ABCDE"[scores.index(max_score)]
        else:
            answers[q + 1] = None  # no answer marked

    return answers


def read_answers(right_thresh, num_questions):
    answers = {}

    for q in range(num_questions):
        y = START_Y + q * DY
        scores = []

        for i, x in enumerate(BUBBLE_X):
            score = bubble_fill_score(right_thresh, x, y, R)
            scores.append(score)

        max_score = max(scores)
        if max_score < 150:
            answers[q + 1] = None
        else:
            answers[q + 1] = scores.index(max_score)

    return answers


def get_student_name(image):
    # Convert PIL image to OpenCV
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Crop the left part of the page (adjust coordinates)
    height, width, _ = img.shape
    left_crop = img[:, : width // 3]  # adjust fraction if needed

    # Optional preprocessing
    gray = cv2.cvtColor(left_crop, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # OCR
    text = pytesseract.image_to_string(thresh, config="--psm 7")  # single line
    return text.strip()


def get_name(image):
    height, width = image.shape[:2]
    left = image[:, : width // 3]  # crop left part for name

    gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    name = pytesseract.image_to_string(thresh, config="--psm 7")
    return name.strip()


def get_answers(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    height, width, _ = img.shape
    right_crop = img[:, width // 3 :]  # right 2/3 for answers

    gray = cv2.cvtColor(right_crop, cv2.COLOR_RGB2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    """
    I am not syre what name_region is below
    Maybe try:
    name_region = extract_name_region(page_img, y1, y2, x1, x2)
    _, name_region = cv2.threshold(name_region, 150, 255, cv2.THRESH_BINARY)

    """
    blurred = cv2.medianBlur(name_region, 3)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    answers = []

    for c in contours:
        # Filter contours by size (ignore tiny noise)
        if 100 < cv2.contourArea(c) < 1000:
            x, y, w, h = cv2.boundingRect(c)
            # Could use y-position to order answers
            answers.append((y, x, w, h))

    # Sort contours by vertical position
    answers = sorted(answers, key=lambda x: x[0])

    # Map them to 1-5 answers per question
    # Here you might need to adjust based on layout
    student_answers = []
    for i in range(0, len(answers), 5):
        question_group = answers[i : i + 5]
        # Find the filled circle (largest black area)
        filled_index = None
        max_black = 0
        for idx, (y, x, w, h) in enumerate(question_group):
            circle_roi = thresh[y : y + h, x : x + w]
            black_pixels = cv2.countNonZero(circle_roi)
            if black_pixels > max_black:
                max_black = black_pixels
                filled_index = idx
        # Answer A-E
        student_answers.append("ABCDE"[filled_index])

    return student_answers


def extract_name_removeVertical(page_img, y1=200, y2=280, x1=120, x2=1400):
    gray = cv2.cvtColor(page_img, cv2.COLOR_RGB2GRAY)

    data = pytesseract.image_to_data(
        gray, output_type=pytesseract.Output.DICT, config="--psm 6"
    )

    name_region = extract_name_region(page_img, y1, y2, x1, x2)
    _, name_region = cv2.threshold(name_region, 150, 255, cv2.THRESH_BINARY)

    bw = cv2.adaptiveThreshold(
        name_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )

    # Remove vertical lines (thin kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    vertical_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical_kernel)

    cleaned = cv2.subtract(bw, vertical_lines)

    return pytesseract.image_to_string(cleaned, config="--psm 7").strip()


def extract_name(page_img, y1=200, y2=280, x1=120, x2=1400):
    gray = cv2.cvtColor(page_img, cv2.COLOR_RGB2GRAY)

    data = pytesseract.image_to_data(
        gray, output_type=pytesseract.Output.DICT, config="--psm 6"
    )

    ### Since our sheets are standard, we have NAME there.
    ### AND, this function did not use x, y, w, h neither.
    # for i, txt in enumerate(data["text"]):
    #     if txt.strip().upper() == "NAME":
    #         x = data["left"][i]
    #         y = data["top"][i]
    #         w = data["width"][i]
    #         h = data["height"][i]

    name_region = extract_name_region(page_img, y1, y2, x1, x2)
    _, name_region = cv2.threshold(name_region, 150, 255, cv2.THRESH_BINARY)
    return pytesseract.image_to_string(name_region, config="--psm 7").strip()

    # return ""


def extract_boxed_name(name_region):
    # Ensure grayscale
    if len(name_region.shape) == 3:
        gray = cv2.cvtColor(name_region, cv2.COLOR_RGB2GRAY)
    else:
        gray = name_region.copy()

    # Binarize
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove horizontal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    H = gray.shape[0]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 0.6 * H and w < 80:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])

    name = ""

    for x, y, w, h in boxes:
        char_img = gray[y : y + h, x : x + w]

        char_img = cv2.resize(char_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        _, char_img = cv2.threshold(
            char_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        char = pytesseract.image_to_string(
            char_img,
            config="--psm 10 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        ).strip()

        if char:
            name += char

    return name


def extract_name_region(image, y1=200, y2=280, x1=120, x2=1400):
    """
    # ---- FIXED COORDINATES (example – adjust once) ----
    y1, y2 = ,  # vertical span of handwritten name
    x1, x2 = ,  # horizontal span of handwritten name
    """
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    name_region = gray[y1:y2, x1:x2]

    # name_region = cv2.GaussianBlur(name_region, (3,3), 0)
    name_region = cv2.medianBlur(name_region, 3)

    _, name_region = cv2.threshold(
        name_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # return pytesseract.image_to_string(name_region, config="--psm 7").strip()
    return name_region


def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
