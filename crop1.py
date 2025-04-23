import cv2
import numpy as np
import os

# Define the output directory
output_dir = './anhmoi'  # Change this to your desired folder path

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the image
image = cv2.imread('./anh/13.jpeg', cv2.IMREAD_GRAYSCALE)

# Apply thresholding to binarize the image
_, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)

# Find contours of the characters
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Counter for naming the output images
count = 0

# Process each contour
for contour in contours:
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Filter out small contours (noise)
    if w > 10 and h > 10:
        # Adjust the bounding box to fit 50x150 pixels, centered on the character
        center_x, center_y = x + w//2, y + h//2
        new_x = max(0, center_x - 25)  # 50/2
        new_y = max(0, center_y - 75)  # 150/2
        
        # Ensure the crop stays within image bounds
        crop = image[new_y:new_y+150, new_x:new_x+50]
        
        # If the crop is smaller than 50x150 (near edges), pad it
        if crop.shape != (150, 50):
            crop = cv2.copyMakeBorder(
                crop,
                top=max(0, 0 - new_y),
                bottom=max(0, (new_y + 150) - image.shape[0]),
                left=max(0, 0 - new_x),
                right=max(0, (new_x + 50) - image.shape[1]),
                borderType=cv2.BORDER_CONSTANT,
                value=255  # White background
            )
            crop = crop[:150, :50]  # Trim to exact size if needed
        
        # Save the cropped image to the specified directory
        output_path = os.path.join(output_dir, f'13_{count}.png')
        cv2.imwrite(output_path, crop)
        count += 1

print(f"Saved {count} character images to {output_dir}.")