
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import remove_small_objects
from tkinter import Tk, Toplevel, Label
from tkinter.filedialog import askopenfilename

# Conversion factor from pixels to millimeters
px_to_mm = 0.1  # Example value, you should replace this with your actual conversion factor
px_to_inch = px_to_mm / 25.4  # 1 inch = 25.4 mm

# Function to select a file
def select_file():
    Tk().withdraw()  # We don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(title='Pick an image file', filetypes=[('All Files', '.')])
    return filename

# Function to show details in a popup window
def show_details(stone_index, stone_area, centroid):
    area_mm = stone_area * (px_to_mm ** 2)
    area_inch = stone_area * (px_to_inch ** 2)
    
    details_window = Toplevel()
    details_window.title(f"Stone {stone_index} Details")
    
    details = [
        f"Stone {stone_index}",
        f"Area: {stone_area} pixels",
        f"Area: {area_mm:.2f} mm²",
        f"Centroid: {centroid}",
        # Add more details as required
    ]
    
    for detail in details:
        Label(details_window, text=detail).pack()

# Callback function for mouse click event
def on_click(event):
    x = int(event.xdata)
    y = int(event.ydata)
    
    stone_index = labels[y, x]
    
    if stone_index > 0:
        stone_area = stats[stone_index, cv2.CC_STAT_AREA]
        centroid = (centroids[stone_index][0], centroids[stone_index][1])
        show_details(stone_index, stone_area, centroid)

# Select a file
filename = select_file()
a = cv2.imread(filename)

# Convert to grayscale
b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

# Apply Gaussian filter
b = cv2.GaussianBlur(b, (5, 5), 0)

# Binary thresholding
_, c = cv2.threshold(b, 20, 255, cv2.THRESH_BINARY)

# Fill holes
d = cv2.morphologyEx(c, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

# Remove small objects
e = remove_small_objects(d.astype(bool), 1000)
e = e.astype(np.uint8)

# Preprocess image
PreprocessedImage = cv2.bitwise_and(a, a, mask=e)

# Adjust image intensity
PreprocessedImage = cv2.normalize(PreprocessedImage, None, alpha=50, beta=255, norm_type=cv2.NORM_MINMAX)

# Convert to grayscale again
uo = cv2.cvtColor(PreprocessedImage, cv2.COLOR_BGR2GRAY)

# Apply median filter
mo = cv2.medianBlur(uo, 3)
bilateral_filtered_image = cv2.bilateralFilter(mo, d=9, sigmaColor=75, sigmaSpace=75)

threshold = 220  # Adjust this value as needed
_, mask = cv2.threshold(bilateral_filtered_image, threshold, 255, cv2.THRESH_BINARY)

# Create an all-white image
masked_image = np.ones_like(bilateral_filtered_image) * 255

# Apply the mask to keep high-intensity parts
masked_image[mask == 255] = bilateral_filtered_image[mask == 255]

# Check if the masked image is blank (i.e., all pixels are the same)
if np.all(masked_image == 255):
    print("No stone detected")
else:
    print("Stone detected")

    red_mask = np.zeros_like(a)
    red_mask[mask == 255] = [0,0, 255]  # Red color in BGR

    # Overlay the red mask on the original image
    highlighted_image = cv2.addWeighted(a, 0.7, red_mask, 0.3, 20)

    # Calculate the size of detected parts
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    stone_areas = []
    for i in range(1, num_labels):  # Skip the background
        area = stats[i, cv2.CC_STAT_AREA]
        area_mm = area * (px_to_mm ** 2)
        area_inch = area * (px_to_inch ** 2)
        stone_areas.append(area)
        print(f"Stone {i}: Area = {area} pixels, {area_mm:.2f} mm²")

    # Optionally, you can visualize the labeled stones
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_channel = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_channel, blank_channel])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0  # Set background label to black

    # Create subplots to display images in one window
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Display original image
    axes[0, 0].imshow(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Display masked image
    axes[0, 1].imshow(masked_image, cmap='gray')
    axes[0, 1].set_title('Masked Image')
    axes[0, 1].axis('off')
    
    # Display highlighted image
    axes[1, 0].imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Stone Highlighted in Red')
    axes[1, 0].axis('off')
    
    # Display labeled stones
    axes[1, 1].imshow(labeled_img)
    axes[1, 1].set_title('Labeled Stones')
    axes[1, 1].axis('off')
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust spacing
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
