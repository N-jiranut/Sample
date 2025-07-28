import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Create a dummy image
img = np.zeros((480, 640, 3), dtype=np.uint8) # Height=480, Width=640
ret, imgs = cap.read()

window_name = "My OpenCV Window"

# Create a named window (optional, but good practice for more control)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # WINDOW_NORMAL allows resizing

# Display the image
cv2.imshow("window_name", imgs)

# It's important to call waitKey() at least once for the window to render
# and for its properties to be accessible.
cv2.waitKey(1) 

# Get the window properties
# You need to query specific properties like width and height.
# In newer OpenCV versions (3.4.1+), you can use cv2.getWindowImageRect()
# to get a tuple of (x, y, width, height) of the image within the window.
# This is often what people mean by "screen size in cv2.imshow".

x, y, width, height = cv2.getWindowImageRect(window_name)

print(f"Window name: {window_name}")
print(f"Image within window dimensions: Width={width}, Height={height}")

# You can also get other properties using cv2.getWindowProperty()
# For example, to check if the window is visible:
is_visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
print(f"Is window visible? {bool(is_visible)}")

# Wait for a key press and then close the window
cv2.waitKey(0) 
cv2.destroyAllWindows()