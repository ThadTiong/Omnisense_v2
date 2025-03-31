import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Inverse Light Positions
A = np.linalg.pinv(np.array([
    [-0.397, 0.903, 0],
    [0, 0.847, -0.51],
    [0.397, 0.903, 0],
]))

distance_to_object = 1

def processing_3drender(frame, ax, fig):
    # Extract individual color channels
    blue_channel, green_channel, red_channel = cv2.split(frame)
    channels = [blue_channel, green_channel, red_channel]

    height, width = channels[0].shape
    surface = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            pixel_intensities = np.array([channel[y, x] for channel in channels])
            I = pixel_intensities.reshape(-1, 1)

            normal = A @ I
            normal_norm = np.linalg.norm(normal)

            if normal_norm < 1e-8:
                surface_height = 0
            else:
                surface_height = (normal[2] * distance_to_object) / normal_norm

            surface[y, x] = surface_height

    # Update the plot
    ax.clear()
    x, y = np.meshgrid(range(surface.shape[1]), range(surface.shape[0]))
    ax.plot_surface(x, y, surface, cmap='viridis')
    plt.pause(0.1)

    # Render the plot as a 2D image
    canvas = FigureCanvas(fig)
    canvas.draw()
    frame = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return frame

if __name__ == "__main__":
    # Test code
    cap = cv2.VideoCapture(0)

    # Set the desired width and height for resizing
    target_width = 80
    target_height = 60

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        frame = cv2.resize(frame, (target_width, target_height))

        processed_frame = processing_3drender(frame, ax, fig)

        # Display the 3D rendering as a 2D image using OpenCV
        cv2.imshow("3D Rendering", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
