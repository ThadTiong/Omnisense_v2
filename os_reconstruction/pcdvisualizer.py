import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

colormap = plt.cm.viridis

class PointCloudVisualizer():
    '''
    A class for visualizing point clouds.

    Parameters
    ----------
    target_width : int, optional -
        The target width of the visualization, by default 320
    target_height : int, optional - 
        The target height of the visualization, by default 240

    Attributes
    ----------
    points : numpy.ndarray -
        Array representing 3D points
    visualizer : open3d.visualization.Visualizer -
        Open3D visualizer instance
    pointcloud : open3d.geometry.PointCloud -
        Open3D point cloud geometry
        
    References
    ----------
    [1] GelSight Robotics GitHub repository. GelSight Inc.
        https://github.com/gelsightinc/gsrobotics
    '''
    def __init__(self, target_width = 320, target_height = 240, name = 'Open3D'):
        # Define X Y Z Axis
        x, y = np.arange(target_width), np.arange(target_height)
        X, Y = np.meshgrid(x, y)
        Z    = np.sin(X)
        
        # Define 3D self.points
        self.points       = np.zeros([target_width * target_height, 3])
        self.points[:, 0] = np.ndarray.flatten(X)
        self.points[:, 1] = np.ndarray.flatten(Y)
        self.points[:, 2] = np.ndarray.flatten(Z)
        
        self.init_visualizer(name)

    def init_visualizer(self, name):
        self.visualizer        = o3d.visualization.Visualizer()
        self.pointcloud        = o3d.geometry.PointCloud() 
        self.pointcloud.points = o3d.utility.Vector3dVector(self.points)
        self.pointcloud.colors = o3d.utility.Vector3dVector(self.points)
        
        self.visualizer.create_window(window_name=name, width=640, height=480, visible=True)
        self.visualizer.add_geometry(self.pointcloud)
        
    def update_visualizer(self, colormap):
        '''
        Update the visualizer with new colors.

        Parameters
        ----------
        colormap : numpy.ndarray -
            Array of RGB colors for each point.
        '''
        self.pointcloud.points = o3d.utility.Vector3dVector(self.points)
        self.pointcloud.colors = o3d.utility.Vector3dVector(colormap)
        self.visualizer.update_geometry(self.pointcloud)
        self.visualizer.poll_events()
        self.visualizer.update_renderer()
        
    def save_pointcloud(self, filename="pc_1.pcd"):
        '''
        Save the point cloud data.

        Parameters
        ----------
        filename : str, optional -
            Name of the file to save the point cloud, by default "pc_1.pcd"
        '''
        save_path = "os_reconstruction/saves/" + filename
        success = o3d.io.write_point_cloud(save_path, self.pointcloud)
        if success:
            print("Point Cloud Saved")
        else:
            print("Failed to save point cloud")

    def capture_image(self, crop_width=640, crop_height=480):
        '''
        Capture an image of the visualization.

        Parameters
        ----------
        crop_width : int, optional -
            Width of the captured image, by default 640
        crop_height : int, optional -
            Height of the captured image, by default 480

        Returns
        -------
        numpy.ndarray -
            Captured image.
        '''
        img = self.visualizer.capture_screen_float_buffer()
        img = (np.asarray(img) * 255).astype(np.uint8)
        img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (crop_width, crop_height))
        return img
    
    def depth2points(self, depthmap):
        '''
        Update the Z coordinate of the points.

        Parameters
        ----------
        depthmap : numpy.ndarray -
            Depth map.
        '''
        self.points[:, 2] = np.ndarray.flatten(depthmap)
        
    def map_greyscale(self, depthmap):
        '''
        Map depth values to grayscale colors.

        Parameters
        ----------
        depthmap : numpy.ndarray -
            Depth map.

        Returns
        -------
        numpy.ndarray -
            Array of grayscale colors.
        '''
        dx, dy = np.gradient(depthmap)
        dx, dy = dx * 0.5, dy * 0.5
        
        np_colors = np.clip(dx + 0.5, 0, 1)
        np_colors = np_colors.flatten()
        
        colors = np.tile(np_colors[:, np.newaxis], (1, 3))
        return colors
    
    def map_color(self, depthmap):
        '''
        Map depth values to colormap colors.

        Parameters
        ----------
        depthmap : numpy.ndarray -
            Depth map.

        Returns
        -------
        numpy.ndarray -
            Array of RGB colors.
        '''
        min_depth = np.min(depthmap)
        max_depth = np.max(depthmap)
        normalized_depth = (depthmap - min_depth) / (max_depth - min_depth)
        
        colors = np.zeros((normalized_depth.size, 3))
        colors[:, 0] = np.clip(2 * normalized_depth.flatten(), 0, 1)
        colors[:, 1] = np.clip(2 * normalized_depth.flatten() - 1, 0, 1)
        colors[:, 2] = np.clip(2 * normalized_depth.flatten() - 2, 0, 1)
        
        #colors = colormap(normalized_depth.flatten())[:, :3]
        return colors