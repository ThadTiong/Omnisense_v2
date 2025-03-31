import cv2
import os
import re

def find_camera_on_linux(device_id):
    '''
    Find camera on Linux system by device ID.

    Parameters
    ----------
    device_id : int -
        Device ID of the camera.

    Returns
    -------
    str or None
        Name of the camera if found, None otherwise.
    '''
    for file in os.listdir("/sys/class/video4linux"):
        real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
        
        with open(real_file, "rt") as name_file:
            name = name_file.read().rstrip()
            
        if int(re.search("\d+$", file).group(0)) == device_id:
            return name
        
    return None

def find_camera_on_windows(device_id):
    '''
    Find camera on Windows system by device ID.

    Parameters
    ----------
    device_id : int -
        Device ID of the camera.

    Returns
    -------
    str or None
        Name of the camera if found, None otherwise.
    '''
    from pygrabber.dshow_graph import FilterGraph
    graph = FilterGraph()

    all_cams = graph.get_input_devices()
    if device_id < len(all_cams):
        return all_cams[device_id]
    else:
        return None

def list_cameras():
    '''
    List available cameras on the system.

    Returns
    -------
    list -
        List of dictionaries containing camera names and IDs.
    '''
    if os.name == 'nt':
        return list_cameras_on_windows()
    else:
        return list_cameras_on_linux()

def list_cameras_on_linux():
    '''
    List available cameras on Linux system.

    Returns
    -------
    list -
        List of dictionaries containing camera names and IDs.
    '''
    cameras = []
    for file in os.listdir("/sys/class/video4linux"):
        real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
        
        with open(real_file, "rt") as name_file:
            name = name_file.read().rstrip()
            
        cameras.append({"name": name, "id": int(re.search("\d+$", file).group(0))})
    
    return cameras

def list_cameras_on_windows():
    '''
    List available cameras on Windows system.

    Returns
    -------
    list -
        List of dictionaries containing camera names and IDs.
    '''
    from pygrabber.dshow_graph import FilterGraph
    graph = FilterGraph()

    all_cams = graph.get_input_devices()
    cameras = [{"name": cam, "id": index} for index, cam in enumerate(all_cams)]

    return cameras

class Camera:
    '''
    A class to handle camera operations.

    Attributes
    ----------
    cam : cv2.VideoCapture -
        OpenCV video capture object.
    data : numpy.ndarray -
        Image data from the camera.
    device_name : str -
        Name of the camera device.
    device_id : int -
        ID of the camera device.
    width : int -
        Width of the captured frame.
    height : int -
        Height of the captured frame.
    is_running : bool -
        Flag indicating if video capture is running.
    '''
    def __init__(self, cam = None, width = 640, height = 480):
        self.cam = cam
        self.data = None
        self.device_name = ''
        self.device_id = 0
        self.width = width
        self.height = height
        self.is_running = True

    def connect(self, device_id=0, backend=cv2.CAP_ANY):
        '''
        Connect to a camera device. 

        Parameters
        ----------
        device_id : int, optional -
            ID of the camera device, by default 0
        backend : int, optional -
            OpenCV backend for video capture, by default cv2.CAP_ANY

        Returns
        -------
        cv2.VideoCapture or None -
            Video capture object if connection is successful, None otherwise.
        '''
        self.cam = cv2.VideoCapture(device_id, backend)
        if self.cam.isOpened():
            self.device_id = device_id
            self.device_name = find_camera_on_windows(device_id)
            return self.cam
        return None

    def switch_camera(self, device_id):
        '''
        Switch to a different camera device.

        Parameters
        ----------
        device_id : int -
            ID of the new camera device.

        Returns
        -------
        bool -
            True if switching is successful, False otherwise.
        '''
        if self.cam is not None:
            self.cam.release()
            
        backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]
        for backend in backends:
            if self.connect(device_id, backend):
                print('Success: Camera connected:', self.device_id, ' - ', self.cam.getBackendName())
                return True
        
        print('Error: Unable to open video source:', self.device_id)
        return False
    
    def get_frame(self):
        '''
        Capture a frame from the camera.

        Returns
        -------
        numpy.ndarray or None -
            Captured frame data if successful, None otherwise.
        '''
        result, frame = self.cam.read()
        
        if result:
            frame = cv2.resize(frame, (self.width, self.height))
            self.data = frame
        else:
            print('ERROR! Reading image from camera')
            
        return self.data

    def save_frame(self, file_name):
        '''
        Save the captured frame to a file.

        Parameters
        ----------
        file_name : str -
            Name of the file to save the frame.
        '''
        cv2.imwrite(file_name, self.data)