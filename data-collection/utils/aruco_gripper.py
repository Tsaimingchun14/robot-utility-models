import cv2
import numpy as np

def detect_gripper_width_aruco(image, marker_id_left=200, marker_id_right=201, dictionary_type=cv2.aruco.DICT_6X6_250):
    """
    Detects ArUco markers on gripper fingers and returns the Euclidean distance in pixels.
    Returns None if markers are not detected.
    """
    # Initialize ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect
    corners, ids, _ = detector.detectMarkers(image)
    
    points = {}
    if ids is not None:
        ids = ids.flatten()
        for i, m_id in enumerate(ids):
            if m_id in [marker_id_left, marker_id_right]:
                # Use the center of the marker for distance
                center = np.mean(corners[i][0], axis=0)
                points[m_id] = center

    if marker_id_left in points and marker_id_right in points:
        p1, p2 = points[marker_id_left], points[marker_id_right]
        dist = np.linalg.norm(p1 - p2)
        return dist
    
    return None
