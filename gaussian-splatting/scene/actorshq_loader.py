import numpy as np
import csv
import collections
import struct
import cv2
import os, sys
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from typing import NamedTuple
from scipy.spatial.transform import Rotation

# Getting .obj files from Actorshq
# https://github.com/synthesiaresearch/humanrf/tree/main/actorshq/toolbox

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    
# BaseImage = collections.namedtuple(
#     "Image", ["id", "name", "w", "h", "rx", "ry", "rz", "tx", "ty", "tz", "fx", "fy", "px", "py"])

BaseImage = collections.namedtuple(
    "Image", ["id", "name", "w", "h", "r_angle", "t", "fx", "fy", "cx", "cy"])


def rvec2R(r_angle):
    return Rotation.from_rotvec(-r_angle).as_matrix()
    # return cv2.Rodrigues(np.array([rx, ry, rz]))[0]

def tvec2T(r_angle, t):
    return -Rotation.from_rotvec(-r_angle).as_matrix() @ t
    
class Image(BaseImage):
    def rvec2R(self):
        return rvec2R(self.r_angle)
    def tvec2T(self):
        return tvec2T(self.r_angle, self.t)
    
def read_calibration_csv(cameras_file):
    
    cams_params = {}    
    f = open(cameras_file, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    next(rdr)
    for line in rdr:
        image_id = int(line[0][3:])
        name = line[0]
        w = int(line[1]); h = int(line[2])
        rx = float(line[3]); ry = float(line[4]);  rz = float(line[5])
        tx = float(line[6]); ty = float(line[7]);  tz = float(line[8])
        fx = float(line[9]); fy = float(line[10])
        cx = float(line[11]) - 0.5; cy = float(line[12]) - 0.5 # cx and cy range is -0.5 to 0.5
        
        cams_params[image_id] = Image(
                    id=image_id, name=name, w=w, h=h, 
                    r_angle = np.array([rx, ry, rz]),
                    t = np.array([tx, ty, tz]),
                    fx=fx, fy=fy, cx=cx, cy=cy)
    f.close()  
    return cams_params

def read_first_points3D_obj(obj_path):
    
    # read .obj file and return xyzs & rbgs
    xyzs = []
    rgbs = None
    # num_points = 0
    
    with open(obj_path, "r") as fid:
        for line in fid:
            if line.startswith('v '):
                parts = line.split()
                xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                xyzs.append(xyz)
                
    xyzs = np.array(xyzs)
    rgbs = np.zeros((xyzs.shape[0], 3), dtype=np.float32)
    
    return xyzs, rgbs