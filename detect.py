import numpy as np
import cv2
import cv2.aruco as aruco
import math
"""
**************************************************************************
*                  E-Yantra Robotics Competition
*                  ================================
*  This software is intended to check version compatiability of open source software
*  Theme: Thirsty Crow
*  MODULE: Task1.1
*  Filename: detect.py
*  Version: 1.0.0  
*  Date: October 31, 2018
*  
*  Author: e-Yantra Project, Department of Computer Science 
*  and Engineering, Indian Institute of Technology Bombay.
*  
*  Software released under Creative Commons CC BY-NC-SA
*
*  For legal information refer to:
*        http://creativecommons.org/licenses/by-nc-sa/4.0/legalcode 
*     
*
*  This software is made available on an “AS IS WHERE IS BASIS”. 
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*  
*  e-Yantra - An MHRD project under National Mission on Education using 
*  ICT(NMEICT)
*
**************************************************************************
"""

####################### Define Utility Functions Here ##########################
"""
Function Name : getCameraMatrix()
Input: None
Output: camera_matrix, dist_coeff
Purpose: Loads the camera calibration file provided and returns the camera and
         distortion matrix saved in the calibration file.
"""
def getCameraMatrix():
        with np.load('System.npz') as X:
                camera_matrix, dist_coeff, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
        return camera_matrix, dist_coeff

"""
Function Name : sin()
Input: angle (in degrees)
Output: value of sine of angle specified
Purpose: Returns the sine of angle specified in degrees
"""
def sin(angle):
        return math.sin(math.radians(angle))

"""
Function Name : cos()
Input: angle (in degrees)
Output: value of cosine of angle specified
Purpose: Returns the cosine of angle specified in degrees
"""
def cos(angle):
        return math.cos(math.radians(angle))

def length(x1,y1,x2,y2) :
        return math.sqrt(((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2)))
################################################################################


"""
Function Name : detect_markers()
Input: img (numpy array), camera_matrix, dist_coeff
Output: aruco list in the form [(aruco_id_1, centre_1, rvec_1, tvec_1),(aruco_id_2,
        centre_2, rvec_2, tvec_2), ()....]
Purpose: This function takes the image in form of a numpy array, camera_matrix and
         distortion matrix as input and detects ArUco markers in the image. For each
         ArUco marker detected in image, paramters such as ID, centre coord, rvec
         and tvec are calculated and stored in a list in a prescribed format. The list
         is returned as output for the function
"""
def detect_markers(img, camera_matrix, dist_coeff):
        markerLength = 100
        aruco_list = []
        ######################## INSERT CODE HERE ########################
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GR)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
        n =len(ids)
        with np.load('System.npz') as X:
                camera_matrix, dist_matrix, _, _ =[ X[i] for i in
                                                    ('mtx', 'dist', 'rvecs', 'tvecs')]
        rvev, tvec, _objPoints= aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, dist_coeff)
        center = (corners[0][0][0]+corners[0][0][1]+corners[0][0][2]+corners[0][0][3])/4

        for i in range(n):
                aruco_list.append([ids[i],center[i],rvev[i],tvec[i]])
       

        
        ##################################################################
        return aruco_list

"""
Function Name : drawAxis()
Input: img (numpy array), aruco_list, aruco_id, camera_matrix, dist_coeff
Output: img (numpy array)
Purpose: This function takes the above specified outputs and draws 3 mutually
         perpendicular axes on the specified aruco marker in the image and
         returns the modified image.
"""
def drawAxis(img, aruco_list, aruco_id, camera_matrix, dist_coeff):
        for x in aruco_list:
                if aruco_id == x[0]:
                        rvec, tvec = x[2], x[3]
        markerLength = 100
        m = markerLength/2
        pts = np.float32([[-m,m,0],[m,m,0],[-m,-m,0],[-m,m,m]])
        pt_dict = {}
        imgpts, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeff)
        for i in range(len(pts)):
                 pt_dict[tuple(pts[i])] = tuple(imgpts[i].ravel())
        src = pt_dict[tuple(pts[0])];   dst1 = pt_dict[tuple(pts[1])];
        dst2 = pt_dict[tuple(pts[2])];  dst3 = pt_dict[tuple(pts[3])];
        
        img = cv2.line(img, src, dst1, (0,255,0), 4)
        img = cv2.line(img, src, dst2, (255,0,0), 4)
        img = cv2.line(img, src, dst3, (0,0,255), 4)
        return img

"""
Function Name : drawCube()
Input: img (numpy array), aruco_list, aruco_id, camera_matrix, dist_coeff
Output: img (numpy array)
Purpose: This function takes the above specified outputs and draws a cube
         on the specified aruco marker in the image and returns the modified
         image.
"""
def drawCube(img, ar_list, ar_id, camera_matrix, dist_coeff):
        for x in ar_list:
                if ar_id == x[0]:
                        rvec, tvec = x[2], x[3]
        markerLength = 100
        m = markerLength/2
        ######################## INSERT CODE HERE ########################
        pts = np.float32([[-m,m,0],[-m,-m,0],[m,-m,0],[m,m,0],[-m,m,m],[-m,-m,m],[m,-m,m],[m,m,m]])
        pt_dict = {}
        imgpts, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeff)
        for i in range(len(pts)):
                 pt_dict[tuple(pts[i])] = tuple(imgpts[i].ravel())
        src = pt_dict[tuple(pts[0])];   dst1 = pt_dict[tuple(pts[1])];
        dst2 = pt_dict[tuple(pts[2])];  dst3 = pt_dict[tuple(pts[3])]; dst4 = pt_dict[tuple(pts[4])]; dst5 = pt_dict[tuple(pts[5])];
        dst6 = pt_dict[tuple(pts[6])]; dst7 = pt_dict[tuple(pts[7])];
        
        img = cv2.line(img, src, dst1, (0,0,255), 4)
        img = cv2.line(img, dst1, dst2, (0,0,255), 4)
        img = cv2.line(img, dst2, dst3, (0,0,255), 4)
        img = cv2.line(img, dst3, dst7, (0,0,255), 4)
        img = cv2.line(img, dst7, dst4, (0,0,255), 4)
        img = cv2.line(img, dst4, dst5, (0,0,255), 4)
        img = cv2.line(img, dst5, dst6, (0,0,255), 4)
        img = cv2.line(img, dst5, dst1, (0,0,255), 4)
        img = cv2.line(img, dst7, dst6, (0,0,255), 4)
        img = cv2.line(img, src, dst4, (0,0,255), 4) 
        img = cv2.line(img, src, dst3, (0,0,255), 4)
        img = cv2.line(img, dst6, dst2, (0,0,255), 4)
        return img
        
        ##################################################################
        return img

"""
Function Name : drawCylinder()
Input: img (numpy array), aruco_list, aruco_id, camera_matrix, dist_coeff
Output: img (numpy array)
Purpose: This function takes the above specified outputs and draws a cylinder
         on the specified aruco marker in the image and returns the modified
         image.
"""
def drawCylinder(img, ar_list, ar_id, camera_matrix, dist_coeff):
        for x in ar_list:
                if ar_id == x[0]:
                        rvec, tvec = x[2], x[3]
        markerLength = 100
        radius = markerLength/2; height = markerLength*1.5
        ######################## INSERT CODE HERE ########################
        pts = np.float32([[0,0,0],[0,radius,0],[-radius*sin(30),radius*cos(30),0],[-radius*sin(60),radius*cos(60),0],[-radius,0,0],[-radius*cos(30),-radius*sin(30),0],
                          [-radius*cos(60),-radius*sin(60),0],[0,-radius,0],[radius*sin(30),-radius*cos(30),0],[radius*sin(60),-radius*cos(60),0],[radius,0,0],
                          [radius*cos(30),radius*sin(30),0],[radius*cos(60),radius*sin(60),0],
                          [0,0,height],[0,radius,height],[-radius*sin(30),radius*cos(30),height],[-radius*sin(60),radius*cos(60),height],[-radius,0,height],
                          [-radius*cos(30),-radius*sin(30),height],[-radius*cos(60),-radius*sin(60),height],[0,-radius,height],[radius*sin(30),-radius*cos(30),height],
                          [radius*sin(60),-radius*cos(60),height],[radius,0,height],[radius*cos(30),radius*sin(30),height],[radius*cos(60),radius*sin(60),height]])
        pt_dict = {}
        imgpts, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeff)
        for i in range(len(pts)):
                 pt_dict[tuple(pts[i])] = tuple(imgpts[i].ravel())
        src = pt_dict[tuple(pts[0])];   dst1 = pt_dict[tuple(pts[1])];  dst2 = pt_dict[tuple(pts[2])];  dst3 = pt_dict[tuple(pts[3])];
        dst4 = pt_dict[tuple(pts[4])];  dst5 = pt_dict[tuple(pts[5])];  dst6 = pt_dict[tuple(pts[6])];  dst7 = pt_dict[tuple(pts[7])];
        dst8 = pt_dict[tuple(pts[8])];  dst9 = pt_dict[tuple(pts[9])];  dst10 = pt_dict[tuple(pts[10])];  dst11 = pt_dict[tuple(pts[11])];
        dst12 = pt_dict[tuple(pts[12])];  dst13 = pt_dict[tuple(pts[13])];  dst14 = pt_dict[tuple(pts[14])];  dst15 = pt_dict[tuple(pts[15])];
        dst16 = pt_dict[tuple(pts[16])];  dst17 = pt_dict[tuple(pts[17])];  dst18 = pt_dict[tuple(pts[18])];  dst19 = pt_dict[tuple(pts[19])];
        dst20 = pt_dict[tuple(pts[20])];  dst21 = pt_dict[tuple(pts[21])];  dst22 = pt_dict[tuple(pts[22])];  dst23 = pt_dict[tuple(pts[23])];
        dst24 = pt_dict[tuple(pts[24])];  dst25 = pt_dict[tuple(pts[25])];

        img = cv2.line(img, src, dst1, (0,0,255), 1)
        img = cv2.line(img, src, dst2, (0,0,255), 1)
        img = cv2.line(img, src, dst3, (0,0,255), 1)
        img = cv2.line(img, src, dst4, (0,0,255), 1)
        img = cv2.line(img, src, dst5, (0,0,255), 1)
        img = cv2.line(img, src, dst6, (0,0,255), 1)
        img = cv2.line(img, src, dst7, (0,0,255), 1)
        img = cv2.line(img, src, dst8, (0,0,255), 1)
        img = cv2.line(img, src, dst9, (0,0,255), 1)
        img = cv2.line(img, src, dst10, (0,0,255), 1)
        img = cv2.line(img, src, dst11, (0,0,255), 1)
        img = cv2.line(img, src, dst12, (0,0,255), 1)
        img = cv2.line(img, src, dst13, (0,0,255), 1)
        img = cv2.line(img, dst13, dst14, (0,0,255), 1)
        img = cv2.line(img, dst13, dst15, (0,0,255), 1)
        img = cv2.line(img, dst13, dst16, (0,0,255), 1)
        img = cv2.line(img, dst13, dst17, (0,0,255), 1)
        img = cv2.line(img, dst13, dst18, (0,0,255), 1)
        img = cv2.line(img, dst13, dst19, (0,0,255), 1)
        img = cv2.line(img, dst13, dst20, (0,0,255), 1)
        img = cv2.line(img, dst13, dst21, (0,0,255), 1)
        img = cv2.line(img, dst13, dst22, (0,0,255), 1)
        img = cv2.line(img, dst13, dst23, (0,0,255), 1)
        img = cv2.line(img, dst13, dst24, (0,0,255), 1)
        img = cv2.line(img, dst13, dst25, (0,0,255), 1)
        img= cv2.line(img, dst1, dst14, (0,0,255),1)
        img= cv2.line(img, dst2, dst15, (0,0,255),1)
        img= cv2.line(img, dst3, dst16, (0,0,255),1)
        img= cv2.line(img, dst4, dst17, (0,0,255),1)
        img= cv2.line(img, dst5, dst18, (0,0,255),1)
        img= cv2.line(img, dst6, dst19, (0,0,255),1)
        img= cv2.line(img, dst7, dst20, (0,0,255),1)
        img= cv2.line(img, dst8, dst21, (0,0,255),1)
        img= cv2.line(img, dst9, dst22, (0,0,255),1)
        img= cv2.line(img, dst10, dst23, (0,0,255),1)
        img= cv2.line(img, dst11, dst24, (0,0,255),1)
        img= cv2.line(img, dst12, dst25, (0,0,255),1)
       
        x=length(dst1[0],dst1[1],dst7[0],dst7[1])
        y=length(dst4[0],dst4[1],dst10[0],dst10[1])
        if x<y :
                minor=x/2
                major=y/2
        else :
                major=x/2
                minor=y/2
        Xsrc = int(src[0])
        Ysrc = int(src[1])
      
        major = int(major)
        minor = int(minor)
       
        img = cv2.ellipse(img, (Xsrc,Ysrc),(major,minor),0,0,360,(0,0,255),1)

        _x=length(dst14[0],dst14[1],dst20[0],dst20[1])
        _y=length(dst17[0],dst17[1],dst23[0],dst23[1])
        if _x<_y :
                _minor=_x/2
                _major=_y/2
        else :
                _major=_x/2
                _minor=_y/2

        Xdst13 = int(dst13[0])
        Ydst13 = int(dst13[1])
        _major = int(_major)
        _minor = int(_minor)
        img = cv2.ellipse(img, (Xdst13,Ydst13), (_major,_minor),0,0,360,(0,0,255),1)
        #img = cv2.circle(img, src, int(radius), (0,0,255), 1)

        #img = cv2.circle(img, dst13, int(radius), (0,0,255), 1)
                
        ##################################################################
        return img

"""
MAIN CODE
This main code reads images from the test cases folder and converts them into
numpy array format using cv2.imread. Then it draws axis, cubes or cylinders on
the ArUco markers detected in the images.
"""


if __name__=="__main__":
        cam, dist = getCameraMatrix()
        img = cv2.imread("..\\TestCases\\image_5.jpg")
        aruco_list = detect_markers(img, cam, dist)
        for i in aruco_list:
                #img = drawAxis(img, aruco_list, i[0], cam, dist)
                #img = drawCube(img, aruco_list, i[0], cam, dist)
                img = drawCylinder(img, aruco_list, i[0], cam, dist)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
