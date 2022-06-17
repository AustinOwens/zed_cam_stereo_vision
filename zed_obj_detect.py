'''
Created on Jun 8, 2022

@author: Austin Owens

Need to make sure opencv-contrib-python libs are installed:
pip install opencv-contrib-python
Note: Opencv comes with opencv-contrib-python

Ran with Python 3.8
'''

import numpy as np
from numpy import array
import cv2
import math as m


# Camera Undistortion Params
r_cam_mtx = array([[375.40059218,   0.        , 327.76067959],
                   [  0.        , 373.50422012, 184.03869394],
                   [  0.        ,   0.        ,   1.        ]])

r_dist_coeffs = array([[-0.13168878, -0.09869939, -0.00558188,  0.00125711,  0.11456413]])

l_cam_mtx = array([[375.40059218,   0.        , 327.76067959],
                   [  0.        , 373.50422012, 184.03869394],
                   [  0.        ,   0.        ,   1.        ]])

l_dist_coeffs = array([[-0.13168878, -0.09869939, -0.00558188,  0.00125711,  0.11456413]])

undist_roi = 9, 13, 652, 346



# Raw Img
cv2.namedWindow('roi_selector', cv2.WINDOW_NORMAL)

# Disparity 3D Img
cv2.namedWindow('disparity_3d', cv2.WINDOW_NORMAL)

# Writing 3D point cloud
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    verts = verts[np.isfinite(verts).all(axis=1)] # Remove inf vertices
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
        
# ROI Trackbars
def nothing(x):
    pass

cv2.namedWindow('1_roi_selector_trackbars', cv2.WINDOW_NORMAL)
cv2.resizeWindow('1_roi_selector_trackbars', 700, 100)

cv2.createTrackbar('scale_percent', '1_roi_selector_trackbars', 100, 100, nothing)
cv2.createTrackbar('min_hue',        '1_roi_selector_trackbars', 0,   180, nothing)
cv2.createTrackbar('max_hue',        '1_roi_selector_trackbars', 180, 180, nothing)
cv2.createTrackbar('min_sat',        '1_roi_selector_trackbars', 0,   255, nothing)
cv2.createTrackbar('max_sat',        '1_roi_selector_trackbars', 255, 255, nothing)
cv2.createTrackbar('min_val',        '1_roi_selector_trackbars', 0,   255, nothing)
cv2.createTrackbar('max_val',        '1_roi_selector_trackbars', 255, 255, nothing)
cv2.createTrackbar('erosion',       '1_roi_selector_trackbars', 0, 10, nothing)
cv2.createTrackbar('dilation',      '1_roi_selector_trackbars', 0, 10, nothing)

# Disparity Map Trackbars
cv2.namedWindow('2_disp_trackbars', cv2.WINDOW_NORMAL)
cv2.resizeWindow('2_disp_trackbars', 700, 0)

cv2.createTrackbar('min_disparity',      '2_disp_trackbars', 10,      25, nothing)
cv2.createTrackbar('num_disparities',    '2_disp_trackbars', 138,    320, nothing)
cv2.createTrackbar('block_size',         '2_disp_trackbars', 2,       50, nothing)
cv2.createTrackbar('p1',                '2_disp_trackbars', 442,   1000, nothing)
cv2.createTrackbar('p2',                '2_disp_trackbars', 2519,   5000, nothing)
cv2.createTrackbar('disp12_max_diff',     '2_disp_trackbars', 0,       25, nothing)
cv2.createTrackbar('pre_filter_cap',      '2_disp_trackbars', 0,       62, nothing)
cv2.createTrackbar('uniqueness_ratio',   '2_disp_trackbars', 0,      100, nothing)
cv2.createTrackbar('speckle_window_size', '2_disp_trackbars', 0,      200, nothing)
cv2.createTrackbar('speckle_range',      '2_disp_trackbars', 0,      100, nothing)
cv2.createTrackbar('lambda',            '2_disp_trackbars', 8000, 10000, nothing)
cv2.createTrackbar('sigma',             '2_disp_trackbars', 20,     100, nothing)
cv2.createTrackbar('erosion',           '2_disp_trackbars', 1,       10, nothing)
cv2.createTrackbar('dilation',          '2_disp_trackbars', 1,       10, nothing)
cv2.createTrackbar('focal',             '2_disp_trackbars', 5,       10, nothing)
cv2.createTrackbar('radius_neighbor',   '2_disp_trackbars', 1,      100, nothing)
cv2.createTrackbar('use_neighbor',      '2_disp_trackbars', 0,        1, nothing)

if __name__ == '__main__':
    # Initialize Camera
    cap = cv2.VideoCapture(1)
    
    # Initialize StereoSGBM Object
    stereo = cv2.StereoSGBM_create()
    
    # Initialize Camshift Track Windows
    l_track_window = (0, 0, 1, 1) # x, y, width, height
    r_track_window = (0, 0, 1, 1) # x, y, width, height
    
    # Create Mouse Click Events
    draw_hsv_roi = False
    show_hsv_roi_cnt = 0
    x1, y1, x2, y2 = 0, 0, 0, 0
    def roi_click_event(event, x, y, flags, params):
        global draw_hsv_roi
        global show_hsv_roi_cnt
        global x1, y1, x2, y2
        
        # Checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1 = x, y
            draw_hsv_roi = True
        
        # Checking if previously clicked left mouse and is held down
        if draw_hsv_roi:
            # Counter for how long hsv roi box should stay on screen after letting go of mouse
            show_hsv_roi_cnt = 15 
            
            x2, y2 = x, y
            
        # Checking if let go of left mouse
        if event == cv2.EVENT_LBUTTONUP:
            
            # Boundary check
            if y1 > y2:
                tmp = y2
                y2 = y1
                y1 = tmp
                
            if x1 > x2:
                tmp = x2
                x2 = x1
                x1 = tmp
                
            if y1 == y2:
                y2 += 1
                
            if x1 == x2:
                x2 += 1
                
            
            roi_bgr = img[y1:y2, x1:x2]
            
            roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            
            min_hsv = np.amin((np.amin(roi_hsv, 1)), 0)
            max_hsv = np.amax((np.amax(roi_hsv, 1)), 0)
            
            tolerance = 3
            cv2.setTrackbarPos('min_hue', '1_roi_selector_trackbars', min_hsv[0] - tolerance)
            cv2.setTrackbarPos('min_sat', '1_roi_selector_trackbars', min_hsv[1] - tolerance) 
            cv2.setTrackbarPos('min_val', '1_roi_selector_trackbars', min_hsv[2] - tolerance) 
            cv2.setTrackbarPos('max_hue', '1_roi_selector_trackbars', max_hsv[0] + tolerance) 
            cv2.setTrackbarPos('max_sat', '1_roi_selector_trackbars', max_hsv[1] + tolerance) 
            cv2.setTrackbarPos('max_val', '1_roi_selector_trackbars', max_hsv[2] + tolerance)
            
            draw_hsv_roi = False
            
    double_click_flag = False
    def ply_click_event(event, x, y, flags, params):
        global double_click_flag
        
        if event == cv2.EVENT_LBUTTONDBLCLK:
            double_click_flag = True
            
    cv2.setMouseCallback('roi_selector', roi_click_event)
    cv2.setMouseCallback('disparity_3d', ply_click_event)
    
    
    def neighbors(mat, radius, row_number, column_number):
        return [[mat[i][j] if  i >= 0 and i < len(mat) and j >= 0 and j < len(mat[0]) else 0
                 for j in range(column_number-radius, column_number+radius)]
                 for i in range(row_number-radius, row_number+radius)]
        
    def reject_outliers(data, m=2.0):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.0
        
        return data[s<m]
    
    while True:
        ret, img = cap.read()
        
        if ret:  
            cv2.imshow('raw_img', img)
            
            ### STEP 1: UNDISTORTION / DOWNSCALING ###
            height = img.shape[0]
            width  = img.shape[1]
            
            l_img = img[0:height, 0:int(width/2)]
            r_img = img[0:height, int(width/2):width]
            
            ## Undistort
            l_img_undistort = cv2.undistort(l_img, l_cam_mtx, l_dist_coeffs)
            r_img_undistort = cv2.undistort(r_img, r_cam_mtx, r_dist_coeffs)
            
            # Crop
            x_undist, y_undist, w_undist, h_undist = undist_roi
            l_img_undistort_crop = l_img_undistort[y_undist:y_undist+h_undist, x_undist:x_undist+w_undist]
            r_img_undistort_crop = r_img_undistort[y_undist:y_undist+h_undist, x_undist:x_undist+w_undist]
            
            # Downsizing to increase processing speed
            scale_percent = cv2.getTrackbarPos('scale_percent', '1_roi_selector_trackbars')
            width = int(l_img_undistort_crop.shape[1] * scale_percent / 100)
            height = int(l_img_undistort_crop.shape[0] * scale_percent / 100)
            dim = (width, height)
            l_img_undistort_crop = cv2.resize(l_img_undistort_crop, dim, cv2.INTER_LINEAR_EXACT)
            r_img_undistort_crop = cv2.resize(r_img_undistort_crop, dim, cv2.INTER_LINEAR_EXACT) 
            
            # Recombine Img
            img = cv2.hconcat([l_img_undistort_crop, r_img_undistort_crop]) 
            
            
            ### STEP 2: HSV FILTERING ###
            min_hue = cv2.getTrackbarPos('min_hue', '1_roi_selector_trackbars')
            max_hue = cv2.getTrackbarPos('max_hue', '1_roi_selector_trackbars')
            min_sat = cv2.getTrackbarPos('min_sat', '1_roi_selector_trackbars')
            max_sat = cv2.getTrackbarPos('max_sat', '1_roi_selector_trackbars')
            min_val = cv2.getTrackbarPos('min_val', '1_roi_selector_trackbars')
            max_val = cv2.getTrackbarPos('max_val', '1_roi_selector_trackbars')
            
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_bound = (min_hue, min_sat, min_val)
            upper_bound = (max_hue, max_sat, max_val)
            mask_img = cv2.inRange(hsv_img, lower_bound , upper_bound) # Binary image (0 or 255)
            
            ### STEP 3: EROSION & DILATION ###
            erosion  = cv2.getTrackbarPos('erosion', '1_roi_selector_trackbars')
            dilation = cv2.getTrackbarPos('dilation', '1_roi_selector_trackbars')
            
            kernel = np.ones((5,5), np.uint8)
            mask_img = cv2.erode(mask_img, kernel, iterations=erosion)
            mask_img = cv2.dilate(mask_img, kernel, iterations=dilation)
            
            # Adding 3rd axis to make RGB compatible and masking orig img with hsv
            hsv_mask_img = np.bitwise_and(img, mask_img[:, :, np.newaxis]) 
            
            
            ### STEP 4: SPLIT ZED CAMERA IMAGE ###
            height = mask_img.shape[0]
            width  = mask_img.shape[1]
            
            l_mask_img = mask_img[0:height, 0:int(width/2)]
            r_mask_img = mask_img[0:height, int(width/2):width]
            
            ### STEP 5: CAMSHIFT LEFT IMG ###
            l_term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            l_track_box, l_track_window = cv2.CamShift(l_mask_img, l_track_window, l_term_crit)
            
            # Makes the ellipse on the image
            cv2.ellipse(hsv_mask_img, l_track_box, (0, 0, 255), 2) 
            cv2.putText(hsv_mask_img, "({:.1f}, {:.1f})".format(l_track_box[0][0], l_track_box[0][1]), (int(l_track_box[0][0]), int(l_track_box[0][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            cv2.putText(hsv_mask_img, "({:.1f}, {:.1f})".format(l_track_box[1][1], l_track_box[1][0]), (int(l_track_box[0][0]), int(l_track_box[0][1]+15)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            cv2.putText(hsv_mask_img, "({:.1f})".format(l_track_box[2]), (int(l_track_box[0][0]), int(l_track_box[0][1]+30)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))  
            
            ### STEP 6: CAMSHIFT RIGHT IMG ###
            r_term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            r_track_box, r_track_window = cv2.CamShift(r_mask_img, r_track_window, r_term_crit)
            
            # Offset track window
            r_track_box_offset = ((r_track_box[0][0]+int(width/2), r_track_box[0][1]), (r_track_box[1][0], r_track_box[1][1]), r_track_box[2])
            
            # Makes the ellipse on the image
            cv2.ellipse(hsv_mask_img, r_track_box_offset, (0, 0, 255), 2) 
            cv2.putText(hsv_mask_img, "({:.1f}, {:.1f})".format(r_track_box[0][0], r_track_box[0][1]), (int(r_track_box_offset[0][0]), int(r_track_box_offset[0][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            cv2.putText(hsv_mask_img, "({:.1f}, {:.1f})".format(r_track_box[1][1], r_track_box[1][0]), (int(r_track_box_offset[0][0]), int(r_track_box_offset[0][1]+15)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            cv2.putText(hsv_mask_img, "({:.1f})".format(r_track_box[2]), (int(r_track_box_offset[0][0]), int(r_track_box_offset[0][1]+30)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))  
            
            ### STEP 7: DRAW ROI HSV FILTER BOX ###
            if show_hsv_roi_cnt >= 0:
                cv2.rectangle(hsv_mask_img, (x1, y1), (x2, y2), (0, 0, 255), int(show_hsv_roi_cnt/4))
                
                # If not draw_hsv_roi, then slowly fade the rectangle out of existance
                if not draw_hsv_roi:
                    show_hsv_roi_cnt -= 5
                  
            cv2.imshow('roi_selector', hsv_mask_img)
            
            
            ### STEP 8: CREATE DISPARITY MAP ###
            height = img.shape[0]
            width  = img.shape[1]
            l_img = img[0:height, 0:int(width/2)]
            r_img = img[0:height, int(width/2):width]
            
            
            # Updating the parameters based on the trackbar positions
            min_disparity       = cv2.getTrackbarPos('min_disparity',       '2_disp_trackbars')
            num_disparities     = cv2.getTrackbarPos('num_disparities',     '2_disp_trackbars')#*16
            block_size          = cv2.getTrackbarPos('block_size',          '2_disp_trackbars')#*2 + 5
            p1                  = cv2.getTrackbarPos('p1',                  '2_disp_trackbars')
            p2                  = cv2.getTrackbarPos('p2',                  '2_disp_trackbars')
            disp12_max_diff     = cv2.getTrackbarPos('disp12_max_diff',     '2_disp_trackbars')
            pre_filter_cap      = cv2.getTrackbarPos('pre_filter_cap',      '2_disp_trackbars')
            uniqueness_ratio    = cv2.getTrackbarPos('uniqueness_ratio',    '2_disp_trackbars')
            speckle_window_size = cv2.getTrackbarPos('speckle_window_size', '2_disp_trackbars')#*2
            speckle_range       = cv2.getTrackbarPos('speckle_range',       '2_disp_trackbars')
            lmbda               = cv2.getTrackbarPos('lambda',              '2_disp_trackbars')
            sigma               = cv2.getTrackbarPos('sigma',               '2_disp_trackbars')/10.0
            erosion             = cv2.getTrackbarPos('erosion',             '2_disp_trackbars')
            dilation            = cv2.getTrackbarPos('dilation',            '2_disp_trackbars')
            focal               = cv2.getTrackbarPos('focal',               '2_disp_trackbars')
            use_neighbor        = cv2.getTrackbarPos('use_neighbor',        '2_disp_trackbars')
            radius_neighbor     = cv2.getTrackbarPos('radius_neighbor',     '2_disp_trackbars')
            
            
            # Setting the updated parameters before computing disparity map
            stereo.setMinDisparity(min_disparity)
            stereo.setNumDisparities(num_disparities)
            stereo.setBlockSize(block_size)
            stereo.setP1(p1)
            stereo.setP2(p2)
            stereo.setDisp12MaxDiff(disp12_max_diff)
            stereo.setPreFilterCap(pre_filter_cap)
            stereo.setUniquenessRatio(uniqueness_ratio)
            stereo.setSpeckleWindowSize(speckle_window_size)
            stereo.setSpeckleRange(speckle_range)
            #stereo.setMode(cv2.STEREO_SGBM_MODE_HH)
            
            
            # Calculating disparity & WLS Filtering
            # Returns a 16bit signed single channel image (CV_16S) containing a disparity map scaled 
            # by 16. It is commonly converted to CV_32F and scaled down 16 times.
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
            right_stereo = cv2.ximgproc.createRightMatcher(stereo)
            
            left_for_matcher = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
            right_for_matcher = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)
            
            left_disp = stereo.compute(left_for_matcher, right_for_matcher)
            right_disp = right_stereo.compute(right_for_matcher, left_for_matcher)
            
            wls_filter.setLambda(lmbda)
            wls_filter.setSigmaColor(sigma)
            disp = wls_filter.filter(left_disp, l_img, disparity_map_right=right_disp).astype(np.float32) / 16.0
            
            # Debug Pre-Filtered Disparity
            #pre_filtered_disp = left_disp.astype(np.float32) / 16.0
            #pre_filtered_disp = (pre_filtered_disp - min_disparity)/num_disparities 
            #cv2.imshow('pre_filtered_disp', pre_filtered_disp)

            
            ### STEP 9: EROSION & DILATION ###
    
            # Perform erosion and dilation
            kernel = np.ones((5,5), np.uint8)
            disp = cv2.erode(disp, kernel, iterations=erosion)
            disp = cv2.dilate(disp, kernel, iterations=dilation)
            
            ### STEP 10: CALCULATE 3D PROJECTION ###
            
            # Crop out blank portion of disparity map caused by num_disparities and min_disparity
            disp = disp[:, (num_disparities+min_disparity):disp.shape[1]]
            h, w = disp.shape
            
            f = (focal/10.0)*w  # Focal length
            
            Q = np.float32([[1, 0, 0, -0.5*w],
                            [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis so that y-axis looks up
                            [0, 0, 0,     -f], 
                            [0, 0, 1,      0]])
            
            points = cv2.reprojectImageTo3D(disp, Q)
                
            
            ### STEP 11: CALCULATE ROI FRAOM CAMSHIFT ###
            track_box_x      = l_track_box[0][0]
            track_box_y      = l_track_box[0][1]
            track_box_width  = l_track_box[1][1]
            track_box_height = l_track_box[1][0]
            orientation      = l_track_box[2]
            
            # Get true width and height of track_box
            rot_angle = m.radians(orientation-90) # Subtract 90 deg since 90 degrees is baseline for camshift width
            roi_w = abs(track_box_width*m.cos(rot_angle)) + abs(track_box_height*m.sin(rot_angle))
            roi_h = abs(track_box_width*m.sin(rot_angle)) + abs(track_box_height*m.cos(rot_angle))
            
            # X cannot got into the num_disparities+min_disparity region (the cropped regions thats caused by 
            # increasing num_disparities & min_disparity)
            #if track_box_x <= num_disparities+min_disparity:
            track_box_x -= num_disparities+min_disparity
                
            # Width cannot stretch passed the num_disparities+min_disparity region
            #if track_box_x-track_box_width/2 <= num_disparities+min_disparity:
            #    track_box_width = track_box_width- #(track_box_x-(num_disparities+min_disparity))*2
                
            # Use nearest neighbor to calculate ROI
            if use_neighbor:
                
                # Get ROI based on nearest neighbor
                roi_x1 = int(round(track_box_x-radius_neighbor, 0))
                roi_x2 = int(round(track_box_x+radius_neighbor, 0))
                roi_y1 = int(round(track_box_y-radius_neighbor, 0))
                roi_y2 = int(round(track_box_y+radius_neighbor, 0))
                
                # Check ROI boundaries
                if roi_x1 <= 0:
                    roi_x1 = 0
                    
                if roi_y1 <= 0:
                    roi_y1 = 0
                    
                if roi_y2 <= roi_y1:
                    roi_y2 = roi_y1+1
                    
                if roi_x2 <= roi_x1:
                    roi_x2 = roi_x1+1
                    
                # Get neighbor X, Y, and Z points within center of camshift
                roi_points = np.zeros((radius_neighbor, radius_neighbor, 3))
                for i in range(1+2*radius_neighbor):
                    for j in range(1+2*radius_neighbor):
                        roi_points[i][j] = points[roi_y1+i][roi_x1+j]
            
            else:
                # Find ROI based on bounded ellipse
                
                
                roi_x1 = track_box_x-(roi_w/2.0)
                roi_y1 = track_box_y-(roi_h/2.0)
                roi_x2 = roi_x1 + roi_w
                roi_y2 = roi_y1 + roi_h
                
                # Round and convert to int
                roi_x1 = int(round(roi_x1, 0))
                roi_x2 = int(round(roi_x2, 0))
                roi_y1 = int(round(roi_y1, 0))
                roi_y2 = int(round(roi_y2, 0))
                
                # Check ROI boundaries
                if roi_x1 <= 0:
                    roi_x1 = 0
                   
                if roi_y1 <= 0:
                    roi_y1 = 0
                
                # Check ROI width and height is always at least 1 in length
                if roi_y2 <= roi_y1:
                    roi_y2 = roi_y1+1
                    
                if roi_x2 <= roi_x1:
                    roi_x2 = roi_x1+1
                    
                roi_points = points[roi_y1:roi_y2, roi_x1:roi_x2]
            
            ### STEP 12: CALCULATE X, Y, and Z ###
            X = roi_points[:, :, 0].flatten()
            Y = roi_points[:, :, 1].flatten()
            Z = roi_points[:, :, 2].flatten()
            
            X = reject_outliers(X)
            Y = reject_outliers(Y)
            Z = reject_outliers(Z)
            
            X = X.mean()
            Y = Y.mean()
            Z = Z.mean()
            
            
            ### STEP 13: CALCULATE 3D POINT CLOUD ###
            if double_click_flag:
                f_name = "3D_Point_Cloud.ply"
                colors = cv2.cvtColor(l_img[:, (num_disparities+min_disparity):l_img.shape[1]], cv2.COLOR_BGR2RGB)
                mask = disp > disp.min()
                out_points = points[mask]
                out_colors = colors[mask]
                write_ply(f_name, out_points, out_colors)
                print("Wrote {}".format(f_name))
                double_click_flag = False
            
            
            ### STEP 14: DISPLAY ###
            
            # Normalizing disparity values
            disp = (disp - min_disparity)/num_disparities 
            
            # Crop image based on ROI
            roi = disp[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Makes the ellipse on the image
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
            cv2.ellipse(disp, ((track_box_x, track_box_y), (track_box_height, track_box_width), orientation), (255, 255, 255), 2) 
            cv2.putText(disp, "({:.1f}, {:.1f}, {:.1f})".format(X, Y, Z), (int(track_box_x), int(track_box_y)), cv2.FONT_HERSHEY_PLAIN, 1, (128,0,128))
            cv2.putText(disp, "({:.1f}, {:.1f})".format(track_box_width, track_box_height), (int(track_box_x), int(track_box_y)+15), cv2.FONT_HERSHEY_PLAIN, 1, (128,0,128))
            cv2.putText(disp, "({:.1f})".format(orientation), (int(track_box_x), int(track_box_y)+30), cv2.FONT_HERSHEY_PLAIN, 1, (128,0,128))  
            
            #Show diparity map, 3D points, and ROI based on camshift
            cv2.imshow('disp', disp)
            cv2.imshow('disparity_3d', points)
            cv2.imshow('roi', roi)
            
            
        if cv2.waitKey(1) == 27:
            break
      
    cap.release()
    
    cv2.destroyAllWindows()
