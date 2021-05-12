import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2

class Messure():

    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        self.Wd, self.Hd =  1280, 720
        if self.device_product_line == 'L500':
            self.Wc, self.Hc =  960, 540
        else:
            self.Wc, self.Hc =  self.Wd, self.Hd

        self.config.enable_stream(rs.stream.depth, self.Wd, self.Hd, rs.format.z16, 30)  # Depth stream
        self.config.enable_stream(rs.stream.color, self.Wc, self.Hc, rs.format.bgr8, 30)  # RGB stream

        # Start streaming
        self.pipeline.start(self.config)

        # Get stream profile and camera intrinsics
        self.profile = self.pipeline.get_active_profile()
        self.depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        self.color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        self.depth_intrinsics = self.depth_profile.get_intrinsics()
        self.color_intrinsics = self.color_profile.get_intrinsics()
        print("depth_intrinsics: \n{}".format(self.depth_intrinsics))
        print("color_intrinsics: \n{}".format(self.color_intrinsics))

        # Getting the depth sensor's depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Create an align object
        self.align_to = rs.stream.color
        #self.align_to = rs.stream.depth

        if self.align_to == rs.stream.depth:
            self.Wf = self.Wd
            self.Hf = self.Hd

        elif self.align_to == rs.stream.color:
            self.Wf = self.Wc
            self.Hf = self.Hc

        self.align = rs.align(self.align_to)

        # setup the filters
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.decimate = rs.decimation_filter()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

        # Geometry params
        self.points_buffer = {}  # 2d points dict
        self.calc_dist_reg = 10  # calculate distance in region of AxA

        self.planes = []  # 3d planes list
        self.mask = np.zeros((self.Hf, self.Wf))
        self.inter_3_planes = None
        self.lines = []
        
        # Misc
        self.stop = False  #stop flag
        self.pause_frame = False  # pause frame flag
        self.font = cv2.FONT_HERSHEY_SIMPLEX  # font


    def click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x >= self.Wf:
                x -= self.Wf

            dist, avg_dist, spatial_sigma, max_dist, min_dist = self.get_smart_dist(x, y)
            self.points_buffer[len(self.points_buffer)] = {'coords': (x,y), 
                                             'dist': dist,
                                             'avg_dist': avg_dist,
                                             'spatial_sigma': spatial_sigma,
                                             'max_dist': max_dist,
                                             'min_dist': min_dist}
            
            #print(self.get_3d_coords(self.points_buffer[len(self.points_buffer) - 1]))


    def get_smart_dist(self, x, y):
        d_mat = np.zeros((2*self.calc_dist_reg-1, 2*self.calc_dist_reg-1))
        for i in range(-self.calc_dist_reg, self.calc_dist_reg):
            for j in range(-self.calc_dist_reg, self.calc_dist_reg):
                d_mat[i][j] = self.depth_frame.get_distance(x+i, y+j)

        return (d_mat[0, 0], 
                np.average(d_mat),
                np.var(d_mat),
                np.max(d_mat),
                np.min(d_mat))
    

    def visuals(self):
        # Print visuals
        if self.inter_3_planes is not None:
            cv2.circle(self.images, self.inter_3_planes, 
                                    2, (20, 200, 20), -1)  # Left image

            for line in self.lines:
                cv2.circle(self.images, (int(line[0]), int(line[1])), 
                                    2, (20, 200, 20), -1)  # Left image

        # Print points
        for i in range(len(self.points_buffer)):
            cv2.circle(self.images, self.points_buffer[i]['coords'], 
                                    2, (20, 20, 200), -1)  # Left image
            cv2.circle(self.images, (self.points_buffer[i]['coords'][0] + self.Wf,
                                    self.points_buffer[i]['coords'][1]), 
                                    2, (20, 20, 200), -1)  # Right image

            #cv2.rectangle(self.images, (self.points_buffer[i]['coords'][0] - self.calc_dist_reg,
            #                            self.points_buffer[i]['coords'][1] - self.calc_dist_reg), 
            #                           (self.points_buffer[i]['coords'][0] + self.calc_dist_reg,
            #                            self.points_buffer[i]['coords'][1] + self.calc_dist_reg), (20, 20, 200), 1)  # Left image

            #cv2.putText(self.images, 
            #            "Sigma: {:.3g}".format(self.points_buffer[i]['spatial_sigma']),
            #            (self.points_buffer[i]['coords'][0] + self.calc_dist_reg, 
            #             self.points_buffer[i]['coords'][1] ),
            #             self.font, 0.3, (20, 20, 200), 1)  # Left image

            # if i > 0:
            #     cv2.line(self.images, self.points_buffer[i]['coords'], self.points_buffer[i-1]['coords'], 
            #                           (20, 20, 200), 1)  # Left image
            #     cv2.line(self.images, (self.points_buffer[i]['coords'][0] + self.Wf,
            #                           self.points_buffer[i]['coords'][1]), 
            #                           (self.points_buffer[i-1]['coords'][0] + self.Wf,
            #                           self.points_buffer[i-1]['coords'][1]), 
            #                           (20, 20, 200), 1)  # Right image

            #     P_prev = self.get_3d_coords(self.points_buffer[i-1])
            #     P_now = self.get_3d_coords(self.points_buffer[i])
            #     dist_3d = self.cal_3d_distance(P_prev, P_now)
            #     #dist_3d_avg = self.cal_3d_distance(P_prev_avg, P_now_avg)

            #     cv2.putText(self.images, 
            #                 "Dist: {}m".format(round(dist_3d,4)),
            #                 (int((self.points_buffer[i]['coords'][0] + self.points_buffer[i-1]['coords'][0]) / 2), 
            #                  int((self.points_buffer[i]['coords'][1] + self.points_buffer[i-1]['coords'][1]) / 2) ),
            #                 self.font, 0.4, (20, 200, 20), 1)  # Left image
            #     cv2.putText(self.images, 
            #                 "Dist: {}m".format(round(dist_3d,4)),
            #                 (int((self.points_buffer[i]['coords'][0] + self.points_buffer[i-1]['coords'][0]) / 2) + self.Wf, 
            #                  int((self.points_buffer[i]['coords'][1] + self.points_buffer[i-1]['coords'][1]) / 2) ),
            #                 self.font, 0.4, (20, 200, 20), 1)  # Right image

        # # Show images
        # cv2.imshow('RealSense-1', self.images) # self.depth_colormap
        # cv2.setMouseCallback('RealSense-1', self.click)

    
    def get_3d_coords(self, p1):
        """get_3d_coords gets 2d points in pixels, z distance and calculates 3d coords

        Arguments:
            p1 {[dic]} -- 2d point

        Returns:
            [X,Y,Z] -- 3d point
        """
        u, v = p1['coords']
        Z = p1['dist']

        if self.align_to == rs.stream.depth:
            X = (u - self.depth_intrinsics.ppx) * Z / self.depth_intrinsics.fx
            Y = (v - self.depth_intrinsics.ppy) * Z / self.depth_intrinsics.fy

        elif self.align_to == rs.stream.color:
            X = (u - self.color_intrinsics.ppx) * Z / self.color_intrinsics.fx
            Y = (v - self.color_intrinsics.ppy) * Z / self.color_intrinsics.fy

        else:
            raise ValueError('align_to not defined')

        return [X,Y,Z]

    
    def get_2d_coords(self, P1):
        if self.align_to == rs.stream.depth:
            f = np.array([[self.depth_intrinsics.fx , 0.0, self.depth_intrinsics.ppx],
                         [0.0 , self.depth_intrinsics.fy, self.depth_intrinsics.ppy],
                         [0.0 , 0.0, 1.0]])

        elif self.align_to == rs.stream.color:
            f = np.array([[self.color_intrinsics.fx , 0.0, self.color_intrinsics.ppx],
                         [0.0 , self.color_intrinsics.fy, self.color_intrinsics.ppy],
                         [0.0 , 0.0, 1.0]])

        else:
            raise ValueError('align_to not defined')

        p_2d = np.dot(f, P1)
        p_2d = p_2d / p_2d[-1]

        return(p_2d)
    

    def cal_3d_distance(self, P1, P2):
        """cal_3d_distance takes two 3d points and calculates distance

        Arguments:
            P1 {[X,Y,Z]} -- point1
            P2 {[X,Y,Z]} -- point2

        Returns:
            [float] -- distance in 3d
        """
        dx = P1[0] - P2[0]
        dy = P1[1] - P2[1]
        dz = P1[2] - P2[2]

        return np.sqrt(dx**2 + dy**2 + dz**2)

    
    def calc_plane(self, points):
        A = np.ndarray((len(points), 3))
        for i in range(len(points)):
            P = self.get_3d_coords(points[i])
            A[i, :] = P

        A = np.append(A, np.ones((len(points),1)), axis=1)
        w, v = np.linalg.eig(np.dot(A.T, A))

        return v[:,np.argmin(w)]


    def calc_plane_ransac(self, ):
        n = 3  # Points for plane
        k = 10  # Number of iterations
        accepted_error = 10**-3  # accepted distance of a point from the plane

        best_plane = None
        best_inliers_percent = None
        best_error = None
        
        for i in range(k):
            # 1. Select n random samples
            sample_number = np.random.choice(len(self.points_buffer), n, replace=False)
            points = []
            for j in sample_number:
                points.append(self.points_buffer[j])
            
            # 2. Calculate plane using selected n points
            plane = self.calc_plane(points)

            # 3. Find inliers
            inliers = []
            for k in range(len(self.points_buffer)):
                X,Y,Z = self.get_3d_coords(self.points_buffer[k])
                error = np.dot(plane, np.array([X, Y, Z, 1]))
                #print(error)
                if abs(error) < accepted_error:
                    inliers.append(self.points_buffer[k])
            
            inliers_percent = len(inliers) / len(self.points_buffer)

            # 4. Recompute plane with new inliers
            plane = self.calc_plane(inliers)
            error = 0
            for point in inliers:
                X,Y,Z = self.get_3d_coords(point)
                error += np.dot(plane, np.array([X, Y, Z, 1])) ** 2
            #print("RANSAC Plane: {}, error: {}, inliers: {}".format(plane, error, inliers_percent))

            # 5. Choose best model:
            if best_inliers_percent is None:  # Choose best model for the first run
                best_plane = plane
                best_inliers_percent = inliers_percent
                best_error = error

            if inliers_percent > best_inliers_percent:
                best_plane = plane
                best_inliers_percent = inliers_percent
                best_error = error

            if inliers_percent == best_inliers_percent:
                if error < best_error:
                    best_plane = plane
                    best_inliers_percent = inliers_percent
                    best_error = error

        #print("RANSAC best Plane: {}, error: {}, inliers: {}".format(best_plane, 
        #                                                             best_error, 
        #                                                             best_inliers_percent))

        return best_plane


    def calc_2plane_inter(self, plane1, plane2):
        cross = np.cross(plane1[:3], plane2[:3])
        A = np.array([plane1[:2], plane2[:2]])
        d = np.array([-plane1[3], -plane2[3]])
        p_inter = np.linalg.solve(A, d).T
        p_inter = np.append(p_inter, np.array([0]))

        return p_inter, cross

    
    def calc_3plane_inter(self, plane1, plane2, plane3):
        A = np.array([plane1[:3], plane2[:3], plane3[:3]])
        d = np.array([-plane1[3], -plane2[3], -plane3[3]])
        p_inter = np.linalg.solve(A, d).T

        return p_inter
    

    def draw_plane(self, plane_eq):
        for x in range(self.Wf-1):
            for y in range(self.Hf-1):

                p = {'coords': (x,y), 
                     'dist': self.depth_frame.get_distance(x, y)}

                X,Y,Z = self.get_3d_coords(p)
                a = np.dot(plane_eq, np.array([X, Y, Z, 1]))
                if abs(a) < 0.001:
                    print("(x,y) = ({},{})| X, Y, Z = {}, {}, {}| a = {}".format(x,y,X,Y,Z,a))
                    self.mask[y, x] = 1
        
        cv2.imshow("mask", self.mask)


    def keyboard_clicks(self):

        key = cv2.waitKey(1)

        if key == ord("q"):  # q quits the program
            self.stop = True

        if key == ord("n"):  # resets the points
            self.points_buffer = {}
            self.planes = []
            self.lines = []
            self.inter_3_planes = None
            self.mask = np.zeros((self.Hf, self.Wf))

        if key == ord(" "):  # pause the video
            if self.pause_frame:
                self.pause_frame = False

            else:
                self.pause_frame = True

        if key == ord("p"):  # calculate plane
            plane = self.calc_plane(self.points_buffer)
            plane_ransac = self.calc_plane_ransac()
            print("Fitted Plane: {}".format(plane))
            print("RANSAC Plane: {}".format(plane_ransac))
            self.planes.append(plane_ransac)

            self.points_buffer = {}
            self.mask = np.zeros((self.Hf, self.Wf))

            if len(self.planes) > 2:
                p_inter_3d = self.calc_3plane_inter(self.planes[0], self.planes[1], self.planes[2])
                p_inter_2d = self.get_2d_coords(p_inter_3d)
                self.inter_3_planes = int(p_inter_2d[0]), int(p_inter_2d[1])
                print("3d: {}".format(p_inter_3d))

                for plane_num in range(-1,2):
                    cross = np.cross(self.planes[plane_num][:3], self.planes[plane_num+1][:3])
                    print("cross: {}".format(cross))

                    for t in range(-100, 100):
                        print(t)
                        p_3d = p_inter_3d + t * cross / 500.0
                        print("p_3d: {}".format(p_3d))

                        p_2d = self.get_2d_coords(p_3d)
                        print("p_2d: {}".format(p_2d))
                        #self.lines.append(p_2d)

                        if p_2d[0] > 0 and p_2d[0] < self.Wf and \
                           p_2d[1] > 0 and p_2d[1] < self.Hf:

                            p_2d_dict = {'coords' : (int(p_2d[0]), int(p_2d[1])),
                                        'dist' : self.depth_frame.get_distance(int(p_2d[0]), int(p_2d[1])) 
                                        }
                            
                            p_3d_real = self.get_3d_coords(p_2d_dict)
                            print("p_3d_real: {}".format(p_3d_real))

                            error = self.cal_3d_distance(p_3d_real, p_3d)
                            print("error: {}".format(error))

                            if error < 0.01: #
                                self.lines.append(p_2d)

                        print('\n')


    def run(self):

        while not self.stop:
            # Wait for a coherent pair of frames: depth and color
            if not self.pause_frame:
                frames = self.pipeline.wait_for_frames()

                # Align frames
                frames = self.align.process(frames)
                
                self.depth_frame = frames.get_depth_frame()
                self.color_frame = frames.get_color_frame()
                
                if not self.depth_frame or not self.color_frame:
                    continue

                # Apply the filters
                self.depth_frame = self.depth_to_disparity.process(self.depth_frame)
                self.depth_frame = self.spatial.process(self.depth_frame)
                self.depth_frame = self.temporal.process(self.depth_frame)
                self.depth_frame = self.disparity_to_depth.process(self.depth_frame)
                self.depth_frame = self.hole_filling.process(self.depth_frame)
                self.depth_frame.__class__ = rs.depth_frame

            # Apply colormap on depth image and convert to numpy
            colorizer = rs.colorizer()
            self.depth_colormap = np.asanyarray(colorizer.colorize(self.depth_frame).get_data())
            self.depth_image_np = np.asanyarray(self.depth_frame.get_data())
            self.color_image = np.asanyarray(self.color_frame.get_data())

            # Dims
            depth_colormap_dim = self.depth_colormap.shape
            color_colormap_dim = self.color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            # Mandatory for L500 without alignment
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(self.color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                self.images = np.hstack((resized_color_image, self.depth_colormap))
            else:
                self.images = np.hstack((self.color_image, self.depth_colormap))
                
            self.visuals()  # display visuals
            self.keyboard_clicks()  # keyboard interaction

            # Show images
            cv2.imshow('RealSense-1', self.images) # self.depth_colormap
            cv2.setMouseCallback('RealSense-1', self.click)


        self.pipeline.stop()  # Stop streaming


if __name__ == '__main__':

    cam = Messure()
    cam.run()
    print("Done")
