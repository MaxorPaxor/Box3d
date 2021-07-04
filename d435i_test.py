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

        self.Wd, self.Hd =  640, 480
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

        self.points = {}  # 2d points dict
        self.pause_frame = False  # pause frame flag
        self.calc_dist_reg = 10  # calculate distance in region of AxA
        self.font = cv2.FONT_HERSHEY_SIMPLEX  # font


    def click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x >= self.Wf:
                x -= self.Wf

            dist, avg_dist, spatial_sigma, max_dist, min_dist = self.get_smart_dist(x, y)
            self.points[len(self.points)] = {'coords': (x,y), 
                                             'dist': dist,
                                             'avg_dist': avg_dist,
                                             'spatial_sigma': spatial_sigma,
                                             'max_dist': max_dist,
                                             'min_dist': min_dist}

    
    def get_smart_dist(self, x, y):
        d_mat = np.zeros((2*self.calc_dist_reg-1, 2*self.calc_dist_reg-1))
        for i in range(-self.calc_dist_reg, self.calc_dist_reg):
            for j in range(-self.calc_dist_reg, self.calc_dist_reg):
                d_mat[i][j] = self.depth_frame.get_distance(x+i, y+j)

        #print("AVG dist: {}, spatial_sigma: {}".format(np.average(d_mat), np.var(d_mat)))
        #print("dist: {}".format(d_mat[0, 0]))
        #print("min: {}".format(np.min(d_mat)))
        #print("min: {}".format(np.max(d_mat)))

        return (d_mat[0, 0], 
                np.average(d_mat),
                np.var(d_mat),
                np.max(d_mat),
                np.min(d_mat))
    

    def visuals(self):
        # Print visuals
        for i in range(len(self.points)):
            cv2.circle(self.images, self.points[i]['coords'], 
                                    2, (20, 20, 200), -1)  # Left image
            cv2.circle(self.images, (self.points[i]['coords'][0] + self.Wf,
                                    self.points[i]['coords'][1]), 
                                    2, (20, 20, 200), -1)  # Right image

            cv2.rectangle(self.images, (self.points[i]['coords'][0] - self.calc_dist_reg,
                                        self.points[i]['coords'][1] - self.calc_dist_reg), 
                                       (self.points[i]['coords'][0] + self.calc_dist_reg,
                                        self.points[i]['coords'][1] + self.calc_dist_reg), (20, 20, 200), 1)  # Left image

            cv2.putText(self.images, 
                        "Sigma: {:.3g}".format(self.points[i]['spatial_sigma']),
                        (self.points[i]['coords'][0] + self.calc_dist_reg, 
                         self.points[i]['coords'][1] ),
                         self.font, 0.3, (20, 20, 200), 1)  # Left image

            if i > 0:
                cv2.line(self.images, self.points[i]['coords'], self.points[i-1]['coords'], 
                                      (20, 20, 200), 1)  # Left image
                cv2.line(self.images, (self.points[i]['coords'][0] + self.Wf,
                                      self.points[i]['coords'][1]), 
                                      (self.points[i-1]['coords'][0] + self.Wf,
                                      self.points[i-1]['coords'][1]), 
                                      (20, 20, 200), 1)  # Right image

                P_prev, P_prev_avg = self.get_3d_coords(self.points[i-1])
                P_now, P_now_avg = self.get_3d_coords(self.points[i])
                dist_3d = self.cal_3d_distance(P_prev, P_now)
                dist_3d_avg = self.cal_3d_distance(P_prev_avg, P_now_avg)

                cv2.putText(self.images, 
                            "Dist: {}m, AVG Dist: {}".format(round(dist_3d,4), round(dist_3d_avg,4)),
                            (int((self.points[i]['coords'][0] + self.points[i-1]['coords'][0]) / 2), 
                             int((self.points[i]['coords'][1] + self.points[i-1]['coords'][1]) / 2) ),
                            self.font, 0.4, (20, 200, 20), 1)  # Left image
                cv2.putText(self.images, 
                            "Dist: {}m, AVG Dist: {}".format(round(dist_3d,4), round(dist_3d_avg,4)),
                            (int((self.points[i]['coords'][0] + self.points[i-1]['coords'][0]) / 2) + self.Wf, 
                             int((self.points[i]['coords'][1] + self.points[i-1]['coords'][1]) / 2) ),
                            self.font, 0.4, (20, 200, 20), 1)  # Right image

        # Show images
        cv2.imshow('RealSense-1', self.images) # self.depth_colormap
        cv2.setMouseCallback('RealSense-1', self.click)

    
    def get_3d_coords(self, p1):
        """get_3d_coords gets 2d points in pixels, z distance and calculates 3d coords

        Arguments:
            p1 {[dic]} -- 2d point

        Returns:
            [X,Y,Z] -- 3d point
        """
        u, v = p1['coords']
        Z = p1['dist']
        Z_avg = p1['avg_dist']

        if self.align_to == rs.stream.depth:
            X = (u - self.depth_intrinsics.ppx) * Z / self.depth_intrinsics.fx
            Y = (v - self.depth_intrinsics.ppy) * Z / self.depth_intrinsics.fy
            X_avg = (u - self.depth_intrinsics.ppx) * Z_avg / self.depth_intrinsics.fx
            Y_avg = (v - self.depth_intrinsics.ppy) * Z_avg / self.depth_intrinsics.fy

        elif self.align_to == rs.stream.color:
            X = (u - self.color_intrinsics.ppx) * Z / self.color_intrinsics.fx
            Y = (v - self.color_intrinsics.ppy) * Z / self.color_intrinsics.fy
            X_avg = (u - self.color_intrinsics.ppx) * Z_avg / self.color_intrinsics.fx
            Y_avg = (v - self.color_intrinsics.ppy) * Z_avg / self.color_intrinsics.fy

        else:
            raise ValueError('align_to not defined')

        return [X,Y,Z], [X_avg, Y_avg, Z_avg]

    
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


    def run(self):

        while True:
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
                
            self.visuals()

            key = cv2.waitKey(1)
            if key == ord("q"):  # q quits the program
                break
            if key == ord("n"):  # resets the points
                self.points = {}
            if key == ord(" "):  # pause the video
                if self.pause_frame:
                    self.pause_frame = False
                else:
                    self.pause_frame = True

        # Stop streaming
        self.pipeline.stop()


if __name__ == '__main__':

    cam = Messure()
    cam.run()
    print("Done")
