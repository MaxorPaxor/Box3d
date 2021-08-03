import pyrealsense2.pyrealsense2 as rs
import pyransac3d as pyrsc
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt

RECORDED = False  # Recorded video should be 640x480 both color and depth. color format should be BGR8.


class Measure:

    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        # self.Wd, self.Hd = 1280, 720
        self.Wd, self.Hd = 640, 480
        # self.Wd, self.Hd = 1024, 768
        if self.device_product_line == 'L500':
            self.Wc, self.Hc = 960, 540
        else:
            self.Wc, self.Hc = self.Wd, self.Hd

        if RECORDED:
            rs.config.enable_device_from_file(self.config, "/home/alum/Documents/20210530_161310.bag")
            self.config.enable_stream(rs.stream.depth, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
        else:
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
        # print("depth_intrinsics: \n{}".format(self.depth_intrinsics))
        # print("color_intrinsics: \n{}".format(self.color_intrinsics))

        # Getting the depth sensor's depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Create an align object
        self.align_to = rs.stream.color
        # self.align_to = rs.stream.depth

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
        self.points_buffer = {}  # 2d points dict {(x, y): dist}
        self.points_3d_buffer_list = []  # 3d points list

        self.planes = []  # 3d planes list
        self.inter_points = []
        self.inter_3d_points = []

        # Misc
        self.stop = False  # stop flag
        self.pause_frame = False  # pause frame flag
        self.font = cv2.FONT_HERSHEY_SIMPLEX  # font
        self.draw = False

    def click(self, event, x, y, flags, param):
        """
        Click function. A paint-like brush tool to color an object
        and collect all of its pixels including depth information
        to the global buffer of the class.
        """

        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.draw:
                if x >= self.Wf:
                    x -= self.Wf

                click_area = 4
                scatter_factor = 4

                if len(self.points_buffer) > 0:
                    for i in range(-click_area+1, click_area):
                        for j in range(-click_area+1, click_area):

                            if np.sum([v['coords'] == (x + i*scatter_factor, y + j*scatter_factor) for v in self.points_buffer.values()]) == 0:
                                # Don't add new points if they exists in the buffer

                                dist = self.depth_frame.get_distance(x+i*scatter_factor, y+j*scatter_factor)

                                if dist != 0:
                                    self.points_buffer[len(self.points_buffer)] = {'coords': (x+i*scatter_factor, y+j*scatter_factor),
                                                                                   'dist': dist}

                                    tmp_3d = self.get_3d_coords(self.points_buffer[len(self.points_buffer) - 1])
                                    self.points_3d_buffer_list.append(tmp_3d)

                else:
                    for i in range(-click_area+1, click_area):
                        for j in range(-click_area+1, click_area):
                            dist = self.depth_frame.get_distance(x + i*scatter_factor, y + j*scatter_factor)

                            if dist != 0:
                                self.points_buffer[len(self.points_buffer)] = {'coords': (x + i*scatter_factor, y + j*scatter_factor),
                                                                               'dist': dist}

                                tmp_3d = self.get_3d_coords(self.points_buffer[len(self.points_buffer) - 1])
                                self.points_3d_buffer_list.append(tmp_3d)

        elif event == cv2.EVENT_LBUTTONUP:
            self.draw = False

    def get_smart_dist(self, x, y):
        """
        Uses smart method to calculate distance. Based on neighbourhood of a single pixel.
        """
        calc_dist_reg = 10
        d_mat = np.zeros((2 * calc_dist_reg - 1, 2 * calc_dist_reg - 1))
        for i in range(-calc_dist_reg, calc_dist_reg):
            for j in range(-calc_dist_reg, calc_dist_reg):
                d_mat[i][j] = self.depth_frame.get_distance(x + i, y + j)

        return (d_mat[0, 0],
                np.average(d_mat),
                np.var(d_mat),
                np.max(d_mat),
                np.min(d_mat))

    def visuals(self):
        """
        visuals draws visuals on the color-depth image pair.
        :return: None
        """
        if len(self.inter_points) > 0:
            i = 1
            for point in self.inter_points:
                cv2.circle(self.images, (point[0], point[1]),
                           4, (20, 200, 20), -1)  # Left image

                cv2.putText(self.images,
                            "{}".format(i),
                            (point[0], point[1]),
                            self.font, 0.6, (20, 20, 200), 1)  # Left image
                i += 1

            # Box lines - Left image
            cv2.line(self.images, tuple(self.inter_points[0]), tuple(self.inter_points[1]), (20, 200, 20), 1)  # 1-2
            cv2.line(self.images, tuple(self.inter_points[0]), tuple(self.inter_points[2]), (20, 200, 20), 1)  # 1-3
            cv2.line(self.images, tuple(self.inter_points[1]), tuple(self.inter_points[3]), (20, 200, 20), 1)  # 2-4
            cv2.line(self.images, tuple(self.inter_points[2]), tuple(self.inter_points[3]), (20, 200, 20), 1)  # 3-4

            cv2.line(self.images, tuple(self.inter_points[4]), tuple(self.inter_points[5]), (20, 200, 20), 1)  # 5-6
            cv2.line(self.images, tuple(self.inter_points[4]), tuple(self.inter_points[6]), (20, 200, 20), 1)  # 5-7
            cv2.line(self.images, tuple(self.inter_points[5]), tuple(self.inter_points[7]), (20, 200, 20), 1)  # 6-8
            cv2.line(self.images, tuple(self.inter_points[6]), tuple(self.inter_points[7]), (20, 200, 20), 1)  # 7-8

            cv2.line(self.images, tuple(self.inter_points[0]), tuple(self.inter_points[4]), (20, 200, 20), 1)  # 1-5
            cv2.line(self.images, tuple(self.inter_points[1]), tuple(self.inter_points[5]), (20, 200, 20), 1)  # 2-6
            cv2.line(self.images, tuple(self.inter_points[2]), tuple(self.inter_points[6]), (20, 200, 20), 1)  # 3-7
            cv2.line(self.images, tuple(self.inter_points[3]), tuple(self.inter_points[7]), (20, 200, 20), 1)  # 4-8

            cv2.putText(self.images, "Distance 1-5: {:.2f} mm".format(round(abs(self.planes[0][3] - self.planes[3][3])*1000, 3)),
                        (10, 20), self.font, 0.6, (20, 200, 20), 1)  # Left image

            cv2.putText(self.images, "Distance 1-2: {:.2f} mm".format(round(abs(self.planes[2][3] - self.planes[5][3])*1000, 3)),
                        (10, 40), self.font, 0.6, (20, 200, 20), 1)  # Left image

            cv2.putText(self.images, "Distance 1-3: {:.2f} mm".format(round(abs(self.planes[1][3] - self.planes[4][3])*1000, 3)),
                        (10, 60), self.font, 0.6, (20, 200, 20), 1)  # Left image

        else:
            # Draw points
            for i in range(len(self.points_buffer)):
                cv2.circle(self.images, self.points_buffer[i]['coords'],
                           2, (20, 20, 200), -1, cv2.LINE_AA)  # Left image
                cv2.circle(self.images, (self.points_buffer[i]['coords'][0] + self.Wf,
                                         self.points_buffer[i]['coords'][1]),
                           2, (20, 20, 200), -1)  # Right image

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

        return [X, Y, Z]

    def get_2d_coords(self, P1):
        """

        :param P1: [X,Y,Z] -- 3d point, float
        :return: p_2d: [u, v] -- 2d point, int
        """
        if self.align_to == rs.stream.depth:
            f = np.array([[self.depth_intrinsics.fx, 0.0, self.depth_intrinsics.ppx],
                          [0.0, self.depth_intrinsics.fy, self.depth_intrinsics.ppy],
                          [0.0, 0.0, 1.0]])

        elif self.align_to == rs.stream.color:
            f = np.array([[self.color_intrinsics.fx, 0.0, self.color_intrinsics.ppx],
                          [0.0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
                          [0.0, 0.0, 1.0]])

        else:
            raise ValueError('align_to not defined')

        p_2d = np.dot(f, P1)
        p_2d = p_2d / p_2d[-1]

        return [int(p_2d[0]), int(p_2d[1])]

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

        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def calc_plane(self, points):
        """
        Calculates plane using least quares
        :param points: list of point dicts {'coords': (x, y), 'dist': dist}
        :return: A, B, C, D -- np(4,1) Ax+ By +Cz +d =0
        """
        A = np.ndarray((len(points), 3))
        for i in range(len(points)):
            P = self.get_3d_coords(points[i])
            A[i, :] = P

        A = np.append(A, np.ones((len(points), 1)), axis=1)
        w, v = np.linalg.eig(np.dot(A.T, A))

        return v[:, np.argmin(w)]

    def calc_plane_ransac(self, ):
        """
        Calculates plane using RANSAC
        :return: A, B, C, D -- np(4,1) Ax+ By +Cz +d =0
        """
        n = 3  # Points for plane
        k = 10  # Number of iterations
        accepted_error = 10 ** -3  # accepted distance of a point from the plane

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
                X, Y, Z = self.get_3d_coords(self.points_buffer[k])
                error = np.dot(plane, np.array([X, Y, Z, 1]))
                if abs(error) < accepted_error:
                    inliers.append(self.points_buffer[k])

            inliers_percent = len(inliers) / len(self.points_buffer)

            # 4. Recompute plane with new inliers
            plane = self.calc_plane(inliers)
            error = 0
            for point in inliers:
                X, Y, Z = self.get_3d_coords(point)
                error += np.dot(plane, np.array([X, Y, Z, 1])) ** 2
            # print("RANSAC Plane: {}, error: {}, inliers: {}".format(plane, error, inliers_percent))

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

        # print("RANSAC best Plane: {}, error: {}, inliers: {}".format(best_plane,
        #                                                             best_error, 
        #                                                             best_inliers_percent))

        return best_plane

    def calc_2plane_inter(self, plane1, plane2):
        """
        Returns intersection line between two planes
        :param plane1: [A, B, C, D]
        :param plane2: [A, B, C, D]
        :return: p_inter - intersection points, cross - intersection vector direction
        """
        cross = np.cross(plane1[:3], plane2[:3])
        A = np.array([plane1[:2], plane2[:2]])
        d = np.array([-plane1[3], -plane2[3]])
        p_inter = np.linalg.solve(A, d).T
        p_inter = np.append(p_inter, np.array([0]))

        return p_inter, cross

    def calc_3plane_inter(self, plane1, plane2, plane3):
        """
        Takes 3 planes and returns 3d intersection point
        :param plane1: [A, B, C, D]
        :param plane2: [A, B, C, D]
        :param plane3: [A, B, C, D]
        :return: [X, Y, Z]
        """
        A = np.array([plane1[:3], plane2[:3], plane3[:3]])
        d = np.array([-plane1[3], -plane2[3], -plane3[3]])
        p_3d_inter = np.linalg.solve(A, d).T
        p_2d_inter = self.get_2d_coords(p_3d_inter)

        self.inter_3d_points.append(p_3d_inter)
        self.inter_points.append(p_2d_inter)

        return p_3d_inter, p_2d_inter

    def calc_8plane_inter(self):
        # Short way
        # for i in range(0, 4, 3):
        #     for j in range(0, 4, 3):
        #         for k in range(0, 4, 3):
        #             self.calc_3plane_inter(self.planes[0+i], self.planes[1+j], self.planes[2+k])

        # Long way
        self.calc_3plane_inter(self.planes[0], self.planes[1], self.planes[2])  # 1
        self.calc_3plane_inter(self.planes[0], self.planes[1], self.planes[2+3])  # 2
        self.calc_3plane_inter(self.planes[0], self.planes[1+3], self.planes[2])  # 3
        self.calc_3plane_inter(self.planes[0], self.planes[1+3], self.planes[2+3])  # 4

        self.calc_3plane_inter(self.planes[0+3], self.planes[1], self.planes[2])  # 5
        self.calc_3plane_inter(self.planes[0+3], self.planes[1], self.planes[2+3])  # 6
        self.calc_3plane_inter(self.planes[0+3], self.planes[1+3], self.planes[2])  # 7
        self.calc_3plane_inter(self.planes[0+3], self.planes[1+3], self.planes[2+3])  # 8

    def add_opposite_plane(self, plane, points):
        """
        Gets a plane and a 3d point cloud, and calculates a second plane: new_plane
        Which is parallel to the plane and bounds all cloud points.
        :param plane: [A, B, C, D]
        :param points:  list of 3d cloud points [[X, Y, Z]i]
        :return: None
        """
        plane_norm = np.array(plane[:3])
        proj = -np.dot(points, plane_norm)
        proj.sort()

        max_proj = np.max(proj)
        min_proj = np.min(proj)
        avg_proj = np.average(proj)

        # print(np.histogram(proj, bins=100))

        new_plane = plane[:3]
        print("max: {}, min: {}, AVG: {}".format(max_proj, min_proj, avg_proj))
        if abs(max_proj - plane[3]) < abs(min_proj - plane[3]):
            new_plane.append(min_proj)
            # new_plane.append(proj[int(0.005 * len(proj))])
            print(
                "Adding new plane. min proj: {}, custom proj : {}".format(min_proj, proj[int(0.005 * len(proj))]))
        else:
            new_plane.append(max_proj)
            # new_plane.append(proj[int(0.995 * len(proj))])
            print(
                "Adding new plane. max proj: {}, custom proj : {}".format(max_proj, proj[int(0.995 * len(proj))]))

        self.planes.append(new_plane)

        return proj

    def display_inlier_outlier(self, cloud, ind):
        """
        Open3d helper function to draw outliers and inliers
        """
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    def keyboard_clicks(self):
        """
        Keyboard key presses during main loop
        """
        key = cv2.waitKey(1)

        if key == ord("q"):  # q quits the program
            self.stop = True

        if key == ord("n"):  # resets the points
            self.points_buffer = {}  # 2d points dict
            self.points_3d_buffer_list = []  # 3d points dict

            self.planes = []  # 3d planes list
            self.inter_points = []
            self.inter_3d_points = []

        if key == ord(" "):  # pause the video
            if self.pause_frame:
                self.pause_frame = False

            else:
                self.pause_frame = True

        if key == ord("c"):  # calculate cuboid from buffer points
            cube = pyrsc.Cuboid()
            best_eq, best_inliers = cube.fit(np.array(self.points_3d_buffer_list), thresh=2e-2, maxIteration=10000)
            self.planes.append(best_eq.tolist())
            self.planes = self.planes[0]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(self.points_3d_buffer_list))
            pcd_good_ransac = pcd.select_by_index(best_inliers)

            ### Filtering
            # cl, ind = pcd_good.remove_radius_outlier(nb_points=len(best_inliers), radius=1)  # Radial filter
            cl, ind = pcd_good_ransac.remove_statistical_outlier(nb_neighbors=int(1*len(best_inliers)),
                                                          std_ratio=2.0)  # Statistical filter
            # self.display_inlier_outlier(pcd_good_ransac, ind)
            pcd_good = pcd_good_ransac.select_by_index(ind)

            ### Clustering
            # with o3d.utility.VerbosityContextManager(
            #         o3d.utility.VerbosityLevel.Debug) as cm:
            #     labels = np.array(
            #         pcd_good.cluster_dbscan(eps=0.1, min_points=100, print_progress=True))
            # max_label = labels.max()
            # print(f"point cloud has {max_label + 1} clusters")
            # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            # colors[labels < 0] = 0
            # pcd_good.colors = o3d.utility.Vector3dVector(colors[:, :3])
            # o3d.visualization.draw_geometries([pcd_good])

            # pcd_good.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(2000, 3)))
            # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_good, voxel_size=0.005)
            # o3d.visualization.draw_geometries([voxel_grid])
            # octree = o3d.geometry.Octree(max_depth=8)
            # octree.create_from_voxel_grid(voxel_grid)
            # o3d.visualization.draw_geometries([octree])

            # octree = o3d.geometry.Octree(max_depth=4)
            # octree.convert_from_point_cloud(pcd_good, size_expand=0.01)
            # o3d.visualization.draw_geometries([octree])

            points = np.asarray(pcd_good.points)

            self.add_opposite_plane(self.planes[0], points)
            self.add_opposite_plane(self.planes[1], points)
            self.add_opposite_plane(self.planes[2], points)

            self.calc_8plane_inter()

            print("BOX DIMENSIONS: {} x {} x {} mm".format(round(abs(self.planes[0][3] - self.planes[3][3])*1000, 2),
                                                       round(abs(self.planes[1][3] - self.planes[4][3])*1000, 2),
                                                       round(abs(self.planes[2][3] - self.planes[5][3])*1000, 2))
                  )

            # pcd_bbox = o3d.geometry.PointCloud()
            # pcd_bbox.points = o3d.utility.Vector3dVector(np.array(self.inter_3d_points))
            # hull, _ = pcd_bbox.compute_convex_hull()
            # hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
            # hull_ls.paint_uniform_color((1, 0, 0))
            # o3d.visualization.draw_geometries([pcd_bbox, pcd_good, hull_ls], point_show_normal=True)

    def run(self):
        """
        The main function in the class that runs the camera,
        filters the obtained depth-image,
        and calls for mouse-click and keyboard-clicks
        """

        cv2.namedWindow("RealSense-1", cv2.WINDOW_AUTOSIZE)
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
                # self.depth_frame = self.temporal.process(self.depth_frame)
                self.depth_frame = self.disparity_to_depth.process(self.depth_frame)
                # self.depth_frame = self.hole_filling.process(self.depth_frame)
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
                resized_color_image = cv2.resize(self.color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                 interpolation=cv2.INTER_AREA)
                self.images = np.hstack((resized_color_image, self.depth_colormap))
            else:
                self.images = np.hstack((self.color_image, self.depth_colormap))

            self.visuals()  # display visuals
            # self.keyboard_clicks()  # keyboard interaction

            # Show images
            cv2.imshow("RealSense-1", self.images)  # self.depth_colormap
            cv2.setMouseCallback("RealSense-1", self.click)
            self.keyboard_clicks()  # keyboard interaction

        self.pipeline.stop()  # Stop streaming


if __name__ == '__main__':
    cam = Measure()
    cam.run()
    print("Done")
