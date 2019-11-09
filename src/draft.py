#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
import ros_numpy
import numpy as np
from scipy.spatial import KDTree
import math
import time


class LocalMap:
    def __init__(self, topic):
        self.subscriber = rospy.Subscriber(topic, PointCloud2, self.callback)
        self.publisher_visualize = rospy.Publisher("/elevation/local_stairs", PointCloud2, queue_size=10)
        self.publisher_stairs = rospy.Publisher("/elevation/stairs", PointCloud2, queue_size=10)

    def callback(self, data):
        cloud_arr = ros_numpy.point_cloud2.pointcloud2_to_array(data)       #read raw data

        new_arr = cloud_arr.copy()  # make new array to be published

        # get coords data arrays
        x_cloud = cloud_arr['x'].reshape(len(cloud_arr['x']), 1)
        y_cloud = cloud_arr['y'].reshape(len(cloud_arr['y']), 1)
        z_cloud = cloud_arr['z'].reshape(len(cloud_arr['z']), 1)

        # merge x and y coords
        xy_2D = np.concatenate((x_cloud, y_cloud), axis=1)

        # convert my 2D x/y array to KDTree to find nearest neighbors
	kdtree_2D = KDTree(xy_2D)


        # init arrays to save detected stairs points
        stairs_points = []
        x_stairs = []
        y_stairs = []
        stairs_indexes = []

        # go trough points
        for i in range(len(cloud_arr)):
            # get current points coords
            xy_point = np.array([cloud_arr[i][0], cloud_arr[i][1]])

            # find nearest neighbors and check their distances
            distance, indexes = kdtree_2D.query(xy_point, 25)

            distance_check = distance < 0.25
            indexes = indexes[distance_check]

            # if there are any neighbors within range
            if indexes.size != 0:
                x = x_cloud[indexes]
                y = y_cloud[indexes]
                z = z_cloud[indexes]
		if True:
                	# fit plane to points
                	X = np.concatenate((x, y, np.ones((x.shape[0], 1))), axis=1)
                	w = np.dot(np.linalg.pinv(X), z)
	    	else:
			# Fit using ransac
			try:
                		X = np.concatenate((x, y), axis=1)
                		reg = RANSACRegressor(random_state=0, max_trials=15, min_samples=1).fit(X, z)
                		w = reg.estimator_.coef_[0]
			except:
				continue

                # calculate the angle from horizontal plane
                alpha = np.arccos(1/math.sqrt(w[0]**2 + w[1]**2 + 1))

                if alpha > 0.15 and alpha < 0.6:
                    # save info about stairs point
                    stairs_indexes.append(i)
                    stairs_points.append(new_arr[i])
                    x_stairs.append(new_arr[i][0])
                    y_stairs.append(new_arr[i][1])

        # init array for definitive stairs points
        final_stairs = []

        # merge x and y coords of stairs points and make KDTree
        x_stairs = np.array(x_stairs).reshape(len(x_stairs), 1)
        y_stairs = np.array(y_stairs).reshape(len(y_stairs), 1)

        xy_2D = np.concatenate((x_stairs, y_stairs), axis=1)
        kdtree_2D = KDTree(xy_2D)


        # go through stairs points and check whether they really are on stairs
        for i in range(len(stairs_points)):
            # get current points coords
            xy_point = np.array([stairs_points[i][0], stairs_points[i][1]])
            # check 9 nearest neighbors in 2D
            distance, indexes = kdtree_2D.query(xy_point, 9)

            distance_check = distance < 0.15
            indexes = indexes[distance_check]

            # if in the neighborhood are all points also qualified as stairs
            if len(indexes) >= 9:
                # prepare stair point to be published
                new_stair_point = stairs_points[i].copy()
                new_stair_point[3] = 1
                final_stairs.append(new_stair_point)
                # change color for visualization
                idx = stairs_indexes[i]
                new_arr[idx][3] = 0b111111110000000001111111

        # convert list to numpy array
        final_stairs = np.array(final_stairs)
        # create pointcloud2 messages
        msg_visualize = ros_numpy.point_cloud2.array_to_pointcloud2(new_arr, stamp=data.header.stamp,
                                                          frame_id=data.header.frame_id)
        msg_stairs = ros_numpy.point_cloud2.array_to_pointcloud2(final_stairs, stamp=data.header.stamp,
                                                          frame_id=data.header.frame_id)
        # publish messages
        self.publisher_visualize.publish(msg_visualize)
        self.publisher_stairs.publish(msg_stairs)

def local_map_evaluation():
    """
    Function creating subscriber nodes on ROS server.
    """
    # init node and get output_dir param from launch file
    rospy.init_node('local_map_node', anonymous=True)


    local_map_node = LocalMap("/elevation/local")

    rospy.spin()

if __name__ == '__main__':
    local_map_evaluation()