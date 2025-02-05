import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
#from semantic_predictor_segformer_multicat import SemanticPredictor
from cv_bridge import CvBridge
import yaml
import tf
import cv2
import numpy as np
import pandas as pd

MAX_DEPTH = 10


def getCameraInfo(filepath):
    with open(filepath, 'r') as f:
        yaml_data = yaml.safe_load(f)
    width = yaml_data['image_width']
    height = yaml_data['image_height']
    D = yaml_data['distortion_coefficients']['data']
    K = yaml_data['camera_matrix']['data']
    R = yaml_data['rectification_matrix']['data']
    P = yaml_data['projection_matrix']['data']
    return CameraInfo(width=width, height=height, D=D, K=K, R=R, P=P)


class HabitatObservationPublisher:

    def __init__(self,
                 rgb_topic=None,
                 depth_topic=None,
                 #semantic_topic=None,
                 camera_info_topic=None,
                 true_pose_topic=None,
                 camera_info_file=None):
        self.cvbridge = CvBridge()

        # Initialize camera info publisher
        if camera_info_topic is not None:
            self.publish_camera_info = True
            self.camera_info_publisher = rospy.Publisher(camera_info_topic, CameraInfo, latch=True, queue_size=100)
            self.camera_info = getCameraInfo(camera_info_file)
        else:
            self.publish_camera_info = False

        # Initialize RGB image publisher
        if rgb_topic is not None:
            self.publish_rgb = True
            self.image_publisher1 = rospy.Publisher('habitat/rgb1/image', Image, latch=True, queue_size=100)
            self.image_publisher2 = rospy.Publisher('habitat/rgb2/image', Image, latch=True, queue_size=100)
            self.image_publisher3 = rospy.Publisher('habitat/rgb3/image', Image, latch=True, queue_size=100)
            self.image_publisher4 = rospy.Publisher('habitat/rgb4/image', Image, latch=True, queue_size=100)
            self.image = Image()
            self.image.is_bigendian = False
        else:
            self.publish_rgb = False

        # Initialize depth image publisher
        if depth_topic is not None:
            self.publish_depth = True
            self.depth_publisher1 = rospy.Publisher('/habitat/depth1/image', Image, latch=True, queue_size=100)
            self.depth_publisher2 = rospy.Publisher('/habitat/depth2/image', Image, latch=True, queue_size=100)
            self.depth_publisher3 = rospy.Publisher('/habitat/depth3/image', Image, latch=True, queue_size=100)
            self.depth_publisher4 = rospy.Publisher('/habitat/depth4/image', Image, latch=True, queue_size=100)
            self.depth = Image()
            self.depth.is_bigendian = True
        else:
            self.publish_depth = False

        # Initialize semantic publisher
        """
        if semantic_topic is not None:
            self.publish_semantic = True
            self.semantic_publisher1 = rospy.Publisher('/semantic_mask1', Image, latch=True, queue_size=100)
            self.semantic_publisher2 = rospy.Publisher('/semantic_mask2', Image, latch=True, queue_size=100)
            self.semantic_publisher3 = rospy.Publisher('/semantic_mask3', Image, latch=True, queue_size=100)
            self.semantic_mask = Image()
            self.semantic_mask.is_bigendian = True
            self.semantic_mask_publisher = SemanticMaskPublisher()
        else:
            self.publish_semantic = False
        """

        # Initialize position publisher
        print('TRUE POSE TOPIC:', true_pose_topic)
        if true_pose_topic is not None:
            self.publish_true_pose = True
            self.pose_publisher = rospy.Publisher(true_pose_topic, PoseStamped, latch=True, queue_size=100)
            self.transform_publisher = rospy.Publisher('/habitat/transform_stamped', TransformStamped, latch=True, queue_size=100)
            self.tfbr = tf.TransformBroadcaster()
        else:
            self.publish_true_pose = False


    def publish(self, observations, cur_time):

        # Publish RGB image
        if self.publish_rgb:
            self.image = self.cvbridge.cv2_to_imgmsg(observations['rgb1'])
            self.image.encoding = 'rgb8'
            self.image.header.stamp = cur_time
            self.image.header.frame_id = 'camera_link1_noised'
            self.image_publisher1.publish(self.image)

            self.image = self.cvbridge.cv2_to_imgmsg(observations['rgb2'])
            self.image.encoding = 'rgb8'
            self.image.header.stamp = cur_time
            self.image.header.frame_id = 'camera_link2_noised'
            self.image_publisher2.publish(self.image)

            self.image = self.cvbridge.cv2_to_imgmsg(observations['rgb3'])
            self.image.encoding = 'rgb8'
            self.image.header.stamp = cur_time
            self.image.header.frame_id = 'camera_link3_noised'
            self.image_publisher3.publish(self.image)

            self.image = self.cvbridge.cv2_to_imgmsg(observations['rgb4'])
            self.image.encoding = 'rgb8'
            self.image.header.stamp = cur_time
            self.image.header.frame_id = 'camera_link4_noised'
            self.image_publisher4.publish(self.image)

        # Publish depth image
        if self.publish_depth:
            depth = observations['depth1'] * 7800 + 200
            #depth = observations['depth'] * 10000
            depth = depth.astype(np.uint16)
            depth[depth == 8000] = 0
            depth[depth == 200] = 0
            self.depth = self.cvbridge.cv2_to_imgmsg(depth)
            self.depth.header.stamp = cur_time
            self.depth.header.frame_id = 'camera_link1'
            self.depth_publisher1.publish(self.depth)

            depth = observations['depth2'] * 7800 + 200
            #depth = observations['depth'] * 10000
            depth = depth.astype(np.uint16)
            depth[depth == 8000] = 0
            depth[depth == 200] = 0
            self.depth = self.cvbridge.cv2_to_imgmsg(depth)
            self.depth.header.stamp = cur_time
            self.depth.header.frame_id = 'camera_link2'
            self.depth_publisher2.publish(self.depth)

            depth = observations['depth3'] * 7800 + 200
            #depth = observations['depth'] * 10000
            depth = depth.astype(np.uint16)
            depth[depth == 8000] = 0
            depth[depth == 200] = 0
            self.depth = self.cvbridge.cv2_to_imgmsg(depth)
            self.depth.header.stamp = cur_time
            self.depth.header.frame_id = 'camera_link3'
            self.depth_publisher3.publish(self.depth)

            depth = observations['depth4'] * 7800 + 200
            #depth = observations['depth'] * 10000
            depth = depth.astype(np.uint16)
            depth[depth == 8000] = 0
            depth[depth == 200] = 0
            self.depth = self.cvbridge.cv2_to_imgmsg(depth)
            self.depth.header.stamp = cur_time
            self.depth.header.frame_id = 'camera_link4'
            self.depth_publisher4.publish(self.depth)

        # Publish semantic mask
        """
        if self.publish_semantic:
            image = observations['rgb1']
            image = cv2.resize(image, (320, 240))
            semantic_mask = self.semantic_mask_publisher.process_image(image)
            self.semantic = self.cvbridge.cv2_to_imgmsg(image)
            self.semantic.header.stamp = cur_time
            self.semantic.header.frame_id = 'camera_link1'
            self.semantic_publisher1.publish(self.semantic)

            image = observations['rgb2']
            image = cv2.resize(image, (320, 240))
            semantic_mask = self.semantic_mask_publisher.process_image(image)
            self.semantic = self.cvbridge.cv2_to_imgmsg(image)
            self.semantic.header.stamp = cur_time
            self.semantic.header.frame_id = 'camera_link2'
            self.semantic_publisher2.publish(self.semantic)

            image = observations['rgb3']
            image = cv2.resize(image, (320, 240))
            semantic_mask = self.semantic_mask_publisher.process_image(image)
            self.semantic = self.cvbridge.cv2_to_imgmsg(image)
            self.semantic.header.stamp = cur_time
            self.semantic.header.frame_id = 'camera_link3'
            self.semantic_publisher3.publish(self.semantic)
        """

        # Publish camera info
        if self.publish_camera_info:
            self.camera_info.header.stamp = cur_time
            self.camera_info_publisher.publish(self.camera_info)

        # Publish true pose
        if self.publish_true_pose:
            x, y = observations['gps']
            cur_z_angle = observations['compass'][0]
            cur_pose = PoseStamped()
            cur_pose.header.stamp = cur_time
            cur_pose.header.frame_id = 'map'
            cur_pose.pose.position.x = x
            cur_pose.pose.position.y = -y
            cur_pose.pose.position.z = 0.1#observations['agent_position'][0][1]
            cur_pose.pose.orientation.x, \
            cur_pose.pose.orientation.y, \
            cur_pose.pose.orientation.z, \
            cur_pose.pose.orientation.w = tf.transformations.quaternion_from_euler(0, 0, cur_z_angle)
            #print('Pose at time {} is ({}, {}, {})'.format(cur_time.to_sec(), x, -y, cur_z_angle))
            self.tfbr.sendTransform((x, -y, observations['agent_position'][0][1]),
                                    tf.transformations.quaternion_from_euler(0, 0, cur_z_angle),
                                    cur_time,
                                    'base_link', 'map')
            self.pose_publisher.publish(cur_pose)

            cur_transform = TransformStamped()
            cur_transform.header = cur_pose.header
            cur_transform.child_frame_id = 'base_link'
            cur_transform.transform.translation = cur_pose.pose.position
            cur_transform.transform.rotation = cur_pose.pose.orientation
            self.transform_publisher.publish(cur_transform)
