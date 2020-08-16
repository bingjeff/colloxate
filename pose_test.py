import unittest

import numpy as np
import pose


class PoseTest(unittest.TestCase):
    def test_multiply(self):
        rotate_90 = pose.make_pose([0, 1, 0], 0.5 * np.pi, [0, 0, 1])
        x_pose = pose.make_pose([1, 0, 0], 0, [0, 0, 1])
        expected_pose = pose.make_pose([0, 2, 0], 0.5 * np.pi, [0, 0, 1])
        np.testing.assert_almost_equal(list(expected_pose), list(pose.multiply(rotate_90, x_pose)))

    def test_hamilton_product(self):
        q37 = pose.axis_angle_to_quaternion(np.deg2rad(37), [0, 0, 1])
        q91 = pose.axis_angle_to_quaternion(np.deg2rad(91), [0, 0, 1])
        q128 = pose.axis_angle_to_quaternion(np.deg2rad(128), [0, 0, 1])
        np.testing.assert_almost_equal(list(q128), list(pose.hamilton_product(q37, q91)))

    def test_quaternion_product(self):
        q37 = pose.axis_angle_to_quaternion(np.deg2rad(37), [0, 0, 1])
        q91 = pose.axis_angle_to_quaternion(np.deg2rad(91), [0, 0, 1])
        q128 = pose.axis_angle_to_quaternion(np.deg2rad(128), [0, 0, 1])
        np.testing.assert_almost_equal(list(q128), list(pose.quaternion_product(q37, q91)))

    def test_vector_rotation_by_quaternion_product(self):
        rotate_90 = pose.axis_angle_to_quaternion(np.deg2rad(90), [0, 0, 1])
        x_axis = np.array([0, 1.0, 0, 0])
        y_axis = np.array([0, 0, 1.0, 0])
        rotated_vector = pose.quaternion_product(rotate_90, pose.quaternion_product(x_axis, pose.quaternion_conjugate(rotate_90)))
        np.testing.assert_almost_equal(list(y_axis), list(rotated_vector))

    def test_vector_rotation_by_hamilton_product(self):
        rotate_90 = pose.axis_angle_to_quaternion(np.deg2rad(90), [0, 0, 1])
        x_axis = np.array([0, 1.0, 0, 0])
        y_axis = np.array([0, 0, 1.0, 0])
        rotated_vector = pose.hamilton_product(rotate_90, pose.hamilton_product(x_axis, pose.quaternion_conjugate(rotate_90)))
        np.testing.assert_almost_equal(list(y_axis), list(rotated_vector))

    def test_vector_rotation(self):
        rotate_90 = pose.axis_angle_to_quaternion(np.deg2rad(90), [0, 0, 1])
        x_axis = np.array([1.0, 0, 0])
        y_axis = np.array([0, 1.0, 0])
        rotated_vector = pose.vector_rotate(rotate_90, x_axis)
        np.testing.assert_almost_equal(list(y_axis), list(rotated_vector))


if __name__ == '__main__':
    unittest.main()