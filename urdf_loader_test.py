import unittest

import numpy as np
import pose
import urdf_loader

THREE_JOINT_PATH = 'data/three_joint.urdf'
THREE_JOINT_ROOT = 'link_0'
THREE_JOINT_TIP = 'link_3'


class UrdfLoaderTest(unittest.TestCase):
    def setUp(self):
        self.chain = urdf_loader.read_chain_from_urdf(THREE_JOINT_PATH, THREE_JOINT_ROOT, THREE_JOINT_TIP)
        self.fk_fun = urdf_loader.make_kinematic_chain_function(self.chain)

    def test_zero_pose(self):
        joint_angles = [0, 0, 0]
        expected_pose = pose.make_pose([0, 0, 1.1], 0, [0, 0, 1])
        np.testing.assert_almost_equal(list(expected_pose), list(self.fk_fun(joint_angles)))

    def test_pitch_pose(self):
        joint_angles = [0, 0.5 * np.pi, 0]
        expected_pose = pose.make_pose([1.0, 0, 0.1], 0.5 * np.pi, [0, 1, 0])
        np.testing.assert_almost_equal(list(expected_pose), list(self.fk_fun(joint_angles)))

    def test_pan_pitch_pose(self):
        joint_angles = [0.5 * np.pi, 0.5 * np.pi, -0.5 * np.pi]
        expected_pose = pose.make_pose([0, 1.0, 0.1], -0.5 * np.pi, [1, 0, 0])
        np.testing.assert_almost_equal(list(expected_pose), list(self.fk_fun(joint_angles)))


if __name__ == '__main__':
    unittest.main()
