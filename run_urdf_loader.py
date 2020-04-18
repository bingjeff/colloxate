from absl import app
from absl import flags

import urdf_loader

import jax.numpy as jnp

FLAGS = flags.FLAGS
flags.DEFINE_string("urdf_path", "data/kuka_iiwa.urdf",
                    "Path of URDF to load.")
flags.DEFINE_string("root_link", "lbr_iiwa_link_0",
                    "Path of URDF to load.")
flags.DEFINE_string("tip_link", "lbr_iiwa_link_7",
                    "Path of URDF to load.")


def main(argv):
    del argv  # Unused.
    print("Loading: {}".format(FLAGS.urdf_path))

    chain = urdf_loader.read_chain_from_urdf(
        FLAGS.urdf_path, FLAGS.root_link, FLAGS.tip_link)
    for joint in chain:
        print('{}: {}'.format(joint.get('name'), joint.get('type')))
    kinematics = urdf_loader.make_kinematic_chain_function(chain)
    zero_pose = jnp.array([0., 0., 0., 0., 0., 0., 0.])
    print('zero pose: {}'.format(kinematics(zero_pose)))
    bent_pose = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    print('bent pose: {}'.format(kinematics(bent_pose)))


if __name__ == '__main__':
    app.run(main)
