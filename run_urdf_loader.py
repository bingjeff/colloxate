from absl import app
from absl import flags

import urdf_loader

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

    chain = urdf_loader.ReadChainFromUrdf(
        FLAGS.urdf_path, FLAGS.root_link, FLAGS.tip_link)
    for joint in chain:
        print('{}: {}'.format(joint.get('name'), joint.get('type')))


if __name__ == '__main__':
    app.run(main)
