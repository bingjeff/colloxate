import jax.numpy as jnp
from typing import Sequence


def make_identity_pose() -> Sequence[float]:
    """Create the identity representation for a pose.

    :return: Array of elements [tx, ty, tz, rw, rx, ry, rz].
    """
    return jnp.array([0., 0., 0., 1.0, 0., 0., 0.])


def make_pose(translation: Sequence[float], rotation: float, axis: Sequence[float]) -> Sequence[float]:
    """Create a pose object from origin, angle and axis.

    :param translation: The origin of the pose, [tx, ty, tz].
    :param rotation: The angle of rotation in radians.
    :param axis: The axis of rotation, [ax, ay, az].
    :return: Pose array of elements, [tx, ty, tz, rw, rx, ry, rz].
    """
    cq = jnp.cos(0.5 * rotation)
    sq = jnp.sin(0.5 * rotation)
    return jnp.array([translation[0], translation[1], translation[2], cq, sq * axis[0], sq * axis[1], sq * axis[2]])


def multiply(pose_left: Sequence[float], pose_right: Sequence[float]) -> Sequence[float]:
    """Basic multiplication between pose objects for transformation.

    Writes out the basic quaternion math with the least number of operations.

    :param pose_left: pose with parent coordinates.
    :param pose_right: pose with child coordinates.
    :return: Pose array of elements, [tx, ty, tz, rw, rx, ry, rz].
    """
    ltx, lty, ltz, lqw, lqx, lqy, lqz = pose_left
    rtx, rty, rtz, rqw, rqx, rqy, rqz = pose_right
    tw = -lqx * rtx - lqy * rty - lqz * rtz
    tx = lqw * rtx + lqy * rtz - lqz * rty
    ty = lqw * rty - lqx * rtz + lqz * rtx
    tz = lqw * rtz + lqx * rty - lqy * rtx

    tx = -tw * lqx + tx * lqw - ty * lqz + tz * lqy + ltx
    ty = -tw * lqy + tx * lqz + ty * lqw - tz * lqx + lty
    tz = -tw * lqz - tx * lqy + ty * lqx + tz * lqw + ltz

    qw = lqw * rqw - lqx * rqx - lqy * rqy - lqz * rqz
    qx = lqw * rqx + lqx * rqw + lqy * rqz - lqz * rqy
    qy = lqw * rqy - lqx * rqz + lqy * rqw + lqz * rqx
    qz = lqw * rqz + lqx * rqy - lqy * rqx + lqz * rqw
    return jnp.array([tx, ty, tz, qw, qx, qy, qz])
