# import jax.numpy as jnp
import numpy as jnp
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


def axis_angle_to_quaternion(rotation: float, axis: Sequence[float]) -> Sequence[float]:
    """Create a quaternion from an axis-angle representation.

    :param rotation: the angle of rotation in radians.
    :param axis: the axis of rotation [ax, ay, az].
    :return: a quaternion [w, x, y, z]
    """
    cq = jnp.cos(0.5 * rotation)
    sq = jnp.sin(0.5 * rotation)
    return jnp.array([cq, sq * axis[0], sq * axis[1], sq * axis[2]])


def hamilton_product(quaternion_left: Sequence[float], quaternion_right: Sequence[float]) -> Sequence[float]:
    """Compute the Hamilton product between two quaternions.

    :param quaternion_left: left-hand-side quaternion, [lw, lx, ly, lz].
    :param quaternion_right: right-hand-side quaternion, [rw, rx, ry, rz].
    :return: Hamilton product of two quaternions.
    """
    lw, lx, ly, lz = quaternion_left
    rw, rx, ry, rz = quaternion_right
    w = lw * rw - lx * rx - ly * ry - lz * rz
    x = lw * rx + lx * rw + ly * rz - lz * ry
    y = lw * ry - lx * rz + ly * rw + lz * rx
    z = lw * rz + lx * ry - ly * rx + lz * rw
    return jnp.array([w, x, y, z])


def vector_rotate(quaternion: Sequence[float], vector: Sequence[float]) -> Sequence[float]:
    """Rotates the vector by the quaternion, per q*v*inv(q).

    :param quaternion: left-hand-side quaternion, [rw, rx, ry, rz].
    :param vector: right-hand-side vector to rotate, [tx, ty, tz].
    :return: rotated vector [x, y, z]
    """
    lw, lx, ly, lz = quaternion
    tx, ty, tz = vector
    rw =  tx * lx + ty * ly + tz * lz
    rx =  tx * lw - ty * lz + tz * ly
    ry =  tx * lz + ty * lw - tz * lx
    rz = -tx * ly + ty * lx + tz * lw

    x = lw * rx + lx * rw + ly * rz - lz * ry
    y = lw * ry - lx * rz + ly * rw + lz * rx
    z = lw * rz + lx * ry - ly * rx + lz * rw
    return jnp.array([x, y, z])


def quaternion_conjugate(quaternion: Sequence[float]) -> Sequence[float]:
    """Compute the conjugate of a quaternion.

    :param quaternion: the quaternion [w, x, y, z].
    :return: the conjugate [w, -x, -y, -z].
    """
    w, x, y, z = quaternion
    return jnp.array([w, -x, -y, -z])


def quaternion_product(quaternion_left: Sequence[float], quaternion_right: Sequence[float]) -> Sequence[float]:
    """Compute the quaternion product between two quaternions.

    :param quaternion_left: left-hand-side quaternion, [lw, lx, ly, lz].
    :param quaternion_right: right-hand-side quaternion, [rw, rx, ry, rz].
    :return: quaternion product of two quaternions.
    """
    # (s + v) (t + w) = (s*t - dot(v, w)) + (s*w + t*v + cross(v, w))
    w = quaternion_left[0] * quaternion_right[0] - jnp.dot(quaternion_left[1:], quaternion_right[1:])
    xyz = quaternion_left[0] * quaternion_right[1:]
    xyz += quaternion_right[0] * quaternion_left[1:]
    xyz += jnp.cross(quaternion_left[1:], quaternion_right[1:])
    return jnp.array([w, xyz[0], xyz[1], xyz[2]])


def multiply(pose_left: Sequence[float], pose_right: Sequence[float]) -> Sequence[float]:
    """Basic multiplication between pose objects for transformation.

    Writes out the basic quaternion math with the least number of operations.

    :param pose_left: pose with parent coordinates.
    :param pose_right: pose with child coordinates.
    :return: Pose array of elements, [tx, ty, tz, rw, rx, ry, rz].
    """
    ltx, lty, ltz, lqw, lqx, lqy, lqz = pose_left
    rtx, rty, rtz, rqw, rqx, rqy, rqz = pose_right

    # Rotate the rhs vector.
    rw =  rtx * lqx + rty * lqy + rtz * lqz
    rx =  rtx * lqw - rty * lqz + rtz * lqy
    ry =  rtx * lqz + rty * lqw - rtz * lqx
    rz = -rtx * lqy + rty * lqx + rtz * lqw

    tx = lqw * rx + lqx * rw + lqy * rz - lqz * ry
    ty = lqw * ry - lqx * rz + lqy * rw + lqz * rx
    tz = lqw * rz + lqx * ry - lqy * rx + lqz * rw

    # Offset the rhs vector.
    tx += ltx
    ty += lty
    tz += ltz

    # Rotate the rhs quaternion.
    qw = lqw * rqw - lqx * rqx - lqy * rqy - lqz * rqz
    qx = lqw * rqx + lqx * rqw + lqy * rqz - lqz * rqy
    qy = lqw * rqy - lqx * rqz + lqy * rqw + lqz * rqx
    qz = lqw * rqz + lqx * rqy - lqy * rqx + lqz * rqw

    return jnp.array([tx, ty, tz, qw, qx, qy, qz])
