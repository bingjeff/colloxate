import jax.numpy as jnp


def MakePose(translation, rotation, axis):
    cq = jnp.cos(0.5 * rotation)
    sq = jnp.sin(0.5 * rotation)
    return jnp.array([translation[0], translation[1], translation[2], cq, sq * axis[0], sq * axis[1], sq * axis[2]])


def MultiplyPoses(p_left, p_right):
    ltx, lty, ltz, lqw, lqx, lqy, lqz = p_left
    rtx, rty, rtz, rqw, rqx, rqy, rqz = p_right
    tw = -lqx*rtx - lqy*rty - lqz*rtz
    tx = lqw*rtx + lqy*rtz - lqz*rty
    ty = lqw*rty - lqx*rtz + lqz*rtx
    tz = lqw*rtz + lqx*rty - lqy*rtx

    tx = -tw*lqx + tx*lqw - ty*lqz + tz*lqy + ltx
    ty = -tw*lqy + tx*lqz + ty*lqw - tz*lqx + lty
    tz = -tw*lqz - tx*lqy + ty*lqx + tz*lqw + ltz

    qw = lqw*rqw - lqx*rqx - lqy*rqy - lqz*rqz
    qx = lqw*rqx + lqx*rqw + lqy*rqz - lqz*rqy
    qy = lqw*rqy - lqx*rqz + lqy*rqw + lqz*rqx
    qz = lqw*rqz + lqx*rqy - lqy*rqx + lqz*rqw
    return jnp.array([tx, ty, tz, qw, qx, qy, qz])
