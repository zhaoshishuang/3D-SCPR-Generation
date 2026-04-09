import numpy as np
from scipy import interpolate
from skimage.transform import warp


def cumulative_distance(points):
    ex_points = np.concatenate([points[:1], points])
    diff = np.diff(ex_points, axis=0)
    dists = np.linalg.norm(diff, ord=2, axis=1)
    cum_dists = np.cumsum(dists)
    return cum_dists


def normalize(x):
    return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)


def vertical_vector(x, d):
    x = normalize(x)
    d = normalize(d)
    x = x - d * np.dot(x, d)
    return normalize(x)


def curve_smooth_and_resample(curve,
                              spacing=None,
                              sample_unit=0,
                              extra_length=0):
    """Perform smoothing and resampling for a vessel center curve.

    Parameters
    ----------
    curve : Nx3 array.
    spacing : None or array of shape (3, ). Spacing is used for
        calculating in world coordinate system.
    sample_unit : 0 or a positive float number (mm). If not 0, the
        value is used for resample the curve points.
    extra_length : 0 or uint (mm). The length is used to expand the
        curve.

    Returns
    ----------
    curve : Mx3 array, the same order with input curve.
    tangent :
    axis1 :
    axis2 :
    """
    if sample_unit < 0:
        raise ValueError('sample_unit should be posivite.')

    if extra_length < 0:
        raise ValueError('extra_length should be posivite.')

    if sample_unit == 0 and extra_length > 0:
        raise ValueError(
            'extra_length > 0 will affect only when sample_unit > 0.')

    if spacing is None:
        spacing = (1., 1., 1.)

    spacing = np.array(spacing)
    curve = curve * spacing

    w = np.ones(len(curve)) / np.mean(spacing)
    smoothness = len(curve) * np.mean(spacing)
    tck, u = interpolate.splprep([curve[:, i] for i in range(3)],
                                 k=3,
                                 w=w,
                                 s=smoothness)
    points = np.stack(interpolate.splev(u, tck), axis=-1)

    if sample_unit > 0:
        cum_dists = cumulative_distance(points)
        dists_target = np.arange(-extra_length,
                                 np.floor(cum_dists[-1]) + extra_length,
                                 sample_unit)
        spl1d = interpolate.splrep(cum_dists, u, k=1)
        u_new = interpolate.splev(dists_target, spl1d)
        points = np.stack(interpolate.splev(u_new, tck), axis=-1)
    else:
        u_new = u

    d1 = np.stack(interpolate.splev(u_new, tck, der=1), axis=-1)
    d1_ = normalize(d1)

    cur_direct = np.array([1., 0., 0.])
    axis1 = list()
    for d in d1_:
        cur_direct = vertical_vector(cur_direct, d)
        axis1.append(cur_direct)

    axis1 = np.stack(axis1)
    axis2 = np.cross(axis1, d1_)

    # points = points / spacing

    return points, d1_, axis1, axis2

def generate_straightened_3d_scpr(image,
                                 spacing,
                                 points,
                                 axis1,
                                 axis2,
                                 theta,
                                 sample_width,
                                 sample_unit,
                                 order=1):
    """Generate straightened cpr by pre-computed points and axes.

    Parameters
    ----------
    image : 3d array.
    spacing : None or array of shape (3, ). None indicates isotropous
    coords.
    points : Nx3 array (mm).
    axis1 : normalized axis
    axis2 : normalized axis
    theta : 0 ~ 2 * pi
    sample_width : uint (pixel)
    sample_unit : positive float (mm)
    order : int (0-5)

    Returns
    ----------
    straightened_3d_scpr : image of shape len(points) * sample_width * sample_width
    """
    gridx = (np.arange(0, sample_width) - sample_width / 2.0 +
             0.5) * sample_unit
    direc1 = axis1 * np.cos(theta) + axis2 * np.sin(theta)
    direc2 = axis2 * np.cos(theta + np.pi / 2.) + axis2 * np.sin(theta +
                                                                 np.pi / 2.)

    cordx = direc1[:, None, :] * gridx[None, :, None]
    cordy = direc2[:, None, :] * gridx[None, :, None]
    grid = points[:, None, None, :] + cordx[:, None, :, :] + cordy[:, :,
                                                                   None, :]

    grid = grid.reshape((-1, sample_width * sample_width, 3))

    if spacing is not None:
        grid_ori = grid / np.array(spacing)
    else:
        grid_ori = grid
    straightened_3d_scpr = warp(
        image, np.moveaxis(grid_ori, -1, 0), order=order)
    straightened_3d_scpr = straightened_3d_scpr.reshape(
        (-1, sample_width, sample_width))

    return straightened_3d_scpr, grid_ori