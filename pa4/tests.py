import numpy as np
import math

from scipy.misc import imread

from util import preprocess_ncc, compute_ncc, project, unproject_corners, \
    pyrdown, pyrup, compute_photometric_stereo

# handle Python 3
import sys
if sys.version_info[0] >= 3:
    xrange = range

def skip_not_implemented(func):
    from nose.plugins.skip import SkipTest

    def wrapper():
        try:
            func()
        except NotImplementedError as exc:
            raise SkipTest(
                "Test {0} is skipped {1}".format(func.__name__, exc))
    wrapper.__name__ = func.__name__
    return wrapper


@skip_not_implemented
def preprocess_ncc_zeros_test():
    ncc_size = 5

    image = np.zeros((2 * ncc_size - 1, 2 * ncc_size - 1, 3), dtype=np.float32)
    n = preprocess_ncc(image, ncc_size)

    assert n.shape == (2 * ncc_size - 1, 2 * ncc_size -
                       1, 3 * ncc_size * ncc_size)
    assert (np.abs(n) < 1e-6).all()


@skip_not_implemented
def preprocess_ncc_delta_test():
    ncc_size = 5
    ncc_half = ncc_size // 2

    image = np.zeros((2 * ncc_size - 1, 2 * ncc_size - 1, 3), dtype=np.float32)
    image[ncc_size - 1, ncc_size - 1, :] = ncc_size ** 2
    n = preprocess_ncc(image, ncc_size)

    correct = np.zeros((2 * ncc_size - 1, 2 * ncc_size - 1,
                        3 * ncc_size ** 2), dtype=np.float32)
    correct[ncc_half:-ncc_half, ncc_half:-ncc_half, :] = - \
        1.0 / (ncc_size * math.sqrt(3 * ncc_size ** 2 - 3))
    x = (ncc_size ** 2 - 1.0) / (ncc_size * math.sqrt(3 * ncc_size ** 2 - 3))
    for i in xrange(ncc_size):
        for j in xrange(ncc_size):
            correct[-(i + ncc_half + 1), -(j + ncc_half + 1), ncc_size **
                    2 * 0 + ncc_size * i + j] = x
            correct[-(i + ncc_half + 1), -(j + ncc_half + 1), ncc_size **
                    2 * 1 + ncc_size * i + j] = x
            correct[-(i + ncc_half + 1), -(j + ncc_half + 1), ncc_size **
                    2 * 2 + ncc_size * i + j] = x

    assert n.shape == (2 * ncc_size - 1, 2 * ncc_size -
                       1, 3 * ncc_size * ncc_size)
    assert (np.abs(n - correct) < 1e-6).all()

@skip_not_implemented
def offset_and_scale_ncc_test():
    ncc_size = 5
    ncc_half = ncc_size // 2

    image1 = np.random.random((2 * ncc_size - 1, 2 * ncc_size - 1, 3))
    image2 = image1 * 2 + 3

    n1 = preprocess_ncc(image1, ncc_size)
    n2 = preprocess_ncc(image2, ncc_size)

    ncc = compute_ncc(n1, n2)

    assert ncc.shape == (2 * ncc_size - 1, 2 * ncc_size - 1)
    assert (np.abs(ncc[:ncc_half, :]) < 1e-6).all()
    assert (np.abs(ncc[-ncc_half:, :]) < 1e-6).all()
    assert (np.abs(ncc[:, :ncc_half]) < 1e-6).all()
    assert (np.abs(ncc[:, -ncc_half:]) < 1e-6).all()
    assert (
        np.abs(ncc[ncc_half:-ncc_half, ncc_half:-ncc_half] - 1) < 1e-6).all()


@skip_not_implemented
def project_Rt_identity_centered_test():
    width = 1
    height = 1
    f = 1

    K = np.array((
                 (f, 0, width / 2.0),
                 (0, f, height / 2.0),
                 (0, 0, 1)
                 ))

    Rt = np.zeros((3, 4), dtype=np.float32)
    Rt[:, :3] = np.identity(3)

    point = np.array(((0, 0, 1), ), dtype=np.float32).reshape((1, 1, 3))

    projection = project(K, Rt, point)

    assert projection.shape == (1, 1, 2)
    assert projection[0][0][0] == width / 2.0
    assert projection[0][0][1] == height / 2.0

@skip_not_implemented
def unproject_Rt_identity_2x2_2xdepth_test():
    width = 2
    height = 2
    f = 1

    K = np.array((
                 (f, 0, width / 2.0),
                 (0, f, height / 2.0),
                 (0, 0, 1)
                 ))

    Rt = np.zeros((3, 4), dtype=np.float32)
    Rt[:, :3] = np.identity(3)

    depth = 2
    point = unproject_corners(K, width, height, depth, Rt)

    assert point.shape == (2, 2, 3)

    assert point[0, 0, 0] == -2
    assert point[0, 0, 1] == -2

    assert point[0, 1, 0] == 2
    assert point[0, 1, 1] == -2

    assert point[1, 0, 0] == -2
    assert point[1, 0, 1] == 2

    assert point[1, 1, 0] == 2
    assert point[1, 1, 1] == 2

    assert (point[:, :, 2] == 2).all()


@skip_not_implemented
def project_unproject_Rt_identity_test():
    width = 20
    height = 10
    f = 1

    K = np.array((
                 (f, 0, width / 2.0),
                 (0, f, height / 2.0),
                 (0, 0, 1)
                 ))

    Rt = np.zeros((3, 4), dtype=np.float32)
    Rt[:, :3] = np.identity(3)

    depth = 1
    point = unproject_corners(K, width, height, depth, Rt)

    projection = project(K, Rt, point)

    assert projection.shape == (2, 2, 2)

    assert np.abs(projection[0, 0, 0]) < 1e-5
    assert np.abs(projection[0, 0, 1]) < 1e-5

    assert np.abs(projection[0, 1, 0] - width) < 1e-5
    assert np.abs(projection[0, 1, 1]) < 1e-5

    assert np.abs(projection[1, 0, 0]) < 1e-5
    assert np.abs(projection[1, 0, 1] - height) < 1e-5

    assert np.abs(projection[1, 1, 0] - width) < 1e-5
    assert np.abs(projection[1, 1, 1] - height) < 1e-5

@skip_not_implemented
def pyrdown_even_test():
    height = 16
    width = 16

    image = np.ones((height, width, 3), dtype=np.float32)

    down = pyrdown(image)

    assert down.shape == (height / 2, width / 2, 3)
    assert (down == 1).all()


@skip_not_implemented
def pyrdown_odd_test():
    height = 17
    width = 17

    image = np.ones((height, width, 3), dtype=np.float32)

    down = pyrdown(image)

    assert down.shape == ((height + 1) / 2, (width + 1) / 2, 3)
    assert (down == 1).all()


@skip_not_implemented
def pyrdown_nonsquare_test():
    height = 16
    width = 31

    image = np.ones((height, width, 3), dtype=np.float32)

    down = pyrdown(image)

    assert down.shape == ((height + 1) // 2, (width + 1) // 2, 3)
    assert (down == 1).all()

@skip_not_implemented
def pyrup_pyrdown_test():
    height = 16
    width = 31

    image = np.random.random((height, width, 3))

    up = pyrup(image)

    assert up.shape == (2 * height, 2 * width, 3)

    down = pyrdown(up)

    assert down.shape == (height, width, 3)


@skip_not_implemented
def compute_photometric_stereo_test():
    image1 = np.array((1,), dtype=np.float32).reshape(1, 1, 1)
    image2 = np.array((0,), dtype=np.float32).reshape(1, 1, 1)
    image3 = np.array((0,), dtype=np.float32).reshape(1, 1, 1)

    images = [image1, image2, image3]

    lights = np.array((
                      (0, 1, 0),
                      (0, 0, 1),
                      (1, 0, 0)
                      ), dtype=np.float32)

    albedo, normals = compute_photometric_stereo(lights, images)

    assert albedo.shape == (1, 1, 1)
    assert normals.shape == (1, 1, 3)

    assert np.allclose(albedo[0, 0, 0], 1)
    assert (normals[0, 0, :] == (0, 0, 1)).all()


@skip_not_implemented
def compute_photometric_stereo_half_albedo_test():
    image1 = np.array((0.5,), dtype=np.float32).reshape(1, 1, 1)
    image2 = np.array((0,), dtype=np.float32).reshape(1, 1, 1)
    image3 = np.array((0,), dtype=np.float32).reshape(1, 1, 1)

    images = [image1, image2, image3]

    lights = np.array((
                      (0, 1, 0),
                      (0, 0, 1),
                      (1, 0, 0)
                      ), dtype=np.float32)

    albedo, normals = compute_photometric_stereo(lights, images)

    assert albedo.shape == (1, 1, 1)
    assert normals.shape == (1, 1, 3)

    assert np.allclose(albedo[0, 0, 0], 0.5)
    assert (normals[0, 0, :] == (0, 0, 1)).all()
