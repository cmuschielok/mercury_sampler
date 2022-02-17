from typing import Optional, Union

import numpy as np
# This import statement is unused but is nevertheless needed!
# We repeat it here for convenience, should it ever disappear:
# import quaternion
import quaternion
from numpy.random import Generator, default_rng


def random_quaternions(sample_size: int = 1000, rng: Optional[Generator] = None,
                       rng_params: Optional[dict] = None) -> np.ndarray:
    """Returns an array of normally distributed unit quaternions.

    Args:
        sample_size: int
        The number of quaternions to be generated.

        rng: Generator, optional
        The random number generator to be used.

        rng_params: dict, optional
        Parameters passed to the random number generator `rng`.

    Returns:
        quaternions: (N, 4) ndarray

    """
    # Deal with the random number generator.
    if rng_params is None:
        rng_params = dict()
    if rng is None:
        rng = default_rng(**rng_params)
    # Obtain a (sample_size, 4) random array with rows to be interpreted as R4-vectors
    # which cover the hypersphere S4.
    points_on_hypersphere = rng.standard_normal(size=(sample_size, 4))
    norm = np.linalg.norm(points_on_hypersphere, axis=1)
    points_on_hypersphere = [p/n for p, n in zip(points_on_hypersphere, norm)]
    # Build quaternions from these points
    quaternions = np.array([np.quaternion(*p) for p in points_on_hypersphere])
    return quaternions


def rotate_vector_by_quaternion(points: np.ndarray, quat: np.quaternion,
                                tolerance: float = 1.0e-6) -> np.ndarray:
    """Use a quaternion to rotate a vector by an angle around an axis.

    Args:
        points: (N, 3) ndarray

        quat: quaternion
        A unit quaternion.

        tolerance: float, default=1.0e-6
        Allowed deviation of the scalar part of the `rotated_points` from zero,
        when `rotated_points` are still quaternions.

    Returns:
        rotated_points: (N, 3) ndarray

    Raises:
        ValueError:
        If the absolute value of the scalar part of any quaternion in
        `rotated_points_quat` is greater than `tolerance`. Ideally, those should
        be pure quaternions, as we use their vector parts as points in
        Euclidean 3-space.

    Author:
        Christoph Muschielok (c.muschielok@tum.de)

    """
    # Make sure that only unit quaternions are used.
    quat = quat.normalized()
    # ***
    # Why did I do this like that? I can just normalize and continue. Also, the
    # assertion is not good coding style in this case.
    # err_str = "This has to be a *unit* quaternion! Found norm: {}"
    # assert (quat - quat.normalized()).norm() < tolerance, err_str.format(quat.norm())
    # ***
    # Make quaternions from the vectors.
    points_quat = np.asarray([np.quaternion(0.0, *point)
                              for point in points
                              ], dtype=np.quaternion)
    # Rotate
    rotated_points_quat = quat.reciprocal() * points_quat * quat
    _tol_conditions = np.abs([r.components[0] for r in rotated_points_quat]) > tolerance
    if any(_tol_conditions):
        _msg = "This rotation resulted in at least one quaternion with first component " \
               "non-zero and above tolerance! Indices: {}"
        raise ValueError(_msg.format(np.where(_tol_conditions)[0]))
    rotated_points = np.array([r.components[1:] for r in rotated_points_quat],
                              dtype=float)
    return rotated_points


def rotate_and_shift(quat: Union[np.quaternion, None],
                     trans: np.ndarray,
                     points: np.ndarray) -> np.ndarray:
    """Rotate points using a quaternion and shift.

    Args:
        quat: Union[quaternion, None]
        A unit quaternion describing an axis and an angle around which a point
        is rotated. If this is explicitly None, only a translation is done.

        trans: (3, ) ndarray
        A 3-vector by which a point is shifted.

        points: (N, 3) ndarray
        An array of points which shall be transformed.

    Returns:
        transformed_points: (N, 3) ndarray
        The corresponding array of transformed points.

    Author:
        Christoph Muschielok (c.muschielok@tum.de)

    """
    if quat is None:
        rotated_points = points
    else:
        rotated_points = rotate_vector_by_quaternion(points, quat)
    transformed_points = np.asarray([r - trans for r in rotated_points], dtype=float)
    return transformed_points
