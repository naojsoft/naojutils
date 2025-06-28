#
# rot.py -- rotation calculation functions
#
#  E. Jeschke
#

# 3rd party
import numpy as np

# mount angle offsets for certain instruments
mount_offsets = dict(FOCAS=0.259, MOIRCS=45.0)
# whether PA should be flipped
mount_flip = dict(FOCAS=True, MOIRCS=True)


def calc_alternate_angle(ang_deg):
    """Calculates the alternative usable angle to the given one.

    Parameters
    ----------
    ang_deg : float or array of float
        The input angle(s) in degrees

    Returns
    -------
    alt_deg : float or array of float
        The output angle(s) in degrees
    """
    alt_deg = ang_deg - np.sign(ang_deg) * 360
    return alt_deg


def normalize_angle(ang_deg, limit=None, ang_offset=0.0):
    """Normalize an angle.

    Parameters
    ----------
    az_deg: float
        A traditional azimuth value where 0 deg == North

    limit: str or None (optional, defaults to None)
        How to limit the range of the result angle

    ang_offset: float (optional, defaults to 0.0)
        Angle to add to the input angle to offset it

    Returns
    -------
    limit: None (-360, 360), 'full' (0, 360), or 'half' (-180, 180)

    To normalize to Subaru azimuth (AZ 0 == S), do
        normalize_angle(ang_deg, limit='half', ang_offset=-180)
    """
    # convert to array if just a scalar
    is_array = isinstance(ang_deg, np.ndarray)
    if not is_array:
        ang_deg = np.array([ang_deg])

    ang_deg = ang_deg + ang_offset

    # constrain to -360, +360
    mask = np.fabs(ang_deg) >= 360.0
    ang_deg[mask] = np.remainder(ang_deg[mask], np.sign(ang_deg[mask]) * 360.0)
    if limit is not None:
        # constrain to 0, +360
        mask = ang_deg < 0.0
        ang_deg[mask] += 360.0
        if limit == 'half':
            # constrain to -180, +180
            mask = ang_deg > 180.0
            ang_deg[mask] -= 360.0

    if not is_array:
        # extract scalar
        ang_deg = ang_deg[0]

    return ang_deg


def check_rotation_limits(rot_start_deg, rot_stop_deg, min_rot_deg, max_rot_deg):
    """Check rotation against limits.

    Parameters
    ----------
    rot_start : float or array of float, NaNs ok
        Rotation start value(s). NaN indicates a non-available angle.

    rot_stop : float or array of float, NaNs ok
        Rotation stop value(s). NaN indicates a non-available angle.

    min_rot_deg : float
        Minimum rotation value

    max_rot_deg : float
        Maximum rotation value

    Returns
    -------
    rot_ok : bool
        True if rotation is allowed, False otherwise
    """
    is_array = isinstance(rot_start_deg, np.ndarray)
    if not is_array:
        rot_start_deg = np.array([rot_start_deg])
        rot_stop_deg = np.array([rot_stop_deg])

    rot_ok = np.logical_and(np.isfinite(rot_start_deg),
                            np.isfinite(rot_stop_deg))
    rot_ok = np.logical_and(rot_ok, min_rot_deg <= rot_start_deg)
    rot_ok = np.logical_and(rot_ok, rot_start_deg <= max_rot_deg)
    rot_ok = np.logical_and(rot_ok, min_rot_deg <= rot_stop_deg)
    rot_ok = np.logical_and(rot_ok, rot_stop_deg <= max_rot_deg)

    if not is_array:
        rot_ok = rot_ok[0]
    return rot_ok


def calc_optimal_rotation(left_start_deg, left_stop_deg,
                          right_start_deg, right_stop_deg,
                          cur_rot_deg, min_rot_deg, max_rot_deg):
    """Find optimal rotation, while checking against limits.

    Parameters
    ----------
    left_start_deg : float array of float, NaNs ok
        Rotation possibility 1 start value(s)

    left_stop_deg : float or array of float, NaNs ok
        Rotation possibility 1 stop value(s)

    right_start_deg : float or array of float, NaNs ok
        Rotation possibility 2 start value(s)

    right_stop_deg : float or array of float, NaNs ok
        Rotation possibility 2 stop value(s)

    cur_rot_deg : float
        Current rotation value

    min_rot_deg : float
        Minimum rotation value

    max_rot_deg : float
        Maximum rotation value

    Returns
    -------
    rot_start, rot_stop : start and stop rotation values
        floats if rotation is allowed, NaN otherwise
    """
    left_ok = check_rotation_limits(left_start_deg, left_stop_deg,
                                    min_rot_deg, max_rot_deg)
    right_ok = check_rotation_limits(right_start_deg, right_stop_deg,
                                     min_rot_deg, max_rot_deg)

    is_array = isinstance(left_start_deg, np.ndarray)
    if not is_array:
        left_start_deg = np.array([left_start_deg])
        left_stop_deg = np.array([left_stop_deg])

    res_start = np.full_like(left_start_deg, np.nan)
    res_stop = np.full_like(left_stop_deg, np.nan)
    idx = np.arange(len(res_start), dtype=int)

    both_ok = np.logical_and(left_ok, right_ok)
    if np.any(both_ok):
        delta_l = np.fabs(cur_rot_deg - left_start_deg[both_ok])
        delta_r = np.fabs(cur_rot_deg - right_start_deg[both_ok])
        favor_l = delta_l < delta_r
        res_start[both_ok] = np.where(favor_l,
                                      left_start_deg[both_ok],
                                      right_start_deg[both_ok])
        res_stop[both_ok] = np.where(favor_l,
                                     left_stop_deg[both_ok],
                                     right_stop_deg[both_ok])

    just_left_ok = np.logical_and(left_ok, np.logical_not(right_ok))
    if np.any(just_left_ok):
        res_start[just_left_ok] = left_start_deg[just_left_ok]
        res_stop[just_left_ok] = left_stop_deg[just_left_ok]

    just_right_ok = np.logical_and(np.logical_not(left_ok), right_ok)
    if np.any(just_right_ok):
        res_start[just_right_ok] = right_start_deg[just_right_ok]
        res_stop[just_right_ok] = right_stop_deg[just_right_ok]

    if not is_array:
        # extract scalars
        res_start, res_stop = res_start[0], res_stop[0]

    return np.array([res_start, res_stop])


def calc_subaru_azimuths(az_deg):
    """Calculate Subaru (0 deg == South) azimuth possibilities.

    Parameters
    ----------
    az_deg: float or array of float
        A traditional azimuth value(s) where 0 deg == North

    Returns
    -------
    (naz_deg, paz_deg): tuple of float or NaN
        possible translated azimuths (0 deg == South), some of them may be NaN

    NOTE: naz_deg is always in the negative direction, paz_deg in the positive
    """
    # limit angle to 0 <= az_deg < 360.0
    az_deg = normalize_angle(az_deg, limit='full', ang_offset=0.0)

    is_array = isinstance(az_deg, np.ndarray)
    if not is_array:
        az_deg = np.array([az_deg])
    az_deg = az_deg.astype(float)

    naz_deg = np.zeros_like(az_deg)
    paz_deg = np.zeros_like(az_deg)

    mask = np.logical_and(0.0 <= az_deg, az_deg <= 90.0)
    naz_deg[mask] = - (180.0 - az_deg[mask])
    paz_deg[mask] = 180.0 + az_deg[mask]

    mask = np.logical_and(90.0 < az_deg, az_deg < 180.0)
    naz_deg[mask] = - (180.0 - az_deg[mask])
    paz_deg[mask] = np.nan

    mask = np.logical_and(180.0 < az_deg, az_deg < 270.0)
    naz_deg[mask] = np.nan
    paz_deg[mask] = az_deg[mask] - 180.0

    mask = np.logical_and(270.0 < az_deg, az_deg < 360.0)
    naz_deg[mask] = -270.0 + (az_deg[mask] - 270.0)
    paz_deg[mask] = az_deg[mask] - 180.0

    if not is_array:
        # extract scalar
        naz_deg, paz_deg = naz_deg[0], paz_deg[0]

    return np.array([naz_deg, paz_deg])


def get_quadrant(az_deg):
    """Get the quadrant (NE, SE, SW, NW) for a given azimuth.

    Parameters
    ----------
    az_deg: float or array of float
        A traditional azimuth value where 0 deg == North

    Returns
    -------
    quadrant : str or array of str
        Quadrant which contains azimuth: 'NE', 'SE', 'SW', 'NW'
    """
    # limit angle to 0 <= az_deg < 360.0
    az_deg = normalize_angle(az_deg, limit='full', ang_offset=0.0)

    is_array = isinstance(az_deg, np.ndarray)
    if not is_array:
        az_deg = np.array([az_deg])
    az_deg = az_deg.astype(float)

    res = np.zeros_like(az_deg, dtype=int)
    idx = np.arange(len(az_deg), dtype=int)

    is_south = np.logical_and(90.0 < az_deg, az_deg < 270.0)
    is_north = np.logical_not(is_south)

    is_east = az_deg[is_south] <= 180.0
    res[idx[is_south][is_east]] = 1  # 'SE'
    res[idx[is_south][np.logical_not(is_east)]] = 2  #'SW'

    is_east = np.logical_and(0.0 <= az_deg[is_north], az_deg[is_north] <= 90.0)
    res[idx[is_north][is_east]] = 3  # 'NE'
    res[idx[is_north][np.logical_not(is_east)]] = 4  #'NW'

    vals = np.array(['Bad', 'SE', 'SW', 'NE', 'NW'])
    res = vals[res]
    if not is_array:
        # extract scalar
        res = res[0]
    return res


def calc_possible_azimuths(dec_deg, az_start_deg, az_stop_deg, obs_lat_deg):
    """Calculate possible azimuth moves.

    Parameters
    ----------
    dec_deg : float
        Declination of target in degrees

    az_start_deg: float
        azimuth for target at start of observation

    az_stop_deg: float
        azimuth for target at stop of observation (end of exposure)

    obs_lat_deg: float
        Observers latitude in degrees

    Returns
    -------
    az_choices : list of (float, float) tuples
        List of possible azimuth start and stops in Subaru (S==0 deg) coordinates
    """
    # circumpolar_deg_limit = 90.0 - obs_lat_deg
    # if dec_deg > circumpolar_deg:
    #     # target in North for whole range
    #     # circumpolar orbit, object may go E to W or W to E
    #     # 2 az directions are possible
    #     if cr1.ha < cr2.ha:
    #         # object moving E to W
    #         pass
    #     else:
    #         # object moving W to E
    #         pass

    # NOTE: this "fudge factor" was added because some objects azimuth
    # as calculated by the ephemeris engine fall outside of the expected
    # ranges--this hopefully allows us to ensure that we can test whether
    # a target is will be truly in the North or South only
    fudge_factor_deg = 0.1

    if dec_deg > obs_lat_deg + fudge_factor_deg:
        # target in North for whole range
        # 2 az directions are possible
        naz_deg_start, paz_deg_start = calc_subaru_azimuths(az_start_deg)

        if not (-270.0 <= naz_deg_start <= -90.0):
            raise ValueError(f"AZ(neg) start value ({naz_deg_start}) out of range for target in North")
        if not (90.0 <= paz_deg_start <= 270.0):
            raise ValueError(f"AZ(pos) start value ({paz_deg_start}) out of range for target in North")

        naz_deg_stop, paz_deg_stop = calc_subaru_azimuths(az_stop_deg)

        if not (-270.0 <= naz_deg_stop <= -90.0):
            raise ValueError(f"AZ(neg) stop value ({naz_deg_stop}) out of range for target in North")
        if not (90.0 <= paz_deg_stop <= 270.0):
            raise ValueError(f"AZ(pos) stop value ({paz_deg_stop}) out of range for target in North")

        return [(naz_deg_start, naz_deg_stop), (paz_deg_start, paz_deg_stop)]

    elif dec_deg < 0.0 - fudge_factor_deg:
        # target in South for whole range
        # only 1 az direction is possible

        naz_deg_start, paz_deg_start = calc_subaru_azimuths(az_start_deg)
        naz_deg_stop, paz_deg_stop = calc_subaru_azimuths(az_stop_deg)

        if not np.isnan(naz_deg_start):
            # <-- target in SE
            if not np.isnan(paz_deg_start):
                raise ValueError(f"target in SE has two AZ start values ({naz_deg_start},{paz_deg_start})")
            if not np.isnan(naz_deg_stop):
                # <-- target finishes in SE
                return [(naz_deg_start, naz_deg_stop)]
            else:
                # <-- target finishes in SW
                return [(naz_deg_start, paz_deg_stop)]
        else:
            # <-- target in SW
            if np.isnan(paz_deg_stop):
                raise ValueError(f"target in SW has no AZ stop value ({paz_deg_stop})")
            if not np.isnan(naz_deg_stop):
                raise ValueError(f"target in SW has neg AZ stop value ({naz_deg_stop})")
            return [(paz_deg_start, paz_deg_stop)]

    else:
        # target could be in N and may dip S, depending on start or exp time
        # 2 az directions are possible if target stays in N
        # else only 1 az direction is possible
        start_quad = get_quadrant(az_start_deg)
        stop_quad = get_quadrant(az_stop_deg)

        naz_deg_start, paz_deg_start = calc_subaru_azimuths(az_start_deg)
        naz_deg_stop, paz_deg_stop = calc_subaru_azimuths(az_stop_deg)

        if start_quad == 'NE':
            # <-- stop_quad can be in NE, SE, SW, NW
            if stop_quad not in ['NE', 'SE', 'SW', 'NW']:
                raise ValueError(f"stop quadrant '{stop_quad}' not valid for target originating in NE")
            if stop_quad == 'NE':
                # <-- two azimuths are possible
                return [(naz_deg_start, naz_deg_stop),
                        (paz_deg_start, paz_deg_stop)]
            elif stop_quad == 'SE':
                # <-- only one azimuth is possible
                return [(naz_deg_start, naz_deg_stop)]
            else:
                # <-- only one azimuth is possible
                return [(naz_deg_start, paz_deg_stop)]

        elif start_quad == 'SE':
            # <-- stop_quad can be in SE, SW, NW
            if stop_quad not in ['SE', 'SW', 'NW']:
                raise ValueError(f"stop quadrant '{stop_quad}' not valid for target originating in SE")
            if stop_quad == 'SE':
                # <-- only one azimuth is possible
                return [(naz_deg_start, naz_deg_stop)]
            else:
                # <-- only one azimuth is possible
                return [(naz_deg_start, paz_deg_stop)]

        elif start_quad == 'SW':
            # <-- stop_quad can be in SW, NW
            if stop_quad not in ['SW', 'NW']:
                raise ValueError(f"stop quadrant '{stop_quad}' not valid for target originating in SW")
            # <-- only one azimuth is possible
            return [(paz_deg_start, paz_deg_stop)]

        elif start_quad == 'NW':
            # <-- stop_quad can be in NW only
            if stop_quad not in ['NW']:
                raise ValueError(f"stop quadrant '{stop_quad}' not valid for target originating in NW")
            # <-- two azimuths are possible
            return [(naz_deg_start, naz_deg_stop),
                    (paz_deg_start, paz_deg_stop)]

        else:
            raise ValueError(f"start quadrant '{start_quad}' type not recognized")


def calc_rotator_angle(pang_deg, pa_deg, flip=False, ins_delta=0.0):
    """Calculate the instrument rotator offset.

    NOTE: DOES NOT NORMALIZE THE ANGLES

    Parameters
    ----------
    pang_deg : float or array of float
        Parallactic angle for target(s) at a certain time

    pa_deg : float or array of float
        The desired position angle(s) in degrees

    flip : bool (optional, defaults to False)
        Whether the image is flipped or not (depends on foci)

    ins_delta : float (optional, defaults to 0.0)
        Instrument mounting offset to apply

    Returns
    -------
    rot_deg : float
        The rotator value for this observation
    """
    if flip:
        # non-mirror image, such as foci Cs, or NsOpt w/ImR
        pa_deg = -pa_deg

    # rotator_angle = parallactic_angle + position_angle
    rot_deg = pang_deg + pa_deg + ins_delta

    return rot_deg


def calc_possible_rotations(start_pang_deg, stop_pang_deg, pa_deg, ins_name,
                            dec_deg, obs_lat_deg):
    """Calculate the possible instrument rotations.

    Parameters
    ----------
    pang_deg_start : float or array of float
        Parallactic angle for target(s) at start of observation

    pang_deg_stop : float or array of float
        Parallactic angle for target(s) at stop of observation (end of exposure)

    pa_deg : float or array of float
        The desired position angle(s) in degrees

    ins_name : str
        Instrument name

    dec_deg : float
        Declination of target in degrees

    ob_lat_deg : float
        Observers latitude in degrees

    Returns
    -------
    possible_rots : array of (float, float)
        The rotator offset angles for this parallactic angle

    NOTE: the possibilities are not guaranteed to be achievable.
    They should be further checked against limits.
    """
    ins_delta = mount_offsets.get(ins_name, 0.0)
    ins_flip = mount_flip.get(ins_name, False)

    is_north = dec_deg > obs_lat_deg

    if is_north and np.sign(start_pang_deg) != np.sign(stop_pang_deg):
        # north target has a discontinuity in parallactic angle as the target
        # passes through the meridian.  If sign is different for a northerly
        # target, then we need to calculate the alternate angle to calculate
        # the correct direction of rotation
        stop_pang_deg = calc_alternate_angle(stop_pang_deg)

    start_rot_deg = calc_rotator_angle(start_pang_deg,
                                       pa_deg, flip=ins_flip,
                                       ins_delta=ins_delta)
    stop_rot_deg = calc_rotator_angle(stop_pang_deg,
                                      pa_deg, flip=ins_flip,
                                      ins_delta=ins_delta)

    rot_diff = stop_rot_deg - start_rot_deg
    # sign of this should indicate the direction of the rotation
    # rot_sign = np.sign(rot_diff)

    # normalize angles to (-360, +360)
    left_start_deg = normalize_angle(start_rot_deg, limit=None)
    left_stop_deg = left_start_deg + rot_diff

    right_start_deg = calc_alternate_angle(left_start_deg)
    right_stop_deg = right_start_deg + rot_diff

    return np.array([(left_start_deg, left_stop_deg),
                     (right_start_deg, right_stop_deg)])


def calc_offset_angle(pang_deg, pa_deg, flip=False, ins_delta=0.0):
    """Calculate the 'offset angle' for the rotator.

    Parameters
    ----------
    pang_deg : float or array of float
        Parallactic angle for target(s) at a certain time

    pa_deg : float or array of float
        The desired position angle(s) in degrees

    flip : bool (optional, defaults to False)
        Whether the image is flipped or not (depends on foci)

    ins_delta : float (optional, defaults to 0.0)
        Instrument mounting offset to apply

    Returns
    -------
    offset_deg : float
        The offset value for this observation
    """
    if flip:
        # non-mirror image, such as foci Cs, or NsOpt w/ImR
        pa_deg = -pa_deg

    # rotator_angle = parallactic_angle + position_angle
    offset_deg = normalize_angle(pa_deg + ins_delta)

    if pang_deg < 0.0:
        # object in the East, offset angle should be positive
        if offset_deg < 0.0:
            offset_deg = calc_alternate_angle(offset_deg)
    else:
        # object in the West, offset angle should be negative
        if offset_deg > 0.0:
            offset_deg = calc_alternate_angle(offset_deg)

    return offset_deg
