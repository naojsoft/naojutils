import numpy as np
from astropy.table import Table

# default magnification
beta = 0.29898169

# see SBR writing functions
sbr_fudge = 1.006


def file2table(path):
    """Read an MDP file into an Astropy table"""

    # read mdp file
    with open(path, 'r') as mdp_f:
        buf = mdp_f.read()
        lines = buf.split('\n')
        lines.pop(-1)

    # separate into arrays of the appropriate type
    tups = [line.split() for line in lines]
    x = np.array([float(tup[0]) for tup in tups])
    y = np.array([float(tup[1]) for tup in tups])
    length = np.array([float(tup[2]) for tup in tups])
    width = np.array([float(tup[3]) for tup in tups])
    ang = np.array([float(tup[4]) for tup in tups])
    priority = np.array([float(tup[5]) for tup in tups])
    otype = np.array([tup[6] for tup in tups])
    comment = np.array([' '.join(tup[7:]) for tup in tups])

    # make a proper table with headers and columns
    tbl = Table([x, y, length, width, ang, priority, otype, comment],
            names=['x', 'y', 'slit_length', 'slit_width',
                   'angle', 'priority', 'type', 'comment'])

    return tbl


def load_mdp(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            row_dict = {
                'type': parts[6].strip(),
                'x': float(parts[0]),
                'y': float(parts[1]),
                'width': float(parts[2]),
                'length': float(parts[3]),
                'angle': float(parts[4]),
                'priority': parts[5],
                'comment': " ".join(parts[7:]) if len(parts) > 7 else ''
            }
            if row_dict['type'].startswith('C'):
                row_dict['diameter'] = row_dict.pop('width')
                row_dict.pop('length')
                row_dict.pop('angle')
            rows.append(row_dict)

    return rows


def rows2table(rows):
    """Convert a list of dicts representing a MOIRCS mask design
    into an Astropy table that can be written out as an SBR file
    with table2sbr().
    """
    # separate into arrays of the appropriate type
    x = np.array([float(dct['x']) for dct in rows])
    y = np.array([float(dct['y']) for dct in rows])
    length = np.array([float(dct['length' if 'length' in dct else 'diameter'])
                       for dct in rows])
    width = np.array([float(dct['width' if 'width' in dct else 'diameter'])
                      for dct in rows])
    ang = np.array([float(dct.get('angle', 0.0)) for dct in rows])
    priority = np.array([float(dct['priority']) for dct in rows])
    otype = np.array([dct['type'] for dct in rows])
    comment = np.array([dct['comment'] for dct in rows])

    # make a proper table with headers and columns
    tbl = Table([x, y, length, width, ang, priority, otype, comment],
            names=['x', 'y', 'slit_length', 'slit_width',
                   'angle', 'priority', 'type', 'comment'])

    return tbl


def table2sbr(table, fov_ctr_px, px_scale):
    """Write a table representing a MOIRCS mask design into a buffer
    that is in the SBR file format.

    Returns a buffer and a list of warnings.
    """
    # offset = np.deg2rad(mos_rot_deg)
    conversion = 0.015 / beta / 0.1038 * px_scale
    fov_x_ctr, fov_y_ctr = fov_ctr_px

    warnings = []
    lines = []
    for i, shape in enumerate(table):
        x = shape['x']
        y = shape['y']
        sl_l = shape['slit_width'] * 0.5
        x1_off = x - sl_l - fov_x_ctr
        x2_off = x + sl_l - fov_x_ctr
        y1_off = y - fov_y_ctr
        y2_off = y - fov_y_ctr
        x1_focus = -x1_off * conversion
        x2_focus = -x2_off * conversion
        y1_focus = y1_off * conversion
        y2_focus = y2_off * conversion
        x1_laser = x1_focus * sbr_fudge
        x2_laser = x2_focus * sbr_fudge
        y1_laser = y1_focus * sbr_fudge
        y2_laser = y2_focus * sbr_fudge

        # Commenting out for now, Ichi-san says this is never used for MOIRCS
        # if mos_rot_deg != 0:
        #     r1 = np.hypot(x1_laser, y1_laser)
        #     theta1 = np.arctan2(y1_laser, x1_laser)
        #     x1_laser = r1 * np.cos(theta1 + offset)
        #     y1_laser = r1 * np.sin(theta1 + offset)
        #     r2 = np.hypot(x2_laser, y2_laser)
        #     theta2 = np.arctan2(y2_laser, x2_laser)
        #     x2_laser = r2 * np.cos(theta2 + offset)
        #     y2_laser = r2 * np.sin(theta2 + offset)

        corners_r = np.array([np.hypot(x1_focus, y1_focus),
                              np.hypot(x1_focus, y2_focus),
                              np.hypot(x2_focus, y1_focus),
                              np.hypot(x2_focus, y2_focus)])
        if np.any(corners_r > 90):
            warnings.append(f"{'Slit' if shape['type'].startswith('B') else 'Hole'} {i} is out of laser FOV.")

        if shape['type'].startswith('B') and np.any(np.abs([x1_focus, x2_focus]) > 60):
            warnings.append(f"Slit {i} is out of MOIRCS FOV.")

        if shape['type'].startswith('B'):
            width = (shape['slit_length'] * px_scale / 2.06218 * sbr_fudge) * 1.08826 - 0.126902
            lines.append(f"B,{x1_laser:9.4f},{y1_laser:9.4f},{x2_laser:9.4f},{y2_laser:9.4f},{width:9.4f}")
        else:
            radius = shape['slit_width'] / 2 * 0.015 / beta / 0.1038 * px_scale
            lines.append(f"C,{(x1_laser + x2_laser) * 0.5:9.4f},{(y1_laser + y2_laser) * 0.5:9.4f},{abs((x2_laser - x1_laser) * 0.5):9.4f}")

    sbr_s = "\n".join(lines) + "\n"
    return sbr_s, warnings


def table2sbr2(table, fov_ctr_px, px_scale):

    # FOV center
    fov_ctr_x, fov_ctr_y = fov_ctr_px

    sl_half = table['slit_length'] * 0.5
    x1_offset_px = (table['x'] - sl_half) - fov_ctr_x
    x2_offset_px = (table['x'] + sl_half) - fov_ctr_x
    sw_half = table['slit_width'] * 0.5
    # y1_offset_px = (table['y'] - sw_half) - fov_ctr_y
    # y2_offset_px = (table['y'] + sw_half) - fov_ctr_y
    y1_offset_px = table['y'] - fov_ctr_y
    y2_offset_px = table['y'] - fov_ctr_y

    conversion = 0.015 / beta / 0.1038 * px_scale
    x1_focus = - x1_offset_px * conversion
    x2_focus = - x2_offset_px * conversion
    y1_focus = y1_offset_px * conversion
    y2_focus = y2_offset_px * conversion

    warnings = []
    # FOV check
    # laser cutter (<180")
    r1 = np.hypot(x1_focus, y1_focus)
    r2 = np.hypot(x1_focus, y2_focus)
    r3 = np.hypot(x2_focus, y1_focus)
    r4 = np.hypot(x2_focus, y2_focus)

    tst1 = np.logical_or(r1 > 90.0, r2 > 90.0)
    tst2 = np.logical_or(r3 > 90.0, r4 > 90.0)
    tst = np.logical_or(tst1, tst2)
    if np.any(tst):
        idxs = np.nonzero(tst)
        idxs = map(str, idxs[0] + 1)
        warnings.append("slits out of FOV: {}".format(",".join(idxs)))

    # MOIRCS (|x|<120")
    tst1 = np.logical_or(np.abs(x1_focus) > 60.0, np.abs(x2_focus) > 60.0)
    if np.any(tst):
        idxs = np.nonzero(tst)
        idxs = map(str, idxs[0] + 1)
        warnings.append("slits out of FOV: {}".format(",".join(idxs)))

    x1_laser = x1_focus * sbr_fudge
    x2_laser = x2_focus * sbr_fudge
    y1_laser = y1_focus * sbr_fudge
    y2_laser = y2_focus * sbr_fudge

    w_laser = (table['slit_width'] * px_scale / 2.06218 * sbr_fudge) * 1.08826 - 0.126902

    lines = ["B,{0: 9.4f},{1: 9.4f},{2: 9.4f},{3: 9.4f},{4: 9.4f}".format(
        x1_laser[i], y1_laser[i], x2_laser[i], y2_laser[i], w_laser[i])
             if table['type'][i].startswith('B') else
             "C,{0: 9.4f},{1: 9.4f},{2: 9.4f}".format((x1_laser[i] + x2_laser[i]) * 0.5,
                                                      (y1_laser[i] + y2_laser[i]) * 0.5,
                                                      np.abs((x2_laser[i] - x1_laser[i]) * 0.5))
             for i in range(len(x1_laser))]
    sbr_s = "\n".join(lines) + "\n"
    return sbr_s, warnings
