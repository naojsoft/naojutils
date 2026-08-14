"""Unit Tests for the naoj.util.rot functions"""

import numpy as np

from naoj.util import rot

pfs_min_rot_deg = -174.0
pfs_max_rot_deg = +174.0


class TestRotations:
    """
    *** NOTE ***: ALL AZIMUTHS INTERPRETED AS 0 deg = South IN TEST
    AND CONVERTED TO 0 deg == North by set_target_az() !
    """

    def setup_class(self):
        # desired position angle
        self.pa_deg = 0.0
        # azimuth start and stop
        self.az_start_deg, self.az_stop_deg = 0.0, 0.0
        # parallactic angle start and stop
        self.pang_start_deg, self.pang_stop_deg = 0.0, 0.0

    def set_pa(self, pa_deg):
        self.pa_deg = pa_deg

    def set_target_pang(self, pang_start_deg, pang_stop_deg):
        self.pang_start_deg = pang_start_deg
        self.pang_stop_deg = pang_stop_deg

    def set_target_az(self, az_start_deg, az_stop_deg):
        # convert to AZ 0 deg == North
        self.az_start_deg = rot.normalize_angle(az_start_deg,
                                                limit='half', ang_offset=180.0)
        self.az_stop_deg = rot.normalize_angle(az_stop_deg,
                                               limit='half', ang_offset=180.0)

    def test_pa0_target_in_N(self):
        self.set_pa(0.0)
        self.set_target_pang(-175, +175)
        self.set_target_az(-178, +178)
        # check calculated rotator angle for this PA
        res = rot.compute_rotator_angles(self.pang_start_deg,
                                         self.pang_stop_deg,
                                         self.pa_deg,
                                         self.az_start_deg,
                                         self.az_stop_deg)
        rot_start_deg, rot_stop_deg, offset_deg = res
        assert np.isclose(rot_start_deg, -175.0, atol=0.01)
        assert np.isclose(rot_stop_deg, -185.0, atol=0.01)
        assert np.isclose(offset_deg, 0.0, atol=0.01)

        # check calculations for both rotations
        res = rot.calc_possible_rotations(self.pang_start_deg,
                                          self.pang_stop_deg,
                                          self.pa_deg,
                                          'PFS',
                                          self.az_start_deg,
                                          self.az_stop_deg)
        # return values are an array 2x3
        assert np.isclose(res[0][0], -175.0, atol=0.01)
        assert np.isclose(res[0][1], -185.0, atol=0.01)
        assert np.isclose(res[0][2], 0.0, atol=0.01)
        assert np.isclose(res[1][0], 185.0, atol=0.01)
        assert np.isclose(res[1][1], 175.0, atol=0.01)
        assert np.isclose(res[1][2], 0.0, atol=0.01)

        # check not capable of this
        end_res = rot.calc_optimal_rotation(res[0][0], res[0][1],
                                            res[1][0], res[1][1], 0.0,
                                            pfs_min_rot_deg,
                                            pfs_max_rot_deg)
        assert len(end_res) == 2
        start_rot, stop_rot = end_res
        assert np.isnan(start_rot)
        assert np.isnan(stop_rot)

    def do_pfs_ng(self, pa_deg, pang_start_deg, pang_stop_deg,
                  az_start_deg, az_stop_deg):
        self.set_pa(pa_deg)
        self.set_target_pang(pang_start_deg, pang_stop_deg)
        self.set_target_az(az_start_deg, az_stop_deg)

        res = rot.calc_possible_rotations(self.pang_start_deg,
                                          self.pang_stop_deg,
                                          self.pa_deg,
                                          'PFS',
                                          self.az_start_deg,
                                          self.az_stop_deg)

        end_res = rot.calc_optimal_rotation(res[0][0], res[0][1],
                                            res[1][0], res[1][1], 0.0,
                                            pfs_min_rot_deg,
                                            pfs_max_rot_deg)
        assert len(end_res) == 2
        start_rot, stop_rot = end_res
        assert np.isnan(start_rot)
        assert np.isnan(stop_rot)

    def do_pfs_ok(self, pa_deg, pang_start_deg, pang_stop_deg,
                  az_start_deg, az_stop_deg):
        self.set_pa(pa_deg)
        self.set_target_pang(pang_start_deg, pang_stop_deg)
        self.set_target_az(az_start_deg, az_stop_deg)

        res = rot.calc_possible_rotations(self.pang_start_deg,
                                          self.pang_stop_deg,
                                          self.pa_deg,
                                          'PFS',
                                          self.az_start_deg,
                                          self.az_stop_deg)

        end_res = rot.calc_optimal_rotation(res[0][0], res[0][1],
                                            res[1][0], res[1][1], 0.0,
                                            pfs_min_rot_deg,
                                            pfs_max_rot_deg)
        assert len(end_res) == 2
        start_rot, stop_rot = end_res
        assert not np.isnan(start_rot)
        assert not np.isnan(stop_rot)

    def test_pa90_target_in_N(self):
        self.do_pfs_ok(90.0, -175, +175, -178, +178)

    def test_pa90_target_in_N_no2(self):
        # specify different azimuths for same area
        self.do_pfs_ok(90.0, -175, +175, -178, -182)

    def test_pa90_target_in_N_no3(self):
        # specify different azimuths for same area
        self.do_pfs_ok(90.0, -175, +175, +182, +178)

    def test_pa90_target_in_N_not_ok(self):
        # specify different azimuths for same area
        self.do_pfs_ok(90.0, -96, -99, -120, -121)
