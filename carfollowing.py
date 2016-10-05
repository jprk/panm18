import numpy as np
import math


class CarFollowingModel(object):

    def __init__(self):
        pass

    def accel_fnc(self, s, v, v_leader):
        raise NotImplementedError('generic accel_fnc not implemented')

    def get_minimum_space_gap(self, v, v_leader):
        raise NotImplementedError('generic get_minimum_space_gap not implemented')

    def get_params(self):
        raise NotImplementedError('generic get_params not implemented')

    def set_params(self, data_tuple):
        raise NotImplementedError('generic set_params not implemented')

    def model_fnc(self, y: tuple, y_leader: tuple, l_leader: float, t: float) -> tuple:
        """
        Generic integrator function that transforms a 2-tuple of vehicle position and speed to a new 2-tuple,
        using the information about leader vehicle position, speed, and vehicle length.
        :param y: actual vehicle position and speed
        :param y_leader: leader vehicle position and speed
        :param l_leader: length of the leader vehicle
        :param t: current time
        :return: updated actual vehicle position and speed at time `t`
        """
        # State vector
        x, v = y
        # Leader state vector
        x_leader, v_leader = y_leader
        # Gap between vehicles
        s = x_leader - x - l_leader
        assert s > 0.0
        # Derivatives
        dxdt = v
        dvdt = self.accel_fnc(s, v, v_leader)
        return dxdt, dvdt


class CarFollowingIDM(CarFollowingModel):

    @staticmethod
    def shorthand():
        return 'idm__'

    def __init__(self, v0, s0, T, a, b):
        # Initialize the parent class
        super(CarFollowingIDM, self).__init__()
        self.v0 = v0
        self.s0 = s0
        self.T = T
        self.a = a
        self.b = b
        self.delta = 4.0
        # math.sqrt is about 4.5 times faster than np.sqrt
        self.sqrt_ab = math.sqrt(a*b)

    def get_params(self):
        return self.v0, self.s0, self.T, self.a, self.b, self.delta, self.sqrt_ab

    def set_params(self, data_tuple):
        self.v0, self.s0, self.T, self.a, self.b, self.delta, self.sqrt_ab = data_tuple

    def s_star(self, v, v_leader):
        dv = v - v_leader
        s_star = self.s0 + v * self.T
        s_star += v * dv / (2 * self.sqrt_ab)
        return s_star

    def accel_fnc(self, s, v, v_leader):
        spf = self.s_star(v, v_leader) / s
        vfd = v / self.v0
        # For stopped vehicles we may sometimes get minimal positive or negative `scale` values due to
        # numerical errors ...
        scale = 1.0 - math.pow(vfd, self.delta) - spf*spf
        if math.fabs(scale) < 1e-12:
            scale = 0.0
        dvdt = self.a * scale
        return dvdt

    def get_minimum_space_gap(self, v, v_leader):
        return max(self.s_star(v, v_leader), self.s0 + v * self.T)


class CarFollowingTest(CarFollowingModel):

    @staticmethod
    def shorthand():
        return 'test_'

    def __init__(self, v0, s0, T, a, b):
        # Initialize the parent class
        super(CarFollowingTest, self).__init__()
        self.v0 = v0
        self.s0 = s0
        self.T = T
        self.a = a
        self.b = b
        self.delta = 4.0
        # math.sqrt is about 4.5 times faster than np.sqrt
        self.sqrt_ab = math.sqrt(a*b)

    def get_params(self):
        return self.v0, self.s0, self.T, self.a, self.b, self.delta, self.sqrt_ab

    def set_params(self, data_tuple):
        self.v0, self.s0, self.T, self.a, self.b, self.delta, self.sqrt_ab = data_tuple

    def model_fnc(self, y: tuple, y_leader: tuple, l_leader: float, t: float) -> tuple:
        """
        Generic integrator function that transforms a 2-tuple of vehicle position and speed to a new 2-tuple,
        using the information about leader vehicle position, speed, and vehicle length.
        :param y: actual vehicle position and speed
        :param y_leader: leader vehicle position and speed
        :param l_leader: length of the leader vehicle
        :param t: current time
        :return: updated actual vehicle position and speed at time `t`
        """
        # State vector
        x, v = y
        # Derivatives
        dxdt = v
        dvdt = math.sin(t)
        return dxdt, dvdt

    def get_minimum_space_gap(self, v, v_leader):
        return 4.0
