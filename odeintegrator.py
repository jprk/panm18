from carfollowing import CarFollowingModel
import panm_globals


class ODEIntegrator(object):

    def __init__(self, h):
        self.model = None
        self.f = None
        self.h = h
        self.h2 = 0.5*h
        self.hh2 = 0.5 * h * h  # for ballistic update

    @classmethod
    def with_model_and_step(cls, cf_model: CarFollowingModel, h: float) -> None:
        integrator = cls(h)
        integrator.model = cf_model
        integrator.f = cf_model.model_fnc
        return integrator

    def stopping_heuristics(self, yx: float, yv: float, dx: float, dv: float, h: float) -> tuple:
        yvh = yv + h * dv
        if dv < 0 and yvh < 0:
            # Do not stop vehicles that are already stopped.
            # TODO: Investigate why this sometimes happens.
            if yv == 0.0:
                yvh = 0.0
                yxh = yx
                panm_globals.LOGGER.debug(
                    'yx = {:f}, yv = {:f}, dx = {:f}, dv = {:f} ..... already stopped vehicle has stopped?'.format(yx, yv, dx, dv))
            else:
                yxh_temp = yx + h * dx
                yxh = yx - dx * dx / dv
                panm_globals.LOGGER.debug(
                    'dv = {:f}, yvh = {:f} ..... projected position {:f}, forced stop'.format(dv, yvh, yxh_temp))
                panm_globals.LOGGER.debug(
                    'vehicle probably stopped at {:f}, setting yvh to zero'.format(yxh))
                panm_globals.LOGGER.debug(
                    'yx = {:f}, yv = {:f}, dx = {:f}, dv = {:f}, h = {:f}'.format(yx, yv, dx, dv, h))
                yvh = 0.0
        else:
            yxh = yx + h * dx
            if yvh < 0:
                panm_globals.LOGGER.warn(
                    'negative speed {:f} with non-negative acceleration {:f} at {:f}'.format(yvh, dv, yxh))
        # Todo: Additional check of the state consistency here?
        return yxh, yvh

    def update_state(self, vehicle_state: tuple, leader_state: tuple, leader_length: float, t: float) -> tuple:
        raise NotImplementedError('integrator update method has to be implemented in child class')

    def initialize_state(self, x: float, v: float) -> tuple:
        raise NotImplementedError('create state method has to be implemented in child class')

    def correct_state(self, vehicle_state_t: tuple, vehicle_state_th: tuple, leader_state_th: tuple, leader_length: float) -> tuple:
        raise NotImplementedError('state checking method has to be implemented in child class')

    @staticmethod
    def _correct(state_t: tuple, state_th: tuple, state_lth: tuple, leader_length: float, s0: float) -> tuple:
        yx, yv = state_t
        yxh, yvh = state_th
        if yxh < yx:
            yxh = yx
            panm_globals.LOGGER.warn('yhx < yx')
        if yvh < 0.0:
            yvh = 0.0
            panm_globals.LOGGER.warn('yvh < 0.0')
        # Now handle the rare case when a vehicle "overshoots" the leader if the leader has been forced to stop
        # by the heuristics in update_state().
        lxh, lvh = state_lth
        leader_back_x = lxh - leader_length
        if yxh + s0 >= leader_back_x + 1e-12:
            # The vehicle is closer to the leader than acceptable. This will result in an attempt to reverse the
            # vehicle movement in the next integration step.
            yxh = leader_back_x - s0
            if yxh < yx:
                # Our vehicle wants to reverse anyway, this is bad
                panm_globals.LOGGER.error('irreversible vehicle overshot')
                yxh = yx
            if lvh > 0.0:
                # Overshot leader vehicle that is still moving
                panm_globals.LOGGER.error('overshot moving leader vehicle')
            yvh = 0.0
        #
        return yxh, yvh

class EulerODEIntegrator(ODEIntegrator):

    @staticmethod
    def shorthand():
        return 'euler'

    @staticmethod
    def order():
        return 1

    def initialize_state(self, x: float, v: float) -> tuple:
        # The state vector for first order method has just the current state values and the "future" value at t+h
        return x, v

    def update_state(self, vehicle_state: tuple, leader_state: tuple, leader_length: float, t: float) -> tuple:
        # Timing shows that decomposition of tuple into variables is faster than indexing...
        yx, yv = vehicle_state
        dx1, dv1 = self.f(vehicle_state, leader_state, leader_length, t)
        th = t + self.h
        yh = self.stopping_heuristics(yx, yv, dx1, dv1, self.h)
        return yh, th

    def correct_state(self, vehicle_state_t: tuple, vehicle_state_th: tuple, leader_state_th: tuple, leader_length: float) -> tuple:
        # For Euler integrator we do not have to decompose the vehicle states and so the
        # EulerODEIntegrator::_correct() mathod can be called directly.
        yxh, yvh = self._correct(vehicle_state_t, vehicle_state_th, leader_state_th, leader_length, self.model.s0)
        # It looks like overkill here, but the state tuple is more complex in higher-order methods
        return yxh, yvh, (yxh, yvh)

class TrapezoidODEIntegrator(ODEIntegrator):

    @staticmethod
    def shorthand():
        return 'trape'

    @staticmethod
    def order():
        return 2

    def initialize_state(self, x: float, v: float) -> tuple:
        # The state vector for explicit trapezoid contains the current state value and the previous state
        # at time step t-h
        return (x, v), (x, v)

    def update_state(self, vehicle_state: tuple, leader_state: tuple, leader_length: float, t: float) -> tuple:
        """
        Update the vehicle state using explicit trapezoidal rule (Heun's method).
        :param vehicle_state: tuple of two states (the state at t-h and the state at t)
        :param leader_state:  tuple of two leader vehicle states; as the leader vehicle has been updated already,
                              the leader state contains states at t and at t+h
        :param leader_length: the length of the leader vehicle
        :param t: current time
        :return: a tuple of state information (2-tuple of states at t and t+h) and shifted time
        """
        # Timing shows that decomposition of tuple into variables is faster than indexing...
        *_, vehicle_state_t = vehicle_state
        yx, yv = vehicle_state_t
        leader_state_t, leader_state_th = leader_state
        k1x, k1v = self.f(vehicle_state_t, leader_state_t, leader_length, t)
        # Temporary position of this vehicle for the second sample
        # yt = (yx + self.h * k1x, yv + self.h * k1v)
        y_pred = self.stopping_heuristics(yx, yv, k1x, k1v, self.h)
        # Temporary position of leader vehicle for the second sample
        k2x, k2v = self.f(y_pred, leader_state_th, leader_length, t + self.h)
        th = t + self.h
        # vehicle_state_th = (yx + self.h2 * (k1x+k2x), yv + self.h2 * (k1v+k2v))
        vehicle_state_th = self.stopping_heuristics(yx, yv, k1x+k2x, k1v+k2v, self.h2)
        yh = vehicle_state_t, vehicle_state_th
        # assert vehicle_state_th[0] > 0
        # assert vehicle_state_th[1] > 0
        return yh, th

    def correct_state(self, vehicle_state_t: tuple, vehicle_state_th: tuple, leader_state_th: tuple, leader_length: float) -> tuple:
        _, state_t = vehicle_state_t
        _, state_th = vehicle_state_th
        _, state_lth = leader_state_th
        yxh, yvh = self._correct(state_t, state_th, state_lth, leader_length, self.model.s0)
        return yxh, yvh, (state_t, (yxh, yvh))


class RungeKutta4ODEIntegrator(ODEIntegrator):

    @staticmethod
    def shorthand():
        return 'rk4__'

    @staticmethod
    def order():
        return 4

    def initialize_state(self, x: float, v: float) -> tuple:
        # The state vector for the fourth-order Runge-Kutta contains the current state value and the previous state
        # at time steps t-h and t-h/2
        return (x, v), (x, v), (x, v)

    def update_state(self, vehicle_state: tuple, leader_state: tuple, leader_length: float, t: float) -> tuple:
        """
        Update the vehicle state using explicit RK4 rule (Heun's method).
        :param vehicle_state: tuple of three states (the state at t-h, t-h/2 and the state at t)
        :param leader_state:  tuple of three leader vehicle states; as the leader vehicle has been updated already,
                              the leader state contains states at t, t+h/2 and at t+h
        :param leader_length: the length of the leader vehicle
        :param t: current time
        :return: a tuple of state information (2-tuple of states at t and t+h) and shifted time
        """
        # Timing shows that decomposition of tuple into variables is faster than indexing...
        *_, vehicle_state_t = vehicle_state
        yx, yv = vehicle_state_t
        leader_state_t, leader_state_th2, leader_state_th = leader_state
        k1x, k1v = self.f(vehicle_state_t, leader_state_t, leader_length, t)
        # Temporary position for the second sample
        # yt = (yx + self.h2 * k1x, yv + self.h2 * k1v)
        panm_globals.LOGGER.debug('RK4 step 1')
        y_pred1 = self.stopping_heuristics(yx, yv, k1x, k1v, self.h2)
        k2x, k2v = self.f(y_pred1, leader_state_th2, leader_length, t + self.h2)
        # Temporary position for the third sample
        # yt = (yx + self.h2 * k2x, yv + self.h2 * k2v)
        panm_globals.LOGGER.debug('RK4 step 2')
        y_pred2 = self.stopping_heuristics(yx, yv, k2x, k2v, self.h2)
        k3x, k3v = self.f(y_pred2, leader_state_th2, leader_length, t + self.h2)
        # Temporary position for the fourth sample
        # yt = (yx + self.h * k3x, yv + self.h * k3v)
        panm_globals.LOGGER.debug('RK4 step 3')
        y_pred3 = self.stopping_heuristics(yx, yv, k3x, k3v, self.h)
        try:
            k4x, k4v = self.f(y_pred3, leader_state_th, leader_length, t + self.h)
        except AssertionError:
            panm_globals.LOGGER.error('vehicle_state_t  = ' + str(vehicle_state_t))
            panm_globals.LOGGER.error('leader_state_t   = ' + str(leader_state_t))
            panm_globals.LOGGER.error('leader_state_th2 = ' + str(leader_state_th2))
            panm_globals.LOGGER.error('leader_state_th  = ' + str(leader_state_th))
            panm_globals.LOGGER.error('k1x, k1v = ({:-f}, {:-f}) ----> ypred1 = {:s}'.format(k1x, k1v, str(y_pred1)))
            panm_globals.LOGGER.error('k2x, k2v = ({:-f}, {:-f}) ----> ypred2 = {:s}'.format(k2x, k2v, str(y_pred2)))
            panm_globals.LOGGER.error('k3x, k3v = ({:-f}, {:-f}) ----> ypred3 = {:s}'.format(k3x, k3v, str(y_pred3)))
            k3xt, k3vt = self.f(y_pred2, leader_state_th2, leader_length, t + self.h2)
            y_pred3 = self.stopping_heuristics(yx, yv, k3xt, k3vt, self.h)
            raise
        th = t + self.h
        # TODO: This it just a wild guess ... the x and v at time h/2 is the average of "middle" samples from RK4.
        # I.e. y_th2 = y + h/4 * (k2+k3)
        panm_globals.LOGGER.debug('RK4 step 4 th2')
        vehicle_state_th2 = self.stopping_heuristics(yx, yv, k2x+k3x, k2v+k3v, 0.5*self.h2)
        # vehicle_state_th  = (yx + self.h * (k1x + 2*k2x + 2*k3x + k4x) / 6.0, yv + self.h * (k1v + 2*k2v + 2*k3v + k4v) / 6.0)
        panm_globals.LOGGER.debug('RK4 step 4 th')
        vehicle_state_th = self.stopping_heuristics(yx, yv, k1x+2*k2x+2*k3x+k4x, k1v + 2*k2v + 2*k3v + k4v, self.h/6)
        yh = vehicle_state_t, vehicle_state_th2, vehicle_state_th
        # assert yh[0] > 0
        # assert yh[1] > 0
        return yh, th

    def correct_state(self, vehicle_state_t: tuple, vehicle_state_th: tuple, leader_state_th: tuple, leader_length: float) -> tuple:
        *_, state_t = vehicle_state_t
        _, state_th2, state_th = vehicle_state_th
        _, state_lth2, state_lth = leader_state_th
        state_th2 = self._correct(state_t, state_th2, state_lth2, leader_length, self.model.s0)
        yxh, yvh = self._correct(state_th2, state_th, state_lth, leader_length, self.model.s0)
        return yxh, yvh, (state_t, state_th2, (yxh, yvh))


class BallisticODEIntegrator(ODEIntegrator):

    @staticmethod
    def shorthand():
        return 'bali_'

    @staticmethod
    def order():
        return 1

    def initialize_state(self, x: float, v: float) -> tuple:
        # The state vector for ballistic methods is the same as for a first order method - it contains just
        # the current state values
        return x, v

    def update_state(self, vehicle_state: tuple, leader_state: tuple, leader_length: float, t: float) -> tuple:
        # Timing shows that decomposition of tuple into variables is faster than indexing...
        yx, yv = vehicle_state
        k1x, k1v = self.f(vehicle_state, leader_state, leader_length, t)
        yh = (yx + self.h * yv + self.hh2 * k1v, yv + self.h * k1v)
        th = t + self.h
        return yh, th

    def correct_state(self, vehicle_state_t: tuple, vehicle_state_th: tuple, leader_back_x: tuple) -> tuple:
        raise NotImplementedError('state checking method has to be implemented in child class')
