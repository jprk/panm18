import panm_globals
from array import array
import scipy.stats
import numpy as np
from odeintegrator import ODEIntegrator


class Vehicle(object):

    def __init__(self, veh_id: int, length: float, actual_speed: float, ode_integrator: ODEIntegrator,
                 distance: float = 0.0, stopped: bool = False) -> None:
        self.id = veh_id
        self.stopped = stopped
        self.x = distance
        self.v = actual_speed
        self.l = length
        self.pos = array('d')  # internal Python array of doubles has noting to do with np.array()
        self.spd = array('d')
        self.gap = array('d')
        self.t_sim = array('d')
        self.ode = ode_integrator
        self.t0 = None
        self.t_zero = None
        self.next_state = None
        if not (self.ode is None):
            self.state = self.ode.initialize_state(self.x, self.v)

    # def __init__(self, speed, accel, decel, length, max_speed, ode_integrator):
    #    self.x = 0
    #    self.v = speed
    #    self.vmax = max_speed
    #    self.a = accel
    #    self.b = decel
    #    self.l = length
    #    self.pos = dict()
    #    self.spd = dict()
    #    self.ode = ode_integrator
    #    self.t0 = None
    #    self.t_zero = None

    def set_entrance_time(self, t_sim):
        self.t_zero = t_sim
        self.t0 = t_sim

    def update(self, t_sim, leader=None):
        # Do not update position of a stopped vehicle
        if self.stopped:
            self.next_state = self.state
            return
        dt = t_sim - self.t0
        if leader is None:
            self.x += self.v*dt
            t_simh = t_sim + dt
        else:
            # State vectors for ODE integrator are pure python tuples.
            # Call the ODE integrator with this vehicle and its leader vehicle state information
            next_state, t_simh = self.ode.update_state(self.state, leader.state, leader.l, t_sim)
            self.x, self.v, self.next_state = self.ode.correct_state(self.state, next_state, leader.next_state, leader.l)
            # Compute the gap between both vehicles
            s = leader.x - leader.l - self.x
            self.gap.append(s)
            if s <= (self.ode.model.s0 - 1e-12):
                panm_globals.LOGGER.warn(
                    'gap between vehicles {:d} and {:d} is {:f} which is {:f} less than s0/{:f}'.format(
                        leader.id, self.id, s, s-self.ode.model.s0, self.ode.model.s0))
        # Remember the position and speed of this vehicle at this time step
        self.t_sim.append(t_simh)
        self.pos.append(self.x)
        self.spd.append(self.v)
        self.t0 = t_sim

    def has_safe_distance(self, leader):
        """

        :return: boolean
        :param leader: Vehicle
        """
        space_gap = leader.x - self.x - leader.l
        min_space_gap = self.ode.model.get_minimum_space_gap(self.v, leader.v)
        panm_globals.LOGGER.debug(
            'gap({:d},{:d}): v = {:6.3f}, v_l = {:6.3f}, space_gap = {:6.3f}, min_space_gap = {:6.3f}'.format(
                self.id, leader.id, self.v, leader.v, space_gap, min_space_gap))
        return (space_gap > min_space_gap) and (space_gap > 0.0)

    def set_initial_position(self, leader):
        initial_position = leader.x - leader.l - self.ode.model.get_minimum_space_gap(self.v, leader.v)
        panm_globals.LOGGER.debug('initial position of the vehicle {:d}: {:g} m'.format(self.id, initial_position))
        if self.v > 0.0:
            assert initial_position >= 0.0
        self.x = initial_position
        # Initialize the state of the vehicle (again) to reflect the position of the vehicle
        self.state = self.ode.initialize_state(self.x, self.v)
        panm_globals.LOGGER.debug('space gap from leader: {:g} m'.format(leader.x - leader.l - self.x))

    def move_to_next_state(self):
        self.state = self.next_state


class VehicleFactory(object):

    next_vehicle_id = 0

    @staticmethod
    def get_truncnorm(mean, sdv, min_limit, max_limit):
        a, b = (min_limit - mean) / sdv, (max_limit - mean) / sdv
        return scipy.stats.truncnorm(a, b, mean, sdv)

    def __init__(self):
        # Recomputation of truncated gaussian limits:
        # a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        # Speeds: minimum 90 km/h = 25 m/s
        #         maximum 144 km/h = 40 m/s
        #         mean 115 km/h = 32 m/s
        #         stdev 21 km/h = 6 m/s
        self.speed_rv = self.get_truncnorm(32.0, 6.0, 25.0, 40.0)
        self.maxspd_rv = self.get_truncnorm(36.0, 1.0, 35.0, 37.0)
        # Acceleration
        self.accel_rv = self.get_truncnorm(2.5, 1.0, 2.0, 4.0)
        # Deceleration
        self.decel_rv = self.get_truncnorm(3.5, 1.0, 2.5, 4.0)
        # Vehicle length
        self.length_rv = self.get_truncnorm(6.0, 1.0, 4.9, 6.5)
        # Vehicle time gap
        self.tgap_rv = self.get_truncnorm(1.2, 0.2, 1.0, 1.8)
        # Vehicle space gap
        self.sgap_rv = self.get_truncnorm(2.0, 0.5, 1.0, 2.5)
        # Random state
        self.random_state = np.random.RandomState(17502)

    def generate(self, integrator_class, car_following_class, h,
                 vehicle_length=None,
                 desired_speed=None, actual_speed=None,
                 accel=None, decel=None,
                 time_gap=None, space_gap=None):
        # If we have no prescribed fixed desired (maximum) speed ...
        if desired_speed is None:
            # ... generate it randomly
            desired_speed = self.maxspd_rv.rvs(random_state=self.random_state)
        # If we have no prescribed fixed actual speed ...
        if actual_speed is None:
            # ... make sure that the vehicle does not exceed its preferred speed when entering the road
            actual_speed = min(desired_speed, self.speed_rv.rvs(random_state=self.random_state))
        if accel is None:
            accel = self.accel_rv.rvs(random_state=self.random_state)
        if decel is None:
            decel = self.decel_rv.rvs(random_state=self.random_state)
        if vehicle_length is None:
            vehicle_length = self.length_rv.rvs(random_state=self.random_state)
        if time_gap is None:
            time_gap = self.tgap_rv.rvs(random_state=self.random_state)
        if space_gap is None:
            space_gap = self.sgap_rv.rvs(random_state=self.random_state)
        #
        cf_model = car_following_class(desired_speed, space_gap, time_gap, accel, decel)
        integrator = integrator_class.with_model_and_step(cf_model, h)
        #
        self.next_vehicle_id += 1
        #
        panm_globals.LOGGER.debug('generating vehicle {:d}:'.format(self.next_vehicle_id))
        panm_globals.LOGGER.debug('  v0={:f}, v={:f}, a={:f}, b={:f}'.format(desired_speed, actual_speed, accel, decel))
        panm_globals.LOGGER.debug('  T={:f}, s0={:f}'.format(time_gap, space_gap))
        panm_globals.LOGGER.debug('  l={:f}'.format(vehicle_length))
        #
        return Vehicle(self.next_vehicle_id, vehicle_length, actual_speed, integrator)
