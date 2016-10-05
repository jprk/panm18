import panm_globals
from array import array
import scipy.stats
import numpy as np
from odeintegrator import ODEIntegrator
import math
import logging
from detector import Detector

class Vehicle(object):

    def __init__(self, veh_id: int, length: float, actual_speed: float = 0.0, ode_integrator: ODEIntegrator = None,
                 distance: float = 0.0, stopped: bool = False, store_data: bool = True) -> None:
        self.id = veh_id
        self.stopped = stopped
        self.x = distance
        self.v = actual_speed
        self.l = length
        self.pos = array('d')  # internal Python array of doubles has nothing to do with np.array()
        self.spd = array('d')
        self.gap = array('d')
        self.acc = array('d')
        self.t_sim = array('d')
        self.store_data = store_data  # For longer simulations we have to skip storing position at every step
        self.ode = ode_integrator
        self.time_step_start = None
        self.generated_time = None
        self.next_state = None
        if not (self.ode is None):
            self.state = self.ode.initialize_state(self.x, self.v)
        self.logger = None
        self.log_file_handler = None

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
    #    self.time_step_start = None
    #    self.generated_time = None

    @staticmethod
    def state_str(state: tuple):
        if len(state) == 2:
            a, b = state
            if isinstance(a, float):
                assert isinstance(b, float)
                state = (state,)
        if len(state) > 1:
            prefix = '('
            postfix = ')'
            infix = ','
        else:
            prefix = ''
            postfix = ''
            infix = ''
        out_str = ""
        out_infix = ""
        for apair in state:
            x, v = apair
            out_str += out_infix
            out_str += '({:+9.4f},{:8.4f})'.format(x,v)
            out_infix = infix
        return prefix + out_str + postfix

    def start_logging(self, leader: 'Vehicle') -> None:
        # New logger name
        logger_name = 'iterations_vh{:02d}_{:s}_{:s}_{:04d}'.format(
            self.id, self.ode.model.shorthand(), self.ode.shorthand(), int(1000*self.ode.h))
        logger_file_name = logger_name+'.log'
        # Create a logger object
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        self.log_file_handler = logging.FileHandler(logger_file_name, mode='w')
        self.log_file_handler.setLevel(logging.DEBUG)
        # create formatters and add it to the handlers
        formatter = logging.Formatter('%(levelname)-8s %(message)s')
        self.log_file_handler.setFormatter(formatter)
        # add the handlers to logger
        self.logger.addHandler(self.log_file_handler)
        self.logger.info("logging started")
        self.leader = leader
        self.leader.logger = self.logger

    def stop_logging(self):
        self.logger.removeHandler(self.log_file_handler)
        self.logger = None
        self.leader.logger = None
        self.leader = None

    def gap_to_leader(self, leader: 'Vehicle' = None) -> float:
        if leader is None:
            gap = float('inf')
        else:
            gap = leader.x - leader.l - self.x
        return gap

    def set_generated_time(self, t_sim: float) -> None:
        self.generated_time = t_sim
        self.time_step_start = t_sim

    def record_state(self, t_sim: float, leader: 'Vehicle'= None) -> None:
        self.t_sim.append(t_sim)
        self.spd.append(self.v)
        self.pos.append(self.x)
        self.acc.append(t_sim)
        self.gap.append(self.gap_to_leader(leader))

    def update(self, t_sim: float, leader: 'Vehicle' = None, detector: Detector = None):
        """

        :param t_sim:
        :param leader:
        :return:
        """
        # The whole update makes no sense in case that the global timestamp points to some other point in time
        # than the expected beginning of the time step
        assert math.fabs(t_sim - self.time_step_start) < 1e-12
        # Continuous accumulating of self.ode.h will lead to rounding errors for small time steps. Consider t_sim
        # to be the reference clock and move just a single step from it.
        self.time_step_start = t_sim + self.ode.h
        # Do not update position of a vehicle that does not move
        if self.stopped:
            # Make sure the state update step generates a correct state for this vehicle
            self.next_state = self.state
            # There is nothing else to do in this case
            return
        # Check if the vehicle shall drive with the free flow speed
        # TODO: This is not quite correct. We shall replace this by calling the self.ode.update_state with the
        # leader position in infinity.
        if leader is None:
            # Compute the vehicle position at the end of this time step
            # self.x += self.v * self.ode.h
            # a = 0.0
            dummy_leader_state = self.ode.initialize_state(math.inf, 0.0)
            prev_speed = self.v
            prev_pos = self.x
            next_state, t_simh = \
                self.ode.update_state(self.state, dummy_leader_state, 0.0, t_sim)
            self.x, self.v, self.next_state = \
                self.ode.correct_state(self.state, next_state, dummy_leader_state, 0.0)
            # Compute back the acceleration
            a = (self.v - prev_speed) / self.ode.h
            # In case that the update scheme is adaptive, the time step will be different. We have to accept
            # the returned t_simh value
            self.time_step_start = t_simh
        else:
            if self.logger is not None:
                self.logger.debug('t_sim={:8.4f}, vehicle {:2d}: {:s}, leader: {:s}'.format(
                    t_sim, self.id, self.state_str(self.state), self.state_str(leader.next_state)))
            # State vectors for ODE integrator are pure python tuples.
            # Call the ODE integrator with this vehicle and its leader vehicle state information
            prev_speed = self.v
            prev_pos = self.x
            next_state, t_simh = \
                self.ode.update_state(self.state, leader.next_state, leader.l, t_sim)
            self.x, self.v, self.next_state = \
                self.ode.correct_state(self.state, next_state, leader.next_state, leader.l)
            # Compute back the acceleration
            a = (self.v - prev_speed)/self.ode.h
            # In case that the update scheme is adaptive, the time step will be different. We have to accept
            # the returned t_simh value
            self.time_step_start = t_simh
        # Compute the gap between both vehicles
        s = self.gap_to_leader(leader)
        if s <= (self.ode.model.s0 - 1e-12):
            panm_globals.LOGGER.warn(
                'gap between vehicles {:d} and {:d} is {:f} which is {:f} less than s0={:f}'.format(
                    leader.id, self.id, s, s - self.ode.model.s0, self.ode.model.s0))
        # Remember the position and speed of this vehicle at this time step
        if self.store_data:
            self.t_sim.append(self.time_step_start)
            #
            self.pos.append(self.x)
            self.spd.append(self.v)
            self.acc.append(a)
            self.gap.append(s)
        # Record detector data
        if detector is not None:
            if self.x >= detector.x and prev_pos < detector.x:
                # The vehicle crossed the detector in this time step
                scale = (self.x - detector.x)/(self.x - prev_pos)
                detector_time = self.time_step_start - scale * self.ode.h
                detector.record_vehicle(detector_time)
                panm_globals.LOGGER.debug('vehicle {:d} passed over detector at {:f}'.format(
                    self.id, detector_time ))

    def update_from_entrance_time(self, t_sim, leader):
        # The value of `time_step_start` contains the entrance time of the vehicle, but we expect that it contains
        # the value that corresponds to `t_sim`
        temp_h = t_sim + self.ode.h - self.generated_time
        # TODO: Possible inconsistency in self.time_step_start, check the initialisation of this value.
        self.time_step_start = self.generated_time
        # Remember the original time step of the ODE integrator, we will have to restore it later
        orig_h = self.ode.get_step()
        # Set a temporary integration step that will move the vehicle from the entrance time to `t_sim`
        # TODO: This does not take into account the middle sample of RK4 which will be incorrect for the leader!
        self.ode.set_step(temp_h)
        # Perform the position and velocity update to put the vehicle onto the prescribed time grid
        self.update(self.generated_time, leader)
        # Return the settings of the ODE integrator to normal
        self.ode.set_step(orig_h)
        # Recompute the position of the next time step, which was incorrectly updated with the temporary time step.
        self.time_step_start = t_sim + self.ode.h

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

    def __init__(self, store_data: bool = True) -> None:
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
        self.store_data = store_data

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
        return Vehicle(self.next_vehicle_id, vehicle_length, actual_speed, integrator, store_data=self.store_data)
