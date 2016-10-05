import sys
import math
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cycler import cycler
import logging
import time
import pickle as pickl
import panm_globals
import vehiclegen as vg
import vehiclefact as vf
import odeintegrator as oi
import carfollowing as cfm
from detector import Detector

# Import typing hints
from typing import List

try:
    import klepto
except ImportError:
    print("ERROR: Cannot import klepto library")
    klepto = None

try:
    import psutil
except ImportError:
    print('WARNING: Cannot import psutil, profiling information will not be available.')
    psutil = None

# Color definitions for figures
plt.rc('axes', prop_cycle=cycler('color', ['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b']))

LOGGER_NAME = "panm18"

# Create a logger object
panm_globals.LOGGER = logging.getLogger(LOGGER_NAME)
panm_globals.LOGGER.setLevel(logging.DEBUG)
# Create console handler with a lower log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# create formatters and add it to the handlers
console_formatter = logging.Formatter('%(threadName)-10s: %(levelname)-8s %(message)s')
console_handler.setFormatter(console_formatter)
# add the handlers to logger
panm_globals.LOGGER.addHandler(console_handler)

panm_globals.LOGGER.info("logging started")


def estimate_size(self):
    panm_globals.LOGGER.debug('Estimating size:')
    total_size = 0
    for var_name in self.__dict__:
        type_name = type(self.__dict__[var_name]).__name__
        elem_size = sys.getsizeof(self.__dict__[var_name])
        panm_globals.LOGGER.debug('  self.{:s} is {:s}, size {:d} bytes'.format(
            var_name, type_name, elem_size))
        if type_name == 'list':
            num_elems = len(self.__dict__[var_name])
            panm_globals.LOGGER.debug('     the list contains {:d} elements, {:f} bytes/elements'.format(
                num_elems, float(elem_size) / float(num_elems)))
        total_size += elem_size
        panm_globals.LOGGER.debug('  ** total size is approximately {:d} bytes'.format(total_size))
    return total_size


def unpickle_experiment_result(experiment_name: str, vehicle_id: int = None) -> tuple:
    file_name = experiment_name+'.pkl'
    panm_globals.LOGGER.info('Reading experiment data from `{:s}`'.format(file_name))
    if vehicle_id is not None:
        panm_globals.LOGGER.info('Will extract data for vehicle {:d}'.format(vehicle_id))
    with open(file_name, mode="rb") as pklfile:
        unpickler = pickl.Unpickler(pklfile)
        cf_model_name = unpickler.load()
        ode_integrator_name = unpickler.load()
        res_h = unpickler.load()
        res_t_max = unpickler.load()
        res_max_vh_ct = unpickler.load()
        res_tot_clck = unpickler.load()
        #
        res_num_vh = unpickler.load()
        panm_globals.LOGGER.info('Data contains {:d} vehicles'.format(res_num_vh))
        res_vehs = list()
        while res_num_vh > 0:
            vid = unpickler.load()
            # panm_globals.LOGGER.debug('Processing data for vehicle {:d}'.format(vid))
            vlen = unpickler.load()
            res_vh = vf.Vehicle(vid, vlen)
            res_vh.t_sim = unpickler.load()
            res_vh.spd = unpickler.load()
            res_vh.pos = unpickler.load()
            res_vh.gap = unpickler.load()
            res_vh.acc = unpickler.load()
            if (vehicle_id is None) or (vehicle_id == vid):
                panm_globals.LOGGER.info('Storing data for vehicle {:d}'.format(vid))
                res_vehs.append(res_vh)
            res_num_vh -= 1
    #
        panm_globals.LOGGER.info('Read experiment data from `{:s}`'.format(file_name))
    return res_h, res_t_max, res_max_vh_ct, res_vehs, res_tot_clck


def pickle_experiment_result(
        result_experiment_name: str,
        cf_mdl_name,
        ode_integ_name,
        step_h: float,
        end_time: float,
        max_vh_cnt: int,
        vehicle_list: List[vf.Vehicle],
        tot_clock: float) -> None:
    file_name = result_experiment_name + '.pkl'
    #
    process_info = None
    first_rss = 0
    vehicle_rss = 0
    if psutil is not None:
        process_info = psutil.Process()
        pinfo = process_info.memory_info()
        first_rss = pinfo.rss
        panm_globals.LOGGER.info('pickle monitor: current process memory {:d} bytes'.format(first_rss))
    #
    with open(file_name, mode="wb") as pklfile:
        pickler = pickl.Pickler(pklfile, protocol=pickl.HIGHEST_PROTOCOL)
        pickler.dump(cf_mdl_name)
        pickler.dump(ode_integ_name)
        pickler.dump(step_h)
        pickler.dump(end_time)
        pickler.dump(max_vh_cnt)
        pickler.dump(tot_clock)
        #
        if process_info is not None:
            pinfo = process_info.memory_info()
            panm_globals.LOGGER.info(
                'pickle monitor: current process memory {:d} bytes, dumping allocated {:d} bytes'.format(
                    pinfo.rss, pinfo.rss-first_rss))
        #
        pickler.dump(len(vehicle_list))
        for veh in vehicle_list:
            if process_info is not None:
                pinfo = process_info.memory_info()
                vehicle_rss = pinfo.rss
            pickler.dump(veh.id)
            pickler.dump(veh.l)
            pickler.dump(veh.t_sim)
            pickler.dump(veh.spd)
            pickler.dump(veh.pos)
            pickler.dump(veh.gap)
            pickler.dump(veh.acc)
            if process_info is not None:
                pinfo = process_info.memory_info()
                panm_globals.LOGGER.info(
                    'pickle monitor: dump of vehicle {:d} consumed {:d} bytes'.format(
                        veh.id, pinfo.rss - vehicle_rss))
                panm_globals.LOGGER.info(
                    '                current process memory {:d} bytes, dumping allocated {:d} bytes'.format(
                        pinfo.rss, pinfo.rss - first_rss))
        panm_globals.LOGGER.info('Saved experiment data to `{:s}`'.format(file_name))
    #
    if process_info is not None:
        pinfo = process_info.memory_info()
        panm_globals.LOGGER.info(
            'pickle monitor: current process memory {:d} bytes, dumping allocated {:d} bytes'.format(
                pinfo.rss, pinfo.rss - first_rss))
    #
    del pickler
    #
    if process_info is not None:
        pinfo = process_info.memory_info()
        panm_globals.LOGGER.info(
            'deleted pickler: current process memory {:d} bytes, allocated {:d} extra bytes'.format(
                pinfo.rss, pinfo.rss - first_rss))


NAMES = 'class_names'
PARAMS = 'simulation_parameters'
NUM_VEHICLES = 'number_of_vehicles'
VID_PREFIX = 'vehicle_id_'


def _vid(vid_param):
    return VID_PREFIX + str(vid_param)


def load_experiment_result(experiment_name, vehicle_id=None):
    file_name = experiment_name+'.kto'
    panm_globals.LOGGER.info('Reading experiment data from `{:s}`'.format(file_name))
    ar = klepto.archives.dir_archive(file_name)
    vehicles = list()
    if vehicle_id is None:
        ar.load()
    else:
        ar.load(NAMES)
        ar.load(PARAMS)
        vid = _vid(vehicle_id)
        ar.load(vid)
        panm_globals.LOGGER.info(str(ar.keys()))
        (vehicle_id, vehicle_l, vehicle_t_sim, vehicle_spd, vehicle_pos, vehicle_gap, vehicle_acc) = ar[vid]
        vehicle = vf.Vehicle(vehicle_id, vehicle_l)
        vehicle.t_sim = vehicle_t_sim
        vehicle.spd = vehicle_spd
        vehicle.pos = vehicle_pos
        vehicle.gap = vehicle_gap
        vehicle.acc = vehicle_acc
        vehicles.append(vehicle)
    res_h, res_t_max, res_max_vh_cnt, res_tot_clk, res_num_vh = ar[PARAMS]
    #
    panm_globals.LOGGER.info('Read experiment data from `{:s}`'.format(file_name))
    return res_h, res_t_max, res_max_vh_cnt, vehicles, res_tot_clk


def save_experiment_result(
        experiment_name,
        cf_model_name,
        ode_integrator_name,
        h,
        t_max,
        max_vehicle_count,
        vehicles,
        total_clock):
    file_name = experiment_name+'.kto'
    panm_globals.LOGGER.info('Saving experiment data to `{:s}`'.format(file_name))
    ar = klepto.archives.dir_archive(file_name, cached=False)
    ar[NAMES] = (cf_model_name, ode_integrator_name)
    ar[PARAMS] = (h, t_max, max_vehicle_count, total_clock, len(vehicles))
    for vehicle in vehicles:
        vehicle_data = (vehicle.id, vehicle.l, vehicle.t_sim, vehicle.spd, vehicle.pos, vehicle.gap, vehicle.acc)
        vid = _vid(vehicle.id)
        ar[vid] = vehicle_data
        panm_globals.LOGGER.debug('Saved vehicle id {:d}'.format(vehicle.id))
    #
    ar.dump()
    # ar.clear()
    panm_globals.LOGGER.info('Saved experiment data to `{:s}`'.format(file_name))


def get_experiment_name(cf_model_class, ode_integrator_class, h):
    return 'ex_{:s}_{:s}_{:04d}'.format(cf_model_class.shorthand(), ode_integrator_class.shorthand(), int(h * 1000))


def run_experiment(cf_model_class, ode_integrator_class, h=1.0, t_max=60.0, max_vehicle_count=9999999,
                   experiment_name=None, experiment_prefix=None, entrances=None, detector_position=None):
    #
    # Logging setup
    #
    # Construct experiment name that will be used as a part of file names generated by this experiment
    if experiment_name is None:
        if experiment_prefix is None:
            experiment_prefix = 'stopgo_'
        experiment_name = experiment_prefix + get_experiment_name(cf_model_class, ode_integrator_class, h)
    # Create file handler which logs even debug messages
    fh = logging.FileHandler(experiment_name + '.log', mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(threadName)-10s: [%(filename)s:%(lineno)4s - %(funcName)20s()] %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    #
    panm_globals.LOGGER.addHandler(fh)
    #
    panm_globals.LOGGER.info('Starting experiment ' + experiment_name)
    panm_globals.LOGGER.info('Car following integrator: ' + cf_model_class.__name__)
    panm_globals.LOGGER.info('ODE integrator: {:s} with step {:f} seconds'.format(ode_integrator_class.__name__, h))
    #
    # Bookkeeping
    #
    vehicles = list()  # type: List[vf.Vehicle]
    entrance_queue = deque()  # type: deque[vf.Vehicle]
    entrance_data = list()
    #
    entrance_generator = vg.VehicleGenerator('0201/sokp-0201-20121001.mat')
    vehicle_factory = vf.VehicleFactory(store_data=(detector_position is not None))
    #
    if detector_position is not None:
        detector = Detector(detector_position)
    else:
        detector = None
    #
    #
    next_vehicle_entrance_time = entrance_generator.generate_next_vehicle(0.0)
    #
    t_sim = 0.0
    t_max = float(t_max)
    t_alt = 0.0
    h = float(h)
    iteration_no = 0
    prev_iteration_no = 1
    #
    start_clock = time.clock()
    info_time_step = 4.0
    info_time = start_clock + info_time_step
    #
    max_vehicle_id = 0
    #
    if not (entrances is None):
        ext_veh = entrances[0]
        next_vehicle_entrance_time = ext_veh[0]
    #
    try:
        while t_sim <= t_max:
            iteration_no += 1
            panm_globals.LOGGER.debug('t_sim: {:f}, t_alt: {:f}, diff: {:g}'.format(t_sim, t_alt, t_sim - t_alt))
            # Write something to the terminal in case that we have not had any output for some time
            if time.clock() > info_time:
                ips = (iteration_no-prev_iteration_no)/info_time_step
                panm_globals.LOGGER.info('t_sim: {:9.4f}, iteration {:d}, {:7.2f} iterations/sec, {:d} vehicles'.format(
                    t_sim, iteration_no, ips, len(vehicles)))
                info_time += info_time_step
                prev_iteration_no = iteration_no
            if detector_position is not None:
                # Reduce the vehicle list by removing the vehicles that passed the detector at the targent gantry
                start_idx = 0
                for vehicle in vehicles:
                    # The list of vehicles is by definition ordered by decreasing distance to the origin, therefore
                    # the first vehicle that has not yet reached the detector means that all remaining vehicles have
                    # not reached the detector as well.
                    if vehicle.x < detector_position:
                        break
                    start_idx += 1
                    panm_globals.LOGGER.debug('t_sim: {:9.4f}, vehicle {:d} behind the detector'.format(
                        t_sim, vehicle.id))
                if start_idx > 0:
                    vehicles = vehicles[start_idx:]
                    panm_globals.LOGGER.debug('                  simulating {:d} vehicles'.format(
                        len(vehicles)))
            # Update the positions of all vehicles for time t+h
            leader = None
            for vehicle in vehicles:
                vehicle.update(t_sim, leader, detector)
                leader = vehicle
            # Entrance queue management
            while len(entrance_queue) > 0:
                leader = vehicles[-1]
                head = entrance_queue[0]
                # Force the speed of the vehicle to be the same as the speed of the leader
                if head.v > leader.v:
                    head.v = leader.v
                # If the vehicle at the beginning of the queue does not have a safe gap to the leader vehicle,
                # all other vehicles can not enter. The processing of the queue is finished for this iteration.
                if not head.has_safe_distance(leader):
                    break
                # Leader vehicle is far away enough, place the vehicle at the head of the entrance queue
                # into simulation
                head.set_initial_position(leader)
                head.record_state(t_sim, leader)
                head.next_state = head.state
                head.time_step_start = t_sim + h
                vehicles.append(head)
                # record vehicle entrance
                entrance_record = (t_sim, head.x, head.v, head.l, head.ode.model.get_params())
                entrance_data.append(entrance_record)
                panm_globals.LOGGER.debug(
                    't_sim={:8.4f} -- inserted vehicle {:d} from queue (t_zero={:8.4f}), {:d} queueing vehicles'.format(
                        t_sim, head.id, head.generated_time, len(entrance_queue)))
                # Remove the vehicle from the entrance queue
                entrance_queue.popleft()  # type: vf.Vehicle
            #
            # Generate a new vehicle if there are still some vehicles to generate
            #
            # We have not completely switched over to the next iteration step, we will therefore now generate vehicles
            # that entered the network between `t_sim` and `t_sim+h`.
            if (max_vehicle_id < max_vehicle_count) and (t_sim+h >= next_vehicle_entrance_time):
                if entrances is None:
                    # No external prescribed entrances.
                    # Generate a new vehicle
                    vehicle = vehicle_factory.generate(ode_integrator_class, cf_model_class, h)
                    # Set the time the vehicle has been generated
                    vehicle.set_generated_time(next_vehicle_entrance_time)
                    max_vehicle_id = vehicle.id
                    skip_vehicle = False
                    # If this is not the first vehicle in the lane ...
                    if len(vehicles) > 0:
                        # ... find the leader vehicle ...
                        leader = vehicles[-1]
                        # ... and test if there are some vehicles in the entrance queue.
                        if len(entrance_queue) > 0:
                            # The entrance queue is not empty, which means that the generated vehicle has to be placed
                            # directly into the tail of the entrance queue and will enter the lane as soon as all
                            # preceding vehicles have entered it.
                            panm_globals.LOGGER.debug(
                                '{:8.4f} -- generated vehicle id {:d} directly into queue'.format(
                                    next_vehicle_entrance_time, vehicle.id))
                            skip_vehicle = True
                            entrance_queue.append(vehicle)
                        # If the entrance queue is empty, check the distance of the new vehicle to its leader. If
                        # the distance is smaller than the acceptable distance for the given vehicle speed ...
                        # TODO: The leader state describes the position of the leader at `t_sim` and not at
                        # `next_vehicle_entrance_time`, meaning that the test is more conservative than it should be.
                        # It also means that the test will be quite inaccurate for large `h`.
                        elif not vehicle.has_safe_distance(leader):
                            # ... we shall probably not insert the vehicle now ...
                            skip_vehicle = True
                            # ... but we can try to change its speed to the speed of the leader and enter it anyway
                            # in case that the speed change caused the distance to become acceptable.
                            if vehicle.v > leader.v:
                                vehicle.v = leader.v
                                panm_globals.LOGGER.debug('vehicle speed changed')
                                # Recompute the safe distance for the case when the speed of both vehicles is equal
                                skip_vehicle = not vehicle.has_safe_distance(leader)
                            if skip_vehicle:
                                s = leader.x - vehicle.x - leader.l
                                panm_globals.LOGGER.debug(
                                    '{:8.4f} -- generated vehicle id {:d} too close (gap {:f} m), adding to queue'.format(
                                        next_vehicle_entrance_time, vehicle.id, s))
                                entrance_queue.append(vehicle)
                    if not skip_vehicle:
                        # The vehicle will enter the network at time `next_vehicle_entrance_time`, record the
                        # parameters at this time point
                        vehicle.record_state(next_vehicle_entrance_time, leader)
                        # Now compute the movement of the vehicle between `next_vehicle_entrance_time` and `t_sim+h`
                        # TODO
                        assert t_sim + h - next_vehicle_entrance_time > 0
                        if t_sim + h - next_vehicle_entrance_time > 1e-12:
                            # Really update the position
                            # TODO: Check the limit, 1e-12 may be too small
                            vehicle.update_from_entrance_time(t_sim, leader)
                        vehicles.append(vehicle)
                        # record vehicle entrance
                        entrance_record = (t_sim, vehicle.x, vehicle.v, vehicle.l, vehicle.ode.model.get_params())
                        entrance_data.append(entrance_record)
                        panm_globals.LOGGER.debug('{:8.4f} -- new vehicle id {:2}, speed {:7.4f}'.format(
                            next_vehicle_entrance_time, vehicle.id, vehicle.v))
                    next_vehicle_entrance_time = entrance_generator.generate_next_vehicle(next_vehicle_entrance_time)
                else:
                    # External prescribed entrances
                    while (max_vehicle_id < max_vehicle_count) and (next_vehicle_entrance_time <= t_sim):
                        max_vehicle_id += 1
                        cf_model = cf_model_class(0, 0, 0, 0, 0)
                        ode_integrator = ode_integrator_class.with_model_and_step(cf_model, h)
                        vehicle = vf.Vehicle(max_vehicle_id, ext_veh[3], ext_veh[2], ode_integrator)
                        vehicle.x = ext_veh[1]
                        vehicle.ode.model.set_params(ext_veh[4])
                        # Update vehicle position wrt to current step
                        # dt = t_sim - ext_veh[0]
                        # vehicle.x += dt*vehicle.v
                        # vehicle.set_zero_time(t_sim)
                        # TODO: Not tested yet
                        vehicle.set_generated_time(ext_veh[0])
                        vehicle.record_state(ext_veh[0], leader)
                        # Append it to queue
                        vehicles.append(vehicle)
                        panm_globals.LOGGER.debug(
                            '{:8.4f} -- new external vehicle id {:2}, speed {:7.4f}, position {:7.4f}'.format(
                                ext_veh[0], vehicle.id, vehicle.v, vehicle.x))
                        if max_vehicle_id < len(entrances):
                            ext_veh = entrances[max_vehicle_id]
                            next_vehicle_entrance_time = ext_veh[0] - 1e-6
                        else:
                            next_vehicle_entrance_time = t_max + 1000.0
            # Positions and speeds of all vehicles have been updated, make the time step
            for vehicle in vehicles:
                vehicle.move_to_next_state()
            t_alt += h
            t_sim = iteration_no*h
        #
        end_clock = time.clock()
        total_clock = end_clock-start_clock
        #
        with open(experiment_name + '_entrances.pkl', 'wb') as epf:
            pickl.dump(entrance_data, epf)
        #
        panm_globals.LOGGER.info('Size of vehicle list for {:d} vehicles: {:d} bytes'.format(
            len(vehicles), sys.getsizeof(vehicles)))
        vehicle_memory = 0
        for vehicle in vehicles:
            obj_data = sys.getsizeof(vehicle)
            spd_data = sys.getsizeof(vehicle.spd)
            pos_data = sys.getsizeof(vehicle.pos)
            gap_data = sys.getsizeof(vehicle.gap)
            tsm_data = sys.getsizeof(vehicle.t_sim)
            tot_data = obj_data+spd_data+pos_data+gap_data+tsm_data
            panm_globals.LOGGER.info('  vehicle {:3d}: {:d} bytes total'.format(vehicle.id, tot_data))
            vehicle_memory += tot_data
        #
        panm_globals.LOGGER.info('Vehicle list memory: {:d} bytes, {:f} kiB, {:f} MiB'.format(
            vehicle_memory, vehicle_memory/1024.0, vehicle_memory/(1024.0*1024.0)))
        panm_globals.LOGGER.info('Total time: {:f} seconds'.format(total_clock))
        panm_globals.LOGGER.info('Step time: {:g}'.format(h))
        panm_globals.LOGGER.info('Simulated time: {:g}'.format(t_sim))
        panm_globals.LOGGER.info('Simulated end time: {:g}'.format(t_max))
        panm_globals.LOGGER.info('Simulated time error: {:g}'.format(t_sim-t_max))
        panm_globals.LOGGER.info('Alt time error: {:g}'.format(t_alt - t_max))
        #
        start_clock = time.clock()
        pickle_experiment_result(experiment_name, cf_model_class.__name__, ode_integrator_class.__name__,
                                 h, t_max, max_vehicle_count, vehicles, total_clock)
        end_clock = time.clock()
        total_clock = end_clock - start_clock
        panm_globals.LOGGER.info('Saving took {:f} seconds'.format(total_clock))
        if detector is not None:
            #
            start_clock = time.clock()
            detector.pickle_data(experiment_name, t_sim)
            end_clock = time.clock()
            total_clock = end_clock - start_clock
            panm_globals.LOGGER.info('Saving detector data took {:f} seconds'.format(total_clock))
    except AssertionError:
        panm_globals.LOGGER.exception('Assertion in run_experiment for step {:f} iteration {:d}'.format(
            h, iteration_no))
        total_clock = None
    #
    panm_globals.LOGGER.removeHandler(fh)
    #
    return vehicles, total_clock


def run_experiment_stopgo(cf_model_class, ode_integrator_class, h=1.0, t_max=60.0, max_vehicle_count=20,
                          experiment_name=None, experiment_prefix=None):
    # Construct experiment name that will be used as a part of file names generated by this experiment
    if experiment_name is None:
        if experiment_prefix is None:
            experiment_prefix = 'stopgo_'
        experiment_name = experiment_prefix + get_experiment_name(cf_model_class, ode_integrator_class, h)
    # Create file handler which logs even debug messages
    fh = logging.FileHandler(experiment_name + '.log', mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(threadName)-10s: [%(filename)s:%(lineno)4s - %(funcName)20s()] %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    #
    panm_globals.LOGGER.addHandler(fh)
    #
    panm_globals.LOGGER.info('Starting experiment ' + experiment_name)
    panm_globals.LOGGER.info('Car following integrator: ' + cf_model_class.__name__)
    panm_globals.LOGGER.info('ODE integrator: {:s} with step {:f} seconds'.format(ode_integrator_class.__name__, h))
    #
    #  Bookkeeping
    #
    vehicles = list()  # type: List[vf.Vehicle]
    vehicle_factory = vf.VehicleFactory()
    # --------------------------------------------------------------------
    # Create a vehicle that blocks the road at 1200m
    # first_vehicle = vf.Vehicle(9999, 0.0, 0.0,
    #                            ode_integrator_class.with_model_and_step(cf_model_class, h),
    #                             distance=1200.0, stopped=True)
    # Create a vehicle that blocks the road at 670m (Section 5. Results of the original paper)
    first_vehicle = vf.Vehicle(9999, 0.0, 0.0,
                               ode_integrator_class.with_model_and_step(cf_model_class, h),
                               distance=670.0, stopped=True)
    first_vehicle.time_step_start = 0.0
    vehicles.append(first_vehicle)
    # Now generate a queue of 20 stopped vehicles
    leader = first_vehicle
    ignore_leader = True
    for i in range(max_vehicle_count):
        # Our stopgo experiment #1
        # vehicle = vehicle_factory.generate(ode_integrator_class, cf_model_class, h, actual_speed=0.0)
        # Fixed parameters according to table 1 of the paper, standard set
        # Assuming fixed vehicle length 6m
        vehicle = vehicle_factory.generate(ode_integrator_class, cf_model_class, h,
                                           desired_speed=15.0, time_gap=1.0, space_gap=2.0,
                                           accel=1.0, decel=1.5, actual_speed=0.0, vehicle_length=6.0)
        # Fixed parameters according to table 1 of the paper, creep-to-stop set
        # Assuming fixed vehicle length 6m
        # vehicle = vehicle_factory.generate(ode_integrator_class, cf_model_class, h,
        #                                    desired_speed=15.0, time_gap=1.0, space_gap=1.0,
        #                                    accel=2.0, decel=1.5, actual_speed=0.0 vehicle_length=6.0)
        if not ignore_leader and leader is not None:
            vehicle.set_initial_position(leader)
        vehicle.set_generated_time(0.0)
        vehicle.record_state(0.0, leader)
        # Conditional logging
        # if i == 9:
        #    vehicle.start_logging(leader)
        #
        vehicles.append(vehicle)
        leader = vehicle
        ignore_leader = False
    # We have set all information for t=0 already
    t_sim = 0.0
    t_max = float(t_max)
    t_alt = 0.0
    h = float(h)
    iteration_no = 0
    prev_iteration_no = 1
    #
    start_clock = time.clock()
    info_time_step = 4.0
    info_time = start_clock + info_time_step
    #
    try:
        while t_sim <= t_max:
            iteration_no += 1
            panm_globals.LOGGER.debug('t_sim: {:f}, t_alt: {:f}, diff: {:g}'.format(t_sim, t_alt, t_sim - t_alt))
            # Write something to the terminal in case that we have not had any output for some time
            if time.clock() > info_time:
                ips = (iteration_no-prev_iteration_no)/info_time_step
                panm_globals.LOGGER.info('t_sim: {:f}, iteration {:d}, {:f} iterations/sec'.format(
                    t_sim, iteration_no, ips))
                info_time += info_time_step
                prev_iteration_no = iteration_no
            # Update the positions of all vehicles for time t+h
            leader = None
            for vehicle in vehicles:
                vehicle.update(t_sim, leader)
                leader = vehicle
            # Positions and speeds of all vehicles have been updated, make the time step
            for vehicle in vehicles:
                vehicle.move_to_next_state()
            t_alt += h
            t_sim = iteration_no * h
        #
        end_clock = time.clock()
        total_clock = end_clock - start_clock
        #
        panm_globals.LOGGER.info('Size of vehicle list for {:d} vehicles: {:d} bytes'.format(
            len(vehicles), sys.getsizeof(vehicles)))
        vehicle_memory = 0
        for vehicle in vehicles:
            obj_data = sys.getsizeof(vehicle)
            spd_data = sys.getsizeof(vehicle.spd)
            pos_data = sys.getsizeof(vehicle.pos)
            gap_data = sys.getsizeof(vehicle.gap)
            tsm_data = sys.getsizeof(vehicle.t_sim)
            tot_data = obj_data + spd_data + pos_data + gap_data + tsm_data
            panm_globals.LOGGER.info('  vehicle {:3d}: {:d} bytes total'.format(vehicle.id, tot_data))
            vehicle_memory += tot_data
            panm_globals.LOGGER.info('Vehicle list memory: {:d} bytes, {:f} kiB, {:f} MiB'.format(
                vehicle_memory, vehicle_memory / 1024.0, vehicle_memory / (1024.0 * 1024.0)))
        panm_globals.LOGGER.info('Total time: {:f} seconds'.format(total_clock))
        panm_globals.LOGGER.info('Average time per iteration: {:f} seconds'.format(total_clock/iteration_no))
        panm_globals.LOGGER.info('Average time per iteration per vehicle: {:f} seconds'.format(
            total_clock/(iteration_no*max_vehicle_count)))
        panm_globals.LOGGER.info('Step time: {:g}'.format(h))
        panm_globals.LOGGER.info('Simulated time: {:g}'.format(t_sim))
        panm_globals.LOGGER.info('Simulated end time: {:g}'.format(t_max))
        panm_globals.LOGGER.info('Simulated time error: {:g}'.format(t_sim - t_max))
        panm_globals.LOGGER.info('Alt time error: {:g}'.format(t_alt - t_max))
        #
        start_clock = time.clock()
        pickle_experiment_result(experiment_name, cf_model_class.__name__, ode_integrator_class.__name__,
                                 h, t_max, max_vehicle_count, vehicles, total_clock)
        end_clock = time.clock()
        total_clock = end_clock - start_clock
        panm_globals.LOGGER.info('Saving took {:f} seconds'.format(total_clock))
    except AssertionError:
        panm_globals.LOGGER.exception('Assertion in run_experiment_stopgo for step {:f} iteration {:d}'.format(
            h, iteration_no))
        total_clock = None
    #
    panm_globals.LOGGER.removeHandler(fh)
    #
    return vehicles, total_clock


def get_experiment_data(experiment_name, vehicle_id, sample_period=0.0, min_time=0.0, max_time=0.0):
    try:
        h, t_max, max_vehicle_count, vehicles, total_clock = unpickle_experiment_result(experiment_name, vehicle_id)
        vehicle = vehicles[0]
        #
        assert vehicle.id == vehicle_id
        #
        panm_globals.LOGGER.info('got {:d} records for vehicle {:d}'.format(len(vehicle.t_sim), vehicle_id))
        #
        ref_t = vehicle.t_sim
        ref_spd = vehicle.spd
        ref_dst = vehicle.pos
        ref_gap = vehicle.gap
        if sample_period > 0:
            panm_globals.LOGGER.info('selecting only samples every {:f} seconds'.format(sample_period))
            max_row = int(t_max / sample_period) + 1
        else:
            max_row = len(ref_t)
        data = np.zeros((max_row, 4), np.float64)
        #
        row = 0
        info_time_step = 4.0
        info_time = time.clock() + info_time_step
        for i in range(len(ref_t)):
            t = ref_t[i]
            if sample_period > 0:
                do_output_this_row = False
                modulo = math.fmod(t, sample_period)
                if modulo < 1e-6 or (sample_period - modulo) < 1e-6:
                    # Recorded time is a multiple of resample_period
                    do_output_this_row = True
            else:
                do_output_this_row = True
            # For some experiments we need a subset of time range, e.g. 28.8--60.0 or 9.0--60.0
            if t < min_time:
                do_output_this_row = False
            if 0.0 < max_time < t:
                do_output_this_row = False
            # Finally store the data
            if do_output_this_row:
                data[row, :] = (t, ref_spd[i], ref_dst[i], ref_gap[i])
                panm_globals.LOGGER.debug('[{:d}] t={:4.1f}, spd={:5.2f}, dst={:5.2f}'.format(
                    row, t, ref_spd[i], ref_dst[i]))
                row += 1
            # A bit of user-friendly display
            if time.clock() > info_time:
                panm_globals.LOGGER.info('processing row {:d}/{:d}'.format(i, len(ref_t)))
                info_time += info_time_step
        panm_globals.LOGGER.info('returning total {:d} rows of data'.format(row))
        return total_clock, np.delete(data, slice(row, max_row), axis=0)
    except EOFError:
        panm_globals.LOGGER.error('Cannot load `{:s}`'.format(experiment_name))


def get_local_error_l1(experiment_data, reference_data):
    return np.fabs(experiment_data[:, 1:] - reference_data[:, 1:])


def get_global_error_l1(experiment_data, reference_data):
    return np.sum(get_local_error_l1(experiment_data, reference_data), axis=0)


def get_global_error_l2(experiment_data, reference_data):
    x = experiment_data[:, 1:] - reference_data[:, 1:]
    return np.linalg.norm(x, axis=0)/len(x)


def get_global_error_linf(experiment_data, reference_data):
    # L-infinity norm on the state variables x and v
    # We have to first compute the vector of absolute difference magnitues in Eclidean sense, and then take
    # the maximum.
    x = experiment_data[:, 1:3] - reference_data[:, 1:3]
    xn = np.linalg.norm(x, axis=1)
    return np.max(xn, axis=0)


def get_color(color: tuple, lightness: float) -> tuple:
    r, g, b = color
    if r == 0.0:
        r = lightness
    if g == 0.0:
        g = lightness
    if b == 0.0:
        b = lightness
    return r, g, b

if False:
    vehgen = vg.VehicleGenerator()
    sim_time_sec = 0.0
    vehicle_count = 0
    headway_sum = 0.0
    while sim_time_sec <= 86400.0:
        next_entrance_time = vehgen.generate_next_vehicle(sim_time_sec)
        headway = next_entrance_time - sim_time_sec
        vehicle_count += 1
        headway_sum += headway
        sim_time_sec = next_entrance_time
        panm_globals.LOGGER.info('t_sim={:6.3f}, vid={:3d}, headway={:6.4f}, avg headway={:6.3f}'.format(
            sim_time_sec, vehicle_count, headway, headway_sum / vehicle_count))

if False:
    # vehicles_ref, clock_ref = run_experiment(
    #    cfm.CarFollowingIDM, oi.RungeKutta4ODEIntegrator,
    #    h=0.0001, t_max=70.0, max_vehicle_count=11, experiment_name='reference')
    run_experiment_stopgo(
        cfm.CarFollowingIDM, oi.RungeKutta4ODEIntegrator,
        h=0.0001, t_max=100.0, experiment_name='stopgofix_reference')
    # run_experiment_stopgo(
    #    cfm.CarFollowingTest, oi.RungeKutta4ODEIntegrator,
    #    h=0.0001, t_max=100.0, max_vehicle_count=15, experiment_name='test_reference')
    # run_experiment_stopgo(
    #     cfm.CarFollowingIDM, oi.RungeKutta4ODEIntegrator,
    #     h=0.0002, t_max=100.0, max_vehicle_count=20, experiment_name='stopgofix_reference_verif')
    # vehicles1, clock1 = run_experiment(
    #   CarFollowingIDM, EulerODEIntegrator,
    #   h=0.05, t_max=240.0, max_vehicle_count=1111, experiment_name='test')
if False:
    run_experiment_stopgo(
        cfm.CarFollowingIDM, oi.RungeKutta4ODEIntegrator,
        h=0.0002, t_max=100.0, experiment_name='stopgofix_reference_verif')

# Create file handler which logs even debug messages
local_log_file_handler = logging.FileHandler('panm18.log', mode='w')
local_log_file_handler.setLevel(logging.DEBUG)
local_formatter = logging.Formatter(
    '%(asctime)s - %(threadName)-10s: [%(filename)s:%(lineno)4s - %(funcName)20s()] %(levelname)-8s %(message)s')
local_log_file_handler.setFormatter(local_formatter)
#
panm_globals.LOGGER.addHandler(local_log_file_handler)

if False:
    # Estimation of global error upper bound (equation (30)).
    global_errors = list()
    for veh_id in range(15):
        ref_time, ref_data = get_experiment_data('stopgofix_reference',
                                                 vehicle_id=veh_id + 1, sample_period=0.0002, max_time=90.0)
        ref_time_v, ref_data_v = get_experiment_data('stopgofix_reference_verif',
                                                     vehicle_id=veh_id + 1, max_time=90.0)
        gee = get_global_error_linf(ref_data_v, ref_data)
        # The value of `gee` is (speed, position, gap), where speed and position are the state variables.
        panm_globals.LOGGER.info('vehicle {:d} global error estimate {:g}'.format(veh_id, gee))
        global_errors.append(gee)
    panm_globals.LOGGER.info('Global error upper bound is {:g}'.format(max(global_errors)))
    #
    fig_err, ax_err = plt.subplots()
    ax_err.plot(global_errors, '-')
    ax_err.set_ylabel('Global error for vehicle [-]')
    ax_err.set_xlabel('Vehicle ID')
    plt.show()

if False:
    # vehicles_ref, clock_ref = run_experiment_stopgo(cfm.CarFollowingIDM, oi.RungeKutta4ODEIntegrator,
    #    h=0.002, t_max=120.0, max_vehicle_count=20)
    #
    # run_experiment(cfm.CarFollowingIDM, oi.RungeKutta4ODEIntegrator, h=0.02, t_max=120.0, max_vehicle_count=20,
    #               experiment_prefix='experiment')
    #
    run_experiment(cfm.CarFollowingIDM, oi.EulerODEIntegrator, h=0.5, t_max=86400.0, detector_position=1200.0,
                   experiment_prefix='experiment')

# times = [2.4, 1.2, 0.6, 0.3, 0.15, 0.075, 0.0375, 0.01875, 0.009375]
times = [2.4, 1.2, 0.8, 0.6, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01, 0.008, 0.006, 0.004, 0.002]
integrators = [
    oi.EulerODEIntegrator,
    oi.BallisticODEIntegrator,
    oi.TrapezoidODEIntegrator,
    oi.RungeKutta4ODEIntegrator]

if False:
    # with open('reference_entrances.pkl','r') as ref:
    #    ref_entrances = pickl.load(ref)
    for integrator in integrators:
        # integrator = oi.EulerODEIntegrator
        # integrator = oi.TrapezoidODEIntegrator
        # integrator = oi.RungeKutta4ODEIntegrator
        for th in times:
            # vehicles1, clock1 = run_experiment(cfm.CarFollowingIDM,
            #                                    integrator, h=th, t_max=70.0, entrances=ref_entrances)
            # run_experiment_stopgo(cfm.CarFollowingIDM, integrator, h=th, t_max=100.0, experiment_prefix='stopgofix_')
            # run_experiment_stopgo(cfm.CarFollowingTest, integrator, h=th, t_max=100.0, experiment_prefix='test_')
            run_experiment(cfm.CarFollowingIDM, integrator, h=th, t_max=86400.0, detector_position=1200.0,
               experiment_prefix='experiment')

if False:
    # Create complexity vs. accuracy graphs for single vehicle
    disp_t_max = 58.0
    veh_id = 10
    #
    disp_exp_prefix = 'stopgofix_'
    output_prefix = disp_exp_prefix + 'vh{:02d}_'.format(veh_id)
    #
    fig_vehicle_spd, ax_cnt = plt.subplots()
    fig_vehicle_pos, ax_vehicle_pos = plt.subplots()
    fig_spd, ax_spd = plt.subplots()
    fig_pos, ax_pos = plt.subplots()
    fig_cp_spd, ax_cp_spd = plt.subplots()
    fig_cp_pos, ax_cp_pos = plt.subplots()
    #
    integrator_colormaps = {
        oi.EulerODEIntegrator: cm.Blues,
        oi.BallisticODEIntegrator: cm.Greys,
        oi.RungeKutta4ODEIntegrator: cm.Purples,
        oi.TrapezoidODEIntegrator: cm.Greens
    }
    #
    ref_time, ref_data = get_experiment_data(disp_exp_prefix + 'reference',
                                             vehicle_id=veh_id, sample_period=2.4, max_time=58.0)
    for integrator in integrators:
        ge_data = np.zeros((len(times), 4))
        cp_data = np.zeros((len(times), 1))
        p = integrator.order()
        data_row = 0
        colormap = integrator_colormaps[integrator]
        for th in times:
            exp_name = disp_exp_prefix + get_experiment_name(cfm.CarFollowingIDM, integrator, th)
            exp_time, exp_data = get_experiment_data(exp_name, vehicle_id=veh_id, sample_period=2.4, max_time=58.0)
            colormap_val = 1.0 - float(data_row) / (len(times) + 4)
            current_color = colormap(colormap_val)
            ax_cnt.plot(exp_data[:, 0], exp_data[:, 1], ls='-', linewidth=0.5, color=current_color)
            ax_vehicle_pos.plot(exp_data[:, 0], exp_data[:, 2], ls='-', linewidth=0.5, color=current_color)
            global_error = get_global_error_l1(exp_data, ref_data)
            # global_error = get_global_error_l2(exp_data, ref_data)
            ge_data[data_row, 0] = exp_time
            ge_data[data_row, 1:4] = global_error
            #
            # complexity
            c = p/th
            cp_data[data_row] = c
            #
            data_row += 1
        #
        ax_spd.loglog(ge_data[:, 0], ge_data[:, 1], '-', label=integrator.__name__)
        ax_pos.loglog(ge_data[:, 0], ge_data[:, 2], '-', label=integrator.__name__)
        ax_cp_spd.loglog(cp_data, ge_data[:, 1], '-', label=integrator.__name__)
        ax_cp_pos.loglog(cp_data, ge_data[:, 2], '-', label=integrator.__name__)
    #
    ax_cnt.plot(ref_data[:, 0], ref_data[:, 1], color='orange', ls='--', linewidth=2.0)
    ax_vehicle_pos.plot(ref_data[:, 0], ref_data[:, 2], color='orange', ls='--', linewidth=2.0)
    #
    ax_cnt.set_xlabel('Simulated time [s]')
    ax_cnt.set_ylabel('Vehicle speed [m/s]')
    ax_vehicle_pos.set_ylabel('Vehicle position [m]')
    ax_vehicle_pos.set_xlabel('Simulated time [s]')
    #
    ax_cp_spd.set_xlabel(r'Numerical complexity [(veh$\cdot$s)$^{-1}$]')
    ax_cp_spd.set_ylabel('Global speed error [m/s]')
    ax_cp_spd.legend(loc='best')
    fig_cp_spd.suptitle('Speed error (L1) vs. complexity for vehicle {:d}'.format(veh_id))
    ax_cp_pos.set_xlabel(r'Numerical complexity [(veh$\cdot$s)$^{-1}$]')
    ax_cp_pos.set_ylabel('Global position error [m/s]')
    ax_cp_pos.legend(loc='best')
    fig_cp_pos.suptitle('Position error (L1) vs. complexity for vehicle {:d}'.format(veh_id))
    ax_spd.set_xlabel('Computational time [s]')
    ax_spd.set_ylabel('Global speed error [m/s]')
    ax_spd.legend()
    ax_pos.set_xlabel('Computational time [s]')
    ax_pos.set_ylabel('Global position error [m]')
    ax_spd.legend()
    # plt.show()
    # fig_pos.set_size_inches(15/2.54,10/2.54)
    fig_vehicle_spd.savefig(output_prefix+'spd.pdf')
    fig_vehicle_pos.savefig(output_prefix+'pos.pdf')
    fig_spd.savefig(output_prefix+'ge_spd.pdf')
    fig_pos.savefig(output_prefix+'ge_pos.pdf')
    fig_cp_spd.savefig(output_prefix+'cp_spd.pdf')
    fig_cp_pos.savefig(output_prefix+'cp_pos.pdf')

if False:
    # Create error graph for 10th vehicle
    th = 0.009375
    sample_spacing = 0.0375
    disp_t_max = 100.0
    veh_id = 10
    # integrator = oi.RungeKutta4ODEIntegrator
    disp_exp_prefix = 'test_'
    # cf_model_class = cfm.CarFollowingIDM
    cfm_class = cfm.CarFollowingTest
    #
    fig_vehicle_spd, ax_cnt = plt.subplots()
    fig_vehicle_pos, ax_vehicle_pos = plt.subplots()
    fig_vehicle_spd2, ax_vehicle_spd2 = plt.subplots()
    fig_vehicle_pos2, ax_vehicle_pos2 = plt.subplots()
    #
    ref_time, ref_data = get_experiment_data(disp_exp_prefix + 'reference',
                                             sample_period=sample_spacing, vehicle_id=veh_id, max_time=disp_t_max)
    ax_vehicle_spd2.plot(ref_data[:, 0], ref_data[:, 1], 'k-',  linewidth=2.0)
    ax_vehicle_pos2.plot(ref_data[:, 0], ref_data[:, 2], 'k-',  linewidth=2.0)
    ax_vehicle_spd_ref = ax_cnt.twinx()
    ax_vehicle_spd_ref.semilogy(ref_data[1:, 0], ref_data[1:, 1], color='orange',  linewidth=2.0, ls="--")
    ax_vehicle_spd_ref.set_ylabel('Reference speed [m/s]')
    ax_vehicle_spd_ref.set_ylim([1e-15, 1e2])
    for integrator in integrators:
        exp_name = disp_exp_prefix + get_experiment_name(cfm_class, integrator, th)
        exp_time, exp_data = get_experiment_data(exp_name,
                                                 sample_period=sample_spacing, vehicle_id=veh_id, max_time=disp_t_max)
        err_data = get_local_error_l1(ref_data, exp_data)
        #
        ax_cnt.semilogy(exp_data[:, 0], err_data[:, 0], '-', label=integrator.__name__)
        ax_vehicle_pos.semilogy(exp_data[:, 0], err_data[:, 1], '-', label=integrator.__name__)
        ax_vehicle_spd2.plot(exp_data[:, 0], exp_data[:, 1], '-', label=integrator.__name__)
        ax_vehicle_pos2.plot(exp_data[:, 0], exp_data[:, 2], '-', label=integrator.__name__)
    #
    ax_cnt.set_xlabel('Simulated time [s]')
    ax_cnt.set_ylabel('Vehicle speed error [m/s]')
    ax_cnt.legend(loc='best')
    ax_cnt.set_ylim([1e-15, 1e2])
    ax_vehicle_pos.set_xlabel('Simulated time [s]')
    ax_vehicle_pos.set_ylabel('Vehicle position error [m]')
    ax_vehicle_pos.legend(loc='best')
    #
    # plt.show()
    # fig_pos.set_size_inches(15/2.54,10/2.54)
    fig_vehicle_pos.savefig(disp_exp_prefix + 'vh10_pos_error.pdf')
    fig_vehicle_spd.savefig(disp_exp_prefix + 'vh10_spd_error.pdf')
    fig_vehicle_pos2.savefig(disp_exp_prefix + 'vh10_pos_ref.pdf')
    fig_vehicle_spd2.savefig(disp_exp_prefix + 'vh10_spd_ref.pdf')

if False:
    # For selected vehicle id, generate vehicle speed and position plots for particular integration scheme and all
    # tested integration steps
    ref_time, ref_data = get_experiment_data('stopgo_reference', vehicle_id=4)
    for integrator in integrators:
        fig_vehicle_spd, ax_cnt = plt.subplots()
        fig_vehicle_pos, ax_vehicle_pos = plt.subplots()
        data_row = 0
        for th in times:
            exp_name = 'stopgo_' + get_experiment_name(cfm.CarFollowingIDM, integrator, th)
            exp_time, exp_data = get_experiment_data(exp_name, vehicle_id=4)
            ax_cnt.plot(exp_data[:, 0], exp_data[:, 1], '-', label=th)
            ax_vehicle_pos.plot(exp_data[:, 0], exp_data[:, 2], '-', label=th)
            #
            data_row += 1
        #
        ax_cnt.plot(ref_data[:, 0], ref_data[:, 1], 'k:', linewidth=2.0, label='reference')
        ax_vehicle_pos.plot(ref_data[:, 0], ref_data[:, 2], 'k:', linewidth=2.0, label='reference')
        ax_cnt.set_xlabel('Simulated time [s]')
        ax_cnt.set_ylabel('Vehicle speed [m/s]')
        ax_cnt.legend()
        ax_vehicle_pos.set_ylabel('Vehicle position [m]')
        ax_vehicle_pos.set_xlabel('Simulated time [s]')
        ax_vehicle_pos.legend()
        #
        fig_vehicle_pos.savefig('stopgo_vh_pos_{:s}.pdf'.format(integrator.shorthand()))
        fig_vehicle_spd.savefig('stopgo_vh_spd_{:s}.pdf'.format(integrator.shorthand()))

if False:
    # Create complexity vs. accuracy graphs
    disp_t_max = 58.0
    veh_id = 10
    #
    disp_exp_prefix = 'stopgofix_'
    cfm_class = cfm.CarFollowingIDM
    #
    integrator_colormaps = {
        oi.EulerODEIntegrator: cm.Blues,
        oi.BallisticODEIntegrator: cm.Greys,
        oi.RungeKutta4ODEIntegrator: cm.Purples,
        oi.TrapezoidODEIntegrator: cm.Greens
    }
    #
    ref_time, ref_data = get_experiment_data(disp_exp_prefix + 'reference',
                                             sample_period=0.001, vehicle_id=veh_id, max_time=disp_t_max)
    for integrator in integrators:
        output_prefix = disp_exp_prefix + 'vh{:02d}_{:s}_'.format(veh_id, integrator.shorthand())
        fig_vehicle_spd, ax_cnt = plt.subplots()
        fig_vehicle_pos, ax_vehicle_pos = plt.subplots()
        pos = 0
        colormap = integrator_colormaps[integrator]
        for th in times:
            exp_name = disp_exp_prefix + get_experiment_name(cfm_class, integrator, th)
            exp_time, exp_data = get_experiment_data(exp_name, vehicle_id=veh_id, max_time=disp_t_max)
            colormap_val = 1.0-float(pos)/(len(times)+4)
            ax_cnt.plot(exp_data[:, 0], exp_data[:, 1], '-', label=th, color=colormap(colormap_val))
            ax_vehicle_pos.plot(exp_data[:, 0], exp_data[:, 2], '-', label=th, color=colormap(colormap_val))
            pos += 1
        ax_cnt.plot(ref_data[:, 0], ref_data[:, 1], color='orange', linewidth=2.0, ls="--", label='reference')
        ax_vehicle_pos.plot(ref_data[:, 0], ref_data[:, 2], color='orange', linewidth=2.0, ls="--", label='reference')
        ax_cnt.set_xlabel('Simulated time [s]')
        ax_cnt.set_ylabel('Vehicle speed [m/s]')
        ax_cnt.set_ylim([0, 16])
        ax_cnt.legend(loc='best')
        fig_vehicle_spd.suptitle('Vehicle {:d}, {:s}'.format(veh_id, integrator.__name__))
        ax_vehicle_pos.set_ylabel('Vehicle position [m]')
        ax_vehicle_pos.set_xlabel('Simulated time [s]')
        ax_vehicle_pos.legend(loc='best')
        fig_vehicle_spd.suptitle('Vehicle {:d}, {:s}'.format(veh_id, integrator.__name__))
        #
        fig_vehicle_spd.savefig(output_prefix+'spd.pdf')
        fig_vehicle_pos.savefig(output_prefix+'pos.pdf')

if False:
    display_vehicles = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30]
    _, _, _, res_vehicles, _ = unpickle_experiment_result('stopgofix_ex_idm___rk4___0009')
    fig_pos, ax_pos = plt.subplots()
    fig_spd, ax_spd = plt.subplots()
    for dv_idx in display_vehicles:
        disp_vh = res_vehicles[dv_idx]
        ax_pos.plot(disp_vh.t_sim, disp_vh.pos, '-')
        ax_spd.plot(disp_vh.t_sim, disp_vh.spd, '-')
    # plt.show(block=False)
    plt.show()

if False:
    # display_vehicles = [2, 4, 6, 8, 10]
    display_vehicles = [1, 5, 10, 15, 20]
    _, _, _, res_vehicles, _ = unpickle_experiment_result('stopgofix_ex_idm___rk4___0009')
    fig_pos, ax_pos = plt.subplots()
    fig_spd, ax_spd = plt.subplots()
    fig_gap, ax_gap = plt.subplots()
    fig_acc, ax_acc = plt.subplots()
    for dv_idx in display_vehicles:
        disp_vh = res_vehicles[dv_idx]
        ax_pos.plot(disp_vh.t_sim, disp_vh.pos, '-')
        ax_spd.plot(disp_vh.t_sim, np.array(disp_vh.spd) * 3.6, '-')
        ax_gap.plot(disp_vh.t_sim, disp_vh.gap, '-')
        ax_acc.plot(disp_vh.t_sim, disp_vh.acc, '-')
    # plt.show(block=False)
    ax_pos.set_xlim([0, 150])
    ax_pos.set_ylim([0, 700])
    ax_pos.set_ylabel('Vehicle position [m]')
    ax_spd.set_xlim([0, 150])
    ax_spd.set_ylim([0, 56])
    ax_spd.set_ylabel('Vehicle speed [km/h]')
    ax_acc.set_xlim([0, 150])
    ax_acc.set_ylim([-3, 1])
    ax_acc.set_ylabel('Vehicle acceleration [m/s2]')
    ax_gap.set_xlim([0, 150])
    ax_gap.set_ylim([0, 36])
    ax_gap.set_ylabel('Spacing [m]')
    plt.show()

if False:
    # Create detector graphs
    output_prefix = 'experimentex_'
    cfm_class = cfm.CarFollowingIDM
    detector = Detector.from_pickled_data('experimentex_idm___rk4___0500')
    #
    fig_cnt, ax_cnt = plt.subplots()
    ax_cnt.plot(detector.data, '-')
    ax_cnt.set_xlabel('Simulated time [s]')
    ax_cnt.set_ylabel('Vehicle count [-]')
    ax_cnt.set_ylim([0, 50])
    # ax_cnt.legend(loc='best')
    fig_cnt.suptitle('Detector data at km 18.7')
    #
    fig_cnt.savefig(output_prefix+'cnt_187.pdf')

if True:
    # Create detector difference graphs
    output_prefix = 'experimentex_'
    cfm_class = cfm.CarFollowingIDM
    reference = Detector.from_mat('0187/sokp-0187-20121001.mat')
    #
    fig_cnt, ax_cnt = plt.subplots()
    #
    limit = 0
    for integrator in integrators:
        exp_name = 'experiment' + get_experiment_name(cfm_class, integrator, 0.5)
        detector = Detector.from_pickled_data(exp_name)
        data_diff = np.array(detector.data[:1440],dtype='int32')-reference.data
        hits = np.sum(np.fabs(data_diff)<1)
        exp_limit = max(data_diff.max(), -data_diff.min())
        limit = max(limit, exp_limit)
        ax_cnt.plot(data_diff, '-', label=integrator.__name__)
        mse = (data_diff**2).mean()
        panm_globals.LOGGER.info('{:s} mse {:f}, exp_limit {:f}, hits {:f}'.format(
            integrator.__name__, mse, exp_limit, hits))
    ax_cnt.set_xlabel('Simulated time [s]')
    ax_cnt.set_ylabel('Vehicle count [-]')
    ax_cnt.set_xlim([0, 1439])
    ax_cnt.set_ylim([-limit-1, limit+1])
    ax_cnt.legend(loc='best')
    fig_cnt.suptitle('Detector data difference at km 18.7')
    #
    fig_cnt.savefig(output_prefix+'cntdif_0187.pdf')

