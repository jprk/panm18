import sys
import math
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import pickle as pickl
import panm_globals
import vehiclegen as vg
import vehiclefact as vf
import odeintegrator as oi
import carfollowing as cfm
try:
    import klepto
except ImportError:
    print("ERROR: Cannot import klepto library")
import psutil


LOGGER_NAME = "panm18"

# Create a logger object
panm_globals.LOGGER = logging.getLogger(LOGGER_NAME)
panm_globals.LOGGER.setLevel(logging.DEBUG)
# Create console handler with a lower log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatters and add it to the handlers
formatter = logging.Formatter('%(threadName)-10s: %(levelname)-8s %(message)s')
ch.setFormatter(formatter)
# add the handlers to logger
panm_globals.LOGGER.addHandler(ch)

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


def unpickle_experiment_result(experiment_name, vehicle_id=None):
    file_name = experiment_name+'.pkl'
    panm_globals.LOGGER.info('Reading experiment data from `{:s}`'.format(file_name))
    with open(file_name, mode="rb") as pklfile:
        unpickler = pickl.Unpickler(pklfile)
        cf_model_name = unpickler.load()
        ode_integrator_name = unpickler.load()
        h = unpickler.load()
        t_max = unpickler.load()
        max_vehicle_count = unpickler.load()
        total_clock = unpickler.load()
        #
        num_vehicles = unpickler.load()
        panm_globals.LOGGER.info('Data contains {:d} vehicles'.format(num_vehicles))
        vehicles = list()
        while num_vehicles > 0:
            vid = unpickler.load()
            vlen = unpickler.load()
            vehicle = vf.Vehicle(vid, vlen, None, None)
            vehicle.t_sim = unpickler.load()
            vehicle.spd = unpickler.load()
            vehicle.pos = unpickler.load()
            vehicle.gap = unpickler.load()
            if (vehicle_id is None) or (vehicle_id == vid):
                vehicles.append(vehicle)
            num_vehicles -= 1
    #
        panm_globals.LOGGER.info('Read experiment data from `{:s}`'.format(file_name))
    return h, t_max, max_vehicle_count, vehicles, total_clock

def pickle_experiment_result(experiment_name, cf_model_name, ode_integrator_name, h, t_max, max_vehicle_count, vehicles, total_clock):
    file_name = experiment_name+'.pkl'
    process_info = psutil.Process()
    pinfo = process_info.memory_info()
    first_rss = pinfo.rss
    panm_globals.LOGGER.info('pickle monitor: current process memory {:d} bytes'.format(first_rss))
    with open(file_name, mode="wb") as pklfile:
        pickler = pickl.Pickler(pklfile, protocol=pickl.HIGHEST_PROTOCOL)
        pickler.dump(cf_model_name)
        pickler.dump(ode_integrator_name)
        pickler.dump(h)
        pickler.dump(t_max)
        pickler.dump(max_vehicle_count)
        pickler.dump(total_clock)
        pinfo = process_info.memory_info()
        panm_globals.LOGGER.info('pickle monitor: current process memory {:d} bytes, dumping allocated {:d} bytes'.format(
            pinfo.rss, pinfo.rss-first_rss))
        #
        pickler.dump(len(vehicles))
        for vehicle in vehicles:
            pinfo = process_info.memory_info()
            vehicle_rss = pinfo.rss
            pickler.dump(vehicle.id)
            pickler.dump(vehicle.l)
            pickler.dump(vehicle.t_sim)
            pickler.dump(vehicle.spd)
            pickler.dump(vehicle.pos)
            pickler.dump(vehicle.gap)
            pinfo = process_info.memory_info()
            panm_globals.LOGGER.info('pickle monitor: dump of vehicle {:d} consumed {:d} bytes'.format(
                vehicle.id, pinfo.rss - vehicle_rss))
            panm_globals.LOGGER.info('                current process memory {:d} bytes, dumping allocated {:d} bytes'.format(
                pinfo.rss, pinfo.rss - first_rss))
        panm_globals.LOGGER.info('Saved experiment data to `{:s}`'.format(file_name))
    pinfo = process_info.memory_info()
    panm_globals.LOGGER.info('pickle monitor: current process memory {:d} bytes, dumping allocated {:d} bytes'.format(
        pinfo.rss, pinfo.rss - first_rss))
    del pickler
    pinfo = process_info.memory_info()
    panm_globals.LOGGER.info('deleted pickler: current process memory {:d} bytes, allocated {:d} extra bytes'.format(
        pinfo.rss, pinfo.rss - first_rss))

NAMES = 'class_names'
PARAMS = 'simulation_parameters'
NUM_VEHICLES = 'number_of_vehicles'
VID_PREFIX = 'vehicle_id_'

def _vid(vid):
    return VID_PREFIX+str(vid)


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
        panm_globals.LOGGER.info( str(ar.keys()))
        (vehicle_id, vehicle_l, vehicle_t_sim, vehicle_spd, vehicle_pos, vehicle_gap) = ar[vid]
        vehicle = vf.Vehicle(vehicle_id, vehicle_l, None, None)
        vehicle.t_sim = vehicle_t_sim
        vehicle.spd = vehicle_spd
        vehicle.pos = vehicle_pos
        vehicle.gap = vehicle_gap
        vehicles.append(vehicle)
    h, t_max, max_vehicle_count, total_clock, num_vehicles = ar[PARAMS]
    #
    panm_globals.LOGGER.info('Read experiment data from `{:s}`'.format(file_name))
    return h, t_max, max_vehicle_count, vehicles, total_clock

def save_experiment_result(experiment_name, cf_model_name, ode_integrator_name, h, t_max, max_vehicle_count, vehicles, total_clock):
    file_name = experiment_name+'.kto'
    panm_globals.LOGGER.info('Saving experiment data to `{:s}`'.format(file_name))
    ar = klepto.archives.dir_archive(file_name, cached=False)
    ar[NAMES] = (cf_model_name, ode_integrator_name)
    ar[PARAMS] = (h, t_max, max_vehicle_count, total_clock, len(vehicles))
    for vehicle in vehicles:
        vehicle_data = (vehicle.id, vehicle.l, vehicle.t_sim, vehicle.spd, vehicle.pos, vehicle.gap)
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
                   experiment_name=None, entrances=None):
    if experiment_name is None:
        experiment_name = get_experiment_name(cf_model_class, ode_integrator_class, h)
    # Create file handler which logs even debug messages
    fh = logging.FileHandler(experiment_name + '.log', mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(threadName)-10s: [%(filename)s:%(lineno)4s - %(funcName)20s()] %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    #
    panm_globals.LOGGER.addHandler(fh)
    #
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatters and add it to the handlers
    formatter = logging.Formatter('%(threadName)-10s: %(levelname)-8s %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to logger
    panm_globals.LOGGER.addHandler(ch)
    #
    panm_globals.LOGGER.info('Starting experiment')
    panm_globals.LOGGER.info('Car following integrator: ' + cf_model_class.__name__)
    panm_globals.LOGGER.info('ODE integrator: ' + ode_integrator_class.__name__)
    # Bookkeeping
    vehicles = list()  # type: list[Vehicle]
    entrance_queue = deque()  # type: deque[Vehicle]
    entrance_data = list()
    #
    entrance_generator = vg.VehicleGenerator()
    vehicle_factory = vf.VehicleFactory()
    #
    next_vehicle_entrance_time = entrance_generator.generate_next_vehicle(0.0)
    t_sim = np.float64(0.0)
    t_max = np.float64(t_max)
    t_alt = np.float64(0.0)
    h = np.float64(h)
    max_vehicle_id = 0
    iteration_no = 0
    #
    start_clock = time.clock()
    info_time_step = 4.0
    info_time = start_clock
    #
    if not (entrances is None):
        ext_veh = entrances[0]
        next_vehicle_entrance_time = ext_veh[0]
    #
    while t_sim <= t_max:
        panm_globals.LOGGER.debug('t_sim: {:f}, t_alt: {:f}, diff: {:g}'.format(t_sim, t_alt, t_sim-t_alt))
        #
        if time.clock() > info_time:
            panm_globals.LOGGER.info('t_sim: {:f}, iteration {:d}'.format(t_sim, iteration_no))
            info_time += info_time_step
        # Update positions of all existing vehicles
        leader = None
        for vehicle in vehicles:
            vehicle.update(t_sim, leader)
            leader = vehicle
        # Entrance queue management
        if len(entrance_queue) > 0:
            leader = vehicles[-1]
            head = entrance_queue[0]
            # Force the speed of the vehicle to be the same as the speed of the leader
            if head.v > leader.v:
                head.v = leader.v
            if head.has_safe_distance(leader):
                # Leader vehicle is far away enough, place the vehicle at the head of the entrance queue
                # into simulation
                headx = entrance_queue.popleft() # type: vf.Vehicle
                # head.set_zero_time(t_sim)
                head.set_initial_position(leader)
                vehicles.append(head)
                # record vehicle entrance
                entrance_record = (t_sim, head.x, head.v, head.l, head.ode.model.get_params())
                entrance_data.append(entrance_record)
                panm_globals.LOGGER.debug(
                    't_sim={:8.4f} -- inserted vehicle {:d} from queue (t_zero={:8.4f}), {:d} queueing vehicles'.format(
                        t_sim, head.id, head.t_zero, len(entrance_queue)))
        # Generate a new vehicle if there are still some vehicles to generate
        if (max_vehicle_id < max_vehicle_count) and (t_sim >= next_vehicle_entrance_time):
            if entrances is None:
                vehicle = vehicle_factory.generate(ode_integrator_class, cf_model_class, h)
                vehicle.set_entrance_time(next_vehicle_entrance_time)
                max_vehicle_id = vehicle.id
                skip_vehicle = False
                if len(vehicles) > 0:
                    leader = vehicles[-1]
                    if len(entrance_queue) > 0:
                        panm_globals.LOGGER.debug('{:8.4f} -- generated vehicle id {:d} directly into queue'.format(next_vehicle_entrance_time, vehicle.id))
                        skip_vehicle = True
                        entrance_queue.append(vehicle)
                    elif not vehicle.has_safe_distance(leader):
                        # We shall probably not insert the vehicle now
                        skip_vehicle = True
                        # But we can try to alter the speed
                        if vehicle.v > leader.v:
                            vehicle.v = leader.v
                            panm_globals.LOGGER.debug('vehicle speed changed')
                            skip_vehicle = not vehicle.has_safe_distance(leader)
                        if skip_vehicle:
                            s = leader.x - vehicle.x - leader.l
                            panm_globals.LOGGER.debug('{:8.4f} -- generated vehicle id {:d} too close (gap {:f} m), adding to queue'.format(next_vehicle_entrance_time, vehicle.id, s))
                            entrance_queue.append(vehicle)
                if not skip_vehicle:
                    vehicle.set_entrance_time(next_vehicle_entrance_time)
                    vehicles.append(vehicle)
                    # record vehicle entrance
                    entrance_record = (t_sim, vehicle.x, vehicle.v, vehicle.l, vehicle.ode.model.get_params())
                    entrance_data.append(entrance_record)
                    panm_globals.LOGGER.debug('{:8.4f} -- new vehicle id {:2}, speed {:7.4f}'.format(next_vehicle_entrance_time, vehicle.id, vehicle.v))
                next_vehicle_entrance_time = entrance_generator.generate_next_vehicle(next_vehicle_entrance_time)
            else:
                # External prescribed entrances
                while (max_vehicle_id < max_vehicle_count) and (next_vehicle_entrance_time <= t_sim):
                    max_vehicle_id += 1
                    cf_model = cf_model_class(0,0,0,0,0)
                    integrator = ode_integrator_class.with_model_and_step(cf_model, h)
                    vehicle = vf.Vehicle(max_vehicle_id, ext_veh[3], ext_veh[2], integrator)
                    vehicle.x = ext_veh[1]
                    vehicle.ode.model.set_params(ext_veh[4])
                    # Update vehicle position wrt to current step
                    #dt = t_sim - ext_veh[0]
                    #vehicle.x += dt*vehicle.v
                    #vehicle.set_zero_time(t_sim)
                    vehicle.set_entrance_time(ext_veh[0])
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
        t_alt += h
        iteration_no += 1
        t_sim = iteration_no*h
    #
    end_clock = time.clock()
    total_clock = end_clock-start_clock
    #
    with open(experiment_name + '_entrances.pkl', 'w') as epf:
        pickl.dump(entrance_data, epf)
    #
    panm_globals.LOGGER.info('Size of vehicle list for {:d} vehicles: {:d} bytes'.format(len(vehicles), sys.getsizeof(vehicles)))
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
    panm_globals.LOGGER.info('Alt time error: {:g}'.format(t_alt-t_max))
    #
    panm_globals.LOGGER.removeHandler(fh)
    #
    save_experiment_result(experiment_name, cf_model_class.__name__, ode_integrator_class.__name__, h, t_max, max_vehicle_count, vehicles, total_clock)
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
    vehicles = list()  # type: list[vf.Vehicle]
    vehicle_factory = vf.VehicleFactory()
    # --------------------------------------------------------------------
    # Create a vehicle that blocks the road at 1200m
    # first_vehicle = vf.Vehicle(9999, 0.0, 0.0, ode_integrator_class.with_model_and_step(cf_model_class, h), distance=1200.0, stopped=True)
    # Create a vehicle that blocks the road at 670m (Section 5. Results of the original paper)
    first_vehicle = vf.Vehicle(9999, 0.0, 0.0, ode_integrator_class.with_model_and_step(cf_model_class, h), distance=1200.0, stopped=True)
    vehicles.append(first_vehicle)
    # Now generate a queue of 20 stopped vehicles
    leader = None
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
        vehicle.set_entrance_time(0.0)
        if leader is not None:
            vehicle.set_initial_position(leader)
        vehicles.append(vehicle)
        leader = vehicle
    #
    t_sim = 0.0
    t_max = float(t_max)
    t_alt = 0.0
    h = float(h)
    iteration_no = 0
    prev_iteration_no = 0
    #
    start_clock = time.clock()
    info_time_step = 4.0
    info_time = start_clock + info_time_step
    #
    try:
        while t_sim <= t_max:
            panm_globals.LOGGER.debug('t_sim: {:f}, t_alt: {:f}, diff: {:g}'.format(t_sim, t_alt, t_sim - t_alt))
            # Update positions of all existing vehicles
            if time.clock() > info_time:
                ips = (iteration_no-prev_iteration_no)/info_time_step
                panm_globals.LOGGER.info('t_sim: {:f}, iteration {:d}, {:f} iterations/sec'.format(t_sim, iteration_no, ips))
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
            iteration_no += 1
            t_sim = iteration_no * h
        #
        end_clock = time.clock()
        total_clock = end_clock - start_clock
        #
        panm_globals.LOGGER.info('Size of vehicle list for {:d} vehicles: {:d} bytes'.format(len(vehicles), sys.getsizeof(vehicles)))
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
        panm_globals.LOGGER.info('Step time: {:g}'.format(h))
        panm_globals.LOGGER.info('Simulated time: {:g}'.format(t_sim))
        panm_globals.LOGGER.info('Simulated end time: {:g}'.format(t_max))
        panm_globals.LOGGER.info('Simulated time error: {:g}'.format(t_sim - t_max))
        panm_globals.LOGGER.info('Alt time error: {:g}'.format(t_alt - t_max))
        #
        start_clock = time.clock()
        pickle_experiment_result(experiment_name, cf_model_class.__name__, ode_integrator_class.__name__, h, t_max,
                               max_vehicle_count, vehicles, total_clock)
        end_clock = time.clock()
        total_clock = end_clock - start_clock
        panm_globals.LOGGER.info('Saving took {:f} seconds'.format(total_clock))
    except AssertionError:
        panm_globals.LOGGER.exception('Assertion in run_experiment_stopgo for step {:f} iteration {:d}'.format(h, iteration_no))
        total_clock = None
    #
    panm_globals.LOGGER.removeHandler(fh)
    #
    return vehicles, total_clock


def get_experiment_data(experiment_name, vehicle_id, resample_period=0.0, min_time=0.0, max_time=0.0):
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
        if resample_period > 0:
            max_row = int(t_max/resample_period)+1
        else:
            max_row = len(ref_t)
        data = np.zeros((max_row,4),np.float64)
        #
        row = 0
        info_time_step = 4.0
        info_time = time.clock() + info_time_step
        for i in range(len(ref_t)):
            t = ref_t[i]
            if resample_period > 0:
                do_output_this_row = False
                modulo = math.fmod(t, resample_period)
                if modulo < 1e-6 or (resample_period - modulo) < 1e-6:
                    # Recorded time is a multiple of resample_period
                    do_output_this_row = True
            else:
                do_output_this_row = True
            # For some experiments we need a subset of time range, e.g. 28.8--60.0 or 9.0--60.0
            if t < min_time:
                do_output_this_row = False
            if max_time > 0.0 and t > max_time:
                do_output_this_row = False
            # Finally store the data
            if do_output_this_row:
                data[row,:] = (t, ref_spd[i], ref_dst[i], ref_gap[i])
                panm_globals.LOGGER.debug('[{:d}] t={:4.1f}, spd={:5.2f}, dst={:5.2f}'.format(row, t, ref_spd[i], ref_dst[i]))
                row += 1
            # A bit of user-friendly display
            if time.clock() > info_time:
                panm_globals.LOGGER.info('processing row {:d}/{:d}'.format(i, len(ref_t)))
                info_time += info_time_step
        panm_globals.LOGGER.info('returning total {:d} rows of data'.format(row))
        return total_clock, np.delete(data,slice(row,max_row),axis=0)
    except EOFError:
        panm_globals.LOGGER.error('Cannot load `{:s}`'.format(experiment_name))


def get_global_error_l1(exp_data, ref_data):
    return np.sum(np.fabs(exp_data[:,1:]-ref_data[:,1:]), axis=0)

def get_global_error_linf(exp_data, ref_data):
    # L-infinity norm on the state variables x and v
    # We have to first compute the vector of absolute difference magnitues in Eclidean sense, and then take
    # the maximum.
    x = exp_data[:,1:3]-ref_data[:,1:3]
    xn = np.linalg.norm(x, axis=1)
    return np.max(xn, axis=0)

if False:
    vehgen = vg.VehicleGenerator()
    t_sim = 0.0
    vehicle_count = 0
    headway_sum = 0.0
    while t_sim <= 86400.0:
        next_entrance_time = vehgen.generate_next_vehicle(t_sim)
        headway = next_entrance_time - t_sim
        vehicle_count += 1
        headway_sum += headway
        t_sim = next_entrance_time
        panm_globals.LOGGER.info('t_sim={:6.3f}, vid={:3d}, headway={:6.4f}, avg headway={:6.3f}'.format(
            t_sim, vehicle_count, headway, headway_sum/vehicle_count))

if False:
    #vehicles_ref, clock_ref = run_experiment(
    #    cfm.CarFollowingIDM, oi.RungeKutta4ODEIntegrator,
    #    h=0.0001, t_max=70.0, max_vehicle_count=11, experiment_name='reference')
    run_experiment_stopgo(
        cfm.CarFollowingIDM, oi.RungeKutta4ODEIntegrator,
        h=0.0001, t_max=80.0, max_vehicle_count=10, experiment_name='stopgo_reference')
    run_experiment_stopgo(
        cfm.CarFollowingIDM, oi.RungeKutta4ODEIntegrator,
        h=0.0002, t_max=80.0, max_vehicle_count=10, experiment_name='stopgo_reference_verif')
    # vehicles1, clock1 = run_experiment(
    #   CarFollowingIDM, EulerODEIntegrator,
    #   h=0.05, t_max=240.0, max_vehicle_count=1111, experiment_name='test')

if False:
    # Estimation of global error upper bound (equation (30)).
    global_errors = list()
    for veh_id in range(10):
        ref_time, ref_data = get_experiment_data('stopgo_reference',
                                                 vehicle_id=veh_id+1, resample_period=0.0002, max_time=80.0)
        ref_time_v, ref_data_v = get_experiment_data('stopgo_reference_verif',
                                                     vehicle_id=veh_id+1, max_time=80.0)
        gee = get_global_error_linf(ref_data_v, ref_data)
        # The value of `gee` is (speed, position, gap), where speed and position are the state variables.
        panm_globals.LOGGER.info('vehicle {:d} global error estimate {:g}'.format(veh_id, gee))
        global_errors.append(gee)
    panm_globals.LOGGER.info('Global error upper bound is {:g}'.format(max(global_errors)))

if False:
    vehicles_ref, clock_ref = run_experiment_stopgo(cfm.CarFollowingIDM, oi.RungeKutta4ODEIntegrator,
        h=0.002, t_max=120.0, max_vehicle_count=20)

times = [2.4, 1.2, 0.6, 0.3, 0.15, 0.075, 0.0375, 0.01875, 0.009375]
integrators = [oi.EulerODEIntegrator, oi.TrapezoidODEIntegrator, oi.RungeKutta4ODEIntegrator, oi.BallisticODEIntegrator]

if False:
    # with open('reference_entrances.pkl','r') as ref:
    #    ref_entrances = pickl.load(ref)
    for integrator in integrators:
        # integrator = oi.TrapezoidODEIntegrator
        # integrator = oi.RungeKutta4ODEIntegrator
        for th in times:
            #vehicles1, clock1 = run_experiment(cfm.CarFollowingIDM, integrator, h=th, t_max=70.0, entrances=ref_entrances)
            run_experiment_stopgo(cfm.CarFollowingIDM, integrator, h=th, t_max=80.0)

if True:
    # Create complexity vs. accuracy graphs
    fig_vehicle_spd, ax_vehicle_spd = plt.subplots()
    fig_vehicle_pos, ax_vehicle_pos = plt.subplots()
    fig_spd, ax_spd = plt.subplots()
    fig_pos, ax_pos = plt.subplots()
    fig_cp_spd, ax_cp_spd = plt.subplots()
    fig_cp_pos, ax_cp_pos = plt.subplots()
    ref_time, ref_data = get_experiment_data('stopgo_reference', vehicle_id=4, resample_period=2.4, max_time=80.0)
    for integrator in integrators:
        ge_data = np.zeros((len(times),4))
        cp_data = np.zeros((len(times),1))
        p = integrator.order()
        row = 0
        for th in times:
            exp_name = 'stopgo_' + get_experiment_name(cfm.CarFollowingIDM, integrator, th)
            exp_time, exp_data = get_experiment_data(exp_name, vehicle_id=4, resample_period=2.4, max_time=80.0)
            ax_vehicle_spd.plot(exp_data[:,0], exp_data[:,1], '-')
            ax_vehicle_pos.plot(exp_data[:,0], exp_data[:,2], '-')
            global_error = get_global_error_l1(exp_data, ref_data)
            ge_data[row,0] = exp_time
            ge_data[row,1:4] = global_error
            #
            # complexity
            c = p/th
            cp_data[row] = c
            #
            row += 1
        ax_spd.loglog(ge_data[:,0], ge_data[:,1], '-', label=integrator.__name__)
        ax_pos.loglog(ge_data[:, 0], ge_data[:, 2], '-', label=integrator.__name__)
        ax_cp_spd.loglog(cp_data, ge_data[:,1], '-', label=integrator.__name__)
        ax_cp_pos.loglog(cp_data, ge_data[:, 1], '-', label=integrator.__name__)
    #
    ax_vehicle_spd.plot(ref_data[:, 0], ref_data[:, 1], 'k-', linewidth=2.0)
    ax_vehicle_pos.plot(ref_data[:, 0], ref_data[:, 2], 'k-', linewidth=2.0)
    ax_vehicle_spd.set_xlabel('Simulated time [s]')
    ax_vehicle_spd.set_ylabel('Vehicle speed [m/s]')
    ax_vehicle_pos.set_ylabel('Vehicle position [m]')
    ax_vehicle_pos.set_xlabel('Simulated time [s]')
    #
    ax_cp_spd.set_xlabel('Numerical complexity [(veh s)^-1]')
    ax_cp_spd.set_ylabel('Global speed error [m/s]')
    ax_cp_spd.legend()
    ax_cp_pos.set_xlabel('Numerical complexity [(veh s)^-1]')
    ax_cp_pos.set_ylabel('Global position error [m/s]')
    ax_cp_pos.legend()
    ax_spd.set_xlabel('Computational time [s]')
    ax_spd.set_ylabel('Global speed error [m/s]')
    ax_spd.legend()
    ax_pos.set_ylabel('Global position error [m]')
    ax_pos.set_xlabel('Computational time [s]')
    ax_spd.legend()
    # plt.show()
    # fig_pos.set_size_inches(15/2.54,10/2.54)
    fig_vehicle_pos.savefig('stopgo_vh_pos.pdf')
    fig_vehicle_spd.savefig('stopgo_vh_spd.pdf')
    fig_pos.savefig('stopgo_ge_pos.pdf')
    fig_spd.savefig('stopgo_ge_spd.pdf')
    fig_cp_spd.savefig('stopgo_cp_spd.pdf')
    fig_cp_pos.savefig('stopgo_cp_pos.pdf')

if False:
    # For selected vehicle id, generate vehicle speed and position plots for particular integration scheme and all
    # tested integration steps
    ref_time, ref_data = get_experiment_data('stopgo_reference', vehicle_id=4)
    for integrator in integrators:
        fig_vehicle_spd, ax_vehicle_spd = plt.subplots()
        fig_vehicle_pos, ax_vehicle_pos = plt.subplots()
        row = 0
        for th in times:
            exp_name = 'stopgo_' + get_experiment_name(cfm.CarFollowingIDM, integrator, th)
            exp_time, exp_data = get_experiment_data(exp_name, vehicle_id=4)
            ax_vehicle_spd.plot(exp_data[:,0], exp_data[:,1], '-', label=th)
            ax_vehicle_pos.plot(exp_data[:,0], exp_data[:,2], '-', label=th)
            #
            row += 1
        #
        ax_vehicle_spd.plot(ref_data[:, 0], ref_data[:, 1], 'k:', linewidth=2.0, label='reference')
        ax_vehicle_pos.plot(ref_data[:, 0], ref_data[:, 2], 'k:', linewidth=2.0, label='reference')
        ax_vehicle_spd.set_xlabel('Simulated time [s]')
        ax_vehicle_spd.set_ylabel('Vehicle speed [m/s]')
        ax_vehicle_spd.legend()
        ax_vehicle_pos.set_ylabel('Vehicle position [m]')
        ax_vehicle_pos.set_xlabel('Simulated time [s]')
        ax_vehicle_pos.legend()
        #
        fig_vehicle_pos.savefig('stopgo_vh_pos_{:s}.pdf'.format(integrator.shorthand()))
        fig_vehicle_spd.savefig('stopgo_vh_spd_{:s}.pdf'.format(integrator.shorthand()))

if False:
    # integrator = oi.EulerODEIntegrator
    integrator = oi.TrapezoidODEIntegrator
    shorthand = integrator.shorthand()
    for th in times:
        exp_base_name = get_experiment_name(cfm.CarFollowingIDM, integrator, th)
        exp_name = 'stopgo_' + exp_base_name
        exp_time, exp_data = get_experiment_data(exp_name, vehicle_id=1)
        fig_vehicle_spd, ax_vehicle_spd = plt.subplots()
        fig_vehicle_pos, ax_vehicle_pos = plt.subplots()
        ax_vehicle_spd.plot(exp_data[:,0], exp_data[:,1], '-', label=th)
        ax_vehicle_pos.plot(exp_data[:,0], exp_data[:,2], '-', label=th)
        ax_vehicle_spd.set_xlabel('Simulated time [s]')
        ax_vehicle_spd.set_ylabel('Vehicle speed [m/s]')
        ax_vehicle_spd.legend()
        ax_vehicle_pos.set_ylabel('Vehicle position [m]')
        ax_vehicle_pos.set_xlabel('Simulated time [s]')
        ax_vehicle_pos.legend()
        #
        fig_vehicle_pos.savefig('stopgo_vh1_pos_'+exp_base_name+'.pdf')
        fig_vehicle_spd.savefig('stopgo_vh1_spd_'+exp_base_name+'.pdf')

if False:
    display_vehicles = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30]
    fig_pos, ax_pos = plt.subplots()
    fig_spd, ax_spd = plt.subplots()
    for dv in display_vehicles:
        vehicle = vehicles1[dv]
        ax_pos.plot(vehicle.t_sim, vehicle.pos, '-')
        ax_spd.plot(vehicle.t_sim, vehicle.spd, '-')
    # plt.show(block=False)
    plt.show()

if False:
    display_vehicles = [0, 1, 2, 4, 6, 8, 10]
    h, t_max, max_vehicle_count, vehicles, total_clock = unpickle_experiment_result('stopgo_ex_idm___rk4___0002')
    fig_pos, ax_pos = plt.subplots()
    fig_spd, ax_spd = plt.subplots()
    for dv in display_vehicles:
        vehicle = vehicles[dv]
        ax_pos.plot(vehicle.t_sim, vehicle.pos, '-')
        ax_spd.plot(vehicle.t_sim, vehicle.spd, '-')
    # plt.show(block=False)
    plt.show()
