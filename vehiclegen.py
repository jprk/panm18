import numpy as np
import scipy
import panm_globals

class VehicleGenerator(object):

    def __init__(self):
        self.headway_rv = None
        self.random_state = np.random.RandomState(27601)

    def generate_next_vehicle(self, time_offset):
        # TIME_STEP = 90.0
        # num_vehicles = 20
        # hsum = 0.0
        # for a in xrange(num_vehicles):
        #    headway = random.expovariate(num_vehicles/TIME_STEP)
        #    hsum = hsum + headway
        #    print a, headway, hsum
        #
        if self.headway_rv is None:
            # Constant flow 1200 vph, 20 vehicles per minute
            num_vehicles_in_slot = 20.0
            slot_length = 60.0
            # Determine the mean headway of vehicles based on input data
            mean_headway = slot_length / num_vehicles_in_slot
            panm_globals.LOGGER.info('vehicle generator initialised with mean headway {:6.4f} sec'.format(mean_headway))
            self.headway_rv = scipy.stats.expon(scale=mean_headway)
        # Compute the actual headway
        headway = self.headway_rv.rvs(random_state=self.random_state)
        #
        return time_offset + headway