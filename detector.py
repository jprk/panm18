from array import array
import pickle
import numpy as np
import scipy.io as sio
import os.path
import panm_globals

DETECTOR_PICKLE_SUFFIX = '_detector'


class Detector(object, panm_globals.MatfileSOKPReader):

    _id = 0

    def __init__(self, distance: float = np.nan, aggregation_period: float = 60) -> None:
        self._id += 1
        self.id = self._id
        self.x = distance
        self.time_step = aggregation_period
        self.data = array('I')
        self.slot_end_time = 0.0

    @classmethod
    def from_pickled_data(cls, experiment_name: str) -> 'Detector':
        detector = cls()
        with open(experiment_name + DETECTOR_PICKLE_SUFFIX + '.pkl', 'rb') as epf:
            detector.id = pickle.load(epf)
            detector.x = pickle.load(epf)
            detector.time_step = pickle.load(epf)
            detector.slot_end_time = pickle.load(epf)
            detector.data = pickle.load(epf)
        return detector

    @classmethod
    def from_mat(cls, file_name: str) -> 'Detector':
        detector = cls()
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()
        file_name = os.path.join('vehiclegen', file_name)
        if ext == '.mat':
            detector.read_mat(file_name)
        else:
            raise TypeError('unsupported input file type `{:s}`'.format(file_name))
        #
        return detector

    def _append_slots_until(self, t_sim: float) -> None:
        while t_sim > self.slot_end_time:
            self.data.append(0)
            self.slot_end_time += self.time_step

    def record_vehicle(self, t_sim: float) -> None:
        self._append_slots_until(t_sim)
        self.data[-1] += 1

    def pickle_data(self, experiment_name: str, t_sim: float) -> None:
        self._append_slots_until(t_sim)
        with open(experiment_name + DETECTOR_PICKLE_SUFFIX + '.pkl', 'wb') as epf:
            pickle.dump(self.id, epf)
            pickle.dump(self.x, epf)
            pickle.dump(self.time_step, epf)
            pickle.dump(self.slot_end_time, epf)
            pickle.dump(self.data, epf)
