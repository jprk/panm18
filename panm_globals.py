import scipy.io as sio

LOGGER = None

class MatfileSOKPReader(object):

    def read_mat(self, file_name: str) -> 'MatfileSOKPReader':
        temp_data = sio.loadmat(file_name)
        # Check if the loaded dictionary has a field named `data`
        if 'data' in temp_data:
            # Looks like a Eltodo SOKP format
            gantry_data_struct = temp_data['data']
            # If it is the format in question, it will have the following field set
            field_list = ['date', 'time_step', 'gantry_id', 'lanemap', 'los', 'cnt', 'occ', 'spd']
            field_set = set(field_list)
            gantry_data_field_set = set(gantry_data_struct.dtype.names)
            if field_set == gantry_data_field_set:
                # Yes, the file has the expected set of fields
                date = str(gantry_data_struct[0, 0]['date'])
                self.time_step = int(gantry_data_struct[0, 0]['time_step'])
                vehicle_counts = gantry_data_struct[0, 0]['cnt']
                pers_vehicles = vehicle_counts[2, 0]
                pv_shape = pers_vehicles.shape
                LOGGER.info('Data {:s} are in SOKP format, date {:s}, time step {:d} seconds'.format(
                    file_name, date, self.time_step))
                self.data = pers_vehicles[:, 0]  # remember that the values are uint8
                del temp_data, gantry_data_struct, vehicle_counts, pers_vehicles
