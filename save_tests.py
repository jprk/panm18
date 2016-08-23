import cPickle as pickl
import klepto
import time
import random
import sys
import numpy as np
import struct

class TestClass(object):

    def __init__(self, num_elements):
        self.pos = list()
        self.spd = list()
        self.gap = list()
        for i in xrange(num_elements):
            n1 = random.random()
            self.pos.append(n1)
            self.spd.append(n1*n1)
            self.gap.append(n1*n1+n1)

    def estimate_size(self):
        print('Estimating size:')
        total_size = 0
        for var_name in self.__dict__:
            type_name = type(self.__dict__[var_name]).__name__
            elem_size = sys.getsizeof(self.__dict__[var_name])
            print('  self.{:s} is {:s}, size {:d} bytes'.format(
                var_name, type_name, elem_size))
            if type_name == 'list':
                num_elems = len(self.__dict__[var_name])
                print('     the list contains {:d} elements, {:f} bytes/elements'.format(
                    num_elems, float(elem_size)/float(num_elems)))
            total_size += elem_size
        print('  ** total size is approximately {:d} bytes'.format(total_size))
        return total_size

def save_with_pickle(archive_name, data):
    file_name = archive_name + '.pkl'
    t1 = time.clock()
    with open(file_name, mode="wb") as pklfile:
        pickler = pickl.Pickler(pklfile, protocol=pickl.HIGHEST_PROTOCOL)
        pickler.dump(data)
    t2 = time.clock()
    total_clock = t2-t1
    print ('Saved list of {:d}*{:s} to `{:s}` in {:7.4} seconds'.format(len(data), type(data[0]).__name__, file_name, total_clock))

def save_with_klepto_dir_archive(archive_name, data):
    file_name = archive_name + '.kto'
    t1 = time.clock()
    ar = klepto.archives.dir_archive(file_name, cached=False)
    ar['data'] = data
    ar.dump()
    t2 = time.clock()
    total_clock = t2-t1
    print ('Saved list of {:d}*{:s} to `{:s}` in {:7.4} seconds'.format(len(data), type(data[0]).__name__, file_name, total_clock))


data = list()
for i in xrange(1200000):
    num = random.random()
    data.append(num)

save_with_pickle('test_plain_list', data)
save_with_klepto_dir_archive('test_plain_list', data)

data = list()
list_size = 0
for i in xrange(10):
    tc = TestClass(60*10000)
    data.append(tc)
    list_size += tc.estimate_size()

print('Total size of class list approximately {:d} bytes'.format(list_size))

save_with_pickle('c:/temp/test_class_list', data)
save_with_klepto_dir_archive('c:/temp/test_class_list', data)


