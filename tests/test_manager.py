''' 
Tests for manager
'''

from clmm.core import manager
from clmm.core.datatypes import *
from clmm.galaxycluster import *

# Function specifications
function_specs = {'test_func_%d'%i:True for i in range(3)}

# Data input specifications
test_specs = {'test%d'%i:True for i in range(3)}
test_table = []
test_data = GCData('test_data', test_specs, test_table)

test_packed_data = GCData('function_to_test', function_specs, test_table)

manager_guy = manager.Manager({'function_to_test':test_data})

# Galaxy cluster example data
test_gc = GalaxyCluster('test_cluster', test_data)
test_gc_out = GalaxyCluster('test_cluster', test_data)
test_gc_out.add_data(test_packed_data)

def function_to_test(data, **argv):
    print('*** Here is your data ***')
    print(data)
    print('* and auxiliary args:')
    for a, i in argv.items():
        print(a, i)
    print('*************************')
    return

from numpy import testing as tst

def test_signcreator():
    tst.assert_equal(manager_guy._signcreator(function_to_test), 'function_to_test')

def test_signspecs():
    tst.assert_equal(manager_guy._signspecs(function_specs), function_specs)

def test_pack() :
    tst.assert_equal(manager_guy._pack(function_to_test, function_specs, test_table), test_packed_data)

def test_unpack() :
    print(manager_guy._unpack(test_gc, function_to_test, function_specs))
    print(test_data.values)
    tst.assert_equal(manager_guy._unpack(test_gc, function_to_test, function_specs), test_data.values)

def test_apply() :
    manager_guy.apply(test_gc, function_to_test, function_specs, test_specs)
    #tst.assert_equal(test_gc, test_gc_out)
    for d1, d2 in zip(test_gc.data, test_gc_out.data):
        print(d1, d2)
        tst.assert_equal(d1, d2)

def test_prepare() :
    pass

def test_deliver() :
    pass


if __name__ == "__main__" :
    test_signcreator()
    test_signspecs()
    test_pack()
    test_unpack()
    test_apply()

