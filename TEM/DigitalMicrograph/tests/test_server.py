import unittest
import Pyro5.api
import sys
sys.path.append('.')
sys.path.append('./tests')
from server_software.server import ArrayServer

class TestArrayServer(unittest.TestCase):

    def test_get_array(self):
        server = ArrayServer()
        array_list, shape, dtype = server.get_array()
        self.assertEqual(shape, (1024, 1024))
        self.assertEqual(dtype, 'int16')

    def test_serialize_array(self):
        import numpy as np
        from server_software.server import serialize_array
        server = ArrayServer()
        array = np.array([[1, 2], [3, 4]])
        array_list, shape, dtype = serialize_array(array)
        self.assertEqual(array_list, [[1, 2], [3, 4]])
        self.assertEqual(shape, (2, 2))
        self.assertEqual(dtype, 'int64')

if __name__ == '__main__':
    unittest.main()