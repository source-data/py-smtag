import unittest

#maybe import https://github.com/pytorch/pytorch/blob/master/test/common.py and use TestCase()

class SmtagTestCase(unittest.TestCase):

    def __init__(self, methodName):
        super(SmtagTestCase, self).__init__(methodName)

    @staticmethod
    def assertTensorEqual(a, b, tolerance=1e-4):
        max_diff = a.sub(b).abs().max()
        if max_diff > tolerance:
            raise AssertionError('Max difference {:.5f} between tensors exceeds tolerance {:.5f}.'.format(max_diff, tolerance))