import unittest

class SmtagTestCase(unittest.TestCase):

    def __init__(self, methodName):
        super(SmtagTestCase, self).__init__(methodName)

    @staticmethod
    def assertTensorEqual(a, b, tolerance=1e-4):
        return a.sub(b).abs().max() < tolerance