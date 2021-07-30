import unittest
import src.segmentationUtils.Analogy_Computing as test
import numpy as np
import math

class testAnalogy_Computing(unittest.TestCase):
    '''
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
    '''

    def setUp(self):
        #print("avant le test")
        A = np.zeros((4,3))
        B = np.zeros((4,2))
        my_str = "string"
        my_int = 2
        my_list = [1,2,3]
        my_float = math.pi

    def tearDown(self):
        #print("apres le test")
        1
    def test_dimention(self):
        self.assertEqual(1,1)

    def test_type(self):
        self.assertRaises(TypeError, test.cnnTab_to_anaTab_(my_str, my_list, my_float))

    def test_dim(self):
        self.assertRaises(ValueError, test.cnnTab_to_anaTab_(A, B, A))