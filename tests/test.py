import unittest
import numpy as np
import sys
sys.path.append('G:/Fall2024/CMPT742/assignments/materials_a3/materials')

from dataset import default_box_generator

class TestArrayEquality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gt_boxes = np.load("GT.npy")

    @staticmethod
    def arrays_are_equal(arr1, arr2):
        if arr1.shape != arr2.shape:
            return False
        return np.allclose(np.sort(arr1, axis=0), np.sort(arr2, axis=0))
    
    def test_equal_boxes(self):
        boxes = default_box_generator([10, 5, 3, 1], [0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7])

        self.assertTrue(self.arrays_are_equal(boxes, self.gt_boxes), "Arrays should be equal")

if __name__ == '__main__':
    unittest.main()

