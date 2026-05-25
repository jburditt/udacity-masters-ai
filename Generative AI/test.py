import unittest
import util

class TestFactorial(unittest.TestCase):
    def test_positive_integer(self):
        self.assertEqual(util.factorial(5), 120)

    def test_zero(self):
        self.assertEqual(util.factorial(0), 1)

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            util.factorial(-3)

# Run the tests
unittest.main(argv=["notebook-tests"], exit=False)