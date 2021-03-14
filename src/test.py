import unittest
from dispatchers import grid_dispatcher
from dispatchers import model_dispatcher
import config
import files
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

class TestGridDispatcher(unittest.TestCase):

    def test_get_havling(self):
        grid = grid_dispatcher.grids["halving"]
        self.assertEqual(type(grid), HalvingRandomSearchCV)

    def test_get_grid_search(self):
        grid = grid_dispatcher.grids["grid"]
        self.assertEqual(type(grid), GridSearchCV)


class TestModelDispatcher(unittest.TestCase):

    def test_get_rf(self):
        clf = model_dispatcher.models["rf"]
        self.assertEqual(type(clf), RandomForestClassifier)

    def test_get_shallow_rf(self):
        clf = model_dispatcher.models["rf_shallow"]
        self.assertEqual(type(clf), RandomForestClassifier)
        self.assertEqual(clf.max_depth, 3)


class TestFiles(unittest.TestCase):

    def test_extract_filename(self):
        filename = "this_is_a_file.bin"
        class_name = files.extract_class_from_training_file(filename)
        self.assertEqual(class_name, "file")


if __name__ == '__main__':
    unittest.main()