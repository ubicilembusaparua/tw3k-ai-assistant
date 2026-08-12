import unittest
import os
import src
from src.storage import init_db, get_dataset_stats

TEST_DB = "test_home.db"


class TestHome(unittest.TestCase):
    def setUp(self):
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

    def tearDown(self):
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

    def test_version(self):
        self.assertEqual(src.__version__, "1.0.0")

    def test_initial_stats(self):
        init_db(TEST_DB)
        stats = get_dataset_stats(TEST_DB)
        self.assertEqual(stats.total_videos, 0)
        self.assertEqual(stats.total_chunks, 0)


if __name__ == "__main__":
    unittest.main()
