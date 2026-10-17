import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data  # noqa: E402

EXPECT = {"bnci2014": (9, 4, 22), "bnci2015": (12, 2, 13), "zhou2016": (4, 3, 14)}
DEV_K = {"bnci2014": 6, "bnci2015": 8, "zhou2016": 3}


class TestData(unittest.TestCase):
    def test_dev_part_excludes_the_holdout_people(self):
        hold = data.holdout()
        for name, K in DEV_K.items():
            d = data.load(name)                                 # part="dev" is the default
            self.assertEqual(len(d["people"]), K, name)
            self.assertFalse(set(d["people"].tolist()) & set(hold[name]), name)
            self.assertEqual(d["name"], name)
            h = data.load(name, part="holdout")
            self.assertEqual(sorted(h["people"].tolist()), sorted(hold[name]))
            self.assertEqual(h["name"], f"{name}_holdout")

    def test_shapes_and_masks(self):
        for name, (K, C, chans) in EXPECT.items():
            d = data.load(name, part="all")
            self.assertEqual((len(d["people"]), d["n_classes"], d["X"].shape[1]), (K, C, chans), name)
            self.assertEqual(d["X"].shape[2], 480)
            self.assertFalse((d["calib"] & d["test"]).any(), name)
            self.assertTrue((d["val"] <= d["calib"]).all(), name)
            for k in range(K):
                m = d["person"] == k
                self.assertGreater((m & d["calib"] & ~d["val"]).sum(), 0, (name, k))
                self.assertGreater((m & d["val"]).sum(), 0, (name, k))
                self.assertGreaterEqual((m & d["test"]).sum(), 150, (name, k))
                self.assertGreaterEqual(len(np.unique(d["unit"][m & d["calib"]])), 2, (name, k))

    def test_single_run_person_split_in_halves(self):
        d = data.load("bnci2015", part="all")
        m = (d["person"] == 0) & d["calib"]
        self.assertEqual(sorted(np.unique(d["unit"][m]).tolist()), [0, 1])
        self.assertEqual((d["unit"][m] == 0).sum(), (d["unit"][m] == 1).sum())
        self.assertTrue((d["val"][m] == (d["unit"][m] == 1)).all())   # the last unit is val

    def test_zhou_calibrates_on_two_sessions(self):
        d = data.load("zhou2016", part="all")
        raw = np.load(data.PATHS["zhou2016"], allow_pickle=True)
        self.assertEqual(sorted(np.unique(raw["session"][d["calib"]]).tolist()), [0, 1])
        self.assertEqual(np.unique(raw["session"][d["test"]]).tolist(), [2])


if __name__ == "__main__":
    unittest.main()
