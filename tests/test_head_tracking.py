import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from head_tracking import HeadTracker, SceneCutDetector, TargetSelector, face_to_head
from demo import read_boxes


def face(x,y=.2):
    return [x,y,x+.1,y+.2,.95]


class TrackingTests(unittest.TestCase):
    def test_ids_survive_detector_order_and_motion(self):
        tracker=HeadTracker()
        first=tracker.update([face(.1),face(.7)])
        for step in range(1,12):
            last=tracker.update([face(.7-step*.005),face(.1+step*.005)])
            self.assertEqual([t['track_id'] for t in last],[t['track_id'] for t in first])
            self.assertLess(last[0]['bbox'][0],last[1]['bbox'][0])

    def test_short_gap_reconnect_long_gap_expires(self):
        tracker=HeadTracker(max_age=4,predict_gap=2)
        original=tracker.update([face(.1)])[0]['track_id']
        for gap in [1,2]:
            row=tracker.update([])[0]
            self.assertEqual((row['source'],row['missed_frames']),('predicted',gap))
        self.assertEqual(tracker.update([]),[])
        self.assertEqual(tracker.update([face(.1)])[0]['track_id'],original)
        for _ in range(5):
            tracker.update([])
        self.assertNotEqual(tracker.update([face(.1)])[0]['track_id'],original)

    def test_spatial_target_never_silently_switches(self):
        tracker=HeadTracker(max_age=0,predict_gap=0)
        selector=TargetSelector('rightmost')
        tracks=tracker.update([face(.1),face(.7)])
        self.assertEqual(selector.select(tracks)[0]['track_id'],2)
        self.assertEqual(selector.select(tracker.update([face(.1)])),[])
        tracker.reset();selector.reset()
        self.assertGreater(selector.select(tracker.update([face(.2)]))[0]['track_id'],2)

    def test_scene_cut_reset_does_not_recycle_ids(self):
        cut=SceneCutDetector();tracker=HeadTracker()
        self.assertFalse(cut.update(np.zeros((72,128,3),dtype=np.uint8)))
        self.assertFalse(cut.update(np.zeros((72,128,3),dtype=np.uint8)))
        self.assertTrue(cut.update(np.full((72,128,3),255,dtype=np.uint8)))
        old=tracker.update([face(.1)])[0]['track_id']
        tracker.reset()
        self.assertGreater(tracker.update([face(.1)])[0]['track_id'],old)

    def test_clips_face_boxes_at_image_boundary(self):
        box=face_to_head((0,0,20,40),100,80)
        self.assertEqual(box[0],0);self.assertEqual(box[1],0)
        self.assertTrue(np.all(box[2:]>box[:2]))
        self.assertTrue(np.all((box>=0)&(box<=1)))

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):HeadTracker(max_age=1,predict_gap=2)
        with self.assertRaises(ValueError):TargetSelector('0')
        with self.assertRaises(ValueError):HeadTracker().update([[0,0,1,1,float('nan')]])

    def test_legacy_csv_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'boxes.csv'
            path.write_text('frame,xmin,ymin,xmax,ymax\n0,0.1,0.2,0.3,0.4\n')
            np.testing.assert_allclose(read_boxes(path),[[.1,.2,.3,.4]])
            path.write_text('frame,xmin,ymin,xmax,ymax\n1,0.1,0.2,0.3,0.4\n')
            with self.assertRaises(ValueError):read_boxes(path)
            path.write_text('frame,xmin,ymin,xmax,ymax\n0,0.3,0.2,0.1,0.4\n')
            with self.assertRaises(ValueError):read_boxes(path)


if __name__=='__main__':unittest.main()
