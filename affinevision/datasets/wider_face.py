import cv2
import numpy as np
import tqdm
from pathlib import Path

from .base import TDataset


class WiderFace(TDataset):
    def __init__(self, data_dir, is_test=False):
        self.is_test = is_test
        self.data = list(tqdm.tqdm(self._parse(Path(data_dir)), desc=f'Parsing {data_dir}'))
    
    def __len__(self,):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        item['image'] = cv2.imread(str(item['path']), cv2.IMREAD_COLOR)
        return item
    
    def _parse(self, data_dir):
        label_file = data_dir / 'label.txt'
        image_dir = data_dir / 'images'
        with open(label_file, 'rt') as f:
            lines = f.readlines()
        i = 0

        while i < len(lines):
            path = lines[i]
            assert path.startswith('#')
            path = path[1:].strip()
            i += 1
            fullpath = image_dir / path
          
            item = {
                'path': str(fullpath),
            }

            if self.is_test:
                yield item

            bboxes = []
            keypoints = []
            flags = []
            while i < len(lines) and not lines[i].startswith('#'):
                records = lines[i].strip().split()
                records = np.array(records).astype(np.float32)
                bbox = records[:4]
                bbox[2:] += bbox[:2]
                bboxes.append(bbox)

                if records.size > 4:
                    keypoints_flag = np.array(records[4:-1]).astype(np.float32).reshape(-1, 3)
                    points = keypoints_flag[:, :2]
                    has_keypoints = (keypoints_flag[:, -1] >= 0).all()
                    keypoints.append(points)
                    flags.append(has_keypoints)
                i += 1
            
            if bboxes:
                bboxes = np.stack(bboxes, axis=0)
                item.update(bboxes=bboxes)
            
            if keypoints:
                keypoints = np.stack(keypoints, axis=0)
                item.update(keypoints=keypoints, flags=np.array(flags))
            
            yield item
