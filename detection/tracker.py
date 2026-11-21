"""
SiteGuard AI - Centroid and IoU Worker Tracker
Maintains consistent identity IDs for workers across video/webcam frames.
Prevents duplicate violation counts and provides persistent bounding box tracks.
"""

import numpy as np

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxB[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    if unionArea <= 0:
        return 0.0
    return interArea / unionArea

class WorkerTracker:
    def __init__(self, max_disappeared=30, iou_threshold=0.3):
        self.next_id = 1
        self.objects = {}       # id -> box [x1, y1, x2, y2]
        self.disappeared = {}   # id -> frame_count
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold

    def register(self, box):
        self.objects[self.next_id] = box
        self.disappeared[self.next_id] = 0
        obj_id = self.next_id
        self.next_id += 1
        return obj_id

    def deregister(self, obj_id):
        if obj_id in self.objects:
            del self.objects[obj_id]
        if obj_id in self.disappeared:
            del self.disappeared[obj_id]

    def update(self, detected_boxes):
        """
        Updates tracked workers with newly detected person boxes.
        Returns: list of dicts: [{'id': int, 'box': [x1, y1, x2, y2]}]
        """
        if len(detected_boxes) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return []

        if len(self.objects) == 0:
            tracked = []
            for box in detected_boxes:
                obj_id = self.register(box)
                tracked.append({'id': obj_id, 'box': box})
            return tracked

        # Match existing objects to detected boxes using IoU
        object_ids = list(self.objects.keys())
        prev_boxes = [self.objects[oid] for oid in object_ids]

        iou_matrix = np.zeros((len(object_ids), len(detected_boxes)), dtype=np.float32)
        for i, prev_box in enumerate(prev_boxes):
            for j, det_box in enumerate(detected_boxes):
                iou_matrix[i, j] = calculate_iou(prev_box, det_box)

        # Greedy match based on max IoU
        matched_obj_indices = set()
        matched_det_indices = set()

        # Sort all pairs by IoU descending
        pairs = []
        for i in range(len(object_ids)):
            for j in range(len(detected_boxes)):
                if iou_matrix[i, j] >= self.iou_threshold:
                    pairs.append((iou_matrix[i, j], i, j))
        pairs.sort(reverse=True, key=lambda x: x[0])

        tracked = []
        for score, i, j in pairs:
            if i in matched_obj_indices or j in matched_det_indices:
                continue
            obj_id = object_ids[i]
            self.objects[obj_id] = detected_boxes[j]
            self.disappeared[obj_id] = 0
            matched_obj_indices.add(i)
            matched_det_indices.add(j)
            tracked.append({'id': obj_id, 'box': detected_boxes[j]})

        # Unmatched existing objects
        for i, obj_id in enumerate(object_ids):
            if i not in matched_obj_indices:
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

        # Unmatched detections -> register new
        for j, box in enumerate(detected_boxes):
            if j not in matched_det_indices:
                obj_id = self.register(box)
                tracked.append({'id': obj_id, 'box': box})

        return tracked

    def reset(self):
        self.next_id = 1
        self.objects.clear()
        self.disappeared.clear()
