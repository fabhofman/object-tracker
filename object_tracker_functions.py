import numpy as np
from collections import Counter
from itertools import chain



# ---helper functions
def bbox_to_meas(bb):
    # [x1,y1,x2,y2] -> [cx,cy,w,h]
    x1, y1, x2, y2 = bb
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    cx = x1 + w/2.0
    cy = y1 + h/2.0
    return np.array([cx, cy, w, h], dtype=np.float32)

def meas_to_bbox(z):
    # [cx,cy,w,h] -> [x1,y1,x2,y2]
    cx, cy, w, h = z
    x1 = cx - w/2.0
    y1 = cy - h/2.0
    x2 = cx + w/2.0
    y2 = cy + h/2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def inflate_bbox(bbox, n_missed, inflate_per_frame=0.2, max_scale=3.0):
    """
    bbox: [x1, y1, x2, y2]
    n_missed: how many frames since last update (time_since_update)
    inflate_per_frame: fractional linear growth per missed frame
    max_scale: cap on total scale factor

    Returns: inflated [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return bbox

    # scale factor grows with missed frames, but capped
    scale = 1.0 + inflate_per_frame * n_missed
    scale = min(scale, max_scale)

    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    new_w = w * scale
    new_h = h * scale

    new_x1 = cx - new_w / 2.0
    new_y1 = cy - new_h / 2.0
    new_x2 = cx + new_w / 2.0
    new_y2 = cy + new_h / 2.0

    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)

# ---- IoU + matching 

def iou_matrix(tracks_bboxes, dets_bboxes):
    '''tracks_bboxes: list of np.ndarray of shape (N,4) with [x1,y1,x2,y2] detections per frame for a single track
    dets_bboxes: list of np.ndarray of shape (N,4) with [x1,y1,x2,y2] detections per frame
    Returns: np.ndarray of shape (N,N) with IoU between each track and detection
    '''

    if len(tracks_bboxes) == 0 or len(dets_bboxes) == 0:
        return np.zeros((len(tracks_bboxes), len(dets_bboxes)), dtype=np.float32)
    T = np.array(tracks_bboxes, dtype=np.float32)
    D = np.array(dets_bboxes, dtype=np.float32)

    # areas
    Ta = (T[:,2]-T[:,0]) * (T[:,3]-T[:,1])
    Da = (D[:,2]-D[:,0]) * (D[:,3]-D[:,1])

    ious = np.zeros((T.shape[0], D.shape[0]), dtype=np.float32)
    for i in range(T.shape[0]):
        xx1 = np.maximum(T[i,0], D[:,0])
        yy1 = np.maximum(T[i,1], D[:,1])
        xx2 = np.minimum(T[i,2], D[:,2])
        yy2 = np.minimum(T[i,3], D[:,3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = Ta[i] + Da - inter
        ious[i] = np.where(union > 0, inter/union, 0.0)
    return ious

def greedy_match(ious, iou_thresh):
    """
    ious: np.ndarray of shape (N,M) with IoU between each track and detection
    iou_thresh: IoU above which we treat detections as duplicates
    Returns: list of (track_idx, det_idx), list_unmatched_tracks, list_unmatched_dets
    """
    matches = []
    if ious.size == 0:
        return matches, list(range(ious.shape[0])), list(range(ious.shape[1]))

    T, D = ious.shape
    used_t = set()
    used_d = set()

    # flatten and sort by IoU desc
    flat = [ (int(i//D), int(i % D), ious[int(i//D), int(i % D)]) for i in range(T*D) ]
    flat.sort(key=lambda x: x[2], reverse=True)

    for ti, di, v in flat:
        if v < iou_thresh: break
        if ti in used_t or di in used_d:
            continue
        matches.append((ti, di))
        used_t.add(ti); used_d.add(di)

    unmatched_t = [i for i in range(T) if i not in used_t]
    unmatched_d = [j for j in range(D) if j not in used_d]
    return matches, unmatched_t, unmatched_d

def remove_duplicate_dets_iou(boxes, scores, iou_thresh=0.7):
    """
    boxes:  np.ndarray of shape (N, 4)  [x1, y1, x2, y2]
    scores: np.ndarray of shape (N,)
    iou_thresh: IoU above which we treat detections as duplicates

    Returns:
        filtered_boxes , filtered_scores. Same format as input boxes and scores but with dupes removed
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    if boxes.shape[0] == 0:
        return boxes, scores

    # Sort detections by score (high to low)
    order = np.argsort(scores)[::-1]

    keep_indices = []

    while len(order) > 0:
        i = order[0]          # index of highest-score box remaining
        keep_indices.append(i)

        if len(order) == 1:
            break

        # Compare this box with all remaining boxes
        rest = order[1:]

        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h

        area_i    = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_rest - inter
        ious = np.where(union > 0, inter / union, 0.0)

        # Keep only those boxes whose IoU with box i is <= threshold
        mask = ious <= iou_thresh
        order = rest[mask]    # drop high-IoU (duplicate) boxes, keep the others

    filtered_boxes = boxes[keep_indices]
    filtered_scores = scores[keep_indices]
    return filtered_boxes, filtered_scores


# -------- Kalman filter for a box (8D, constant velocity)
class KalmanBox:
    def __init__(self, z0, dt=1/5.0, process_var=1.0, meas_var_pos=5.0, meas_var_size=10.0, vel_var_scale=10.0,
                 init_pos_var=100.0, init_vel_scale=100.0):
        '''
        z0: [cx,cy,w,h]
        dt: time step in seconds
        process_var: process noise covariance
        meas_var_pos: measurement noise covariance for position
        meas_var_size: measurement noise covariance for size
        vel_var_scale: velocity noise covariance scale
        init_pos_var: initial position variance
        init_vel_scale: initial velocity variance
        '''
        self.dt = float(dt)
        self.x = np.zeros((8,1), dtype=np.float32)  # [cx,cy,w,h,vx,vy,vw,vh]
        self.x[:4,0] = z0

        dt = self.dt
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i+4] = dt

        self.H = np.zeros((4,8), dtype=np.float32)
        self.H[0,0] = self.H[1,1] = self.H[2,2] = self.H[3,3] = 1.0

        q_pos = process_var
        q_vel = process_var * vel_var_scale
        self.Q = np.diag([q_pos,q_pos,q_pos,q_pos,q_vel,q_vel,q_vel,q_vel]).astype(np.float32)
        self.R = np.diag([meas_var_pos, meas_var_pos, meas_var_size, meas_var_size]).astype(np.float32)

        self.P = np.eye(8, dtype=np.float32) * init_pos_var
        self.P[4:,4:] *= init_vel_scale
        self.I = np.eye(8, dtype=np.float32)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z=None):
        if z is None:
            return self.x.copy()
        z = z.reshape(4,1).astype(np.float32)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        return self.x.copy()

    def predicted_bbox(self):
        return meas_to_bbox(self.x[:4,0])

# -------- Track wrapper 
class Track:
    def __init__(self, track_id, z0, dt, process_var, meas_var_pos, meas_var_size, vel_var_scale,
                 init_pos_var, init_vel_scale):
        '''
        track_id: unique identifier for the track
        the rest are the same as above in KalmanBox class
        '''

        self.id = track_id
        self.kf = KalmanBox(z0, dt=dt, process_var=process_var,
                            meas_var_pos=meas_var_pos, meas_var_size=meas_var_size, vel_var_scale=vel_var_scale,
                            init_pos_var=init_pos_var, init_vel_scale=init_vel_scale)
        self.hits = 1               # times updated with a detection
        self.age = 1                # frames since creation
        self.time_since_update = 0  # frames since last detection
        self.bbox = meas_to_bbox(z0)
        self.confirmed = False      # becomes True after min_hits

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self.bbox = self.kf.predicted_bbox()
        return self.bbox

    def update(self, det_bbox):
        z = bbox_to_meas(det_bbox)
        self.kf.update(z)
        self.bbox = meas_to_bbox(self.kf.x[:4,0])
        self.hits += 1
        self.time_since_update = 0




#--------Track stitching code
##this stuff adds marginal value, can remove if lower computational cost is desired

def invert(frames, IDs):
    '''
    frames: list of np.ndarray of shape (N,4) with [x1,y1,x2,y2] detections per frame
    IDs: list of dicts {det_idx: track_id} per frame (output of kalman_track_frames)
    Returns: dict of tracks {track_id: list[(frame_idx, bbox)]}
    '''
    tracks = {}
    for f_idx, (boxes, frame_IDs) in enumerate(zip(frames, IDs)):
        for det_idx, track_ID in frame_IDs.items():
            bb = boxes[det_idx]
            tracks.setdefault(track_ID, []).append((f_idx, bb))
    return tracks

def track_stats(track):
    '''
    track = list[(frame_idx, bbox)]
    Returns: duration, disp, mean_area
    '''
    # track = list[(frame_idx, bbox)]
    frames = np.array([t[0] for t in track])
    bbs    = np.array([t[1] for t in track], dtype=np.float32)

    # duration in frames
    duration = frames.max() - frames.min() + 1

    # centre positions
    cx = (bbs[:,0] + bbs[:,2]) / 2
    cy = (bbs[:,1] + bbs[:,3]) / 2
    # total displacement (pixel distance between first and last)
    disp = np.hypot(cx[-1] - cx[0], cy[-1] - cy[0])

    # average area
    areas = (bbs[:,2] - bbs[:,0]) * (bbs[:,3] - bbs[:,1])
    mean_area = areas.mean()

    return duration, disp, mean_area


def filter_tracks(tracks, frames, IDs, min_duration_frames=5, min_displacement_percent=0.01, dimensions=(720, 460)):
    """
    tracks: dict of tracks {track_id: list[(frame_idx, bbox)]}
    frames: list of np.ndarray of shape (N,4) with [x1,y1,x2,y2] detections per frame
    IDs: list of dicts {det_idx: track_id} per frame (output of kalman_track_frames)
    min_duration_frames: minimum duration of a track in frames
    min_displacement_percent: minimum displacement of a track in percentage of the image diagonal
    dimensions: dimensions of the image


    Drop tracks that are too short or basically static.
    Also removes corresponding bounding boxes from frames.

    Returns:
        good_tracks: dict of filtered tracks
        filtered_frames: list of np.ndarray with filtered bounding boxes
        filtered_IDs: list of dicts with remapped detection indices (0, 1, 2, ...)
    """
    w, h = dimensions
    diag = np.hypot(w, h)

    good_tracks = {}
    for track_ID, traj in tracks.items():
        duration, disp, _ = track_stats(traj)
        disp_frac = disp / diag if diag > 0 else 0.0

        if duration >= min_duration_frames and disp_frac >= min_displacement_percent:
            good_tracks[track_ID] = traj

    # Build set of good track IDs for fast lookup
    good_track_ids = set(good_tracks.keys())

    # Filter frames and remap detection indices to be continuous
    filtered_frames = []
    filtered_IDs = []

    for frame_boxes, frame_IDs in zip(frames, IDs):
        # Find which detection indices to keep
        keep_indices = [det_idx for det_idx, track_id in frame_IDs.items()
                        if track_id in good_track_ids]
        keep_indices.sort()  # maintain order

        # Create new frame with only kept boxes
        if len(keep_indices) > 0:
            new_boxes = frame_boxes[keep_indices]
        else:
            new_boxes = np.zeros((0, 4), dtype=np.float32)

        # Remap detection indices to be continuous (0, 1, 2, ...)
        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}

        # Build new ID mapping with remapped indices
        new_frame_IDs = {old_to_new_idx[det_idx]: track_id
                         for det_idx, track_id in frame_IDs.items()
                         if det_idx in old_to_new_idx}

        filtered_frames.append(new_boxes)
        filtered_IDs.append(new_frame_IDs)

    return good_tracks, filtered_frames, filtered_IDs




def bbox_center_area(bb):
    x1,y1,x2,y2 = bb
    cx = (x1+x2)/2
    cy = (y1+y2)/2
    area = (x2-x1)*(y2-y1)
    return cx, cy, area


def merge_score(traj_a, traj_b,
                max_frame_gap=10,
                max_center_dist_percent=0.1,
                max_area_ratio=2.0,
                dimensions=(720, 460)):
    '''
    traj_a: list[(frame_idx, bbox)]
    traj_b: list[(frame_idx, bbox)]
    max_frame_gap: maximum gap in frames between the two tracks
    max_center_dist_percent: maximum distance in percentage of the image diagonal between the two tracks
    max_area_ratio: maximum ratio of the areas of the two tracks
    dimensions: dimensions of the image
    

    If traj_a and traj_b could be the same person (a before b), return a
    distance score (smaller is better). Otherwise return None.
    '''
    traj_a = sorted(traj_a, key=lambda x: x[0])
    traj_b = sorted(traj_b, key=lambda x: x[0])

    f_a, bb_a = traj_a[-1]
    f_b, bb_b = traj_b[0]

    # time ordering & max gap
    if f_b <= f_a or (f_b - f_a) > max_frame_gap:
        return None

    cx_a, cy_a, area_a = bbox_center_area(bb_a)
    cx_b, cy_b, area_b = bbox_center_area(bb_b)

    # distance in image-diagonal units
    w, h = dimensions
    diag = np.hypot(w, h)
    dist = np.hypot(cx_b - cx_a, cy_b - cy_a)
    dist_frac = dist / diag if diag > 0 else 0.0
    if dist_frac > max_center_dist_percent:
        return None

    # similar scale
    ratio = max(area_a, area_b) / max(1.0, min(area_a, area_b))
    if ratio > max_area_ratio:
        return None

    # score: you can combine dist + small area penalty if you like
    # for now just use centre distance fraction
    return dist_frac


def merge_tracks(tracks,
                 max_frame_gap=10,
                 max_center_dist_percent=0.1,
                 max_area_ratio=2.0,
                 dimensions=(720, 460)):
    """
    tracks: dict track_id -> list[(frame_idx, bbox)]
    returns: list of merged track groups, each group is a list of original IDs
    """
    tids = sorted(tracks.keys())
    trajs = {tid: sorted(traj, key=lambda x: x[0]) for tid, traj in tracks.items()}

    # union-find / disjoint set
    parent = {tid: tid for tid in tids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # --- build candidate merges with scores ---
    candidates = []  # (score, earlier_tid, later_tid)

    for i, ta in enumerate(tids):
        for tb in tids[i+1:]:
            score = merge_score(trajs[ta], trajs[tb],
                                max_frame_gap=max_frame_gap,
                                max_center_dist_percent=max_center_dist_percent,
                                max_area_ratio=max_area_ratio,
                                dimensions=dimensions)
            if score is not None:
                # make sure ta is the earlier track in time
                if trajs[ta][-1][0] <= trajs[tb][0][0]:
                    earlier, later = ta, tb
                else:
                    earlier, later = tb, ta
                candidates.append((score, earlier, later))

    # sort by "closeness" (best first)
    candidates.sort(key=lambda x: x[0])

    used_earlier = set()
    used_later = set()

    # --- greedy one-to-one matching ---
    for score, earlier, later in candidates:
        if earlier in used_earlier or later in used_later:
            continue  # already matched in another pair
        union(earlier, later)
        used_earlier.add(earlier)
        used_later.add(later)

    # --- collect groups ---
    groups = {}
    for tid in tids:
        r = find(tid)
        groups.setdefault(r, []).append(tid)

    return list(groups.values())



def create_new_ID_tracks(frames, IDs,
                         min_duration_frames=3,
                         min_displacement_percent=0.01,
                         max_frame_gap=10,
                         max_center_dist_percent=0.1,
                         max_area_ratio=2.0,
                         dimensions=(720, 460)):
    """
    frames: list of np.ndarray (N_f, 4)  [x1,y1,x2,y2] per frame
    IDs:    list of dicts {det_idx: track_id} per frame (output of kalman_track_frames)

    Returns:
        frames_new: filtered frames with only valid tracks' bounding boxes
        IDs_new:    list of dicts {det_idx: merged_person_id} per frame
        groups:     list of lists, each group is original track IDs merged into one person
    """

    # 1) Build track trajectories from (frames, IDs)
    tracks = invert(frames, IDs)

    # 2) Filter out tiny / static tracks (likely junk) AND remove their boxes
    tracks_filtered, frames_filtered, IDs_filtered = filter_tracks(
        tracks, frames, IDs,
        min_duration_frames=min_duration_frames,
        min_displacement_percent=min_displacement_percent,
        dimensions=dimensions
    )

    # 3) Merge track IDs that look like fragments of the same person
    groups = merge_tracks(tracks_filtered,
                          max_frame_gap=max_frame_gap,
                          max_center_dist_percent=max_center_dist_percent,
                          max_area_ratio=max_area_ratio,
                          dimensions=dimensions)

    # 4) Build mapping from old track IDs to new merged IDs (person IDs)
    old_to_new = {}
    for new_id, group in enumerate(groups):
        for tid in group:
            old_to_new[tid] = new_id

    # 5) Build new per-frame ID dicts with merged person IDs
    IDs_new = []
    for frame_IDs in IDs_filtered:
        new_map = {}
        for det_idx, old_tid in frame_IDs.items():
            if old_tid in old_to_new:
                new_map[det_idx] = old_to_new[old_tid]
        IDs_new.append(new_map)

    # frames_filtered now has matching number of boxes with IDs_new
    return frames_filtered, IDs_new, groups




