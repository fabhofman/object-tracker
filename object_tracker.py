from object_tracker_functions import remove_duplicate_dets_iou, greedy_match, iou_matrix, inflate_bbox, bbox_to_meas, Track, create_new_ID_tracks
import numpy as np

# ---------- Main tracker ----------
def kalman_track_frames(frames, scores, dimensions,
                        sfps=10.0,
                        iou_cleaning_threshold=0.4,
                        iou_threshold=0.3,
                        min_hits=2,
                        max_age=3,
                        process_var=1.0,
                        meas_var_pos=5.0,
                        meas_var_size=10.0,
                        vel_var_scale=10.0,
                        init_pos_var=100.0,
                        init_vel_scale=100.0,
                        return_tentative=True,
                        inflate_factor = 0.2,
                        max_inflate_scale = 2,
                        iou_decay_per_miss=0.5,
                        min_iou=0.1,
                        min_duration_frames=5,
                        min_displacement_percent=0.001,
                        max_frame_gap=20,
                        max_center_dist_percent=0.2,
                        max_area_ratio=2.5):
    """
    frames: list of np.ndarray of shape (N,4) with [x1,y1,x2,y2] detections per frame
    returns: list of dicts -> for each frame: { detection_index : track_id }
    """

    clean_boxes_list = []
    clean_scores_list = []

    for each_frame_boxes, each_frame_scores in zip(frames, scores):
        fb, fs = remove_duplicate_dets_iou(each_frame_boxes, each_frame_scores, iou_thresh=iou_cleaning_threshold)
        clean_boxes_list.append(fb)
        clean_scores_list.append(fs)

    frames = clean_boxes_list
    scores = clean_scores_list



    dt = 1.0 / float(sfps)
    next_id = 0
    tracks = []  # list[Track]
    id_maps = [] # per-frame mapping: det_idx -> id

    for t_idx, dets in enumerate(frames):
        # normalize dets
        if dets is None:
            dets = np.zeros((0,4), dtype=np.float32)
        dets = np.array(dets, dtype=np.float32).reshape(-1,4)

        # 1) predict all tracks and build "matching boxes"
        predicted_boxes = []
        matching_boxes = []

        for tr in tracks:
            pb = tr.predict()  # updates tr.bbox internally
            predicted_boxes.append(pb)

            # Use inflated box for matching if the track has been virtual for a while
            if (tr.time_since_update - 1) > 0:
                mb = inflate_bbox(pb,
                                  n_missed=(tr.time_since_update-1),
                                  inflate_per_frame=inflate_factor,  # tune this
                                  max_scale=max_inflate_scale)  # tune this
            else:
                mb = pb

            matching_boxes.append(mb)

        matching_boxes = np.array(matching_boxes, dtype=np.float32)

        # 2) match inflated predicted boxes to detections

        ious = iou_matrix(matching_boxes, dets)

        if len(tracks) > 0 and len(dets) > 0:
            # --- per-track IoU thresholds ---
            base_iou = iou_threshold  # e.g. 0.3

            per_track_thresh = np.zeros(len(tracks), dtype=np.float32)
            for ti, tr in enumerate(tracks):
                eff = base_iou - iou_decay_per_miss * (tr.time_since_update-1)
                per_track_thresh[ti] = max(min_iou, eff)

            # apply per-track IoU thresholds: zero out too-low IoUs
            for ti in range(len(tracks)):
                mask = ious[ti, :] < per_track_thresh[ti]
                ious[ti, mask] = 0.0

        matches, unmatched_tracks_idx, unmatched_dets_idx = greedy_match(ious, 0.000001)   #is this incorrectly replacing pred_box with inflate_box?


        # 3) update matched tracks
        det_to_id = {}
        for ti, di in matches:
            tr = tracks[ti]
            tr.update(dets[di])
            if (tr.hits >= min_hits) or return_tentative:
                det_to_id[di] = tr.id

        # 4) increment age for unmatched tracks (already done in predict); remove old
        alive_tracks = []
        for idx, tr in enumerate(tracks):
            if idx in unmatched_tracks_idx:
                # no update this frame
                if tr.time_since_update <= max_age:
                    alive_tracks.append(tr)
                # else drop
            else:
                alive_tracks.append(tr)
        tracks = alive_tracks

        # 5) create new tracks for unmatched detections
        for di in unmatched_dets_idx:
            z0 = bbox_to_meas(dets[di])
            tr = Track(next_id, z0, dt, process_var, meas_var_pos, meas_var_size,
                       vel_var_scale, init_pos_var, init_vel_scale)
            # confirm if min_hits==1 or we allow tentative return
            if tr.hits >= min_hits:
                tr.confirmed = True
            tracks.append(tr)
            if return_tentative or tr.confirmed:
                det_to_id[di] = tr.id
            next_id += 1

        # 6) update confirmed flags
        for tr in tracks:
            if not tr.confirmed and tr.hits >= min_hits:
                tr.confirmed = True

        # 7) map for this frame (det index -> id)
        id_maps.append(det_to_id)

    frames_new, IDs_new, groups = create_new_ID_tracks(
        frames, id_maps,
        min_duration_frames=min_duration_frames,
        min_displacement_percent=min_displacement_percent,
        max_frame_gap=max_frame_gap,
        max_center_dist_percent=max_center_dist_percent,
        max_area_ratio=max_area_ratio,
        dimensions=dimensions
    )

    return frames_new, IDs_new, groups
