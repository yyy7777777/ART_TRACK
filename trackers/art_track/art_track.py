import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import logging

from .aimm_kf_wrapper import AIMMKalmanFilter
from .matching import *
from .basetrack import BaseTrack, TrackState
from typing import List


class STrackAIMM(BaseTrack):
    """Enhanced STrack with AIMM-UKF support."""

    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.observations = {}
        self.motion_history = []
        self.maneuver_count = 0
        self.last_aimm_summary = {}

    def predict(self):
        """Run prediction with AIMM support."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0

        prev_pos = self.mean[:2].copy() if self.mean is not None else None

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

        if prev_pos is not None:
            current_pos = self.mean[:2]
            displacement = np.linalg.norm(current_pos - prev_pos)
            self.motion_history.append(displacement)

            if len(self.motion_history) > 10:
                self.motion_history.pop(0)

        if hasattr(self.kalman_filter, 'get_aimm_state_summary'):
            self.last_aimm_summary = self.kalman_filter.get_aimm_state_summary()

            if self.track_id == 1 and self.tracklet_len % 10 == 0:
                logging.debug(
                    f"[DEBUG] Track {self.track_id} Frame {self.frame_id}: "
                    f"AIMM Summary = {self.last_aimm_summary}"
                )

            if self.last_aimm_summary.get('is_maneuvering', False):
                self.maneuver_count += 1

    @staticmethod
    def multi_predict(stracks):
        """Run prediction for a list of tracks."""
        if len(stracks) > 0:
            for st in stracks:
                if st.state != TrackState.Tracked and st.mean is not None:
                    st.mean[7] = 0
                st.predict()

    def activate(self, kalman_filter, frame_id):
        """Start a new track."""
        self.kalman_filter = kalman_filter
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.observations[frame_id] = self._tlwh

        self.tracklet_len = 1
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.start_frame = frame_id

        self.track_id = self.next_id()
        self.is_activated = True
        logging.debug(f"AIMM track {self.track_id} activated at frame {frame_id}")

    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-activate a lost track."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        self.observations[frame_id] = new_track.tlwh

        logging.debug(f"AIMM track {self.track_id} re-activated at frame {frame_id}")

    def re_activate_with_oru(self, new_track, frame_id):
        """
        Re-activate a lost track with Observation-Centric Re-Update.
        """
        last_seen_frame = self.end_frame
        lost_duration = frame_id - last_seen_frame

        if lost_duration <= 0:
            logging.error(
                f"Attempted to re-activate track {self.track_id} with ORU in the same or a future frame. "
                f"Current frame: {frame_id}, last seen: {last_seen_frame}. "
                f"Falling back to simple re-activation."
            )
            self.re_activate(new_track, frame_id)
            return

        last_seen_obs_tlwh = self.observations.get(last_seen_frame, self.tlwh)
        new_obs_tlwh = new_track.tlwh

        current_mean, current_cov = self.mean.copy(), self.covariance.copy()

        if lost_duration > 1:
            for i in range(1, lost_duration):
                alpha = i / lost_duration
                virtual_obs_tlwh = (1 - alpha) * last_seen_obs_tlwh + alpha * new_obs_tlwh

                current_mean, current_cov = self.kalman_filter.predict(current_mean, current_cov)
                current_mean, current_cov = self.kalman_filter.update(
                    current_mean, current_cov, self.tlwh_to_xyah(virtual_obs_tlwh)
                )

        self.mean, self.covariance = self.kalman_filter.predict(current_mean, current_cov)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_obs_tlwh)
        )

        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.frame_id = frame_id
        self.tracklet_len = 0
        self.observations[frame_id] = new_obs_tlwh

        if hasattr(self.kalman_filter, 'get_aimm_state_summary'):
            self.last_aimm_summary = self.kalman_filter.get_aimm_state_summary()

        num_lost_frames = lost_duration - 1
        # logging.warning(
        #     f"AIMM track {self.track_id} re-activated with ORU after being lost for {num_lost_frames} frames."
        # )

    def update(self, new_track, frame_id):
        """Update a track."""
        if not self.is_activated:
            logging.error(
                f"Updating unactivated track {self.track_id} at frame {frame_id}. "
                f"This indicates a bug in the tracking logic."
            )

        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        if len(self.observations) > 0:
            last_frame = max(self.observations.keys())
            last_tlwh = self.observations[last_frame]

            displacement = np.linalg.norm(new_tlwh[:2] - last_tlwh[:2])
            self.motion_history.append(displacement)
            if len(self.motion_history) > 10:
                self.motion_history.pop(0)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )

        self.observations[frame_id] = new_tlwh
        self.state = TrackState.Tracked
        self.score = new_track.score

        if hasattr(self.kalman_filter, 'get_aimm_state_summary'):
            self.last_aimm_summary = self.kalman_filter.get_aimm_state_summary()

    def get_motion_analysis(self):
        """Return motion analysis statistics."""
        return {
            'avg_displacement': np.mean(self.motion_history) if self.motion_history else 0,
            'displacement_std': np.std(self.motion_history) if len(self.motion_history) > 1 else 0,
            'maneuver_count': self.maneuver_count,
            'maneuver_frequency': self.maneuver_count / max(1, self.tracklet_len),
            'aimm_summary': self.last_aimm_summary.copy(),
        }

    @property
    def tlwh(self):
        """Get current position in tlwh format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to tlbr format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xyah(self):
        """Convert bounding box to xyah format."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert tlwh to xyah format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def get_center(self):
        """Return the bounding box center."""
        if self.mean is not None:
            return self.mean[:2].copy()
        x, y, w, h = self._tlwh
        return np.array([x + w / 2, y + h / 2])

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @property
    def velocity(self):
        """Return the predicted velocity vector."""
        if self.mean is None:
            return np.array([0, 0])

        velocity = self.mean[4:6].copy()

        if self.last_aimm_summary:
            dominant_model = self.last_aimm_summary.get('dominant_model', 'CV')
            if dominant_model == 'CT':
                velocity *= 1.1
            elif self.last_aimm_summary.get('is_maneuvering', False):
                velocity *= 0.9

        return velocity

    def __repr__(self):
        return 'AIMM_OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

    @property
    def positional_uncertainty(self) -> float:
        """Return a scalar position uncertainty based on covariance trace."""
        if self.covariance is None:
            return 10.0

        pos_cov = self.covariance[0:2, 0:2]
        uncertainty_scalar = np.trace(pos_cov)
        return max(0.1, uncertainty_scalar)

    @property
    def is_maneuvering(self) -> bool:
        """Return whether the track is currently maneuvering."""
        if not self.last_aimm_summary:
            return False
        return self.last_aimm_summary.get('is_maneuvering', False)

    def get_dominant_motion_mode(self) -> str:
        """Return the dominant motion mode."""
        if not self.last_aimm_summary:
            return 'UNKNOWN'

        dominant_model = self.last_aimm_summary.get('dominant_model', None)
        if dominant_model:
            return dominant_model

        model_probs = self.last_aimm_summary.get('model_probabilities', [])

        if isinstance(model_probs, dict):
            if not model_probs:
                return 'UNKNOWN'
            return max(model_probs.items(), key=lambda x: x[1])[0]
        elif isinstance(model_probs, (list, np.ndarray)):
            if len(model_probs) < 3:
                return 'UNKNOWN'
            model_names = ['CV', 'CA', 'CT']
            return model_names[np.argmax(model_probs)]
        else:
            logging.warning(f"Unexpected model_probabilities format: {type(model_probs)}")
            return 'UNKNOWN'


class DARTrackerAIMM(object):
    """Enhanced tracker with AIMM-UKF."""

    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh
        self.buffer_size = int(frame_rate / 30.0 * 5 * args.track_buffer)
        self.max_time_lost = self.buffer_size

        self.kalman_filter = AIMMKalmanFilter(use_aimm=True)

        self.total_maneuvers_detected = 0
        self.model_usage_stats = {'CV': 0, 'CA': 0, 'CT': 0}

        self.img_diag = None
        self.max_detected = 0
        BaseTrack.reset_id()

        self.oc_sort_thresh = 1
        self.w_iou = 0.4
        self.w_motion = 0.6

        self.uncertainty_k = 0.1
        self.uncertainty_midpoint = 25.0
        self.base_weight = 0.5
        self.dynamic_range = 0.7

        logging.info("DARTrackerAIMM initialized with AIMM-Driven Cascade")

    def reset(self):
        """Reset tracker state."""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0

        self.total_maneuvers_detected = 0
        self.model_usage_stats = {'CV': 0, 'CA': 0, 'CT': 0}

        self.max_detected = 0
        self.img_diag = None

        BaseTrack.reset_id()
        logging.info("OCByteTracker-AIMM reset to initial state")

    def _split_tracks_by_aimm_state(self, tracks):
        """Split tracks by maneuvering state."""
        stable_tracks = []
        maneuvering_tracks = []

        cv_probs = []
        maneuver_count = 0

        for track in tracks:
            model_probs = track.last_aimm_summary.get('model_probabilities', [])
            if isinstance(model_probs, (list, np.ndarray)) and len(model_probs) > 0:
                cv_probs.append(model_probs[0])

            if track.is_maneuvering:
                maneuvering_tracks.append(track)
                maneuver_count += 1
            else:
                stable_tracks.append(track)

        if len(tracks) > 0:
            maneuver_ratio = maneuver_count / len(tracks)
            logging.info(
                f"Track split: stable={len(stable_tracks)}, maneuvering={len(maneuvering_tracks)} "
                f"(ratio={maneuver_ratio:.1%})"
            )

            if cv_probs:
                logging.debug(
                    f"CV probability: mean={np.mean(cv_probs):.3f}, median={np.median(cv_probs):.3f}"
                )

        return stable_tracks, maneuvering_tracks

    def update(self, output_results, img_info, img_size):
        """
        Main tracking update.

        Step 1: stable tracks + all detections
        Step 2: maneuvering tracks + lost tracks + remaining detections
        Step 3: unmatched tracks + remaining high-score detections
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        newly_created_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]

        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        self.img_diag = float(np.hypot(img_w, img_h))

        valid_inds = scores > 0.35
        dets = bboxes[valid_inds]
        scores_keep = scores[valid_inds]
        high_score_mask = scores_keep >= self.args.track_thresh

        self.max_detected = max(self.max_detected, int(len(dets)))

        if len(dets) > 0:
            detections = [
                STrackAIMM(STrackAIMM.tlbr_to_tlwh(tlbr), score)
                for tlbr, score in zip(dets, scores_keep)
            ]
        else:
            detections = []

        tracked_stracks = self.tracked_stracks.copy()
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrackAIMM.multi_predict(strack_pool)

        stable_tracks, maneuvering_tracks = self._split_tracks_by_aimm_state(tracked_stracks)

        logging.info(
            f"Frame {self.frame_id}: "
            f"stable={len(stable_tracks)}, maneuvering={len(maneuvering_tracks)}, "
            f"lost={len(self.lost_stracks)}, detections={len(detections)}"
        )

        logging.debug("Step 1: stable track matching")
        if len(stable_tracks) > 0 and len(detections) > 0:
            dists_stable = iou_distance(stable_tracks, detections)
            strict_thresh = self.args.match_thresh
            matches_step1, u_track_step1, u_detection_step1 = linear_assignment(
                dists_stable, thresh=strict_thresh
            )

            for itracked, idet in matches_step1:
                track = stable_tracks[itracked]
                det = detections[idet]
                track.update(det, self.frame_id)
                activated_starcks.append(track)

            unmatched_stable = [stable_tracks[i] for i in u_track_step1]
            remaining_detections_step1 = [detections[i] for i in u_detection_step1]

            logging.info(f"Step 1: matched {len(matches_step1)} pairs")
        else:
            unmatched_stable = stable_tracks
            remaining_detections_step1 = detections

        all_candidates_step2 = maneuvering_tracks + self.lost_stracks

        logging.debug(
            f"Step 2: maneuvering + lost matching - "
            f"tracks={len(all_candidates_step2)} "
            f"(maneuvering={len(maneuvering_tracks)}, lost={len(self.lost_stracks)}), "
            f"detections={len(remaining_detections_step1)}"
        )

        if len(all_candidates_step2) > 0 and len(remaining_detections_step1) > 0:
            n_maneuvering = len(maneuvering_tracks)
            n_lost = len(self.lost_stracks)

            dists_step2 = self._compute_aimm_distance(all_candidates_step2, remaining_detections_step1)

            if n_maneuvering > 0:
                adaptive_thresh = self._compute_adaptive_threshold(maneuvering_tracks)
            else:
                adaptive_thresh = self.args.match_thresh * 1.2

            if n_lost > 0:
                avg_lost_duration = np.mean([self.frame_id - t.end_frame for t in self.lost_stracks])
                if avg_lost_duration > 10:
                    adaptive_thresh = max(adaptive_thresh, self.oc_sort_thresh * 1.5)
                elif avg_lost_duration > 5:
                    adaptive_thresh = max(adaptive_thresh, self.oc_sort_thresh * 1.2)

            matches_step2, u_track_step2, u_detection_step2 = linear_assignment(
                dists_step2, thresh=adaptive_thresh
            )

            for itracked, idet in matches_step2:
                track = all_candidates_step2[itracked]
                det = remaining_detections_step1[idet]

                if itracked < n_maneuvering:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate_with_oru(det, self.frame_id)
                    refind_stracks.append(track)

            unmatched_step2_tracks = [all_candidates_step2[i] for i in u_track_step2]
            remaining_detections_step2 = [remaining_detections_step1[i] for i in u_detection_step2]

            unmatched_maneuvering = [t for t in unmatched_step2_tracks if t in maneuvering_tracks]
            unmatched_lost_step2 = [t for t in unmatched_step2_tracks if t in self.lost_stracks]

            logging.info(
                f"Step 2: matched {len(matches_step2)} pairs "
                f"(maneuvering={sum(1 for i, _ in matches_step2 if i < n_maneuvering)}, "
                f"lost={sum(1 for i, _ in matches_step2 if i >= n_maneuvering)})"
            )
        else:
            unmatched_maneuvering = maneuvering_tracks
            unmatched_lost_step2 = self.lost_stracks
            remaining_detections_step2 = remaining_detections_step1

        all_unmatched_tracked = unmatched_stable + unmatched_maneuvering
        all_candidates_step3 = all_unmatched_tracked + unmatched_lost_step2

        high_score_detections_step3 = [
            det for det, is_high in zip(
                remaining_detections_step2,
                [high_score_mask[detections.index(det)] for det in remaining_detections_step2]
            ) if is_high
        ]

        logging.debug(
            f"Step 3: ID recovery - tracks={len(all_candidates_step3)} "
            f"(tracked={len(all_unmatched_tracked)}, lost={len(unmatched_lost_step2)}), "
            f"high-score detections={len(high_score_detections_step3)}"
        )

        if len(all_candidates_step3) > 0 and len(high_score_detections_step3) > 0:
            n_tracked = len(all_unmatched_tracked)
            n_lost = len(unmatched_lost_step2)

            dists_step3, valid_mask = self._compute_lost_track_cost(
                all_candidates_step3, high_score_detections_step3
            )

            if n_lost > 0:
                avg_lost_duration = np.mean([self.frame_id - t.end_frame for t in unmatched_lost_step2])
                if avg_lost_duration > 10:
                    recovery_thresh = self.oc_sort_thresh * 2.0
                elif avg_lost_duration > 5:
                    recovery_thresh = self.oc_sort_thresh * 1.5
                else:
                    recovery_thresh = self.oc_sort_thresh * 1.2
            else:
                recovery_thresh = self.args.match_thresh * 1.5

            matches_step3, u_track_step3, u_detection_step3 = linear_assignment(
                dists_step3, thresh=recovery_thresh
            )

            for itracked, idet in matches_step3:
                track = all_candidates_step3[itracked]
                det = high_score_detections_step3[idet]

                if itracked < n_tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate_with_oru(det, self.frame_id)
                    refind_stracks.append(track)

            final_unmatched_tracks = [all_candidates_step3[i] for i in u_track_step3]
            remaining_high_score_detections = [high_score_detections_step3[i] for i in u_detection_step3]

            logging.info(
                f"Step 3: matched {len(matches_step3)} pairs "
                f"(tracked={sum(1 for i, _ in matches_step3 if i < n_tracked)}, "
                f"lost={sum(1 for i, _ in matches_step3 if i >= n_tracked)})"
            )
        else:
            final_unmatched_tracks = all_candidates_step3
            remaining_high_score_detections = high_score_detections_step3

        for track in final_unmatched_tracks:
            if track.state == TrackState.Tracked:
                track.mark_lost()
                lost_stracks.append(track)

        logging.info(
            f"Step 4: marked "
            f"{len([t for t in final_unmatched_tracks if t.state == TrackState.Tracked])} tracked tracks as lost"
        )

        new_track_detections = [
            det for det in remaining_high_score_detections if det.score >= self.det_thresh
        ]

        if len(new_track_detections) > 0:
            sorted_new = sorted(new_track_detections, key=lambda t: t.score, reverse=True)
            for track in sorted_new:
                track.activate(self.kalman_filter, self.frame_id)
                newly_created_stracks.append(track)

            logging.info(f"Step 5: created {len(sorted_new)} new tracks")

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        logging.info(f"Step 6: removed {len(removed_stracks)} expired tracks")

        self._update_aimm_statistics(activated_starcks + refind_stracks)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, newly_created_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logging.info(
            f"Frame {self.frame_id} summary: "
            f"Step1={len(matches_step1) if 'matches_step1' in locals() else 0}, "
            f"Step2={len(matches_step2) if 'matches_step2' in locals() else 0}, "
            f"Step3={len(matches_step3) if 'matches_step3' in locals() else 0} | "
            f"activated={len(activated_starcks)}, refound={len(refind_stracks)}, "
            f"new={len(newly_created_stracks)}, lost={len(lost_stracks)}, removed={len(removed_stracks)}"
        )

        return output_stracks

    def _compute_aimm_distance(self, tracks, detections):
        """Compute uncertainty-adaptive fusion distance."""
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((len(tracks), len(detections)), dtype=np.float32)

        iou_dist = iou_distance(tracks, detections)
        motion_dist = self._compute_motion_distance(tracks, detections)

        combined_dist = np.zeros((len(tracks), len(detections)), dtype=np.float32)

        for i, track in enumerate(tracks):
            uncertainty = track.positional_uncertainty
            norm_uncertainty = 1.0 / (
                1.0 + np.exp(-self.uncertainty_k * (uncertainty - self.uncertainty_midpoint))
            )

            alpha = self.base_weight + self.dynamic_range * norm_uncertainty
            alpha = np.clip(alpha, 0.1, 0.9)
            beta = 1.0 - alpha

            combined_dist[i, :] = alpha * iou_dist[i, :] + beta * motion_dist[i, :]

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(
                    f"Track {track.track_id}: uncertainty={uncertainty:.2f}, "
                    f"norm={norm_uncertainty:.3f}, alpha={alpha:.3f}, beta={beta:.3f}"
                )

        return combined_dist

    def _compute_motion_distance(self, tracks, detections):
        """Compute motion distance with AIMM state adjustment."""
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)

        motion_dist = np.zeros((len(tracks), len(detections)), dtype=np.float32)

        for i, track in enumerate(tracks):
            track_pos = track.tlwh[:2] + track.tlwh[2:] / 2
            track_vel = track.velocity

            for j, det in enumerate(detections):
                det_pos = det.tlwh[:2] + det.tlwh[2:] / 2
                predicted_pos = track_pos + track_vel
                pos_dist = np.linalg.norm(det_pos - predicted_pos)

                if hasattr(track, 'last_aimm_summary'):
                    summary = track.last_aimm_summary
                    if summary.get('is_maneuvering', False):
                        pos_dist *= 0.8

                    dominant_model = summary.get('dominant_model', 'CV')
                    if dominant_model == 'CT':
                        pos_dist *= 0.5

                img_scale = max(track.tlwh[2], track.tlwh[3], 1.0)
                motion_dist[i, j] = pos_dist / img_scale

        return motion_dist

    def _compute_lost_track_cost(self, lost_tracks, detections):
        """Compute matching cost for lost tracks."""
        if len(lost_tracks) == 0 or len(detections) == 0:
            return (
                np.empty((len(lost_tracks), len(detections)), dtype=np.float32),
                np.zeros((len(lost_tracks), len(detections)), dtype=bool)
            )

        last_obs_boxes = []
        for track in lost_tracks:
            if len(track.observations) > 0:
                last_frame = max(track.observations.keys())
                last_tlwh = track.observations[last_frame]
                last_tlbr = last_tlwh.copy()
                last_tlbr[2:] += last_tlbr[:2]
            else:
                last_tlbr = track.tlbr
            last_obs_boxes.append(last_tlbr)

        last_obs_boxes = np.array(last_obs_boxes)
        det_boxes = np.array([det.tlbr for det in detections])

        iou_cost = 1.0 - iou_batch(last_obs_boxes, det_boxes)
        motion_cost = np.zeros((len(lost_tracks), len(detections)), dtype=np.float32)

        delta_t = min(3, self.max_time_lost // 3)

        for i, track in enumerate(lost_tracks):
            track_velocity = None

            if len(track.observations) >= 2:
                sorted_frames = sorted(track.observations.keys())

                for dt in range(min(delta_t, len(sorted_frames) - 1), 0, -1):
                    if len(sorted_frames) > dt:
                        prev_frame = sorted_frames[-(dt + 1)]
                        curr_frame = sorted_frames[-1]

                        prev_box = track.observations[prev_frame]
                        curr_box = track.observations[curr_frame]

                        prev_center = prev_box[:2] + prev_box[2:] / 2
                        curr_center = curr_box[:2] + curr_box[2:] / 2

                        speed = curr_center - prev_center
                        norm = np.linalg.norm(speed)

                        if norm > 1e-6:
                            track_velocity = speed / norm
                            break

            if track_velocity is None:
                raw_velocity = track.velocity
                norm = np.linalg.norm(raw_velocity)
                if norm > 1e-6:
                    track_velocity = raw_velocity / norm

            for j, det in enumerate(detections):
                det_center = det.tlwh[:2] + det.tlwh[2:] / 2

                if len(track.observations) > 0:
                    last_frame = max(track.observations.keys())
                    track_center = track.observations[last_frame][:2] + track.observations[last_frame][2:] / 2
                else:
                    track_center = track.get_center()

                virtual_speed = det_center - track_center
                virtual_norm = np.linalg.norm(virtual_speed)

                if virtual_norm > 1e-6:
                    virtual_velocity = virtual_speed / virtual_norm

                    if track_velocity is not None:
                        cosine_sim = np.dot(track_velocity, virtual_velocity)
                        direction_cost = (1.0 - cosine_sim) / 2.0
                    else:
                        direction_cost = 0.5

                    time_lost = self.frame_id - track.end_frame
                    expected_dist = virtual_norm / max(1, time_lost)
                    img_scale = max(track.tlwh[2], track.tlwh[3], 1.0)

                    dist_penalty = min(expected_dist / img_scale, 1.0) * 0.3
                    motion_cost[i, j] = 0.7 * direction_cost + 0.3 * dist_penalty
                else:
                    motion_cost[i, j] = 0.1

        total_cost = self.w_iou * iou_cost + self.w_motion * motion_cost
        valid_mask = np.ones((len(lost_tracks), len(detections)), dtype=bool)

        for i in range(len(lost_tracks)):
            for j in range(len(detections)):
                if iou_cost[i, j] > 0.95 and motion_cost[i, j] > 0.9:
                    valid_mask[i, j] = False
                    total_cost[i, j] = 1e6

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            valid_count = np.sum(valid_mask)
            logging.debug(
                f"Lost track cost: "
                f"{len(lost_tracks)} tracks × {len(detections)} detections, "
                f"valid pairs={valid_count}/{valid_mask.size}, "
                f"IoU=[{iou_cost.min():.3f}, {iou_cost.max():.3f}], "
                f"Motion=[{motion_cost.min():.3f}, {motion_cost.max():.3f}], "
                f"Total=[{total_cost[valid_mask].min():.3f}, {total_cost[valid_mask].max():.3f}]"
            )

        return total_cost, valid_mask

    def _compute_adaptive_threshold(self, tracks):
        """Compute adaptive matching threshold."""
        base_thresh = self.args.match_thresh

        if not tracks:
            return base_thresh

        maneuver_ratio = sum(
            1 for t in tracks if t.last_aimm_summary.get('is_maneuvering', False)
        ) / len(tracks)

        if maneuver_ratio > 0.3:
            return base_thresh * 1.5

        return base_thresh * 1.1

    def _update_aimm_statistics(self, tracks):
        """Update AIMM-related statistics."""
        for track in tracks:
            if hasattr(track, 'last_aimm_summary'):
                summary = track.last_aimm_summary

                if summary.get('is_maneuvering', False):
                    self.total_maneuvers_detected += 1

                dominant_model = summary.get('dominant_model', 'CV')
                if dominant_model in self.model_usage_stats:
                    self.model_usage_stats[dominant_model] += 1

    def get_tracking_statistics(self):
        """Return tracking statistics."""
        stats = {
            'frame_id': self.frame_id,
            'tracked_count': len(self.tracked_stracks),
            'lost_count': len(self.lost_stracks),
            'total_maneuvers': self.total_maneuvers_detected,
            'model_usage': self.model_usage_stats.copy(),
            'active_tracks_analysis': []
        }

        for track in self.tracked_stracks:
            if hasattr(track, 'get_motion_analysis'):
                analysis = track.get_motion_analysis()
                stats['active_tracks_analysis'].append({
                    'track_id': track.track_id,
                    'analysis': analysis
                })

        return stats

    def _xyxy2tlwh(self, bboxes):
        """Convert bbox format from xyxy to tlwh."""
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        return bboxes


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb