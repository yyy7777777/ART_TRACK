# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg
from .aimm_ukf_filter import AIMUKFFilter
import logging

"""
AIMM-UKF Kalman filter wrapper.

Compatible with the original KalmanFilter interface while providing
AIMM-UKF-based enhancements.
"""

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class AIMMKalmanFilter(object):
    """
    Adaptive Interacting Multiple Model Unscented Kalman Filter wrapper.

    Compared with the original KalmanFilter, this wrapper adds:
    1. Nonlinear motion modeling
    2. Adaptive noise covariance
    3. Automatic model switching
    4. Better support for fast and maneuvering targets

    The interface remains compatible with the original implementation.
    """

    def __init__(self, dt=1.0, use_aimm=True):
        """
        Initialize the filter.

        Parameters
        ----------
        dt : float
            Time step.
        use_aimm : bool
            Whether to enable AIMM-UKF mode. If False, the filter falls back
            to the traditional Kalman filter.
        """
        self.dt = dt
        self.use_aimm = use_aimm

        if use_aimm:
            self.aimm_filter = AIMUKFFilter(dt)
            logging.info("AIMM-UKF mode enabled")
        else:
            self._init_traditional_mode()
            logging.info("Traditional Kalman Filter mode")

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        self._aimm_means = None
        self._aimm_covariances = None
        self._is_initialized = False

    def _init_traditional_mode(self):
        """Initialize the traditional Kalman filter mode."""
        ndim = 4
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = self.dt
        self._update_mat = np.eye(ndim, 2 * ndim)

    def initiate(self, measurement):
        """Create track from an unassociated measurement."""
        if self.use_aimm:
            self._aimm_means, self._aimm_covariances = self.aimm_filter.initiate(measurement)
            self._is_initialized = True

            mean = self._aimm_means[0]
            covariance = self._aimm_covariances[0]

            logging.debug(f"AIMM-UKF track initiated with measurement: {measurement}")
            return mean, covariance
        else:
            mean_pos = measurement
            mean_vel = np.zeros_like(mean_pos)
            mean = np.r_[mean_pos, mean_vel]

            std = [
                2 * self._std_weight_position * measurement[3],
                2 * self._std_weight_position * measurement[3],
                1e-2,
                2 * self._std_weight_position * measurement[3],
                10 * self._std_weight_velocity * measurement[3],
                10 * self._std_weight_velocity * measurement[3],
                1e-5,
                10 * self._std_weight_velocity * measurement[3],
            ]
            covariance = np.diag(np.square(std))
            return mean, covariance

    def predict(self, mean, covariance):
        """Run the prediction step."""
        if self.use_aimm and self._is_initialized:
            try:
                self._update_aimm_state_from_cv(mean, covariance)
                self._aimm_means, self._aimm_covariances = self.aimm_filter.predict_no_update(
                    self._aimm_means, self._aimm_covariances
                )

                predicted_mean = self._aimm_means[0]
                predicted_covariance = self._aimm_covariances[0]

                state_summary = self.aimm_filter.get_state_summary()
                logging.debug(
                    f"AIMM prediction - dominant model: {state_summary['dominant_model']}, "
                    f"occlusion time: {state_summary['occlusion_time']}"
                )

                return predicted_mean, predicted_covariance

            except Exception as e:
                logging.warning(f"AIMM prediction failed: {e}, falling back to traditional")
                return self._traditional_predict(mean, covariance)
        else:
            return self._traditional_predict(mean, covariance)

    def _traditional_predict(self, mean, covariance):
        """Run the traditional prediction step."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space."""
        if not self.use_aimm:
            self._update_mat = np.eye(4, 8)

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        projected_mean = mean[:4]
        projected_cov = covariance[:4, :4]

        return projected_mean, projected_cov + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run the correction step."""
        if self.use_aimm and self._is_initialized:
            try:
                self._update_aimm_state_from_cv(mean, covariance)

                updated_mean, updated_covariance = self.aimm_filter.update(
                    self._aimm_means, self._aimm_covariances, measurement
                )

                self._aimm_means[0] = updated_mean
                self._aimm_covariances[0] = updated_covariance

                state_summary = self.aimm_filter.get_state_summary()
                logging.debug(
                    f"AIMM update - model probs: {state_summary['model_probabilities']}, "
                    f"maneuvering: {state_summary['is_maneuvering']}"
                )

                return updated_mean, updated_covariance

            except Exception as e:
                logging.warning(f"AIMM update failed: {e}, falling back to traditional")
                return self._traditional_update(mean, covariance, measurement)
        else:
            return self._traditional_update(mean, covariance, measurement)

    def _traditional_update(self, mean, covariance, measurement):
        """Run the traditional update step."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )

        H = np.eye(4, 8)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, H.T).T, check_finite=False
        ).T

        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )

        return new_mean, new_covariance

    def _update_aimm_state_from_cv(self, cv_mean, cv_covariance):
        """Update the internal AIMM state from the CV-compatible state."""
        if not self._is_initialized:
            return

        self._aimm_means[0] = cv_mean
        self._aimm_covariances[0] = cv_covariance

        for i in range(1, len(self._aimm_means)):
            common_dim = min(8, len(self._aimm_means[i]))
            self._aimm_means[i][:common_dim] = cv_mean[:common_dim]

            cov_dim = min(8, self._aimm_covariances[i].shape[0])
            self._aimm_covariances[i][:cov_dim, :cov_dim] = cv_covariance[:cov_dim, :cov_dim]

    def multi_predict(self, mean, covariance):
        """Run vectorized prediction."""
        if self.use_aimm:
            logging.warning(
                "AIMM mode does not support vectorized prediction. "
                "Use individual predict() calls."
            )
            return self._traditional_multi_predict(mean, covariance)
        else:
            return self._traditional_multi_predict(mean, covariance)

    def _traditional_multi_predict(self, mean, covariance):
        """Run vectorized prediction in traditional mode."""
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements."""
        mean, covariance = self.project(mean, covariance)

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        if self.use_aimm and self._is_initialized:
            state_summary = self.aimm_filter.get_state_summary()
            uncertainty_factor = 1.0 + state_summary.get('adaptive_noise_factor', 1.0) * 0.1
            covariance *= uncertainty_factor

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            try:
                cholesky_factor = np.linalg.cholesky(covariance)
                z = scipy.linalg.solve_triangular(
                    cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
                )
                squared_maha = np.sum(z * z, axis=0)
                return squared_maha
            except np.linalg.LinAlgError:
                return np.sum(d * d, axis=1)
        else:
            raise ValueError('invalid distance metric')

    def get_aimm_state_summary(self):
        """Return the AIMM state summary."""
        if self.use_aimm and self._is_initialized:
            return self.aimm_filter.get_state_summary()
        else:
            return {"mode": "traditional", "aimm_enabled": False}

    def reset_aimm_state(self):
        """Reset the AIMM state."""
        if self.use_aimm:
            self._is_initialized = False
            self._aimm_means = None
            self._aimm_covariances = None
            logging.debug("AIMM state reset")


class KalmanFilter(AIMMKalmanFilter):
    """Backward-compatible KalmanFilter class."""

    def __init__(self, use_aimm=True):
        super().__init__(dt=1.0, use_aimm=use_aimm)


def create_aimm_kalman_filter(dt=1.0, enable_aimm=True):
    """Create an AIMM Kalman filter."""
    return AIMMKalmanFilter(dt=dt, use_aimm=enable_aimm)


def create_traditional_kalman_filter():
    """Create a traditional Kalman filter."""
    return AIMMKalmanFilter(dt=1.0, use_aimm=False)