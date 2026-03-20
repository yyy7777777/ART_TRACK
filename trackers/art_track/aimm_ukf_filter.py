# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg
from scipy.linalg import sqrtm, cholesky
import logging
from typing import List, Tuple, Dict

"""
Adaptive Interacting Multiple Model Unscented Kalman Filter (AIMM-UKF)

Core innovations:
1. Interacting Multiple Model (IMM) framework for handling nonlinear motion mode switching
2. Unscented Kalman Filter (UKF) for accurate nonlinear state propagation
3. Adaptive noise covariance for occlusion and sudden maneuvers

Design goals:
- Fast motion: multiple models cover different motion regimes
- Frequent occlusion: adaptive noise covariance adjustment
- High maneuverability: UKF handles nonlinear turning dynamics
"""

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
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
    9: 16.919
}


class MotionModel:
    """Base class for motion models."""
    def __init__(self, name: str, state_dim: int, dt: float = 1.0):
        self.name = name
        self.state_dim = state_dim
        self.dt = dt

    def f(self, state: np.ndarray) -> np.ndarray:
        """State transition function f(x_k)."""
        raise NotImplementedError

    def h(self, state: np.ndarray) -> np.ndarray:
        """Observation function h(x_k)."""
        # Default observation model: observe position and size [x, y, a, h]
        return state[:4]

    def get_process_noise(self, state: np.ndarray) -> np.ndarray:
        """Return the process noise covariance matrix Q."""
        raise NotImplementedError


class ConstantVelocityModel(MotionModel):
    """Constant velocity motion model (CV)."""
    def __init__(self, dt: float = 1.0):
        super().__init__("CV", 8, dt)

    def f(self, state: np.ndarray) -> np.ndarray:
        """CV state transition: [x, y, a, h, vx, vy, va, vh] -> next state."""
        F = np.eye(8)
        F[0, 4] = self.dt  # x += vx * dt
        F[1, 5] = self.dt  # y += vy * dt
        F[2, 6] = self.dt  # a += va * dt
        F[3, 7] = self.dt  # h += vh * dt
        return F @ state

    def get_process_noise(self, state: np.ndarray) -> np.ndarray:
        """Process noise for the CV model."""
        # Adaptive noise based on current target size
        std_pos = 0.05 * state[3]   # Position noise proportional to height
        std_vel = 0.1 * state[3]    # Velocity noise
        std_size = 0.01 * state[3]  # Size noise

        q = np.array([
            std_pos, std_pos, std_size, std_pos,
            std_vel, std_vel, std_size * 0.1, std_vel
        ])
        return np.diag(q ** 2)


class ConstantAccelerationModel(MotionModel):
    """Constant acceleration motion model (CA)."""
    def __init__(self, dt: float = 1.0):
        super().__init__("CA", 10, dt)  # [x, y, a, h, vx, vy, va, vh, ax, ay]

    def f(self, state: np.ndarray) -> np.ndarray:
        """CA state transition."""
        F = np.eye(10)
        dt = self.dt
        dt2 = dt * dt / 2

        # Position += velocity * dt + 0.5 * acceleration * dt^2
        F[0, 4] = dt
        F[0, 8] = dt2
        F[1, 5] = dt
        F[1, 9] = dt2
        F[2, 6] = dt
        F[3, 7] = dt

        # Velocity += acceleration * dt
        F[4, 8] = dt
        F[5, 9] = dt

        return F @ state

    def get_process_noise(self, state: np.ndarray) -> np.ndarray:
        """Process noise for the CA model."""
        std_pos = 0.08 * state[3]
        std_vel = 0.15 * state[3]
        std_acc = 0.2 * state[3]
        std_size = 0.01 * state[3]

        q = np.array([
            std_pos, std_pos, std_size, std_pos,
            std_vel, std_vel, std_size * 0.1, std_vel,
            std_acc, std_acc
        ])
        return np.diag(q ** 2)


class CoordinatedTurnModel(MotionModel):
    """Coordinated turn model (CT) for nonlinear turning motion."""
    def __init__(self, dt: float = 1.0):
        super().__init__("CT", 9, dt)  # [x, y, a, h, vx, vy, va, vh, ω]

    def f(self, state: np.ndarray) -> np.ndarray:
        """CT state transition with trigonometric nonlinear dynamics."""
        next_state = state.copy()
        dt = self.dt

        x, y, a, h = state[0], state[1], state[2], state[3]
        vx, vy, va, vh = state[4], state[5], state[6], state[7]
        omega = state[8]  # Angular velocity

        if abs(omega) < 1e-5:
            # Degenerates to the CV model when angular velocity is near zero
            next_state[0] = x + vx * dt
            next_state[1] = y + vy * dt
        else:
            # Nonlinear turning motion
            sin_wt = np.sin(omega * dt)
            cos_wt = np.cos(omega * dt)

            next_state[0] = x + (vx * sin_wt + vy * (cos_wt - 1)) / omega
            next_state[1] = y + (vy * sin_wt - vx * (cos_wt - 1)) / omega
            next_state[4] = vx * cos_wt - vy * sin_wt
            next_state[5] = vx * sin_wt + vy * cos_wt

        # Update scale and height linearly
        next_state[2] = a + va * dt
        next_state[3] = h + vh * dt

        return next_state

    def get_process_noise(self, state: np.ndarray) -> np.ndarray:
        """Process noise for the CT model."""
        std_pos = 0.1 * state[3]
        std_vel = 0.2 * state[3]
        std_omega = 0.1  # Angular velocity noise (rad/s)
        std_size = 0.01 * state[3]

        q = np.array([
            std_pos, std_pos, std_size, std_pos,
            std_vel, std_vel, std_size * 0.1, std_vel,
            std_omega
        ])
        return np.diag(q ** 2)


class UnscentedKalmanFilter:
    """Unscented Kalman Filter for nonlinear motion models."""

    def __init__(self, motion_model: MotionModel, alpha: float = 1e-3,
                 beta: float = 2.0, kappa: float = None):
        self.motion_model = motion_model
        self.n = motion_model.state_dim

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa if kappa is not None else 3 - self.n

        # Compute scaling term and weights
        self.lambda_ = self.alpha ** 2 * (self.n + self.kappa) - self.n
        self.Wm, self.Wc = self._compute_weights()

        # Measurement noise covariance (fixed)
        std_obs = [0.1, 0.1, 0.01, 0.1]  # Observation noise for [x, y, a, h]
        self.R = np.diag(np.array(std_obs) ** 2)

    def _compute_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute unscented transform weights."""
        Wm = np.zeros(2 * self.n + 1)
        Wc = np.zeros(2 * self.n + 1)

        Wm[0] = self.lambda_ / (self.n + self.lambda_)
        Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha ** 2 + self.beta)

        for i in range(1, 2 * self.n + 1):
            Wm[i] = 1 / (2 * (self.n + self.lambda_))
            Wc[i] = 1 / (2 * (self.n + self.lambda_))

        return Wm, Wc

    def _generate_sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Generate sigma points."""
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = mean

        # Ensure the covariance matrix is positive definite
        cov_scaled = (self.n + self.lambda_) * cov

        # Add a small regularization term for numerical stability
        cov_scaled += np.eye(self.n) * 1e-8

        try:
            # Try Cholesky decomposition first
            sqrt = cholesky(cov_scaled, lower=True)
        except np.linalg.LinAlgError:
            # Fall back to SVD if Cholesky fails
            U, s, Vt = np.linalg.svd(cov_scaled)
            s = np.maximum(s, 1e-8)
            sqrt = U @ np.diag(np.sqrt(s))

        sqrt = np.real(sqrt)

        for i in range(self.n):
            sigma_points[i + 1] = mean + sqrt[i]
            sigma_points[i + 1 + self.n] = mean - sqrt[i]

        return sigma_points

    def predict(self, mean: np.ndarray, cov: np.ndarray, Q: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """UKF prediction step."""
        if Q is None:
            Q = self.motion_model.get_process_noise(mean)

        # Generate sigma points
        sigma_points = self._generate_sigma_points(mean, cov)

        # Propagate sigma points
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(sigma_points.shape[0]):
            sigma_points_pred[i] = self.motion_model.f(sigma_points[i])

        # Predicted mean and covariance
        mean_pred = np.sum(self.Wm[:, np.newaxis] * sigma_points_pred, axis=0)

        cov_pred = Q.copy()
        for i in range(sigma_points_pred.shape[0]):
            diff = sigma_points_pred[i] - mean_pred
            cov_pred += self.Wc[i] * np.outer(diff, diff)

        cov_pred = (cov_pred + cov_pred.T) / 2
        cov_pred += np.eye(cov_pred.shape[0]) * 1e-8

        return mean_pred, cov_pred

    def update(self, mean_pred: np.ndarray, cov_pred: np.ndarray,
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """UKF update step."""
        # Generate sigma points from predicted state
        sigma_points = self._generate_sigma_points(mean_pred, cov_pred)

        # Predict measurements
        obs_dim = len(measurement)
        gamma = np.zeros((sigma_points.shape[0], obs_dim))
        for i in range(sigma_points.shape[0]):
            gamma[i] = self.motion_model.h(sigma_points[i])

        # Predicted measurement mean
        z_pred = np.sum(self.Wm[:, np.newaxis] * gamma, axis=0)

        # Innovation covariance S
        S = self.R.copy()
        for i in range(gamma.shape[0]):
            diff = gamma[i] - z_pred
            S += self.Wc[i] * np.outer(diff, diff)

        # Cross covariance T
        T = np.zeros((self.n, obs_dim))
        for i in range(sigma_points.shape[0]):
            state_diff = sigma_points[i] - mean_pred
            obs_diff = gamma[i] - z_pred
            T += self.Wc[i] * np.outer(state_diff, obs_diff)

        # Regularize S for stability
        S_reg = S + np.eye(S.shape[0]) * 1e-8

        # Kalman gain
        K = T @ np.linalg.inv(S_reg)

        # Innovation
        innovation = measurement - z_pred

        # State update
        mean_updated = mean_pred + K @ innovation
        cov_updated = cov_pred - K @ S_reg @ K.T

        cov_updated = (cov_updated + cov_updated.T) / 2

        # Log-likelihood for model probability update
        log_likelihood = self._compute_log_likelihood(innovation, S)

        return mean_updated, cov_updated, log_likelihood

    def _compute_log_likelihood(self, innovation: np.ndarray, S: np.ndarray) -> float:
        """Compute Gaussian log-likelihood."""
        try:
            S_reg = S + np.eye(S.shape[0]) * 1e-8
            log_likelihood = -0.5 * (
                innovation.T @ np.linalg.inv(S_reg) @ innovation +
                np.log(np.linalg.det(2 * np.pi * S_reg))
            )
            return float(log_likelihood)
        except Exception:
            return -np.inf


class AIMUKFFilter:
    """Adaptive Interacting Multiple Model Unscented Kalman Filter."""

    def __init__(self, dt: float = 1.0):
        self.dt = dt

        # Build the motion model set
        self.models = [
            ConstantVelocityModel(dt),
            ConstantAccelerationModel(dt),
            CoordinatedTurnModel(dt)
        ]

        # Create one UKF per motion model
        self.ukf_filters = [UnscentedKalmanFilter(model) for model in self.models]

        # Number of models
        self.num_models = len(self.models)

        # Model transition probability matrix (Markov chain)
        # Diagonal-dominant: prefer staying in the current mode
        self.transition_prob = np.array([
            [0.70, 0.20, 0.10],  # CV -> CV, CA, CT
            [0.20, 0.70, 0.10],  # CA -> CV, CA, CT
            [0.35, 0.25, 0.40]   # CT -> CV, CA, CT
        ])

        # Initial model probabilities (biased toward CV)
        self.model_prob = np.array([0.7, 0.2, 0.1])

        # Adaptive noise parameters
        self.base_noise_factor = 2.0
        self.occlusion_noise_factor = 2.0   # Noise amplification during occlusion
        self.maneuver_noise_factor = 3.0    # Noise amplification during maneuvers

        # Maneuver detection parameters
        self.maneuver_threshold = 0.3
        self.prev_model_prob = self.model_prob.copy()

        # State bookkeeping
        self.occlusion_time = 0
        self.is_maneuvering = False

        logging.info("AIMM-UKF Filter initialized with models: CV, CA, CT")

    def initiate(self, measurement: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Initialize multi-model states from the first measurement."""
        means = []
        covariances = []

        for model in self.models:
            if model.name == "CV":
                # CV: [x, y, a, h, vx, vy, va, vh]
                mean = np.zeros(8)
                mean[:4] = measurement
                std = np.array([
                    measurement[3] * 0.1, measurement[3] * 0.1, 0.01, measurement[3] * 0.1,
                    measurement[3] * 0.5, measurement[3] * 0.5, 0.001, measurement[3] * 0.5
                ])

            elif model.name == "CA":
                # CA: [x, y, a, h, vx, vy, va, vh, ax, ay]
                mean = np.zeros(10)
                mean[:4] = measurement
                std = np.array([
                    measurement[3] * 0.1, measurement[3] * 0.1, 0.01, measurement[3] * 0.1,
                    measurement[3] * 0.5, measurement[3] * 0.5, 0.001, measurement[3] * 0.5,
                    measurement[3] * 0.1, measurement[3] * 0.1
                ])

            elif model.name == "CT":
                # CT: [x, y, a, h, vx, vy, va, vh, ω]
                mean = np.zeros(9)
                mean[:4] = measurement
                std = np.array([
                    measurement[3] * 0.1, measurement[3] * 0.1, 0.01, measurement[3] * 0.1,
                    measurement[3] * 0.5, measurement[3] * 0.5, 0.001, measurement[3] * 0.5,
                    0.1  # Initial angular velocity uncertainty
                ])

            covariance = np.diag(std ** 2)
            means.append(mean)
            covariances.append(covariance)

        return means, covariances

    def predict(self, means: List[np.ndarray], covariances: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """IMM prediction step."""
        # Step 1: compute mixing probabilities
        c_j = self.transition_prob.T @ self.model_prob
        omega = np.zeros((self.num_models, self.num_models))

        for i in range(self.num_models):
            for j in range(self.num_models):
                if c_j[j] > 1e-10:
                    omega[i, j] = (self.transition_prob[i, j] * self.model_prob[i]) / c_j[j]

        # Step 2: mix states
        mixed_means = []
        mixed_covariances = []

        for j in range(self.num_models):
            target_dim = self.models[j].state_dim
            mixed_mean = np.zeros(target_dim)

            # Weighted mean with dimension alignment
            total_weight = 0
            for i in range(self.num_models):
                weight = omega[i, j]
                if weight > 1e-10:
                    source_mean = (
                        means[i][:target_dim]
                        if len(means[i]) >= target_dim
                        else np.pad(means[i], (0, target_dim - len(means[i])))
                    )
                    mixed_mean += weight * source_mean
                    total_weight += weight

            if total_weight > 1e-10:
                mixed_mean /= total_weight

            # Mixed covariance
            mixed_cov = np.eye(target_dim) * 1e-4
            for i in range(self.num_models):
                weight = omega[i, j]
                if weight > 1e-10:
                    source_cov = covariances[i]
                    if source_cov.shape[0] != target_dim:
                        # Adjust covariance dimension
                        if source_cov.shape[0] > target_dim:
                            source_cov = source_cov[:target_dim, :target_dim]
                        else:
                            temp_cov = np.eye(target_dim) * 1e-4
                            min_dim = min(source_cov.shape[0], target_dim)
                            temp_cov[:min_dim, :min_dim] = source_cov[:min_dim, :min_dim]
                            source_cov = temp_cov

                    source_mean = (
                        means[i][:target_dim]
                        if len(means[i]) >= target_dim
                        else np.pad(means[i], (0, target_dim - len(means[i])))
                    )
                    diff = source_mean - mixed_mean
                    mixed_cov += weight * (source_cov + np.outer(diff, diff))

            mixed_cov = (mixed_cov + mixed_cov.T) / 2
            mixed_cov += np.eye(target_dim) * 1e-8

            mixed_means.append(mixed_mean)
            mixed_covariances.append(mixed_cov)

        # Step 3: predict each model independently
        pred_means = []
        pred_covariances = []

        for j in range(self.num_models):
            Q = self._get_adaptive_process_noise(j, mixed_means[j])
            mean_pred, cov_pred = self.ukf_filters[j].predict(
                mixed_means[j], mixed_covariances[j], Q
            )
            pred_means.append(mean_pred)
            pred_covariances.append(cov_pred)

        # Update predicted model probabilities
        self.model_prob = c_j

        return pred_means, pred_covariances

    def update(self, pred_means: List[np.ndarray], pred_covariances: List[np.ndarray],
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """IMM update step."""
        # A measurement is available; reset occlusion counter
        self.occlusion_time = 0

        # Step 1: update each model independently
        updated_means = []
        updated_covariances = []
        likelihoods = []

        for j in range(self.num_models):
            mean_updated, cov_updated, log_likelihood = self.ukf_filters[j].update(
                pred_means[j], pred_covariances[j], measurement
            )

            updated_means.append(mean_updated)
            updated_covariances.append(cov_updated)
            likelihoods.append(np.exp(log_likelihood))

        # Step 2: update model probabilities
        likelihoods = np.array(likelihoods)
        likelihoods = np.maximum(likelihoods, 1e-300)

        unnormalized_prob = self.model_prob * likelihoods
        normalization_factor = np.sum(unnormalized_prob)

        if normalization_factor > 1e-300:
            self.model_prob = unnormalized_prob / normalization_factor
        else:
            # Fall back to uniform distribution if all likelihoods are too small
            self.model_prob = np.ones(self.num_models) / self.num_models

        # Step 3: maneuver detection
        prob_change = np.linalg.norm(self.model_prob - self.prev_model_prob)
        self.is_maneuvering = prob_change > self.maneuver_threshold
        self.prev_model_prob = self.model_prob.copy()

        # Step 4: fuse model states into CV format for compatibility
        cv_mean = np.zeros(8)
        cv_cov = np.eye(8) * 1e-4

        for j in range(self.num_models):
            weight = self.model_prob[j]

            # Convert each model state into CV format
            if self.models[j].name == "CV":
                model_mean_cv = updated_means[j]
                model_cov_cv = updated_covariances[j]
            elif self.models[j].name == "CA":
                # CA -> CV: drop acceleration terms
                model_mean_cv = updated_means[j][:8]
                model_cov_cv = updated_covariances[j][:8, :8]
            elif self.models[j].name == "CT":
                # CT -> CV: drop angular velocity term
                model_mean_cv = updated_means[j][:8]
                model_cov_cv = updated_covariances[j][:8, :8]

            cv_mean += weight * model_mean_cv
            diff = model_mean_cv - cv_mean
            cv_cov += weight * (model_cov_cv + np.outer(diff, diff))

        cv_cov = (cv_cov + cv_cov.T) / 2
        cv_cov += np.eye(8) * 1e-8

        logging.debug(
            f"Model probabilities: CV={self.model_prob[0]:.3f}, "
            f"CA={self.model_prob[1]:.3f}, CT={self.model_prob[2]:.3f}"
        )

        return cv_mean, cv_cov

    def predict_no_update(self, means: List[np.ndarray], covariances: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Prediction only, without measurement update (during occlusion)."""
        self.occlusion_time += 1
        return self.predict(means, covariances)

    def _get_adaptive_process_noise(self, model_idx: int, state: np.ndarray) -> np.ndarray:
        """Compute adaptive process noise covariance."""
        # Base process noise
        Q_base = self.models[model_idx].get_process_noise(state)

        # Adaptive scaling factor
        adaptive_factor = self.base_noise_factor

        # Occlusion adaptation: longer occlusion -> larger noise
        if self.occlusion_time > 0:
            occlusion_factor = 1.0 + self.occlusion_noise_factor * np.tanh(self.occlusion_time / 5.0)
            adaptive_factor *= occlusion_factor

        # Maneuver adaptation: increase noise when maneuvering is detected
        if self.is_maneuvering:
            adaptive_factor *= self.maneuver_noise_factor

        return adaptive_factor * Q_base

    def get_state_summary(self) -> Dict:
        """Return a summary of the current filter state."""
        return {
            "model_probabilities": self.model_prob.tolist(),
            "dominant_model": self.models[np.argmax(self.model_prob)].name,
            "occlusion_time": self.occlusion_time,
            "is_maneuvering": self.is_maneuvering,
            "adaptive_noise_factor": self._compute_current_noise_factor()
        }

    def _compute_current_noise_factor(self) -> float:
        """Compute the current adaptive noise scaling factor."""
        factor = self.base_noise_factor
        if self.occlusion_time > 0:
            factor *= (1.0 + self.occlusion_noise_factor * np.tanh(self.occlusion_time / 5.0))
        if self.is_maneuvering:
            factor *= self.maneuver_noise_factor
        return factor


# Convenience factory function
def create_aimm_ukf_filter(dt: float = 1.0) -> AIMUKFFilter:
    """Create an AIMM-UKF filter instance."""
    return AIMUKFFilter(dt)
