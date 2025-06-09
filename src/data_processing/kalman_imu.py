import numpy as np
from tqdm import tqdm

class KalmanIMU:
    def __init__(self, acc, gyr, frequency, Q_angle=0.001, Q_gyro=0.003, R_angle=0.03):
        assert acc.shape == gyr.shape and acc.shape[1] == 3
        self.acc = acc.astype(np.float32)
        self.gyr = gyr.astype(np.float32)
        self.dt = 1.0 / frequency

        self.Q_angle = Q_angle
        self.Q_gyro = Q_gyro
        self.R_angle = R_angle

        self.num_samples = len(acc)
        self.pitch = np.zeros(self.num_samples, dtype=np.float32)
        self.roll = np.zeros(self.num_samples, dtype=np.float32)

        # Kalman filter state variables for pitch and roll
        self.angle = np.zeros(2)  # [pitch, roll]
        self.bias = np.zeros(2)
        self.P = np.zeros((2, 2, 2))  # error covariance for pitch and roll

        self._run_filter()

    def _run_filter(self):
        for i in tqdm(range(self.num_samples), desc="Running Kalman Filter"):
            ax, ay, az = self.acc[i]
            gx, gy, gz = self.gyr[i]

            # Calculate pitch and roll from accelerometer
            acc_pitch = np.arctan2(ax, np.sqrt(ay**2 + az**2))
            acc_roll = np.arctan2(ay, np.sqrt(ax**2 + az**2))

            # --- Pitch Axis ---
            self._kalman_update(0, acc_pitch, gy)
            self.pitch[i] = self.angle[0]

            # --- Roll Axis ---
            self._kalman_update(1, acc_roll, gx)
            self.roll[i] = self.angle[1]

    def _kalman_update(self, idx, acc_angle, gyro_rate):
        dt = self.dt
        angle = self.angle[idx]
        bias = self.bias[idx]
        P = self.P[idx]

        # Predict
        rate = gyro_rate - bias
        angle += dt * rate
        P[0][0] += dt * (dt*P[1][1] - P[0][1] - P[1][0] + self.Q_angle)
        P[0][1] -= dt * P[1][1]
        P[1][0] -= dt * P[1][1]
        P[1][1] += self.Q_gyro * dt

        # Update
        y = acc_angle - angle
        S = P[0][0] + self.R_angle
        K = np.array([P[0][0] / S, P[1][0] / S])

        angle += K[0] * y
        bias += K[1] * y
        P00_temp = P[0][0]
        P01_temp = P[0][1]

        P[0][0] -= K[0] * P00_temp
        P[0][1] -= K[0] * P01_temp
        P[1][0] -= K[1] * P00_temp
        P[1][1] -= K[1] * P01_temp

        self.angle[idx] = angle
        self.bias[idx] = bias
        self.P[idx] = P

    @property
    def roll_pitch(self):
        return self.roll, self.pitch