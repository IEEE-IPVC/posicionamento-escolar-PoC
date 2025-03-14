from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

@dataclass
class AccessPoint:
    x: float
    y: float
    rssi: float
    distance: float = 0.0

def rssi_to_distance(rssi: float, A: float = -40, n: float = 2) -> float:
    return 10 ** ((A - rssi) / (10 * n))

class PositionEstimator:
    def __init__(self, access_points: List[AccessPoint]):
        self.access_points = access_points
        for ap in self.access_points:
            ap.distance = rssi_to_distance(ap.rssi)

    def _error_function(self, pos: Tuple[float, float]) -> float:
        x, y = pos
        return sum((np.sqrt((x - ap.x) ** 2 + (y - ap.y) ** 2) - ap.distance) ** 2 for ap in self.access_points)

    def estimate_position(self, initial_guess: Tuple[float, float]) -> Tuple[float, float]:
        result = minimize(self._error_function, initial_guess, method='Nelder-Mead')
        return result.x

class DirectionVectorCalculator:
    @staticmethod
    def calculate_direction_vector(estimated_x: float, estimated_y: float, x_new: float, y_new: float) -> Tuple[float, float]:
        dx = x_new - estimated_x
        dy = y_new - estimated_y
        magnitude = np.sqrt(dx**2 + dy**2)
        return (dx / magnitude, dy / magnitude)

access_points: List[AccessPoint] = [
    AccessPoint(x=0, y=0, rssi=-40),
    AccessPoint(x=10, y=0, rssi=-50),
    AccessPoint(x=5, y=10, rssi=-45),
]

position_estimator = PositionEstimator(access_points)
initial_guess: Tuple[float, float] = (5, 5)
estimated_x, estimated_y = position_estimator.estimate_position(initial_guess)
print(f"Posição Estimada: ({estimated_x:.2f}, {estimated_y:.2f})")

x_new, y_new = 7, 7  
direction_vector = DirectionVectorCalculator.calculate_direction_vector(estimated_x, estimated_y, x_new, y_new)
print(f"Vetor Direcional: {direction_vector}")
