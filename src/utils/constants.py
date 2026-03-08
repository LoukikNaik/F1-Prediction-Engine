"""Constants and enums for the F1 Prediction Engine."""

from enum import Enum


class CircuitType(str, Enum):
    STREET = "street"
    HIGH_SPEED = "high_speed"
    TECHNICAL = "technical"
    HYBRID = "hybrid"


class StandingType(str, Enum):
    DRIVER = "driver"
    CONSTRUCTOR = "constructor"


class SessionType(str, Enum):
    PRACTICE_1 = "FP1"
    PRACTICE_2 = "FP2"
    PRACTICE_3 = "FP3"
    QUALIFYING = "Q"
    SPRINT = "S"
    SPRINT_QUALIFYING = "SQ"
    RACE = "R"


class PredictionStage(str, Enum):
    PRE_WEEKEND = "pre_weekend"
    POST_QUALIFYING = "post_qualifying"
    POST_SPRINT = "post_sprint"
    RACE_EVE = "race_eve"
    LIVE = "live"


# Number of grid positions (22 from 2026 with Cadillac as 11th team)
GRID_SIZE = 22

# Live prediction constants
LIVE_POLL_INTERVAL_SECONDS = 90
LIVE_MONTE_CARLO_SIMULATIONS = 5_000
LIVE_MIN_LAPS_FOR_PACE = 3

# Points system (2025+)
POINTS_SYSTEM = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1,
}

# Sprint race points system
SPRINT_POINTS_SYSTEM = {
    1: 8, 2: 7, 3: 6, 4: 5, 5: 4,
    6: 3, 7: 2, 8: 1,
}

# DNF position encoding for ML
DNF_POSITION = 22
