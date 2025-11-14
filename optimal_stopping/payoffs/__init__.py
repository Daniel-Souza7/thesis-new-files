"""
Option Payoff Functions

This package contains various option payoff implementations for optimal stopping problems.
"""

# Base payoff class
from optimal_stopping.payoffs.payoff import Payoff

# Standard (vanilla) options
from optimal_stopping.payoffs.standard import (
    BasketCall,
    BasketPut,
    MaxCall,
    MaxPut,
    MinCall,
    MinPut,
    GeometricBasketCall,
    GeometricBasketPut,
)

# Barrier options (single barrier)
from optimal_stopping.payoffs.barriers import (
    # Up-and-Out
    UpAndOutBasketCall,
    UpAndOutBasketPut,
    UpAndOutMaxCall,
    UpAndOutMaxPut,
    UpAndOutMinCall,
    UpAndOutMinPut,
    UpAndOutGeometricBasketCall,
    UpAndOutGeometricBasketPut,
    # Down-and-Out
    DownAndOutBasketCall,
    DownAndOutBasketPut,
    DownAndOutMaxCall,
    DownAndOutMaxPut,
    DownAndOutMinCall,
    DownAndOutMinPut,
    DownAndOutGeometricBasketCall,
    DownAndOutGeometricBasketPut,
    # Up-and-In
    UpAndInBasketCall,
    UpAndInBasketPut,
    UpAndInMaxCall,
    UpAndInMaxPut,
    UpAndInMinCall,
    UpAndInMinPut,
    UpAndInGeometricBasketCall,
    UpAndInGeometricBasketPut,
    # Down-and-In
    DownAndInBasketCall,
    DownAndInBasketPut,
    DownAndInMaxCall,
    DownAndInMaxPut,
    DownAndInMinCall,
    DownAndInMinPut,
    DownAndInGeometricBasketCall,
    DownAndInGeometricBasketPut,
)

# Lookback options
from optimal_stopping.payoffs.lookbacks import (
    LookbackFloatingStrikeCall,
    LookbackFloatingStrikePut,
    LookbackFixedStrikeCall,
    LookbackFixedStrikePut,
)

# Double barrier options
from optimal_stopping.payoffs.double_barriers import (
    DoubleKnockOutCall,
    DoubleKnockOutPut,
    DoubleKnockInCall,
    DoubleKnockInPut,
    UpInDownOutCall,
    UpInDownOutPut,
    UpOutDownInCall,
    UpOutDownInPut,
    PartialTimeBarrierCall,
    StepBarrierCall,
    DoubleKnockOutLookbackFloatingCall,
    DoubleKnockOutLookbackFloatingPut,
)

# Niche/specialized options
from optimal_stopping.payoffs.niche import (
    BestOfKCall,
    WorstOfKCall,
    RankWeightedBasketCall,
    ChooserBasketOption,
    RangeCall,
    DispersionCall,
)

# Leveraged positions
from optimal_stopping.payoffs.leverage import (
    LeveragedBasketLongPosition,
    LeveragedBasketShortPosition,
    LeveragedBasketLongStopLoss,
    LeveragedBasketShortStopLoss,
)

__all__ = [
    # Base
    'Payoff',
    # Standard
    'BasketCall', 'BasketPut', 'MaxCall', 'MaxPut', 'MinCall', 'MinPut',
    'GeometricBasketCall', 'GeometricBasketPut',
    # Barriers - Up-and-Out
    'UpAndOutBasketCall', 'UpAndOutBasketPut', 'UpAndOutMaxCall', 'UpAndOutMaxPut',
    'UpAndOutMinCall', 'UpAndOutMinPut', 'UpAndOutGeometricBasketCall', 'UpAndOutGeometricBasketPut',
    # Barriers - Down-and-Out
    'DownAndOutBasketCall', 'DownAndOutBasketPut', 'DownAndOutMaxCall', 'DownAndOutMaxPut',
    'DownAndOutMinCall', 'DownAndOutMinPut', 'DownAndOutGeometricBasketCall', 'DownAndOutGeometricBasketPut',
    # Barriers - Up-and-In
    'UpAndInBasketCall', 'UpAndInBasketPut', 'UpAndInMaxCall', 'UpAndInMaxPut',
    'UpAndInMinCall', 'UpAndInMinPut', 'UpAndInGeometricBasketCall', 'UpAndInGeometricBasketPut',
    # Barriers - Down-and-In
    'DownAndInBasketCall', 'DownAndInBasketPut', 'DownAndInMaxCall', 'DownAndInMaxPut',
    'DownAndInMinCall', 'DownAndInMinPut', 'DownAndInGeometricBasketCall', 'DownAndInGeometricBasketPut',
    # Lookbacks
    'LookbackFloatingStrikeCall', 'LookbackFloatingStrikePut',
    'LookbackFixedStrikeCall', 'LookbackFixedStrikePut',
    # Double Barriers
    'DoubleKnockOutCall', 'DoubleKnockOutPut', 'DoubleKnockInCall', 'DoubleKnockInPut',
    'UpInDownOutCall', 'UpInDownOutPut', 'UpOutDownInCall', 'UpOutDownInPut',
    'PartialTimeBarrierCall', 'StepBarrierCall',
    'DoubleKnockOutLookbackFloatingCall', 'DoubleKnockOutLookbackFloatingPut',
    # Niche
    'BestOfKCall', 'WorstOfKCall', 'RankWeightedBasketCall',
    'ChooserBasketOption', 'RangeCall', 'DispersionCall',
    # Leverage
    'LeveragedBasketLongPosition', 'LeveragedBasketShortPosition',
    'LeveragedBasketLongStopLoss', 'LeveragedBasketShortStopLoss',
]
