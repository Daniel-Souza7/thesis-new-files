from dataclasses import dataclass
import typing
from typing import Iterable
import numpy as np

FigureType = typing.NewType('FigureType', str)
TablePrice = FigureType("TablePrice")
TableDuration = FigureType("TableDuration")
PricePerNbPaths = FigureType("PricePerNbPaths")

'''
Algos:NLSM, LSM, LS2, DOS, HP, LSPI, FQI, FQIR, LN, LN2, LND
New algos = RLSM, FQI, RFQI, LNfast
'''

class Seed:
    def __init__(self):
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed

    def get_seed(self):
        return self.seed


path_gen_seed = Seed()


@dataclass
class _DefaultConfig:
  algos: Iterable[str] = ('NLSM',
                          'LSM',
                          'DOS',
                          'FQI',
                          'RFQI',
                          'LNfast',
                          'LN2',
                          'RLSM', 'SRFQI', 'SRLSM')
  dividends: Iterable[float] = (0.0,)
  nb_dates: Iterable[int] = (10,)
  drift: Iterable[float] = (0.06,)
  risk_free_rate: Iterable[float] = (None,)  # Risk-free rate for discounting (None = drift - 0.04 = 0.02)
  mean: Iterable[float] = (0.01,)
  speed: Iterable[float] = (2,)
  correlation: Iterable[float] = (-0.3,)
  hurst: Iterable[float] = (0.75,)
  stock_models: Iterable[str] = ('BlackScholes',)
  strikes: Iterable[float] = (100,)
  maturities: Iterable[float] = (1,)
  nb_paths: Iterable[int] = (20000,)
  nb_runs: int = 10
  nb_stocks: Iterable[int] = (1,)
  payoffs: Iterable[str] = ('MaxCall',)
  spots: Iterable[float] = (100,)
  volatilities: Iterable[float] = (0.2,)
  hidden_size: Iterable[int] = (20,)
  nb_epochs: Iterable[int] = (30,)
  factors: Iterable[Iterable[float]] = ((1.,1.,1.),)
  ridge_coeff: Iterable[float] = (1.,)
  train_ITM_only: Iterable[bool] = (True,)
  use_path: Iterable[bool] = (False,)
  use_payoff_as_input: Iterable[bool] = (False,)
  barriers: Iterable[float] = (1,)  # Barrier level (None = use default)
  barriers_up: Iterable[float] = (1,)  # Upper barrier level for double barriers
  barriers_down: Iterable[float] = (1,)  # Lower barrier level for double barriers
  k: Iterable[int] = (2,)  # Number of assets for best-of-k/worst-of-k options
  weights: Iterable[tuple] = (None,)  # Custom weights for rank-weighted options (None = use formula)
  step_param1: Iterable[float] = (-1,)  # Lower bound for step barrier random walk
  step_param2: Iterable[float] = (1,)  # Upper bound for step barrier random walk
  step_param3: Iterable[float] = (-1,)  # Lower bound for double step barrier
  step_param4: Iterable[float] = (1,)  # Upper bound for double step barrier
  representations: Iterable[str] = ('TablePriceDuration',)

  # When adding a filter here, also add to filtering.py and read_data.py


'''
Test config for risk_free_rate functionality
'''

test_risk_free_rate = _DefaultConfig(
    algos=('RLSM', 'RFQI'), payoffs=('BasketCall',), nb_stocks=(50,),
    drift=(0.05,), risk_free_rate=(None, 0.05, 0.1, 0.2, 0.3,),
    nb_paths=(50000,), nb_dates=(10,), nb_runs=3
)

'''
Test config for EOP algorithm
'''
test_eop = _DefaultConfig(
    nb_runs=5,
    nb_paths=(100000,),
    nb_stocks=(1,),
    algos=('RLSM', 'LSM', 'RFQI', 'FQI', 'EOP'),
    payoffs=('Call',)
)

'''
Exercise times analysis with varying hidden sizes
'''
Exercisetimes = _DefaultConfig(
    nb_runs=5,
    nb_paths=(100000,),
    nb_stocks=(1,),
    algos=('RLSM', 'LSM', 'RFQI', 'FQI'),
    payoffs=('Call',),
    hidden_size=(1, 2, 4, 20, 100)
)

'''
Diagnostic config to debug early exercise issue
Tests with train_ITM_only=False and more basis functions
'''
debug_early_exercise = _DefaultConfig(
    nb_runs=5,
    nb_paths=(100000,),
    nb_stocks=(1,),
    algos=('RLSM', 'LSM', 'RFQI', 'FQI', 'EOP'),
    payoffs=('Call',),
    train_ITM_only=(True, False),  # Train on ALL paths, not just ITM
    hidden_size=(20, 100)  # More basis functions for better approximation
)

video_testing2 = _DefaultConfig(
    nb_paths=(40000,),
    nb_stocks=(1,),
    algos=('LSM',),
    payoffs=('Call',),
    train_ITM_only=(True,),
)


'''
Comparison prices and computation time
'''

@dataclass
class _DimensionTable(_DefaultConfig):
  algos: Iterable[str] = ('NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS', 'EOP')
  nb_stocks: Iterable[int] = (5, 10, 50, 100)

@dataclass
class _FastTable(_DefaultConfig):
  algos: Iterable[str] = ('NLSM', 'RFQI', 'RLSM', 'DOS', 'EOP')
  nb_stocks: Iterable[int] = (500, 1000, 1500)

@dataclass
class _FasterTable(_DefaultConfig):
  algos: Iterable[str] = ('RFQI', 'RLSM', 'EOP', 'SRFQI', 'SRLSM')
  nb_stocks: Iterable[int] = (100, 500, 1000)


# tables with basis functions
@dataclass
class _SmallDimensionTable(_DefaultConfig):
  algos: Iterable[str] = ('LSM', 'FQI')
  nb_stocks: Iterable[int] = (5, 10, 50, 100)


@dataclass
class _VerySmallDimensionTable(_DefaultConfig):
  nb_stocks: Iterable[int] = (5, 10, 20)
  algos: Iterable[str] = ('NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS')



'''
Comparison prices and computation time 
'''

doublebarriers= _FasterTable(
    algos=['SRLSM', 'SRFQI'],  # Only path-dependent algorithms
    payoffs=[
        'DoubleBarrierBasketCall',
    ],
    nb_stocks=[15, 50],
    strikes=[100],
    spots=[100],
    nb_paths=[10000],
    nb_dates=[9],
    barriers_up=[110],
    barriers_down=[90]
)




smoketest = _FasterTable(
    algos = ('SRFQI', 'SRLSM'),
    nb_stocks =(1, 3),
    payoffs = (

       'DoubleKnockOutCall', 'DoubleKnockOutPut',
        'DoubleKnockInCall', 'DoubleKnockInPut',
        'UpInDownOutCall', 'UpInDownOutPut',
        'UpOutDownInCall', 'UpOutDownInPut',

       'LookbackFixedCall',
       'LookbackFixedPut',
       'LookbackFloatCall',
       'LookbackFloatPut',
       'LookbackMaxCall',
       'LookbackMinPut',
        # Up-And-Out (8)
        'UpAndOutMaxCall', 'UpAndOutMaxPut',
        'UpAndOutMinCall', 'UpAndOutMinPut',
        'UpAndOutBasketCall', 'UpAndOutBasketPut',
        'UpAndOutGeometricBasketCall', 'UpAndOutGeometricBasketPut',
        # Down-And-Out (8)
        'DownAndOutMaxCall', 'DownAndOutMaxPut',
        'DownAndOutMinCall', 'DownAndOutMinPut',
        'DownAndOutBasketCall', 'DownAndOutBasketPut',
        'DownAndOutGeometricBasketCall', 'DownAndOutGeometricBasketPut',
        # Up-And-In (8)
        'UpAndInMaxCall', 'UpAndInMaxPut',
        'UpAndInMinCall', 'UpAndInMinPut',
        'UpAndInBasketCall', 'UpAndInBasketPut',
        'UpAndInGeometricBasketCall', 'UpAndInGeometricBasketPut',
        # Down-And-In (8)
        'DownAndInMaxCall', 'DownAndInMaxPut',
        'DownAndInMinCall', 'DownAndInMinPut',
        'DownAndInBasketCall', 'DownAndInBasketPut',
        'DownAndInGeometricBasketCall', 'DownAndInGeometricBasketPut',
    ),
    barriers = (80, 90, 95, 100, 105, 110, 120),
    strikes = (100,),
    spots = (100,),
    use_payoff_as_input = (True,),
    nb_runs = 10,
    nb_paths = (20000,),
    nb_dates = (6,),
    barriers_up=[110],
    barriers_down=[90]
)

smokefest = _FasterTable(
    algos = ('SRFQI', 'SRLSM'),
    nb_stocks =(1, 3),
    payoffs = (
'DoubleKnockOutLookbackFloatingPut', 'DoubleKnockOutLookbackFloatingCall'
    ),
    strikes = (100,),
    spots = (100,),
    use_payoff_as_input = (True,),
    nb_runs = 10,
    nb_paths = (20000,),
    nb_dates = (6,),
    barriers_up=[110],
    barriers_down=[90]
)

debug = _FasterTable(
    algos = ('SRLSM', 'SRFQI'),
    nb_stocks =(5,),
    payoffs = (
        # Up-And-Out (8)
        'UpAndOutMaxCall', 'UpAndOutMaxPut',
        'UpAndOutMinCall', 'UpAndOutMinPut',
        'UpAndOutBasketCall', 'UpAndOutBasketPut',
        'UpAndOutGeometricBasketCall', 'UpAndOutGeometricBasketPut',

    ),
    barriers = (80, 90, 95, 100, 105, 110, 120),
    strikes = (100,),
    spots = (100,),
    use_payoff_as_input = (True,),
    nb_runs = 1,
    nb_paths = (10000,),
    nb_dates = (3,), hidden_size=(100,) )

debug3 = _FasterTable(
    algos = ('SRLSM', 'SRFQI'),
    nb_stocks =(2, 5,),
    payoffs = (
        # Up-And-Out (8)
        'UpAndOutMaxCall', 'UpAndOutMaxPut',
        'UpAndOutMinCall', 'UpAndOutMinPut',
        'UpAndOutBasketCall', 'UpAndOutBasketPut',
        'UpAndOutGeometricBasketCall', 'UpAndOutGeometricBasketPut',

    ),
    barriers = (80, 90, 95, 100, 105, 110, 120),
    strikes = (100,),
    spots = (100,),
    use_payoff_as_input = (True,),
    nb_runs = 1,
    nb_paths = (10000,),
    nb_dates = (3,), hidden_size=(100,) )


debug2 = _FasterTable(
    algos = ('RFQI', 'RLSM'),
    nb_stocks =(15, 25, 50),
    payoffs = (# Standard (8)
        'MaxCall', 'MaxPut', 'MinCall', 'MinPut',
        'BasketCall', 'BasketPut',
        'GeometricBasketCall', 'GeometricBasketPut'


    ),

    strikes = (100,),
    spots = (100,),
    use_payoff_as_input = (True,),
    nb_runs = 5,
    nb_paths = (10000,),
    nb_dates = (8,))

debug22 = _FasterTable(
    algos = ('SRFQI', 'SRLSM'),
    nb_stocks =(15, 25, 50),
    payoffs = (# Standard (8)
        # Up-And-Out (8)
        'UpAndOutMaxCall', 'UpAndOutMaxPut',
        'UpAndOutMinCall', 'UpAndOutMinPut',
        'UpAndOutBasketCall', 'UpAndOutBasketPut',
        'UpAndOutGeometricBasketCall', 'UpAndOutGeometricBasketPut',
        # Down-And-Out (8)
        'DownAndOutMaxCall', 'DownAndOutMaxPut',
        'DownAndOutMinCall', 'DownAndOutMinPut',
        'DownAndOutBasketCall', 'DownAndOutBasketPut',
        'DownAndOutGeometricBasketCall', 'DownAndOutGeometricBasketPut',
        # Up-And-In (8)
        'UpAndInMaxCall', 'UpAndInMaxPut',
        'UpAndInMinCall', 'UpAndInMinPut',
        'UpAndInBasketCall', 'UpAndInBasketPut',
        'UpAndInGeometricBasketCall', 'UpAndInGeometricBasketPut',
        # Down-And-In (8)
        'DownAndInMaxCall', 'DownAndInMaxPut',
        'DownAndInMinCall', 'DownAndInMinPut',
        'DownAndInBasketCall', 'DownAndInBasketPut',
        'DownAndInGeometricBasketCall', 'DownAndInGeometricBasketPut',
    ),
    barriers = (1, 80, 90, 95, 100, 105, 110, 120, 100000),
    strikes = (100,),
    spots = (100,),
    use_payoff_as_input = (True,),
    nb_runs = 5,
    nb_paths = (10000,),
    nb_dates = (8,))

debug22neg = _FasterTable(
    algos = ('SRFQI', 'SRLSM'),
    nb_stocks =(15, 25, 50),
    payoffs = (# Standard (8)
        # Up-And-Out (8)
        'UpAndOutMaxCall', 'UpAndOutMaxPut',
        'UpAndOutMinCall', 'UpAndOutMinPut',
        'UpAndOutBasketCall', 'UpAndOutBasketPut',
        'UpAndOutGeometricBasketCall', 'UpAndOutGeometricBasketPut',
        # Down-And-Out (8)
        'DownAndOutMaxCall', 'DownAndOutMaxPut',
        'DownAndOutMinCall', 'DownAndOutMinPut',
        'DownAndOutBasketCall', 'DownAndOutBasketPut',
        'DownAndOutGeometricBasketCall', 'DownAndOutGeometricBasketPut',
        # Up-And-In (8)
        'UpAndInMaxCall', 'UpAndInMaxPut',
        'UpAndInMinCall', 'UpAndInMinPut',
        'UpAndInBasketCall', 'UpAndInBasketPut',
        'UpAndInGeometricBasketCall', 'UpAndInGeometricBasketPut',
        # Down-And-In (8)
        'DownAndInMaxCall', 'DownAndInMaxPut',
        'DownAndInMinCall', 'DownAndInMinPut',
        'DownAndInBasketCall', 'DownAndInBasketPut',
        'DownAndInGeometricBasketCall', 'DownAndInGeometricBasketPut',
    ),
    barriers = (1, 80, 90, 95, 100, 105, 110, 120, 100000),
    strikes = (100,),
    spots = (100,),
    use_payoff_as_input = (True,),
    nb_runs = 5,
    nb_paths = (10000,),
    nb_dates = (8,),
    drift=[-0.02])


debug222 = _FasterTable(
    algos = ('SRFQI', 'SRLSM'),
    nb_stocks =(15, 25, 50),
    payoffs = (# Standard (8)
        # Up-And-Out (8)
        'UpAndOutMaxCall', 'UpAndOutMaxPut',
        'UpAndOutMinCall', 'UpAndOutMinPut',
        'UpAndOutBasketCall', 'UpAndOutBasketPut',
        'UpAndOutGeometricBasketCall', 'UpAndOutGeometricBasketPut',
        # Down-And-Out (8)
        'DownAndOutMaxCall', 'DownAndOutMaxPut',
        'DownAndOutMinCall', 'DownAndOutMinPut',
        'DownAndOutBasketCall', 'DownAndOutBasketPut',
        'DownAndOutGeometricBasketCall', 'DownAndOutGeometricBasketPut',
        # Up-And-In (8)
        'UpAndInMaxCall', 'UpAndInMaxPut',

        'UpAndInBasketCall', 'UpAndInBasketPut',
        'UpAndInGeometricBasketCall', 'UpAndInGeometricBasketPut',
        # Down-And-In (8)
        'DownAndInMaxCall', 'DownAndInMaxPut',

        'DownAndInBasketCall', 'DownAndInBasketPut',
        'DownAndInGeometricBasketCall', 'DownAndInGeometricBasketPut',
    ),
    barriers = (1,),
    strikes = (100,),
    spots = (100,),
    use_payoff_as_input = (True,),
    nb_runs = 5,
    nb_paths = (10000,),
    nb_dates = (8,))

smoketest2 = _FasterTable(
    stock_models= ['RoughHeston'],
    algos = ('RFQI', 'RLSM'),
    nb_stocks =(10,100, 1000),
    payoffs = (
        # Standard (8)
        'MaxCall', 'MaxPut', 'MinCall', 'MinPut',
        'BasketCall', 'BasketPut',
        'GeometricCall', 'GeometricPut',

        # Up-And-Out (8)
        'UpAndOutMaxCall', 'UpAndOutMaxPut',
        'UpAndOutMinCall', 'UpAndOutMinPut',
        'UpAndOutBasketCall', 'UpAndOutBasketPut',
        'UpAndOutGeometricCall', 'UpAndOutGeometricPut',
        # Down-And-Out (8)
        'DownAndOutMaxCall', 'DownAndOutMaxPut',
        'DownAndOutMinCall', 'DownAndOutMinPut',
        'DownAndOutBasketCall', 'DownAndOutBasketPut',
        'DownAndOutGeometricCall', 'DownAndOutGeometricPut',
        # Up-And-In (8)
        'UpAndInMaxCall', 'UpAndInMaxPut',
        'UpAndInMinCall', 'UpAndInMinPut',
        'UpAndInBasketCall', 'UpAndInBasketPut',
        'UpAndInGeometricCall', 'UpAndInGeometricPut',
        # Down-And-In (8)
        'DownAndInMaxCall', 'DownAndInMaxPut',
        'DownAndInMinCall', 'DownAndInMinPut',
        'DownAndInBasketCall', 'DownAndInBasketPut',
        'DownAndInGeometricCall', 'DownAndInGeometricPut',
        # Lookback (6)
        'LookbackMaxCall', 'LookbackMaxPut',
        'LookbackMinCall', 'LookbackMinPut',
        'LookbackBasketCall', 'LookbackBasketPut'
    ),
    barriers = (80, 90, 95, 100, 105, 110, 120),
    strikes = (100,),
    spots = (100,),
    use_payoff_as_input = (True,),
    nb_runs = 5,
    nb_paths = (20000,),
    nb_dates = (10,), drift=(0.01,), hurst = [0.05])

lookback_config = _FasterTable(
    algos=['SRLSM', 'SRFQI'],  # Only path-dependent algorithms
    payoffs=[
        'LookbackFixedCall',
        'LookbackFixedPut',
        'LookbackFloatCall',
        'LookbackFloatPut',
        'LookbackMaxCall',
        'LookbackMinPut',
    ],
    nb_stocks=[5, 10],
    strikes=[100],
    barriers=[1,],
    spots=[100],
    nb_paths=[10000],
    nb_dates=[9],
)
# teste servidores
teste_servidores4747 = _FasterTable(payoffs=['MaxCall', 'BasketCall'], use_payoff_as_input=(True, False))

teste_servidores47 = _FasterTable(payoffs=['DownAndOutMaxCall'], use_payoff_as_input=(True, False),)
teste_servidores474 = _FasterTable(payoffs=['LookbackMaxCall'], use_payoff_as_input=(True, False),)

teste_servidores7447 = _FasterTable(payoffs=['DownAndOutMaxCall'], use_payoff_as_input=(True, False), barriers=(85,), nb_runs=1, algos=['RFQI', 'RLSM', 'EOP'])
teste_servidores4744 = _FasterTable(payoffs=['LookbackMaxCall'], use_payoff_as_input=(True, False),nb_runs=1)
teste_servidores747 = _FasterTable(payoffs=['MinPut'], use_payoff_as_input=(True, False), stock_models=['BlackScholes', 'Heston'], drift=(-0.02,))

muitos_parametros_diferentes = _DimensionTable(
    algos= ['RFQI', 'RLSM', 'EOP'], hidden_size= (10, 20, 50, 200,), use_payoff_as_input=(True, False),
    stock_models=['BlackScholes'], drift=(0.0,), nb_stocks=(100, 500, 1000,), nb_paths=(20000, 40000, 100000,), nb_epochs=(30, 40, 50,), nb_dates=(10, 25, 50,) )
# BS and Heston MaxCall
table_spots_Dim_BS_MaxCallr0 = _DimensionTable(
    spots=[80, 100, 120], drift=(0.0,), use_payoff_as_input=(True, False))
table_Dim_Heston_MaxCallr0 = _DimensionTable(
    stock_models=['Heston'], drift=(0.0,), use_payoff_as_input=(True, False))
#   -- table for do
algos = ['DOS',]
table_spots_Dim_BS_MaxCallr0_do = _DimensionTable(
    algos=algos, drift=(0.0,), use_payoff_as_input=(True, False),
    spots=[80, 100,  120])
table_Dim_Heston_MaxCallr0_do = _DimensionTable(
    algos=algos, drift=(0.0,), use_payoff_as_input=(True, False),
    stock_models=['Heston'])
#   -- tables with basis functions
table_spots_Dim_BS_MaxCallr0_bf = _SmallDimensionTable(
    spots=[80, 100,  120], drift=(0.0,), use_payoff_as_input=(True, False))
table_Dim_Heston_MaxCallr0_bf = _SmallDimensionTable(
    stock_models=['Heston'], drift=(0.0,), use_payoff_as_input=(True, False))
#   -- true price
table_spots_Dim_MaxCallr0_ref = _DimensionTable(
    spots=[80, 100, 120], drift=(0.0,), algos=["EOP"],
    stock_models=['BlackScholes', 'Heston'])
#   -- tables to generate output tables
algos = ['NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS', 'EOP']
table_spots_Dim_BS_MaxCallr0_gt1 = _DimensionTable(
    spots=[80, 100,  120], algos=algos,
    drift=(0.0,), use_payoff_as_input=(True, False,))
table_Dim_Heston_MaxCallr0_gt1 = _DimensionTable(
    stock_models=['Heston'], algos=algos,
    drift=(0.0,), use_payoff_as_input=(True, False))


#replicação

replicacao_servidores = _DimensionTable(
    spots=[80, 100, 120], drift=(0.0,), use_payoff_as_input=(True, False))

replicacao_servidores2 = _FastTable(
    spots=[80, 100, 120], drift=(0.0,), use_payoff_as_input=(True, False))


otimization00 = _FasterTable()
otimaztionpayoff = _FasterTable(use_payoff_as_input=(True, False))
otimization1 = _FasterTable(use_payoff_as_input=(True, False), nb_paths=(15000, 20000, 25000, 30000))


otimization2 = _FasterTable(use_payoff_as_input=(True, False), nb_dates=(8, 10, 12, 18))

otimization3 = _FasterTable(use_payoff_as_input=(True, False), hidden_size=(15, 20, 25, 30))


otimization4 = _FasterTable(use_payoff_as_input=(True, False), nb_epochs=(24, 30, 38, 45))

otimization5 = _FasterTable(use_payoff_as_input=(True, False), nb_runs=20)


# Heston with Var
table_Dim_HestonV_MaxCallr0_1 = _DimensionTable(
    algos=algos,
    stock_models=['HestonWithVar'], drift=(0.0,), nb_stocks=(5, 10, 50,), use_payoff_as_input=(True, False))
# Heston with Var2

table_Dim_HestonV_MaxCallr0_2 = _DimensionTable(
    algos= ['NLSM', 'RFQI', 'RLSM', 'DOS', 'EOP'],
    stock_models=['HestonWithVar'], drift=(0.0,), nb_stocks=(100,), use_payoff_as_input=(True, False))
table_Dim_HestonV_MaxCallr0_3 = _DimensionTable(
    algos= ['NLSM', 'RFQI', 'RLSM', 'DOS', 'EOP'],
    stock_models=['HestonWithVar'], drift=(0.0,), nb_stocks=(500,), use_payoff_as_input=(True, False))
table_Dim_HestonV_MaxCallr0_4 = _DimensionTable(
    algos= ['NLSM', 'RFQI', 'RLSM', 'DOS', 'EOP'],
    stock_models=['HestonWithVar'], drift=(0.0,), nb_stocks=(1000,), use_payoff_as_input=(True, False))
table_Dim_HestonV_MaxCallr0_5 = _DimensionTable(
    algos= ['RFQI', 'RLSM', 'DOS', 'EOP'],
    stock_models=['HestonWithVar'], drift=(0.0,), nb_stocks=(1000,), use_payoff_as_input=(True, False))
#   -- tables with basis functions
table_Dim_HestonV_MaxCallr0_bf = _SmallDimensionTable(
    stock_models=['HestonWithVar'], drift=(0.0,), nb_stocks=(5, 10, 50,),
    use_payoff_as_input=(True, False))
#   -- true price
table_spots_Dim_HestonV_MaxCallr0_ref = _DimensionTable(
    drift=(0.0,), algos=["EOP"],
    stock_models=['HestonWithVar'])
#   -- tables to generate output tables
table_Dim_HestonV_MaxCallr0_gt1 = _DimensionTable(
    stock_models=['HestonWithVar'], algos=algos, ridge_coeff=[1, np.nan, None],
    drift=(0.0,), use_payoff_as_input=(True, False))


# RoughHeston model
table_Dim_RoughHeston_MaxCallr0 = _DimensionTable(
    nb_stocks=[5, 10, 50, 100], train_ITM_only=[False,],
    stock_models=['RoughHeston'], dividends=[0.1], drift=(0.05,), hurst=(0.05,),
    use_payoff_as_input=(False, True))
table_Dim_RoughHeston_MaxCallr0_do = _DimensionTable(
    algos=['DOS',], dividends=[0.1], drift=(0.05,), hurst=(0.05,),
    nb_stocks=[5, 10, 50, 100],
    use_payoff_as_input=(False, True), train_ITM_only=[False,],
    stock_models=['RoughHeston'])
table_Dim_RoughHeston_MaxCallr0_dopath = _DimensionTable(
    algos=['pathDOS',], dividends=[0.1], drift=(0.05,), hurst=(0.05,),
    nb_stocks=[5, 10, 50, 100], use_path=[True],
    use_payoff_as_input=(False, True), train_ITM_only=[False,],
    stock_models=['RoughHeston'])
table_Dim_RoughHeston_MaxCallr0_bf = _SmallDimensionTable(
    stock_models=['RoughHeston'], dividends=[0.1], drift=(0.05,), hurst=(0.05,),
    use_payoff_as_input=(False, True), nb_stocks=[5, 10, 50, 100],
    train_ITM_only=[False,],)
table_Dim_RoughHeston_MaxCallr0_RRLSM = _DimensionTable(
    algos=['RRLSM', 'RRFQI'], train_ITM_only=[False], nb_stocks=[5, 10, 50, 100],
    factors=[[0.0008, 0.11], [0.0001, 0.3]], hurst=(0.05,),
    stock_models=['RoughHeston'], dividends=[0.1], drift=(0.05,),
    use_payoff_as_input=(False, True))
table_Dim_RoughHeston_MaxCallr0_gt1 = _DimensionTable(
    train_ITM_only=[False,], nb_stocks=[5, 10, 50, 100],
    factors=[str((1., 1., 1.)), str([0.0001, 0.3])],
    stock_models=['RoughHeston'], hurst=(0.05,),
    algos=['NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS', 'RRLSM', 'pathDOS'],
    ridge_coeff=[1, np.nan, None],
    dividends=[0.1], drift=(0.05,), use_payoff_as_input=(False, True))

# RoughHestonWithVar model
table_Dim_RoughHestonV_MaxCall = _DimensionTable(
    algos=['NLSM', 'RFQI', 'RLSM', 'DOS'],
    nb_stocks=[5, 10, 50, 100], train_ITM_only=[False,],
    stock_models=['RoughHestonWithVar'], dividends=[0.1], drift=(0.05,),
    hurst=(0.05,), use_payoff_as_input=(False, True))
table_Dim_RoughHestonV_MaxCall_dopath = _DimensionTable(
    algos=['pathDOS',], dividends=[0.1], drift=(0.05,), hurst=(0.05,),
    nb_stocks=[5, 10, 50, 100], use_path=[True],
    use_payoff_as_input=(False, True), train_ITM_only=[False,],
    stock_models=['RoughHestonWithVar'])
table_Dim_RoughHestonV_MaxCall_bf = _SmallDimensionTable(
    stock_models=['RoughHestonWithVar'], dividends=[0.1], drift=(0.05,),
    hurst=(0.05,), use_payoff_as_input=(False, True), nb_stocks=[5, 10, 50,],
    train_ITM_only=[False,],)
table_Dim_RoughHestonV_MaxCall_RRLSM = _DimensionTable(
    algos=['RRLSM', 'RRFQI'], train_ITM_only=[False], nb_stocks=[5, 10, 50, 100],
    factors=[[0.0008, 0.11],], hurst=(0.05,),
    stock_models=['RoughHestonWithVar'], dividends=[0.1], drift=(0.05,),
    use_payoff_as_input=(False, True))
table_Dim_RoughHestonV_MaxCall_gt1 = _DimensionTable(
    train_ITM_only=[False,], nb_stocks=[5, 10, 50, 100],
    stock_models=['RoughHestonWithVar'], hurst=(0.05,),
    algos=['NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS', 'RRLSM', 'pathDOS'],
    ridge_coeff=[1, np.nan, None], factors=["(1.0, 1.0, 1.0)", "[0.0008, 0.11]"],
    dividends=[0.1], drift=(0.05,), use_payoff_as_input=(False, True))


# GeoPut
table_smallDim_BS_GeoPut = _VerySmallDimensionTable(
    payoffs=['GeometricPut'],
    nb_stocks=[100],
    algos=['RFQI',  ],
    stock_models=['Heston', ],
    use_payoff_as_input=(True, False))
table_smallDim_BS_GeoPut_BS = _VerySmallDimensionTable(
    payoffs=['GeometricPut'],
    nb_stocks=[5,10,20,50,100,],
    algos=['LSM', 'FQI', ],
    stock_models=['Heston', 'BlackScholes'],
    use_payoff_as_input=(True, False))
#   -- true price GeoPut
#      ATTENTION: after running, the payoff, nb_dates, nb_stocks, dividends,
#      volatility, need to
#      be changed in the metric file to get the correct tables
table_smallDim_BS_GeoPut_ref1 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[1], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
table_smallDim_BS_GeoPut_ref2 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[5], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
table_smallDim_BS_GeoPut_ref3 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[10], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
table_smallDim_BS_GeoPut_ref4 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[20], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
table_smallDim_BS_GeoPut_ref5 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[50], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
table_smallDim_BS_GeoPut_ref6 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[100], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
table_smallDim_BS_GeoPut_ref7 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[500], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
table_smallDim_BS_GeoPut_ref8 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[1000], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
table_smallDim_BS_GeoPut_ref9 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[2000], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
# -- overview table
table_GeoPut_payoffs_gt1 = _DimensionTable(
    payoffs=['GeometricPut'], algos=algos,
    stock_models=['BlackScholes', 'Heston', ],
    nb_stocks=[5, 10, 20, 50, 100,],
    nb_dates=[10, 10000],
    dividends=[0.0,],
    volatilities=[0.2,],
    drift=(0.02,), use_payoff_as_input=(True, False))


# BasketCall BS
table_Dim_BS_BasktCallr0 = _DimensionTable(
    payoffs=['BasketCall'], algos=('NLSM', 'RFQI', 'RLSM', 'DOS'),
    drift=(0.0,), use_payoff_as_input=(True, False))
table_Dim_BS_BasktCallr0_bf = _SmallDimensionTable(
    payoffs=['BasketCall'], drift=(0.0,), use_payoff_as_input=(True, False))
#  -- true price
table_spots_Dim_BasktCallr0_ref = _DimensionTable(
    drift=(0.0,), algos=["EOP"], payoffs=['BasketCall'])
#  -- overview tables
table_BasketCall_payoffsr0_gt = _DimensionTable(
    payoffs=['BasketCall',], algos=algos,
    stock_models=['BlackScholes',],
    nb_stocks=[5, 10, 20, 50, 100, 500, 1000, 2000],
    drift=(0.0,), use_payoff_as_input=(False,))
table_BasketCall_payoffsr0_gt1 = _DimensionTable(
    payoffs=['BasketCall',], algos=algos,
    stock_models=['BlackScholes',],
    nb_stocks=[5, 10, 20, 50, 100, 500, 1000, 2000],
    drift=(0.0,), use_payoff_as_input=(True, False))


# other payoffs HestonWithVar
table_smallDim_HestonV_GeoPut = _VerySmallDimensionTable(
    payoffs=['GeometricPut'],
    algos=['NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS', ],
    stock_models=['HestonWithVar',],
    use_payoff_as_input=(True, False))
table_GeoPut_HestonV_payoffs_gt = _DimensionTable(
    payoffs=['GeometricPut'], algos=algos,
    stock_models=['HestonWithVar', ],
    nb_stocks=[5, 10, 20,],
    dividends=[0.0,],
    volatilities=[0.2,],
    nb_dates=[10],
    drift=(0.02,), use_payoff_as_input=(False,))
table_GeoPut_HestonV_payoffs_gt1 = _DimensionTable(
    payoffs=['GeometricPut'], algos=algos,
    stock_models=['HestonWithVar', ],
    nb_stocks=[5, 10, 20,],
    dividends=[0.0,],
    volatilities=[0.2,],
    nb_dates=[10],
    drift=(0.02,), use_payoff_as_input=(False,True))


# many dates tables
algos = ['NLSM', 'RFQI', 'RLSM', 'LSM', 'DOS']
table_manyDates_BS_MaxCallr0_1 = _VerySmallDimensionTable(
    algos=algos, nb_stocks=[10, 50, ], nb_dates=[50, 100],
    drift=(0.0,), use_payoff_as_input=(True, False))
table_manyDates_BS_MaxCallr0_FQI = _VerySmallDimensionTable(
    algos=["FQI"], nb_stocks=[10, 50, ], nb_dates=[50, 100],
    drift=(0.0,), use_payoff_as_input=(True, False))
algos_ = ['NLSM', 'RFQI', 'RLSM', 'DOS']
table_manyDates_BS_MaxCallr0_2 = _VerySmallDimensionTable(
    algos=algos_, nb_stocks=[100, 500,], nb_dates=[50, 100],
    drift=(0.0,), use_payoff_as_input=(True, False))
table_manyDates_BS_MaxCallr0_ref = _VerySmallDimensionTable(
    algos= ['EOP'], nb_stocks=[10, 50, 100, 500,], nb_dates=[50, 100],
    drift=(0.0,), use_payoff_as_input=(False,))
table_manyDates_BS_MaxCallr0_gt = _VerySmallDimensionTable(
    algos=['NLSM', 'RFQI', 'RLSM', 'LSM', 'DOS', 'FQI', 'EOP'],
    nb_stocks=[10, 50, 100, 500,], nb_dates=[10, 50, 100],
    hidden_size=[20], ridge_coeff=[1, np.nan, None],
    drift=(0.0,), use_payoff_as_input=(True,))
table_manyDates_BS_MaxCallr0_gt1 = _VerySmallDimensionTable(
    algos=['NLSM', 'RFQI', 'RLSM', 'LSM', 'DOS', 'FQI', 'EOP'],
    nb_stocks=[10, 50, 100, 500,], nb_dates=[10, 50, 100],
    hidden_size=[20], ridge_coeff=[1, np.nan, None],
    drift=(0.0,), use_payoff_as_input=(True, False))
# -- many dates tables, BS with dividend
table_manyDates_BS_MaxCall_div_1 = _VerySmallDimensionTable(
    algos = ['NLSM', 'RLSM', 'LSM', 'DOS'],
    nb_stocks=[10, 50, ], nb_dates=[50, 100],
    dividends=[0.1], drift=(0.05,), use_payoff_as_input=(True,))
table_manyDates_BS_MaxCall_div_FQI = _VerySmallDimensionTable(
    algos=["FQI", "RFQI"], nb_stocks=[10, 50, ], nb_dates=[50, 100],
    dividends=[0.1], drift=(0.05,), use_payoff_as_input=(False,))
table_manyDates_BS_MaxCall_div_2 = _VerySmallDimensionTable(
    algos=['NLSM', 'RLSM', 'DOS'],
    nb_stocks=[100, 500,], nb_dates=[50, 100],
    dividends=[0.1], drift=(0.05,), use_payoff_as_input=(True, ))
table_manyDates_BS_MaxCall_div_FQI_2 = _VerySmallDimensionTable(
    algos=["RFQI"], nb_stocks=[100, 500, ], nb_dates=[50, 100],
    dividends=[0.1], drift=(0.05,), use_payoff_as_input=(False,))
table_manyDates_BS_MaxCall_div_gt1 = _VerySmallDimensionTable(
    algos=['NLSM', 'RFQI', 'RLSM', 'LSM', 'DOS', 'FQI',],
    nb_stocks=[10, 50, 100, 500,], nb_dates=[10, 50, 100],
    hidden_size=[20], ridge_coeff=[1, np.nan, None],
    dividends=[0.1], drift=(0.05,),
    use_payoff_as_input=(True, False))



# BS MinPut
table_spots_Dim_BS_MinPut = _DimensionTable(
    payoffs=["MinPut"],
    spots=[80, 100, 120], drift=(0.02,), use_payoff_as_input=(True, False))
table_spots_Dim_BS_MinPut_do = _DimensionTable(
    payoffs=["MinPut"],
    algos=['DOS',], drift=(0.02,), use_payoff_as_input=(True, False),
    spots=[80, 100, 120])
#   -- tables with basis functions
table_spots_Dim_BS_MinPut_bf = _SmallDimensionTable(
    payoffs=["MinPut"],
    spots=[80, 100, 120], drift=(0.02,), use_payoff_as_input=(True, False))
#   -- tables to generate output tables
algos = ['NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS']
table_spots_Dim_BS_MinPut_gt = _DimensionTable(
    payoffs=["MinPut"],
    spots=[80, 100, 120], algos=algos, ridge_coeff=[1, np.nan, None],
    drift=(0.02,), use_payoff_as_input=(False,))
table_spots_Dim_BS_MinPut_gt1 = _DimensionTable(
    payoffs=["MinPut"],
    spots=[80, 100, 120], algos=algos, ridge_coeff=[1, np.nan, None],
    drift=(0.02,), use_payoff_as_input=(True, False,))


# BS MaxCall with dividend
table_Dim_BS_MaxCall_div = _DimensionTable(
    payoffs=['MaxCall'], algos=('NLSM', 'RFQI', 'RLSM', 'DOS'),
    dividends=[0.1],
    drift=(0.05,), use_payoff_as_input=(True, False))
table_Dim_BS_MaxCall_div_bf = _SmallDimensionTable(
    dividends=[0.1],
    payoffs=['MaxCall'], drift=(0.05,), use_payoff_as_input=(True, False))
table_Dim_BS_MaxCall_div_gt = _DimensionTable(
    payoffs=['MaxCall',], algos=algos, dividends=[0.1],
    drift=(0.05,), use_payoff_as_input=(False,))
table_Dim_BS_MaxCall_div_gt1 = _DimensionTable(
    payoffs=['MaxCall',], algos=algos, dividends=[0.1],
    drift=(0.05,), use_payoff_as_input=(True, False))


# Heston MinPut
table_spots_Dim_Heston_MinPut = _DimensionTable(
    payoffs=["MinPut"], stock_models=['Heston',],
    algos=('NLSM', 'RFQI', 'RLSM', 'DOS'),
    spots=[100], drift=(0.02,), use_payoff_as_input=(True, False))
#   -- tables with basis functions
table_spots_Dim_Heston_MinPut_bf = _SmallDimensionTable(
    payoffs=["MinPut"], stock_models=['Heston',],
    spots=[100], drift=(0.02,), use_payoff_as_input=(True, False))
#   -- tables to generate output tables
algos = ['NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS']
table_spots_Dim_Heston_MinPut_gt = _DimensionTable(
    payoffs=["MinPut"], stock_models=['Heston',],
    spots=[100], algos=algos, ridge_coeff=[1, np.nan, None],
    drift=(0.02,), use_payoff_as_input=(False,))
table_spots_Dim_Heston_MinPut_gt1 = _DimensionTable(
    payoffs=["MinPut"], stock_models=['Heston',],
    spots=[100], algos=algos, ridge_coeff=[1, np.nan, None],
    drift=(0.02,), use_payoff_as_input=(True, False,))


# Heston MaxCall with dividend
table_Dim_Heston_MaxCall_div = _DimensionTable(
    payoffs=['MaxCall'], algos=('RFQI', ),
    nb_stocks=(1000,),
    dividends=[0.1], stock_models=['Heston',],
    drift=(0.05,), use_payoff_as_input=(True, False))
table_Dim_Heston_MaxCall_div_bf = _SmallDimensionTable(
    dividends=[0.1], stock_models=['Heston',],
    payoffs=['MaxCall'], drift=(0.05,), use_payoff_as_input=(True, False))
table_Dim_Heston_MaxCall_div_gt1 = _DimensionTable(
    stock_models=['Heston',],
    payoffs=['MaxCall',], algos=algos, dividends=[0.1],
    drift=(0.05,), use_payoff_as_input=(True, False))


# HestonV MinPut
table_spots_Dim_HestonV_MinPut = _DimensionTable(
    payoffs=["MinPut"], stock_models=['HestonWithVar',],
    algos=('NLSM', 'RFQI', 'RLSM', 'DOS'),
    spots=[100], drift=(0.02,), use_payoff_as_input=(True, False))
#   -- tables with basis functions
table_spots_Dim_HestonV_MinPut_bf = _SmallDimensionTable(
    payoffs=["MinPut"], stock_models=['HestonWithVar',], nb_stocks=(5, 10, 50,),
    spots=[100], drift=(0.02,), use_payoff_as_input=(True, False))
#   -- tables to generate output tables
table_spots_Dim_HestonV_MinPut_gt1 = _DimensionTable(
    payoffs=["MinPut"], stock_models=['HestonWithVar',],
    spots=[100], algos=algos, ridge_coeff=[1, np.nan, None],
    drift=(0.02,), use_payoff_as_input=(True, False,))


# HestonV MaxCall with dividend
table_Dim_HestonV_MaxCall_div = _DimensionTable(
    payoffs=['MaxCall'], algos=('NLSM', 'RFQI', 'RLSM', 'DOS'),
    dividends=[0.1], stock_models=['HestonWithVar',],
    drift=(0.05,), use_payoff_as_input=(True, False))
table_Dim_HestonV_MaxCall_div_bf = _SmallDimensionTable(
    dividends=[0.1], stock_models=['HestonWithVar',], nb_stocks=(5, 10, 50,),
    payoffs=['MaxCall'], drift=(0.05,), use_payoff_as_input=(True, False))
table_Dim_HestonV_MaxCall_div_gt1 = _DimensionTable(
    stock_models=['HestonWithVar',],
    payoffs=['MaxCall',], algos=algos, dividends=[0.1],
    drift=(0.05,), use_payoff_as_input=(True, False))




'''
Empirical convergence study
'''

@dataclass
class _DefaultPlotNbPaths(_DefaultConfig):
  nb_runs: int = 20
  nb_stocks = [5, ]
  maturities: Iterable[int] = (1,)
  representations: Iterable[str] = ("ConvergenceStudy",)


table_conv_study_Heston_LND = _DefaultPlotNbPaths(
    nb_paths=list(200 * 2**np.array(range(8))),
    hidden_size=(10,50,100,),
    algos=("RLSM", ),
    stock_models=['Heston'],
)
table_conv_study_BS_LND = _DefaultPlotNbPaths(
    nb_paths=list(200 * 2**np.array(range(8))),
    hidden_size=(10,50,100,),
    algos=("RLSM", ),
    stock_models=['BlackScholes'],
)
table_conv_study_Heston_FQIR = _DefaultPlotNbPaths(
    nb_paths=list(200 * 2**np.array(range(8))),
    hidden_size=(5, 10, 50, 100),
    algos=("RFQI",),
    stock_models=['Heston'],
)
table_conv_study_BS_FQIR = _DefaultPlotNbPaths(
    nb_paths=list(200 * 2**np.array(range(8))),
    hidden_size=(5, 10, 50, 100),
    algos=("RFQI",),
    stock_models=['BlackScholes'],
)


'''
Tests for Sensitivity to Randomness of hidden layers 
'''
# ------------- Randomness Comparison (fixed paths)
SensRand_greeks_table1 = _DimensionTable(
    nb_runs=10, nb_epochs=[10],
    payoffs=["MaxCall"], volatilities=[0.2], drift=[0.02],
    strikes=[100], spots=[100], nb_dates=[10],
    hidden_size=[20], use_payoff_as_input=(False,), train_ITM_only=[True],
    algos=["RLSMSoftplus", "RLSMSoftplusReinit"], nb_stocks=[1],
    nb_paths=[100000])
SensRand_greeks_table1_1 = _DimensionTable(
    nb_runs=10, nb_epochs=[10, 30, 50],
    payoffs=["MaxCall"], volatilities=[0.2], drift=[0.02],
    strikes=[100], spots=[100], nb_dates=[10],
    hidden_size=[20], use_payoff_as_input=(True,), train_ITM_only=[True],
    algos=["NLSM"], nb_stocks=[1], nb_paths=[100000])



'''
Test for the FBM case of DOS
'''
# RNNLeastSquares
hurst = list(np.linspace(0, 1, 21))
hurst[0] = 0.01

table_RNN_DOS = _DefaultConfig(
    payoffs=['Identity'], nb_stocks=[1], spots=[0], nb_epochs=[30],
    hurst=hurst, train_ITM_only=[False],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'NLSM',
        'DOS',
        'RLSM',
        'RFQI',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
table_RNN_DOS_PD = _DefaultConfig(
    payoffs=['Identity'], nb_stocks=[1], spots=[0], nb_epochs=[30],
    hurst=hurst, train_ITM_only=[False],
    stock_models=['FractionalBrownianMotionPathDep'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'DOS',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
table_RNN_DOS_bf = _DefaultConfig(
    payoffs=['Identity'], nb_stocks=[1], spots=[0], nb_epochs=[30],
    hurst=hurst, train_ITM_only=[False],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'LSM',
        'FQI'
    ], nb_runs=10,
    representations=['TablePriceDuration']
)

factors0 = []
for a in [0.0001]:
    for b in [0.3]:
        factors0 += [[a,b]]
table_RNN_DOS_randRNN = _DefaultConfig(
    payoffs=['Identity'], nb_stocks=[1], spots=[0], nb_epochs=[30],
    hurst=hurst, train_ITM_only=[False],
    factors=factors0,
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'RRLSM',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
table_RNN_DOS_FQIR_PD = _DefaultConfig(
    payoffs=['Identity'], nb_stocks=[1], spots=[0], nb_epochs=[30],
    hurst=hurst, train_ITM_only=[False],
    stock_models=['FractionalBrownianMotionPathDep'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'RFQI',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
table_RNN_DOS_FQIRRNN = _DefaultConfig(
    payoffs=['Identity'], nb_stocks=[1], spots=[0], nb_epochs=[30],
    hurst=hurst, train_ITM_only=[False],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'RRFQI'
    ], nb_runs=10,
    representations=['TablePriceDuration']
)




'''
higher dimensional FBM
'''
table_highdim_hurst0 = _DefaultConfig(
    payoffs=['Identity'],
    nb_stocks=[1],
    spots=[0], nb_epochs=[30],
    hurst=[0.05], train_ITM_only=[False],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'DOS',
        'RLSM',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
table_highdim_hurst_PD0 = _DefaultConfig(
    payoffs=['Identity'],
    nb_stocks=[1],
    spots=[0], nb_epochs=[30],
    hurst=[0.05], train_ITM_only=[False],
    use_path=[True],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'DOS',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
factors = []
for a in [0.0008,]:
    for b in [0.11,]:
        factors += [[a,b]]
table_highdim_hurst_RNN0 = _DefaultConfig(
    payoffs=['Identity'],
    nb_stocks=[1],
    spots=[0], nb_epochs=[30],
    hurst=[0.05], train_ITM_only=[False],
    factors=factors,
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'RRLSM',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)

hurst1 = [0.05]
table_highdim_hurst = _DefaultConfig(
    payoffs=['Max', 'Mean'],
    nb_stocks=[5, 10, ],
    spots=[0], nb_epochs=[30],
    hurst=hurst1, train_ITM_only=[False],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    use_payoff_as_input=[True],
    algos=[
        'DOS',
        'RLSM',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
table_highdim_hurst_PD = _DefaultConfig(
    payoffs=['Max', 'Mean'],
    nb_stocks=[5, 10,],
    spots=[0], nb_epochs=[30],
    hurst=hurst1, train_ITM_only=[False],
    use_path=[True],
    use_payoff_as_input=[True],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'pathDOS',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
factors = []
for a in [0.0008,]:
    for b in [0.11,]:
        factors += [[a,b]]
table_highdim_hurst_RNN = _DefaultConfig(
    payoffs=['Max', 'Mean'],
    nb_stocks=[5, 10,],
    spots=[0], nb_epochs=[30],
    hurst=hurst1, train_ITM_only=[False],
    factors=factors,
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    use_payoff_as_input=[True],
    algos=[
        'RRLSM',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)

factors1 = [str((1.,1.,1.))] + [str(x) for x in factors+factors0]
table_highdim_hurst_gt = _DefaultConfig(
    payoffs=['Identity', 'Max', 'Mean'],
    nb_stocks=[1, 5, 10, ],
    spots=[0], nb_epochs=[30],
    hurst=[0.05,], train_ITM_only=[False],
    stock_models=['FractionalBrownianMotion',],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100], factors=factors1,
    use_payoff_as_input=[True, False],
    algos=[
        'DOS',
        'pathDOS',
        'RLSM',
        'RRLSM',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)


'''
test with Ridge regression
'''
table_Ridge_MaxCall = _SmallDimensionTable(
    spots=[100], algos=["LSMRidge", "RLSMRidge"], ridge_coeff=[1., 0.5, 2.],
    representations=['TablePriceDuration'],
)


'''
test with other basis functions
'''
table_OtherBasis_MaxCall = _SmallDimensionTable(
    spots=[100], algos=[
        "LSMLaguerre", "LSM", "FQILaguerre", "FQI", ],
    nb_stocks=(5, 10, 50,),
    nb_runs=10,
    representations=['TablePriceDuration'],
)


'''
greeks computations
'''
algos = ["NLSM", "DOS", "RLSMSoftplus", "RFQI", "RFQISoftplus"]
test_table_greeks_1 = _DimensionTable(
    nb_runs=1, nb_epochs=[20],
    payoffs=["MinPut"], volatilities=[0.2], drift=[0.06],
    strikes=[36,], spots=[40], nb_dates=[10],
    hidden_size=[100], use_payoff_as_input=(True, False,), train_ITM_only=[True],
    algos=["RLSMSoftplus"], nb_stocks=[1], nb_paths=[100000])

algos = ["FQI", "RFQI"] +\
        ["RLSMSoftplus","RLSMElu", "RLSMSilu","RLSMGelu","RLSMTanh"]+\
        ["RLSM", "LSM", "NLSM", "DOS"]
table_greeks_1 = _DimensionTable(
    nb_runs=10, nb_epochs=[10],
    payoffs=["MinPut"], volatilities=[0.2], drift=[0.06],
    strikes=[36,40, 44], spots=[40], nb_dates=[10],
    hidden_size=[10], use_payoff_as_input=(True,), train_ITM_only=[True],
    algos=algos, nb_stocks=[1], nb_paths=[100000])
algos = ["NLSM", "DOS"]
table_greeks_1_2 = _DimensionTable(
    nb_runs=10, nb_epochs=[10],
    payoffs=["MinPut"], volatilities=[0.2], drift=[0.06],
    strikes=[36,40,44], spots=[40], nb_dates=[10],
    hidden_size=[10,], use_payoff_as_input=(True,), train_ITM_only=[True],
    algos=algos, nb_stocks=[1], nb_paths=[100000])

table_greeks_binomial = _DimensionTable(
    nb_runs=1, algos=["B"],
    payoffs=["Put1Dim"], volatilities=[0.2], drift=[0.06],
    strikes=[36,40,44], spots=[40], nb_dates=[10000, 50000],
    nb_stocks=[1])


spots = np.linspace(20, 60, 41)
table_greeks_plots = _DimensionTable(
    nb_runs=5, nb_epochs=[10],
    payoffs=["MinPut"],
    volatilities=[0.1, 0.2, 0.3, 0.4],
    maturities=[1, 0.5, 2, 4, 8],
    drift=[0.06],
    strikes=[40],
    spots=spots.tolist(), nb_dates=[10],
    hidden_size=[10], use_payoff_as_input=(True,), train_ITM_only=[True],
    algos=["RLSMSoftplus"], nb_stocks=[1], nb_paths=[100000])
table_greeks_plots_binomial = _DimensionTable(
    nb_runs=1,
    payoffs=["Put1Dim"],
    volatilities=[0.1, 0.2, 0.3, 0.4],
    maturities=[1, 0.5, 2, 4, 8],
    drift=[0.06],
    strikes=[40],
    spots=spots.tolist(), nb_dates=[10000],
    hidden_size=[10],
    algos=["B"], nb_stocks=[1],)


'''
upper bound computations
'''
table_price_lower_upper_1 = _DimensionTable(
    nb_runs=10, nb_epochs=[10],
    payoffs=["MaxCall"], volatilities=[0.2], drift=[0.05], dividends=[0.1],
    strikes=[100,], spots=[90, 100, 110], nb_dates=[9], maturities=[3],
    hidden_size=[100], use_payoff_as_input=(True,), train_ITM_only=[False],
    algos=["RLSMSoftplus",],
    nb_stocks=[2, 3, 5, 10, 20,],
    nb_paths=[100000])
table_price_lower_upper_1_1 = _DimensionTable(
    nb_runs=10, nb_epochs=[10],
    payoffs=["MaxCall"], volatilities=[0.2], drift=[0.05], dividends=[0.1],
    strikes=[100,], spots=[90, 100, 110], nb_dates=[9], maturities=[3],
    hidden_size=[-5], use_payoff_as_input=(True,), train_ITM_only=[False],
    algos=["RLSMSoftplus",], nb_stocks=[2, 3, 5, 10, 20, 30, 50, 100, 200, 500],
    nb_paths=[20000])

# Add this BEFORE the test_table definitions at the end of configs.py

table_article_d5_d10 = _DimensionTable(
    spots=[80, 100, 120],
    drift=(0.0,),
    algos=['NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS', 'EOP'],
    nb_stocks=[2,4],  # Only d=5 and d=10
    use_payoff_as_input=(True, False),
    nb_runs=3,
)
# ==============================================================================
test_table = _SmallDimensionTable(
    spots=[10], strikes=[10],
    algos=[
        'NLSM', 'LSM', 'DOS', 'FQI', 'RFQI', 'RLSM',
        "LSMLaguerre", "FQILaguerre", "LSMRidge", "RLSMRidge", "RLSMTanh",
        "FQIRidge", "RFQIRidge",
        "RRLSM", "RRLSMmix", "RFQITanh", "RRFQI",
        "LSPI",
        'FQIDeg1', 'LSMDeg1',
    ],
    nb_stocks=(5,), nb_dates=(5,), nb_paths=(100,),
    use_payoff_as_input=(True, False),
    nb_runs=1, factors=((0.001,0.001,0.001),),
    representations=['TablePriceDuration'],
)


test_table2 = _SmallDimensionTable(
    spots=[10], strikes=[10],
    algos=[
        'NLSM', 'LSM', 'DOS', 'FQI', 'RFQI', 'RLSM',
        "LSMLaguerre", "FQILaguerre", "LSMRidge", "RLSMRidge", "RLSMTanh",
        "RRLSM", "RRLSMmix", "RFQITanh", "RRFQI",
        "LSPI",
        'FQIDeg1', 'LSMDeg1',
        'pathDOS'
    ],
    stock_models=["Heston", "RoughHeston",
                  "HestonWithVar", "RoughHestonWithVar"],
    hurst=[0.25],
    nb_stocks=(5,), nb_dates=(5,), nb_paths=(100,),
    use_payoff_as_input=(True, False),
    nb_runs=1, factors=((0.001,0.001,0.001),),
    representations=['TablePriceDuration'],
)

# ==============================================================================
# QUICK TESTS FOR NEW 408 PAYOFF SYSTEM
# ==============================================================================

# Test 1: Base payoffs (no barriers) - Tests simple non-path-dependent options
quick_test_base_payoffs = _DefaultConfig(
    algos=['RFQI', 'RLSM'],  # Fast algorithms for non-path-dependent
    payoffs=[
        'BasketCall', 'BasketPut',           # Simple basket
        'GeometricCall', 'GeometricPut',     # Geometric
        'MaxCall', 'MinPut',                 # Max/Min
        'Call', 'Put',                       # Single asset
    ],
    nb_stocks=[2, 3],
    nb_paths=[2000],
    nb_dates=[5],
    nb_runs=2,
    strikes=[100],
    spots=[100],
    barriers=[100000],  # High barrier = effectively no barrier
    volatilities=[0.2],
    drift=[0.02],
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)

# Test 2: Barrier payoffs - Tests path-dependent barrier options
quick_test_barriers = _DefaultConfig(
    algos=['SRFQI', 'SRLSM'],  # Path-dependent algorithms
    payoffs=[
        # Single barriers
        'UO_BasketCall', 'DO_BasketPut',
        'UI_GeometricCall', 'DI_MaxCall',
        # Double barriers
        'UODO_Call', 'UIDI_Put',
        # Step barriers
        'StepB_BasketCall', 'DStepB_GeometricCall',
    ],
    nb_stocks=[2],
    nb_paths=[2000],
    nb_dates=[5],
    nb_runs=2,
    strikes=[100],
    spots=[100],
    barriers=[110],      # Single barrier
    barriers_up=[110],   # Upper barrier
    barriers_down=[90],  # Lower barrier
    step_param1=[-1],    # Step barrier lower bound
    step_param2=[1],     # Step barrier upper bound
    step_param3=[-1],    # Double step barrier lower bound
    step_param4=[1],     # Double step barrier upper bound
    volatilities=[0.2],
    drift=[0.02],
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)

# Test 3: Advanced payoffs with parameters - Tests rank, Asian, lookback
quick_test_advanced = _DefaultConfig(
    algos=['SRFQI', 'SRLSM'],  # Path-dependent algorithms
    payoffs=[
        # Rank-based (uses k parameter)
        'BestOfKCall', 'WorstOfKPut',
        # Asian path-dependent
        'AsianFixedStrikeCall', 'AsianFloatingStrikePut',
        # Lookback
        'LookbackFixedCall', 'LookbackFloatPut',
        # Range (path-dependent)
        'RangeCall', 'RangeCall_Single',
    ],
    nb_stocks=[3],
    nb_paths=[2000],
    nb_dates=[5],
    nb_runs=2,
    strikes=[100],
    spots=[100],
    barriers=[100000],  # High barrier = no barrier
    k=[2],              # Rank parameter (best-of-2, worst-of-2)
    volatilities=[0.2],
    drift=[0.02],
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)
# ============================================================================
# VALIDATION TESTS - Comprehensive parameter and convergence verification
# ============================================================================

# Test 1: Barrier Convergence - Verify UO/DO with extreme barriers → vanilla
validation_barrier_convergence = _DefaultConfig(
    algos=['RFQI', 'RLSM', 'SRFQI', 'SRLSM'],
    payoffs=[
        # Vanilla baselines
        'BasketCall', 'BasketPut', 'Call', 'Put',
        # Up-and-Out (should converge to vanilla when barrier is very high)
        'UO_BasketCall', 'UO_BasketPut', 'UO_Call', 'UO_Put',
        # Down-and-Out (should converge to vanilla when barrier is very low)
        'DO_BasketCall', 'DO_BasketPut', 'DO_Call', 'DO_Put',
    ],
    nb_stocks=[10],
    nb_paths=[5000],
    nb_dates=[10],
    nb_runs=5,
    strikes=[100],
    spots=[100],
    # Test multiple barrier levels: extreme (converge) vs moderate (different prices)
    barriers=[10000, 150, 50],  # 10000=vanilla, 150=high, 50=low
    volatilities=[0.2],
    drift=[0.05],
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)

# Test 2: K Sensitivity - Rank options with different k values
validation_k_sensitivity = _DefaultConfig(
    algos=['RFQI', 'RLSM'],
    payoffs=[
        'BestOfKCall', 'WorstOfKPut',
        'RankWeightedBasketCall', 'RankWeightedBasketPut',
    ],
    nb_stocks=[10],  # Out of 10 stocks
    nb_paths=[5000],
    nb_dates=[10],
    nb_runs=5,
    strikes=[100],
    spots=[100],
    barriers=[100000],
    # Test k = 2, 5, 8 out of 10 stocks
    # BestOfK should have HIGHER prices with larger k (more optionality)
    # WorstOfK should have LOWER prices with larger k (worst of more = worse)
    k=[2, 5, 8],
    volatilities=[0.2, 0.4],  # Low and high vol to see differences
    drift=[0.05],  # Match validation_large_basket drift
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)

# Test 4: Step Barrier Drift - Different random walk parameters
validation_step_barriers = _DefaultConfig(
    algos=['SRFQI', 'SRLSM'],
    payoffs=[
        # Single step barrier
        'StepB_BasketCall', 'StepB_BasketPut',
        'StepB_Call', 'StepB_Put',
        # Double step barrier
        'DStepB_BasketCall', 'DStepB_Call',
    ],
    nb_stocks=[10],
    nb_paths=[5000],
    nb_dates=[10],
    nb_runs=5,
    strikes=[100],
    spots=[100],
    barriers=[120],  # Starting barrier
    barriers_up=[120],
    barriers_down=[80],
    # Test different drift ranges for step barriers
    # Positive drift [0, 2]: barrier drifts UP → easier to hit for UO
    # Negative drift [-2, 0]: barrier drifts DOWN → easier to hit for DO
    # Symmetric [-2, 2]: no drift bias
    step_param1=[-2, -1, 0],
    step_param2=[0, 1, 2],
    step_param3=[-2],
    step_param4=[2],
    volatilities=[0.25],
    drift=[0.05],
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)

# Test 5: Large Basket - Test all payoff types with d=10
validation_large_basket = _DefaultConfig(
    algos=['RFQI', 'RLSM', 'SRFQI', 'SRLSM'],
    payoffs=[
        # Simple basket
        'BasketCall', 'BasketPut', 'GeometricCall', 'MaxCall', 'MinPut',
        # Asian
        'AsianFixedStrikeCall', 'AsianFloatingStrikePut',
        # Rank
        'BestOfKCall', 'WorstOfKPut',
        # Range/Dispersion
        'RangeCall', 'DispersionCall',
        # Barriers
        'UO_BasketCall', 'DO_BasketPut', 'UODO_BasketCall',
    ],
    nb_stocks=[10],
    nb_paths=[5000],
    nb_dates=[10],
    nb_runs=5,
    strikes=[100],
    spots=[100],
    barriers=[10000, 130],  # High and moderate
    barriers_up=[130],
    barriers_down=[70],
    k=[3, 7],  # Best/worst of 3 and 7 out of 10
    volatilities=[0.2, 0.4],  # Low and high vol
    drift=[0.05],
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)

# Test 6: UI/DI Near-Barrier Convergence
validation_in_barriers = _DefaultConfig(
    algos=['SRFQI', 'SRLSM'],
    payoffs=[
        # Vanilla baseline
        'BasketCall', 'BasketPut',
        # Up-and-In (should converge to vanilla when barrier is very LOW → always hit)
        'UI_BasketCall', 'UI_BasketPut',
        # Down-and-In (should converge to vanilla when barrier is very HIGH → always hit)
        'DI_BasketCall', 'DI_BasketPut',
    ],
    nb_stocks=[10],
    nb_paths=[5000],
    nb_dates=[10],
    nb_runs=5,
    strikes=[100],
    spots=[100],
    # UI with barrier=80 (low) → always hit → vanilla
    # DI with barrier=120 (high) → always hit → vanilla
    # UI with barrier=150 (high) → rarely hit → near zero
    # DI with barrier=50 (low) → rarely hit → near zero
    barriers=[80, 120, 150, 50],
    volatilities=[0.2],
    drift=[0.05],
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)

# Test 7: Lookback vs Asian vs Vanilla - Ordering verification
validation_payoff_ordering = _DefaultConfig(
    algos=['RFQI', 'RLSM', 'SRFQI', 'SRLSM'],  # Both standard and path-dependent algos
    payoffs=[
        # Ordering: Lookback > Asian > Vanilla (for calls)
        # NOTE: Call/BasketCall use RFQI/RLSM, Asian/Lookback use SRFQI/SRLSM
        'Call', 'AsianFixedStrikeCall', 'LookbackFixedCall',
        'BasketCall',
        # Floating strike variants
        'AsianFloatingStrikeCall', 'LookbackFloatCall',
    ],
    nb_stocks=[10],
    nb_paths=[5000],
    nb_dates=[10],
    nb_runs=5,
    strikes=[100],
    spots=[100],
    barriers=[100000],
    volatilities=[0.2, 0.4],  # Low and high vol to see differences
    drift=[0.05],
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)

# Video Testing: Simple config for testing video generation
video_testing = _DefaultConfig(
    algos=['RLSM'],
    nb_stocks=[2],
    nb_runs=1,  # FIX: nb_runs must be int, not list
    payoffs=['BasketPut'],
)

# Benchmark Algorithms: Test all benchmark algorithms
benchmark_algorithms = _DefaultConfig(
    algos=['RLSM', 'RFQI', 'LSM', 'FQI', 'NLSM', 'DOS'],
    nb_stocks=[1, 2],
    nb_paths=[10000],
    nb_dates=[20],
    nb_runs=3,  # FIX: nb_runs must be int, not list
    payoffs=['BasketCall', 'BasketPut'],
    nb_epochs=[20],
    hidden_size=[50],
)

# Test LSM only
test_lsm = _DefaultConfig(
    algos=['LSM'],
    nb_stocks=[1],
    nb_paths=[1000],
    nb_dates=[10],
    nb_runs=1,  # FIX: nb_runs must be int, not list
    payoffs=['BasketCall'],
    nb_epochs=[5],
    hidden_size=[20],
)

# ==============================================================================
# STORED PATHS TESTS
# ==============================================================================
# NOTE: First store paths using: python -m optimal_stopping.data.store_paths
# Then update the storage ID below and run experiments

# Example stored paths config - UPDATE THE STORAGE ID!
test_stored = _DefaultConfig(
    stock_models=['RealDataStored1763383103077'],  # Use stored ID
    nb_stocks=[3],    # Can use subset (≤50)
    nb_paths=[5000],          # Can use subset (≤100000)
    nb_dates=[252],            # Must match exactly
    maturities=[1.0],          # Must match exactly
    spots=[100],      # Will automatically rescale!
    payoffs=['BasketCall', 'MaxCall', 'MinPut'],
    algos=['RLSM', 'LSM'],
    nb_runs=2,
)

# ==============================================================================
# REAL DATA MODEL TESTS
# ==============================================================================
# NOTE: RealData requires yfinance: pip install yfinance

# Quick test: Minimal config for fast testing
test_real_data_quick = _DefaultConfig(
    algos=['RLSM'],
    stock_models=['RealData'],
    nb_stocks=[3],  # AAPL, MSFT, GOOGL
    nb_paths=[1000],
    nb_dates=[20],
    nb_runs=1,
    payoffs=['BasketCall'],
    spots=[100],
    strikes=[100],
    maturities=[0.25],
    drift=(None,),  # Use empirical drift from historical data
    volatilities=(None,),  # Use empirical volatility from historical data
    nb_epochs=[10],
    hidden_size=[30],
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)

# Test 1: Basic Real Data with FAANG stocks
test_real_data_basic = _DefaultConfig(
    algos=['RLSM', 'RFQI', 'LSM'],
    stock_models=['RealData'],
    nb_stocks=[5],  # AAPL, MSFT, GOOGL, AMZN, NVDA
    nb_paths=[10000],
    nb_dates=[252],  # 1 year of daily steps
    nb_runs=3,
    payoffs=['BasketCall', 'BasketPut', 'MaxCall', 'MinPut'],
    spots=[100],
    strikes=[100, 110],  # ATM and OTM
    maturities=[1.0],
    drift=(None,),  # Use empirical drift from historical data
    volatilities=(None,),  # Use empirical volatility from historical data
    nb_epochs=[30],
    hidden_size=[50],
    train_ITM_only=[True],
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)

# Test 2: Crisis vs Non-Crisis comparison
test_real_data_crisis = _DefaultConfig(
    algos=['RLSM', 'RFQI'],
    stock_models=['RealData'],
    nb_stocks=[3],
    nb_paths=[5000],
    nb_dates=[63],  # ~3 months
    nb_runs=5,
    payoffs=['BasketPut', 'MaxCall'],  # Puts more interesting during crises
    spots=[100],
    strikes=[100],
    maturities=[0.25],
    drift=(None,),  # Use empirical drift from historical data
    volatilities=(None,),  # Use empirical volatility from historical data
    nb_epochs=[20],
    hidden_size=[50],
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
    # Note: To actually test crisis vs non-crisis, you'd need to modify
    # the RealDataModel initialization in run_algo.py to pass exclude_crisis parameter
)

# Test 3: Real Data vs Black-Scholes comparison
# Note: drift/volatilities are used by BlackScholes and as overrides for RealData
# This makes the models comparable (same drift/vol parameters)
test_real_data_vs_bs = _DefaultConfig(
    algos=['RLSM', 'RFQI', 'SRFQI', 'SRLSM'],
    stock_models=['RealData', 'BlackScholes'],
    nb_stocks=[5, 10],
    nb_paths=[10000],
    nb_dates=[50, 100],
    nb_runs=5,
    payoffs=['BasketCall', 'AsianFixedStrikeCall', 'LookbackFixedCall'],
    spots=[100],
    strikes=[90, 100, 110],  # ITM, ATM, OTM
    maturities=[1.0],
    volatilities=[0.2, 0.3],
    drift=[0.05],
    nb_epochs=[30],
    hidden_size=[50],
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)

# Test 4: High-dimensional Real Data (many stocks)
test_real_data_multidim = _DefaultConfig(
    algos=['RLSM', 'RFQI', 'SRFQI', 'SRLSM'],
    stock_models=['RealData'],
    nb_stocks=[10, 25, 50],  # Test scaling
    nb_paths=[5000, 10000],
    nb_dates=[50],
    nb_runs=3,
    payoffs=['BasketCall', 'MaxCall', 'MinPut', 'AsianFixedStrikeCall'],
    spots=[100],
    strikes=[100],
    maturities=[1.0],
    drift=(None,),  # Use empirical drift from historical data
    volatilities=(None,),  # Use empirical volatility from historical data
    nb_epochs=[20],
    hidden_size=[50, 100],
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)

# Test 5: Real Data with exotic payoffs
test_real_data_exotics = _DefaultConfig(
    algos=['RLSM', 'RFQI', 'SRFQI', 'SRLSM'],
    stock_models=['RealData'],
    nb_stocks=[5],
    nb_paths=[10000],
    nb_dates=[100],
    nb_runs=3,
    payoffs=[
        # Basket options
        'BasketCall', 'BasketPut',
        # Best/worst of
        'MaxCall', 'MinPut',
        # Path-dependent
        'AsianFixedStrikeCall', 'LookbackFixedCall',
        # Barriers
        'UI_BasketCall', 'DI_BasketPut',
    ],
    spots=[100],
    strikes=[100],
    maturities=[1.0],
    drift=(None,),  # Use empirical drift from historical data
    volatilities=(None,),  # Use empirical volatility from historical data
    barriers=[80, 120],  # For barrier options
    nb_epochs=[30],
    hidden_size=[50],
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)

# Test 6: Real Data short maturity (high frequency)
test_real_data_short = _DefaultConfig(
    algos=['RLSM', 'RFQI'],
    stock_models=['RealData'],
    nb_stocks=[3],
    nb_paths=[10000],
    nb_dates=[21],  # 1 month, daily
    nb_runs=5,
    payoffs=['BasketCall', 'BasketPut', 'MaxCall'],
    spots=[100],
    strikes=[95, 100, 105],
    maturities=[1/12],  # 1 month
    drift=(None,),  # Use empirical drift from historical data
    volatilities=(None,),  # Use empirical volatility from historical data
    nb_epochs=[20],
    hidden_size=[30],
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)

# Test 7: Real Data algorithm comparison (all algorithms)
test_real_data_all_algos = _DefaultConfig(
    algos=['RLSM', 'RFQI', 'LSM', 'FQI', 'NLSM', 'DOS', 'SRFQI', 'SRLSM'],
    stock_models=['RealData'],
    nb_stocks=[5],
    nb_paths=[10000],
    nb_dates=[50],
    nb_runs=3,
    payoffs=['BasketCall', 'BasketPut'],
    spots=[100],
    strikes=[100],
    maturities=[0.5],
    drift=(None,),  # Use empirical drift from historical data
    volatilities=(None,),  # Use empirical volatility from historical data
    nb_epochs=[20, 30],
    hidden_size=[50],
    train_ITM_only=[True, False],  # Compare ITM-only vs all paths
    use_payoff_as_input=[True],
    representations=['TablePriceDuration'],
)
