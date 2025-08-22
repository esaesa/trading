# indicator_manager.py
import numpy as np
import pandas as pd
import ta

from ta.volatility import AverageTrueRange, BollingerBands
from numba import njit

class IndicatorManager:
    """
    Efficient manager for computing common technical indicators with minimal overhead.
    
    Methods:
        - compute_rsi(...)
        - compute_ema(...)
        - compute_atr(...)
        - compute_atr_percentage(...)
        - compute_coefficient_of_variation(...)
        
    Usage in a Backtesting.py strategy:
    
        def init(self):
            df = self.data.df  # the real DataFrame from self.data
            self.indicator_manager = IndicatorManager(df)
            
            rsi_series = self.indicator_manager.compute_rsi(window=14)
            self.rsi_dynamic = self.I(lambda _: rsi_series, self.data.Close, name="RSI")
            
            # and so on...
    """

    def __init__(self, data: pd.DataFrame):
        """
        :param data: A real pandas DataFrame with columns: 
                     'Open', 'High', 'Low', 'Close', [Volume optional],
                     and a DatetimeIndex.
                     
        We store it directly to avoid overhead from copying. 
        Please don't mutate it externally after passing it in!
        """
        self.data = data

        # Optionally store the computed series if you need to access them:
        self.rsi_series = None
        self.ema_series = None
        self.atr_series = None
        self.atr_pct_series = None
        self.cv_series = None

    def _resample_ohlc(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """
        Internal helper to resample OHLC data if interval is provided.
        Aggregates standard columns if present. 
        If you only need 'Close', consider a simpler approach.
        """
        if not interval:
            return df
        
        agg_dict = {}
        # Only aggregate columns that exist in df
        if 'Open' in df.columns:
            agg_dict['Open'] = 'first'
        if 'High' in df.columns:
            agg_dict['High'] = 'max'
        if 'Low' in df.columns:
            agg_dict['Low'] = 'min'
        if 'Close' in df.columns:
            agg_dict['Close'] = 'last'
        if 'Volume' in df.columns:
            agg_dict['Volume'] = 'sum'
        
        # Resample and drop rows that are all NaN
        resampled = df.resample(interval).agg(agg_dict).dropna(how='all')
        resampled.ffill(inplace=True)
        return resampled
    
    # Tiny helper to return a safe NaN series aligned to self.data.index
    def _nan_series_like_data(self) -> pd.Series:
        return pd.Series(np.nan, index=self.data.index, dtype=float)

    # ----------------------------------------------------------------
    # RSI (guarded)
    # ----------------------------------------------------------------
    def compute_rsi(self, window: int, resample_interval: str = None) -> pd.Series:
        df = self.data[['Open', 'High', 'Low', 'Close']]
        df_res = self._resample_ohlc(df, resample_interval)

        # Not enough bars → return NaNs aligned to original index
        if len(df_res) < max(2, int(window or 1)):
            return self._nan_series_like_data()

        rsi_calc = ta.momentum.RSIIndicator(
            close=df_res['Close'],
            window=window
        ).rsi()

        if resample_interval:
            rsi_calc = rsi_calc.reindex(self.data.index).ffill()

        self.rsi_series = rsi_calc
        return rsi_calc

    # ----------------------------------------------------------------
    # EMA (guarded)
    # ----------------------------------------------------------------
    def compute_ema(self, window: int, resample_interval: str = None) -> pd.Series:
        df = self.data[['Close']]
        df_res = self._resample_ohlc(df, resample_interval)

        if len(df_res) < max(2, int(window or 1)):
            return self._nan_series_like_data()

        ema_calc = ta.trend.EMAIndicator(
            close=df_res['Close'],
            window=window
        ).ema_indicator()

        if resample_interval:
            ema_calc = ema_calc.reindex(self.data.index).ffill()

        self.ema_series = ema_calc
        return ema_calc

    # ----------------------------------------------------------------
    # ATR (guarded)
    # ----------------------------------------------------------------
    def compute_atr(self, window: int, resample_interval: str = None) -> pd.Series:
        cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in self.data.columns]
        df = self.data[cols]
        df_res = self._resample_ohlc(df, resample_interval)

        # Need High/Low/Close present and enough rows
        needed = {'High', 'Low', 'Close'}
        if not needed.issubset(df_res.columns) or len(df_res) < max(2, int(window or 1)):
            return self._nan_series_like_data()

        try:
            atr_calc = AverageTrueRange(
                high=df_res['High'],
                low=df_res['Low'],
                close=df_res['Close'],
                window=window
            ).average_true_range()
        except Exception:
            return self._nan_series_like_data()

        if resample_interval:
            atr_calc = atr_calc.reindex(self.data.index).ffill()

        self.atr_series = atr_calc
        return atr_calc

    def compute_atr_percentage(self, window: int, resample_interval: str = None) -> pd.Series:
        atr_raw = self.compute_atr(window=window, resample_interval=resample_interval)
        if atr_raw.isna().all():
            self.atr_pct_series = self._nan_series_like_data()
            return self.atr_pct_series

        close = self.data['Close'].astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            atr_pct = (atr_raw / close) * 100.0
        self.atr_pct_series = atr_pct
        return atr_pct

    # ----------------------------------------------------------------
    # CV (guarded)
    # ----------------------------------------------------------------
    def compute_coefficient_of_variation(
        self,
        window: int,
        resample_interval: str = None,
        price_col: str = 'Close'
    ) -> pd.Series:
        df = self.data[[price_col]]
        df_res = self._resample_ohlc(df, resample_interval)

        if len(df_res) < max(2, int(window or 1)):
            return self._nan_series_like_data()

        roll_std = df_res[price_col].rolling(window=window, min_periods=window).std()
        roll_mean = df_res[price_col].rolling(window=window, min_periods=window).mean()
        cv_calc = roll_std / roll_mean

        if resample_interval:
            cv_calc = cv_calc.reindex(self.data.index).ffill()

        self.cv_series = cv_calc
        return cv_calc

    # ----------------------------------------------------------------
    # BBW (guarded)
    # ----------------------------------------------------------------
    def compute_bbw(self, window: int = 20, num_std: int = 2, resample_interval: str = None) -> pd.Series:
        df = self.data[['Close']]
        df_res = self._resample_ohlc(df, resample_interval)

        if len(df_res) < max(2, int(window or 1)):
            return self._nan_series_like_data()

        try:
            bb = BollingerBands(close=df_res['Close'], window=window, window_dev=num_std)
            upper_band = bb.bollinger_hband()
            lower_band = bb.bollinger_lband()
            middle_band = bb.bollinger_mavg()
            bbw = (upper_band - lower_band) / middle_band * 100.0
        except Exception:
            return self._nan_series_like_data()

        if resample_interval:
            bbw = bbw.reindex(self.data.index).ffill()

        self.bbw_series = bbw
        return bbw



    # ----------------------------------------------------------------
    # Laguerre RSI
    # ----------------------------------------------------------------
    def compute_laguerre_rsi(self, gamma: float = 0.5, resample_interval: str = None) -> pd.Series:
        """
        Compute Laguerre RSI over the Close prices.
        
        Math Model:
            L0[i] = (1 - gamma) * price[i] + gamma * L0[i-1]
            L1[i] = -gamma * L0[i] + L0[i-1] + gamma * L1[i-1]
            L2[i] = -gamma * L1[i] + L1[i-1] + gamma * L2[i-1]
            L3[i] = -gamma * L2[i] + L2[i-1] + gamma * L3[i-1]
            
            CU = max(L0 - L1, 0) + max(L1 - L2, 0) + max(L2 - L3, 0)
            CD = max(L1 - L0, 0) + max(L2 - L1, 0) + max(L3 - L2, 0)
            Laguerre RSI = CU / (CU + CD)
        """
        df = self.data[['Close']]
        df_res = self._resample_ohlc(df, resample_interval)
        prices = df_res['Close'].values

        # Call the static JIT-compiled Laguerre RSI function
        lag_rsi = IndicatorManager.laguerre_rsi_fast(prices, gamma)
        if resample_interval:
            lag_rsi = pd.Series(lag_rsi, index=df_res.index).reindex(self.data.index).ffill()
        else:
            lag_rsi = pd.Series(lag_rsi, index=self.data.index)
        self.laguerre_rsi_series = lag_rsi
        return lag_rsi

    @staticmethod
    @njit
    def laguerre_rsi_fast(prices, gamma):
        n = len(prices)
        L0 = np.empty(n)
        L1 = np.empty(n)
        L2 = np.empty(n)
        L3 = np.empty(n)
        RSI = np.empty(n)

        # Initialize the first elements
        L0[0] = prices[0]
        L1[0] = prices[0]
        L2[0] = prices[0]
        L3[0] = prices[0]
        RSI[0] = 0.0

        for i in range(1, n):
            L0[i] = (1 - gamma) * prices[i] + gamma * L0[i - 1]
            L1[i] = -gamma * L0[i] + L0[i - 1] + gamma * L1[i - 1]
            L2[i] = -gamma * L1[i] + L1[i - 1] + gamma * L2[i - 1]
            L3[i] = -gamma * L2[i] + L2[i - 1] + gamma * L3[i - 1]

            cu = 0.0
            cd = 0.0

            diff0 = L0[i] - L1[i]
            diff1 = L1[i] - L2[i]
            diff2 = L2[i] - L3[i]

            if diff0 >= 0:
                cu += diff0
            else:
                cd += -diff0

            if diff1 >= 0:
                cu += diff1
            else:
                cd += -diff1

            if diff2 >= 0:
                cu += diff2
            else:
                cd += -diff2

            RSI[i] = cu / (cu + cd) if (cu + cd) > 0 else 0.0
        return RSI
    
    # ----------------------------------------------------------------
    # Bollinger Band Width (BBW)
    # ----------------------------------------------------------------
    def compute_bbw(self, window: int = 20, num_std: int = 2, resample_interval: str = None) -> pd.Series:
        """
        Compute Bollinger Band Width (BBW) as (Upper Band - Lower Band) / Middle Band.
        
        :param window: Rolling period for the SMA and standard deviation. Default 20.
        :param num_std: Number of standard deviations for the bands. Default 2.
        :param resample_interval: Optional resampling interval (e.g., '1h', '1D').
        :return: A pandas Series of BBW values, aligned to self.data's index.
        """
        df = self.data[['Close']]
        df_res = self._resample_ohlc(df, resample_interval)

        # Compute Bollinger Bands using TA library
        bb = BollingerBands(
            close=df_res['Close'], 
            window=window, 
            window_dev=num_std
        )
        upper_band = bb.bollinger_hband()
        lower_band = bb.bollinger_lband()
        middle_band = bb.bollinger_mavg()

        bbw = (upper_band - lower_band) / middle_band *100

        # Align to original index if resampled
        if resample_interval:
            bbw = bbw.reindex(self.data.index).ffill()

        self.bbw_series = bbw
        return bbw