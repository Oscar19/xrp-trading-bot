
"""
BACKTEST ESTRATEGIA RSI + ESTOCÁSTICO - XRP/USDT
=================================================
Código corregido sin lookahead bias, con comisiones reales,
gestión de riesgo y métricas profesionales.

INSTALACIÓN:
    pip install ccxt ta pandas numpy matplotlib

USO:
    python xrp_backtest.py
"""

import pandas as pd
import numpy as np
import ccxt
import ta
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================
# CONFIGURACIÓN
# ============================

CONFIG = {
    'symbol': 'XRP/USDT',
    'timeframe': '5m',
    'limit': 1000,

    # Indicadores
    'rsi_period': 14,
    'stoch_k': 14,
    'stoch_d': 3,
    'ema_trend': 200,

    # Gestión de riesgo
    'initial_balance': 1000,
    'risk_per_trade': 0.02,      # 2% del balance por operación
    'stop_loss': 0.015,          # 1.5% stop loss
    'take_profit': 0.03,         # 3% take profit (ratio 1:2)

    # Costes reales Binance Spot
    'fee': 0.001,                # 0.1%
    'slippage': 0.0005,          # 0.05% slippage estimado
}

# ============================
# CLASES DE DATOS
# ============================

@dataclass
class Trade:
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    side: str
    size: float
    margin: float
    pnl: float
    pnl_pct: float
    reason: str
    fees: float

@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: pd.Series
    metrics: Dict
    config: Dict

# ============================
# 1. OBTENER DATOS (BINANCE REAL)
# ============================

def get_data(symbol: str = 'XRP/USDT', timeframe: str = '5m', limit: int = 1000) -> pd.DataFrame:
    """Obtiene datos OHLCV de Binance con cache local."""

    os.makedirs('cache', exist_ok=True)
    cache_file = f"cache/{symbol.replace('/', '_')}_{timeframe}_{limit}.csv"

    # Verificar cache (válido por 1 hora)
    if os.path.exists(cache_file):
        cache_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).seconds
        if cache_age < 3600:
            print(f"📁 Usando datos en cache")
            df = pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)
            return df

    print(f"🌐 Descargando {symbol} {timeframe} desde Binance...")
    exchange = ccxt.binance({'enableRateLimit': True})

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        print(f"❌ Error descargando datos: {e}")
        raise

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    df.to_csv(cache_file)
    print(f"✅ Datos descargados: {len(df)} velas | {df.index[0]} → {df.index[-1]}")

    return df

# ============================
# 2. INDICADORES
# ============================

def add_indicators(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Añade RSI, Estocástico y EMA de tendencia."""

    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=config['rsi_period']).rsi()

    # Estocástico
    stoch = ta.momentum.StochasticOscillator(
        high=df['high'], low=df['low'], close=df['close'],
        window=config['stoch_k'], smooth_window=config['stoch_d']
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # EMA filtro de tendencia
    df['ema_trend'] = ta.trend.EMAIndicator(df['close'], window=config['ema_trend']).ema_indicator()

    return df

# ============================
# 3. SEÑALES (SIN LOOKAHEAD BIAS)
# ============================

def generate_signals(df: pd.DataFrame, config: Dict) -> pd.Series:
    """
    Genera señales SIN usar información futura.
    La señal para la vela i se basa en datos de velas CERRADAS anteriores.
    """
    signals = pd.Series(index=df.index, dtype=object)
    signals[:] = None

    for i in range(2, len(df)):
        # Datos disponibles al INICIO de la vela i
        prev = df.iloc[i-1]      # Vela anterior (cerrada)
        prev_prev = df.iloc[i-2] # Hace 2 velas

        if pd.isna(prev_prev['rsi']) or pd.isna(prev['stoch_k']) or pd.isna(prev['stoch_d']):
            continue

        # CONDICIÓN DE COMPRA:
        # 1. RSI < 30 (sobrevendido)
        # 2. Cruce estocástico alcista
        # 3. Precio > EMA200 (tendencia alcista)

        rsi_oversold = prev_prev['rsi'] < 30
        stoch_cross = (prev_prev['stoch_k'] < prev_prev['stoch_d']) and (prev['stoch_k'] > prev['stoch_d'])
        trend_filter = prev['close'] > prev['ema_trend']

        if rsi_oversold and stoch_cross and trend_filter:
            signals.iloc[i] = 'BUY'

    return signals

# ============================
# 4. BACKTEST ROBUSTO
# ============================

class BacktestEngine:
    def __init__(self, config: Dict):
        self.initial_balance = config['initial_balance']
        self.fee = config['fee']
        self.slippage = config['slippage']
        self.risk_per_trade = config['risk_per_trade']
        self.stop_loss = config['stop_loss']
        self.take_profit = config['take_profit']

    def run(self, df: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        balance = self.initial_balance
        position = None
        trades = []
        equity = [self.initial_balance]

        for i in range(2, len(df)):
            current = df.iloc[i]

            # Actualizar equity (balance + valor no realizado)
            if position:
                unrealized = (current['close'] - position['entry_price']) / position['entry_price'] * position['margin']
                equity.append(balance + unrealized)
            else:
                equity.append(balance)

            # ENTRADA: Señal en vela actual, ejecutamos al APERTURA
            if signals.iloc[i] == 'BUY' and position is None and balance > 10:
                position = self._open_position(current, balance)
                if position:
                    balance -= position['margin']

            # GESTIÓN DE POSICIÓN
            if position:
                exit_price, reason = self._check_exit(position, current)
                if exit_price:
                    trade = self._close_position(position, current.name, exit_price, reason)
                    balance += trade.pnl + position['margin']
                    trades.append(trade)
                    position = None

        equity_curve = pd.Series(equity, index=df.index[:len(equity)])

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=self._calc_metrics(trades, equity_curve),
            config=self.__dict__
        )

    def _open_position(self, candle, balance: float) -> Optional[Dict]:
        """Abre posición con gestión de riesgo por operación."""

        entry_price = candle['open'] * (1 + self.slippage)
        risk_amount = balance * self.risk_per_trade
        stop_distance = self.stop_loss

        size = risk_amount / (entry_price * stop_distance)
        margin = size * entry_price

        if margin > balance * 0.95:
            size = (balance * 0.95) / entry_price
            margin = size * entry_price

        return {
            'side': 'LONG',
            'entry_price': entry_price,
            'entry_time': candle.name,
            'size': size,
            'margin': margin,
            'stop_loss': entry_price * (1 - self.stop_loss),
            'take_profit': entry_price * (1 + self.take_profit)
        }

    def _check_exit(self, position: Dict, candle) -> Tuple[Optional[float], Optional[str]]:
        """Verifica SL/TP usando high/low de la vela."""

        if position['side'] == 'LONG':
            if candle['low'] <= position['stop_loss']:
                exit_price = min(position['stop_loss'], candle['open'])
                return exit_price, 'stop_loss'
            if candle['high'] >= position['take_profit']:
                exit_price = max(position['take_profit'], candle['open'])
                return exit_price, 'take_profit'
        return None, None

    def _close_position(self, position: Dict, exit_time, exit_price: float, reason: str) -> Trade:
        """Cierra posición con comisiones reales."""

        actual_exit = exit_price * (1 - self.slippage)
        gross_pnl = (actual_exit - position['entry_price']) * position['size']

        entry_fee = position['entry_price'] * position['size'] * self.fee
        exit_fee = actual_exit * position['size'] * self.fee
        total_fees = entry_fee + exit_fee

        net_pnl = gross_pnl - total_fees

        return Trade(
            entry_time=str(position['entry_time']),
            exit_time=str(exit_time),
            entry_price=round(position['entry_price'], 6),
            exit_price=round(actual_exit, 6),
            side=position['side'],
            size=round(position['size'], 2),
            margin=round(position['margin'], 2),
            pnl=round(net_pnl, 2),
            pnl_pct=round(net_pnl / self.initial_balance * 100, 3),
            reason=reason,
            fees=round(total_fees, 4)
        )

    def _calc_metrics(self, trades: List[Trade], equity: pd.Series) -> Dict:
        if not trades:
            return {'error': 'No trades executed'}

        returns = equity.pct_change().dropna()
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)

        return {
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': round(len(wins) / len(trades) * 100, 2),
            'profit_factor': round(abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)), 2) if losses else float('inf'),
            'total_return_pct': round((equity.iloc[-1] / self.initial_balance - 1) * 100, 2),
            'total_pnl_usd': round(total_pnl, 2),
            'avg_trade_usd': round(total_pnl / len(trades), 2),
            'avg_win_usd': round(sum(t.pnl for t in wins) / len(wins), 2) if wins else 0,
            'avg_loss_usd': round(sum(t.pnl for t in losses) / len(losses), 2) if losses else 0,
            'max_drawdown_pct': round((equity / equity.cummax() - 1).min() * 100, 2),
            'sharpe_ratio': round(returns.mean() / returns.std() * np.sqrt(252 * 288), 2) if returns.std() != 0 else 0,
            'final_balance': round(equity.iloc[-1], 2),
        }

# ============================
# 5. RESULTADOS
# ============================

def print_results(result: BacktestResult, config: Dict):
    m = result.metrics
    print("\n" + "="*60)
    print("📊 RESULTADOS DEL BACKTEST - XRP/USDT")
    print("="*60)
    print(f"\n💰 BALANCE FINAL: ${m['final_balance']} (Inicial: ${config['initial_balance']})")
    print(f"📈 RETORNO TOTAL: {m['total_return_pct']}%")
    print(f"📉 MAX DRAWDOWN: {m['max_drawdown_pct']}%")
    print(f"\n🎯 OPERACIONES:")
    print(f"   Total: {m['total_trades']} | Ganadas: {m['winning_trades']} | Perdidas: {m['losing_trades']}")
    print(f"   Win Rate: {m['win_rate']}%")
    print(f"   Profit Factor: {m['profit_factor']}")
    print(f"\n💵 P&L:")
    print(f"   Total: ${m['total_pnl_usd']}")
    print(f"   Promedio por trade: ${m['avg_trade_usd']}")
    print(f"   Promedio ganadora: ${m['avg_win_usd']}")
    print(f"   Promedio perdedora: ${m['avg_loss_usd']}")
    print(f"\n📊 RATIO DE SHARPE: {m['sharpe_ratio']}")
    print(f"\n📋 ÚLTIMAS 5 OPERACIONES:")
    print("-"*80)
    for t in result.trades[-5:]:
        emoji = "🟢" if t.pnl > 0 else "🔴"
        print(f"{emoji} {t.entry_time[:19]} → {t.exit_time[:19]} | ${t.pnl} ({t.pnl_pct}%) | {t.reason.upper()}")
    print("\n" + "="*60)

def plot_results(result: BacktestResult, config: Dict):
    """Genera gráfico de equity curve y drawdown."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Equity Curve
        axes[0].plot(result.equity_curve.index, result.equity_curve.values, 
                     linewidth=1.5, color='#00d4aa', label='Balance')
        axes[0].axhline(y=config['initial_balance'], color='gray', linestyle='--', alpha=0.5)
        axes[0].fill_between(result.equity_curve.index, result.equity_curve.values, 
                             config['initial_balance'], alpha=0.2, color='green')

        for t in result.trades:
            entry_time = pd.to_datetime(t.entry_time)
            exit_time = pd.to_datetime(t.exit_time)
            color = '#00ff00' if t.pnl > 0 else '#ff0000'
            axes[0].scatter([entry_time], [result.equity_curve.loc[entry_time]], 
                           color=color, marker='^', s=80, zorder=5)
            axes[0].scatter([exit_time], [result.equity_curve.loc[exit_time]], 
                           color=color, marker='v', s=80, zorder=5)

        axes[0].set_title('Equity Curve - XRP/USDT RSI+Stoch', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Balance (USD)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Drawdown
        drawdown = (result.equity_curve / result.equity_curve.cummax() - 1) * 100
        axes[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.4, color='red')
        axes[1].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        axes[1].set_title('Drawdown (%)')
        axes[1].set_ylabel('Drawdown %')
        axes[1].set_xlabel('Fecha')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/xrp_backtest_chart.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("📈 Gráfico guardado en results/xrp_backtest_chart.png")
    except ImportError:
        print("⚠️ matplotlib no instalado. Gráfico omitido.")

# ============================
# MAIN
# ============================

def main():
    print("🚀 BACKTEST RSI+STOCH XRP/USDT")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Datos reales de Binance
    df = get_data(CONFIG['symbol'], CONFIG['timeframe'], CONFIG['limit'])

    # 2. Indicadores
    df = add_indicators(df, CONFIG)

    # 3. Señales
    signals = generate_signals(df, CONFIG)
    print(f"📡 Señales: {signals.notna().sum()} de {len(signals)}")

    # 4. Backtest
    engine = BacktestEngine(CONFIG)
    result = engine.run(df, signals)

    # 5. Resultados
    print_results(result, CONFIG)

    # 6. Guardar
    os.makedirs('results', exist_ok=True)

    trades_dict = [asdict(t) for t in result.trades]
    with open('results/xrp_backtest_trades.json', 'w') as f:
        json.dump(trades_dict, f, indent=2, default=str)

    result.equity_curve.to_csv('results/xrp_backtest_equity.csv')

    with open('results/xrp_backtest_metrics.json', 'w') as f:
        json.dump(result.metrics, f, indent=2)

    # 7. Gráfico
    plot_results(result, CONFIG)

    print(f"\n💾 Todo guardado en /results/")
    return result

if __name__ == "__main__":
    result = main()
