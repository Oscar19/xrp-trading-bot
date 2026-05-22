"""
BOT RSI CANALES - BACKTEST v2.1 (EMA CRUCE)
==============================
Estrategia: Cruce alcista EMA 9/21
- EMA9 cruza por encima de EMA21
- Confirmación con volumen
- Solo LONG
"""

import pandas as pd
import numpy as np
import ccxt
import os
from datetime import datetime

CONFIG = {
    'symbol': os.environ.get('BITGET_SYMBOL', 'ADA/USDT'),
    'timeframe': '4h',
    'balance': float(os.environ.get('BALANCE', '1000')),
    'leverage': int(os.environ.get('LEVERAGE', '5')),
    'risk_per_trade': float(os.environ.get('RISK_PER_TRADE', '0.02')),
    'sl_pct': float(os.environ.get('SL_PCT', '0.008')),
    'tp_pct': float(os.environ.get('TP_PCT', '0.015')),
    'trailing_stop_pct': 0.008,
    'breakeven_trigger': 0.010,
    'time_exit_max': 20,
    'cooldown_horas': 8,
    'volumen_min_ratio': float(os.environ.get('VOLUMEN_MIN_RATIO', '1.2')),
    'volumen_lookback': 10,
    'ema_rapida': 9,
    'ema_lenta': 21,
}

LOG_FILE = None

def init_logging():
    global LOG_FILE
    log_path = os.environ.get('LOG_FILE', 'backtest_ema.txt')
    log_path = log_path.replace('/', '_')
    dir_name = os.path.dirname(log_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    LOG_FILE = open(log_path, 'w', encoding='utf-8')
    return log_path

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line)
    if LOG_FILE:
        LOG_FILE.write(line + '\n')
        LOG_FILE.flush()

def close_logging():
    if LOG_FILE:
        LOG_FILE.close()

def fetch_data(exchange, symbol, timeframe, limit=1000):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        log(f"Error descargando {timeframe}: {e}")
        return None

def get_millis(timeframe):
    mapping = {'1m': 60000, '5m': 300000, '15m': 900000, '30m': 1800000,
               '1h': 3600000, '2h': 7200000, '4h': 14400000, '1d': 86400000}
    return mapping.get(timeframe, 14400000)

def fetch_data_batches(exchange, symbol, timeframe, total_velas=2000):
    all_data = []
    limit = 1000
    log(f"Descargando batch 1 ({limit} velas)...")
    df1 = fetch_data(exchange, symbol, timeframe, limit=limit)
    if df1 is None or len(df1) == 0:
        return None
    all_data.append(df1)
    if total_velas > limit and len(df1) > 0:
        since = int(df1.index[0].timestamp() * 1000) - (limit * get_millis(timeframe))
        if since > 0:
            log(f"Descargando batch 2 desde {pd.to_datetime(since, unit='ms')}...")
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                if ohlcv and len(ohlcv) > 0:
                    df2 = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms')
                    df2.set_index('timestamp', inplace=True)
                    df2 = df2[df2.index < df1.index[0]]
                    if len(df2) > 0:
                        all_data.insert(0, df2)
                        log(f"Batch 2: {len(df2)} velas")
            except Exception as e:
                log(f"No se pudo descargar batch 2: {e}")
    df_combined = pd.concat(all_data)
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    df_combined.sort_index(inplace=True)
    return df_combined

def simular_trade(df, idx_entrada, entry_price, direccion='long'):
    if idx_entrada >= len(df) - 1:
        return None
    leverage = CONFIG['leverage']
    if direccion == 'long':
        sl_price = entry_price * (1 - CONFIG['sl_pct'])
        tp_price = entry_price * (1 + CONFIG['tp_pct'])
        min_recent = df['low'].iloc[max(0, idx_entrada - 3):idx_entrada].min()
        sl_final = max(sl_price, min_recent * 0.998)
        max_price = entry_price
        breakeven_activado = False
        sl_trailing = sl_final
        for i in range(1, min(CONFIG['time_exit_max'], len(df) - idx_entrada)):
            vela = df.iloc[idx_entrada + i]
            if vela['high'] > max_price:
                max_price = vela['high']
            ganancia_pct = (max_price - entry_price) / entry_price
            if ganancia_pct >= CONFIG['breakeven_trigger'] and not breakeven_activado:
                breakeven_activado = True
                sl_trailing = entry_price * 1.001
            if breakeven_activado:
                nuevo_sl = max_price * (1 - CONFIG['trailing_stop_pct'])
                if nuevo_sl > sl_trailing:
                    sl_trailing = nuevo_sl
            if vela['low'] <= sl_trailing:
                pnl_pct_raw = (sl_trailing - entry_price) / entry_price * 100
                pnl_pct = pnl_pct_raw * leverage
                return {'resultado': 'SL', 'exit_price': sl_trailing, 'pnl_pct': pnl_pct,
                        'pnl_pct_raw': pnl_pct_raw, 'velas_duracion': i,
                        'fecha_salida': df.index[idx_entrada + i],
                        'tipo_salida': 'trailing' if breakeven_activado else 'sl_fijo'}
            if vela['high'] >= tp_price:
                pnl_pct_raw = CONFIG['tp_pct'] * 100
                pnl_pct = pnl_pct_raw * leverage
                return {'resultado': 'TP', 'exit_price': tp_price, 'pnl_pct': pnl_pct,
                        'pnl_pct_raw': pnl_pct_raw, 'velas_duracion': i,
                        'fecha_salida': df.index[idx_entrada + i], 'tipo_salida': 'tp_fijo'}
    idx_salida = idx_entrada + min(CONFIG['time_exit_max'], len(df) - idx_entrada - 1)
    exit_price = df['close'].iloc[idx_salida]
    pnl_pct_raw = (exit_price - entry_price) / entry_price * 100
    pnl_pct = pnl_pct_raw * leverage
    return {'resultado': 'TIME_EXIT', 'exit_price': exit_price, 'pnl_pct': pnl_pct,
            'pnl_pct_raw': pnl_pct_raw, 'velas_duracion': min(CONFIG['time_exit_max'], len(df) - idx_entrada - 1),
            'fecha_salida': df.index[idx_salida], 'tipo_salida': 'time_exit'}

def backtest_ema_cruce(df):
    log("\n" + "="*60)
    log("BACKTEST: EMA_CRUCE (9/21)")
    log("="*60)

    # Calcular EMAs
    df['ema9'] = df['close'].ewm(span=CONFIG['ema_rapida'], adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=CONFIG['ema_lenta'], adjust=False).mean()

    trades = []
    rechazos = {'cruce': 0, 'volumen': 0, 'cooldown': 0}

    closes = df['close'].values
    ema9 = df['ema9'].values
    ema21 = df['ema21'].values
    ventana_min = max(CONFIG['ema_lenta'] + 10, 50)
    ultimo_trade_fecha = None

    for i in range(ventana_min, len(df) - 1):
        fecha = df.index[i]

        if ultimo_trade_fecha is not None:
            horas = (fecha - ultimo_trade_fecha).total_seconds() / 3600
            if horas < CONFIG['cooldown_horas']:
                rechazos['cooldown'] += 1
                continue

        # Cruce alcista: ema9[-2] < ema21[-2] y ema9[-1] > ema21[-1]
        if not (ema9[i-1] < ema21[i-1] and ema9[i] > ema21[i]):
            rechazos['cruce'] += 1
            continue

        # Volumen
        volumen_actual = df['volume'].iloc[i]
        volumen_media = df['volume'].iloc[max(0, i - CONFIG['volumen_lookback']):i].mean()
        if volumen_media > 0:
            ratio_volumen = volumen_actual / volumen_media
            if ratio_volumen < CONFIG['volumen_min_ratio']:
                rechazos['volumen'] += 1
                continue
        else:
            ratio_volumen = 0

        entry_price = closes[i]
        resultado = simular_trade(df, i, entry_price, 'long')

        if resultado:
            ultimo_trade_fecha = fecha
            trades.append({
                'fecha_entrada': fecha,
                'idx': i,
                'entry': entry_price,
                'estrategia': 'EMA_CRUCE',
                'ema9': ema9[i],
                'ema21': ema21[i],
                'ratio_volumen': ratio_volumen,
                **resultado
            })

    log(f"Trades EMA_CRUCE: {len(trades)}")
    for k, v in rechazos.items():
        log(f"  {k}: {v}")

    return trades

def calcular_metricas(trades, balance_inicial=1000):
    if not trades:
        return None
    n_trades = len(trades)
    ganadores = [t for t in trades if t['pnl_pct'] > 0]
    perdedores = [t for t in trades if t['pnl_pct'] <= 0]
    n_ganadores = len(ganadores)
    n_perdedores = len(perdedores)
    win_rate = n_ganadores / n_trades * 100 if n_trades > 0 else 0
    pnl_total = sum(t['pnl_pct'] for t in trades)
    pnl_promedio = pnl_total / n_trades if n_trades > 0 else 0
    balance = balance_inicial
    max_balance = balance
    max_drawdown = 0
    max_drawdown_usd = 0
    leverage = CONFIG['leverage']
    for trade in trades:
        riesgo = balance * CONFIG['risk_per_trade']
        pnl_usd = riesgo * (trade['pnl_pct'] / (CONFIG['sl_pct'] * 100 * leverage))
        balance += pnl_usd
        if balance > max_balance:
            max_balance = balance
        drawdown = (max_balance - balance) / max_balance * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_usd = max_balance - balance
    ganancias_totales = sum(t['pnl_pct'] for t in ganadores)
    perdidas_totales = abs(sum(t['pnl_pct'] for t in perdedores))
    profit_factor = ganancias_totales / perdidas_totales if perdidas_totales > 0 else float('inf')
    avg_win = np.mean([t['pnl_pct'] for t in ganadores]) if ganadores else 0
    avg_loss = np.mean([t['pnl_pct'] for t in perdedores]) if perdedores else 0
    expectancy = (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss)
    returns = [t['pnl_pct'] for t in trades]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252*6) if np.std(returns) > 0 else 0
    return {
        'n_trades': n_trades, 'n_ganadores': n_ganadores, 'n_perdedores': n_perdedores,
        'win_rate': win_rate, 'pnl_total_pct': (balance - balance_inicial) / balance_inicial * 100,
        'pnl_promedio': pnl_promedio, 'balance_inicial': balance_inicial, 'balance_final': balance,
        'max_drawdown': max_drawdown, 'max_drawdown_usd': max_drawdown_usd,
        'profit_factor': profit_factor, 'expectancy': expectancy, 'sharpe': sharpe,
        'avg_win': avg_win, 'avg_loss': avg_loss,
    }

def main():
    init_logging()
    try:
        log("="*60)
        log("BOT RSI CANALES - BACKTEST EMA CRUCE v2.1")
        log(f"Par: {CONFIG['symbol']}")
        log(f"Timeframe: {CONFIG['timeframe']}")
        log(f"EMA: {CONFIG['ema_rapida']}/{CONFIG['ema_lenta']}")
        log(f"SL: {CONFIG['sl_pct']*100:.1f}% | TP: {CONFIG['tp_pct']*100:.1f}%")
        log(f"Apalancamiento: {CONFIG['leverage']}x")
        log("="*60)

        try:
            exchange = ccxt.bitget({'options': {'defaultType': 'swap'}, 'timeout': 30000, 'enableRateLimit': True})
            log("Conectado a Bitget")
        except Exception as e:
            log(f"Error conectando: {e}"); return

        log(f"Descargando datos...")
        df = fetch_data_batches(exchange, CONFIG['symbol'], CONFIG['timeframe'], total_velas=2000)
        if df is None:
            df = fetch_data(exchange, CONFIG['symbol'], CONFIG['timeframe'], limit=1000)
        if df is None:
            log("Error descargando datos"); return

        log(f"Datos: {len(df)} velas | {df.index[0]} -> {df.index[-1]}")

        trades = backtest_ema_cruce(df)
        metricas = calcular_metricas(trades)

        log("\n" + "="*60)
        log("RESULTADOS EMA_CRUCE")
        log("="*60)

        if metricas:
            log(f"Trades: {metricas['n_trades']} (G:{metricas['n_ganadores']} P:{metricas['n_perdedores']})")
            log(f"Win Rate: {metricas['win_rate']:.1f}%")
            log(f"Profit Factor: {metricas['profit_factor']:.2f}")
            log(f"Expectancy: {metricas['expectancy']:.2f}%")
            log(f"Sharpe: {metricas['sharpe']:.2f}")
            log(f"Max DD: {metricas['max_drawdown']:.2f}%")
            log(f"Return: {metricas['pnl_total_pct']:.2f}%")
            log(f"Balance: ${metricas['balance_inicial']:.0f} -> ${metricas['balance_final']:.2f}")

            if trades:
                log("\nDETALLE DE TRADES:")
                log("-" * 60)
                for i, t in enumerate(trades, 1):
                    emoji = "🟢" if t['pnl_pct'] > 0 else "🔴"
                    log(f"{emoji} #{i} | {t['fecha_entrada'].strftime('%Y-%m-%d %H:%M')}")
                    log(f"   Entrada: ${t['entry']:.4f} -> Salida: ${t['exit_price']:.4f}")
                    log(f"   Resultado: {t['resultado']} | P&L {CONFIG['leverage']}x: {t['pnl_pct']:+.2f}%")
                    log(f"   EMA9: {t['ema9']:.4f} | EMA21: {t['ema21']:.4f}")
                    log(f"   Volumen: {t['ratio_volumen']:.1f}x | Duración: {t['velas_duracion']} velas")
                    log("")
        else:
            log("No hubo trades")

        log("="*60)
        log("Backtest finalizado")
        log("="*60)
    finally:
        close_logging()

if __name__ == "__main__":
    main()
