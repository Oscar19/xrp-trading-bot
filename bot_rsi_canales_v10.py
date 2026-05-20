
"""
BOT RSI CANALES - BACKTEST v10.3
==============================
- Apalancamiento REAL (5x) en cálculos de P&L
- Par: SOL/USDT por defecto
- Timeframe: 4h
- Rupturas de diagonales RSI + Volumen
"""

import pandas as pd
import numpy as np
import ccxt
import os
from datetime import datetime, timedelta
from scipy import stats

CONFIG = {
    'symbol': os.environ.get('BITGET_SYMBOL', 'SOL/USDT'),
    'timeframe': '4h',
    'balance': float(os.environ.get('BALANCE', '1000')),
    'leverage': int(os.environ.get('LEVERAGE', '5')),  # <-- AHORA SÍ SE USA
    'risk_per_trade': float(os.environ.get('RISK_PER_TRADE', '0.02')),
    'sl_pct': float(os.environ.get('SL_PCT', '0.008')),
    'tp_pct': float(os.environ.get('TP_PCT', '0.015')),
    'trailing_stop_pct': 0.008,
    'breakeven_trigger': 0.010,
    'time_exit_max': 20,
    'cooldown_horas': 8,

    'volumen_min_ratio': 1.5,      # <-- BAJADO a 1.5x para más señales
    'volumen_lookback': 10,
    'rsi_period': 14,
    'pivot_ventana': 3,
    'min_puntos_diagonal': 3,
    'pendiente_max_diagonal': 0.05,
    'r2_min_diagonal': 0.15,
    'zona_sobreventa': 40,
    'zona_sobrecompra': 60,

    'direccion': os.environ.get('DIRECCION', 'both'),
    'debug': os.environ.get('DEBUG', 'false').lower() == 'true',
}

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

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

def calcular_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detectar_diagonal_maximos_rsi(rsi_values, ventana_pivot=3, min_puntos=3,
                                   pendiente_max=-0.01, r2_min=0.15):
    rsi = np.array(rsi_values)
    n = len(rsi)

    if n < ventana_pivot * 2 + 1:
        return None, "Datos insuficientes"

    maximos_idx = []
    maximos_val = []

    for i in range(ventana_pivot, n - ventana_pivot):
        ventana = rsi[i - ventana_pivot:i + ventana_pivot + 1]
        if rsi[i] == ventana.max() and rsi[i] > 30:
            if not maximos_idx or (i - maximos_idx[-1]) >= 5:
                maximos_idx.append(i)
                maximos_val.append(rsi[i])

    if len(maximos_idx) < min_puntos:
        return None, f"Pocos máximos ({len(maximos_idx)} < {min_puntos})"

    best_diagonal = None
    best_score = -999

    for n_usar in range(min_puntos, min(20, len(maximos_idx)) + 1):
        x = np.array(maximos_idx[-n_usar:])
        y = np.array(maximos_val[-n_usar:])

        if len(x) < 2:
            continue

        m, b, r, p, se = stats.linregress(x, y)

        if m > pendiente_max:
            continue

        r2 = r ** 2
        if r2 < r2_min:
            continue

        score = r2 * 100 + n_usar * 2 + (x[-1] / n) * 10

        if score > best_score:
            best_score = score
            best_diagonal = {
                'm': m, 'b': b, 'r2': r2,
                'maximos_idx': maximos_idx, 'maximos_val': maximos_val,
                'n_usados': n_usar,
                'ultimo_max_idx': x[-1], 'ultimo_max_val': y[-1],
                'primero_max_idx': x[0], 'primero_max_val': y[0],
            }

    if best_diagonal is None and len(maximos_idx) >= 2:
        x = np.array(maximos_idx[-2:])
        y = np.array(maximos_val[-2:])
        m, b, r, p, se = stats.linregress(x, y)
        if m < 0:
            return {
                'm': m, 'b': b, 'r2': r**2,
                'maximos_idx': maximos_idx, 'maximos_val': maximos_val,
                'n_usados': 2, 'ultimo_max_idx': x[-1], 'ultimo_max_val': y[-1],
                'primero_max_idx': x[0], 'primero_max_val': y[0],
            }, "OK (fallback 2 puntos)"

    return best_diagonal, "OK" if best_diagonal else f"No diagonal válida"

def detectar_diagonal_minimos_rsi(rsi_values, ventana_pivot=3, min_puntos=3,
                                   pendiente_min=0.01, r2_min=0.15):
    rsi = np.array(rsi_values)
    n = len(rsi)

    if n < ventana_pivot * 2 + 1:
        return None, "Datos insuficientes"

    minimos_idx = []
    minimos_val = []

    for i in range(ventana_pivot, n - ventana_pivot):
        ventana = rsi[i - ventana_pivot:i + ventana_pivot + 1]
        if rsi[i] == ventana.min() and rsi[i] < 70:
            if not minimos_idx or (i - minimos_idx[-1]) >= 5:
                minimos_idx.append(i)
                minimos_val.append(rsi[i])

    if len(minimos_idx) < min_puntos:
        return None, f"Pocos mínimos ({len(minimos_idx)} < {min_puntos})"

    best_diagonal = None
    best_score = -999

    for n_usar in range(min_puntos, min(20, len(minimos_idx)) + 1):
        x = np.array(minimos_idx[-n_usar:])
        y = np.array(minimos_val[-n_usar:])

        if len(x) < 2:
            continue

        m, b, r, p, se = stats.linregress(x, y)

        if m < pendiente_min:
            continue

        r2 = r ** 2
        if r2 < r2_min:
            continue

        score = r2 * 100 + n_usar * 2 + (x[-1] / n) * 10

        if score > best_score:
            best_score = score
            best_diagonal = {
                'm': m, 'b': b, 'r2': r2,
                'minimos_idx': minimos_idx, 'minimos_val': minimos_val,
                'n_usados': n_usar,
                'ultimo_min_idx': x[-1], 'ultimo_min_val': y[-1],
                'primero_min_idx': x[0], 'primero_min_val': y[0],
            }

    if best_diagonal is None and len(minimos_idx) >= 2:
        x = np.array(minimos_idx[-2:])
        y = np.array(minimos_val[-2:])
        m, b, r, p, se = stats.linregress(x, y)
        if m > 0:
            return {
                'm': m, 'b': b, 'r2': r**2,
                'minimos_idx': minimos_idx, 'minimos_val': minimos_val,
                'n_usados': 2, 'ultimo_min_idx': x[-1], 'ultimo_min_val': y[-1],
                'primero_min_idx': x[0], 'primero_min_val': y[0],
            }, "OK (fallback 2 puntos)"

    return best_diagonal, "OK" if best_diagonal else "No diagonal válida"

def detectar_ruptura_diagonal_long(rsi_values, diagonal, df, idx, 
                                    umbral_ruptura=0.5, volumen_ratio_min=1.5, 
                                    volumen_lookback=10):
    if not diagonal:
        return False, None

    n = len(rsi_values)
    if n < 3 or idx < 2:
        return False, None

    val_diag_2 = diagonal['m'] * (n - 2) + diagonal['b']
    val_diag_1 = diagonal['m'] * (n - 1) + diagonal['b']

    rsi_2 = rsi_values[-2]
    rsi_1 = rsi_values[-1]

    condicion_base = (rsi_2 <= val_diag_2 + umbral_ruptura) and (rsi_1 > val_diag_1 + umbral_ruptura)
    momentum = rsi_1 > rsi_2

    if not (condicion_base and momentum):
        return False, None

    if df is not None and 'volume' in df.columns and idx < len(df):
        volumen_actual = df['volume'].iloc[idx]
        volumen_media = df['volume'].iloc[max(0, idx - volumen_lookback):idx].mean()

        if volumen_media > 0:
            ratio_volumen = volumen_actual / volumen_media
            volumen_confirmado = ratio_volumen >= volumen_ratio_min
        else:
            ratio_volumen = 0
            volumen_confirmado = False
    else:
        ratio_volumen = 0
        volumen_confirmado = False

    return True, {
        'tipo': 'ruptura_long',
        'rsi_antes': rsi_2,
        'rsi_despues': rsi_1,
        'diag_antes': val_diag_2,
        'diag_despues': val_diag_1,
        'diferencia': rsi_1 - val_diag_1,
        'ratio_volumen': ratio_volumen,
        'volumen_confirmado': volumen_confirmado,
        'pendiente_diagonal': diagonal['m'],
        'r2_diagonal': diagonal['r2'],
        'n_puntos': diagonal['n_usados'],
    }

def detectar_ruptura_diagonal_short(rsi_values, diagonal, df, idx,
                                     umbral_ruptura=0.5, volumen_ratio_min=1.5,
                                     volumen_lookback=10):
    if not diagonal:
        return False, None

    n = len(rsi_values)
    if n < 3 or idx < 2:
        return False, None

    val_diag_2 = diagonal['m'] * (n - 2) + diagonal['b']
    val_diag_1 = diagonal['m'] * (n - 1) + diagonal['b']

    rsi_2 = rsi_values[-2]
    rsi_1 = rsi_values[-1]

    condicion_base = (rsi_2 >= val_diag_2 - umbral_ruptura) and (rsi_1 < val_diag_1 - umbral_ruptura)
    momentum = rsi_1 < rsi_2

    if not (condicion_base and momentum):
        return False, None

    if df is not None and 'volume' in df.columns and idx < len(df):
        volumen_actual = df['volume'].iloc[idx]
        volumen_media = df['volume'].iloc[max(0, idx - volumen_lookback):idx].mean()
        ratio_volumen = volumen_actual / volumen_media if volumen_media > 0 else 0
        volumen_confirmado = ratio_volumen >= volumen_ratio_min
    else:
        ratio_volumen = 0
        volumen_confirmado = False

    return True, {
        'tipo': 'ruptura_short',
        'rsi_antes': rsi_2,
        'rsi_despues': rsi_1,
        'diag_antes': val_diag_2,
        'diag_despues': val_diag_1,
        'diferencia': val_diag_1 - rsi_1,
        'ratio_volumen': ratio_volumen,
        'volumen_confirmado': volumen_confirmado,
        'pendiente_diagonal': diagonal['m'],
        'r2_diagonal': diagonal['r2'],
        'n_puntos': diagonal['n_usados'],
    }

def detectar_tendencia_precio(df, idx, periodos=30):
    if idx < periodos:
        return 'lateral'

    precios = df['close'].iloc[idx - periodos:idx + 1].values
    sma = np.mean(precios)

    if precios[-1] > sma * 1.015:
        return 'alcista'
    elif precios[-1] < sma * 0.985:
        return 'bajista'
    else:
        return 'lateral'

def simular_trade(df, idx_entrada, entry_price, direccion='long'):
    if idx_entrada >= len(df) - 1:
        return None

    leverage = CONFIG['leverage']  # <-- AHORA SE USA EL APALANCAMIENTO

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
                pnl_pct = pnl_pct_raw * leverage  # <-- APLICAR APALANCAMIENTO
                return {'resultado': 'SL', 'exit_price': sl_trailing, 'pnl_pct': pnl_pct,
                        'pnl_pct_raw': pnl_pct_raw,  # <-- Guardar P&L sin apalancamiento
                        'velas_duracion': i, 'fecha_salida': df.index[idx_entrada + i],
                        'tipo_salida': 'trailing' if breakeven_activado else 'sl_fijo',
                        'direccion': 'long'}

            if vela['high'] >= tp_price:
                pnl_pct_raw = CONFIG['tp_pct'] * 100
                pnl_pct = pnl_pct_raw * leverage
                return {'resultado': 'TP', 'exit_price': tp_price, 'pnl_pct': pnl_pct,
                        'pnl_pct_raw': pnl_pct_raw,
                        'velas_duracion': i, 'fecha_salida': df.index[idx_entrada + i],
                        'tipo_salida': 'tp_fijo', 'direccion': 'long'}

    else:
        sl_price = entry_price * (1 + CONFIG['sl_pct'])
        tp_price = entry_price * (1 - CONFIG['tp_pct'])
        max_recent = df['high'].iloc[max(0, idx_entrada - 3):idx_entrada].max()
        sl_final = min(sl_price, max_recent * 1.002)

        min_price = entry_price
        breakeven_activado = False
        sl_trailing = sl_final

        for i in range(1, min(CONFIG['time_exit_max'], len(df) - idx_entrada)):
            vela = df.iloc[idx_entrada + i]

            if vela['low'] < min_price:
                min_price = vela['low']

            ganancia_pct = (entry_price - min_price) / entry_price
            if ganancia_pct >= CONFIG['breakeven_trigger'] and not breakeven_activado:
                breakeven_activado = True
                sl_trailing = entry_price * 0.999

            if breakeven_activado:
                nuevo_sl = min_price * (1 + CONFIG['trailing_stop_pct'])
                if nuevo_sl < sl_trailing:
                    sl_trailing = nuevo_sl

            if vela['high'] >= sl_trailing:
                pnl_pct_raw = (entry_price - sl_trailing) / entry_price * 100
                pnl_pct = pnl_pct_raw * leverage
                return {'resultado': 'SL', 'exit_price': sl_trailing, 'pnl_pct': pnl_pct,
                        'pnl_pct_raw': pnl_pct_raw,
                        'velas_duracion': i, 'fecha_salida': df.index[idx_entrada + i],
                        'tipo_salida': 'trailing' if breakeven_activado else 'sl_fijo',
                        'direccion': 'short'}

            if vela['low'] <= tp_price:
                pnl_pct_raw = CONFIG['tp_pct'] * 100
                pnl_pct = pnl_pct_raw * leverage
                return {'resultado': 'TP', 'exit_price': tp_price, 'pnl_pct': pnl_pct,
                        'pnl_pct_raw': pnl_pct_raw,
                        'velas_duracion': i, 'fecha_salida': df.index[idx_entrada + i],
                        'tipo_salida': 'tp_fijo', 'direccion': 'short'}

    idx_salida = idx_entrada + min(CONFIG['time_exit_max'], len(df) - idx_entrada - 1)
    exit_price = df['close'].iloc[idx_salida]

    if direccion == 'long':
        pnl_pct_raw = (exit_price - entry_price) / entry_price * 100
    else:
        pnl_pct_raw = (entry_price - exit_price) / entry_price * 100

    pnl_pct = pnl_pct_raw * leverage

    return {
        'resultado': 'TIME_EXIT',
        'exit_price': exit_price,
        'pnl_pct': pnl_pct,
        'pnl_pct_raw': pnl_pct_raw,
        'velas_duracion': min(CONFIG['time_exit_max'], len(df) - idx_entrada - 1),
        'fecha_salida': df.index[idx_salida],
        'tipo_salida': 'time_exit',
        'direccion': direccion
    }

def backtest(df_4h):
    log("="*60)
    log(f"INICIANDO BACKTEST v10.3 - Rupturas Diagonales RSI 4H + Volumen + Apalancamiento {CONFIG['leverage']}x")
    log("="*60)

    trades = []
    rechazos = {
        'diagonal_long': 0, 'diagonal_short': 0,
        'ruptura_long': 0, 'ruptura_short': 0,
        'volumen_long': 0, 'volumen_short': 0,
        'zona_rsi': 0, 'cooldown': 0,
        'direccion': 0, 'tendencia': 0
    }

    rsi = df_4h['rsi'].values
    closes = df_4h['close'].values

    ventana_min = 50
    ultimo_trade_fecha = None

    log(f"Total velas 4h: {len(df_4h)}")
    log(f"Periodo: {df_4h.index[0]} -> {df_4h.index[-1]}")
    log(f"Dirección: {CONFIG['direccion']}")
    log(f"Apalancamiento: {CONFIG['leverage']}x")
    log(f"Analizando...")

    for i in range(ventana_min, len(df_4h) - 1):
        fecha = df_4h.index[i]

        if ultimo_trade_fecha is not None:
            horas_desde_ultimo = (fecha - ultimo_trade_fecha).total_seconds() / 3600
            if horas_desde_ultimo < CONFIG['cooldown_horas']:
                rechazos['cooldown'] += 1
                continue

        rsi_window = rsi[:i + 1]
        rsi_actual = rsi[i]

        tendencia = detectar_tendencia_precio(df_4h, i)

        diagonal_max, msg_max = detectar_diagonal_maximos_rsi(
            rsi_window,
            ventana_pivot=CONFIG['pivot_ventana'],
            min_puntos=CONFIG['min_puntos_diagonal'],
            pendiente_max=CONFIG['pendiente_max_diagonal'],
            r2_min=CONFIG['r2_min_diagonal']
        )

        diagonal_min, msg_min = detectar_diagonal_minimos_rsi(
            rsi_window,
            ventana_pivot=CONFIG['pivot_ventana'],
            min_puntos=CONFIG['min_puntos_diagonal'],
            pendiente_min=abs(CONFIG['pendiente_max_diagonal']),
            r2_min=CONFIG['r2_min_diagonal']
        )

        senal_long = False
        info_long = None

        if diagonal_max:
            ruptura_long, info_long = detectar_ruptura_diagonal_long(
                rsi_window, diagonal_max, df_4h, i,
                umbral_ruptura=0.5,
                volumen_ratio_min=CONFIG['volumen_min_ratio'],
                volumen_lookback=CONFIG['volumen_lookback']
            )

            if not ruptura_long:
                rechazos['ruptura_long'] += 1
            elif not info_long.get('volumen_confirmado', False):
                rechazos['volumen_long'] += 1
                ruptura_long = False
            elif rsi_actual >= CONFIG['zona_sobreventa']:
                rechazos['zona_rsi'] += 1
                ruptura_long = False
            elif tendencia == 'bajista':
                rechazos['tendencia'] += 1
                ruptura_long = False
            else:
                senal_long = True
        else:
            rechazos['diagonal_long'] += 1

        senal_short = False
        info_short = None

        if diagonal_min:
            ruptura_short, info_short = detectar_ruptura_diagonal_short(
                rsi_window, diagonal_min, df_4h, i,
                umbral_ruptura=0.5,
                volumen_ratio_min=CONFIG['volumen_min_ratio'],
                volumen_lookback=CONFIG['volumen_lookback']
            )

            if not ruptura_short:
                rechazos['ruptura_short'] += 1
            elif not info_short.get('volumen_confirmado', False):
                rechazos['volumen_short'] += 1
                ruptura_short = False
            elif rsi_actual <= CONFIG['zona_sobrecompra']:
                rechazos['zona_rsi'] += 1
                ruptura_short = False
            elif tendencia == 'alcista':
                rechazos['tendencia'] += 1
                ruptura_short = False
            else:
                senal_short = True
        else:
            rechazos['diagonal_short'] += 1

        direccion_final = None
        info_ruptura = None

        if senal_long and senal_short:
            if info_long['ratio_volumen'] >= info_short['ratio_volumen']:
                direccion_sugerida = 'long'
                info_ruptura = info_long
            else:
                direccion_sugerida = 'short'
                info_ruptura = info_short
        elif senal_long:
            direccion_sugerida = 'long'
            info_ruptura = info_long
        elif senal_short:
            direccion_sugerida = 'short'
            info_ruptura = info_short
        else:
            continue

        if CONFIG['direccion'] == 'both':
            direccion_final = direccion_sugerida
        elif CONFIG['direccion'] == direccion_sugerida:
            direccion_final = direccion_sugerida
        else:
            rechazos['direccion'] += 1
            continue

        entry_price = closes[i]

        resultado_trade = simular_trade(df_4h, i, entry_price, direccion_final)

        if resultado_trade:
            ultimo_trade_fecha = fecha
            trades.append({
                'fecha_entrada': fecha,
                'idx': i,
                'entry': entry_price,
                'sl': entry_price * (1 - CONFIG['sl_pct']) if direccion_final == 'long' else entry_price * (1 + CONFIG['sl_pct']),
                'tp': entry_price * (1 + CONFIG['tp_pct']) if direccion_final == 'long' else entry_price * (1 - CONFIG['tp_pct']),
                'rsi': rsi_actual,
                'direccion': direccion_final,
                'tendencia': tendencia,
                'ratio_volumen': info_ruptura['ratio_volumen'],
                'volumen_confirmado': info_ruptura['volumen_confirmado'],
                'pendiente_diagonal': info_ruptura['pendiente_diagonal'],
                'r2_diagonal': info_ruptura['r2_diagonal'],
                'n_puntos_diagonal': info_ruptura['n_puntos'],
                **resultado_trade
            })

    log(f"")
    log(f"DEBUG - Rechazos:")
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
    win_rate = n_ganadores / n_trades * 100

    pnl_total = sum(t['pnl_pct'] for t in trades)
    pnl_promedio = pnl_total / n_trades

    balance = balance_inicial
    max_balance = balance
    max_drawdown = 0
    max_drawdown_usd = 0
    balances = [balance]

    leverage = CONFIG['leverage']

    for trade in trades:
        # El riesgo se calcula sobre el balance, pero el P&L ya incluye apalancamiento
        riesgo = balance * CONFIG['risk_per_trade']
        # Ajustar el riesgo por apalancamiento: con 5x, el SL real es 5 veces más grande
        pnl_usd = riesgo * (trade['pnl_pct'] / (CONFIG['sl_pct'] * 100 * leverage))
        balance += pnl_usd
        balances.append(balance)

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

    longs = [t for t in trades if t.get('direccion') == 'long']
    shorts = [t for t in trades if t.get('direccion') == 'short']

    return {
        'n_trades': n_trades,
        'n_ganadores': n_ganadores,
        'n_perdedores': n_perdedores,
        'win_rate': win_rate,
        'pnl_total_pct': (balance - balance_inicial) / balance_inicial * 100,
        'pnl_promedio': pnl_promedio,
        'balance_inicial': balance_inicial,
        'balance_final': balance,
        'max_drawdown': max_drawdown,
        'max_drawdown_usd': max_drawdown_usd,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'sharpe': sharpe,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'balances': balances,
        'longs': len(longs),
        'shorts': len(shorts),
        'pnl_longs': sum(t['pnl_pct'] for t in longs),
        'pnl_shorts': sum(t['pnl_pct'] for t in shorts),
    }

def main():
    log("="*60)
    log("BOT RSI CANALES - BACKTEST v10.3")
    log(f"Par: {CONFIG['symbol']}")
    log(f"Timeframe: {CONFIG['timeframe']}")
    log(f"SL: {CONFIG['sl_pct']*100:.1f}% | TP: {CONFIG['tp_pct']*100:.1f}%")
    log(f"Apalancamiento: {CONFIG['leverage']}x")
    log(f"Dirección: {CONFIG['direccion']}")
    log(f"Volumen mínimo: {CONFIG['volumen_min_ratio']}x media")
    log(f"Zona LONG: RSI < {CONFIG['zona_sobreventa']}")
    log(f"Zona SHORT: RSI > {CONFIG['zona_sobrecompra']}")
    log(f"R2 mínimo diagonal: {CONFIG['r2_min_diagonal']}")
    log(f"Trailing stop: {CONFIG['trailing_stop_pct']*100:.1f}%")
    log(f"Cooldown: {CONFIG['cooldown_horas']}h")
    log(f"Filtro tendencia: ACTIVO (no contratendencia)")
    log("="*60)

    try:
        exchange = ccxt.bitget({
            'options': {'defaultType': 'swap'},
            'timeout': 30000,
            'enableRateLimit': True
        })
        log("Conectado a Bitget")
    except Exception as e:
        log(f"Error conectando: {e}")
        return

    log(f"Descargando datos {CONFIG['timeframe']}...")

    df = fetch_data_batches(exchange, CONFIG['symbol'], CONFIG['timeframe'], total_velas=2000)

    if df is None or len(df) == 0:
        df = fetch_data(exchange, CONFIG['symbol'], CONFIG['timeframe'], limit=1000)

    if df is None:
        log("Error descargando datos")
        return

    log(f"{CONFIG['timeframe']}: {len(df)} velas | {df.index[0]} -> {df.index[-1]}")

    df['rsi'] = calcular_rsi(df['close'], period=CONFIG['rsi_period'])

    trades = backtest(df)

    log("")
    log("="*60)
    log("RESULTADOS DEL BACKTEST v10.3")
    log("="*60)
    log(f"Periodo: {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")
    log(f"Total velas {CONFIG['timeframe']}: {len(df)}")
    log(f"")
    log(f"TRADES ENCONTRADOS: {len(trades)}")
    log(f"")

    if len(trades) == 0:
        log("NO HUBO NINGUNA SEÑAL EN ESTE PERIODO")
        log("")
        log("SUGERENCIAS:")
        log("1. Reducir min_puntos_diagonal (actual: 3)")
        log("2. Reducir r2_min_diagonal (actual: 0.15)")
        log("3. Aumentar ventana de datos (más histórico)")
        log("4. Reducir volumen_min_ratio (actual: 1.5)")
    else:
        metricas = calcular_metricas(trades)
        leverage = CONFIG['leverage']

        log("METRICAS GLOBALES:")
        log("-" * 60)
        log(f"Trades totales: {metricas['n_trades']}")
        log(f"  LONGS: {metricas['longs']} | SHORTS: {metricas['shorts']}")
        log(f"Ganadores: {metricas['n_ganadores']} | Perdedores: {metricas['n_perdedores']}")
        log(f"Win Rate: {metricas['win_rate']:.1f}%")
        log(f"Profit Factor: {metricas['profit_factor']:.2f}")
        log(f"Expectancy: {metricas['expectancy']:.2f}% por trade")
        log(f"Sharpe (anualizado): {metricas['sharpe']:.2f}")
        log(f"Max Drawdown: {metricas['max_drawdown']:.2f}% (${metricas['max_drawdown_usd']:.2f})")
        log(f"")
        log(f"Balance inicial: ${metricas['balance_inicial']:.2f}")
        log(f"Balance final: ${metricas['balance_final']:.2f}")
        log(f"Return total: {metricas['pnl_total_pct']:.2f}%")
        log(f"P&L LONGS: {metricas['pnl_longs']:+.2f}% | P&L SHORTS: {metricas['pnl_shorts']:+.2f}%")
        log(f"Promedio ganador: {metricas['avg_win']:.2f}% (sin apalancamiento: {metricas['avg_win']/leverage:.2f}%)")
        log(f"Promedio perdedor: {metricas['avg_loss']:.2f}% (sin apalancamiento: {metricas['avg_loss']/leverage:.2f}%)")
        log(f"")

        log("DETALLE DE TRADES:")
        log("-" * 60)
        for i, t in enumerate(trades, 1):
            emoji = "🟢" if t['pnl_pct'] > 0 else "🔴"
            dir_icon = "📈" if t.get('direccion') == 'long' else "📉"
            salida_icon = "⏱️" if t['resultado'] == 'TIME_EXIT' else ("🎯" if t['resultado'] == 'TP' else "🛑")
            vol_icon = "💥" if t.get('volumen_confirmado') else "📊"
            tendencia_icon = "🟢" if t.get('tendencia') == 'alcista' else ("🔴" if t.get('tendencia') == 'bajista' else "⚪")

            pnl_raw = t.get('pnl_pct_raw', t['pnl_pct'] / leverage)

            log(f"{emoji} #{i} {dir_icon} | {t['fecha_entrada'].strftime('%Y-%m-%d %H:%M')} {salida_icon} {vol_icon} {tendencia_icon}")
            log(f"   Entrada: ${t['entry']:.4f} -> Salida: ${t['exit_price']:.4f}")
            log(f"   Resultado: {t['resultado']} ({t['tipo_salida']})")
            log(f"   P&L con {leverage}x: {t['pnl_pct']:+.2f}% | P&L spot: {pnl_raw:+.2f}%")
            log(f"   Duración: {t['velas_duracion']} velas 4h")
            log(f"   RSI: {t['rsi']:.1f} | Volumen: {t.get('ratio_volumen', 0):.1f}x media")
            log(f"   Diagonal: pendiente={t.get('pendiente_diagonal', 0):.4f}, R²={t.get('r2_diagonal', 0):.2f}, puntos={t.get('n_puntos_diagonal', 0)}")
            log(f"   Tendencia: {t.get('tendencia', 'N/A')}")
            log(f"")

    log("="*60)
    log("Backtest finalizado")
    log("="*60)

if __name__ == "__main__":
    main()
