"""
BOT RSI CANALES - BACKTEST v3.2.2
==================================
Correccion: RSI_DIVERGENCIA solo genera 1 señal por divergencia detectada.
Se registra el indice del segundo pivot para evitar señales repetidas.

Estrategias:
1. EMA_CRUCE (con filtro de divergencia RSI opcional)
2. RSI_CANALES (con filtro de divergencia RSI opcional)
3. RSI_DIVERGENCIA (1 señal por divergencia, con cooldown)
"""

import pandas as pd
import numpy as np
import ccxt
import os
import json
from datetime import datetime
from scipy import stats
from scipy.signal import argrelextrema

CONFIG = {
    'symbol': os.environ.get('BITGET_SYMBOL', 'ADA/USDT'),
    'timeframe': os.environ.get('TIMEFRAME', '4h'),
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
    'rsi_period': 14,
    'pivot_ventana': 2,
    'min_puntos_diagonal': 2,
    'r2_min_diagonal': 0.10,
    'umbral_ruptura': 0.8,
    'zona_sobreventa': 45,

    'ema_rapida': 9,
    'ema_lenta': 21,
    'ema_tendencia': 50,

    # DIVERGENCIA RSI
    'div_order': int(os.environ.get('DIV_ORDER', '5')),
    'div_lookback': int(os.environ.get('DIV_LOOKBACK', '15')),
    'div_rsi_max': float(os.environ.get('DIV_RSI_MAX', '35')),
    'div_usar_como_filtro': os.environ.get('DIV_USAR_FILTRO', 'true').lower() == 'true',
    'div_estrategia_independiente': os.environ.get('DIV_ESTRATEGIA_IND', 'true').lower() == 'true',
}

LOG_FILE = None

def init_logging():
    global LOG_FILE
    log_path = os.environ.get('LOG_FILE', 'backtest_v3.2.2.txt')
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

# ============================================================================
# DATOS
# ============================================================================

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

# ============================================================================
# INDICADORES
# ============================================================================

def calcular_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detectar_tendencia(df, idx):
    if idx < 50:
        return 'N/A', '⚪'
    precio = df['close'].iloc[idx]
    ema9 = df['ema9'].iloc[idx] if 'ema9' in df.columns else None
    ema21 = df['ema21'].iloc[idx] if 'ema21' in df.columns else None
    ema50 = df['ema50'].iloc[idx] if 'ema50' in df.columns else None
    if ema50 is None:
        return 'N/A', '⚪'
    if precio > ema50 * 1.02:
        if ema9 and ema21 and ema9 > ema21:
            return 'Alcista fuerte', '🟢'
        return 'Alcista', '🟩'
    elif precio < ema50 * 0.98:
        if ema9 and ema21 and ema9 < ema21:
            return 'Bajista fuerte', '🔴'
        return 'Bajista', '🟥'
    else:
        return 'Lateral', '⚪'

# ============================================================================
# DIVERGENCIA RSI - v3.2.2
# ============================================================================

def detectar_pivots(serie, order=3):
    valores = serie.values
    max_idx = argrelextrema(valores, np.greater_equal, order=order)[0]
    min_idx = argrelextrema(valores, np.less_equal, order=order)[0]
    return max_idx, min_idx

def detectar_divergencia_alcista(df, rsi_values, order=3, lookback=25, rsi_max=40):
    """
    Detecta TODAS las divergencias alcistas en el dataframe.
    Retorna la más reciente y una lista de todas las detectadas.
    """
    n = len(df)
    if n < lookback + order * 2 + 5:
        return False, None, []

    precio = df['close'].values
    rsi = rsi_values

    price_max_idx, price_min_idx = detectar_pivots(pd.Series(precio), order=order)
    rsi_max_idx, rsi_min_idx = detectar_pivots(pd.Series(rsi), order=order)

    if len(price_min_idx) < 2 or len(rsi_min_idx) < 2:
        return False, None, []

    divergencias = []

    for i in range(len(price_min_idx) - 1, 0, -1):
        idx2 = price_min_idx[i]
        idx1 = price_min_idx[i - 1]

        if (idx2 - idx1) > lookback:
            continue

        precio1 = precio[idx1]
        precio2 = precio[idx2]

        if precio2 >= precio1 * 0.999:
            continue

        rsi_idx1 = None
        rsi_idx2 = None

        for r_idx in rsi_min_idx:
            if abs(r_idx - idx1) <= order + 2:
                rsi_idx1 = r_idx
            if abs(r_idx - idx2) <= order + 2:
                rsi_idx2 = r_idx

        if rsi_idx1 is None or rsi_idx2 is None:
            continue

        rsi1 = rsi[rsi_idx1]
        rsi2 = rsi[rsi_idx2]

        if rsi2 <= rsi1 * 1.001:
            continue

        if rsi2 > rsi_max:
            continue

        divergencias.append({
            'precio_idx1': int(idx1), 'precio_idx2': int(idx2),
            'precio1': float(precio1), 'precio2': float(precio2),
            'rsi_idx1': int(rsi_idx1), 'rsi_idx2': int(rsi_idx2),
            'rsi1': float(rsi1), 'rsi2': float(rsi2),
            'fuerza': float((rsi2 - rsi1) / (precio1 - precio2)) if (precio1 - precio2) > 0 else 0
        })

    if divergencias:
        return True, divergencias[-1], divergencias

    return False, None, []

# ============================================================================
# ESTRATEGIAS
# ============================================================================

def check_ema_cruce(df, idx, config, divergencias_usadas=None):
    if idx < config['ema_lenta'] + 5:
        return False, "Datos insuficientes", None
    ema9 = df['ema9'].values
    ema21 = df['ema21'].values
    if not (ema9[idx-1] < ema21[idx-1] and ema9[idx] > ema21[idx]):
        return False, "Sin cruce EMA", None
    volumen_actual = df['volume'].iloc[idx]
    volumen_media = df['volume'].iloc[max(0, idx - config['volumen_lookback']):idx].mean()
    ratio_volumen = volumen_actual / volumen_media if volumen_media > 0 else 0
    if ratio_volumen < config['volumen_min_ratio']:
        return False, f"Volumen insuficiente ({ratio_volumen:.1f}x)", None
    info_div = None
    if config['div_usar_como_filtro']:
        hay_div, info_div, _ = detectar_divergencia_alcista(
            df.iloc[:idx+1], df['rsi'].values[:idx+1],
            order=config['div_order'], lookback=config['div_lookback'], rsi_max=config['div_rsi_max']
        )
        if not hay_div:
            return False, "Sin divergencia RSI confirmada", None
    return True, {"ratio_volumen": ratio_volumen, "divergencia": info_div}, info_div

def detectar_diagonal_maximos_rsi(rsi_values, ventana_pivot=2, min_puntos=2, pendiente_max=-0.01, r2_min=0.10):
    rsi = np.array(rsi_values)
    n = len(rsi)
    if n < ventana_pivot * 2 + 1:
        return None
    maximos_idx = []
    maximos_val = []
    for i in range(ventana_pivot, n - ventana_pivot):
        ventana = rsi[i - ventana_pivot:i + ventana_pivot + 1]
        if rsi[i] == ventana.max() and rsi[i] > 25:
            if not maximos_idx or (i - maximos_idx[-1]) >= 3:
                maximos_idx.append(i)
                maximos_val.append(rsi[i])
    if len(maximos_idx) < min_puntos:
        return None
    best_diagonal = None
    best_score = -999
    for n_usar in range(min_puntos, min(15, len(maximos_idx)) + 1):
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
                'm': m, 'b': b, 'r2': r2, 'n_usados': n_usar,
                'ultimo_max_idx': x[-1], 'ultimo_max_val': y[-1],
                'primero_max_idx': x[0], 'primero_max_val': y[0],
            }
    if best_diagonal is None and len(maximos_idx) >= 2:
        x = np.array(maximos_idx[-2:])
        y = np.array(maximos_val[-2:])
        m, b, r, p, se = stats.linregress(x, y)
        if m < 0:
            return {'m': m, 'b': b, 'r2': r**2, 'n_usados': 2,
                    'ultimo_max_idx': x[-1], 'ultimo_max_val': y[-1],
                    'primero_max_idx': x[0], 'primero_max_val': y[0]}
    return best_diagonal

def detectar_ruptura_diagonal_long(rsi_values, diagonal, df, idx, umbral_ruptura=0.8, volumen_ratio_min=1.2, volumen_lookback=10):
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
        'ratio_volumen': ratio_volumen, 'volumen_confirmado': volumen_confirmado,
        'pendiente_diagonal': diagonal['m'], 'r2_diagonal': diagonal['r2'],
        'n_puntos': diagonal['n_usados'], 'rsi_antes': rsi_2, 'rsi_despues': rsi_1,
        'diag_valor': val_diag_1,
    }

def check_rsi_canales(df, idx, config):
    rsi_values = df['rsi'].values[:idx + 1]
    rsi_actual = rsi_values[idx]
    if rsi_actual >= config['zona_sobreventa']:
        return False, f"RSI {rsi_actual:.1f} >= {config['zona_sobreventa']}", None
    diagonal = detectar_diagonal_maximos_rsi(
        rsi_values, ventana_pivot=config['pivot_ventana'],
        min_puntos=config['min_puntos_diagonal'], pendiente_max=-0.01,
        r2_min=config['r2_min_diagonal']
    )
    if not diagonal:
        return False, "No hay diagonal", None
    ruptura, info = detectar_ruptura_diagonal_long(
        rsi_values, diagonal, df, idx,
        umbral_ruptura=config['umbral_ruptura'],
        volumen_ratio_min=config['volumen_min_ratio'],
        volumen_lookback=config['volumen_lookback']
    )
    if not ruptura or not info or not info.get('volumen_confirmado', False):
        return False, "Sin ruptura o volumen", None
    info_div = None
    if config['div_usar_como_filtro']:
        hay_div, info_div, _ = detectar_divergencia_alcista(
            df.iloc[:idx+1], df['rsi'].values[:idx+1],
            order=config['div_order'], lookback=config['div_lookback'], rsi_max=config['div_rsi_max']
        )
        if not hay_div:
            return False, "Sin divergencia RSI confirmada", None
    return True, info, info_div

def check_rsi_divergencia(df, idx, config, divergencias_usadas):
    """
    Estrategia RSI_DIVERGENCIA v3.2.2:
    - Detecta todas las divergencias en el dataframe hasta idx
    - Solo genera señal si hay una divergencia NUEVA (no usada antes)
    - La señal se genera en la vela donde se completa el segundo pivot
    """
    if idx < 50:
        return False, "Datos insuficientes", None

    rsi_values = df['rsi'].values[:idx + 1]
    precio_values = df['close'].values[:idx + 1]

    hay_div, info_div, todas_divs = detectar_divergencia_alcista(
        df.iloc[:idx+1], rsi_values,
        order=config['div_order'], lookback=config['div_lookback'], rsi_max=config['div_rsi_max']
    )

    if not hay_div or not todas_divs:
        return False, "Sin divergencia alcista", None

    # Buscar una divergencia NUEVA (cuyo segundo pivot no haya sido usado)
    div_nueva = None
    for div in reversed(todas_divs):  # De la más reciente a la más antigua
        pivot_key = div['precio_idx2']  # Usamos el índice del segundo pivot como ID único
        if pivot_key not in divergencias_usadas:
            div_nueva = div
            break

    if div_nueva is None:
        return False, "Divergencia ya utilizada", None

    # IMPORTANTE: Solo generar señal si estamos en la vela del segundo pivot o justo después
    # (para evitar señales retardadas)
    if idx < div_nueva['precio_idx2'] or idx > div_nueva['precio_idx2'] + 2:
        return False, f"Fuera de ventana de señal (pivot en {div_nueva['precio_idx2']}, actual {idx})", None

    volumen_actual = df['volume'].iloc[idx]
    volumen_media = df['volume'].iloc[max(0, idx - config['volumen_lookback']):idx].mean()
    ratio_volumen = volumen_actual / volumen_media if volumen_media > 0 else 0

    if ratio_volumen < config['volumen_min_ratio']:
        return False, f"Volumen insuficiente ({ratio_volumen:.1f}x)", None

    return True, {"ratio_volumen": ratio_volumen, "divergencia": div_nueva}, div_nueva

# ============================================================================
# SIMULACIÓN
# ============================================================================

def simular_trade(df, idx_entrada, entry_price, config):
    if idx_entrada >= len(df) - 1:
        return None
    leverage = config['leverage']
    sl_price = entry_price * (1 - config['sl_pct'])
    tp_price = entry_price * (1 + config['tp_pct'])
    min_recent = df['low'].iloc[max(0, idx_entrada - 3):idx_entrada].min()
    sl_final = max(sl_price, min_recent * 0.998)
    max_price = entry_price
    breakeven_activado = False
    sl_trailing = sl_final
    for i in range(1, min(config['time_exit_max'], len(df) - idx_entrada)):
        vela = df.iloc[idx_entrada + i]
        if vela['high'] > max_price:
            max_price = vela['high']
        ganancia_pct = (max_price - entry_price) / entry_price
        if ganancia_pct >= config['breakeven_trigger'] and not breakeven_activado:
            breakeven_activado = True
            sl_trailing = entry_price * 1.001
        if breakeven_activado:
            nuevo_sl = max_price * (1 - config['trailing_stop_pct'])
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
            pnl_pct_raw = config['tp_pct'] * 100
            pnl_pct = pnl_pct_raw * leverage
            return {'resultado': 'TP', 'exit_price': tp_price, 'pnl_pct': pnl_pct,
                    'pnl_pct_raw': pnl_pct_raw, 'velas_duracion': i,
                    'fecha_salida': df.index[idx_entrada + i], 'tipo_salida': 'tp_fijo'}
    idx_salida = idx_entrada + min(config['time_exit_max'], len(df) - idx_entrada - 1)
    exit_price = df['close'].iloc[idx_salida]
    pnl_pct_raw = (exit_price - entry_price) / entry_price * 100
    pnl_pct = pnl_pct_raw * leverage
    return {'resultado': 'TIME_EXIT', 'exit_price': exit_price, 'pnl_pct': pnl_pct,
            'pnl_pct_raw': pnl_pct_raw, 'velas_duracion': min(config['time_exit_max'], len(df) - idx_entrada - 1),
            'fecha_salida': df.index[idx_salida], 'tipo_salida': 'time_exit'}

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def backtest_estrategia(df, nombre_estrategia, config):
    log(f"\\n{'='*60}")
    log(f"BACKTEST: {nombre_estrategia}")
    log(f"{'='*60}")

    trades = []
    rechazos = {'cooldown': 0}
    if nombre_estrategia == 'EMA_CRUCE':
        rechazos.update({'cruce': 0, 'volumen': 0, 'divergencia': 0})
    elif nombre_estrategia == 'RSI_CANALES':
        rechazos.update({'diagonal': 0, 'ruptura': 0, 'volumen': 0, 'zona_rsi': 0, 'divergencia': 0})
    else:
        rechazos.update({'sin_divergencia': 0, 'divergencia_usada': 0, 'fuera_ventana': 0, 'volumen': 0})

    closes = df['close'].values
    ventana_min = max(config['ema_lenta'] + 10, 50)
    ultimo_trade_fecha = None
    divergencias_usadas = set()  # v3.2.2: Registrar divergencias ya usadas

    for i in range(ventana_min, len(df) - 1):
        fecha = df.index[i]
        if ultimo_trade_fecha is not None:
            horas = (fecha - ultimo_trade_fecha).total_seconds() / 3600
            if horas < config['cooldown_horas']:
                rechazos['cooldown'] += 1
                continue

        if nombre_estrategia == 'EMA_CRUCE':
            ok, msg, info_div = check_ema_cruce(df, i, config)
        elif nombre_estrategia == 'RSI_CANALES':
            ok, msg, info_div = check_rsi_canales(df, i, config)
        else:
            ok, msg, info_div = check_rsi_divergencia(df, i, config, divergencias_usadas)

        if not ok:
            msg_lower = str(msg).lower()
            if "cruce" in msg_lower:
                rechazos['cruce'] = rechazos.get('cruce', 0) + 1
            elif "volumen" in msg_lower:
                rechazos['volumen'] = rechazos.get('volumen', 0) + 1
            elif "divergencia ya utilizada" in msg_lower:
                rechazos['divergencia_usada'] = rechazos.get('divergencia_usada', 0) + 1
            elif "fuera de ventana" in msg_lower:
                rechazos['fuera_ventana'] = rechazos.get('fuera_ventana', 0) + 1
            elif "sin divergencia" in msg_lower or "divergencia" in msg_lower:
                if 'sin_divergencia' in rechazos:
                    rechazos['sin_divergencia'] += 1
                else:
                    rechazos['divergencia'] = rechazos.get('divergencia', 0) + 1
            elif "diagonal" in msg_lower:
                rechazos['diagonal'] = rechazos.get('diagonal', 0) + 1
            elif "ruptura" in msg_lower:
                rechazos['ruptura'] = rechazos.get('ruptura', 0) + 1
            elif "rsi" in msg_lower and ">=" in msg_lower:
                rechazos['zona_rsi'] = rechazos.get('zona_rsi', 0) + 1
            continue

        # Registrar divergencia como usada
        if info_div and 'precio_idx2' in info_div:
            divergencias_usadas.add(info_div['precio_idx2'])

        entry_price = closes[i]
        resultado = simular_trade(df, i, entry_price, config)

        if resultado:
            ultimo_trade_fecha = fecha
            trade = {
                'fecha_entrada': fecha, 'idx': i, 'entry': entry_price,
                'estrategia': nombre_estrategia, 'divergencia': info_div is not None,
                'divergencia_info': info_div, **msg, **resultado
            }
            trades.append(trade)

    log(f"Trades {nombre_estrategia}: {len(trades)}")
    for k, v in rechazos.items():
        if v > 0:
            log(f"  {k}: {v}")

    return trades

# ============================================================================
# MÉTRICAS
# ============================================================================

def calcular_metricas(trades, balance_inicial=1000, config=None):
    if not trades or config is None:
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
    leverage = config['leverage']
    for trade in trades:
        riesgo = balance * config['risk_per_trade']
        pnl_usd = riesgo * (trade['pnl_pct'] / (config['sl_pct'] * 100 * leverage))
        balance += pnl_usd
        if balance > max_balance:
            max_balance = balance
        drawdown = (max_balance - balance) / max_balance * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
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
        'max_drawdown': max_drawdown, 'profit_factor': profit_factor,
        'expectancy': expectancy, 'sharpe': sharpe, 'avg_win': avg_win, 'avg_loss': avg_loss,
        'trades_con_divergencia': len([t for t in trades if t.get('divergencia', False)]),
    }

def print_metricas(nombre, metricas, trades):
    if not metricas:
        log(f"\\n{nombre}: 0 trades")
        return
    log(f"\\n{'='*60}")
    log(f"{nombre}")
    log(f"{'='*60}")
    log(f"  Trades: {metricas['n_trades']} (G:{metricas['n_ganadores']} P:{metricas['n_perdedores']})")
    log(f"  Win Rate: {metricas['win_rate']:.1f}%")
    log(f"  Profit Factor: {metricas['profit_factor']:.2f}")
    log(f"  Expectancy: {metricas['expectancy']:.2f}%")
    log(f"  Sharpe: {metricas['sharpe']:.2f}")
    log(f"  Max DD: {metricas['max_drawdown']:.2f}%")
    log(f"  Return: {metricas['pnl_total_pct']:.2f}%")
    log(f"  Balance: ${metricas['balance_inicial']:.0f} -> ${metricas['balance_final']:.2f}")
    if metricas['trades_con_divergencia'] > 0:
        log(f"  Trades con divergencia: {metricas['trades_con_divergencia']}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    init_logging()
    try:
        log("="*60)
        log("BOT RSI CANALES - BACKTEST v3.2.2")
        log("CORRECCION: RSI_DIVERGENCIA solo 1 señal por divergencia")
        log(f"Par: {CONFIG['symbol']}")
        log(f"Timeframe: {CONFIG['timeframe']}")
        log(f"EMA: {CONFIG['ema_rapida']}/{CONFIG['ema_lenta']}")
        log(f"RSI: Periodo {CONFIG['rsi_period']}, Zona < {CONFIG['zona_sobreventa']}")
        log(f"Divergencia: order={CONFIG['div_order']}, lookback={CONFIG['div_lookback']}, rsi_max={CONFIG['div_rsi_max']}")
        log(f"Filtro divergencia: {CONFIG['div_usar_como_filtro']}")
        log(f"Estrategia divergencia: {CONFIG['div_estrategia_independiente']}")
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

        df['rsi'] = calcular_rsi(df['close'], period=CONFIG['rsi_period'])
        df['ema9'] = df['close'].ewm(span=CONFIG['ema_rapida'], adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=CONFIG['ema_lenta'], adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=CONFIG['ema_tendencia'], adjust=False).mean()

        estrategias = ['EMA_CRUCE', 'RSI_CANALES']
        if CONFIG['div_estrategia_independiente']:
            estrategias.append('RSI_DIVERGENCIA')

        todos_trades = {}
        todas_metricas = {}

        for estrategia in estrategias:
            trades = backtest_estrategia(df, estrategia, CONFIG)
            metricas = calcular_metricas(trades, CONFIG['balance'], CONFIG)
            todos_trades[estrategia] = trades
            todas_metricas[estrategia] = metricas
            print_metricas(estrategia, metricas, trades)

        log("\\n" + "="*60)
        log("RESULTADOS COMPARATIVOS v3.2.2")
        log("="*60)

        for estrategia in estrategias:
            m = todas_metricas[estrategia]
            if m:
                log(f"\\n{estrategia}:")
                log(f"  Trades: {m['n_trades']} | Win Rate: {m['win_rate']:.1f}% | PF: {m['profit_factor']:.2f}")
                log(f"  Return: {m['pnl_total_pct']:.2f}% | Max DD: {m['max_drawdown']:.2f}% | Sharpe: {m['sharpe']:.2f}")

        resultados_json = {
            'version': '3.2.2',
            'config': {k: str(v) for k, v in CONFIG.items()},
            'metricas': {k: v for k, v in todas_metricas.items() if v},
            'resumen': {
                'symbol': CONFIG['symbol'],
                'timeframe': CONFIG['timeframe'],
                'velas_totales': len(df),
                'fecha_inicio': str(df.index[0]),
                'fecha_fin': str(df.index[-1]),
            }
        }

        output_dir = os.environ.get('OUTPUT_DIR', 'resultados')
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, 'backtest_v3.2.2_resultados.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(resultados_json, f, indent=2, default=str)
        log(f"\\n✓ Resultados guardados en: {json_path}")

        mejor_estrategia = max(estrategias, key=lambda e: todas_metricas[e]['pnl_total_pct'] if todas_metricas[e] else -999)
        trades_mejor = todos_trades[mejor_estrategia]

        if trades_mejor:
            log(f"\\nDETALLE TRADES {mejor_estrategia} (mejor estrategia):")
            log("-" * 60)
            for i, t in enumerate(trades_mejor[:10], 1):
                emoji = "🟢" if t['pnl_pct'] > 0 else "🔴"
                div_mark = " [🔄DIV]" if t.get('divergencia') else ""
                log(f"{emoji} #{i}{div_mark} | {t['fecha_entrada'].strftime('%Y-%m-%d %H:%M')}")
                log(f"   Entrada: ${t['entry']:.4f} -> Salida: ${t['exit_price']:.4f}")
                log(f"   Resultado: {t['resultado']} | P&L {CONFIG['leverage']}x: {t['pnl_pct']:+.2f}%")
                if t.get('divergencia_info'):
                    d = t['divergencia_info']
                    log(f"   Divergencia: Precio ${d['precio1']:.4f}→${d['precio2']:.4f} | RSI {d['rsi1']:.1f}→{d['rsi2']:.1f}")
                log("")

        log("="*60)
        log("Backtest v3.2.2 finalizado")
        log("="*60)
    finally:
        close_logging()

if __name__ == "__main__":
    main()
