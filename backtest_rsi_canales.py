"""
BOT RSI CANALES - BACKTEST v3
==============================
Fixes:
- Datos 1h ampliados a 2200+ velas (3 meses)
- Filtros relajados o removibles
- Debug mode para ver por qué se rechazan señales
"""

import pandas as pd
import numpy as np
import ccxt
import os
from datetime import datetime
from scipy import stats

# CONFIGURACION
CONFIG = {
    'symbol': os.environ.get('BITGET_SYMBOL', 'XRP/USDT'),
    'balance': float(os.environ.get('BALANCE', '1000')),
    'leverage': int(os.environ.get('LEVERAGE', '5')),
    'risk_per_trade': float(os.environ.get('RISK_PER_TRADE', '0.02')),
    'sl_pct': float(os.environ.get('SL_PCT', '0.015')),
    'tp_pct': float(os.environ.get('TP_PCT', '0.03')),
    'min_r2_canal': float(os.environ.get('MIN_R2_CANAL', '0.05')),      # Muy relajado
    'min_r2_diagonal': float(os.environ.get('MIN_R2_DIAGONAL', '0.10')),  # Muy relajado
    'umbral_rebote': float(os.environ.get('UMBRAL_REBOTE', '1.0')),      # Sin margen extra
    'trailing_stop_pct': 0.01,
    'breakeven_trigger': 0.015,
    'time_exit_max': 30,
    'debug': os.environ.get('DEBUG', 'false').lower() == 'true',  # Modo debug
    'use_filters': os.environ.get('USE_FILTERS', 'true').lower() == 'true',  # Puedes desactivar filtros
}

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def fetch_data(exchange, symbol, timeframe, limit=1000):
    """Descargar datos OHLCV de Bitget"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        log(f"Error descargando {timeframe}: {e}")
        return None

def calcular_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calcular_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def detectar_canal_4h(rsi_values, ventana_pivot=3, min_puntos=3,
                       pendiente_max=-0.005, diff_max=0.5, r2_min=0.05):
    """
    Detectar canal descendente en RSI 4h.
    CORRECCION: Más permisivo para encontrar más señales
    """
    rsi = np.array(rsi_values)
    n = len(rsi)
    
    techos_idx, techos_val = [], []
    suelos_idx, suelos_val = [], []
    
    for i in range(ventana_pivot, n - ventana_pivot):
        ventana = rsi[i-ventana_pivot:i+ventana_pivot+1]
        if rsi[i] == ventana.max() and rsi[i] > 40:  # Bajado de 45 a 40
            techos_idx.append(i)
            techos_val.append(rsi[i])
        if rsi[i] == ventana.min() and rsi[i] < 65:  # Subido de 60 a 65
            suelos_idx.append(i)
            suelos_val.append(rsi[i])
    
    if len(techos_idx) < min_puntos or len(suelos_idx) < min_puntos:
        return None, f"Pocos pivots (techos={len(techos_idx)}, suelos={len(suelos_idx)})"
    
    n_techos = min(8, len(techos_idx))
    n_suelos = min(8, len(suelos_idx))
    
    x_t = np.array(techos_idx[-n_techos:])
    y_t = np.array(techos_val[-n_techos:])
    x_s = np.array(suelos_idx[-n_suelos:])
    y_s = np.array(suelos_val[-n_suelos:])
    
    m_t, b_t, r_t, p_t, se_t = stats.linregress(x_t, y_t)
    m_s, b_s, r_s, p_s, se_s = stats.linregress(x_s, y_s)
    
    # Canal descendente: techos bajando
    if m_t > pendiente_max:
        return None, f"Techo no descendente ({m_t:.4f} > {pendiente_max})"
    
    # Suelos pueden subir ligeramente (canal convergente)
    if m_s > 0.02:
        return None, f"Suelo subiendo demasiado ({m_s:.4f})"
    
    if abs(m_t - m_s) > diff_max:
        return None, f"Lineas no paralelas (diff={abs(m_t-m_s):.4f})"
    
    if r_t**2 < r2_min:
        return None, f"Correlacion baja (R2={r_t**2:.3f})"
    
    return {
        'techo_m': m_t, 'techo_b': b_t,
        'suelo_m': m_s, 'suelo_b': b_s,
        'techo_r2': r_t**2, 'suelo_r2': r_s**2,
        'techos_idx': techos_idx, 'techos_val': techos_val,
        'suelos_idx': suelos_idx, 'suelos_val': suelos_val
    }, "OK"

def detectar_rebote_4h(rsi_values, canal, umbral_zona=1.0):
    if not canal:
        return False, None
    
    x = len(rsi_values) - 1
    suelo = canal['suelo_m'] * x + canal['suelo_b']
    techo = canal['techo_m'] * x + canal['techo_b']
    rsi_actual = rsi_values[-1]
    
    # Más permisivo: cerca del suelo O simplemente RSI < 42
    en_zona = rsi_actual <= suelo * umbral_zona and rsi_actual < 48  # Subido de 45 a 48
    en_zona_baja = rsi_actual < 42  # Subido de 40 a 42
    
    return en_zona or en_zona_baja, {
        'suelo': suelo, 'techo': techo,
        'rsi_actual': rsi_actual,
        'ancho_canal': techo - suelo
    }

def detectar_diagonal_1h(rsi_values, ventana_pivot=5, min_puntos=3,
                          pendiente_max=0.01, r2_min=0.10):  # pendiente_max positivo = más permisivo
    rsi = np.array(rsi_values)
    n = len(rsi)
    
    maximos_idx, maximos_val = [], []
    for i in range(ventana_pivot, n - ventana_pivot):
        ventana = rsi[i-ventana_pivot:i+ventana_pivot+1]
        if rsi[i] == ventana.max() and rsi[i] > 30:  # Bajado de 35 a 30
            if not maximos_idx or rsi[i] <= maximos_val[-1] * 1.15:  # Más permisivo (1.15 vs 1.08)
                maximos_idx.append(i)
                maximos_val.append(rsi[i])
    
    if len(maximos_idx) < min_puntos:
        return None, f"Pocos maximos ({len(maximos_idx)})"
    
    for n_usar in range(min_puntos, min(20, len(maximos_idx)) + 1):  # Hasta 20 puntos
        x = np.array(maximos_idx[-n_usar:])
        y = np.array(maximos_val[-n_usar:])
        m, b, r, p, se = stats.linregress(x, y)
        
        # Aceptar líneas casi planas o ligeramente descendentes
        if m <= pendiente_max and r**2 >= r2_min:
            return {
                'm': m, 'b': b, 'r2': r**2,
                'maximos_idx': maximos_idx, 'maximos_val': maximos_val,
                'n_usados': n_usar
            }, "OK"
    
    # Fallback muy permisivo
    x = np.array(maximos_idx[-min_puntos:])
    y = np.array(maximos_val[-min_puntos:])
    m, b, r, p, se = stats.linregress(x, y)
    if m <= 0.08:
        return {
            'm': m, 'b': b, 'r2': r**2,
            'maximos_idx': maximos_idx, 'maximos_val': maximos_val,
            'n_usados': min_puntos
        }, "OK (fallback)"
    
    return None, f"Tendencia muy ascendente ({m:.4f})"

def detectar_ruptura_1h(rsi_values, diagonal, umbral=0.0):  # Umbral 0 = más permisivo
    if not diagonal:
        return False, None
    
    n = len(rsi_values)
    val_diag_2 = diagonal['m'] * (n-2) + diagonal['b']
    val_diag_1 = diagonal['m'] * (n-1) + diagonal['b']
    
    rsi_2 = rsi_values[-2]
    rsi_1 = rsi_values[-1]
    
    # Ruptura: RSI cruza por encima de la diagonal
    ruptura = rsi_2 <= val_diag_2 and rsi_1 > val_diag_1
    
    if ruptura:
        return True, {
            'rsi_antes': rsi_2, 'rsi_despues': rsi_1,
            'diag_antes': val_diag_2, 'diag_despues': val_diag_1,
            'diferencia': rsi_1 - val_diag_1
        }
    
    # Ruptura previa confirmada
    val_diag_3 = diagonal['m'] * (n-3) + diagonal['b']
    if rsi_1 > val_diag_1 and len(rsi_values) > 3 and rsi_values[-3] > val_diag_3:
        return True, {
            'rsi_antes': rsi_values[-3], 'rsi_despues': rsi_1,
            'diag_antes': val_diag_3, 'diag_despues': val_diag_1,
            'diferencia': rsi_1 - val_diag_1,
            'nota': 'Ruptura previa confirmada'
        }
    
    return False, None

def simular_trade(df_1h, idx_entrada, entry_price):
    """
    Simulación simplificada con trailing + breakeven
    """
    if idx_entrada >= len(df_1h) - 1:
        return None
    
    sl_price = entry_price * (1 - CONFIG['sl_pct'])
    tp_price = entry_price * (1 + CONFIG['tp_pct'])
    
    # SL más ajustado
    min_recent = df_1h['low'].iloc[max(0, idx_entrada-5):idx_entrada].min()
    sl_final = max(sl_price, min_recent * 0.999)
    
    max_price = entry_price
    breakeven_activado = False
    sl_trailing = sl_final
    
    max_velas = CONFIG['time_exit_max']
    
    for i in range(1, min(max_velas, len(df_1h) - idx_entrada)):
        vela = df_1h.iloc[idx_entrada + i]
        vela_low = vela['low']
        vela_high = vela['high']
        
        if vela_high > max_price:
            max_price = vela_high
        
        ganancia_pct = (max_price - entry_price) / entry_price
        if ganancia_pct >= CONFIG['breakeven_trigger'] and not breakeven_activado:
            breakeven_activado = True
            sl_trailing = entry_price * 1.001
        
        if breakeven_activado:
            nuevo_sl = max_price * (1 - CONFIG['trailing_stop_pct'])
            if nuevo_sl > sl_trailing:
                sl_trailing = nuevo_sl
        
        if vela_low <= sl_trailing:
            pnl_pct = (sl_trailing - entry_price) / entry_price * 100
            return {
                'resultado': 'SL',
                'exit_price': sl_trailing,
                'pnl_pct': pnl_pct,
                'velas_duracion': i,
                'fecha_salida': df_1h.index[idx_entrada + i],
                'tipo_salida': 'trailing' if breakeven_activado else 'sl_fijo'
            }
        
        if vela_high >= tp_price:
            return {
                'resultado': 'TP',
                'exit_price': tp_price,
                'pnl_pct': CONFIG['tp_pct'] * 100,
                'velas_duracion': i,
                'fecha_salida': df_1h.index[idx_entrada + i],
                'tipo_salida': 'tp_fijo'
            }
    
    idx_salida = idx_entrada + min(max_velas, len(df_1h) - idx_entrada - 1)
    exit_price = df_1h['close'].iloc[idx_salida]
    pnl_pct = (exit_price - entry_price) / entry_price * 100
    
    return {
        'resultado': 'TIME_EXIT',
        'exit_price': exit_price,
        'pnl_pct': pnl_pct,
        'velas_duracion': min(max_velas, len(df_1h) - idx_entrada - 1),
        'fecha_salida': df_1h.index[idx_salida],
        'tipo_salida': 'time_exit'
    }

def backtest_completo(df_4h, df_1h):
    log("="*60)
    log("INICIANDO BACKTEST v3")
    log("="*60)
    
    trades = []
    rechazos = {
        'canal': 0, 'rebote': 0, 'diagonal': 0, 'ruptura': 0,
        'filtro_vol': 0, 'filtro_tendencia': 0, 'datos_1h': 0
    }
    
    rsi_4h = df_4h['rsi'].values
    closes_4h = df_4h['close'].values
    rsi_1h = df_1h['rsi'].values
    closes_1h = df_1h['close'].values
    
    # Precalcular indicadores
    df_4h['ema20'] = df_4h['close'].ewm(span=20).mean()
    df_1h['atr'] = calcular_atr(df_1h)
    df_1h['atr_avg'] = df_1h['atr'].rolling(50).mean()
    df_1h['vol_avg'] = df_1h['volume'].rolling(20).mean()
    
    ventana_min = 50
    
    log(f"Total velas 4h: {len(df_4h)}")
    log(f"Total velas 1h: {len(df_1h)}")
    log(f"Periodo 4h: {df_4h.index[0]} -> {df_4h.index[-1]}")
    log(f"Periodo 1h: {df_1h.index[0]} -> {df_1h.index[-1]}")
    log(f"Analizando...")
    
    for i in range(ventana_min, len(df_4h) - 1):
        rsi_4h_window =
