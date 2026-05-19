"""
BOT RSI CANALES - BACKTEST COMPLETO
====================================
Revisa todo el histórico y cuenta señales
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
    'min_r2_canal': float(os.environ.get('MIN_R2_CANAL', '0.3')),
    'min_r2_diagonal': float(os.environ.get('MIN_R2_DIAGONAL', '0.3')),
    'umbral_rebote': float(os.environ.get('UMBRAL_REBOTE', '1.15')),
}

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def fetch_data(exchange, symbol, timeframe, since=None, limit=1000):
    """Descargar datos OHLCV de Bitget"""
    try:
        if since:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        else:
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

def detectar_canal_4h(rsi_values, ventana_pivot=3, min_puntos=3,
                       pendiente_max=-0.01, diff_max=0.3, r2_min=0.3):
    rsi = np.array(rsi_values)
    n = len(rsi)
    
    techos_idx, techos_val = [], []
    suelos_idx, suelos_val = [], []
    
    for i in range(ventana_pivot, n - ventana_pivot):
        ventana = rsi[i-ventana_pivot:i+ventana_pivot+1]
        if rsi[i] == ventana.max() and rsi[i] > 45:
            techos_idx.append(i)
            techos_val.append(rsi[i])
        if rsi[i] == ventana.min() and rsi[i] < 60:
            suelos_idx.append(i)
            suelos_val.append(rsi[i])
    
    if len(techos_idx) < min_puntos or len(suelos_idx) < min_puntos:
        return None, f"Pocos pivots (techos={len(techos_idx)}, suelos={len(suelos_idx)})"
    
    x_t = np.array(techos_idx[-8:])
    y_t = np.array(techos_val[-8:])
    x_s = np.array(suelos_idx[-8:])
    y_s = np.array(suelos_val[-8:])
    
    m_t, b_t, r_t, p_t, se_t = stats.linregress(x_t, y_t)
    m_s, b_s, r_s, p_s, se_s = stats.linregress(x_s, y_s)
    
    if m_t > pendiente_max:
        return None, f"Techo no descendente ({m_t:.4f} > {pendiente_max})"
    
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

def detectar_rebote_4h(rsi_values, canal, umbral_zona=1.15):
    if not canal:
        return False, None
    
    x = len(rsi_values) - 1
    suelo = canal['suelo_m'] * x + canal['suelo_b']
    techo = canal['techo_m'] * x + canal['techo_b']
    rsi_actual = rsi_values[-1]
    
    en_zona = rsi_actual < suelo * umbral_zona and rsi_actual < 45
    en_zona_baja = rsi_actual < 40
    
    return en_zona or en_zona_baja, {
        'suelo': suelo, 'techo': techo,
        'rsi_actual': rsi_actual
    }

def detectar_diagonal_1h(rsi_values, ventana_pivot=5, min_puntos=3,
                          pendiente_max=-0.01, r2_min=0.3):
    rsi = np.array(rsi_values)
    n = len(rsi)
    
    maximos_idx, maximos_val = [], []
    for i in range(ventana_pivot, n - ventana_pivot):
        ventana = rsi[i-ventana_pivot:i+ventana_pivot+1]
        if rsi[i] == ventana.max() and rsi[i] > 35:
            if not maximos_idx or rsi[i] <= maximos_val[-1] * 1.08:
                maximos_idx.append(i)
                maximos_val.append(rsi[i])
    
    if len(maximos_idx) < min_puntos:
        return None, f"Pocos maximos ({len(maximos_idx)})"
    
    for n_usar in range(min_puntos, min(15, len(maximos_idx)) + 1):
        x = np.array(maximos_idx[-n_usar:])
        y = np.array(maximos_val[-n_usar:])
        m, b, r, p, se = stats.linregress(x, y)
        
        if m <= pendiente_max and r**2 >= r2_min:
            return {
                'm': m, 'b': b, 'r2': r**2,
                'maximos_idx': maximos_idx, 'maximos_val': maximos_val,
                'n_usados': n_usar
            }, "OK"
    
    x = np.array(maximos_idx[-min_puntos:])
    y = np.array(maximos_val[-min_puntos:])
    m, b, r, p, se = stats.linregress(x, y)
    if m <= 0.05:
        return {
            'm': m, 'b': b, 'r2': r**2,
            'maximos_idx': maximos_idx, 'maximos_val': maximos_val,
            'n_usados': min_puntos
        }, "OK (fallback)"
    
    return None, f"Tendencia ascendente ({m:.4f})"

def detectar_ruptura_1h(rsi_values, diagonal, umbral=0.02):
    if not diagonal:
        return False, None
    
    n = len(rsi_values)
    val_diag_2 = diagonal['m'] * (n-2) + diagonal['b']
    val_diag_1 = diagonal['m'] * (n-1) + diagonal['b']
    
    rsi_2 = rsi_values[-2]
    rsi_1 = rsi_values[-1]
    
    ruptura = rsi_2 <= val_diag_2 and rsi_1 > val_diag_1
    
    if ruptura:
        return True, {
            'rsi_antes': rsi_2, 'rsi_despues': rsi_1,
            'diag_antes': val_diag_2, 'diag_despues': val_diag_1,
            'diferencia': rsi_1 - val_diag_1
        }
    
    val_diag_3 = diagonal['m'] * (n-3) + diagonal['b']
    if rsi_1 > val_diag_1 and len(rsi_values) > 3 and rsi_values[-3] > val_diag_3:
        return True, {
            'rsi_antes': rsi_values[-3], 'rsi_despues': rsi_1,
            'diag_antes': val_diag_3, 'diag_despues': val_diag_1,
            'diferencia': rsi_1 - val_diag_1,
            'nota': 'Ruptura previa confirmada'
        }
    
    return False, None

def backtest_completo(df_4h, df_1h):
    """
    Backtest: revisar cada vela 4h del pasado para ver si hubo señal
    """
    log("="*60)
    log("INICIANDO BACKTEST COMPLETO")
    log("="*60)
    
    señales = []
    rsi_4h = df_4h['rsi'].values
    rsi_1h = df_1h['rsi'].values
    closes_4h = df_4h['close'].values
    closes_1h = df_1h['close'].values
    
    # Ventana mínima para detectar canal: 50 velas 4h
    ventana_min = 50
    
    log(f"Total velas 4h: {len(df_4h)}")
    log(f"Total velas 1h: {len(df_1h)}")
    log(f"Analizando desde vela {ventana_min} hasta {len(df_4h)-1}...")
    
    # Para cada vela 4h, mirar hacia atrás y ver si había setup
    for i in range(ventana_min, len(df_4h) - 1):
        # RSI 4h hasta esta vela
        rsi_4h_window = rsi_4h[:i+1]
        
        # Detectar canal
        canal, msg = detectar_canal_4h(rsi_4h_window)
        if not canal:
            continue
        
        # Verificar rebote
        rebote, info_4h = detectar_rebote_4h(rsi_4h_window, canal)
        if not rebote:
            continue
        
        # Encontrar ventana 1h correspondiente a esta vela 4h
        # Cada vela 4h = 4 velas 1h
        idx_1h = min((i + 1) * 4, len(rsi_1h) - 1)
        if idx_1h < 50:
            continue
        
        rsi_1h_window = rsi_1h[:idx_1h+1]
        
        # Detectar diagonal
        diagonal, msg = detectar_diagonal_1h(rsi_1h_window)
        if not diagonal:
            continue
        
        # Verificar ruptura
        ruptura, info_1h = detectar_ruptura_1h(rsi_1h_window, diagonal)
        if not ruptura:
            continue
        
        # SEÑAL ENCONTRADA EN EL PASADO
        fecha = df_4h.index[i]
        entry = closes_1h[idx_1h]
        
        señales.append({
            'fecha': fecha,
            'idx_4h': i,
            'idx_1h': idx_1h,
            'entry': entry,
            'rsi_4h': info_4h['rsi_actual'],
            'rsi_1h': info_1h['rsi_despues'],
            'techo_canal': info_4h['techo'],
            'suelo_canal': info_4h['suelo'],
        })
    
    return señales

def main():
    log("="*60)
    log("BOT RSI CANALES - BACKTEST 3 MESES")
    log(f"Par: {CONFIG['symbol']}")
    log("="*60)
    
    # Conectar a Bitget
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
    
    # Descargar datos amplios (3 meses)
    log("Descargando datos históricos...")
    
    # 4h: ~3 meses = ~540 velas
    df_4h = fetch_data(exchange, CONFIG['symbol'], '4h', limit=600)
    # 1h: ~3 meses = ~2160 velas  
    df_1h = fetch_data(exchange, CONFIG['symbol'], '1h', limit=2200)
    
    if df_4h is None or df_1h is None:
        log("Error descargando datos")
        return
    
    log(f"4h: {len(df_4h)} velas | {df_4h.index[0]} -> {df_4h.index[-1]}")
    log(f"1h: {len(df_1h)} velas | {df_1h.index[0]} -> {df_1h.index[-1]}")
    
    # Calcular RSI
    df_4h['rsi'] = calcular_rsi(df_4h['close'])
    df_1h['rsi'] = calcular_rsi(df_1h['close'])
    
    # BACKTEST
    señales = backtest_completo(df_4h, df_1h)
    
    # RESULTADOS
    log("")
    log("="*60)
    log("RESULTADOS DEL BACKTEST")
    log("="*60)
    log(f"Periodo analizado: {df_4h.index[0].strftime('%Y-%m-%d')} -> {df_4h.index[-1].strftime('%Y-%m-%d')}")
    log(f"Total velas 4h: {len(df_4h)}")
    log(f"Total velas 1h: {len(df_1h)}")
    log(f"")
    log(f"SEÑALES ENCONTRADAS: {len(señales)}")
    log(f"")
    
    if len(señales) == 0:
        log("NO HUBO NINGUNA SEÑAL EN ESTE PERIODO")
        log("")
        log("Posibles causas:")
        log("- XRP no formó canales descendentes claros en 4h")
        log("- El mercado estuvo en tendencia lateral sin estructura")
        log("- Los parámetros son muy estrictos")
        log("")
        log("Recomendación: Probar con otros pares (ADA, SOL, DOGE)")
    else:
        log("DETALLE DE SEÑALES:")
        log("-" * 60)
        for i, s in enumerate(señales, 1):
            log(f"#{i} | {s['fecha'].strftime('%Y-%m-%d %H:%M')}")
            log(f"   Entrada: ${s['entry']:.4f}")
            log(f"   RSI 4h: {s['rsi_4h']:.1f} (suelo: {s['suelo_canal']:.1f}, techo: {s['techo_canal']:.1f})")
            log(f"   RSI 1h: {s['rsi_1h']:.1f}")
            log(f"")
        
        log("="*60)
        log(f"Frecuencia: 1 señal cada {len(df_4h)//len(señales)} velas 4h (~{len(df_4h)//len(señales)//6} días)")
    
    log("="*60)
    log("Backtest finalizado")
    log("="*60)

if __name__ == "__main__":
    main()
