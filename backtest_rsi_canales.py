"""
BOT RSI CANALES - BACKTEST v8
==============================
Cambios radicales:
- Canal más permisivo: acepta techos planos o ligeramente descendentes
- Descarga de datos en batches para más histórico
- Filtro de rebote más flexible
- Opción de operar sin canal (solo diagonal + zona RSI)
"""

import pandas as pd
import numpy as np
import ccxt
import os
from datetime import datetime, timedelta
from scipy import stats

# CONFIGURACION
CONFIG = {
    'symbol': os.environ.get('BITGET_SYMBOL', 'XRP/USDT'),
    'balance': float(os.environ.get('BALANCE', '1000')),
    'leverage': int(os.environ.get('LEVERAGE', '5')),
    'risk_per_trade': float(os.environ.get('RISK_PER_TRADE', '0.02')),
    'sl_pct': float(os.environ.get('SL_PCT', '0.008')),
    'tp_pct': float(os.environ.get('TP_PCT', '0.015')),
    'trailing_stop_pct': 0.005,
    'breakeven_trigger': 0.008,
    'time_exit_max': 20,
    'cooldown_horas': 6,  # Reducido para más trades
    
    # NUEVO: Modo de operación
    # 'canal' = requiere canal + diagonal (original)
    # 'zona' = solo zona de sobreventa/sobrecompra + diagonal (más señales)
    # 'solo_diagonal' = solo ruptura de diagonal (máximas señales)
    'modo': os.environ.get('MODO', 'zona'),
    
    'direccion': os.environ.get('DIRECCION', 'both'),
    'debug': os.environ.get('DEBUG', 'false').lower() == 'true',
}

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def fetch_data_batches(exchange, symbol, timeframe, total_velas=2000):
    """
    Descargar datos en batches para obtener más histórico
    Bitget limita a ~1000 velas por request
    """
    all_data = []
    limit = 1000
    
    # Primera descarga: las últimas 1000 velas
    log(f"Descargando batch 1 ({limit} velas)...")
    df1 = fetch_data(exchange, symbol, timeframe, limit=limit)
    if df1 is None or len(df1) == 0:
        return None
    
    all_data.append(df1)
    
    # Si queremos más datos, descargar desde el inicio del primero
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
                    # Evitar duplicados
                    df2 = df2[df2.index < df1.index[0]]
                    if len(df2) > 0:
                        all_data.insert(0, df2)
                        log(f"Batch 2: {len(df2)} velas")
            except Exception as e:
                log(f"No se pudo descargar batch 2: {e}")
    
    # Combinar
    df_combined = pd.concat(all_data)
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    df_combined.sort_index(inplace=True)
    
    return df_combined

def get_millis(timeframe):
    """Convertir timeframe a milisegundos"""
    mapping = {'1m': 60000, '5m': 300000, '15m': 900000, '30m': 1800000,
               '1h': 3600000, '2h': 7200000, '4h': 14400000, '1d': 86400000}
    return mapping.get(timeframe, 3600000)

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

def calcular_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detectar_canal_rsi_permisivo(rsi_values, ventana_pivot=3, min_puntos=3,
                                  pendiente_max=0.02, diff_max=0.8, r2_min=0.03):
    """
    Canal mucho más permisivo:
    - Acepta techos planos o ligeramente descendentes (pendiente <= 0.02)
    - Menos puntos requeridos
    - R2 más bajo
    - Diff máximo más amplio
    """
    rsi = np.array(rsi_values)
    n = len(rsi)
    
    techos_idx, techos_val = [], []
    suelos_idx, suelos_val = [], []
    
    for i in range(ventana_pivot, n - ventana_pivot):
        ventana = rsi[i-ventana_pivot:i+ventana_pivot+1]
        if rsi[i] == ventana.max() and rsi[i] > 35:
            techos_idx.append(i)
            techos_val.append(rsi[i])
        if rsi[i] == ventana.min() and rsi[i] < 70:
            suelos_idx.append(i)
            suelos_val.append(rsi[i])
    
    if len(techos_idx) < min_puntos or len(suelos_idx) < min_puntos:
        return None, f"Pocos pivots (techos={len(techos_idx)}, suelos={len(suelos_idx)})"
    
    n_techos = min(6, len(techos_idx))
    n_suelos = min(6, len(suelos_idx))
    
    x_t = np.array(techos_idx[-n_techos:])
    y_t = np.array(techos_val[-n_techos:])
    x_s = np.array(suelos_idx[-n_suelos:])
    y_s = np.array(suelos_val[-n_suelos:])
    
    m_t, b_t, r_t, p_t, se_t = stats.linregress(x_t, y_t)
    m_s, b_s, r_s, p_s, se_s = stats.linregress(x_s, y_s)
    
    # MUCHO más permisivo: techos pueden estar planos o bajando ligeramente
    if m_t > pendiente_max:
        return None, f"Techo subiendo ({m_t:.4f} > {pendiente_max})"
    
    if m_s > 0.05:
        return None, f"Suelo subiendo mucho ({m_s:.4f})"
    
    if abs(m_t - m_s) > diff_max:
        return None, f"Lineas divergentes (diff={abs(m_t-m_s):.4f})"
    
    if r_t**2 < r2_min:
        return None, f"Correlacion muy baja (R2={r_t**2:.3f})"
    
    ancho_canal = (m_t * (n-1) + b_t) - (m_s * (n-1) + b_s)
    
    return {
        'techo_m': m_t, 'techo_b': b_t,
        'suelo_m': m_s, 'suelo_b': b_s,
        'techo_r2': r_t**2, 'suelo_r2': r_s**2,
        'ancho_canal': ancho_canal,
        'techos_idx': techos_idx, 'techos_val': techos_val,
        'suelos_idx': suelos_idx, 'suelos_val': suelos_val,
    }, "OK"

def detectar_zona_rsi(rsi_values):
    """
    Alternativa simple al canal: detectar si RSI está en zona extrema
    """
    rsi_actual = rsi_values[-1]
    
    # Zona de sobreventa (para long)
    sobreventa = rsi_actual < 35
    # Zona de sobrecompra (para short)
    sobrecompra = rsi_actual > 65
    
    return {
        'sobreventa': sobreventa,
        'sobrecompra': sobrecompra,
        'rsi_actual': rsi_actual,
        'suelo': 30,  # Zona de referencia
        'techo': 70,
    }

def detectar_diagonal_rsi(rsi_values, ventana_pivot=3, min_puntos=2,
                           pendiente_max=0.03, r2_min=0.05):
    """
    Diagonal más permisiva:
    - Menos puntos requeridos (2)
    - Ventana de pivot más pequeña
    - R2 más bajo
    """
    rsi = np.array(rsi_values)
    n = len(rsi)
    
    maximos_idx, maximos_val = [], []
    for i in range(ventana_pivot, n - ventana_pivot):
        ventana = rsi[i-ventana_pivot:i+ventana_pivot+1]
        if rsi[i] == ventana.max() and rsi[i] > 25:
            if not maximos_idx or rsi[i] <= maximos_val[-1] * 1.20:
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
    
    # Fallback muy permisivo
    x = np.array(maximos_idx[-min_puntos:])
    y = np.array(maximos_val[-min_puntos:])
    m, b, r, p, se = stats.linregress(x, y)
    if m <= 0.10:
        return {
            'm': m, 'b': b, 'r2': r**2,
            'maximos_idx': maximos_idx, 'maximos_val': maximos_val,
            'n_usados': min_puntos
        }, "OK (fallback)"
    
    return None, f"Tendencia muy ascendente ({m:.4f})"

def detectar_ruptura_diagonal(rsi_values, diagonal, umbral=0.0):
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
    
    return False, None

def detectar_tendencia_precio(df, idx, periodos=30):
    """
    Tendencia más simple y rápida
    """
    if idx < periodos:
        return 'lateral'
    
    precios = df['close'].iloc[idx-periodos:idx+1].values
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
    
    if direccion == 'long':
        sl_price = entry_price * (1 - CONFIG['sl_pct'])
        tp_price = entry_price * (1 + CONFIG['tp_pct'])
        min_recent = df['low'].iloc[max(0, idx_entrada-3):idx_entrada].min()
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
                pnl_pct = (sl_trailing - entry_price) / entry_price * 100
                return {'resultado': 'SL', 'exit_price': sl_trailing, 'pnl_pct': pnl_pct,
                        'velas_duracion': i, 'fecha_salida': df.index[idx_entrada + i],
                        'tipo_salida': 'trailing' if breakeven_activado else 'sl_fijo',
                        'direccion': 'long'}
            
            if vela['high'] >= tp_price:
                return {'resultado': 'TP', 'exit_price': tp_price, 'pnl_pct': CONFIG['tp_pct'] * 100,
                        'velas_duracion': i, 'fecha_salida': df.index[idx_entrada + i],
                        'tipo_salida': 'tp_fijo', 'direccion': 'long'}
    
    else:  # short
        sl_price = entry_price * (1 + CONFIG['sl_pct'])
        tp_price = entry_price * (1 - CONFIG['tp_pct'])
        max_recent = df['high'].iloc[max(0, idx_entrada-3):idx_entrada].max()
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
                pnl_pct = (entry_price - sl_trailing) / entry_price * 100
                return {'resultado': 'SL', 'exit_price': sl_trailing, 'pnl_pct': pnl_pct,
                        'velas_duracion': i, 'fecha_salida': df.index[idx_entrada + i],
                        'tipo_salida': 'trailing' if breakeven_activado else 'sl_fijo',
                        'direccion': 'short'}
            
            if vela['low'] <= tp_price:
                return {'resultado': 'TP', 'exit_price': tp_price, 'pnl_pct': CONFIG['tp_pct'] * 100,
                        'velas_duracion': i, 'fecha_salida': df.index[idx_entrada + i],
                        'tipo_salida': 'tp_fijo', 'direccion': 'short'}
    
    # Time exit
    idx_salida = idx_entrada + min(CONFIG['time_exit_max'], len(df) - idx_entrada - 1)
    exit_price = df['close'].iloc[idx_salida]
    
    if direccion == 'long':
        pnl_pct = (exit_price - entry_price) / entry_price * 100
    else:
        pnl_pct = (entry_price - exit_price) / entry_price * 100
    
    return {
        'resultado': 'TIME_EXIT',
        'exit_price': exit_price,
        'pnl_pct': pnl_pct,
        'velas_duracion': min(CONFIG['time_exit_max'], len(df) - idx_entrada - 1),
        'fecha_salida': df.index[idx_salida],
        'tipo_salida': 'time_exit',
        'direccion': direccion
    }

def backtest(df_1h):
    log("="*60)
    log(f"INICIANDO BACKTEST v8 - Modo: {CONFIG['modo']}")
    log("="*60)
    
    trades = []
    rechazos = {
        'canal': 0, 'rebote': 0, 'zona': 0, 'diagonal': 0, 'ruptura': 0,
        'cooldown': 0, 'tendencia': 0
    }
    
    rsi = df_1h['rsi'].values
    closes = df_1h['close'].values
    
    ventana_min = 30  # Reducido
    ultimo_trade_fecha = None
    
    log(f"Total velas 1h: {len(df_1h)}")
    log(f"Periodo: {df_1h.index[0]} -> {df_1h.index[-1]}")
    log(f"Modo: {CONFIG['modo']} | Dirección: {CONFIG['direccion']}")
    log(f"Analizando...")
    
    for i in range(ventana_min, len(df_1h) - 1):
        fecha = df_1h.index[i]
        
        # COOLDOWN
        if ultimo_trade_fecha is not None:
            horas_desde_ultimo = (fecha - ultimo_trade_fecha).total_seconds() / 3600
            if horas_desde_ultimo < CONFIG['cooldown_horas']:
                rechazos['cooldown'] += 1
                continue
        
        rsi_window = rsi[:i+1]
        rsi_reciente = rsi[max(0, i-20):i+1]  # Ventana corta para diagonal
        
        # === MODO CANAL (más restrictivo) ===
        if CONFIG['modo'] == 'canal':
            canal, msg = detectar_canal_rsi_permisivo(rsi_window)
            if not canal:
                rechazos['canal'] += 1
                continue
            
            rebote, info = detectar_zona_rsi(rsi_window)
            # Usar info del canal
            info_canal = {
                'rsi_actual': rsi[i],
                'suelo': canal['suelo_m'] * i + canal['suelo_b'],
                'techo': canal['techo_m'] * i + canal['techo_b'],
                'ancho_canal': canal['ancho_canal']
            }
            
            # Verificar si está en zona de rebote
            en_zona = rsi[i] < info_canal['suelo'] * 1.05 or rsi[i] < 40
            if not en_zona:
                rechazos['rebote'] += 1
                continue
        
        # === MODO ZONA (intermedio) ===
        elif CONFIG['modo'] == 'zona':
            info_zona = detectar_zona_rsi(rsi_window)
            info_canal = {
                'rsi_actual': rsi[i],
                'suelo': 35,
                'techo': 65,
                'ancho_canal': 30
            }
            # No requiere canal, solo zona extrema
            if not (info_zona['sobreventa'] or info_zona['sobrecompra']):
                rechazos['zona'] += 1
                continue
        
        # === MODO SOLO_DIAGONAL (más permisivo) ===
        else:
            info_canal = {
                'rsi_actual': rsi[i],
                'suelo': 30,
                'techo': 70,
                'ancho_canal': 40
            }
        
        # Detectar diagonal (siempre requerida)
        diagonal, msg = detectar_diagonal_rsi(rsi_reciente)
        if not diagonal:
            rechazos['diagonal'] += 1
            continue
        
        # Verificar ruptura
        ruptura, info_diag = detectar_ruptura_diagonal(rsi_reciente, diagonal)
        if not ruptura:
            rechazos['ruptura'] += 1
            continue
        
        # Determinar dirección
        tendencia = detectar_tendencia_precio(df_1h, i)
        
        if CONFIG['direccion'] == 'both':
            # RSI bajo = LONG, RSI alto = SHORT
            if rsi[i] < 40:
                direccion_final = 'long'
            elif rsi[i] > 60:
                direccion_final = 'short'
            else:
                direccion_final = 'long'  # Default
        elif CONFIG['direccion'] == 'short':
            direccion_final = 'short'
        else:
            direccion_final = 'long'
        
        # SEÑAL ENCONTRADA
        entry_price = closes[i]
        
        resultado_trade = simular_trade(df_1h, i, entry_price, direccion_final)
        
        if resultado_trade:
            ultimo_trade_fecha = fecha
            trades.append({
                'fecha_entrada': fecha,
                'idx': i,
                'entry': entry_price,
                'sl': entry_price * (1 - CONFIG['sl_pct']) if direccion_final == 'long' else entry_price * (1 + CONFIG['sl_pct']),
                'tp': entry_price * (1 + CONFIG['tp_pct']) if direccion_final == 'long' else entry_price * (1 - CONFIG['tp_pct']),
                'rsi': rsi[i],
                'techo_canal': info_canal['techo'],
                'suelo_canal': info_canal['suelo'],
                'ancho_canal': info_canal['ancho_canal'],
                'tendencia': tendencia,
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
    min_balance = balance
    max_drawdown = 0
    max_drawdown_usd = 0
    
    balances = [balance]
    
    for trade in trades:
        riesgo = balance * CONFIG['risk_per_trade']
        pnl_usd = riesgo * (trade['pnl_pct'] / (CONFIG['sl_pct'] * 100))
        balance += pnl_usd
        
        balances.append(balance)
        
        if balance > max_balance:
            max_balance = balance
        
        drawdown = (max_balance - balance) / max_balance * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_usd = max_balance - balance
        
        if balance < min_balance:
            min_balance = balance
    
    ganancias_totales = sum(t['pnl_pct'] for t in ganadores)
    perdidas_totales = abs(sum(t['pnl_pct'] for t in perdedores))
    profit_factor = ganancias_totales / perdidas_totales if perdidas_totales > 0 else float('inf')
    
    avg_win = np.mean([t['pnl_pct'] for t in ganadores]) if ganadores else 0
    avg_loss = np.mean([t['pnl_pct'] for t in perdedores]) if perdedores else 0
    expectancy = (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss)
    
    returns = [t['pnl_pct'] for t in trades]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252*24) if np.std(returns) > 0 else 0
    
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
    log("BOT RSI CANALES - BACKTEST v8")
    log(f"Par: {CONFIG['symbol']}")
    log(f"SL: {CONFIG['sl_pct']*100:.1f}% | TP: {CONFIG['tp_pct']*100:.1f}%")
    log(f"Modo: {CONFIG['modo']} | Dirección: {CONFIG['direccion']}")
    log(f"Debug: {CONFIG['debug']}")
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
    
    log("Descargando datos 1h...")
    
    # Intentar descargar más datos con batches
    df_1h = fetch_data_batches(exchange, CONFIG['symbol'], '1h', total_velas=2000)
    
    if df_1h is None or len(df_1h) == 0:
        # Fallback a descarga simple
        df_1h = fetch_data(exchange, CONFIG['symbol'], '1h', limit=1000)
    
    if df_1h is None:
        log("Error descargando datos")
        return
    
    log(f"1h: {len(df_1h)} velas | {df_1h.index[0]} -> {df_1h.index[-1]}")
    
    # Calcular RSI
    df_1h['rsi'] = calcular_rsi(df_1h['close'])
    
    # BACKTEST
    trades = backtest(df_1h)
    
    # RESULTADOS
    log("")
    log("="*60)
    log("RESULTADOS DEL BACKTEST v8")
    log("="*60)
    log(f"Periodo: {df_1h.index[0].strftime('%Y-%m-%d')} -> {df_1h.index[-1].strftime('%Y-%m-%d')}")
    log(f"Total velas 1h: {len(df_1h)}")
    log(f"")
    log(f"TRADES ENCONTRADOS: {len(trades)}")
    log(f"")
    
    if len(trades) == 0:
        log("NO HUBO NINGUNA SEÑAL EN ESTE PERIODO")
        log("")
        log("SUGERENCIAS:")
        log("1. Probar MODO=zona (más señales)")
        log("2. Probar MODO=solo_diagonal (máximas señales)")
        log("3. Probar DIRECCION=both")
        log("4. Probar otros pares: SOL, ADA, DOGE")
    else:
        metricas = calcular_metricas(trades)
        
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
        log(f"Promedio ganador: {metricas['avg_win']:.2f}%")
        log(f"Promedio perdedor: {metricas['avg_loss']:.2f}%")
        log(f"")
        
        log("DETALLE DE TRADES:")
        log("-" * 60)
        for i, t in enumerate(trades, 1):
            emoji = "🟢" if t['pnl_pct'] > 0 else "🔴"
            dir_icon = "📈" if t.get('direccion') == 'long' else "📉"
            salida_icon = "⏱️" if t['resultado'] == 'TIME_EXIT' else ("🎯" if t['resultado'] == 'TP' else "🛑")
            log(f"{emoji} #{i} {dir_icon} | {t['fecha_entrada'].strftime('%Y-%m-%d %H:%M')} {salida_icon}")
            log(f"   Entrada: ${t['entry']:.4f} -> Salida: ${t['exit_price']:.4f}")
            log(f"   Resultado: {t['resultado']} ({t['tipo_salida']}) | P&L: {t['pnl_pct']:+.2f}%")
            log(f"   Duración: {t['velas_duracion']} velas 1h")
            log(f"   RSI: {t['rsi']:.1f} | Ancho canal: {t['ancho_canal']:.1f}")
            log(f"   Tendencia: {t.get('tendencia', 'N/A')}")
            log(f"")
    
    log("="*60)
    log("Backtest finalizado")
    log("="*60)

if __name__ == "__main__":
    main()
