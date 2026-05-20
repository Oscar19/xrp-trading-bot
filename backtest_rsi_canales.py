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
        rsi_4h_window = rsi_4h[:i+1]
        
        # Detectar canal
        canal, msg = detectar_canal_4h(rsi_4h_window)
        if not canal:
            rechazos['canal'] += 1
            if CONFIG['debug'] and i > len(df_4h) - 10:
                log(f"  [i={i}] Canal rechazado: {msg}")
            continue
        
        # Verificar rebote
        rebote, info_4h = detectar_rebote_4h(rsi_4h_window, canal)
        if not rebote:
            rechazos['rebote'] += 1
            continue
        
        # Encontrar ventana 1h correspondiente
        # CORRECCION: Mapeo temporal preciso
        fecha_4h = df_4h.index[i]
        # Buscar el índice 1h más cercano a esta fecha 4h
        idx_1h_candidatos = df_1h.index.get_indexer([fecha_4h], method='nearest')
        if len(idx_1h_candidatos) == 0 or idx_1h_candidatos[0] == -1:
            continue
        idx_1h = idx_1h_candidatos[0]
        
        if idx_1h < 50:
            continue
        
        # Verificar que tenemos datos 1h suficientes
        if idx_1h >= len(rsi_1h):
            rechazos['datos_1h'] += 1
            continue
        
        # FILTROS (opcionales)
        if CONFIG['use_filters']:
            # Filtro tendencia 4h
            ema20_4h = df_4h['ema20'].iloc[i]
            if closes_4h[i] < ema20_4h * 0.95:  # Más permisivo (0.95 vs 0.98)
                rechazos['filtro_tendencia'] += 1
                continue
            
            # Filtro volatilidad
            atr_actual = df_1h['atr'].iloc[idx_1h]
            atr_avg = df_1h['atr_avg'].iloc[idx_1h]
            if pd.notna(atr_avg) and atr_actual < atr_avg * 0.3:  # Más permisivo (0.3 vs 0.5)
                rechazos['filtro_vol'] += 1
                continue
        
        rsi_1h_window = rsi_1h[:idx_1h+1]
        
        # Detectar diagonal
        diagonal, msg = detectar_diagonal_1h(rsi_1h_window)
        if not diagonal:
            rechazos['diagonal'] += 1
            if CONFIG['debug'] and i > len(df_4h) - 10:
                log(f"  [i={i}] Diagonal rechazada: {msg}")
            continue
        
        # Verificar ruptura
        ruptura, info_1h = detectar_ruptura_1h(rsi_1h_window, diagonal)
        if not ruptura:
            rechazos['ruptura'] += 1
            continue
        
        # SEÑAL ENCONTRADA
        fecha = df_4h.index[i]
        entry_price = closes_1h[idx_1h]
        
        resultado_trade = simular_trade(df_1h, idx_1h, entry_price)
        
        if resultado_trade:
            trades.append({
                'fecha_entrada': fecha,
                'idx_4h': i,
                'idx_1h': idx_1h,
                'entry': entry_price,
                'sl': entry_price * (1 - CONFIG['sl_pct']),
                'tp': entry_price * (1 + CONFIG['tp_pct']),
                'rsi_4h': info_4h['rsi_actual'],
                'rsi_1h': info_1h['rsi_despues'],
                'techo_canal': info_4h['techo'],
                'suelo_canal': info_4h['suelo'],
                'ancho_canal': info_4h.get('ancho_canal', 0),
                **resultado_trade
            })
    
    # Log de rechazos para debug
    if CONFIG['debug'] or len(trades) < 3:
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
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
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
    }

def main():
    log("="*60)
    log("BOT RSI CANALES - BACKTEST v3")
    log(f"Par: {CONFIG['symbol']}")
    log(f"Debug: {CONFIG['debug']} | Filtros: {CONFIG['use_filters']}")
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
    
    log("Descargando datos históricos...")
    
    # CORRECCION: Pedir más datos 1h para cubrir el mismo periodo que 4h
    # 600 velas 4h = 2400 velas 1h. Pedimos 2500 para margen.
    df_4h = fetch_data(exchange, CONFIG['symbol'], '4h', limit=600)
    df_1h = fetch_data(exchange, CONFIG['symbol'], '1h', limit=2500)
    
    if df_4h is None or df_1h is None:
        log("Error descargando datos")
        return
    
    log(f"4h: {len(df_4h)} velas | {df_4h.index[0]} -> {df_4h.index[-1]}")
    log(f"1h: {len(df_1h)} velas | {df_1h.index[0]} -> {df_1h.index[-1]}")
    
    # Calcular RSI
    df_4h['rsi'] = calcular_rsi(df_4h['close'])
    df_1h['rsi'] = calcular_rsi(df_1h['close'])
    
    # BACKTEST
    trades = backtest_completo(df_4h, df_1h)
    
    # RESULTADOS
    log("")
    log("="*60)
    log("RESULTADOS DEL BACKTEST v3")
    log("="*60)
    log(f"Periodo: {df_4h.index[0].strftime('%Y-%m-%d')} -> {df_4h.index[-1].strftime('%Y-%m-%d')}")
    log(f"Total velas 4h: {len(df_4h)}")
    log(f"")
    log(f"TRADES ENCONTRADOS: {len(trades)}")
    log(f"")
    
    if len(trades) == 0:
        log("NO HUBO NINGUNA SEÑAL EN ESTE PERIODO")
        log("")
        log("SUGERENCIAS:")
        log("1. Ejecutar con DEBUG=true para ver rechazos:")
        log("   DEBUG=true python backtest_rsi_canales.py")
        log("2. Desactivar filtros:")
        log("   USE_FILTERS=false python backtest_rsi_canales.py")
        log("3. Probar otros pares: ADA, SOL, DOGE")
        log("4. Aumentar periodo a 6 meses")
        log("5. Reducir aun más min_r2_canal=0.01 min_r2_diagonal=0.05")
    else:
        metricas = calcular_metricas(trades)
        
        log("METRICAS GLOBALES:")
        log("-" * 60)
        log(f"Trades totales: {metricas['n_trades']}")
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
        log(f"Promedio ganador: {metricas['avg_win']:.2f}%")
        log(f"Promedio perdedor: {metricas['avg_loss']:.2f}%")
        log(f"")
        
        log("DETALLE DE TRADES:")
        log("-" * 60)
        for i, t in enumerate(trades, 1):
            emoji = "🟢" if t['pnl_pct'] > 0 else "🔴"
            salida_icon = "⏱️" if t['resultado'] == 'TIME_EXIT' else ("🎯" if t['resultado'] == 'TP' else "🛑")
            log(f"{emoji} #{i} | {t['fecha_entrada'].strftime('%Y-%m-%d %H:%M')} {salida_icon}")
            log(f"   Entrada: ${t['entry']:.4f} -> Salida: ${t['exit_price']:.4f}")
            log(f"   Resultado: {t['resultado']} ({t['tipo_salida']}) | P&L: {t['pnl_pct']:+.2f}%")
            log(f"   Duración: {t['velas_duracion']} velas 1h")
            log(f"   RSI 4h: {t['rsi_4h']:.1f} | RSI 1h: {t['rsi_1h']:.1f}")
            log(f"   Ancho canal: {t['ancho_canal']:.1f}")
            log(f"")
    
    log("="*60)
    log("Backtest finalizado")
    log("="*60)

if __name__ == "__main__":
    main()
