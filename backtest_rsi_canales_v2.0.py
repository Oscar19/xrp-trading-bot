"""
BOT RSI CANALES - BACKTEST v2.0
==============================
Backtest combinado:
1. RSI_CANALES: Ruptura de diagonal descendente en RSI
2. DIVERGENCIA_RSI: Precio hace mínimos más bajos, RSI más altos

Compara resultados de ambas estrategias por separado y combinadas.
"""

import pandas as pd
import numpy as np
import ccxt
import os
from datetime import datetime
from scipy import stats

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

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
    'rsi_period': 14,
    'pivot_ventana': 2,
    'min_puntos_diagonal': 2,
    'pendiente_max_diagonal': 0.10,
    'r2_min_diagonal': 0.10,
    'zona_sobreventa': 45,
    'zona_sobrecompra': 55,
    'umbral_ruptura': 0.8,

    # Config divergencia
    'div_ventana': 80,
    'div_ratio_precio': 0.99,
    'div_ratio_rsi': 1.02,
    'div_max_rsi': 40,

    'direccion': 'long',
    'debug': os.environ.get('DEBUG', 'false').lower() == 'true',
}

# ============================================================================
# LOGGING
# ============================================================================

LOG_FILE = None

def init_logging():
    global LOG_FILE
    log_path = os.environ.get('LOG_FILE', 'backtest_resultados.txt')
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

# ============================================================================
# ESTRATEGIA 1: RSI CANALES
# ============================================================================

def detectar_diagonal_maximos_rsi(rsi_values, ventana_pivot=2, min_puntos=2,
                                   pendiente_max=-0.01, r2_min=0.10):
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
            }

    return best_diagonal

def detectar_ruptura_diagonal_long(rsi_values, diagonal, df, idx,
                                    umbral_ruptura=0.8,
                                    volumen_ratio_min=1.2,
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
        'ratio_volumen': ratio_volumen,
        'volumen_confirmado': volumen_confirmado,
        'pendiente_diagonal': diagonal['m'],
        'r2_diagonal': diagonal['r2'],
        'n_puntos': diagonal['n_usados'],
    }

# ============================================================================
# ESTRATEGIA 2: DIVERGENCIA RSI
# ============================================================================

def encontrar_minimos_locales(valores, ventana=5):
    minimos = []
    for i in range(ventana, len(valores) - ventana):
        ventana_local = valores[i - ventana:i + ventana + 1]
        if valores[i] == ventana_local.min():
            minimos.append(i)
    return minimos

def detectar_divergencia_rsi(df, idx, config):
    """
    Detecta divergencia alcista RSI en el punto idx.
    Retorna (señal_bool, info_dict)
    """
    if idx < 60:
        return False, None

    precios = df['low'].values[:idx + 1]
    rsi_vals = df['rsi'].values[:idx + 1]

    ventana_busqueda = config['div_ventana']
    if len(precios) < ventana_busqueda:
        return False, None

    # Buscar mínimos locales en precio
    minimos_precio = encontrar_minimos_locales(precios[-ventana_busqueda:], ventana=3)
    if len(minimos_precio) < 2:
        return False, None

    offset = len(precios) - ventana_busqueda
    minimos_precio = [m + offset for m in minimos_precio]

    # Últimos 2 mínimos
    min1_idx = minimos_precio[-2]
    min2_idx = minimos_precio[-1]

    # Verificar distancia mínima entre mínimos
    if min2_idx - min1_idx < 5:
        return False, None

    precio_min1 = precios[min1_idx]
    precio_min2 = precios[min2_idx]

    # Precio debe hacer mínimo más bajo
    if precio_min2 >= precio_min1 * config['div_ratio_precio']:
        return False, None

    # RSI en esos puntos
    rsi_min1 = rsi_vals[min1_idx]
    rsi_min2 = rsi_vals[min2_idx]

    # RSI debe hacer mínimo más alto (divergencia)
    if rsi_min2 <= rsi_min1 * config['div_ratio_rsi']:
        return False, None

    # RSI actual en zona de sobreventa
    rsi_actual = rsi_vals[idx]
    if rsi_actual > config['div_max_rsi']:
        return False, None

    # Momentum actual
    if rsi_vals[idx] <= rsi_vals[idx - 1]:
        return False, None

    return True, {
        'precio_min1': precio_min1,
        'precio_min2': precio_min2,
        'rsi_min1': rsi_min1,
        'rsi_min2': rsi_min2,
        'idx_min1': min1_idx,
        'idx_min2': min2_idx,
    }

# ============================================================================
# SIMULACIÓN DE TRADE
# ============================================================================

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
                        'pnl_pct_raw': pnl_pct_raw,
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

    idx_salida = idx_entrada + min(CONFIG['time_exit_max'], len(df) - idx_entrada - 1)
    exit_price = df['close'].iloc[idx_salida]
    pnl_pct_raw = (exit_price - entry_price) / entry_price * 100
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

# ============================================================================
# BACKTEST ESTRATEGIA 1
# ============================================================================

def backtest_rsi_canales(df_4h):
    log("\n" + "="*60)
    log("BACKTEST: RSI_CANALES")
    log("="*60)

    trades = []
    rechazos = {
        'diagonal_long': 0, 'ruptura_long': 0,
        'volumen_long': 0, 'zona_rsi': 0,
        'cooldown': 0, 'tendencia': 0
    }

    rsi = df_4h['rsi'].values
    closes = df_4h['close'].values
    ventana_min = 50
    ultimo_trade_fecha = None

    for i in range(ventana_min, len(df_4h) - 1):
        fecha = df_4h.index[i]

        if ultimo_trade_fecha is not None:
            horas_desde_ultimo = (fecha - ultimo_trade_fecha).total_seconds() / 3600
            if horas_desde_ultimo < CONFIG['cooldown_horas']:
                rechazos['cooldown'] += 1
                continue

        rsi_window = rsi[:i + 1]
        rsi_actual = rsi[i]

        if rsi_actual >= CONFIG['zona_sobreventa']:
            rechazos['zona_rsi'] += 1
            continue

        diagonal = detectar_diagonal_maximos_rsi(
            rsi_window,
            ventana_pivot=CONFIG['pivot_ventana'],
            min_puntos=CONFIG['min_puntos_diagonal'],
            pendiente_max=-0.01,
            r2_min=CONFIG['r2_min_diagonal']
        )

        if not diagonal:
            rechazos['diagonal_long'] += 1
            continue

        ruptura, info = detectar_ruptura_diagonal_long(
            rsi_window, diagonal, df_4h, i,
            umbral_ruptura=CONFIG['umbral_ruptura'],
            volumen_ratio_min=CONFIG['volumen_min_ratio'],
            volumen_lookback=CONFIG['volumen_lookback']
        )

        if not ruptura:
            rechazos['ruptura_long'] += 1
            continue

        if not info.get('volumen_confirmado', False):
            rechazos['volumen_long'] += 1
            continue

        entry_price = closes[i]
        resultado = simular_trade(df_4h, i, entry_price, 'long')

        if resultado:
            ultimo_trade_fecha = fecha
            trades.append({
                'fecha_entrada': fecha,
                'idx': i,
                'entry': entry_price,
                'estrategia': 'RSI_CANALES',
                'rsi': rsi_actual,
                **info,
                **resultado
            })

    log(f"Trades RSI_CANALES: {len(trades)}")
    for k, v in rechazos.items():
        log(f"  {k}: {v}")

    return trades

# ============================================================================
# BACKTEST ESTRATEGIA 2
# ============================================================================

def backtest_divergencia(df_4h):
    log("\n" + "="*60)
    log("BACKTEST: DIVERGENCIA_RSI")
    log("="*60)

    trades = []
    rechazos = {
        'pocos_minimos': 0, 'precio_no_bajo': 0,
        'rsi_no_subio': 0, 'zona_rsi': 0,
        'sin_momentum': 0, 'cooldown': 0
    }

    rsi = df_4h['rsi'].values
    closes = df_4h['close'].values
    ventana_min = 60
    ultimo_trade_fecha = None

    for i in range(ventana_min, len(df_4h) - 1):
        fecha = df_4h.index[i]

        if ultimo_trade_fecha is not None:
            horas_desde_ultimo = (fecha - ultimo_trade_fecha).total_seconds() / 3600
            if horas_desde_ultimo < CONFIG['cooldown_horas']:
                rechazos['cooldown'] += 1
                continue

        divergencia, info = detectar_divergencia_rsi(df_4h, i, CONFIG)

        if not divergencia:
            if info is None:
                rechazos['pocos_minimos'] += 1
            continue

        rsi_actual = rsi[i]
        if rsi_actual > CONFIG['div_max_rsi']:
            rechazos['zona_rsi'] += 1
            continue

        if rsi[i] <= rsi[i - 1]:
            rechazos['sin_momentum'] += 1
            continue

        entry_price = closes[i]
        resultado = simular_trade(df_4h, i, entry_price, 'long')

        if resultado:
            ultimo_trade_fecha = fecha
            trades.append({
                'fecha_entrada': fecha,
                'idx': i,
                'entry': entry_price,
                'estrategia': 'DIVERGENCIA_RSI',
                'rsi': rsi_actual,
                **info,
                **resultado
            })

    log(f"Trades DIVERGENCIA_RSI: {len(trades)}")
    for k, v in rechazos.items():
        log(f"  {k}: {v}")

    return trades

# ============================================================================
# MÉTRICAS
# ============================================================================

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
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    init_logging()

    try:
        log("="*60)
        log("BOT RSI CANALES - BACKTEST v2.0 (COMBINADO)")
        log(f"Par: {CONFIG['symbol']}")
        log(f"Timeframe: {CONFIG['timeframe']}")
        log(f"SL: {CONFIG['sl_pct']*100:.1f}% | TP: {CONFIG['tp_pct']*100:.1f}%")
        log(f"Apalancamiento: {CONFIG['leverage']}x")
        log(f"Cooldown: {CONFIG['cooldown_horas']}h")
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

        if df is None:
            df = fetch_data(exchange, CONFIG['symbol'], CONFIG['timeframe'], limit=1000)

        if df is None:
            log("Error descargando datos")
            return

        log(f"{CONFIG['timeframe']}: {len(df)} velas | {df.index[0]} -> {df.index[-1]}")
        df['rsi'] = calcular_rsi(df['close'], period=CONFIG['rsi_period'])

        # Ejecutar ambos backtests
        trades_canales = backtest_rsi_canales(df)
        trades_divergencia = backtest_divergencia(df)
        trades_combined = trades_canales + trades_divergencia
        trades_combined.sort(key=lambda x: x['fecha_entrada'])

        # Métricas individuales
        metricas_canales = calcular_metricas(trades_canales)
        metricas_div = calcular_metricas(trades_divergencia)
        metricas_combined = calcular_metricas(trades_combined)

        # RESULTADOS
        log("\n" + "="*60)
        log("RESULTADOS COMPARATIVOS")
        log("="*60)

        def print_metricas(nombre, metricas, trades):
            if not metricas:
                log(f"\n{nombre}: 0 trades")
                return
            log(f"\n{nombre}:")
            log(f"  Trades: {metricas['n_trades']} (G:{metricas['n_ganadores']} P:{metricas['n_perdedores']})")
            log(f"  Win Rate: {metricas['win_rate']:.1f}%")
            log(f"  Profit Factor: {metricas['profit_factor']:.2f}")
            log(f"  Expectancy: {metricas['expectancy']:.2f}%")
            log(f"  Sharpe: {metricas['sharpe']:.2f}")
            log(f"  Max DD: {metricas['max_drawdown']:.2f}%")
            log(f"  Return: {metricas['pnl_total_pct']:.2f}%")
            log(f"  Balance: ${metricas['balance_inicial']:.0f} -> ${metricas['balance_final']:.2f}")

        print_metricas("RSI_CANALES", metricas_canales, trades_canales)
        print_metricas("DIVERGENCIA_RSI", metricas_div, trades_divergencia)
        print_metricas("COMBINADO", metricas_combined, trades_combined)

        # Detalle de trades combinados
        if trades_combined:
            log("\nDETALLE TRADES COMBINADOS:")
            log("-" * 60)
            for i, t in enumerate(trades_combined, 1):
                emoji = "🟢" if t['pnl_pct'] > 0 else "🔴"
                est_icon = "📉" if t['estrategia'] == 'RSI_CANALES' else "🔄"
                log(f"{emoji} #{i} {est_icon} {t['estrategia'][:12]} | {t['fecha_entrada'].strftime('%Y-%m-%d %H:%M')}")
                log(f"   Entrada: ${t['entry']:.4f} -> Salida: ${t['exit_price']:.4f}")
                log(f"   Resultado: {t['resultado']} | P&L {CONFIG['leverage']}x: {t['pnl_pct']:+.2f}%")
                log(f"   RSI: {t['rsi']:.1f} | Duración: {t['velas_duracion']} velas")
                log("")

        log("="*60)
        log("Backtest finalizado")
        log("="*60)

    finally:
        close_logging()

if __name__ == "__main__":
    main()
