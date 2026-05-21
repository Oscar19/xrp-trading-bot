"""
BOT RSI CANALES - ALERTAS v1.1 (FIX)
==============================
Sistema de alertas en tiempo real para ADA, SOL, XRP
- Evalúa cada hora desde GitHub Actions
- Envía señales a Telegram cuando detecta ruptura de diagonal RSI
- Basado en estrategia v10.5.1 (LONGS ONLY)
"""

import pandas as pd
import numpy as np
import ccxt
import os
import requests
from datetime import datetime, timedelta
from scipy import stats

# ============================================================================
# CONFIGURACIÓN TELEGRAM
# ============================================================================

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# ============================================================================
# CONFIGURACIÓN DE ACTIVOS
# ============================================================================

ACTIVOS = {
    'ADA/USDT': {'min_rsi': 45, 'nombre': 'ADA'},
    'SOL/USDT': {'min_rsi': 45, 'nombre': 'SOL'},
    'XRP/USDT': {'min_rsi': 45, 'nombre': 'XRP'},
}

CONFIG = {
    'timeframe': '4h',
    'leverage': 5,
    'sl_pct': 0.008,
    'tp_pct': 0.015,
    'volumen_min_ratio': float(os.environ.get('VOLUMEN_MIN_RATIO', '1.2')),
    'volumen_lookback': 10,
    'rsi_period': 14,
    'pivot_ventana': 2,
    'min_puntos_diagonal': 2,
    'r2_min_diagonal': 0.10,
    'umbral_ruptura': 0.8,
    'cooldown_horas': 8,
}

# ============================================================================
# TELEGRAM
# ============================================================================

def enviar_telegram(mensaje, foto=None):
    """Envía mensaje a Telegram. Si foto es None, envía texto."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[ALERTA] Telegram no configurado. Mensaje: {mensaje[:100]}...")
        return False

    try:
        if foto:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {'photo': foto}
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': mensaje, 'parse_mode': 'HTML'}
            response = requests.post(url, files=files, data=data, timeout=30)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                'chat_id': TELEGRAM_CHAT_ID,
                'text': mensaje,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            response = requests.post(url, json=data, timeout=30)

        if response.status_code == 200:
            print(f"[OK] Telegram enviado")
            return True
        else:
            print(f"[ERROR] Telegram: {response.status_code} - {response.text[:200]}")
            return False
    except Exception as e:
        print(f"[ERROR] Envío Telegram: {e}")
        return False

# ============================================================================
# DATOS
# ============================================================================

def fetch_data(exchange, symbol, timeframe, limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")
        return None

def calcular_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ============================================================================
# DETECCIÓN DE DIAGONALES (IGUAL A v10.5.1)
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
                'n_usados': 2,
                'ultimo_max_idx': x[-1], 'ultimo_max_val': y[-1],
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
        return False, {
            'rsi_antes': rsi_2,
            'rsi_despues': rsi_1,
            'diag_valor': val_diag_1,
        }

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
        'rsi_antes': rsi_2,
        'rsi_despues': rsi_1,
        'diag_valor': val_diag_1,
    }

# ============================================================================
# ANÁLISIS POR ACTIVO
# ============================================================================

def analizar_activo(exchange, symbol, config_activo):
    """Analiza un activo y devuelve señal si hay ruptura."""
    print(f"\n[ANALIZANDO] {symbol}...")

    df = fetch_data(exchange, symbol, CONFIG['timeframe'], limit=100)
    if df is None or len(df) < 50:
        return None

    df['rsi'] = calcular_rsi(df['close'], period=CONFIG['rsi_period'])

    rsi_values = df['rsi'].values
    idx = len(df) - 1
    rsi_actual = rsi_values[idx]
    precio_actual = df['close'].iloc[idx]

    print(f"  Precio: ${precio_actual:.4f} | RSI: {rsi_actual:.1f}")

    # Zona RSI
    if rsi_actual >= config_activo['min_rsi']:
        print(f"  [SKIP] RSI {rsi_actual:.1f} >= {config_activo['min_rsi']} (no en zona sobreventa)")
        return None

    # Detectar diagonal
    diagonal = detectar_diagonal_maximos_rsi(
        rsi_values,
        ventana_pivot=CONFIG['pivot_ventana'],
        min_puntos=CONFIG['min_puntos_diagonal'],
        pendiente_max=-0.01,
        r2_min=CONFIG['r2_min_diagonal']
    )

    if not diagonal:
        print(f"  [SKIP] No hay diagonal válida")
        return None

    print(f"  Diagonal: pendiente={diagonal['m']:.4f}, R²={diagonal['r2']:.2f}, puntos={diagonal['n_usados']}")

    # Detectar ruptura
    ruptura, info = detectar_ruptura_diagonal_long(
        rsi_values, diagonal, df, idx,
        umbral_ruptura=CONFIG['umbral_ruptura'],
        volumen_ratio_min=CONFIG['volumen_min_ratio'],
        volumen_lookback=CONFIG['volumen_lookback']
    )

    # FIX: Manejar info=None
    if info is None:
        print(f"  [SKIP] No hay ruptura (info=None)")
        return None

    if not ruptura:
        print(f"  [SKIP] No hay ruptura (RSI_antes={info.get('rsi_antes', 'N/A'):.1f}, diag={info.get('diag_valor', 'N/A'):.1f})")
        return None

    if not info.get('volumen_confirmado', False):
        print(f"  [SKIP] Volumen insuficiente ({info.get('ratio_volumen', 0):.1f}x)")
        return None

    # ¡SEÑAL!
    print(f"  [🟢 SEÑAL] Ruptura detectada!")

    return {
        'symbol': symbol,
        'nombre': config_activo['nombre'],
        'precio': precio_actual,
        'rsi': rsi_actual,
        'sl': precio_actual * (1 - CONFIG['sl_pct']),
        'tp': precio_actual * (1 + CONFIG['tp_pct']),
        **info
    }

# ============================================================================
# FORMATO DE ALERTA
# ============================================================================

def formatear_alerta(señal):
    emoji = "🟢"
    nombre = señal['nombre']
    symbol = señal['symbol']
    precio = señal['precio']

    sl_pct = CONFIG['sl_pct'] * 100
    tp_pct = CONFIG['tp_pct'] * 100
    leverage = CONFIG['leverage']

    sl_precio = señal['sl']
    tp_precio = señal['tp']

    pnl_sl = -sl_pct * leverage
    pnl_tp = tp_pct * leverage

    mensaje = f"""{emoji} <b>ALERTA LONG - {nombre}</b>

📊 <b>{symbol}</b>
💰 Precio: <code>${precio:.4f}</code>
📈 RSI: <code>{señal['rsi']:.1f}</code>
📊 Volumen: <code>{señal['ratio_volumen']:.1f}x</code> media
📉 Diagonal: <code>R²={señal['r2_diagonal']:.2f}</code>, <code>{señal['n_puntos']} puntos</code>

🎯 <b>TRADE SETUP</b>
🟥 SL: <code>${sl_precio:.4f}</code> ({sl_pct:.1f}%)
🟩 TP: <code>${tp_precio:.4f}</code> ({tp_pct:.1f}%)
⚡ Apalancamiento: <code>{leverage}x</code>

💵 P&L estimado:
   SL: {pnl_sl:.1f}% | TP: +{pnl_tp:.1f}%

⏰ Timeframe: 4H
🔔 Alerta: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"""

    return mensaje

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("BOT RSI CANALES - ALERTAS v1.1 (FIX)")
    print(f"Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    if not TELEGRAM_BOT_TOKEN:
        print("[ADVERTENCIA] TELEGRAM_BOT_TOKEN no configurado")
    if not TELEGRAM_CHAT_ID:
        print("[ADVERTENCIA] TELEGRAM_CHAT_ID no configurado")

    try:
        exchange = ccxt.bitget({
            'options': {'defaultType': 'swap'},
            'timeout': 30000,
            'enableRateLimit': True
        })
        print("[OK] Conectado a Bitget")
    except Exception as e:
        print(f"[ERROR] Conectando: {e}")
        return

    señales = []
    for symbol, config in ACTIVOS.items():
        señal = analizar_activo(exchange, symbol, config)
        if señal:
            señales.append(señal)

    print(f"\n{'='*60}")
    print(f"SEÑALES ENCONTRADAS: {len(señales)}")
    print(f"{'='*60}")

    if señales:
        for señal in señales:
            mensaje = formatear_alerta(señal)
            print(f"\n[ENVIANDO] {señal['symbol']}")
            enviar_telegram(mensaje)
    else:
        print("No hay señales en este momento.")
        # Enviar heartbeat cada 6 horas (a las 00:00, 06:00, 12:00, 18:00)
        hora = datetime.now().hour
        if hora % 6 == 0:
            mensaje = f"🔍 <b>Heartbeat Alertas</b>\n\nSin señales en este momento.\nHora: {datetime.now().strftime('%H:%M')}\nActivos monitoreados: ADA, SOL, XRP"
            enviar_telegram(mensaje)

    print("\n[FINALIZADO]")

if __name__ == "__main__":
    main()
