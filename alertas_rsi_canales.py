"""
BOT RSI CANALES - ALERTAS v3.1
==============================
Mejoras:
- Informa tendencia del activo (alcista/bajista/lateral)
- Añade ETH/USDT
- Estrategias en PARALELO: EMA_CRUCE + RSI_CANALES
"""

import pandas as pd
import numpy as np
import ccxt
import os
import requests
from datetime import datetime
from scipy import stats

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
MODO_TEST = os.environ.get('MODO_TEST', 'false').lower() == 'true'

ACTIVOS = {
    'ADA/USDT': {'nombre': 'ADA'},
    'SOL/USDT': {'nombre': 'SOL'},
    'XRP/USDT': {'nombre': 'XRP'},
    'ETH/USDT': {'nombre': 'ETH'},
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
    'zona_sobreventa': 45,
    'ema_rapida': 9,
    'ema_lenta': 21,
    'ema_tendencia': 50,       # Para detectar tendencia general
    'cooldown_horas': 8,
}

# ============================================================================
# TELEGRAM
# ============================================================================

def enviar_telegram(mensaje):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[SKIP] Telegram no configurado")
        return False

    try:
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
            print(f"[ERROR] Telegram {response.status_code}: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"[ERROR] Telegram: {e}")
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
# TENDENCIA
# ============================================================================

def detectar_tendencia(df, idx):
    """
    Detecta tendencia basada en EMAs:
    - Alcista: precio > EMA50 y EMA9 > EMA21
    - Bajista: precio < EMA50 y EMA9 < EMA21
    - Lateral: mixto o cerca de EMA50
    """
    if idx < 50:
        return 'N/A', '⚪'

    precio = df['close'].iloc[idx]
    ema9 = df['ema9'].iloc[idx] if 'ema9' in df.columns else None
    ema21 = df['ema21'].iloc[idx] if 'ema21' in df.columns else None
    ema50 = df['ema50'].iloc[idx] if 'ema50' in df.columns else None

    if ema50 is None:
        return 'N/A', '⚪'

    # Determinar tendencia
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
# ESTRATEGIA 1: EMA CRUCE
# ============================================================================

def estrategia_ema_cruce(exchange, symbol, config_activo):
    """Estrategia EMA 9/21 cruce alcista"""
    print(f"\n  [EMA_CRUCE] {symbol}...")

    df = fetch_data(exchange, symbol, CONFIG['timeframe'], limit=60)
    if df is None or len(df) < 30:
        return None, "Sin datos", None

    df['ema9'] = df['close'].ewm(span=CONFIG['ema_rapida'], adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=CONFIG['ema_lenta'], adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=CONFIG['ema_tendencia'], adjust=False).mean()

    idx = len(df) - 1
    ema9 = df['ema9'].values
    ema21 = df['ema21'].values

    # Tendencia
    tendencia, emoji_tend = detectar_tendencia(df, idx)

    # Cruce alcista: ema9[-2] < ema21[-2] y ema9[-1] > ema21[-1]
    if not (ema9[idx-1] < ema21[idx-1] and ema9[idx] > ema21[idx]):
        return None, f"Sin cruce (Tendencia: {tendencia} {emoji_tend})", (tendencia, emoji_tend)

    # Volumen
    volumen_actual = df['volume'].iloc[idx]
    volumen_media = df['volume'].iloc[max(0, idx - CONFIG['volumen_lookback']):idx].mean()
    ratio_volumen = volumen_actual / volumen_media if volumen_media > 0 else 0

    if ratio_volumen < CONFIG['volumen_min_ratio']:
        return None, f"Volumen insuficiente ({ratio_volumen:.1f}x) | Tendencia: {tendencia} {emoji_tend}", (tendencia, emoji_tend)

    precio = df['close'].iloc[idx]

    print(f"  🟢 SEÑAL EMA: {symbol} @ ${precio:.4f} | Tendencia: {tendencia}")

    return {
        'symbol': symbol,
        'nombre': config_activo['nombre'],
        'precio': precio,
        'rsi': 0,
        'sl': precio * (1 - CONFIG['sl_pct']),
        'tp': precio * (1 + CONFIG['tp_pct']),
        'estrategia': 'EMA_CRUCE',
        'ratio_volumen': ratio_volumen,
        'volumen_confirmado': True,
        'ema9': ema9[idx],
        'ema21': ema21[idx],
        'tendencia': tendencia,
        'tendencia_emoji': emoji_tend,
    }, "OK", (tendencia, emoji_tend)

# ============================================================================
# ESTRATEGIA 2: RSI CANALES
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
            'diag_antes': val_diag_2,
            'diag_despues': val_diag_1,
            'condicion_base': condicion_base,
            'momentum': momentum,
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

def estrategia_rsi_canales(exchange, symbol, config_activo):
    """Estrategia ruptura diagonal RSI"""
    print(f"\n  [RSI_CANALES] {symbol}...")

    df = fetch_data(exchange, symbol, CONFIG['timeframe'], limit=100)
    if df is None or len(df) < 50:
        return None, "Sin datos", None

    df['rsi'] = calcular_rsi(df['close'], period=CONFIG['rsi_period'])
    df['ema50'] = df['close'].ewm(span=CONFIG['ema_tendencia'], adjust=False).mean()

    rsi_values = df['rsi'].values
    idx = len(df) - 1
    rsi_actual = rsi_values[idx]
    precio_actual = df['close'].iloc[idx]

    # Tendencia
    tendencia, emoji_tend = detectar_tendencia(df, idx)

    if rsi_actual >= CONFIG['zona_sobreventa']:
        return None, f"RSI {rsi_actual:.1f} >= {CONFIG['zona_sobreventa']} | Tendencia: {tendencia} {emoji_tend}", (tendencia, emoji_tend)

    diagonal = detectar_diagonal_maximos_rsi(
        rsi_values,
        ventana_pivot=CONFIG['pivot_ventana'],
        min_puntos=CONFIG['min_puntos_diagonal'],
        pendiente_max=-0.01,
        r2_min=CONFIG['r2_min_diagonal']
    )

    if not diagonal:
        return None, f"No hay diagonal | Tendencia: {tendencia} {emoji_tend}", (tendencia, emoji_tend)

    ruptura, info = detectar_ruptura_diagonal_long(
        rsi_values, diagonal, df, idx,
        umbral_ruptura=CONFIG['umbral_ruptura'],
        volumen_ratio_min=CONFIG['volumen_min_ratio'],
        volumen_lookback=CONFIG['volumen_lookback']
    )

    if info is None:
        return None, f"Info=None | Tendencia: {tendencia} {emoji_tend}", (tendencia, emoji_tend)

    if not ruptura:
        motivo = []
        if not info['condicion_base']:
            motivo.append(f"RSI no cruzó diagonal")
        if not info['momentum']:
            motivo.append("Sin momentum")
        return None, f"{' | '.join(motivo)} | Tendencia: {tendencia} {emoji_tend}", (tendencia, emoji_tend)

    if not info.get('volumen_confirmado', False):
        return None, f"Volumen insuficiente ({info['ratio_volumen']:.1f}x) | Tendencia: {tendencia} {emoji_tend}", (tendencia, emoji_tend)

    print(f"  🟢 SEÑAL RSI: {symbol} @ ${precio_actual:.4f} | Tendencia: {tendencia}")

    return {
        'symbol': symbol,
        'nombre': config_activo['nombre'],
        'precio': precio_actual,
        'rsi': rsi_actual,
        'sl': precio_actual * (1 - CONFIG['sl_pct']),
        'tp': precio_actual * (1 + CONFIG['tp_pct']),
        'estrategia': 'RSI_CANALES',
        'tendencia': tendencia,
        'tendencia_emoji': emoji_tend,
        **info
    }, "OK", (tendencia, emoji_tend)

# ============================================================================
# ORQUESTADOR
# ============================================================================

ESTRATEGIAS = {
    'EMA_CRUCE': estrategia_ema_cruce,
    'RSI_CANALES': estrategia_rsi_canales,
}

# ============================================================================
# FORMATO ALERTAS
# ============================================================================

def formatear_alerta(señal):
    nombre = señal['nombre']
    symbol = señal['symbol']
    precio = señal['precio']
    estrategia = señal['estrategia']
    tendencia = señal.get('tendencia', 'N/A')
    tend_emoji = señal.get('tendencia_emoji', '⚪')

    sl_pct = CONFIG['sl_pct'] * 100
    tp_pct = CONFIG['tp_pct'] * 100
    leverage = CONFIG['leverage']

    sl_precio = señal['sl']
    tp_precio = señal['tp']

    pnl_sl = -sl_pct * leverage
    pnl_tp = tp_pct * leverage

    if estrategia == 'EMA_CRUCE':
        emoji = "📈"
        icono_est = "📊"
        detalle = f"EMA9({señal['ema9']:.4f}) > EMA21({señal['ema21']:.4f})"
    else:
        emoji = "🟢"
        icono_est = "📉"
        detalle = f"Ruptura diagonal RSI | R²={señal['r2_diagonal']:.2f}, {señal['n_puntos']} pts"

    mensaje = f"""{emoji} <b>ALERTA LONG - {nombre}</b>

📊 <b>{symbol}</b> {icono_est} <code>{estrategia}</code>
{tend_emoji} <b>Tendencia:</b> {tendencia}
💰 Precio: <code>${precio:.4f}</code>
📈 RSI: <code>{señal['rsi']:.1f}</code>
📊 Volumen: <code>{señal['ratio_volumen']:.1f}x</code> media
📉 {detalle}

🎯 <b>TRADE SETUP</b>
🟥 SL: <code>${sl_precio:.4f}</code> ({sl_pct:.1f}%)
🟩 TP: <code>${tp_precio:.4f}</code> ({tp_pct:.1f}%)
⚡ Apalancamiento: <code>{leverage}x</code>

💵 P&L estimado:
   SL: {pnl_sl:.1f}% | TP: +{pnl_tp:.1f}%

⏰ Timeframe: 4H
🔔 Alerta: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"""

    return mensaje

def formatear_resumen(resultados, hora):
    lineas = ["🔍 <b>Resumen Alertas v3.1</b>", ""]

    total_señales = 0
    for symbol, estrategias in resultados.items():
        # Obtener tendencia del primer análisis disponible
        tendencia = "N/A"
        tend_emoji = "⚪"
        for nombre_est, (señal, motivo) in estrategias.items():
            if señal and 'tendencia' in señal:
                tendencia = señal['tendencia']
                tend_emoji = señal.get('tendencia_emoji', '⚪')
                break
            elif isinstance(motivo, tuple):
                pass  # motivo es (tendencia, emoji) en algunos casos

        lineas.append(f"📊 <b>{symbol}</b> {tend_emoji} {tendencia}:")
        for nombre_est, (señal, motivo) in estrategias.items():
            if señal:
                emoji = "📈" if nombre_est == 'EMA_CRUCE' else "🟢"
                lineas.append(f"   {emoji} {nombre_est}: <b>SEÑAL</b> @ ${señal['precio']:.4f}")
                total_señales += 1
            else:
                # Extraer solo el mensaje, no la tupla de tendencia
                if isinstance(motivo, tuple):
                    msg = motivo[0] if len(motivo) > 0 else str(motivo)
                else:
                    msg = str(motivo)
                lineas.append(f"   ⚪ {nombre_est}: {msg[:40]}")
        lineas.append("")

    lineas.append(f"📈 Total señales: {total_señales}")
    lineas.append(f"⏰ {hora.strftime('%Y-%m-%d %H:%M UTC')}")
    lineas.append("📊 Activos: ADA, SOL, XRP, ETH")
    lineas.append("📈 Estrategias: EMA_CRUCE + RSI_CANALES")

    return "\n".join(lineas)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("BOT RSI CANALES - ALERTAS v3.1")
    print("Tendencia + ETH añadido")
    print("Estrategias: EMA_CRUCE + RSI_CANALES")
    print(f"Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

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

    resultados = {}
    todas_señales = []

    for symbol, config in ACTIVOS.items():
        print(f"\n{'='*40}")
        print(f"ANALIZANDO: {symbol}")
        print(f"{'='*40}")

        resultados[symbol] = {}

        for nombre_est, funcion in ESTRATEGIAS.items():
            señal, motivo, tendencia_info = funcion(exchange, symbol, config)
            resultados[symbol][nombre_est] = (señal, motivo)

            if señal:
                todas_señales.append(señal)

    print(f"\n{'='*60}")
    print(f"TOTAL SEÑALES: {len(todas_señales)}")
    print(f"{'='*60}")

    # Enviar resumen SIEMPRE
    hora = datetime.now()
    resumen = formatear_resumen(resultados, hora)
    enviar_telegram(resumen)

    # Enviar alertas individuales
    for señal in todas_señales:
        mensaje = formatear_alerta(señal)
        enviar_telegram(mensaje)

    # Test mode
    if MODO_TEST:
        test_msg = "🧪 <b>TEST MODE v3.1</b>\n\nEstrategias paralelas activas:\n• EMA_CRUCE\n• RSI_CANALES\n• Tendencia incluida\n• ETH añadido\n\n" + hora.strftime('%H:%M UTC')
        enviar_telegram(test_msg)

    print("\n[FINALIZADO]")

if __name__ == "__main__":
    main()
