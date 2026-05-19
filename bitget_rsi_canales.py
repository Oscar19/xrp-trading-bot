"""
BOT RSI CANALES MULTI-TIMEFRAME
================================
Estrategia:
- 4h: Detectar canal descendente en RSI, esperar rebote en suelo
- 1h: Detectar diagonal verde (máximos decrecientes), esperar ruptura
- Señal: SOLO cuando ambas condiciones se alinean
"""

import pandas as pd
import numpy as np
import ccxt
import requests
import os
from datetime import datetime
import json

# CONFIGURACIÓN
CONFIG = {
    'symbol': os.environ.get('BITGET_SYMBOL', 'XRP/USDT'),
    'leverage': int(os.environ.get('LEVERAGE', '5')),
    'risk_per_trade': float(os.environ.get('RISK_PER_TRADE', '0.02')),
    'balance': float(os.environ.get('BALANCE', '1000')),
    'telegram_token': os.environ.get('TELEGRAM_TOKEN', ''),
    'telegram_chat_id': os.environ.get('TELEGRAM_CHAT_ID', ''),
    'demo_mode': os.environ.get('DEMO_MODE', 'true').lower() == 'true',
    'min_r2_canal': float(os.environ.get('MIN_R2_CANAL', '0.3')),
    'min_r2_diagonal': float(os.environ.get('MIN_R2_DIAGONAL', '0.3')),
    'umbral_rebote': float(os.environ.get('UMBRAL_REBOTE', '1.15')),
}

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def send_telegram(message):
    if not CONFIG['telegram_token'] or not CONFIG['telegram_chat_id']:
        log("No hay token o chat ID de Telegram")
        return False
    
    url = f"https://api.telegram.org/bot{CONFIG['telegram_token']}/sendMessage"
    payload = {
        'chat_id': CONFIG['telegram_chat_id'],
        'text': message,
        'parse_mode': 'HTML',
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        log(f"Error Telegram: {e}")
        return False

def fetch_data(exchange, symbol, timeframe, limit=500):
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
    """Calcular RSI manualmente"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detectar_canal_4h(rsi_values, ventana_pivot=3, min_puntos=3, 
                       pendiente_max=-0.01, diff_max=0.3, r2_min=0.3):
    """
    Detectar canal descendente en RSI 4h
    """
    rsi = np.array(rsi_values)
    n = len(rsi)
    
    # Encontrar máximos locales (techo)
    techos_idx, techos_val = [], []
    for i in range(ventana_pivot, n - ventana_pivot):
        ventana = rsi[i-ventana_pivot:i+ventana_pivot+1]
        if rsi[i] == ventana.max() and rsi[i] > 45:
            techos_idx.append(i)
            techos_val.append(rsi[i])
    
    # Encontrar mínimos locales (suelo)
    suelos_idx, suelos_val = [], []
    for i in range(ventana_pivot, n - ventana_pivot):
        ventana = rsi[i-ventana_pivot:i+ventana_pivot+1]
        if rsi[i] == ventana.min() and rsi[i] < 60:
            suelos_idx.append(i)
            suelos_val.append(rsi[i])
    
    if len(techos_idx) < min_puntos or len(suelos_idx) < min_puntos:
        return None, f"Pocos pivots (techos={len(techos_idx)}, suelos={len(suelos_idx)})"
    
    # Ajustar líneas
    x_t = np.array(techos_idx[-8:])
    y_t = np.array(techos_val[-8:])
    x_s = np.array(suelos_idx[-8:])
    y_s = np.array(suelos_val[-8:])
    
    m_t, b_t, r_t, _, _ = np.polyfit(x_t, y_t, 1, full=False)
    # Usar scipy para R²
    from scipy import stats
    m_t, b_t, r_t, p_t, se_t = stats.linregress(x_t, y_t)
    m_s, b_s, r_s, p_s, se_s = stats.linregress(x_s, y_s)
    
    # Validaciones
    if m_t > pendiente_max:
        return None, f"Techo no descendente ({m_t:.4f} > {pendiente_max})"
    
    if abs(m_t - m_s) > diff_max:
        return None, f"Líneas no paralelas (diff={abs(m_t-m_s):.4f})"
    
    if r_t**2 < r2_min:
        return None, f"Correlación baja (R²={r_t**2:.3f})"
    
    return {
        'techo_m': m_t, 'techo_b': b_t,
        'suelo_m': m_s, 'suelo_b': b_s,
        'techo_r2': r_t**2, 'suelo_r2': r_s**2,
        'techos_idx': techos_idx, 'techos_val': techos_val,
        'suelos_idx': suelos_idx, 'suelos_val': suelos_val
    }, "OK"

def detectar_rebote_4h(rsi_values, canal, umbral_zona=1.15):
    """Detectar si RSI está en zona de rebote del canal"""
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
    """
    Detectar diagonal verde: máximos locales decrecientes en RSI 1h
    """
    from scipy import stats
    
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
        return None, f"Pocos máximos ({len(maximos_idx)})"
    
    # Buscar subconjunto descendente
    for n_usar in range(min_puntos, min(15, len(maximos_idx)) + 1):
        x = np.array(maximos_idx[-n_usar:])
        y = np.array(maximos_val[-n_usar:])
        m, b, r, _, _ = stats.linregress(x, y)
        
        if m <= pendiente_max and r**2 >= r2_min:
            return {
                'm': m, 'b': b, 'r2': r**2,
                'maximos_idx': maximos_idx, 'maximos_val': maximos_val,
                'n_usados': n_usar
            }, "OK"
    
    # Fallback
    x = np.array(maximos_idx[-min_puntos:])
    y = np.array(maximos_val[-min_puntos:])
    m, b, r, _, _ = stats.linregress(x, y)
    if m <= 0.05:
        return {
            'm': m, 'b': b, 'r2': r**2,
            'maximos_idx': maximos_idx, 'maximos_val': maximos_val,
            'n_usados': min_puntos
        }, "OK (fallback)"
    
    return None, f"Tendencia ascendente ({m:.4f})"

def detectar_ruptura_1h(rsi_values, diagonal, umbral=0.02):
    """Detectar ruptura de diagonal hacia arriba"""
    if not diagonal:
        return False, None
    
    n = len(rsi_values)
    val_diag_2 = diagonal['m'] * (n-2) + diagonal['b']
    val_diag_1 = diagonal['m'] * (n-1) + diagonal['b']
    
    rsi_2 = rsi_values[-2]
    rsi_1 = rsi_values[-1]
    
    # Ruptura: anterior <= diagonal, actual > diagonal
    ruptura = rsi_2 <= val_diag_2 and rsi_1 > val_diag_1
    
    if ruptura:
        return True, {
            'rsi_antes': rsi_2, 'rsi_despues': rsi_1,
            'diag_antes': val_diag_2, 'diag_despues': val_diag_1,
            'diferencia': rsi_1 - val_diag_1
        }
    
    # También si ya está arriba desde hace rato
    val_diag_3 = diagonal['m'] * (n-3) + diagonal['b']
    if rsi_1 > val_diag_1 and len(rsi_values) > 3 and rsi_values[-3] > val_diag_3:
        return True, {
            'rsi_antes': rsi_values[-3], 'rsi_despues': rsi_1,
            'diag_antes': val_diag_3, 'diag_despues': val_diag_1,
            'diferencia': rsi_1 - val_diag_1,
            'nota': 'Ruptura previa confirmada'
        }
    
    return False, None

def analizar():
    """Análisis completo multi-timeframe"""
    log("="*60)
    log("INICIANDO ANÁLISIS RSI CANALES")
    log("="*60)
    
    # Conectar a Bitget
    try:
        exchange = ccxt.bitget({
            'options': {'defaultType': 'swap'},
            'timeout': 30000,
            'enableRateLimit': True
        })
        log("✅ Conectado a Bitget")
    except Exception as e:
        log(f"❌ Error conectando a Bitget: {e}")
        return None
    
    # Descargar datos
    log(f"📥 Descargando datos {CONFIG['symbol']}...")
    
    df_4h = fetch_data(exchange, CONFIG['symbol'], '4h', limit=200)
    df_1h = fetch_data(exchange, CONFIG['symbol'], '1h', limit=400)
    
    if df_4h is None or df_1h is None:
        log("❌ Error descargando datos")
        return None
    
    log(f"   4h: {len(df_4h)} velas | {df_4h.index[0]} → {df_4h.index[-1]}")
    log(f"   1h: {len(df_1h)} velas | {df_1h.index[0]} → {df_1h.index[-1]}")
    
    # Calcular RSI
    df_4h['rsi'] = calcular_rsi(df_4h['close'])
    df_1h['rsi'] = calcular_rsi(df_1h['close'])
    
    log(f"   RSI 4h actual: {df_4h['rsi'].iloc[-1]:.2f}")
    log(f"   RSI 1h actual: {df_1h['rsi'].iloc[-1]:.2f}")
    
    # ANÁLISIS 4h
    log("\n🔍 ANÁLISIS 4h:")
    canal_4h, msg_4h = detectar_canal_4h(df_4h['rsi'].dropna().values)
    
    if not canal_4h:
        log(f"   ❌ No hay canal: {msg_4h}")
        return {'señal': False, 'razon': msg_4h}
    
    log(f"   ✅ Canal detectado (R² techo={canal_4h['techo_r2']:.3f})")
    
    rebote_4h, info_4h = detectar_rebote_4h(df_4h['rsi'].dropna().values, canal_4h)
    
    if not rebote_4h:
        log(f"   ❌ No en zona de rebote (RSI={info_4h['rsi_actual']:.1f}, suelo={info_4h['suelo']:.1f})")
        return {'señal': False, 'razon': '4h no en rebote', 'info_4h': info_4h}
    
    log(f"   ✅ En zona de rebote! (RSI={info_4h['rsi_actual']:.1f}, suelo={info_4h['suelo']:.1f})")
    
    # ANÁLISIS 1h
    log("\n🔍 ANÁLISIS 1h:")
    diagonal_1h, msg_1h = detectar_diagonal_1h(df_1h['rsi'].dropna().values)
    
    if not diagonal_1h:
        log(f"   ❌ No hay diagonal: {msg_1h}")
        return {'señal': False, 'razon': msg_1h, 'info_4h': info_4h}
    
    log(f"   ✅ Diagonal detectada (R²={diagonal_1h['r2']:.3f}, {diagonal_1h['n_usados']} puntos)")
    
    ruptura_1h, info_1h = detectar_ruptura_1h(df_1h['rsi'].dropna().values, diagonal_1h)
    
    if not ruptura_1h:
        log(f"   ❌ Sin ruptura (RSI={df_1h['rsi'].iloc[-1]:.1f})")
        return {'señal': False, 'razon': '1h no rompe diagonal', 'info_4h': info_4h, 'diagonal_1h': diagonal_1h}
    
    log(f"   ✅ RUPTURA! (+{info_1h['diferencia']:.1f} puntos)")
    
    # SEÑAL COMPLETA
    entry = df_1h['close'].iloc[-1]
    sl = df_1h['low'].iloc[-5:].min()
    ratio_tp = info_4h['techo'] / info_4h['rsi_actual']
    tp = entry * (ratio_tp * 0.9)
    
    # Calcular margen
    risk_amount = CONFIG['balance'] * CONFIG['risk_per_trade']
    notional = risk_amount / 0.015  # SL 1.5%
    margin = notional / CONFIG['leverage']
    
    señal = {
        'señal': True,
        'direccion': 'largo',
        'entry': round(entry, 4),
        'sl': round(sl, 4),
        'tp': round(tp, 4),
        'margin': round(margin, 2),
        'info_4h': info_4h,
        'info_1h': info_1h,
        'timestamp': datetime.now().isoformat()
    }
    
    log(f"\n{'='*60}")
    log("🚀🚀🚀 SEÑAL DETECTADA 🚀🚀🚀")
    log(f"{'='*60}")
    log(f"   Entrada: ${señal['entry']}")
    log(f"   SL: ${señal['sl']}")
    log(f"   TP: ${señal['tp']}")
    log(f"   Margen: ${señal['margin']} | {CONFIG['leverage']}x")
    
    return señal

def build_message(result):
    """Construir mensaje para Telegram"""
    if not result['señal']:
        return None
    
    lines = []
    lines.append(f"🚀 <b>SEÑAL RSI CANALES - {CONFIG['symbol']}</b>")
    lines.append(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append("")
    lines.append(f"📊 <b>ANÁLISIS:</b>")
    lines.append(f"   4h: RSI en rebote de canal (RSI={result['info_4h']['rsi_actual']:.1f})")
    lines.append(f"   1h: RSI rompió diagonal verde (+{result['info_1h']['diferencia']:.1f} pts)")
    lines.append("")
    lines.append(f"📋 <b>ORDEN LARGO:</b>")
    lines.append(f"   Entrada: ${result['entry']}")
    lines.append(f"   SL: ${result['sl']}")
    lines.append(f"   TP: ${result['tp']}")
    lines.append(f"   Margen: ${result['margin']} | {CONFIG['leverage']}x")
    lines.append("")
    lines.append(f"⚠️ Modo: {'DEMO' if CONFIG['demo_mode'] else 'LIVE'}")
    lines.append(f"💡 Abre Bitget → Futuros {CONFIG['symbol']} → Comprar/Largo")
    
    return "\n".join(lines)

def main():
    log("="*60)
    log("BOT RSI CANALES - Bitget")
    log(f"Par: {CONFIG['symbol']} | Balance sim: ${CONFIG['balance']}")
    log(f"Modo: {'DEMO' if CONFIG['demo_mode'] else 'LIVE'}")
    log("="*60)
    
    result = analizar()
    
    if result and result['señal']:
        message = build_message(result)
        log(f"\nMensaje:\n{message}")
        
        if not CONFIG['demo_mode']:
            log("Enviando alerta a Telegram...")
            success = send_telegram(message)
            if success:
                log("✅ Alerta enviada!")
            else:
                log("❌ Error enviando alerta")
        else:
            log("🎓 Modo DEMO: No se envía a Telegram")
    else:
        razon = result['razon'] if result else 'Error'
        log(f"\n⏳ Sin señal: {razon}")
    
    log("="*60)
    log("Bot finalizado")
    log("="*60)

if __name__ == "__main__":
    main()
