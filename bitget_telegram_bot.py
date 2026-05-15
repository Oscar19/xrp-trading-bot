"""
BOT DE ALERTAS BITGET -> TELEGRAM
=================================
Version para GitHub Actions con variables de entorno.
"""

import pandas as pd
import numpy as np
import ccxt
import ta
import json
import os
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================
# CONFIGURACION (desde variables de entorno o defaults)
# ============================

CONFIG = {
    'symbol': os.environ.get('BITGET_SYMBOL', 'XRP/USDT'),
    'timeframe': '5m',
    'leverage': int(os.environ.get('LEVERAGE', '5')),
    'risk_per_trade': float(os.environ.get('RISK_PER_TRADE', '0.02')),
    'stop_loss_pct': float(os.environ.get('STOP_LOSS', '0.015')),
    'take_profit_pct': float(os.environ.get('TAKE_PROFIT', '0.03')),
    'balance': float(os.environ.get('BALANCE', '1000')),

    # Telegram (OBLIGATORIO desde secrets)
    'telegram_token': os.environ.get('TELEGRAM_TOKEN', ''),
    'telegram_chat_id': os.environ.get('TELEGRAM_CHAT_ID', ''),
}

# ============================
# TELEGRAM
# ============================

def send_telegram(message):
    """Envia mensaje a Telegram."""
    if not CONFIG['telegram_token'] or not CONFIG['telegram_chat_id']:
        print("⚠️  No hay token o chat ID configurado")
        return False

    url = f"https://api.telegram.org/bot{CONFIG['telegram_token']}/sendMessage"

    payload = {
        'chat_id': CONFIG['telegram_chat_id'],
        'text': message,
        'parse_mode': 'HTML',
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return True
        else:
            print(f"❌ Error Telegram: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error enviando Telegram: {e}")
        return False

# ============================
# ANALISIS
# ============================

def analyze():
    """Analiza mercado y genera alerta si hay señal."""
    exchange = ccxt.bitget({'options': {'defaultType': 'swap'}})

    # Descargar datos
    ohlcv = exchange.fetch_ohlcv(CONFIG['symbol'], CONFIG['timeframe'], limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Indicadores
    df['ema9'] = ta.trend.EMAIndicator(df['close'], 9).ema_indicator()
    df['ema21'] = ta.trend.EMAIndicator(df['close'], 21).ema_indicator()
    typical = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical * df['volume']).cumsum() / df['volume'].cumsum()
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma20']

    # Condiciones
    prev = df.iloc[-2]
    prev_prev = df.iloc[-3]
    current = df.iloc[-1]

    cross_vwap = (prev_prev['close'] < prev_prev['vwap']) and (prev['close'] > prev['vwap'])
    volume_ok = prev['vol_ratio'] > 1.3 if not pd.isna(prev['vol_ratio']) else False
    momentum = prev['ema9'] > prev['ema21']

    # Calcular
    entry_price = current['close']
    sl_price = entry_price * (1 - CONFIG['stop_loss_pct'])
    tp_price = entry_price * (1 + CONFIG['take_profit_pct'])

    risk_amount = CONFIG['balance'] * CONFIG['risk_per_trade']
    notional = risk_amount / CONFIG['stop_loss_pct']
    margin = notional / CONFIG['leverage']
    liq_price = entry_price * (1 - 0.9/CONFIG['leverage'])

    return {
        'time': datetime.now().strftime('%H:%M:%S'),
        'price': round(entry_price, 4),
        'vwap': round(current['vwap'], 4),
        'vol_ratio': round(current['vol_ratio'], 2),
        'signal': cross_vwap and volume_ok and momentum,
        'entry': round(entry_price, 4),
        'sl': round(sl_price, 4),
        'tp': round(tp_price, 4),
        'margin': round(margin, 2),
        'liquidation': round(liq_price, 4),
        'conditions': {
            'cross_vwap': cross_vwap,
            'volume_ok': volume_ok,
            'momentum': momentum,
        }
    }

# ============================
# MENSAJES
# ============================

def build_alert_message(result):
    """Construye mensaje de alerta para Telegram."""
    emoji = "🟢" if result['signal'] else "🔴"

    msg = f"""{emoji} <b>ALERTA {CONFIG['symbol']}</b> {emoji}

⏰ <b>{result['time']} UTC</b>

📊 <b>MERCADO:</b>
   Precio: <code>${result['price']}</code>
   VWAP:   <code>${result['vwap']}</code>
   Vol:    <code>{result['vol_ratio']}x</code> media

📋 <b>CONDICIONES:</b>
   VWAP Cruz: {'✅' if result['conditions']['cross_vwap'] else '❌'}
   Volumen:    {'✅' if result['conditions']['volume_ok'] else '❌'}
   Momentum:   {'✅' if result['conditions']['momentum'] else '❌'}
"""

    if result['signal']:
        msg += f"""
🔥 <b>SEÑAL DE COMPRA</b> 🔥

📋 <b>ORDEN:</b>
   Entrada: <code>${result['entry']}</code>
   SL:      <code>${result['sl']}</code> ({CONFIG['stop_loss_pct']*100}%)
   TP:      <code>${result['tp']}</code> ({CONFIG['take_profit_pct']*100}%)

💰 <b>MARGEN:</b> <code>${result['margin']}</code> | {CONFIG['leverage']}x
⚠️ <b>LIQ:</b>   <code>${result['liquidation']}</code>

💡 Abre Bitget y coloca orden manual
"""
    else:
        msg += "
⏳ Sin señal de compra
"

    return msg

# ============================
# MAIN
# ============================

def main():
    print("🔔 BITGET ALERT BOT -> TELEGRAM")
    print(f"📡 Chat ID: {CONFIG['telegram_chat_id']}")
    print(f"⏳ Analizando {CONFIG['symbol']}...
")

    # Analizar
    result = analyze()

    # Construir mensaje
    message = build_alert_message(result)

    # Mostrar en terminal
    print(message)
    print("
" + "="*50)

    # Enviar a Telegram
    print("📤 Enviando a Telegram...")
    success = send_telegram(message)

    if success:
        print("✅ Alerta enviada!")

        # Guardar historial
        history = []
        if os.path.exists('results/alerts_telegram.json'):
            with open('results/alerts_telegram.json', 'r') as f:
                history = json.load(f)

        history.append({
            'time': datetime.now().isoformat(),
            'signal': result['signal'],
            'price': result['price'],
        })

        os.makedirs('results', exist_ok=True)
        with open('results/alerts_telegram.json', 'w') as f:
            json.dump(history, f, indent=2)
    else:
        print("❌ No se pudo enviar a Telegram")

    print("
✅ Bot finalizado")

if __name__ == "__main__":
    main()
