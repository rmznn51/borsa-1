import pandas as pd
import requests
import joblib
from sqlalchemy import create_engine
import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# .env dosyasından telegram bilgilerini çeker
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

engine = create_engine('sqlite:///borsa_verileri.db')

def telegram_mesaj_gonder(mesaj):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram Token veya Chat ID bulunamadı! (.env dosyanı kontrol et)")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': mesaj, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Gönderim hatası: {e}")

def canli_radar():
    print("\n📡 CANLI RADAR TARAMASI BAŞLADI (TELEGRAM) 📡")
    # Diğer dosyadan fonksiyonları ödünç alıyoruz ki kod tekrarı olmasın
    from canli_simulasyon_v6 import ozellik_cikar, TICKERS, BARAJ
    
    try:
        m_4h = joblib.load('konsey_modelleri/konsey_4h_model.joblib')
        m_1d = joblib.load('konsey_modelleri/konsey_1d_model.joblib')
        m_1w = joblib.load('konsey_modelleri/konsey_1w_model.joblib')
        m_1m = joblib.load('konsey_modelleri/konsey_1m_model.joblib')
        m_yargic = joblib.load('konsey_modelleri/yargic_ceo_v6.joblib')
    except: 
        print("❌ Modeller eksik! Önce modelleri eğitmelisin.")
        return

    X_cols = ['Fiyat_Degisimi', 'Mum_Gucu', 'Dalga_Boyu', 'Hacim_Degisimi', 'ATR_Norm', 'RSI', 'SMA20_Uzaklik', 'MACD_Norm', 'MACD_Sig_Norm']
    sinyaller = []

    for ticker in TICKERS:
        try:
            d4, d1, dw, dm = [pd.read_sql(f"SELECT * FROM {ticker}_{p} ORDER BY date DESC LIMIT 50", engine).iloc[::-1] for p in ['4h', '1d', '1w', '1m']]
            d4, d1, dw, dm = [ozellik_cikar(x) for x in [d4, d1, dw, dm]]
            
            son_tarih = d1.index[-1] # En güncel günlük mum kapanışı
            
            if son_tarih in d4.index and son_tarih in dw.index and son_tarih in dm.index:
                p4 = m_4h.predict_proba(d4.loc[[son_tarih]][X_cols])[0, 1]
                p1 = m_1d.predict_proba(d1.loc[[son_tarih]][X_cols])[0, 1]
                pw = m_1w.predict_proba(dw.loc[[son_tarih]][X_cols])[0, 1]
                pm = m_1m.predict_proba(dm.loc[[son_tarih]][X_cols])[0, 1]
                
                ceo_in = pd.DataFrame([[p4, p1, pw, pm, d1.loc[son_tarih, 'ATR'], d1.loc[son_tarih, 'RSI']]], columns=['Sinyal_4H', 'Sinyal_1D', 'Sinyal_1W', 'Sinyal_1M', 'ATR', 'RSI'])
                ceo_score = m_yargic.predict_proba(ceo_in)[0, 1]

                if ceo_score > BARAJ:
                    fiyat = d1.loc[son_tarih, 'close']
                    sinyaller.append(f"🟢 *{ticker.upper()}* | Fiyat: {fiyat:.2f} | CEO Güveni: %{ceo_score*100:.1f}")
        except: continue

    if sinyaller:
        mesaj = "🚀 *ÜÇLÜ KONSEY V6 - GÜNCEL SİNYALLER* 🚀\n\n" + "\n".join(sinyaller)
        print("Sinyaller bulundu, Telegram'a uçuruluyor...")
        telegram_mesaj_gonder(mesaj)
        print("✅ Telegram'a iletildi.")
    else:
        print("😴 Şu an CEO'nun barajını geçen hisse yok.")

if __name__ == "__main__":
    canli_radar()
