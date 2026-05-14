from tvDatafeed import TvDatafeed, Interval
import pandas as pd
from sqlalchemy import create_engine
import time
import random
import logging
import os
from dotenv import load_dotenv

# Loglama: Ekranda ne olup bittiğini şık bir şekilde görmek için
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- 1. VERİTABANI AYARI (KURULUM GEREKTİRMEZ) ---
# Veriler kodun olduğu klasöre "borsa_verileri.db" dosyası olarak kaydedilecek.
DB_URL = 'sqlite:///borsa_verileri.db'
engine = create_engine(DB_URL)

# --- 2. TRADINGVIEW BİLGİLERİN (GÜVENLİ KASADAN ÇEKİLİYOR) ---
load_dotenv() # .env dosyasındaki gizli şifreleri sisteme yükle

TV_USERNAME = os.getenv("TV_USERNAME")
TV_PASSWORD = os.getenv("TV_PASSWORD")

logging.info("TradingView'a bağlanılıyor...")
try:
    tv = TvDatafeed(TV_USERNAME, TV_PASSWORD)
    logging.info("✅ TradingView bağlantısı başarılı!")
except Exception as e:
    logging.error(f"❌ Bağlantı hatası: {e}")
    exit()

# --- 3. İZLENECEK HİSRELER VE ZAMAN DİLİMLERİ ---
TICKERS = [
    'thyao', 'tuprs', 'eregl', 'asels', 'sise', 'akbnk', 'bimas', 'ekgyo', 'froto', 'garan',
    'halkb', 'isctr', 'kchol', 'krdmd', 'petkm', 'pgsus', 'sahol', 'sasa', 'tcell', 'tkfen', 
    'toaso', 'ttkom', 'vakbn', 'ykbnk', 'arclk', 'enkai', 'gubrf', 'hekts', 'kontr', 'odas', 
    'oyakc', 'alark', 'enjsa', 'bryat', 'cimsa', 'doas', 'mavi', 'tavhl', 'kozal',      'otkar', 'ulker', 'vesbe', 'vestl', 'tknsa', 'skbnk', 'tskb', 'xu100'
]
EXCHANGE = 'BIST'

TIMEFRAMES = {
    '4h': (Interval.in_4_hour, 1000),  # Avcı için 4H mumlar
    '1d': (Interval.in_daily, 1500),   # Sörfçü için Günlük mumlar
    '1w': (Interval.in_weekly, 800),   # Balina için Haftalık mumlar
    '1m': (Interval.in_monthly, 300)   # Kasa için Aylık mumlar
}

def veri_cek(ticker, exchange, interval, n_bars, retries=3):
    """Ban yememek için hata durumunda bekleyip tekrar deneyen kalkan"""
    for attempt in range(retries):
        try:
            logging.info(f"[{ticker}] {interval.name} verisi çekiliyor...")
            data = tv.get_hist(symbol=ticker, exchange=exchange, interval=interval, n_bars=n_bars)
            
            if data is not None and not data.empty:
                data.reset_index(inplace=True)
                data.rename(columns={'datetime': 'date'}, inplace=True)
                if 'symbol' in data.columns:
                    data.drop('symbol', axis=1, inplace=True)
                return data
            else:
                logging.warning(f"[{ticker}] Veri boş. Limitlere takılmış olabiliriz.")
        except Exception as e:
            logging.error(f"[{ticker}] Hata: {e}")
            
        time.sleep(random.uniform(10, 15)) # Hata alırsak 5-10 saniye bekle
    return None

def hasat_basla():
    print("\n" + "="*50)
    print("🚜 DEVASA HASAT MAKİNESİ ÇALIŞTIRILDI 🚜")
    print("="*50 + "\n")
    
    for ticker in TICKERS:
        for tf_name, (interval, n_bars) in TIMEFRAMES.items():
            df = veri_cek(ticker, EXCHANGE, interval, n_bars)
            
            if df is not None:
                table_name = f"{ticker.lower()}_{tf_name}"
                df.to_sql(table_name, con=engine, if_exists='replace', index=False)
                logging.info(f"💾 {table_name} veritabanına kaydedildi. ({len(df)} Satır)")
            
            time.sleep(random.uniform(1, 3)) # Alt zaman dilimlerinde kısa mola
            
        logging.info(f"⏳ {ticker} tamamlandı. Diğer hisseye geçmeden önce dinleniyoruz...\n")
        time.sleep(random.uniform(5, 8)) # Hisse geçişlerinde uzun mola

    print("\n" + "="*50)
    print("✅ TÜM VERİLER BAŞARIYLA ÇEKİLDİ VE KASAYA ALINDI!")
    print("="*50 + "\n")

if __name__ == "__main__":
    hasat_basla()
