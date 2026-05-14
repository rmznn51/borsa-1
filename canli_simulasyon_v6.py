import pandas as pd
import pandas_ta as ta
import joblib
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings("ignore")
engine = create_engine('sqlite:///borsa_verileri.db')

BARAJ = 0.30 
TICKERS = ['thyao', 'tuprs', 'eregl', 'asels', 'sise', 'akbnk', 'bimas', 'ekgyo', 'froto', 'garan', 'halkb', 'isctr', 'kchol', 'krdmd', 'petkm', 'pgsus', 'sahol', 'sasa', 'tcell', 'tkfen', 'toaso', 'ttkom', 'vakbn', 'ykbnk', 'arclk', 'enkai', 'gubrf', 'hekts', 'kontr', 'odas', 'oyakc', 'alark', 'enjsa', 'bryat', 'cimsa', 'doas', 'mavi', 'tavhl', 'kozal', 'otkar', 'ulker', 'vesbe', 'vestl', 'tknsa', 'skbnk', 'tskb', 'xu100']

# KOMİSYON VE KAYMA MALİYETİ (Örn: Binde 2 - Alış ve satış için toplam)
KOMISYON_ORANI = 0.002 

def ozellik_cikar(df):
    df.columns = [col.lower() for col in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['SMA_20'] = ta.sma(df['close'], length=20)
    macd = ta.macd(df['close'])
    df['MACD_Norm'] = (macd.iloc[:, 0] / df['close']) * 100 if macd is not None else 0
    df['MACD_Sig_Norm'] = (macd.iloc[:, 1] / df['close']) * 100 if macd is not None else 0
    df['Fiyat_Degisimi'] = df['close'].pct_change() * 100
    df['Mum_Gucu'] = ((df['close'] - df['open']) / df['open']) * 100
    df['Dalga_Boyu'] = ((df['high'] - df['low']) / df['low']) * 100
    df['Hacim_Degisimi'] = df['volume'].pct_change() * 100
    df['ATR_Norm'] = (df['ATR'] / df['close']) * 100
    df['SMA20_Uzaklik'] = ((df['close'] - df['SMA_20']) / df['SMA_20']) * 100
    return df.dropna()

def backtest_calistir(ticker):
    try:
        m_4h = joblib.load('konsey_modelleri/konsey_4h_model.joblib')
        m_1d = joblib.load('konsey_modelleri/konsey_1d_model.joblib')
        m_1w = joblib.load('konsey_modelleri/konsey_1w_model.joblib')
        m_1m = joblib.load('konsey_modelleri/konsey_1m_model.joblib')
        m_yargic = joblib.load('konsey_modelleri/yargic_ceo_v6.joblib')
    except: 
        return None

    toplam_pnl, toplam_islem, basarili = 0, 0, 0
    X_cols = ['Fiyat_Degisimi', 'Mum_Gucu', 'Dalga_Boyu', 'Hacim_Degisimi', 'ATR_Norm', 'RSI', 'SMA20_Uzaklik', 'MACD_Norm', 'MACD_Sig_Norm']

    try:
        d4, d1, dw, dm = [pd.read_sql(f"SELECT * FROM {ticker}_{p} ORDER BY date DESC LIMIT 1000", engine).iloc[::-1] for p in ['4h', '1d', '1w', '1m']]
        d4, d1, dw, dm = [ozellik_cikar(x) for x in [d4, d1, dw, dm]]
        
        common_dates = d1.index.intersection(d4.index).intersection(dw.index).intersection(dm.index)
        
        # YENİ EKLENEN KISIM: Sadece Out-of-Sample (Modelin eğitilirken görmediği) veride test et
        common_dates = common_dates[common_dates > '2025-12-31']
        
        for date in common_dates:
            p4 = m_4h.predict_proba(d4.loc[[date]][X_cols])[0, 1]
            p1 = m_1d.predict_proba(d1.loc[[date]][X_cols])[0, 1]
            pw = m_1w.predict_proba(dw.loc[[date]][X_cols])[0, 1]
            pm = m_1m.predict_proba(dm.loc[[date]][X_cols])[0, 1]
            
            ceo_in = pd.DataFrame([[p4, p1, pw, pm, d1.loc[date, 'ATR'], d1.loc[date, 'RSI']]], columns=['Sinyal_4H', 'Sinyal_1D', 'Sinyal_1W', 'Sinyal_1M', 'ATR', 'RSI'])
            ceo_score = m_yargic.predict_proba(ceo_in)[0, 1]

            if ceo_score > BARAJ: 
                toplam_islem += 1
                fiyat = d1.loc[date, 'close']
                try:
                    gelecek_fiyat = d1.iloc[d1.index.get_loc(date) + 1]['close']
                    
                    # YENİ EKLENEN KISIM: Komisyon Düşülmüş Gerçek Net Kâr/Zarar
                    maliyet = (fiyat + gelecek_fiyat) * (KOMISYON_ORANI / 2)
                    pnl = (gelecek_fiyat - fiyat) - maliyet
                    
                    toplam_pnl += pnl * 100
                    if pnl > 0: basarili += 1
                except: continue
        
        return {
            'kapanan_islem': toplam_islem,
            'basarili': basarili,
            'basarisiz': toplam_islem - basarili,
            'net_pnl': toplam_pnl
        }
    except Exception: return None

def genel_simulasyon():
    print(f"\n🚀 SİMÜLASYON BAŞLIYOR (BARAJ: {BARAJ} | OOS TESTİ: SADECE 2026 VE SONRASI) 🚀")
    genel_islem, genel_pnl, genel_basari = 0, 0, 0
    
    for ticker in TICKERS:
        res = backtest_calistir(ticker)
        if res and res['kapanan_islem'] > 0:
            genel_islem += res['kapanan_islem']
            genel_pnl += res['net_pnl']
            genel_basari += res['basarili']
            
    wr = (genel_basari / genel_islem * 100) if genel_islem > 0 else 0
    print(f"\n📊 GERÇEKÇİ OOS ÖZETİ: {genel_islem} İşlem | Net PnL: {genel_pnl:.2f} TL | Win Rate: %{wr:.2f}")

if __name__ == "__main__":
    genel_simulasyon()
