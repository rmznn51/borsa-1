import pandas as pd
import pandas_ta as ta
import numpy as np
from sqlalchemy import create_engine
from xgboost import XGBClassifier
import joblib
import os
import warnings

warnings.filterwarnings("ignore")
engine = create_engine('sqlite:///borsa_verileri.db')
TICKERS = ['thyao', 'tuprs', 'eregl', 'asels', 'sise', 'akbnk', 'bimas', 'ekgyo', 'froto', 'garan', 'halkb', 'isctr', 'kchol', 'krdmd', 'petkm', 'pgsus', 'sahol', 'sasa', 'tcell', 'tkfen', 'toaso', 'ttkom', 'vakbn', 'ykbnk', 'arclk', 'enkai', 'gubrf', 'hekts', 'kontr', 'odas', 'oyakc', 'alark', 'enjsa', 'bryat', 'cimsa', 'doas', 'mavi', 'tavhl', 'kozal','otkar', 'ulker', 'vesbe', 'vestl', 'tknsa', 'skbnk', 'tskb', 'xu100']

def guvenli_zaman_koprusu(df_gunluk, df_ust_tf, kolon_adi):
    ust_sinyal = df_ust_tf[kolon_adi].shift(1)
    ust_sinyal.name = kolon_adi
    ust_gunluk_yayilmis = ust_sinyal.resample('D').ffill()
    
    if kolon_adi in df_gunluk.columns: 
        df_gunluk.drop(columns=[kolon_adi], inplace=True)
        
    df_gunluk = df_gunluk.join(ust_gunluk_yayilmis)
    return df_gunluk

def ozellik_cikar(df):
    df.columns = [col.lower() for col in df.columns]
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['SMA_20'] = ta.sma(df['close'], length=20)
    macd = ta.macd(df['close'])
    if macd is not None:
        df['MACD'] = macd.iloc[:, 0]; df['MACD_Signal'] = macd.iloc[:, 1]
    else: df['MACD'] = 0; df['MACD_Signal'] = 0
    df['Fiyat_Degisimi'] = df['close'].pct_change() * 100
    df['Mum_Gucu'] = ((df['close'] - df['open']) / df['open']) * 100
    df['Dalga_Boyu'] = ((df['high'] - df['low']) / df['low']) * 100
    df['Hacim_Degisimi'] = df['volume'].pct_change() * 100
    df['ATR_Norm'] = (df['ATR'] / df['close']) * 100
    df['SMA20_Uzaklik'] = ((df['close'] - df['SMA_20']) / df['SMA_20']) * 100
    df['MACD_Norm'] = (df['MACD'] / df['close']) * 100
    df['MACD_Sig_Norm'] = (df['MACD_Signal'] / df['close']) * 100
    return df.dropna()

def ceo_egit():
    print("\n⚖️ YARGIÇ CEO (XGBOOST V6) İNŞA EDİLİYOR ⚖️")
    m_4h = joblib.load('konsey_modelleri/konsey_4h_model.joblib')
    m_1d = joblib.load('konsey_modelleri/konsey_1d_model.joblib')
    m_1w = joblib.load('konsey_modelleri/konsey_1w_model.joblib')
    m_1m = joblib.load('konsey_modelleri/konsey_1m_model.joblib')

    tum_veriler = pd.DataFrame()
    X_cols = ['Fiyat_Degisimi', 'Mum_Gucu', 'Dalga_Boyu', 'Hacim_Degisimi', 'ATR_Norm', 'RSI', 'SMA20_Uzaklik', 'MACD_Norm', 'MACD_Sig_Norm']
    
    for ticker in TICKERS:
        try:
            df_4h = pd.read_sql(f"SELECT * FROM {ticker}_4h", engine)
            df_1d = pd.read_sql(f"SELECT * FROM {ticker}_1d", engine)
            df_1w = pd.read_sql(f"SELECT * FROM {ticker}_1w", engine)
            df_1m = pd.read_sql(f"SELECT * FROM {ticker}_1m", engine)
        except: continue
        
        df_4h = ozellik_cikar(df_4h); df_4h.index = df_4h.index.normalize()
        df_1d = ozellik_cikar(df_1d); df_1d.index = df_1d.index.normalize()
        df_1w = ozellik_cikar(df_1w); df_1w.index = df_1w.index.normalize()
        df_1m = ozellik_cikar(df_1m); df_1m.index = df_1m.index.normalize()

        df_4h['Sinyal_4H'] = m_4h.predict_proba(df_4h[X_cols])[:, 1]
        df_1d['Sinyal_1D'] = m_1d.predict_proba(df_1d[X_cols])[:, 1]
        df_1w['Sinyal_1W'] = m_1w.predict_proba(df_1w[X_cols])[:, 1]
        df_1m['Sinyal_1M'] = m_1m.predict_proba(df_1m[X_cols])[:, 1]

        s_4h = df_4h.resample('D').last().ffill()['Sinyal_4H']
        s_4h.name = 'Sinyal_4H'
        df_1d = df_1d.join(s_4h)

        df_1d = guvenli_zaman_koprusu(df_1d, df_1w, 'Sinyal_1W')
        df_1d = guvenli_zaman_koprusu(df_1d, df_1m, 'Sinyal_1M')
        df_1d.dropna(inplace=True)

        df_1d['Gelecek_Getiri'] = df_1d['close'].shift(-3) / df_1d['close'] - 1 
        df_1d['Yargic_Hedefi'] = np.where(df_1d['Gelecek_Getiri'] > 0.02, 1, 0)
        tum_veriler = pd.concat([tum_veriler, df_1d.dropna()])

    egitim_verisi = tum_veriler[tum_veriler.index <= '2025-12-31']
    X_yargic = egitim_verisi[['Sinyal_4H', 'Sinyal_1D', 'Sinyal_1W', 'Sinyal_1M', 'ATR', 'RSI']]
    y_yargic = egitim_verisi['Yargic_Hedefi']

    # YENİ EKLENEN KISIM: Veri Dengesizliğini (Imbalance) çözmek için sınıf ağırlığını dinamik hesaplıyoruz.
    negatif_sayisi = len(y_yargic[y_yargic == 0])
    pozitif_sayisi = len(y_yargic[y_yargic == 1])
    dengesizlik_orani = negatif_sayisi / pozitif_sayisi if pozitif_sayisi > 0 else 1

    # scale_pos_weight XGBoost'a bu asimetriyi öğretir.
    yargic_modeli = XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05, 
        eval_metric='logloss', 
        scale_pos_weight=dengesizlik_orani, 
        random_state=42
    )
    yargic_modeli.fit(X_yargic, y_yargic)

    os.makedirs("konsey_modelleri", exist_ok=True)
    joblib.dump(yargic_modeli, 'konsey_modelleri/yargic_ceo_v6.joblib')
    print(f"👑 CEO (XGBOOST) BAŞARIYLA KAYDEDİLDİ! (Scale Pos Weight: {dengesizlik_orani:.2f})\n")

if __name__ == "__main__": ceo_egit()
