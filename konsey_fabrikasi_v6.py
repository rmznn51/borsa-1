import pandas as pd
import pandas_ta as ta
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import concurrent.futures
import logging
import warnings
import os

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DB_URL = 'sqlite:///borsa_verileri.db'
engine = create_engine(DB_URL)

TICKERS = ['thyao', 'tuprs', 'eregl', 'asels', 'sise', 'akbnk', 'bimas', 'ekgyo', 'froto', 'garan', 'halkb', 'isctr', 'kchol', 'krdmd', 'petkm', 'pgsus', 'sahol', 'sasa', 'tcell', 'tkfen', 'toaso', 'ttkom', 'vakbn', 'ykbnk', 'arclk', 'enkai', 'gubrf', 'hekts', 'kontr', 'odas', 'oyakc', 'alark', 'enjsa', 'bryat', 'cimsa', 'doas', 'mavi', 'tavhl', 'kozal', 'otkar', 'ulker', 'vesbe', 'vestl', 'tknsa', 'skbnk', 'tskb', 'xu100']

MODELLER_AYAR = {
    '4h': {'isim': 'Konsey_4H', 'shift': -2, 'kar': 1.015, 'zarar': 0.985},
    '1d': {'isim': 'Konsey_1D', 'shift': -3, 'kar': 1.03,  'zarar': 0.97},
    '1w': {'isim': 'Konsey_1W', 'shift': -2, 'kar': 1.08,  'zarar': 0.92},
    '1m': {'isim': 'Konsey_1M', 'shift': -1, 'kar': 1.15,  'zarar': 0.85}
}

def ozellik_muhendisligi(df, settings):
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
    
    df['gelecek_fiyat'] = df['close'].shift(settings['shift'])
    df['Hedef'] = 0 
    df.loc[df['gelecek_fiyat'] > df['close'] * settings['kar'], 'Hedef'] = 1   
    df.loc[df['gelecek_fiyat'] < df['close'] * settings['zarar'], 'Hedef'] = -1  
    
    return df.dropna()

def konsey_egit(tf, settings):
    isim = settings['isim']
    logging.info(f"🏛️ [{isim}] ÜÇLÜ KONSEY TOPLANIYOR...")
    
    master_df = pd.DataFrame()
    for ticker in TICKERS:
        try:
            df = pd.read_sql(f"SELECT * FROM {ticker}_{tf}", engine)
            df = ozellik_muhendisligi(df, settings)
            master_df = pd.concat([master_df, df])
        except: pass
    
    if master_df.empty: return

    egitim_verisi = master_df[master_df.index <= '2025-12-31']

    X_cols = ['Fiyat_Degisimi', 'Mum_Gucu', 'Dalga_Boyu', 'Hacim_Degisimi', 'ATR_Norm', 'RSI', 'SMA20_Uzaklik', 'MACD_Norm', 'MACD_Sig_Norm']
    X = egitim_verisi[X_cols]
    y = egitim_verisi['Hedef']

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
    # YENİ EKLENEN KISIM: Hedef değişkeni -1, 0, 1 şeklinde multiclass olduğu için mlogloss kullanıldı.
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, eval_metric='mlogloss', random_state=42)
    svm_pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', probability=True, C=1.0, random_state=42))])

    konsey = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('svm', svm_pipe)], voting='soft')
    konsey.fit(X, y)
    
    os.makedirs("konsey_modelleri", exist_ok=True)
    joblib.dump(konsey, f"konsey_modelleri/{isim.lower()}_model.joblib")
    logging.info(f"✅ [{isim}] KONSEY EĞİTİLDİ VE KAYDEDİLDİ!")

if __name__ == "__main__":
    print("\n" + "="*55 + "\n🔥 V6 ÜÇLÜ KONSEY MODEL FABRİKASI 🔥\n" + "="*55)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for tf, settings in MODELLER_AYAR.items():
            executor.submit(konsey_egit, tf, settings)
