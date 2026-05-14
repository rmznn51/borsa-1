import subprocess, sys

def b(d, a):
    print(f"\n🚀 {a}..."); subprocess.run([sys.executable, d], check=False)

while True:
    print("\n🌟 BİST ÜÇLÜ KONSEY (V6) ANA KUMANDA 🌟")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("[1] Hasat Makinesi (Veri Güncelle)")
    print("[2] Modelleri Eğit (Konsey + CEO)")
    print("[3] Simülasyon Çalıştır (Backtest)")
    print("[4] PERFORMANS RAPORU (Tahsilatçı)")
    print("[5] CANLI RADAR (Telegram'a Gönder)")
    print("[0] Çıkış")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    s = input("👉 Seçimin: ")
    
    if s == '1': b("hasat_makinesi_v1.py", "Veri Çekimi")
    elif s == '2': b("konsey_fabrikasi_v6.py", "Üçlü Konsey"); b("ceo_egitimi_v6.py", "Yargıç CEO")
    elif s == '3': b("canli_simulasyon_v6.py", "Simülasyon")
    elif s == '4': b("performans_raporu_v6.py", "Performans Raporu")
    elif s == '5': b("telegram_postaci_v6.py", "Telegram Radar")
    elif s == '0': break
