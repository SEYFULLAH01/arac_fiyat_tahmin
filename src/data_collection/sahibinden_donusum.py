"""
Sahibinden ham CSV (yalnızca fiyat) → proje ile uyumlu minimal şemaya dönüştürme.

Kullanım:
python -c "from src.data_collection.sahibinden_donusum import donustur; donustur('data/raw/sahibinden_XXXX.csv')"
"""

from pathlib import Path
import pandas as pd


def donustur(girdi: str, cikti: str = "data/kaynak_sahibinden_min.csv") -> str:
    df = pd.read_csv(girdi)
    if 'fiyat_tl' not in df.columns:
        raise ValueError("Beklenen kolon bulunamadı: fiyat_tl")

    out = pd.DataFrame({
        "marka": pd.Series(dtype=str),
        "seri": pd.Series(dtype=str),
        "model": pd.Series(dtype=str),
        "model_yili": pd.Series(dtype='float'),
        "kilometre": pd.Series(dtype='float'),
        "motor_hacmi": pd.Series(dtype='float'),
        "yakit_turu": pd.Series(dtype=str),
        "vites_turu": pd.Series(dtype=str),
        "kasatipi": pd.Series(dtype=str),
        "motorgucu": pd.Series(dtype='float'),
        "cekis": pd.Series(dtype=str),
        "fiyat": df["fiyat_tl"].astype(float)
    })

    Path(cikti).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cikti, index=False, encoding="utf-8")
    print(f"Dönüştürülen dosya: {cikti} | Kayıt: {len(out)}")
    return cikti


if __name__ == "__main__":
    pass


