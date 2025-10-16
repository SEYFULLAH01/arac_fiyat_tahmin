"""
Sahibinden entegrasyonu: Arama sonuç URL'inden fiyatları çekip CSV'ye kaydeder.

Kullanım:
1) Önce kütüphaneyi kurun:
   pip install git+https://github.com/keremkoseoglu/sahibinden.git
2) Bu dosyayı çalıştırın (URL'i değiştirin):
   python src/data_collection/sahibinden_entegrasyon.py
"""

from pathlib import Path
import csv
import datetime as dt
from typing import List


def _import_spider():
    try:
        # Paket yapısında function değil module; gerçek fonksiyon search.search
        import sahibinden.search as s  # type: ignore
        if hasattr(s, "search"):
            return s.search  # type: ignore
        # Alternatif API: s.run veya benzeri olursa destekle
        raise ImportError("search fonksiyonu bulunamadı")
    except Exception as exc:
        raise RuntimeError(
            "sahibinden paketi bulunamadı. Kurulum: pip install git+https://github.com/keremkoseoglu/sahibinden.git"
        ) from exc


def fiyatlari_cek(url: str, post_sleep: int = 2) -> List[float]:
    search = _import_spider()
    prices = search(url, post_sleep=post_sleep)
    # Sayfa bazı durumlarda string döndürebilir; güvenli dönüştürme
    out: List[float] = []
    for p in prices:
        try:
            out.append(float(str(p).replace(".", "").replace(",", ".")))
        except Exception:
            continue
    return out


def kaydet_csv(url: str, fiyatlar: List[float]) -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path("data/raw") / f"sahibinden_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kaynak", "toplama_tarihi", "arama_url", "fiyat_tl"])
        for fyt in fiyatlar:
            writer.writerow(["sahibinden", ts, url, fyt])
    return str(out_path)


if __name__ == "__main__":
    # Hazır filtre: Tüm otomobil ilanları (geniş liste)
    # İsterseniz daha spesifik bir arama linkiyle değiştirin (marka/seri/model/yıl).
    URL = "https://www.sahibinden.com/otomobil"
    fiyat_listesi = fiyatlari_cek(URL, post_sleep=2)
    dosya = kaydet_csv(URL, fiyat_listesi)
    print(f"Kaydedilen dosya: {dosya} | Toplam fiyat: {len(fiyat_listesi)}")


