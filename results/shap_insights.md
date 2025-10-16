# SHAP İçgörü Raporu

Bu rapor, modelin tahminlerine en çok etki eden özellikleri ve yönlerini kısaca açıklar.

## En Etkili Özellikler ve Yönleri
- Güç / hacim: değer arttıkça fiyat artma eğiliminde
- Model yılı: değer arttıkça fiyat artma eğiliminde
- Araç yaşı: değer arttıkça fiyat düşme eğiliminde
- Model: değer arttıkça fiyat artma eğiliminde
- Cekis: değer arttıkça fiyat düşme eğiliminde
- Motor gücü (HP): değer arttıkça fiyat artma eğiliminde
- Güç / km: değer arttıkça fiyat artma eğiliminde
- Vites Turu: değer arttıkça fiyat artma eğiliminde
- Kilometre / motor hacmi: değer arttıkça fiyat düşme eğiliminde
- Marka: değer arttıkça fiyat artma eğiliminde

## Kısa Alan Yorumları (Domain Insights)
- Daha yeni model (ya da daha düşük araç yaşı) tipik olarak daha yüksek fiyata yol açar.
- Kilometre yükseldikçe, aşınma ve yıpranma nedeniyle fiyat genelde düşme eğilimindedir.
- Daha yüksek motor gücü aynı segmentte daha yüksek fiyatla ilişkilidir.
- Marka/seri/model ve yakıt/vites/gövde gibi kategorik değişkenler, segment farklılıkları nedeniyle fiyatı belirgin etkiler.

Not: Yönler, özelliğin değeri ile ilgili SHAP değerlerinin korelasyon işaretinden türetilmiştir; doğrusal olmayan etkileşimler nedeniyle tek başına deterministik değildir.