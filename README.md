# KAÜ - HASTALIK TAHMİNİ DOSTU

Flask tabanlı web uygulaması; sabit backend Google Sheets verisini okuyup Apriori association rules ve makine öğrenmesi sınıflandırma modelleriyle hastalık tahmini yapar.

## Öne Çıkanlar

- Google Sheets linki sadece backend config içinde tutulur, frontend’e yazılmaz.
- Belirti seçimi modern toggle-card UI ile yapılır.
- Apriori: `mlxtend.frequent_patterns.apriori` ve `association_rules`.
- ML: RandomForestClassifier, LogisticRegression, DecisionTreeClassifier.
- En iyi model accuracy ile seçilir, `models/best_disease_model.joblib` olarak kaydedilir.
- Sonuçta hastalık, olasılık, confidence, support, lift ve alternatifler döner.
- Tek yönlendirme mesajı üretir: doktor branşı veya acil servis.
- Kars nöbetçi eczaneleri `https://www.eczaneler.gen.tr/nobetci-kars` kaynağından canlı çekilir.
- Dark mode, custom cursor, ses sistemi, medical pulse loader ve eczane carousel bulunur.

## Kurulum

```bash
cd project
python3 -m pip install -r requirements.txt
python3 run.py
```

Port 5000 doluysa:

```bash
PORT=5001 python3 run.py
```

## Kullanım

Tarayıcı:

```text
http://127.0.0.1:5000
```

İlk açılışta sabit Google Sheets verisi backend tarafından okunur, Apriori kuralları çıkarılır, üç ML modeli eğitilir ve grafikler `outputs/` klasörüne kaydedilir.

## API

Tahmin:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms":["Ateş","Öksürük"]}'
```

Kurallar:

```bash
curl http://127.0.0.1:5000/rules
```

Analiz:

```bash
curl http://127.0.0.1:5000/analytics
```

Nöbetçi eczaneler:

```bash
curl http://127.0.0.1:5000/pharmacies
```

## Sayfalar

- `/` ana sayfa
- `/predict` belirti seçimi ve AJAX tahmin
- `/analysis` analiz paneli
- `/rules`, `/analytics`, `/pharmacies` JSON API

## Çıktılar

- `models/best_disease_model.joblib`
- `outputs/top_symptoms.png`
- `outputs/disease_distribution.png`
- `outputs/correlation_heatmap.png`
- `outputs/confusion_matrix.png`
- `outputs/support_confidence.png`

## Not

Bu sistem klinik tanı koymaz; veri madenciliği ve makine öğrenmesi projesi olarak karar destek çıktısı üretir.

“Sağlık, insanın en büyük zenginliğidir.”

Vildan Çam, 2026 tüm hakları saklıdır
