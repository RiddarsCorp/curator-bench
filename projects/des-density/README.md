# Плотность DES

## Устройство
- `data/density_measurements.csv` — измерения плотности; `data/components.csv` — компоненты.
- `mining_descriptors.py` — дескрипторы компонентов → `data/density_preprocessed.csv`.
- `notebooks/density_eda.ipynb` — разведочный анализ.
- `notebooks/density_ml.ipynb` — разбиение, обучение, оценка.

## Как запустить
```
pip install -r requirements.txt
python mining_descriptors.py
jupyter nbconvert --execute notebooks/density_ml.ipynb
```

## Документы проекта
- `RESEARCH.md` — суть исследования.
- `PLAN.md` — план по этапам.
