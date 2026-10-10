# SelTox — антибактериальная активность наночастиц

## Устройство
- `Data and Preprocessing/raw/` — исходные данные.
- `Data and Preprocessing/preprocessing.py` — кодирование и масштабирование признаков.
- `model/pipeline.py` — конвейер подготовки признаков.
- `train_model.py` — кросс-валидация и итоговая оценка.

Проект ведётся в PyCharm (папка `.idea`).

## Как запустить
```
pip install -r requirements.txt
python train_model.py
```

## Документы проекта
- `RESEARCH.md` — суть исследования.
- `PLAN.md` — план по этапам.
