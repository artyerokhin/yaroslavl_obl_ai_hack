# yaroslavl_obl_ai_hack
Solution for https://hacks-ai.ru/championships/758240

# Решение для соревнования Ярославской области, Цифровой прорыв

1. Установка пакетов -r requirements.txt
2. Запуск решения python main.py
3. Если нужен новый подбор параметров, то python main.py --load_pickle False

## Общий процесс решения:
1. Добавляется некоторый набор признаков + замены значений (см. utils);
2. С помощью optuna для каждой задачи подбираются оптимальные параметры lightgbm по метрике roc-auc;
3. С помощью optuna подбираются оптимальные параметры округления (порог отсечения 0 или 1) по целевой метрике соревнования;
4. Делается предсказание на тестовых данных.
