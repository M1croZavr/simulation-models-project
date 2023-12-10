Итоговый ноутбук с симуляциями, Range Accrual и графиками в ноутбуке Range accrual pricing.ipynb

Для создания модели 3-ч скоррелированных между собой риск-факторов были взяты дневные данные по процентным ставкам USD - SOFR; RUR - MOSPRIME.

Для оценки параметров для модели Cox-Ingersoll-Ross (волатильность, уровень среднего, скорость возврата к среднему) были использованы исторические данные и метод наименьших квадратов, так как полученные таким образом параметры учитывают паттерн данных для дальнейшей генерации симуляций.
Волатильность для логарифмической модели FX(обменного курса) была выбрана на основе экспертной оценки результатов.
Дата начала симуляций: 10.12.2021.

В модели Range Accrual и симуляциях временной считается как 1 / кол-во дней которое действует контракт.
Количество симуляций вычисляется по среднему значению payout. (Дополнительные комментарии в ноутбуке)
