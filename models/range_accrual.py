import numpy as np

from .cir import CoxIngersollRossModel
from .fx import fx_rate_predict, simulation_correlation


def range_accrual(
    notional, n_fix_dates, n_simulations, 
    fx_corr, upper_bound=0, lower_bound=0
):
    '''
    Функция считает матожидание payout для продукта range accrual
    для этого для каждой симуляции на каждую точку симуляции (кроме нулевой) проверяется условие
    нахождения обменного курса внутри диапозона (задаваемого параметрами lower_bound и upper_bound)
    затем считается доля точек, которые удовлетворяют этому условию. После чего доля умножается на
    номинал (notional)
    Все полученные значения payout скаладываются и делятся на кол-во симуляций
    Ибо все симуляции равновероятны
    '''
    sum_of_payouts = 0
    for fx_sim in fx_corr[1:].T:
        if upper_bound > 0:
            share_of_win = np.sum((lower_bound <= fx_sim) & (upper_bound >= fx_sim)) / n_fix_dates
        else:
            share_of_win = np.sum((lower_bound <= fx_sim)) / n_fix_dates
        sum_of_payouts += notional * share_of_win
    return sum_of_payouts / n_simulations


def range_accrual_pricing(
    history_data, n_simulations, n_fix_dates, chol_matrix,
    notional, upper_bound=0, lower_bound=0
):
    '''
    Объединям все функции
    На выход получаем симуляции для процентной ставки валют, обменного курса и справедливую стоимость
    '''
    r_start_rur = history_data.sort_values(by=['Date'], ascending=False)['RUR'][0]
    r_start_usd = history_data.sort_values(by=['Date'], ascending=False)['USD'][0]
    fx0 = history_data.sort_values(by=['Date'], ascending=False)['FX-Rate'][0]
    delta_t = 1 / n_fix_dates
    
    cir = CoxIngersollRossModel(delta_t=delta_t)
    cir.estimate_ols(history_data['RUR'])
    simulated_rur_rate, expected_value_rur, expected_error_rur = cir.make_interest_rate_simulations(r_start_rur, n_simulations, n_fix_dates)
    
    cir = CoxIngersollRossModel(delta_t=delta_t)
    cir.estimate_ols(history_data['USD'])
    simulated_usd_rate, expected_value_usd, expected_error_usd = cir.make_interest_rate_simulations(r_start_usd, n_simulations, n_fix_dates)

    simulated_fx_rate = fx_rate_predict(n_simulations, n_fix_dates, fx0, simulated_rur_rate, simulated_usd_rate, 0.05)

    corr_simulated_usd_rate, corr_simulated_rur_rate, corr_simulated_fx_rate = \
        simulation_correlation(simulated_usd_rate, simulated_rur_rate, simulated_fx_rate, n_simulations, chol_matrix)

    fair_value = range_accrual(notional, n_fix_dates, n_simulations, corr_simulated_fx_rate, upper_bound, lower_bound)

    return corr_simulated_usd_rate, corr_simulated_rur_rate, corr_simulated_fx_rate, fair_value
