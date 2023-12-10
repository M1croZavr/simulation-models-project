import numpy as np
from scipy import stats


def fx_rate_predict(n_simulations, t_steps, fx0, domestic_rate,foreign_rate, sigma):
    '''
    Функция предоставляет численное решение для стохастического дифференциального
    уравнения, которое описывает поведение обменного курса на основе разности между
    ставкой инотсранной foreign_rate валюты и домашней domestic_rate
    Начальная точка fx0 считается известной
    параметр sigma задает уровень волатильности
    На выходе функция выдает заданное кол-во (n_simulations) симуляций на горизонт,
    задаваемый параметром t_steps
    '''
    dt = 1 / t_steps  # шаг симуляции
    fx = fx0 * np.ones((n_simulations, 1))
    for tm in range(t_steps):
        wiener_multiplier = stats.norm().rvs(size=(n_simulations, 1))
        rt_domestic = domestic_rate[:,tm].reshape(-1, 1)
        rt_foreign = foreign_rate[:,tm].reshape(-1, 1)
        fx_previous = fx[:, -1].reshape(-1, 1)
        fx = np.hstack(
            (
                fx,
                fx_previous + fx_previous * (((rt_foreign - rt_domestic) / 100) * dt + sigma * wiener_multiplier)
            )
        )
    last_t_simulation = fx[:, -1]
    return fx


def simulation_correlation(
    simulated_usd_rate, simulated_rur_rate, simulated_fx_rate,
    n_simulations, chol_matrix
):
    '''
    Функция приводит полученные раннее симуляции процентных ставок (simulated_usd_rate, simulated_rur_rate)
    и обменного курса (simulated_fx_rate) к скоррелированному виду при помощи разложения холецкого,
    расчитанного на исторических данных (chol_matrix)
    На выходе функция выдает множество симуляций обменного курса с поправкой на корреляцию
    '''
    usd_vector = simulated_usd_rate[0].reshape(-1, 1)
    rur_vector = simulated_rur_rate[0].reshape(-1, 1)
    fx_vector = simulated_fx_rate[0].reshape(-1, 1)
    
    uncorr_sim = np.concatenate((usd_vector, rur_vector, fx_vector), axis=1)
    corr_sim = np.matmul(chol_matrix, uncorr_sim.T).T
    
    usd_corr = corr_sim.T[0].reshape(-1, 1)
    rur_corr = corr_sim.T[1].reshape(-1, 1)
    fx_corr = corr_sim.T[2].reshape(-1, 1)
    
    for n_sim in range(1, n_simulations):
        usd_vector = simulated_usd_rate[n_sim].reshape(-1, 1)
        rur_vector = simulated_rur_rate[n_sim].reshape(-1, 1)
        fx_vector = simulated_fx_rate[n_sim].reshape(-1, 1)
        
        uncorr_sim = np.concatenate((usd_vector, rur_vector, fx_vector), axis=1)
        corr_sim = np.matmul(chol_matrix, uncorr_sim.T).T
        usd_corr_element = corr_sim.T[0].reshape(-1, 1)
        rur_corr_element = corr_sim.T[1].reshape(-1, 1)
        fx_corr_element = corr_sim.T[2].reshape(-1, 1)
    
        usd_corr = np.concatenate((usd_corr, usd_corr_element), axis=1)
        rur_corr = np.concatenate((rur_corr, rur_corr_element), axis=1)
        fx_corr = np.concatenate((fx_corr, fx_corr_element), axis=1)

    return usd_corr, rur_corr, fx_corr
