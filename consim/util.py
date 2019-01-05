import numba
import numpy as np

epsilon = 1e-5


@numba.njit(fastmath=True)
def best_weights(max_weight, scores):
    """
    Раздаёт веса лучшим источникам в соответствии с их оценками и макс. весами. Даёт максимальный возможный вес
    лучшему источнику, потом сколько осталось второму источнику, и т.п., пока не раздаст весь единичный вес
    :param max_weight: Масксимальные веса источников
    :param scores: Оценки источников, чем выше тем лучше
    :return: Идеальные веса источников, массив размером max_weight
    """
    budget = 1
    best_idx = np.argsort(-scores)
    sum_best_weight = 0
    n = len(scores)
    result = np.full_like(max_weight, 0)
    for s in range(n):
        idx = best_idx[s]
        w = max_weight[idx]
        if (sum_best_weight + w) > budget:
            if budget == 1:
                w = budget - sum_best_weight
            else:
                break
        sum_best_weight += w
        result[idx] = w
        if np.abs(sum_best_weight - budget) < budget * epsilon:
            break
    return result


@numba.njit(fastmath=True)
def distribute_equal_weight(max_weights, budget=1.0):
    """
    Пытается распределить вес поровну между источниками. Если в источник не влезает такой "равный" вес,
    источнику даётся максимально возможный вес, а оставшийся бюджет распределяется между остальными источниками,
    и т.д.
    :param max_weights: Максимальные веса источников
    :param budget: Сколько бюджета надо распределить (от 1 и меньше)
    :return: Расепределенные веса, массив размера как у max_weights
    """
    assert max_weights.sum() >= budget * (1 - epsilon)
    n = len(max_weights)
    result = np.empty(n)
    # Самые низкие макс. веса идут первыми
    sorted_idx = np.argsort(max_weights)
    # Какой вес уже распределён
    used_weight = 0
    for i in range(n):
        # распределяем вес начиная от самых маленьких мин весов
        idx = sorted_idx[i]
        # Какой равный вес будет у всех оставшихся эл-тов
        candidate_weight = (budget - used_weight) / (n - i)
        # Проверяем, получится ли поровну распределить вес начиная с текущего эл-та
        max_weight = max_weights[idx]
        if max_weight >= candidate_weight:
            # Получилось, выставляем одинаковые веса всем оставшимся и выходим
            result[sorted_idx[i:]] = candidate_weight
            break
        else:
            # Не получилось, выставляем вес в маскимально возможный, и учитываем его как распределенный
            result[idx] = max_weight
            used_weight += max_weight
    # assert np.allclose(result.sum(), budget)
    return result
