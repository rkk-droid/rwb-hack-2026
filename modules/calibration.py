import numpy as np
from scipy.optimize import minimize



def calibrate_global_scale(val_y_true, val_y_pred, test_y_pred):
    """
    Находит единственный глобальный множитель k = sum(true) / sum(pred)
    на валидации и применяет его к тесту.

    Это обнуляет RelBias на валидации и, как правило, улучшает WAPE,
    если модель систематически занижает или завышает прогнозы.
    Один параметр — не переобучается.

    Returns
    -------
    calibrated_preds : np.ndarray
    k                : float — найденный множитель
    """
    val_y_true  = np.asarray(val_y_true,  dtype=np.float64)
    val_y_pred  = np.asarray(val_y_pred,  dtype=np.float64)
    test_y_pred = np.asarray(test_y_pred, dtype=np.float64)

    k = val_y_true.sum() / val_y_pred.sum()

    def metric(y_true, y_pred):
        wape  = np.abs(y_pred - y_true).sum() / y_true.sum()
        rbias = abs(y_pred.sum() / y_true.sum() - 1)
        return wape + rbias, wape, rbias

    m_before = metric(val_y_true, val_y_pred)
    m_after  = metric(val_y_true, val_y_pred * k)
    print(f"Глобальный множитель k = {k:.6f}")
    print(f"Метрика на валидации ДО:    {m_before[0]:.6f}  "
          f"(WAPE={m_before[1]:.4f}, RBias={m_before[2]:.4f})")
    print(f"Метрика на валидации ПОСЛЕ: {m_after[0]:.6f}  "
          f"(WAPE={m_after[1]:.4f}, RBias={m_after[2]:.4f})")

    return test_y_pred * k, k


def calibrate_per_group(
    val_y_true, val_y_pred, val_route_ids, val_steps,
    test_y_pred, test_route_ids, test_steps,
):
    """
    Находит аддитивный bias для каждой пары (route_id, step), минимизируя
    WAPE + |RelBias| на валидационной выборке, затем применяет к тесту.

    Returns
    -------
    calibrated_preds : np.ndarray
    biases           : dict {(route_id, step): bias}
    """
    val_y_true     = np.asarray(val_y_true,     dtype=np.float64)
    val_y_pred     = np.asarray(val_y_pred,     dtype=np.float64)
    test_y_pred    = np.asarray(test_y_pred,    dtype=np.float64)
    val_route_ids  = np.asarray(val_route_ids)
    val_steps      = np.asarray(val_steps)
    test_route_ids = np.asarray(test_route_ids)
    test_steps     = np.asarray(test_steps)

    # Нормализация: градиент w.r.t. b ~ 1/Y_total (~1e-9) без нормировки,
    # что ниже gtol — оптимизатор не двигается. После нормировки ~ n_k/N ~ 1e-4.
    scale = val_y_true.mean()
    yn    = val_y_true / scale
    pn    = val_y_pred / scale

    unique_groups = sorted(set(zip(val_route_ids.tolist(), val_steps.tolist())))
    group2idx     = {g: i for i, g in enumerate(unique_groups)}
    n_groups      = len(unique_groups)

    val_group_idx = np.array([group2idx[(r, s)]
                               for r, s in zip(val_route_ids, val_steps)])
    Y_total = yn.sum()

    def objective_and_grad(biases):
        adjusted  = pn + biases[val_group_idx]
        residuals = adjusted - yn

        wape      = np.abs(residuals).sum() / Y_total
        wape_grad = np.zeros(n_groups)
        np.add.at(wape_grad, val_group_idx, np.sign(residuals))
        wape_grad /= Y_total

        pred_sum   = adjusted.sum()
        rbias      = abs(pred_sum / Y_total - 1)
        rbias_sign = np.sign(pred_sum - Y_total)
        rbias_grad = np.zeros(n_groups)
        np.add.at(rbias_grad, val_group_idx, rbias_sign / Y_total)

        return wape + rbias, wape_grad + rbias_grad

    val_before = objective_and_grad(np.zeros(n_groups))[0]
    result = minimize(
        objective_and_grad,
        x0=np.zeros(n_groups),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10},
    )
    print(f"Статус: {result.message}")
    print(f"Метрика на валидации ДО:    {val_before:.6f}")
    print(f"Метрика на валидации ПОСЛЕ: {result.fun:.6f}")

    biases_arr = result.x * scale
    biases = {g: biases_arr[i] for g, i in group2idx.items()}

    calibrated_preds = test_y_pred.copy()
    for i, (r, s) in enumerate(zip(test_route_ids, test_steps)):
        key = (r, s)
        if key in group2idx:
            calibrated_preds[i] += biases_arr[group2idx[key]]

    return calibrated_preds, biases


def calibrate_per_group_scale_bias(
    val_y_true, val_y_pred, val_route_ids, val_steps,
    test_y_pred, test_route_ids, test_steps,
):
    """
    Совместная оптимизация per-group скаляра и bias: k_{r,s} * pred + b_{r,s}.

    Скаляр исправляет пропорциональные ошибки модели (корреляция остатков
    с предсказаниями), bias — константные сдвиги.
    Итого 16 000 параметров; k ограничен [0, 3], b — без ограничений.

    Returns
    -------
    calibrated_preds : np.ndarray
    scales           : dict {(route_id, step): k}
    biases           : dict {(route_id, step): b}
    """
    val_y_true     = np.asarray(val_y_true,     dtype=np.float64)
    val_y_pred     = np.asarray(val_y_pred,     dtype=np.float64)
    test_y_pred    = np.asarray(test_y_pred,    dtype=np.float64)
    val_route_ids  = np.asarray(val_route_ids)
    val_steps      = np.asarray(val_steps)
    test_route_ids = np.asarray(test_route_ids)
    test_steps     = np.asarray(test_steps)

    scale = val_y_true.mean()
    yn    = val_y_true / scale
    pn    = val_y_pred / scale

    unique_groups = sorted(set(zip(val_route_ids.tolist(), val_steps.tolist())))
    group2idx     = {g: i for i, g in enumerate(unique_groups)}
    n_groups      = len(unique_groups)

    val_group_idx = np.array([group2idx[(r, s)]
                               for r, s in zip(val_route_ids, val_steps)])
    Y_total = yn.sum()

    def objective_and_grad(params):
        k = params[:n_groups]          # скаляры (безразмерные)
        b = params[n_groups:]          # bias в нормализованном пространстве

        adjusted  = k[val_group_idx] * pn + b[val_group_idx]
        residuals = adjusted - yn
        signs     = np.sign(residuals)

        # WAPE
        wape         = np.abs(residuals).sum() / Y_total
        wape_grad_k  = np.zeros(n_groups)
        wape_grad_b  = np.zeros(n_groups)
        np.add.at(wape_grad_k, val_group_idx, signs * pn)
        np.add.at(wape_grad_b, val_group_idx, signs)
        wape_grad_k /= Y_total
        wape_grad_b /= Y_total

        # RelBias
        pred_sum   = adjusted.sum()
        rbias      = abs(pred_sum / Y_total - 1)
        rbias_sign = np.sign(pred_sum - Y_total)
        rbias_grad_k = np.zeros(n_groups)
        rbias_grad_b = np.zeros(n_groups)
        np.add.at(rbias_grad_k, val_group_idx, rbias_sign * pn / Y_total)
        np.add.at(rbias_grad_b, val_group_idx, rbias_sign / Y_total)

        grad = np.concatenate([
            wape_grad_k + rbias_grad_k,
            wape_grad_b + rbias_grad_b,
        ])
        return wape + rbias, grad

    x0     = np.concatenate([np.ones(n_groups), np.zeros(n_groups)])
    bounds = [(0.0, 3.0)] * n_groups + [(None, None)] * n_groups

    val_before = objective_and_grad(x0)[0]
    result = minimize(
        objective_and_grad,
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10},
    )
    print(f"Статус: {result.message}")
    print(f"Метрика на валидации ДО:    {val_before:.6f}")
    print(f"Метрика на валидации ПОСЛЕ: {result.fun:.6f}")

    k_arr = result.x[:n_groups]            # безразмерный
    b_arr = result.x[n_groups:] * scale    # обратно в исходные единицы

    scales = {g: k_arr[i] for g, i in group2idx.items()}
    biases = {g: b_arr[i] for g, i in group2idx.items()}

    # Векторизованное применение к тесту
    test_route_ids = np.asarray(test_route_ids)
    test_steps     = np.asarray(test_steps)
    calibrated_preds = test_y_pred.copy()
    for i, (r, s) in enumerate(zip(test_route_ids, test_steps)):
        key = (r, s)
        if key in group2idx:
            idx = group2idx[key]
            calibrated_preds[i] = k_arr[idx] * test_y_pred[i] + b_arr[idx]

    return calibrated_preds, scales, biases
