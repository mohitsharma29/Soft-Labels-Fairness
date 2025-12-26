import numpy as np
import matplotlib.pyplot as plt

def soft_estimator(soft_y,a):
    # Return pr(A=a), pr(Y=y|A=a)
    hat_a_1 = np.mean(a)
    group_1_indices = np.where(a == 1)[0]
    hat_y_1_a_1 = np.mean(soft_y[group_1_indices])
    group_1_indices = np.where(a == 0)[0]
    hat_y_1_a_0 = np.mean(soft_y[group_1_indices])
    return {
        'a_1': hat_a_1,
        'a_0': 1 - hat_a_1,
        'y_1_a_1': hat_y_1_a_1,
        'y_0_a_1': 1 - hat_y_1_a_1,
        'y_1_a_0': hat_y_1_a_0,
        'y_0_a_0': 1 - hat_y_1_a_0
    }

def positive_rate_estimator(alpha, pi, C, eps=0.001):
    positive_rate = 0
    for i in range(C.shape[0]):
        positive_rate += (C[i] >= alpha)
    return positive_rate / C.shape[0]

def true_positive_rate_estimator(alpha, pi, C, eps=0.001):
    true_positive_rate = 0
    for i in range(C.shape[0]):
        true_positive_rate += (C[i] >= alpha) * C[i]
    
    # Conditional estimate
    # return true_positive_rate / C.shape[0]
    #return (true_positive_rate/C.shape[0])
    return (true_positive_rate/C.shape[0])/pi

def bayes_error_estimator(alphas, priors, C, A):
    err = 0
    pi = priors['y_1_a_1']*priors['a_1'] + priors['y_1_a_0']*priors['a_0']
    for i in range(A.shape[0]):
        #err += min((1 - alphas[A[i]])*C[i], alphas[A[i]]*(1 - C[i]))
        err += (C[i] >= alphas[A[i]])*(1 - 2*C[i])
    return pi + err / A.shape[0]

def threshold_estimator(priors, lam, group, constraint='dp'):
    if group == 0:
        pi_a = priors['a_0']
        pi_y_a = priors['y_1_a_0']
    else:
        pi_a = priors['a_1']
        pi_y_a = priors['y_1_a_1']
    if constraint == 'dp':
        temp_threshold = 1/2 + ((2*group - 1)*lam)/(2*pi_a)
        if temp_threshold > 1-1e-5:
            return 1
        elif temp_threshold < 1e-5:
            return 0
        else:
            return temp_threshold
    elif constraint == 'eop':
        temp_threshold = 1/(2 - ((2*group - 1)*lam)/(pi_y_a*pi_a)) # Because we have conditional estimates
        if temp_threshold > 1-1e-5:
            return 1
        elif temp_threshold < 1e-5:
            return 0
        else:
            return temp_threshold

# Unit tests
def test_monotonicity(priors, D, constraint='eop'):
    if constraint == 'dp':
        lam_max = max(priors['a_0'], priors['a_1'])
    elif constraint == 'eop':
        lam_max = max(priors['y_1_a_0']*priors['a_0'], priors['y_1_a_1']*priors['a_1']) # Because we have conditional estimates
    lams = np.linspace(-1, 1, 50)
    errs = []
    unfs = []
    for lam in lams:
        if constraint == 'dp':
            alphas = [threshold_estimator(priors, lam, 0, constraint='dp'), threshold_estimator(priors, lam, 1, constraint='dp')]
            err = bayes_error_estimator(alphas, priors, D['C'], D['A'])
            # print(alphas, err)
            # continue
            errs.append(err)
            unfs.append(positive_rate_estimator(alphas[1], priors['a_1'], D['C'][D['A'] == 1]) - positive_rate_estimator(alphas[0], priors['a_0'], D['C'][D['A'] == 0]))
        elif constraint == 'eop':
            alphas = [threshold_estimator(priors, lam, 0, constraint='eop'), threshold_estimator(priors, lam, 1, constraint='eop')]
            err = bayes_error_estimator(alphas, priors, D['C'], D['A'])
            errs.append(err)
            unfs.append(true_positive_rate_estimator(alphas[1], priors['y_1_a_1'], D['C'][D['A'] == 1]) - true_positive_rate_estimator(alphas[0], priors['y_1_a_0'], D['C'][D['A'] == 0]))
    plt.figure()
    plt.plot(lams, errs)
    # draw vertical lines at -lam_max and lam_max
    plt.axvline(x=-lam_max, color='r', linestyle='--')
    plt.axvline(x=lam_max, color='r', linestyle='--')
    plt.xlabel('Lambda')
    plt.ylabel('Bayes Error')
    plt.title('Monotonicity Test of Bayes Error with respect to Lambda')
    plt.grid()
    plt.show()
    plt.figure()
    plt.plot(lams, unfs)
    # draw vertical lines at -lam_max and lam_max
    plt.axvline(x=-lam_max, color='r', linestyle='--')
    plt.axvline(x=lam_max, color='r', linestyle='--')
    plt.xlabel('Lambda')
    plt.ylabel('Unfairness')
    plt.title('Monotonicity Test of Unfairness with respect to Lambda')
    plt.grid()
    plt.show()
    return None

# Some inspiration from https://github.com/XianliZeng/Bayes-Optimal-Fair-Classification/
def return_tradeoff(D, priors, constraint='dp', delta=0.001, eps=0.00001, T=100):
    alpha_0 = threshold_estimator(priors, 0, 0, constraint)
    alpha_1 = threshold_estimator(priors, 0, 1, constraint)
    alphas = {
        0: alpha_0,
        1: alpha_1
    }
    bayes_error = bayes_error_estimator(alphas, priors, D['C'], D['A'])
    if constraint == 'dp':
        bayes_unf = positive_rate_estimator(alpha_1, priors['a_1'], D['C'][np.where(D['A'] == 1)[0]]) - positive_rate_estimator(alpha_0, priors['a_0'], D['C'][np.where(D['A'] == 0)[0]])
        global_lam_max = max(priors['a_0'], priors['a_1'])
    elif constraint == 'eop':
        bayes_unf = true_positive_rate_estimator(alpha_1, priors['y_1_a_1'], D['C'][np.where(D['A'] == 1)[0]]) - true_positive_rate_estimator(alpha_0, priors['y_1_a_0'], D['C'][np.where(D['A'] == 0)[0]])
        global_lam_max = max(priors['y_1_a_0']*priors['a_0'], priors['y_1_a_1']*priors['a_1'])
    print(bayes_error, bayes_unf, global_lam_max)
    
    points = {
        -1: [bayes_error, bayes_unf, 0]
    }
    
    if abs(bayes_unf) < delta:
        return points
    else:
        if bayes_unf > delta:
            lam_min = 0
            lam_max = global_lam_max
            mid = (lam_max + lam_min)/2
            counter = 0
            while (lam_max - lam_min) > eps:
                counter += 1
                alpha_0 = threshold_estimator(priors, mid, 0, constraint)
                alpha_1 = threshold_estimator(priors, mid, 1, constraint)
                alphas = {
                    0: alpha_0,
                    1: alpha_1
                }
                error = bayes_error_estimator(alphas, priors, D['C'], D['A'])
                if constraint == 'dp':
                    unf = positive_rate_estimator(alpha_1, priors['a_1'], D['C'][np.where(D['A'] == 1)[0]]) - positive_rate_estimator(alpha_0, priors['a_0'], D['C'][np.where(D['A'] == 0)[0]])
                elif constraint == 'eop':
                    unf = true_positive_rate_estimator(alpha_1, priors['y_1_a_1'], D['C'][np.where(D['A'] == 1)[0]]) - true_positive_rate_estimator(alpha_0, priors['y_1_a_0'], D['C'][np.where(D['A'] == 0)[0]])
                print(f'lambda: {mid}, Alpha_0: {alpha_0}, Alpha_1: {alpha_1}, Error: {error}, Unfairness: {unf}, Lam Min: {lam_min}, Lam Max: {lam_max}')
                points[counter] = [error, unf, mid]
                if unf > delta:
                    lam_min = mid
                elif unf < -delta: # Sign flipped
                    lam_max = mid
                mid = (lam_max + lam_min)/2
                # if mid < 0:
                #     # Make sure to not cross lam_min
                #     mid = max(mid, -global_lam_max)
                # else:
                #     mid = min(mid, global_lam_max)
                if abs(unf) < delta:
                    break
                if counter > T:
                    print('Max iterations reached')
                    break
        else:
            lam_min = -global_lam_max
            lam_max = 0
            mid = (lam_max + lam_min)/2
            counter = 0
            while (lam_max - lam_min) > eps:
                counter += 1
                alpha_0 = threshold_estimator(priors, mid, 0, constraint)
                alpha_1 = threshold_estimator(priors, mid, 1, constraint)
                alphas = {
                    0: alpha_0,
                    1: alpha_1
                }
                error = bayes_error_estimator(alphas, priors, D['C'], D['A'])
                if constraint == 'dp':
                    unf = positive_rate_estimator(alpha_1, priors['a_1'], D['C'][np.where(D['A'] == 1)[0]]) - positive_rate_estimator(alpha_0, priors['a_0'], D['C'][np.where(D['A'] == 0)[0]])
                elif constraint == 'eop':
                    unf = true_positive_rate_estimator(alpha_1, priors['y_1_a_1'], D['C'][np.where(D['A'] == 1)[0]]) - true_positive_rate_estimator(alpha_0, priors['y_1_a_0'], D['C'][np.where(D['A'] == 0)[0]])
                print(f'lambda: {mid}, Alpha_0: {alpha_0}, Alpha_1: {alpha_1}, Error: {error}, Unfairness: {unf}, Lam Min: {lam_min}, Lam Max: {lam_max}')
                points[counter] = [error, unf, mid]
                # Check sign flip and update bounds (Because of convexity sign flip cannot happen!)
                if unf > delta: # Sign flipped!
                    #mid = (mid + points[counter - 1][2])/2
                    lam_min = mid
                elif unf < -delta: # Still negative
                    lam_max = mid
                mid = (lam_max + lam_min)/2
                # if mid < 0:
                #     # Make sure to not cross lam_min
                #     mid = max(mid, -global_lam_max)
                # else:
                #     mid = min(mid, global_lam_max)
                if abs(unf) < delta:
                    break
                if counter > T:
                    print('Max iterations reached')
                    break
    return points, bayes_error, bayes_unf

def plot_tradeoff(tradeoff, tradeoff_est, true_bayes, true_unf, predicted_points=None):
    x = []
    y = []
    colors = []
    for key in tradeoff:
        #if key != -1:
        x.append(tradeoff[key][0])
        y.append(tradeoff[key][1])
        colors.append(tradeoff[key][2])
    x = np.array(x)
    y = np.array(y)
    plt.scatter(x, y, c=colors, cmap='viridis', label='True Trade-off')
    # x_y_spline = interp1d(x, y, kind='cubic')
    # X_ = np.linspace(x.min(), x.max(), 500)
    # Y_ = x_y_spline(X_)
    # plt.plot(X_, Y_, color='blue', linestyle='--', label='True Trade-off (spline)')
    # Draw a vertical line at the bayes optimal point
    plt.axvline(x=true_bayes, color='black', linestyle='--')
    plt.axhline(y=true_unf, color='black', linestyle='--')
    plt.colorbar(label='Lambda')
    x_est = []
    y_est = []
    colors_est = []
    for key in tradeoff_est:
        #if key != -1:
        x_est.append(tradeoff_est[key][0])
        y_est.append(tradeoff_est[key][1])
        colors_est.append(tradeoff_est[key][2])
    x_est = np.array(x_est)
    y_est = np.array(y_est)
    plt.scatter(x_est, y_est, c=colors_est, cmap='plasma', marker='x', label='Estimated Trade-off')
    # remove duplicates from x_est, y_est, colors_est
    # unique_indices = np.unique(x_est, return_index=True)[1]
    # u_x_est = x_est[unique_indices]
    # u_y_est = y_est[unique_indices]
    # u_colors_est = np.array(colors_est)[unique_indices]
    # x_y_spline_est = interp1d(u_x_est, u_y_est, kind='cubic')
    # X_est_ = np.linspace(x_est.min(), x_est.max(), 500)
    # Y_est_ = x_y_spline_est(X_est_)
    # plt.plot(X_est_, Y_est_, color='orange', linestyle='--', label='Estimated Trade-off (spline)')
    plt.colorbar(label='Lambda (Estimated)')
    if predicted_points is not None:
        unfs, errs, eps = predicted_points['exp_grad']
        unfs = np.array(unfs)
        errs = np.array(errs)
        # Remove duplicates
        # unique_indices_exp = np.unique(errs, return_index=True)[1]
        # u_errs = errs[unique_indices_exp]
        # u_unfs = unfs[unique_indices_exp]
        # x_y_spline_exp = interp1d(u_errs, u_unfs, kind='cubic')
        # X_exp_ = np.linspace(errs.min(), errs.max(), 500)
        # Y_exp_ = x_y_spline_exp(X_exp_)
        # plt.plot(X_exp_, Y_exp_, color='green', linestyle='--', label='Fairlearn Exp-Grad (spline)')
        # unfs_fairret, errs_fairret, eps_fairret = predicted_points['fairret']
        # unfs_fairret = np.array(unfs_fairret)
        # errs_fairret = np.array(errs_fairret)
        # Remove duplicates
        # unique_indices_fairret = np.unique(errs_fairret, return_index=True)[1]
        # u_errs_fairret = errs_fairret[unique_indices_fairret]
        # u_unfs_fairret = unfs_fairret[unique_indices_fairret]
        # x_y_spline_fairret = interp1d(u_errs_fairret, u_unfs_fairret, kind='cubic')
        # X_fairret_ = np.linspace(errs_fairret.min(), errs_fairret.max(), 500)
        # Y_fairret_ = x_y_spline_fairret(X_fairret_)
        # plt.plot(X_fairret_, Y_fairret_, color='red', linestyle='--', label='FairRet (spline)')
        plt.scatter(errs, unfs, c='green', marker='^', label='Fairlearn Exp-Grad')
        #plt.scatter(errs_fairret, unfs_fairret, c='red', marker='s', label='FairRet')
    plt.xlabel('Bayes Error')
    plt.ylabel('Unfairness')
    plt.ylim(-0.05, 0.15)
    plt.xlim(0.075, 0.15)
    plt.title('Trade-off Curve')
    plt.legend()