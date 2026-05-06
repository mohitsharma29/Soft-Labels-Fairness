import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'FairPareto'))
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import pandas as pd
from oxonfair import FairPredictor
from sklearn.linear_model import LogisticRegression
from oxonfair import group_metrics as gm
from FairPareto.fairpareto.core import FairParetoClassifier
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, TruePositiveRateParity
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from fairlearn.metrics import true_positive_rate


# Todo: Check MIFPO code

def train_oxon(train_x, train_y, train_a, test_x, test_y, test_a, metric='dp', classifier='lr', tolerance=0.0001):
    # https://arxiv.org/pdf/2407.13710
    # https://github.com/oxfordinternetinstitute/oxonfair/blob/master/examples/fairret.ipynb
    # Split train into a val set (30%)
    subgroups = {
        '00': [],
        '01': [],
        '10': [],
        '11': []
    }
    for i in range(len(train_y)):
        if type(train_y) == pd.core.series.Series:
            y_val = int(train_y.iloc[i])
            a_val = int(train_a.iloc[i])
        else:
            y_val = int(train_y[i])
            a_val = int(train_a[i])
        key = f"{y_val}{a_val}"
        subgroups[key].append(i)
    train_indices = []
    val_indices = []
    for key, indices in subgroups.items():
        np.random.shuffle(indices)
        split_point = int(0.7 * len(indices))
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])
    # x is a numpy array, y and a are pandas series
    if type(train_x) == pd.core.frame.DataFrame:
        val_x = train_x[val_indices]
        val_y = train_y.iloc[val_indices]
        val_a = train_a.iloc[val_indices]
        train_x = train_x[train_indices]
        train_y = train_y.iloc[train_indices]
        train_a = train_a.iloc[train_indices]
        # Data is a numpy arrat, convert to pandas dataframe
        x_train_df = pd.DataFrame(train_x, columns=[f'feature_{i}' for i in range(train_x.shape[1])])
        x_train_df['A'] = train_a.values
        # convert target to array
        y_train = train_y.values
        x_val_df = pd.DataFrame(val_x, columns=[f'feature_{i}' for i in range(val_x.shape[1])])
        x_val_df['A'] = val_a.values
        y_val = val_y.values
        x_test_df = pd.DataFrame(test_x, columns=[f'feature_{i}' for i in range(test_x.shape[1])])
        x_test_df['A'] = test_a.values
        y_test = test_y.values
    else:
        val_x = train_x[val_indices]
        val_y = train_y[val_indices]
        val_a = train_a[val_indices]
        train_x = train_x[train_indices]
        train_y = train_y[train_indices]
        train_a = train_a[train_indices]
        # Data is a numpy arrat, convert to pandas dataframe
        x_train_df = pd.DataFrame(train_x, columns=[f'feature_{i}' for i in range(train_x.shape[1])])
        x_train_df['A'] = train_a
        # convert target to array
        y_train = train_y
        x_val_df = pd.DataFrame(val_x, columns=[f'feature_{i}' for i in range(val_x.shape[1])])
        x_val_df['A'] = val_a
        y_val = val_y
        x_test_df = pd.DataFrame(test_x, columns=[f'feature_{i}' for i in range(test_x.shape[1])])
        x_test_df['A'] = test_a
        y_test = test_y

    if classifier == 'lr':
        predictor = LogisticRegression()
        predictor.fit(x_train_df, y_train)
    else:
        assert False, "Only logistic regression is supported currently"
    
    fair_predictor = FairPredictor(
        predictor=predictor,
        validation_data={
            'data': x_val_df,
            'target': y_val,
            'groups': 'A'
        },
        groups='A',

    )

    if metric == 'DP':
        fair_metric = gm.demographic_parity
    elif metric == 'EOP':
        fair_metric = gm.equal_opportunity
    fair_predictor.fit(
        objective=gm.accuracy,
        constraint=fair_metric,
        value=tolerance
    )
    tradeoff = fair_predictor.evaluate(
        data={
            'data': x_test_df,
            'target': y_test,
            'groups': 'A'
        }
    )
    tradeoff = fair_predictor.frontier_scores()
    oxon_errs = 1 - tradeoff[0]
    oxon_unfs = tradeoff[1]
    return oxon_errs, oxon_unfs

def train_fairret(x,y,a,x_test,y_test,a_test, fairness_strength=0.1, metric='DP'):
    # Convert sensitive to two dimensional
    new_a = []
    new_a_test = []
    for i in range(a.shape[0]):
        if a[i] == 0:
            new_a.append([1,0])
        else:
            new_a.append([0,1])
    for i in range(a_test.shape[0]):
        if a_test[i] == 0:
            new_a_test.append([1,0])
        else:
            new_a_test.append([0,1])
    a = np.asarray(new_a)
    a_test = np.asarray(new_a_test)
    x,y,a,x_test, y_test, a_test = map(lambda z: torch.tensor(z, dtype=torch.float32), (x,y,a,x_test, y_test, a_test))
    y = y.view(-1,1)
    y_test = y_test.view(-1,1)
    #a = a.view(-1,1)
    #a_test = a_test.view(-1,1)
    h_layer_dim = 16
    lr = 1e-3
    batch_size = 1024

    # Mimic a logistic regression model
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], h_layer_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(h_layer_dim, 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to('cuda')

    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(x, y, a)
    test_dataset = TensorDataset(x_test, y_test, a_test)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    #test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    from fairret.statistic import TruePositiveRate, PositiveRate
    from fairret.metric import LinearFractionalParity
    from fairret.loss import NormLoss
    if metric == 'EOP':
        statistic = TruePositiveRate()
        #eop = LinearFractionalParity(TruePositiveRate(), 1)
    elif metric == 'DP':
        statistic = PositiveRate()
        #dp = LinearFractionalParity(PositiveRate(), 1)
    norm_loss = NormLoss(statistic)

    for epoch in range(50):
        losses = []
        for batch_feat, batch_label, batch_sens in dataloader:
            batch_feat = batch_feat.to('cuda')
            batch_sens = batch_sens.to('cuda')
            batch_label = batch_label.to('cuda')
            optimizer.zero_grad()
                    
            logit = model(batch_feat)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, batch_label)
            if metric == 'eop':
                loss += fairness_strength * norm_loss(logit, batch_sens, batch_label)
            elif metric == 'dp':
                loss += fairness_strength * norm_loss(logit, batch_sens)
            loss.backward()
                    
            optimizer.step()
            losses.append(loss.item())
        #print(f"Epoch: {epoch}, loss: {np.mean(losses)}")

    x_test = x_test.to('cuda')
    y_test = y_test.to('cuda')
    a_test = a_test.to('cuda')
    pred = torch.sigmoid(model(x_test))
    if metric == 'DP':
        stat_per_group = statistic(pred, a_test)
    elif metric == 'EOP':
        stat_per_group = statistic(pred, a_test, y_test)
    # Compute err = 1 - acc
    test_err = 1 - torch.mean(((pred >= 0.5).float() == y_test).float()).item()
    test_unf = stat_per_group[1] - stat_per_group[0]
    return test_err, test_unf.item()

def train_mifpo(all_embeddings, all_labels, all_attrs, all_soft):
    # Load your data (X should include the sensitive attribute column)
    # Modified the class to work with custom soft labels
    #feature_group = np.column_stack((samples, sensitive))
    data_df = pd.DataFrame(all_embeddings)
    data_df['sensitive'] = all_attrs

    clf = FairParetoClassifier(soft=[
        all_soft[np.where(all_attrs == 0)[0]],
        all_labels[np.where(all_attrs == 1)[0]]
    ], sensitive_column='sensitive', gamma_values= list(np.linspace(0, 0.5, 31)), verbose=True)
    clf.fit(data_df, all_labels)

    return clf.get_pareto_front()

def signed_dp(y_true, y_pred, sensitive_features):
    group_0 = np.where(sensitive_features == 0)[0]
    group_1 = np.where(sensitive_features == 1)[0]
    pr_1_a0 = np.mean(y_pred[group_0])
    pr_1_a1 = np.mean(y_pred[group_1])
    return pr_1_a1 - pr_1_a0

def train_exp_grad(train_X, train_Y, train_A, test_X, test_Y, test_A, metric='DP'):
    eps_cls = {}
    # for eps in tqdm([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]):
    #     #constraint = DemographicParity()
    #     constraint = TruePositiveRateParity()
    #     mitigator = ExponentiatedGradient(LogisticRegression(), constraint, eps=eps)
    #     mitigator.fit(samples, labels, sensitive_features=sensitive)
    #     eps_cls[eps] = mitigator
    for eps in tqdm([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]):
        #for eps in tqdm([0.001, 0.002]):
        if metric == 'DP':
            constraint = DemographicParity()
        elif metric == 'EOP':
            constraint = TruePositiveRateParity()
        mitigator = ExponentiatedGradient(LogisticRegression(), constraint, eps=eps)
        mitigator.fit(train_X, train_Y, sensitive_features=train_A)
        eps_cls[eps] = mitigator
    
    unfair_model = LogisticRegression().fit(train_X, train_Y)
    preds = unfair_model.predict(test_X)
    unfair_err = 1 - accuracy_score(test_Y, preds)
    if metric == 'DP':
        unfair_unf = signed_dp(test_Y, preds, sensitive_features=test_A)
    elif metric == 'EOP':
        unfair_unf = true_positive_rate(test_Y[np.where(test_A == 1)[0]], preds[np.where(test_A == 1)[0]]) - true_positive_rate(test_Y[np.where(test_A == 0)[0]], preds[np.where(test_A == 0)[0]])
    
    unfs = []
    errs = []
    eps = []

    for e in eps_cls.keys():
        mitigated_test_preds = eps_cls[e].predict(test_X)
        if metric == 'DP':
            unfs.append(signed_dp(test_Y, mitigated_test_preds, test_A))
        elif metric == 'EOP':
            unfs.append(true_positive_rate(test_Y[np.where(test_A == 1)[0]], mitigated_test_preds[np.where(test_A == 1)[0]]) - true_positive_rate(test_Y[np.where(test_A == 0)[0]], mitigated_test_preds[np.where(test_A == 0)[0]]))
        errs.append(1 - accuracy_score(test_Y, mitigated_test_preds))
        eps.append(e)

    # Add the unfair model point
    unfs.append(unfair_unf)
    errs.append(unfair_err)
    eps.append(0)

    # Sort by errors
    temp = sorted(zip(unfs, errs, eps), key=lambda x: x[1])
    unfs, errs, eps = zip(*temp)
    return unfs, errs, eps