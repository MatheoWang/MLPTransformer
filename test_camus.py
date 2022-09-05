import os
import numpy as np
import torch
import pandas as pd
from metrics import evalAllmetric, det_jac

def test_camus(model, test_dataset, device, transformer, model_dir, best_epoch):
    # Test data metrics
    metric_ed = []
    metric_es = []
    det_jac_es2ed_lv = []
    det_jac_es2ed_myo = []
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(test_dataset, 1):
            _, _, inputs, mask_es_gt, y_true, mask_ed_gt, sampling = sample
            inputs = [torch.from_numpy(d).to(device).float() for d in inputs]
            y_true = [torch.from_numpy(d).to(device).float() for d in y_true]
            mask_ed = torch.unsqueeze(torch.from_numpy(mask_ed_gt).to(device).float(), 0)
            mask_es = torch.unsqueeze(torch.from_numpy(mask_es_gt).to(device).float(), 0)
            # run inputs through the model to produce a warped image and flow field
            y_pred = model(*inputs)
            pos_flow = y_pred[2]
            neg_flow = y_pred[3]
            trans_maskes = transformer(mask_es, pos_flow)
            trans_masked = transformer(mask_ed, neg_flow)

            # mask metrics
            trans_masked = trans_masked[0].cpu().numpy()
            trans_masked[0] = (trans_masked[1] + trans_masked[2]) > 0
            trans_maskes = trans_maskes[0].cpu().numpy()
            trans_maskes[0] = (trans_maskes[1] + trans_maskes[2]) > 0
            mask_es_gt[0] = (mask_es_gt[1] + mask_es_gt[2]) > 0
            mask_ed_gt[0] = (mask_ed_gt[1] + mask_ed_gt[2]) > 0
            metes = evalAllmetric(trans_masked, mask_es_gt, sampling)
            meted = evalAllmetric(trans_maskes, mask_ed_gt, sampling)
            metric_ed.append(meted)
            metric_es.append(metes)

            # flow metrics
            # take flow for es-->ed
            detjac = det_jac(pos_flow[0].cpu().numpy()[::-1])
            idxlv = np.where(mask_ed[0, 1].cpu().numpy())
            idxmyo = np.where(mask_ed[0, 2].cpu().numpy())
            det_jac_es2ed_lv.append(detjac[idxlv])
            det_jac_es2ed_myo.append(detjac[idxmyo])

            # break

    det_jac_es2ed_lv = np.concatenate(det_jac_es2ed_lv)
    det_jac_es2ed_myo = np.concatenate(det_jac_es2ed_myo)
    np.save(model_dir + '/epoch' + str(best_epoch) + '_det_jac_es2ed_lv.npy', det_jac_es2ed_lv)
    np.save(model_dir + '/epoch' + str(best_epoch) + '_det_jac_es2ed_myo.npy', det_jac_es2ed_myo)

    metric_ed = np.array(metric_ed)
    metric_es = np.array(metric_es)
    metric_ed = np.concatenate([metric_ed, np.array([np.mean(metric_ed, axis=0)])])
    metric_es = np.concatenate([metric_es, np.array([np.mean(metric_es, axis=0)])])

    # save metrics
    regions = ['LV+MYO', 'LV', 'MYO', 'LA']
    mets = ['Dice', 'HD', 'MSD']
    pd_cols = []
    for r in regions:
        for m in mets:
            pd_cols.append(r + ' ' + m)
    ed_pd = pd.DataFrame(metric_ed, columns=pd_cols)
    es_pd = pd.DataFrame(metric_es, columns=pd_cols)
    ed_pd.to_csv(model_dir + '/epoch' + str(best_epoch) + '_gt_ed_metrics.csv')
    es_pd.to_csv(model_dir + '/epoch' + str(best_epoch) + '_gt_es_metrics.csv')

    metric_all = np.concatenate([metric_ed[:-1], metric_es[:-1]])
    all_pd = pd.DataFrame(metric_all, columns=pd_cols)
    all_pd.to_csv(model_dir + '/epoch' + str(best_epoch) + '_all_metrics.csv')