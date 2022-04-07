import torch

@torch.no_grad()
def compute_pose_errors(gt, pred):
    RE = 0
    for (current_gt, current_pred) in zip(gt, pred):
        snippet_length = current_gt.shape[0]
        scale_factor = torch.sum(current_gt[..., -1] * current_pred[..., -1]) / torch.sum(current_pred[..., -1] ** 2)
        ATE = torch.norm((current_gt[..., -1] - scale_factor * current_pred[..., -1]).reshape(-1)).cpu().numpy()
        R = current_gt[..., :3] @ current_pred[..., :3].transpose(-2, -1)
        for gt_pose, pred_pose in zip(current_gt, current_pred):
            # Residual matrix to which we compute angle's sin and cos
            R = (gt_pose[:, :3] @ torch.inverse(pred_pose[:, :3])).cpu().numpy()
            s = np.linalg.norm([R[0, 1]-R[1, 0],
                                R[1, 2]-R[2, 1],
                                R[0, 2]-R[2, 0]])
            c = np.trace(R) - 1
            # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
            RE += np.arctan2(s, c)

    return [ATE/snippet_length, RE/snippet_length]
