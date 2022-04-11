import torch
import numpy as np
from models.inverse_wrap import pose_vec2mat

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

@torch.no_grad()
def validate_with_pose_only(args, val_loader, pose_net, epoch, device):
    # switch to evaluate mode
    # disp_net.eval()
    with open('./draft.txt', 'w') as f:
        f.write('Epoch,ATE,RTE\n')

    pose_net.eval()
    for i, (tgt_img, ref_imgs, gt_poses) in enumerate(val_loader):
        tgt_img = tgt_img.to(device) #[B, 3, h, w]
        gt_poses = gt_poses.to(device) #[B, num_seq, 3, 4]
        ref_imgs = [img.to(device) for img in ref_imgs]
        b = tgt_img.shape[0]

        poses_i = ref_imgs[:pose_net.nb_ref_imgs//2] + [tgt_img] + ref_imgs[pose_net.nb_ref_imgs//2:]
        poses_i = torch.cat(poses_i, 1) # (B, 3*seq_length, h, w)

        # compute output
        output_poses = pose_net(poses_i) #[B, num_seq-1, 6]
        # [B, num_seq (num_ref_imgs+1), 6]
        reordered_output_poses = torch.cat([output_poses[:, :gt_poses.shape[1]//2],
                                            torch.zeros(b, 1, 6).to(output_poses),
                                            output_poses[:, gt_poses.shape[1]//2:]], dim=1)

        # pose_vec2mat only takes B, 6 tensors, so we simulate a batch dimension of B * seq_length
        # [B * num_seq , 6]
        unravelled_poses = reordered_output_poses.reshape(-1, 6)
        # [B * num_seq , 12]
        unravelled_matrices = pose_vec2mat(unravelled_poses, rotation_mode=args.rotation_mode)
        # [B, num_seq, 3, 4] (T target to sources)
        inv_transform_matrices = unravelled_matrices.reshape(b, -1, 3, 4)
        # [B, num_seq, 3, 3] (R sources to target) (R^-1)
        rot_matrices = inv_transform_matrices[..., :3].transpose(-2, -1)
        # [B, num_seq, 3, 1] (translation sources to target) 
        tr_vectors = -rot_matrices @ inv_transform_matrices[..., -1:]
        # [B, num_seq, 3, 4] (T sources to target)
        transform_matrices = torch.cat([rot_matrices, tr_vectors], axis=-1)
        # [B, 1, 3, 4] (T target to first source ) 
        first_inv_transform = inv_transform_matrices.reshape(b, -1, 3, 4)[:, :1]
        # [B, num_seq, 3, 4] (T target -> first source) dot (All T sources -> target) = [Idenity, T2nd->1st, T3rd->1st, .... Ttarget->1st,.... Tnumseq->1st] ~ All odom wrt the first frame 
        final_poses = first_inv_transform[..., :3] @ transform_matrices
        final_poses[..., -1:] += first_inv_transform[..., -1:]
        final_poses = final_poses.reshape(b, -1, 3, 4) #[B, num_seq, 3 , 4]

        ate, rte = compute_pose_errors(gt_poses, final_poses)
        print(f"ATE: {ate}, RTE: {rte}")
        with open('./draft.txt', 'a') as f:
            f.write(f'{epoch}/{i},{ate},{rte}\n')
    
    return ate, rte
        