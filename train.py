import torch
import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import argparse
from tqdm import tqdm
from torchsummary import summary


from datasets import custom_transforms
from datasets.sequence_folders import SequenceFolder


from models.gan import GANVO

import torch.backends.cudnn as cudnn
import cv2
from logger import AverageMeter

def train_on_batch(gan: GANVO, tgt_img, ref_imgs, intrinsics):
    """
    Training GAN on a single batch
    """
    tgt_img = tgt_img.to(gan.device)    
    ref_imgs = [img.to(gan.device) for img in ref_imgs]
    intrinsics = intrinsics.to(gan.device)
    

    # Generate synthesized tgt views:
    warped_imgs, _, _, _ = gan.G(tgt_img, ref_imgs, intrinsics)
    warped_imgs = torch.cat(warped_imgs, 0)
    #Generate input for Discriminator
    x_D = torch.cat([
        tgt_img, warped_imgs
    ], dim=0)

    # Create labels:
    y_real = torch.cat([torch.ones(tgt_img.size(0), 1),
                        torch.zeros(warped_imgs.size(0), 1)], dim=0).to(gan.device)
    y_fake = torch.ones(warped_imgs.size(0), 1).to(gan.device)
    
    # Train discriminator with truth labels
    gan.D.train(True)
    D_loss = gan.D.train_step(x_D, y_real)
    # Train GAN with fake labels
    gan.train(True)
    gan.D.train(False)
    G_batch_loss, reconstruction_loss, warped_imgs, diff_maps = gan.train_step(tgt_img, ref_imgs, intrinsics, y_fake)

    return D_loss, G_batch_loss, reconstruction_loss, warped_imgs, diff_maps

def train(start=True):
    parser = argparse.ArgumentParser(description='GANVO training on KITTI-formatted Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
    parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
    parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
    
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size')    
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
    
    parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')    
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')

    args = parser.parse_args()

    # Configure device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Datasets
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5]) # Normalize into [-1,1]
    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length
    )

    #Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    #Network
    ganvo = GANVO(seq_length=args.sequence_length)

    #Optimizer
    D_optmizer = torch.optim.Adam(  params=[{'params':ganvo.D.parameters(), 'lr': args.lr}],
                                    betas=(args.momentum, args.beta),
                                    weight_decay=args.weight_decay)

    G_optmizer = torch.optim.Adam(  params=[{'params':ganvo.parameters(), 'lr': args.lr}],
                                    betas=(args.momentum, args.beta),
                                    weight_decay=args.weight_decay )

    # Keras-like Compilation
    ganvo.D.compile(loss=torch.nn.BCELoss(), optimizer = D_optmizer, device=device)
    ganvo.compile(loss=torch.nn.BCELoss(),   optimizer = G_optmizer, device=device) 
    
    cudnn.benchmark = True
    # ganvo.G = torch.nn.DataParallel(ganvo.G)
    # ganvo.D = torch.nn.DataParallel(ganvo.D)

    if start:
        print("Start training")
        for epoch in range(args.epochs):
            print(f'epoch {epoch}')
            #Training on a single Epoch    
            # running_D_loss = 0.0
            # running_G_loss = 0.0
            # running_GAN_loss = 0.0
            for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(tqdm(train_loader)):
                #  Training on a single batch
                batch_size = tgt_img.size(0)
                D_loss, G_batch_loss, reconstruction_loss, warped_imgs, diff_maps = train_on_batch(ganvo, tgt_img, ref_imgs, intrinsics)
                # running_D_loss += D_loss.item() * batch_size * 3
                # running_G_loss += reconstruction.item() * batch_size * 2
                # running_GAN_loss += G_batch_loss.item() * batch_size * 2
                if i%100 == 0 and i !=0 :
                    print(f'D loss: {D_loss}, Reconstruction loss: {reconstruction_loss}, Gan loss: {G_batch_loss}')
                    # cv2.imshow('Warped image', warped_imgs[0].detach().cpu().permute(1,2,0).numpy())
                    print(tgt_img[0, 2, 100, 100])
                    print(warped_imgs[0][2, 100, 100])
                    
            # running_D_loss /= 
    else:
        summary(ganvo.D, input_size=(3, 480, 640))
        summary(ganvo.G.depth_generator, input_size=(3, 480, 640))
        summary(ganvo.G.pose_regressor, input_size=(9, 480, 640))



@torch.no_grad()
def validate_with_gt_pose(args, val_loader, disp_net, pose_exp_net, epoch, logger, tb_writer, sample_nb_to_log=3):
    global device
    batch_time = AverageMeter()
    depth_error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    depth_errors = AverageMeter(i=len(depth_error_names), precision=4)
    # Absolute Trajectory and Relative Trajectory?
    pose_error_names = ['ATE', 'RTE']
    pose_errors = AverageMeter(i=2, precision=4)
    log_outputs = sample_nb_to_log > 0
    # Output the logs throughout the whole dataset
    batches_to_log = list(np.linspace(0, len(val_loader), sample_nb_to_log).astype(int))
    poses_values = np.zeros(((len(val_loader)-1) * args.batch_size * (args.sequence_length-1), 6))
    disp_values = np.zeros(((len(val_loader)-1) * args.batch_size * 3))

    # switch to evaluate mode
    disp_net.eval()
    pose_exp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, ref_imgs, gt_depth, gt_poses) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        gt_depth = gt_depth.to(device)
        gt_poses = gt_poses.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        b = tgt_img.shape[0]

        # compute output
        output_disp = disp_net(tgt_img)
        output_depth = 1/output_disp
        explainability_mask, output_poses = pose_exp_net(tgt_img, ref_imgs) #[B, num_seq-1, 6]

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
        final_poses = final_poses.reshape(b, -1, 3, 4)

        if log_outputs and i in batches_to_log:  # log first output of wanted batches
            index = batches_to_log.index(i)
            if epoch == 0:
                for j, ref in enumerate(ref_imgs):
                    tb_writer.add_image('val Input {}/{}'.format(j, index), tensor2array(tgt_img[0]), 0)
                    tb_writer.add_image('val Input {}/{}'.format(j, index), tensor2array(ref[0]), 1)

            log_output_tensorboard(tb_writer, 'val', index, '', epoch, output_depth, output_disp, None, None, explainability_mask)

        if log_outputs and i < len(val_loader)-1:
            step = args.batch_size*(args.sequence_length-1)
            poses_values[i * step:(i+1) * step] = output_poses.cpu().view(-1, 6).numpy()
            step = args.batch_size * 3
            disp_unraveled = output_disp.cpu().view(args.batch_size, -1)
            disp_values[i * step:(i+1) * step] = torch.cat([disp_unraveled.min(-1)[0],
                                                            disp_unraveled.median(-1)[0],
                                                            disp_unraveled.max(-1)[0]]).numpy()

        depth_errors.update(compute_depth_errors(gt_depth, output_depth[:, 0]))
        pose_errors.update(compute_pose_errors(gt_poses, final_poses))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write(
                'valid: Time {} Abs Error {:.4f} ({:.4f}), ATE {:.4f} ({:.4f})'.format(batch_time,
                                                                                       depth_errors.val[0],
                                                                                       depth_errors.avg[0],
                                                                                       pose_errors.val[0],
                                                                                       pose_errors.avg[0]))
    if log_outputs:
        prefix = 'valid poses'
        coeffs_names = ['tx', 'ty', 'tz']
        if args.rotation_mode == 'euler':
            coeffs_names.extend(['rx', 'ry', 'rz'])
        elif args.rotation_mode == 'quat':
            coeffs_names.extend(['qx', 'qy', 'qz'])
        for i in range(poses_values.shape[1]):
            tb_writer.add_histogram('{} {}'.format(prefix, coeffs_names[i]), poses_values[:, i], epoch)
        tb_writer.add_histogram('disp_values', disp_values, epoch)
    logger.valid_bar.update(len(val_loader))
    return depth_errors.avg + pose_errors.avg, depth_error_names + pose_error_names



if __name__=='__main__':
    train(start=True)