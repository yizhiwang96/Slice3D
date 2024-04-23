import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_model', type=str, default='slicenet', choices=['slicenet', 'disn', 'gtslice'])
    # dataset related
    parser.add_argument('--dir_data', type=str, default='../data')
    parser.add_argument('--name_dataset', type=str, default='shapenet', choices=['objaverse', 'shapenet', 'custom', 'custom_sin_img'])
    parser.add_argument('--name_single', type=str, default='fertility', help='name of the single shape')
    parser.add_argument('--n_wk', type=int, default=16, help='number of workers in dataloader')
    parser.add_argument('--categories_train', type=str, default='objaverse,', help='the training and validation categories of objects for ShapeNet datasets')
    parser.add_argument('--categories_test', type=str, default='objaverse,', help='the testing categories of objects for ShapeNet datasets')
    parser.add_argument('--add_noise', type=float, default=0, help='the std of noise added to the point clouds')
    parser.add_argument('--gt_source', type=str, default='imnet', choices=['imnet', 'occnet'], help='using which query-occ groundtruth when training on Shapenet')

    parser.add_argument('--img_size', type=int, default=128, help='img_size')
    parser.add_argument('--n_qry', type=int, default=256, help='the number of query points for per shape when training')
    parser.add_argument('--n_slices', type=int, default=12, help='the number of slices for each shape')
    parser.add_argument('--n_views', type=int, default=12, help='the number of views for each shape')
    parser.add_argument('--pred_type', type=str, default='sdf', choices=['occ', 'sdf'], help='predict occupancy (occ) or signed distance field (sdf), sdf only works for abc dataset')

    # common hyper-parameters
    parser.add_argument('--name_exp', type=str, default='2023_07_04_chairs_vggptloss')
    parser.add_argument('--name_exp_cam', type=str, default='2023_1107_airplanes_est') # 2023_07_010_camera_pose_est_ or 2023_07_010_camera_pose_est_rot_shift 2023_1107_airplanes_est
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--n_bs', type=int, default=16, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=600, help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='init learning rate')
    parser.add_argument('--n_dim', type=int, default=128, help='the dimension of hidden layer features')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--freq_ckpt', type=int, default=4, help='frequency of epoch saving checkpoint')
    parser.add_argument('--freq_log', type=int, default=200, help='frequency of outputing training logs')
    parser.add_argument('--freq_decay', type=int, default=100, help='decaying the lr evey freq_decay epochs')
    parser.add_argument('--weight_decay', type=float, default=0.5, help='weight decay')
    parser.add_argument('--tboard', type=bool, default=True, help='whether use tensorboard to visualize loss')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, help='resume training')
    parser.add_argument('--est_campose', action=argparse.BooleanOptionalAction, help='whether to use gt camera poses')

    parser.add_argument('--back_bone_cam_est', type=str, default='vgg16_bn', choices=['vgg16_bn', 'resnet50'])

    # training related
    parser.add_argument('--use_white_bg', action=argparse.BooleanOptionalAction, help='whether to use gt camera poses')

    # Marching Cube realted
    parser.add_argument('--mc_chunk_size', type=int, default=3000, help='the number of query points in a chunk when doing marching cube, set it according to your GPU memory')
    parser.add_argument('--mc_res0', type=int, default=64, help='start resolution for MISE')
    parser.add_argument('--mc_up_steps', type=int, default=2, help='number of upsampling steps')
    parser.add_argument('--mc_threshold', type=float, default=0.5, help='the threshold for network output values')
    # testing related
    parser.add_argument('--name_ckpt', type=str, default='10_5511_0.0876_0.9612.ckpt')
    parser.add_argument('--name_ckpt_cam', type=str, default='570_225545_1.969e-05.ckpt') #  480_276575_0.0003403.ckpt 500_288075_0.0009608.ckpt 570_225545_1.969e-05.ckpt
    parser.add_argument('--from_which_slices', type=str, default='gt', choices=['gt', 'gt_rec', 'gen'], help='using which kind of slices')
    parser.add_argument('--overwrite_res', action=argparse.BooleanOptionalAction, help='whether to overwrite the results')
    return parser