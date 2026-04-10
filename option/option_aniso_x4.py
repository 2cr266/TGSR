import argparse

parser = argparse.ArgumentParser(description='Classic options')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=1,
                    help='number of threads for data loading')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')


# Data specifications
parser.add_argument('--data_dir', type=str, default='/hy-tmp/datasets',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='DF2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set5',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-3450/801-810',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=int, default=4,
                    help='super-resolution scale')
parser.add_argument('--patch_size', type=int, default=64,
                    help='output patch size, the input image size is patch size x scale')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--augment', type=bool, default=True,
                    help='use data augmentation')
parser.add_argument('--repeat', type=int, default=9,
                    help='image repeat in dataloader')
parser.add_argument('--patch_cut', type=bool, default=True,
                    help='image repeat in dataloader')
parser.add_argument('--patch_cut_two', type=bool, default=False,
                    help='image two cut in training dataloader')


# Degradation specifications
parser.add_argument('--kernel_size', type=int, default=21,
                    help='size of blur kernels')
parser.add_argument('--blur_type_prob', type=list, default=[0, 1],
                    help='prob of blur type')
parser.add_argument('--blur_type_list', type=list, default=['iso_gaussian', 'aniso_gaussian'],
                    help='list of blur type')
parser.add_argument('--sigma_min', type=float, default=0.2,
                    help='')
parser.add_argument('--sigma_max', type=float, default=4.0,
                    help='')
parser.add_argument('--down_sample_list', type=list, default=['nearest', 'area', 'bilinear', 'bicubic'],
                    help='')
parser.add_argument('--down_sample_prob', type=list, default=[0, 0, 0, 1],
                    help='')
parser.add_argument('--noise_min', type=float, default=0.0,
                    help='noise level range')
parser.add_argument('--noise_max', type=float, default=10.0,
                    help='noise level range')

# pca settings
parser.add_argument('--pca_path', type=str, default='/home/CATADCLS-main/pca/pca_matrix_x4.pth',
                    help='pca path')

# Training specifications
parser.add_argument('--project_path', type=str, default='/home/CATADCLS-main',
                    help='project path')
parser.add_argument('--resume', type=bool, default=False,
                    help='resume from specific checkpoint')
parser.add_argument('--resume_path', type=str, default='/home/CATADCLS-main/experiment/CATAdcls_2.9M_aniso_x4/model/model_0020.pth.tar',
                    help='resume path')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs_encoder', type=int, default=100,
                    help='number of epochs to train the degradation encoder')
parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs to train the whole network')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')



# Optimization specifications
parser.add_argument('--lr_sr', type=float, default= 2e-4,
                    help='learning rate to train the whole network')
parser.add_argument('--lr_decay_sr', type=int, default=100,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma_sr', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

parser.add_argument('--pixel_loss_type', type=str, default='L1Loss', help='pixel loss type (L1Loss/MSELoss)')
parser.add_argument('--pixel_loss_weight', type=float, default=1.0, help='pixel loss weight')
parser.add_argument('--pixel_loss_reduction', type=str, default='mean', help='pixel loss reduction (mean/sum/none)')


# Save and Log specifications
parser.add_argument('--model_name', type=str, default='test_aniso_x4',
                    help='file name to save')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', default=False,
                    help='save output results')

args = parser.parse_args()
