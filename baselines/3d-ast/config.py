# device settings
device = 'cuda' # or 'cpu'

# data settings
dataset_dir = '/path/to/your/dataset/' # parent directory of class folders
feature_dir = 'data/features/' # directory where features are stored and loaded from
use_3D_dataset = True # is MVTec 3D used?
pre_extracted = True # were feature pre-extracted with extract_features? (recommended)
modelname = "my_experiment" # export evaluations/logs with this name
print(modelname)

# inputs
img_len = 768 # width/height of input image
img_size = (img_len, img_len) 
img_dims = [3] + list(img_size)
depth_len = img_len // 4 # width/height of depth maps
depth_downscale = 8 # to which factor depth maps are downsampled by unshuffling
depth_channels = depth_downscale ** 2 # channels per pixel after unshuffling
map_len = img_len // 32 # feature map width/height (dependent on feature extractor!)
extract_layer = 35 # layer from which features are extracted
img_feat_dims = 304 # number of image features (dependent on feature extractor!)

if not use_3D_dataset:
    mode = 'RGB' # force RGB if no 3D data is available
else:
    mode = ['RGB', 'depth', 'combi'][2]
    
n_feat = {'RGB': img_feat_dims, 'depth': depth_channels, 'combi': img_feat_dims + depth_channels}[mode]

training_mask = (mode != 'RGB') # use foreground mask for training?
eval_mask = (mode != 'RGB') # use foreground mask for evaluation?

# 3D settings
dilate_mask = True
dilate_size = 8
n_fills = 3
bg_thresh = 7e-3

# transformation settings
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# network hyperparameters
clamp = 1.9 # clamping parameter
n_coupling_blocks = 4 # higher = more flexible = more unstable
channels_hidden_teacher = 64 # number of neurons in hidden layers of internal networks
channels_hidden_student = 1024 # number of neurons in hidden layers of student
use_gamma = True
kernel_sizes = [3] * (n_coupling_blocks - 1) + [5]
pos_enc = True # use positional encoding
pos_enc_dim = 32 # number of dimensions of positional encoding
asymmetric_student = True
n_st_blocks = 4 # number of residual blocks in student

# training parameters
lr = 2e-4 # learning rate
batch_size = 8
eval_batch_size = batch_size * 2
# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 3 # total epochs = meta_epochs * sub_epochs
sub_epochs = 24 #batch_size # evaluate after this number of epochs

# output settings
verbose = True
hide_tqdm_bar = True
save_model = True

# eval settings
localize = True