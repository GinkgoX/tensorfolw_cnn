#parameters for artistic style

#set content image
CONTENT_IMAGE = 'images/content.jpg'
#set style image
STYLE_IMAGE = 'images/style.jpg'
#set the output image
OUTPUT_IMAGE = 'images/output'
#set vgg model path
VGG_MODEL_PATH = 'imagenet-vgg-verydeep-19.mat'
#define image height
IMAGE_HEIGHT = 224
#define image width
IMAGE_WIDTH = 224
#define vgg_19 content layer loss parameters [conv_layer, weights]
CONTENT_LOSS_LAYERS = [('conv4_2', 0.5), ('conv5_2', 0.5)]
#define vgg_19 style layer parameters [conv_layer, weights]
STYLE_LOSS_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]
#define noize rate
NOIZE = 0.5
#define image mean value
IMAGE_MEAN_VALUE = [128, 128, 128]
#define content loss weight
ALPHA = 1
#define style loss weight
BETA = 200
#define train steps
TRAIN_STEPS = 300
