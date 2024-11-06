from csbdeep.utils import axes_dict
from csbdeep.models import ProjectionConfig, ProjectionCARE
from csbdeep.utils.tf import keras_import, BACKEND as K
from csbdeep.internals import nets
Model = keras_import('models', 'Model')
Input, Conv3D, MaxPooling3D, UpSampling3D, UpSampling2D, Lambda, Multiply = keras_import('layers', 'Input', 'Conv3D', 'MaxPooling3D', 'UpSampling3D', 'UpSampling2D', 'Lambda', 'Multiply')
softmax = keras_import('activations', 'softmax')



class ProjectionUpsamplingConfig(ProjectionConfig):
    
    def __init__(self, axes='ZYX', 
                n_channel_in=1, 
                n_channel_out=1, 
                probabilistic=False,
                unet_n_depth = 3,
                train_loss = 'mse',
                unet_n_first = 48,
                unet_kern_size=3,
                train_epochs=400,
                train_batch_size=4,
                train_learning_rate=0.0001,
                upsampling_factor = 2):
        
        self.axes = axes
        self.n_channel_in = n_channel_in
        self.n_channel_out = n_channel_out
        self.probabilistic = probabilistic
        self.unet_n_depth = unet_n_depth
        self.unet_n_first = unet_n_first
        self.unet_kern_size = unet_kern_size
        self.train_batch_size = train_batch_size
        self.train_epochs = train_epochs
        self.train_learning_rate = train_learning_rate
        self.train_loss = train_loss
        self.upsampling_factor = upsampling_factor
        super().__init__(
            axes = axes,
            n_channel_in = n_channel_in,
            n_channel_out = n_channel_out,
            probabilistic = probabilistic,
            unet_n_depth=unet_n_depth,
            train_epochs=self.train_epochs,
            train_batch_size=self.train_batch_size,
            unet_n_first=self.unet_n_first,
            train_loss=train_loss,
            unet_kern_size=self.unet_kern_size,
            train_learning_rate=self.train_learning_rate,
            train_reduce_lr={"patience": 5, "factor": 0.5},
        )
        
class ProjectionUpsampling(ProjectionCARE):
    
    
    def _build(self):
        # get parameters
        proj = self.proj_params
        proj_axis = axes_dict(self.config.axes)[proj.axis]

        # define surface projection network (3D -> 2D)
        inp = u = Input(self.config.unet_input_shape)
        def conv_layers(u):
            for _ in range(proj.n_conv_per_depth):
                u = Conv3D(proj.n_filt, proj.kern, padding='same', activation='relu')(u)
            return u
        # down
        for _ in range(proj.n_depth):
            u = conv_layers(u)
            u = MaxPooling3D(proj.pool)(u)
        # middle
        u = conv_layers(u)
        # up
        for _ in range(proj.n_depth):
            u = UpSampling3D(proj.pool)(u)
            u = conv_layers(u)
        u = Conv3D(1, proj.kern, padding='same', activation='linear')(u)
        # convert learned features along Z to surface probabilities
        # (add 1 to proj_axis because of batch dimension in tensorflow)
        u = Lambda(lambda x: softmax(x, axis=1+proj_axis))(u)
        # multiply Z probabilities with Z values in input stack
        u = Multiply()([inp, u])
        # perform surface projection by summing over weighted Z values
        u = Lambda(lambda x: K.sum(x, axis=1+proj_axis))(u)
        
        # Perform upsampling
        upsampling_factor = self.config.upsampling_factor
        u = UpSampling2D(size=(upsampling_factor, upsampling_factor))(u)
        
        model_projection = Model(inp, u)

        # define denoising network (2D -> 2D)
        # (remove projected axis from input_shape)
        input_shape = list(self.config.unet_input_shape)
        del input_shape[proj_axis]
        model_denoising = nets.common_unet(
            n_dim           = self.config.n_dim-1,
            n_channel_out   = self.config.n_channel_out,
            prob_out        = self.config.probabilistic,
            residual        = self.config.unet_residual,
            n_depth         = self.config.unet_n_depth,
            kern_size       = self.config.unet_kern_size,
            n_first         = self.config.unet_n_first,
            last_activation = self.config.unet_last_activation,
        )(tuple(input_shape))

        # chain models together
        return Model(inp, model_denoising(model_projection(inp)))
    