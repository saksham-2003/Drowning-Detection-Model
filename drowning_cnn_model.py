import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2, InceptionV3, EfficientNetB0
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization,
    Input, Conv2D, MaxPooling2D, Concatenate, Reshape, Activation,
    Add, AveragePooling2D, SpatialDropout2D, DepthwiseConv2D,
    LeakyReLU, PReLU, ReLU, GaussianNoise, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import math


class CustomLearningRateScheduler:
    def __init__(self, initial_lr=0.001, decay_steps=1000, decay_rate=0.96):
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
    
    def __call__(self, epoch):
        new_lr = self.initial_lr * (self.decay_rate ** (epoch / self.decay_steps))
        return max(new_lr, 1e-6)


class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, **kwargs):
        super(InceptionModule, self).__init__(**kwargs)
        
        self.conv1 = Conv2D(filters//4, 1, padding='same', activation='relu', kernel_regularizer=l2(0.0001))
        
        self.conv3_1x1 = Conv2D(filters//4, 1, padding='same', activation='relu', kernel_regularizer=l2(0.0001))
        self.conv3_3x3 = Conv2D(filters//4, 3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))
        
        self.conv5_1x1 = Conv2D(filters//4, 1, padding='same', activation='relu', kernel_regularizer=l2(0.0001))
        self.conv5_5x5 = Conv2D(filters//4, 5, padding='same', activation='relu', kernel_regularizer=l2(0.0001))
        
        self.pool_conv = Sequential([
            MaxPooling2D(3, strides=1, padding='same'),
            Conv2D(filters//4, 1, padding='same', activation='relu', kernel_regularizer=l2(0.0001))
        ])
        
        self.batch_norm = BatchNormalization()
        
    def call(self, x):
        branch1 = self.conv1(x)
        
        branch3 = self.conv3_1x1(x)
        branch3 = self.conv3_3x3(branch3)
        
        branch5 = self.conv5_1x1(x)
        branch5 = self.conv5_5x5(branch5)
        
        branch_pool = self.pool_conv(x)
        
        outputs = Concatenate()([branch1, branch3, branch5, branch_pool])
        outputs = self.batch_norm(outputs)
        
        return outputs


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        
        self.conv1 = Conv2D(filters, 3, strides=strides, padding='same', kernel_regularizer=l2(0.0001))
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        
        self.conv2 = Conv2D(filters, 3, padding='same', kernel_regularizer=l2(0.0001))
        self.bn2 = BatchNormalization()
        
        if strides != 1:
            self.downsample = Sequential([
                Conv2D(filters, 1, strides=strides, kernel_regularizer=l2(0.0001)),
                BatchNormalization()
            ])
        else:
            self.downsample = lambda x: x
        
        self.relu2 = ReLU()
        
    def call(self, x):
        residual = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = Add()([out, residual])
        out = self.relu2(out)
        
        return out


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.conv = Conv2D(1, kernel_size, padding='same', activation='sigmoid', kernel_regularizer=l2(0.0001))
        
    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = Concatenate()([avg_pool, max_pool])
        feature = self.conv(concat)
        return x * feature


class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.avg_pool = GlobalAveragePooling2D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.fc1 = Dense(channels // reduction, activation='relu', kernel_regularizer=l2(0.0001))
        self.fc2 = Dense(channels, activation='sigmoid', kernel_regularizer=l2(0.0001))
        
    def call(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        
        avg_out = self.fc2(self.fc1(avg_out))
        max_out = self.fc2(self.fc1(max_out))
        
        out = avg_out + max_out
        out = tf.reshape(out, [-1, 1, 1, tf.shape(x)[-1]])
        
        return x * out


def create_advanced_drowning_detection_model(input_shape=(224, 224, 3)):
    # Input layer with noise augmentation
    inputs = Input(shape=input_shape)
    x = GaussianNoise(0.1)(inputs)
    
    # Initial convolution for feature extraction
    x = Conv2D(64, 7, strides=2, padding='same', kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = ResidualBlock(64)(x)
    x = ResidualBlock(64)(x)
    x = SpatialDropout2D(0.2)(x)
    
    x = ResidualBlock(128, strides=2)(x)
    x = ResidualBlock(128)(x)
    x = SpatialDropout2D(0.3)(x)
    
    # Inception modules for multi-scale feature extraction
    x = InceptionModule(256)(x)
    x = InceptionModule(256)(x)
    x = SpatialDropout2D(0.4)(x)
    
    # Attention mechanisms
    x = ChannelAttention(256)(x)
    x = SpatialAttention()(x)
    
    # Additional residual blocks
    x = ResidualBlock(512, strides=2)(x)
    x = ResidualBlock(512)(x)
    x = SpatialDropout2D(0.5)(x)
    
    # Feature pyramid for capturing spatial information at different scales
    branch1 = Conv2D(256, 1, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    branch1 = AveragePooling2D(2)(branch1)
    
    branch2 = Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    branch2 = AveragePooling2D(4)(branch2)
    
    branch3 = Conv2D(256, 5, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(x)
    branch3 = AveragePooling2D(8)(branch3)
    
    # Global pooling for each branch
    branch1 = GlobalAveragePooling2D()(branch1)
    branch2 = GlobalAveragePooling2D()(branch2)
    branch3 = GlobalAveragePooling2D()(branch3)
    
    # Main branch
    main_branch = GlobalAveragePooling2D()(x)
    
    # Concatenate all features
    merged = Concatenate()([main_branch, branch1, branch2, branch3])
    
    # Dense classifier with advanced regularization
    x = Dense(1024, kernel_regularizer=l2(0.0001))(merged)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)
    
    x = Dense(512, kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    return model


detector_model = create_advanced_drowning_detection_model()

# Custom metrics
def f1_score(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.cast(y_true, 'float32') * tf.round(y_pred))
    possible_positives = tf.reduce_sum(tf.cast(y_true, 'float32'))
    predicted_positives = tf.reduce_sum(tf.round(y_pred))
    
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

# Compile with custom optimizer and metrics
optimizer = Adam(learning_rate=0.001)

detector_model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score]
)

# Callbacks with advanced configurations
callbacks = [
    EarlyStopping(
        monitor='val_f1_score',
        patience=10,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ModelCheckpoint(
        'best_drowning_detector.h5',
        monitor='val_f1_score',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    LearningRateScheduler(CustomLearningRateScheduler(initial_lr=0.001))
]

detector_model.summary()

history = detector_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)