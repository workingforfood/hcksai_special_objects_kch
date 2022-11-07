from efficientnet.model import EfficientNetB2
from keras.layers import BatchNormalization, Activation, GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Nadam


def efficientnet_classifierB2(input_shape=(128, 128, 3), num_of_cls=8):
    optimizer = Nadam(lr=1e-3)
    model = EfficientNetB2(input_shape=input_shape, include_top=False, weights="imagenet")
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(100)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(num_of_cls)(x)
    out = Activation('softmax')(x)
    model = Model(inputs=model.input, outputs=out, name="encoder")
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model
