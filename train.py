import tensorflow as tf
from sklearn.model_selection import train_test_split
from dataset import LoadData, PreprocessData
from model import EncoderMiniBlock, DecoderMiniBlock, UNetCompiled

path1 = 'Enter_the_path_to_images'
path2 = 'Enter_the_path_to_masks'
img, mask = LoadData (path1, path2)

target_shape_img = [960, 960, 3]
target_shape_mask = [960, 960, 1]

X, y = PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=77)

# build the U-Net
unet = UNetCompiled(input_size=(960,960,3), n_filters=32, n_classes=1)
unet.summary()

unet.compile(optimizer=tf.keras.optimizers.Adam(), 
             loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# start model training
results = unet.fit(X_train, y_train, batch_size=2, epochs=200, validation_data=(X_valid, y_valid))

# model evaluation
unet.evaluate(X_valid, y_valid)