from lib.lib import *
from core.data import *

class Model():
    def __init__(self, epochs=5, checkpoint_path="training_1/cp.ckpt"):
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        
    def createModel(self):
        model = keras.Sequential()
        # 32 convolution filters used each of size 3x3
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        # 64 convolution filters used each of size 3x3
        model.add(Conv2D(64, (3, 3), activation='relu'))
        # choose the best features via pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # randomly turn neurons on and off to improve convergence
        model.add(Dropout(0.25))
        # flatten since too many dimensions, we only want a classification output
        model.add(Flatten())
        # fully connected to get all relevant data
        model.add(Dense(128, activation='relu'))
        # one more dropout
        model.add(Dropout(0.5))
        # output a softmax to squash the matrix into output probabilities
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer='Adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model
    
    def trainModel(self):
        # Train & Save Checkpoint
        Data = data()
        train_images, train_labels, test_images, test_labels = Data.dataTrain()
        
        model = self.createModel()
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        cp_callback = keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)

        model.fit(train_images, train_labels, epochs=self.epochs, validation_data=(test_images, test_labels), callbacks=[cp_callback])
        loss, acc = model.evaluate(test_images, test_labels, verbose=2)
        print("Accuracy: ", acc)
        
    def testWithImage(self, imagePath="./data/so_4.png"):
        model = self.createModel()
        model.load_weights(self.checkpoint_path).expect_partial()
        image = Image.open(imagePath)
        image_array = np.asarray(image)

        plt.figure()
        plt.imshow(image_array, cmap='gray')
        B = cv2.resize(image_array[:,:,0], dsize=(28, 28),interpolation=cv2.INTER_CUBIC)
        plt.imshow(B)
        preds = model.predict(B.reshape(1,28,28,1))
        guess = np.argmax(preds)
        print("PREDICT: NUMBER ", guess)