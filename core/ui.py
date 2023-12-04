from lib.lib import *
from core.model import *

m = Model()
model = m.createModel()
model.load_weights("training_1/cp.ckpt").expect_partial()

def preprocess_images(imgs): 
    sample_img = imgs if len(imgs.shape) == 2 else imgs[0]
    assert sample_img.shape in [(28, 28, 1), (28, 28)], sample_img.shape 
    return imgs / 255.0

def preprocess_input_image(image):
    # Preprocess the input image for prediction
    image = np.array(image)
    image = cv2.resize(image[:,:,0], dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    image = image.reshape(1, 28, 28, 1)
    image = preprocess_images(image)
    return image

def predict_number(image):
    # Use the loaded model to predict the number
    image = preprocess_input_image(image)
    preds = model.predict(image)
    return np.argmax(preds)

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Number Prediction")

        self.canvas = Canvas(root, width=200, height=200, bg="white")
        self.canvas.pack()

        self.label = tk.Label(root, text="Draw a number and click Predict")
        self.label.pack()

        self.predict_button = Button(root, text="Predict", command=self.predict)
        self.predict_button.pack()

        self.clear_button = Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.image = Image.new("RGB", (200, 200), color="white")
        self.draw = ImageDraw.Draw(self.image)

        self.points = []
        self.last_x, self.last_y = None, None
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.reset_coordinates)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill="black", width=5)
            self.draw.line([self.last_x, self.last_y, x, y], fill="black", width=5)
            
        self.points.append((x, y))
        self.last_x, self.last_y = x, y
        # x1, y1 = (event.x - 1), (event.y - 1)
        # x2, y2 = (event.x + 1), (event.y + 1)
        # self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
        # self.draw.line([x1, y1, x2, y2], fill="black", width=5)
        
    def reset_coordinates(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (200, 200), color="white")
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="Draw a number and click Predict")
        self.last_x, self.last_y = None, None

    def predict(self):
        input_image = self.image.resize((28, 28))
        number = predict_number(input_image)
        self.label.config(text=f"Predicted Number: {number}")