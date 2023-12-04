from lib.lib import *
from core.model import *
from core.ui import *

if __name__ == "__main__":
    model = Model(epochs=5)
    # model.trainModel()
    model.testWithImage("./data/so_4.png")
    # root = tk.Tk()
    # app = DrawingApp(root)
    # root.mainloop()