# Handwriting Recognition
Create venv
```sh
py -m venv venv
```
Let’s install the library
```sh
pip install -r requirements.txt
```
- To train the model:
  - Comment the line `from core.ui import *` inside `main.py`
  - Open the line comment `model.trainModel()`
  - After training the model, you need to annotate the line `model.trainModel()`
- Try testing with data that is photos from the archive you use `model.testWithImage(path)`
- Use with your handwriting:
  - Comment line `model.testWithImage(path)`
  - Open the line comment
    ```sh
      root = tk.Tk()
      app = DrawingApp(root)
      root.mainloop()
    ```
