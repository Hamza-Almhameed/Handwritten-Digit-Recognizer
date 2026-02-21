import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import joblib

model = joblib.load("digit_model.pkl")

window = tk.Tk()
window.title("Draw a digit (0-9)")

canvas_width = 200
canvas_height = 200

canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="black")
canvas.pack()

image = Image.new("L", (canvas_width, canvas_height), "black")
draw = ImageDraw.Draw(image)

def draw_lines(event):
    x, y = event.x, event.y
    r = 10
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
    draw.ellipse([x-r, y-r, x+r, y+r], fill="white")

canvas.bind("<B1-Motion>", draw_lines)



def predict_digit():
    img_array = np.array(image)


    coords = np.column_stack(np.where(img_array > 20))
    if coords.size == 0:
        result_label.config(text="Draw something!")
        return

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img_array[y_min:y_max+1, x_min:x_max+1]

    img_cropped = Image.fromarray(cropped)

    img_resized = img_cropped.resize((8, 8), Image.Resampling.LANCZOS)

    img_array = np.array(img_resized)


    img_array = (img_array / 255.0) * 16

    img_flat = img_array.flatten().reshape(1, -1)

    print("Min:", img_flat.min(), "Max:", img_flat.max())

    prediction = model.predict(img_flat)

    result_label.config(text=f"Prediction: {prediction[0]}")

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill="black")
    result_label.config(text="")

btn_predict = tk.Button(window, text="Predict", command=predict_digit)
btn_predict.pack()

btn_clear = tk.Button(window, text="Clear", command=clear_canvas)
btn_clear.pack()

result_label = tk.Label(window, text="", font=("Arial", 20))
result_label.pack()

window.mainloop()