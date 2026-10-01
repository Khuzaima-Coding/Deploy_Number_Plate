from ultralytics import YOLO
import gradio as gr 


model = YOLO("best.pt")

def pred_image(image):
    img = model.predict(image)
    return img[0].plot()


app= gr.Interface(fn = pred_image, inputs = 'image', outputs = "image" )
app.launch()

#break the terminal -- ctrl+c
#download libraries -- pip install -r libraries
# run the command that open browser---- python app.py(my file)