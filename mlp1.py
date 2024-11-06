import js
import numpy as np

from pyodide.ffi import create_proxy


canvasNN1 = js.document.getElementById('myNNCanvas1')
canvasNN1.style.width  = canvasNN1.width
canvasNN1.style.height = canvasNN1.height
ctxNN1 = canvasNN1.getContext('2d')
canvasDS1 = js.document.getElementById('myDataCanvas1')
canvasDS1.style.width  = canvasDS1.width
canvasDS1.style.height = canvasDS1.height
ctxDS1 = canvasDS1.getContext('2d')

inputField1 = js.document.getElementById('inputField1')

from mlpmain import MLP, MLPDraw, ClassificationOutput

net1 = MLP([2, 1])
drawer1 = MLPDraw(net1, canvasNN1, "inputField1")

or_data_list = [[0,0,0], [0,1,1], [1,0,1], [1,1,1]]
dataset_drawer1 = ClassificationOutput(net1, canvasDS1, data=or_data_list, xrange=[-0.5, 1.5], yrange=[-0.5, 1.5])

def draw1(event=None):
    global drawer1
    drawer1.draw(event)
    dataset_drawer1.draw()

def mousedown1(event):
    drawer1.handle_press(event.offsetX, event.offsetY)
    dataset_drawer1.draw()

def mouseup1(event):
    drawer1.draw(None)
    drawer1.selected = None
    dataset_drawer1.draw()

def move1(event):
    if drawer1.selected is not None:
        drawer1.selected.handle_press_do(event.offsetX, event.offsetY)
    dataset_drawer1.draw()

draw1(None)
dataset_drawer1.draw()


mousedown_proxy1 = create_proxy(mousedown1)
mouseup_proxy1 = create_proxy(mouseup1)
mouseover_proxy1 = create_proxy(move1)

# Attach event listeners to handle drawing
canvasNN1.addEventListener("mouseover", mouseover_proxy1)
canvasNN1.addEventListener("dragmove",  mouseover_proxy1)
canvasNN1.addEventListener("mousemove", mouseover_proxy1)
canvasNN1.addEventListener('mousedown', mousedown_proxy1)
canvasNN1.addEventListener("mouseup",   mouseup_proxy1)
canvasNN1.addEventListener("mouseup",   mouseup_proxy1)
inputField1.addEventListener("keyup",   mouseup_proxy1)
inputField1.addEventListener("keydown", mouseup_proxy1)
