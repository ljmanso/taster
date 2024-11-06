import js
import numpy as np

from pyodide.ffi import create_proxy


canvasNN2 = js.document.getElementById('myNNCanvas2')
canvasNN2.style.width  = canvasNN2.width
canvasNN2.style.height = canvasNN2.height
ctxNN2 = canvasNN2.getContext('2d')
canvasDS2 = js.document.getElementById('myDataCanvas2')
canvasDS2.style.width  = canvasDS2.width
canvasDS2.style.height = canvasDS2.height
ctxDS2 = canvasDS2.getContext('2d')

inputField2 = js.document.getElementById('inputField2')

from mlpmain import MLP, MLPDraw, ClassificationOutput

net2 = MLP([2, 1])
drawer2 = MLPDraw(net2, canvasNN2, "inputField2")

xor_data_list = [[0,0,0], [0,1,1], [1,0,1], [1,1,0]]
dataset_drawer2 = ClassificationOutput(net2, canvasDS2, data=xor_data_list, xrange=[-0.5, 1.5], yrange=[-0.5, 1.5])


def draw2(event=None):
    global drawer2
    drawer2.draw(event)
    dataset_drawer2.draw()

def mousedown2(event):
    drawer2.handle_press(event.offsetX, event.offsetY)
    dataset_drawer2.draw()

def mouseup2(event):
    drawer2.draw(None)
    drawer2.selected = None
    dataset_drawer2.draw()

def move2(event):
    if drawer2.selected is not None:
        drawer2.selected.handle_press_do(event.offsetX, event.offsetY)
    dataset_drawer2.draw()

draw2(None)
dataset_drawer2.draw()


mousedown_proxy2 = create_proxy(mousedown2)
mouseup_proxy2 = create_proxy(mouseup2)
mouseover_proxy2 = create_proxy(move2)

# Attach event listeners to handle drawing
canvasNN2.addEventListener("mouseover", mouseover_proxy2)
canvasNN2.addEventListener("dragmove",  mouseover_proxy2)
canvasNN2.addEventListener("mousemove", mouseover_proxy2)
canvasNN2.addEventListener('mousedown', mousedown_proxy2)
canvasNN2.addEventListener("mouseup",   mouseup_proxy2)
inputField2.addEventListener("keyup",   mouseup_proxy2)
inputField2.addEventListener("keydown", mouseup_proxy2)

