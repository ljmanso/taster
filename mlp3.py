import js
import numpy as np

from pyodide.ffi import create_proxy


canvasNN3 = js.document.getElementById('myNNCanvas3')
canvasNN3.style.width  = canvasNN3.width
canvasNN3.style.height = canvasNN3.height
ctxNN3 = canvasNN3.getContext('2d')
# canvasDS3 = js.document.getElementById('myDataCanvas3')
# canvasDS3.style.width  = canvasDS3.width
# canvasDS3.style.height = canvasDS3.height
# ctxDS3 = canvasDS3.getContext('2d')

from mlpmain import MLP, MLPDraw, ClassificationOutput

net3 = MLP([5, 8, 8, 1])


params = {
    "DISTANCE_TO_CONNECTION_THRESHOLD": 10,
    "UNIT_SEPARATION": 57,
    "LAYER_SEPARATION": 230,
    "VALUES_SEPARATION": 30
    }

drawer3 = MLPDraw(net3, canvasNN3, "inputField3", params=params)


# dataset_drawer3 = ClassificationOutput(net3, canvasDS3, xrange=[-0.5, 1.5], yrange=[-0.5, 1.5])


def draw3(event=None):
    global drawer3
    drawer3.draw(event)

def mousedown3(event):
    drawer3.handle_press(event.offsetX, event.offsetY)

def mouseup3(event):
    drawer3.draw(event)
    drawer3.selected = None

def move3(event):
    if drawer3.selected is not None:
        drawer3.selected.handle_press_do(event.offsetX, event.offsetY)

draw3(None)
#dataset_drawer3.draw()


mousedown_proxy3 = create_proxy(mousedown3)
mouseup_proxy3 = create_proxy(mouseup3)
mouseover_proxy3 = create_proxy(move3)

# Attach event listeners to handle drawing
canvasNN3.addEventListener("mouseover", mouseover_proxy3)
canvasNN3.addEventListener("dragmove", mouseover_proxy3)
canvasNN3.addEventListener("mousemove", mouseover_proxy3)
canvasNN3.addEventListener('mousedown', mousedown_proxy3)
canvasNN3.addEventListener("mouseup",  mouseup_proxy3)

