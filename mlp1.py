import js
import csv
import math

import numpy as np

from pyodide.ffi import create_proxy

or_data_list = [[0,0,0], [0,1,1], [1,0,1], [1,1,1]]
xor_data_list = [[0,0,0], [0,1,1], [1,0,1], [1,1,0]]

canvasNN = js.document.getElementById('myNNCanvas1')
canvasNN.style.width  = canvasNN.width
canvasNN.style.height = canvasNN.height
ctxNN = canvasNN.getContext('2d')
canvasDS = js.document.getElementById('myDataCanvas1')
canvasDS.style.width  = canvasDS.width
canvasDS.style.height = canvasDS.height
ctxDS = canvasDS.getContext('2d')

# button = js.document.getElementById('myButton')

input_values = None

DISTANCE_TO_CONNECTION_THRESHOLD = 10
UNIT_SEPARATION = 120
LAYER_SEPARATION = 300
VALUES_SEPARATION = 30
MAX_WEIGHT = 5.
def numpy_array_to_latex(matrix: np.ndarray) -> str:
    latex_matrix = "\\\\begin{bmatrix}"
    for row in matrix:
        latex_matrix += " & ".join(map(str, row)) + " \\\\\\\\"
    latex_matrix += " \\\\end{bmatrix}"
    return latex_matrix

def project_point_to_segment(x1, y1, x2, y2, x, y):
    # Calculate the square of the length of the segment
    segment_length_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2
    # If the segment length is zero, return the distance between the point and the single endpoint
    if segment_length_squared == 0:
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    # Calculate the projection of point (x, y) onto the line defined by the segment
    t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / segment_length_squared
    # Clamp t to the range [0, 1] to ensure the projection falls on the segment
    t = max(0, min(1, t))
    # Find the coordinates of the projection point on the segment
    prj_x = x1 + t * (x2 - x1)
    prj_y = y1 + t * (y2 - y1)
    return prj_x, prj_y, t


def distance_point_to_segment(x1, y1, x2, y2, x, y):
    prj_x, prj_y, _ = project_point_to_segment(x1, y1, x2, y2, x, y)
    # Calculate the distance between the point and the projection
    distance = math.sqrt((x - prj_x) ** 2 + (y - prj_y) ** 2)    
    return distance


class MLP(object):
    def __init__(self, units, leak=0.1):
        super().__init__()
        self.units = units
        assert len(units) > 1, "MLP needs more than one layer (i.e., first one is input)."
        self.matrices = []
        self.leak = leak

        # First step is to flatten and a linear
        self.matrices.append(np.random.rand(units[1], units[0]))
        # After that, ReLU plus some more layers
        for i in range(len(units)-2):
            M = np.random.rand(units[i+2], units[i+1])
            M = (M-0.5)*2. # normalise -1 to 1
            self.matrices.append(M)
     
    def forward(self, x):
        t = np.array(x)
        # hidden units
        for idx, M in enumerate(self.matrices[:-1]):
            t = np.matmul(M, t) # linear
            t[t<0] *= self.leak # activation
        # output units
        t = np.matmul(self.matrices[-1], t) # linear
        t[t<0] *= self.leak # ReLu activation
        # t = 1./(1+np.exp(-t)) # Sigmoid activation
        return t


class NeuronDraw(object):
    def __init__(self, drawer, layer, unit, ctx):
        self.drawer = drawer
        self.canvas_width  = self.drawer.canvas.width
        self.canvas_height = self.drawer.canvas.height
        self.layer = layer
        self.unit = unit
        self.ctx = ctx
        self.value = None

        if layer == 0:
            units_in_layer = drawer.mlp.matrices[0].shape[1]
        else:
            units_in_layer = drawer.mlp.matrices[layer-1].shape[0]
        total_width = LAYER_SEPARATION*(len(drawer.mlp.matrices))
        total_height = UNIT_SEPARATION*(units_in_layer-1)
        x_base = (self.canvas_width - total_width) / 2
        y_base = (self.canvas_height - total_height) / 2
        self.x = LAYER_SEPARATION*(self.layer) + x_base
        self.y = UNIT_SEPARATION*(self.unit) + y_base

    def pos(self):
        return self.x, self.y

    def set_value(self, value):
        self.value = value

    def draw(self, event):
        self.ctx.beginPath()
        self.ctx.arc(self.x, self.y, 20, 0, 6.28)
        self.ctx.fillStyle = "#99000088"
        self.ctx.fill()
        self.ctx.stroke()
        self.ctx.font = "16px arial"
        self.ctx.textBaseline = "middle"
        self.ctx.textAlign = "center"
        self.ctx.fillStyle = "#000000"
        if self.value is not None:
            value = f"{self.value}"
            if len(value)>5:
                value = value[:5]
            self.ctx.fillText(value, self.x-40, self.y)

class ConnectionDraw(object):
    def __init__(self, drawer, layerA, unitA, xA, yA, layerB, unitB, xB, yB, ctx):
        self.drawer = drawer
        self.layerA = layerA
        self.unitA = unitA
        self.xA, self.yA = xA, yA
        self.layerB = layerB
        self.unitB = unitB
        self.xB, self.yB = xB, yB
        self.ctx = ctx

    def draw(self, event):
        self.ctx.beginPath()
        self.ctx.moveTo(self.xA, self.yA)
        self.ctx.lineTo(self.xB, self.yB)
        self.ctx.stroke()
        m = self.drawer.mlp.matrices[self.layerA]
        weight = m[self.unitB,self.unitA]
        if weight < -MAX_WEIGHT:
            weight = -MAX_WEIGHT
        if weight > MAX_WEIGHT:
            weight = MAX_WEIGHT
        xinc = self.xB - self.xA
        xbase = self.xA + xinc/4
        xp = xbase + xinc*(weight/MAX_WEIGHT)/4
        yinc = self.yB - self.yA
        ybase = self.yA + yinc/4
        yp = ybase + yinc*(weight/MAX_WEIGHT)/4
        self.ctx.beginPath()
        self.ctx.arc(self.xA, self.yA, 4, 0, 6.28) # base
        self.ctx.fillStyle = "#444444ff"
        self.ctx.fill()
        self.ctx.stroke()
        self.ctx.beginPath()
        self.ctx.arc(self.xA*0.75+self.xB*0.25, self.yA*0.75+self.yB*0.25, 3, 0, 6.28) # mid
        self.ctx.fillStyle = "#444444ff"
        self.ctx.fill()
        self.ctx.stroke()
        self.ctx.beginPath()
        self.ctx.arc((self.xA+self.xB)*0.5, (self.yA+self.yB)*0.5, 4, 0, 6.28) # end
        self.ctx.fillStyle = "#444444ff"
        self.ctx.fill()
        self.ctx.stroke()
        self.ctx.beginPath()
        self.ctx.arc(xp, yp, 8, 0, 6.28)  #slider circle
        self.ctx.fillStyle = "#44444444"
        self.ctx.fill()
        self.ctx.stroke()
        self.ctx.font = "14px arial"
        self.ctx.textBaseline = "middle"
        self.ctx.textAlign = "center"
        self.ctx.fillStyle = "#000000"
        value = f"{weight}"
        if len(value)>6:
            value = value[:6]
        if value[0] != "-":
            value = value[:-1]
        self.ctx.fillText(value, xp, yp-20)

            
    def update_model(self, v):
        m = self.drawer.mlp.matrices[self.layerA]
        m[self.unitB,self.unitA] = v

    def handle_press_check(self, x, y):
        x1 = self.xA
        y1 = self.yA
        x2 = (self.xA+self.xB)*0.5
        y2 = (self.yA+self.yB)*0.5
        d = distance_point_to_segment(x1, y1, x2, y2, x, y)
        if d < DISTANCE_TO_CONNECTION_THRESHOLD or self.drawer.selected == self:
            _, _, t = project_point_to_segment(x1, y1, x2, y2, x, y)
            if d < self.drawer.selected_distance:
                self.drawer.selected_distance = d
                self.drawer.selected = self

    def handle_press_do(self, x, y):
        x1 = self.xA
        y1 = self.yA
        x2 = (self.xA+self.xB)*0.5
        y2 = (self.yA+self.yB)*0.5
        _, _, t = project_point_to_segment(x1, y1, x2, y2, x, y)
        w = (t-0.5)*(MAX_WEIGHT*2)
        self.update_model(w)
        self.drawer.draw(None)


class MLPDraw(object):
    def __init__(self, mlp, canvas):
        super().__init__()
        self.mlp = mlp
        self.selected = None
        self.canvas = canvas
        self.ctx = self.canvas.getContext('2d')

        self.nodes = []
        self.connections = []

        # Create node and connection widgets
        for layer, m in enumerate(self.mlp.matrices):
            # For the first layer, we'll need to create the input nodes
            if layer == 0:
                layer_nodes = []
                for unit_input in range(m.shape[1]):
                    layer_nodes.append(NeuronDraw(self, layer, unit_input, self.ctx))
                self.nodes.append(layer_nodes)
            # Once we do the input layer, we continue with the first matrix and forwards
            layer_nodes = []
            for unit_dst in range(m.shape[0]):
                layer_nodes.append(NeuronDraw(self, layer+1, unit_dst, self.ctx))
            self.nodes.append(layer_nodes)
            # At this point we know all nodes have been created, proceed with the connections
            layer_connections = []
            for unit_dst in range(m.shape[0]):
                for unit_src in range(m.shape[1]):
                    cd = ConnectionDraw(self,
                                        layer, unit_src,
                                        *self.nodes[layer][unit_src].pos(),
                                        layer+1, unit_dst,
                                        *self.nodes[layer+1][unit_dst].pos(),
                                        self.ctx)
                    layer_connections.append(cd)
            self.connections.append(layer_connections)


    def draw(self, event):
        global input_values
        self.ctx.clearRect(0, 0, self.canvas.width, self.canvas.height)
        for connection_layer in self.connections:
            for connection in connection_layer:
                connection.draw(event)
        for layer_idx, layer in enumerate(self.nodes):
            for node_idx, node in enumerate(layer):
                if layer_idx == 0 and input_values != None:
                    node.set_value(input_values[node_idx][0])
                node.draw(event)

    def handle_press(self, x, y):
        self.selected = None
        self.selected_distance = math.inf
        for connection_layer in self.connections:
            for connection in connection_layer:
                connection.handle_press_check(x, y)

        if self.selected is not None:
            self.selected.handle_press_do(x,y)

        self.forward()

    def handle_move(self, x, y):
        if self.selected is not None:
            self.selected.handle_press_do(x, y)

    def forward(self, v=None):
        if v is None:
            text = js.document.getElementById("inputField1").value
            global input_values
            if len(text)>0:
                input_values = text.split(",")
                input_values = [ [float(v)] for v in input_values]
                v = np.array(input_values)

        if v is not None:
            ret = net.forward(v)
            text = numpy_array_to_latex(ret)
            # js.eval(f'katex.render("{text}", document.getElementById("outputText"));')
            output_layer = self.nodes[-1]
            self.ctx.font = "16px arial"
            self.ctx.textBaseline = "middle"
            self.ctx.textAlign = "left"
            self.ctx.fillStyle = "#000000"
            for idx in range(ret.shape[0]):
                neuron = output_layer[idx]
                pos = neuron.pos()
                value = f"{ret[idx][0]}"
                if len(value)>6:
                    value = value[:6]
                self.ctx.fillText(value, pos[0]+VALUES_SEPARATION, pos[1])


    def handle_release(self, x, y):
        self.selected = None


class ClassificationOutput(object):
    def __init__(self, mlp, canvas, xrange=[-5, 5], yrange=[-5, 5]):
        super().__init__()
        self.mlp = mlp
        self.canvas = canvas
        self.ctx = self.canvas.getContext('2d')
        self.xrange = xrange
        self.yrange = yrange
        self.data = np.array(or_data_list)

    def x2canvas(self, x):
        return (x-self.xrange[0]) / (self.xrange[1]-self.xrange[0]) * self.canvas.width

    def y2canvas(self, y):
        return (1 - (y-self.yrange[0]) / (self.yrange[1]-self.yrange[0])) * self.canvas.height

    def draw(self, event=None):
        def rgbFromValue(value):
            vv = float(value)
            if vv > 1:
                vv = 1
            elif vv < 0:
                vv = 0
            r = 255 - int(255.*vv)
            g = int(255.*vv)
            b = 0
            return '#%02x%02x%02x' % (r,g,b)

        self.ctx.clearRect(0, 0, self.canvas.width, self.canvas.height)
        self.ctx.lineWidth = 4

        for row_idx in range(self.data.shape[0]):
            row = self.data[row_idx]
            x = self.x2canvas(row[0])
            y = self.y2canvas(row[1])
            value = self.mlp.forward(row[:-1])[0]
            js.console.log(f"{row} ---> {value}")
            self.ctx.beginPath()
            self.ctx.arc(x, y, 10, 0, 6.28)
            self.ctx.fillStyle = rgbFromValue(value)
            self.ctx.strokeStyle = rgbFromValue(row[2])
            self.ctx.fill()
            self.ctx.stroke()
            self.ctx.font = "16px arial"
            self.ctx.textBaseline = "middle"
            self.ctx.textAlign = "center"
            
            self.ctx.fillStyle = "#000000"
            if value is not None:
                values = str(value)
                if len(values)>5:
                    values = values[:5]
                self.ctx.fillText(values, x-40, y)



net = MLP([2, 1])
ret = net.forward(np.array([[1.0],[2.0]]))
drawer = MLPDraw(net, canvasNN)

dataset_drawer = ClassificationOutput(net, canvasDS, xrange=[-0.5, 1.5], yrange=[-0.5, 1.5])

def recompute(event=None):
    drawer.forward()
    dataset_drawer.draw()

def draw(event):
    global drawer
    drawer.draw(event)

def mousedown(event):
    drawer.handle_press(event.offsetX, event.offsetY)

def mouseup(event):
    drawer.draw(event)
    drawer.selected = None
    recompute()

def move(event):
    if drawer.selected is not None:
        drawer.selected.handle_press_do(event.offsetX, event.offsetY)

draw(None)
dataset_drawer.draw()


mousedown_proxy = create_proxy(mousedown)
mouseup_proxy = create_proxy(mouseup)
mouseover_proxy = create_proxy(move)

# Attach event listeners to handle drawing
canvasNN.addEventListener("mouseover", mouseover_proxy)
canvasNN.addEventListener("dragmove", mouseover_proxy)
canvasNN.addEventListener("mousemove", mouseover_proxy)

canvasNN.addEventListener('mousedown', mousedown_proxy)

canvasNN.addEventListener("mouseup",  mouseup_proxy)
# canvasNN.addEventListener("mouseout", mouseup_proxy)
