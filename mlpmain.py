import js
import math


from pyodide.ffi import create_proxy

canvas = js.document.getElementById('myCanvas')
canvas.style.width = canvas.width
canvas.style.height = canvas.height

# button = js.document.getElementById('myButton')
ctx = canvas.getContext('2d')

import numpy as np

DISTANCE_TO_CONNECTION_THRESHOLD = 10
UNIT_SEPARATION = 80
LAYER_SEPARATION = 200
VALUES_SEPARATION = 30
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
        t = x*1
        # hidden units
        for idx, M in enumerate(self.matrices[:-1]):
            t = np.matmul(M, t) # linear
            t[t<0] *= self.leak # activation
        # output units
        t = np.matmul(self.matrices[-1], t) # linear
        t = 1./(1+np.exp(-t)) # sigmoid
        return t


class NeuronDraw(object):
    def __init__(self, drawer, layer, unit, ctx):
        self.drawer = drawer
        self.canvas_width  = self.drawer.canvas.width
        self.canvas_height = self.drawer.canvas.height
        self.layer = layer
        self.unit = unit
        self.ctx = ctx

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

    def draw(self, event):
        self.ctx.beginPath()
        self.ctx.arc(self.x, self.y, 20, 0, 6.28)
        self.ctx.fillStyle = "#99000088"
        self.ctx.fill()
        self.ctx.stroke()


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
        #print("Getting matrix", self.layerA)
        m = self.drawer.mlp.matrices[self.layerA]
        #print("matrix shape", m.shape)
        #print("accessing", self.unitB,self.unitA)
        weight = m[self.unitB,self.unitA]
        if weight < -1:
            weight = -1
        if weight > 1:
            weight = 1
        xinc = self.xB - self.xA
        xbase = self.xA + xinc/4
        xp = xbase + xinc*weight/4
        yinc = self.yB - self.yA
        ybase = self.yA + yinc/4
        yp = ybase + yinc*weight/4
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
        self.ctx.arc(xp, yp, 8, 0, 6.28)  #slider
        self.ctx.fillStyle = "#44444444"
        self.ctx.fill()
        self.ctx.stroke()

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
        js.console.log(self)
        x1 = self.xA
        y1 = self.yA
        x2 = (self.xA+self.xB)*0.5
        y2 = (self.yA+self.yB)*0.5
        _, _, t = project_point_to_segment(x1, y1, x2, y2, x, y)
        w = (t-0.5)*2
        self.update_model(w)
        self.drawer.draw(None)


class MLPDraw(object):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        self.selected = None
        self.canvas = js.document.getElementById('myCanvas')
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
        self.ctx.clearRect(0, 0, self.canvas.width, self.canvas.height)
        for connection_layer in self.connections:
            for connection in connection_layer:
                connection.draw(event)
        for node_layer in self.nodes:
            for node in node_layer:
                node.draw(event)

    def handle_press(self, x, y):
        self.selected = None
        self.selected_distance = math.inf
        for connection_layer in self.connections:
            for connection in connection_layer:
                connection.handle_press_check(x, y)

        self.selected.handle_press_do(x,y)

        self.forward()

    def handle_move(self, x, y):
        if self.selected is not None:
            self.selected.handle_press_do(x, y)

    def forward(self):
        text = js.document.getElementById("inputField").value
        values = text.split(",")
        values = [ [float(v)] for v in values]
        v = np.array(values)
        ret = net.forward(v)
        text = numpy_array_to_latex(ret)
        js.eval(f'katex.render("{text}", document.getElementById("outputText"));')

        output_layer = self.nodes[-1]
        #     def pos(self):
        # return self.x, self.y
    
        ctx.font = "16px arial"
        ctx.textBaseline = "middle"
        ctx.textAlign = "left"
        ctx.fillStyle = "#000000"
        for idx in range(ret.shape[0]):
            neuron = output_layer[idx]
            pos = neuron.pos()
            value = f"{ret[idx][0]}"
            if len(value)>10:
                value = value[:10]
            ctx.fillText(value, pos[0]+VALUES_SEPARATION, pos[1])


    def handle_release(self, x, y):
        self.selected = None


net = MLP([2, 4, 2])
ret = net.forward(np.array([[1.0],[2.0]]))
drawer = MLPDraw(net)

def recompute(event=None):
    drawer.forward()

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


mousedown_proxy = create_proxy(mousedown)
mouseup_proxy = create_proxy(mouseup)
mouseover_proxy = create_proxy(move)

# Attach event listeners to handle drawing
canvas.addEventListener("mouseover", mouseover_proxy)
canvas.addEventListener("dragmove", mouseover_proxy)
canvas.addEventListener("mousemove", mouseover_proxy)

canvas.addEventListener('mousedown', mousedown_proxy)

canvas.addEventListener("mouseup",  mouseup_proxy)
canvas.addEventListener("mouseout", mouseup_proxy)
