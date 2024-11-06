import js
import numpy as np

from pyodide.ffi import create_proxy

import cma

canvasNN4 = js.document.getElementById('myNNCanvas4')
canvasNN4.style.width  = canvasNN4.width
canvasNN4.style.height = canvasNN4.height
ctxNN4 = canvasNN4.getContext('2d')
canvasDS4 = js.document.getElementById('myDataCanvas4')
canvasDS4.style.width  = canvasDS4.width
canvasDS4.style.height = canvasDS4.height
ctxDS4 = canvasDS4.getContext('2d')

button4 = js.document.getElementById('myButton4')

from mlpmain import MLP, MLPDraw, ClassificationOutput, MAX_WEIGHT

def generate_spiral(N):
    theta = np.sqrt(np.random.rand(N))*2*np.pi
    r_a = 2*theta + np.pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + np.random.randn(N,2)
    r_b = -2*theta - np.pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = data_b + np.random.randn(N,2)
    res_a = np.append(x_a, np.zeros((N,1)), axis=1)
    res_b = np.append(x_b, np.ones((N,1)), axis=1)
    ret = np.append(res_a, res_b, axis=0)
    np.random.shuffle(ret)
    xmin = ret[:,0].min()
    xmax = ret[:,0].max()
    ret[:,0] = (ret[:,0]-xmin)/(xmax-xmin)-0.5
    ymin = ret[:,1].min()
    ymax = ret[:,1].max()
    ret[:,1] = (ret[:,1]-ymin)/(ymax-ymin)-0.5
    return ret.tolist()

data = generate_spiral(40)


SIGMA = 1.
POPULATION_SIZE = 50

config = [2, 8, 5, 1]
bestNet4 = MLP(config)
tempNet4 = MLP(config)
netS4 = [MLP(config) for _ in range(POPULATION_SIZE)]
best_fitness = None

def assess(solution, data):
    # js.console.log(f"Solution size {solution.shape=}")
    loss = 0
    count = 0
    for item in data:
        input = np.expand_dims(np.array(item[:2]), axis=1)
        tempNet4.set_params_from_list(solution)
        est = tempNet4.forward(input)[0,0]
        gt = item[2]
        loss += (gt-est)*(gt-est)
        count += 1
    fitness = loss/count
    return fitness

es4 = cma.CMAEvolutionStrategy(len(netS4[0].params_to_list())*[0], SIGMA, {'popsize': POPULATION_SIZE, 'verb_disp': 0, 'verbose': -1} )


params4 = {
    "DISTANCE_TO_CONNECTION_THRESHOLD": 10,
    "UNIT_SEPARATION": 50,
    "LAYER_SEPARATION": 98,
    "VALUES_SEPARATION": 40
    }

drawer4 = MLPDraw(bestNet4, canvasNN4, "inputField4", params=params4)


dataset_drawer4 = ClassificationOutput(bestNet4, canvasDS4, data=data, xrange=[-0.6, 0.6], yrange=[-0.6, 0.6], notext=True, threshold=0.5)


def draw4(event=None):
    global drawer4
    drawer4.draw(event)

def mousedown4(event):
    drawer4.handle_press(event.offsetX, event.offsetY)

def mouseup4(event):
    drawer4.draw(event)
    drawer4.selected = None

def move4(event):
    if drawer4.selected is not None:
        drawer4.selected.handle_press_do(event.offsetX, event.offsetY)

draw4(None)
dataset_drawer4.draw()

iteration4 = 0
def evolve4(event=None):
    global MAX_WEIGHT
    global es4
    global iteration4
    global bestNet4
    global dataset_drawer4
    global best_fitness
    global best4
    MAX_WEIGHT = 10.
    solutions4 = es4.ask()
    fitnesses4 = [assess(x, data) for x in solutions4]
    best_index4 = np.argmin(fitnesses4)
    if best_fitness is None or fitnesses4[best_index4] < best_fitness:
        best_fitness = fitnesses4[best_index4]
        best4 = solutions4[best_index4]
    es4.tell(solutions4, fitnesses4)
    # print(f"({iteration4})-->{fitnesses4[best_index4]}", end="  ")
    # js.console.log(f"{len(fitnesses4)=}  {len(solutions4[0])=}")
    # js.console.log(f"({iteration4})-->{fitnesses4[best_index4-1:best_index4+2]}")

    iteration4 += 1
    js.document.getElementById('myButton4').innerHTML = f"({iteration4}) {best_fitness}"
    bestNet4.set_params_from_list(best4)
    draw4(None)
    dataset_drawer4.draw()

mousedown_proxy4 = create_proxy(mousedown4)
mouseup_proxy4 = create_proxy(mouseup4)
mouseover_proxy4 = create_proxy(move4)
evolve_proxy4 = create_proxy(evolve4)
# Attach event listeners to handle drawing
canvasNN4.addEventListener("mouseover", mouseover_proxy4)
canvasNN4.addEventListener("dragmove", mouseover_proxy4)
canvasNN4.addEventListener("mousemove", mouseover_proxy4)
canvasNN4.addEventListener('mousedown', mousedown_proxy4)
canvasNN4.addEventListener("mouseup",  mouseup_proxy4)
button4.addEventListener("click", evolve_proxy4)

