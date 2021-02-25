
import visdom
import numpy as np

class Visualizer():
    def __init__(self):
        self.vis = visdom.Visdom(use_incoming_socket=False)

    def display_current_results(self, images, k=1):
        for i, (label, image) in enumerate(images.items(), 1):
            self.vis.images(image, win=i+k, opts=dict(title=label))

    def plot_current_loss(self, loss):
        if not hasattr(self, 'plot_data'):
            self.plot_data = []
        self.plot_data.append(loss)
        try:
            self.vis.line(
                Y=np.array(self.plot_data),
                opts={
                    'title': ' loss over time',
                    'xlabel': 'iteration',
                    'ylabel': 'loss',
                },
                win=0,
            )
        except ConnectionError:
            print('Could not connect to Visdom server')
            exit(1)

    def plot_current_eval(self, loss):
        if not hasattr(self, 'plot_data'):
            self.plot_data = []
        self.plot_data.append(loss)
        try:
            self.vis.line(
                Y=np.array(self.plot_data),
                opts={
                    'title': ' eval performance over time',
                    'xlabel': 'iteration',
                    'ylabel': 'eval performance',
                },
                win=1,
            )
        except ConnectionError:
            print('Could not connect to Visdom server')
            exit(1)

