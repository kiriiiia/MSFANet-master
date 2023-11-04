import numpy as np
import os
import ntpath
import time
import utils
from tensorboardX import SummaryWriter


class TFVisualizer():
    def __init__(self, opt):
        self.opt = opt
        self.saved = False
        self.ncols = 4

        self.log_name = os.path.join(opt.logDir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def print_logs(self, message):
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

