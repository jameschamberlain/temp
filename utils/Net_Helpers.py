class EarlyStopper:
    def __init__(self,tolerance):
        self.tolerance = tolerance
        self.counter = 0
        self.best_value = None

    # We should also probably save the model
    def stop(self, dev_loss):
        if self.best_value is None:
            self.best_value = dev_loss
            return False
        if dev_loss < self.best_value:
            print("NEW BEST VALIDATION LOSS")
            self.best_value = dev_loss
            self.counter = 0
            return False
        if dev_loss > self.best_value:
            self.counter += 1
            if self.counter >= self.tolerance:
                return True
            else:
                return False