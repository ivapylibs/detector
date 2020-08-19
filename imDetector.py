class imDetector(object):
    def __init__(self):
        self.preprocessor = []
        self.postprocessor = []

    def predict(self):
        raise NotImplementedError

    def measure(self, I):
        raise NotImplementedError

    def correct(self):
        raise NotImplementedError

    def adapt(self):
        raise NotImplementedError

    def process(self, I):
        self.predict()
        self.measure(I)
        self.correct()
        self.adapt()