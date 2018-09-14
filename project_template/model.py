import cranial


class Model(cranial.base.StatefulModel):
    """A simple model that adds outputs containing the number of training
    examples seen."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state.train_count = 0
        self.state.live_count = 0

    def train(self, iterable):
        for i in iterable:
            self.state.train_count += 1

    def transform(self, record):
        self.state.live_count += 1

        record['train_count'] = self.state.train_count
        record['live_count'] = self.state.live_count

        return record
