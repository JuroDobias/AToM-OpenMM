import logging


class MaxLevelStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None, max_level=logging.INFO):
        super().__init__(stream)
        self.max_level = logging._checkLevel(max_level)

    def handle(self, record):
        if record.levelno > self.max_level:
            return False
        return super().handle(record)
