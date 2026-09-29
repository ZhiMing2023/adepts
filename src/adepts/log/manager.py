import logging


class LogManager:
    """Small console logger used by the solvers."""

    def __init__(self):
        self.logger = logging.getLogger("adepts")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

    def log_h1(self, text, level="INFO", width=100, pad="#"):
        del width, pad
        self.logger.log(getattr(logging, level.upper()), text)

    def log_body(self, text, indent=0, level="INFO"):
        self.logger.log(getattr(logging, level.upper()), "%s%s", " " * indent, text)

    inv_h1 = log_h1
    inv_body = log_body
