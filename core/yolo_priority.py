import threading
from contextlib import contextmanager


class YOLOPriorityGate:
    def __init__(self):
        self.condition = threading.Condition()
        self.busy = False
        self.high_waiting = 0

    # CSRT YOLO 우선
    @contextmanager
    def high(self):
        with self.condition:
            self.high_waiting += 1

            try:
                while self.busy:
                    self.condition.wait()

                self.busy = True

            finally:
                self.high_waiting -= 1

        try:
            yield

        finally:
            with self.condition:
                self.busy = False
                self.condition.notify_all()

    # 갱신하기 YOLO 후순위
    @contextmanager
    def low(self):
        with self.condition:
            while self.busy or self.high_waiting > 0:
                self.condition.wait()

            self.busy = True

        try:
            yield

        finally:
            with self.condition:
                self.busy = False
                self.condition.notify_all()