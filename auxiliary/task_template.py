# using templates as base class for other tasks
from abc import ABC, abstractmethod
from auxiliary.model_interpreter import ModelInterpreter


class TaskTemplate(ABC):
    model_interpreter: ModelInterpreter

    @abstractmethod
    def generate_raw_outputs(self):
        pass
