"""Base abstract class defining the interface for all multimodal models."""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Define the interface for all multimodal models."""

    @abstractmethod
    def prepare_input(self, inputs):
        """Prepare model input from multi-modal data."""
        pass

    @abstractmethod
    def generate(self):
        """Generate a response from the model given prepared inputs."""
        pass
