from abc import ABC, abstractmethod


class BaseBuffer(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass
