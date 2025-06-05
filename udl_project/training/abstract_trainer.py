"""
This module defines an abstract base class for model trainers.

Classes:
    Trainer (ABC): An abstract base class that enforces the implementation of a 'train' method for all subclasses.
"""

from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def train(self):
        pass
