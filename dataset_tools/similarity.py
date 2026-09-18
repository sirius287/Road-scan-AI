"""Lightweight duplicate candidates; not physical-pothole tracking."""
from __future__ import annotations
import cv2
import numpy as np


def dhash64(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    result = 0
    for bit in (small[:, 1:] > small[:, :-1]).flat:
        result = (result << 1) | int(bit)
    return result


class BKTree:
    """Hamming-distance search without an N by N image-similarity matrix."""
    def __init__(self) -> None:
        self.root = None

    def add(self, value: int, item: str) -> None:
        if self.root is None:
            self.root = [value, [item], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(item)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [item], {}]
                return
            node = node[2][distance]

    def query(self, value: int, radius: int):
        stack = [self.root] if self.root is not None else []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                for item in node[1]:
                    yield item, distance
            for edge, child in node[2].items():
                if distance - radius <= edge <= distance + radius:
                    stack.append(child)
