# -*- coding: utf-8 -*-
from enum import Enum


class LevelOrder(Enum):
    """Define order of levels indexing

    TOP_TO_BOTTOM is for axis with 0 at the top
    BOTTOM_TO_TOP is for axis with 0 at the ground level
    """

    TOP_TO_BOTTOM = -1
    BOTTOM_TO_TOP = 1
