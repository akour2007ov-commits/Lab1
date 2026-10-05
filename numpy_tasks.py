"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    n = len(matrices[0])
    result = np.zeros((n, 1))
    for A, x in zip(matrices, vectors):
        result += A @ x
    return result


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    M = np.array(matrix)
    return np.where(M > threshold, 1, 0)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    M = np.array(matrix)
    return [np.unique(row).tolist() for row in M]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    M = np.array(matrix)
    return [np.unique(col).tolist() for col in M.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))

    mean_rows = np.mean(matrix, axis=1)
    var_rows = np.var(matrix, axis=1)

    mean_columns = np.mean(matrix, axis=0)
    var_columns = np.var(matrix, axis=0)

    return MatrixStatistics(
        matrix=matrix,
        mean_rows=mean_rows,
        mean_columns=mean_columns,
        var_rows=var_rows,
        var_columns=var_columns
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    matrix = np.full((rows, columns), second)
    row_indices, col_indices = np.ogrid[0:rows, 0:columns]
    chess_mask = (row_indices + col_indices) % 2 == 0
    matrix[chess_mask] = first
    return matrix


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)

    center_y = image_height // 2
    center_x = image_width // 2

    top = center_y - height // 2
    bottom = top + height
    left = center_x - width // 2
    right = left + width

    image[top:bottom, left:right] = shape_color
    return image

def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)

    y0 = (image_height - 1) / 2.0
    x0 = (image_width - 1) / 2.0

    y_indices, x_indices = np.ogrid[0:image_height, 0:image_width]
    ellipse_mask = ((x_indices - x0) ** 2) / (semi_axis_x ** 2) + ((y_indices - y0) ** 2) / (semi_axis_y ** 2) <= 1

    image[ellipse_mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    ts = np.array(values, dtype=float)

    mean_val = float(np.mean(ts))
    var_val = float(np.var(ts))
    std_val = float(np.std(ts))

    is_max = (ts[1:-1] > ts[:-2]) & (ts[1:-1] > ts[2:])
    is_min = (ts[1:-1] < ts[:-2]) & (ts[1:-1] < ts[2:])

    max_indices = (np.where(is_max)[0] + 1).tolist()
    min_indices = (np.where(is_min)[0] + 1).tolist()

    moving_avg = np.convolve(ts, np.ones(window), mode='valid') / window

    return TimeSeriesStatistics(
        mean=mean_val,
        variance=var_val,
        std_deviation=std_val,
        local_maxima=max_indices,
        local_minima=min_indices,
        moving_average=moving_avg.tolist()
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels_arr = np.array(labels, dtype=int)
    return np.eye(class_count)[labels_arr]
