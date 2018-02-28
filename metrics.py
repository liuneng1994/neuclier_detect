import numpy as np


def iou(label, predict, threadshold):
    if not isinstance(label, np.ndarray):
        label = np.asarray(label).astype(np.int32)
    batchs = label.shape[0]
    label = label.reshape((batchs, -1))
    if not isinstance(predict, np.ndarray):
        predict = np.asarray(predict)
    predict = predict.reshape((batchs, -1))
    predict = predict > threadshold
    correct = predict == label
    incorrect = predict != label
    tp = np.sum(correct.astype(np.int32) * label.astype(np.int32), axis=1).astype(np.float32)
    fp = np.sum(incorrect.astype(np.int32) * label.astype(np.int32), axis=1).astype(np.float32)
    fn = np.sum(incorrect.astype(np.int32) * np.where(label.astype(np.int32) == 1, 0, 1), axis=1).astype(np.float32)
    iou = tp / (tp + fp + fn)
    return iou


def mean_iou(label, predict):
    total = 0
    num = 0
    thredsholds = list(np.arange(0.5, 1.0, 0.05))
    for thredshold in thredsholds:
        result = iou(label,predict,thredshold)
        num = len(result)
        total += np.sum(result)
    return total / (num * len(thredsholds))


if __name__ == '__main__':
    predict = [[[1, 1], [0, 0]], [[1, 1], [0, 0]]]
    label = [[[True, True], [False, False]], [[True, True], [False, False]]]
    result = mean_iou(label, predict)
    print(result)
