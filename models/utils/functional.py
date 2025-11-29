import torch


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        #  torch.index_select输出的是输入张量的指定维度的指定索引号进行索引的张量子集
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def get_tpfpfn(pr, gt, n_classes, device):
    """Calculate TP、FP and FN
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        n_classes: number of class
        device: "cuda" or "cpu"
    Returns:
        float: tp, fp, fn, n_pixels
    """
    tp = torch.zeros(n_classes)
    fp = torch.zeros(n_classes)
    fn = torch.zeros(n_classes)
    n_pixels = torch.zeros(n_classes)
    tp = tp.to(device)
    fp = fp.to(device)
    fn = fn.to(device)
    n_pixels = n_pixels.to(device)
    pr = pr.flatten()
    gt = gt.flatten()

    for c in range(n_classes):
        tp[c] += torch.sum((pr == c) * (gt == c))
        fp[c] += torch.sum((pr == c) * (gt != c))
        fn[c] += torch.sum((pr != c) * (gt == c))
        n_pixels[c] += torch.sum(gt == c)

    return tp, fp, fn, n_pixels


def iou(pr, gt, eps=1e-7, n_classes=2, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """
    # pr = _threshold(pr, threshold=threshold)
    # pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp, fp, fn, n_pixels = get_tpfpfn(pr, gt, n_classes, "cuda")
    cl_wise_score = tp / (tp + fp + fn + eps)
    score = torch.mean(cl_wise_score)

    # intersection = torch.sum(gt * pr)
    # union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    # return (intersection + eps) / union
    return score

def FWiou(pr, gt, eps=1e-7, n_classes=2, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """
    # pr = _threshold(pr, threshold=threshold)
    # pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp, fp, fn, n_pixels = get_tpfpfn(pr, gt, n_classes, "cuda")

    freq = (tp + fn) / torch.sum(n_pixels + eps)
    cl_wise_score = tp / (tp + fp + fn + eps)
    #
    # print(freq)
    # print(cl_wise_score)
    # print(freq * cl_wise_score)
    fwioU = torch.sum(freq * cl_wise_score)

    return fwioU

jaccard = iou


def f_score(pr, gt, beta=1, eps=1e-7, n_classes=2, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """
    # tp = torch.sum(gt * pr)
    # fp = torch.sum(pr) - tp
    # fn = torch.sum(gt) - tp
    tp, fp, fn, n_pixels = get_tpfpfn(pr, gt, n_classes, "cuda")
    # tp, fp, fn, n_pixels = get_tpfpfn(pr, gt, 3, "cpu")
    f_score = ((1 + beta ** 2) * tp + eps) \
              / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)
    return torch.mean(f_score)

def f_score_each(pr, gt, beta=1, eps=1e-7, n_classes=2, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """
    # tp = torch.sum(gt * pr)
    # fp = torch.sum(pr) - tp
    # fn = torch.sum(gt) - tp
    tp, fp, fn, n_pixels = get_tpfpfn(pr, gt, n_classes, "cuda")
    # tp, fp, fn, n_pixels = get_tpfpfn(pr, gt, 3, "cpu")
    f_score = ((1 + beta ** 2) * tp + eps) \
              / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)
    return f_score

def accuracy(pr, gt, eps=1e-7, n_classes=2, threshold=0.5,  ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    # pr = _threshold(pr, threshold=threshold)
    # pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    # tp_tn = torch.sum(gt == pr, dtype=pr.dtype)
    # score = tp_tn / gt.view(-1).shape[0]
    tp, fp, fn, n_pixels = get_tpfpfn(pr, gt, n_classes, "cuda")
    # tp, fp, fn, n_pixels = get_tpfpfn(pr, gt, 3, "cpu")
    acc = torch.sum(tp + eps) / torch.sum(n_pixels + eps)
    return acc


def precision(pr, gt, eps=1e-7, n_classes=2, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    # pr = _threshold(pr, threshold=threshold)
    # pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    # tp = torch.sum(gt * pr)
    # fp = torch.sum(pr) - tp
    tp, fp, fn, n_pixels = get_tpfpfn(pr, gt, n_classes, "cuda")
    # tp, fp, fn, n_pixels = get_tpfpfn(pr, gt, 3, "cpu")
    prec = (tp + eps) / (tp + fp + eps)
    return torch.mean(prec)


def recall(pr, gt, eps=1e-7, n_classes=2, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """
    # pr = _threshold(pr, threshold=threshold)
    # pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    # tp = torch.sum(gt * pr)
    # fn = torch.sum(gt) - tp
    tp, fp, fn, n_pixels = get_tpfpfn(pr, gt, n_classes, "cuda")
    # tp, fp, fn, n_pixels = get_tpfpfn(pr, gt, 3, "cpu")
    recall = (tp + eps) / (tp + fn + eps)
    return torch.mean(recall)


def kappa(pr, gt, eps=1e-7, n_classes=2, threshold=None, ignore_channels=None):
    """Calculate kappa score between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: kappa score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    # tp = torch.sum(gt * pr)
    # fp = torch.sum(pr) - tp
    # fn = torch.sum(gt) - tp
    # tn = torch.sum((1 - gt)*(1 - pr))
    tp, fp, fn, tn = get_tpfpfn(pr, gt, n_classes, "cuda")
    # tp, fp, fn, tn = get_tpfpfn(pr, gt, 3, "cpu")

    N = tp + tn + fp + fn
    p0 = (tp + tn) / N
    pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (N * N)
    score = (p0 - pe) / (1 - pe)
    return torch.mean(score)


def dice(pr, gt, eps=1e-7, n_classes=2, threshold=None, ignore_channels=None):
    """Calculate dice score between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: dice score
    """
    # pr = _threshold(pr, threshold=threshold)
    # pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    # tp = torch.sum(gt * pr)
    # fp = torch.sum(pr) - tp
    # fn = torch.sum(gt) - tp
    _precision = precision(pr, gt, eps=eps, threshold=threshold, ignore_channels=ignore_channels)
    _recall = recall(pr, gt, eps=eps, threshold=threshold, ignore_channels=ignore_channels)
    dice = 2 * _precision * _recall / (_precision + _recall)
    return dice
