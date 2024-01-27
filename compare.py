from dataset import Dataset

def compute_confusion_matrix(train_ds: Dataset, test_ds: Dataset, model, **additional_params):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for (d, f, t) in test_ds.tuples():
        predicted = model(train_ds, freeman=f, **additional_params)
        