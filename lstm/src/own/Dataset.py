class Dataset:
    def __init__(self, dataset, cols, datelist_train):
        self.dataset = dataset.astype('float32')
        self.cols = cols
        self.datelist_train = datelist_train

        print("Training set shape == {}".format(dataset))
        print("All timestamps == {}".format(len(datelist_train)))
        print("Features selected: {}".format(cols))
        print("Prepared dataset: ", self.dataset)
