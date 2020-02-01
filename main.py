from dataset import CocoDataset

if __name__ == '__main__':
    coco = CocoDataset()
    item = coco.__getitem__(0)
    print(item)
