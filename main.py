from dataset import CocoDataset
from model import RetinaNet

if __name__ == '__main__':
    coco = CocoDataset()
    item = coco.__getitem__(0)
    net = RetinaNet()
    losses = net(item['img'].data.unsqueeze(0), item['img_meta'].data, item['gt_bboxes'].data, item['gt_labels'].data)
    print(losses)
