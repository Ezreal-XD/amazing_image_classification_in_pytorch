

#### CIFAR10

| MODEL       | ACC       | PARAS   | P.S.    |
| ----------- | --------- | ------- | ------- |
| VGG12 + GAP | 94.30     | 7.32 M  | 2 hrs   |
| SimpleDLA   | 95.22     | 15.14 M | 5 hrs   |
| DLA         | 95.46     | 16.29 M | 6 hrs   |
| Inception   | **95.60** | 6.17 M  | 13 hrs  |
| ResNet34    | 95.45     | 21.28 M | 5.3 hrs |
| ResNet50    | 94.69     | 23.52 M | 11 hrs  |
| Xception    | 94.69     | 20.83 M | 10 hrs  |
| SENet       |           | 11.26 M | 3.5 hrs |

注：默认

* MAX_EPOCH = 200
* lr = 0.1
* lr_schedule = poly
* weight dacay = 5e-4
* batch size = 128
* optim = sgd

