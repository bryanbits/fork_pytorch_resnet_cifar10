import random
from collections import defaultdict
from torch.utils.data import Dataset, Subset, random_split
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

def split_train_val(dataset, val_size=0.2):
    """
    Returns a tuple of train and validation datasets.
    """
    n = len(dataset)
    n_val = int(n * val_size)
    n_train = n - n_val
    return random_split(dataset, [n_train, n_val])

def get_cifar10_dataset(fpath="data/cifar10", train=True, transform=ToTensor()):
    return datasets.CIFAR10(root=fpath, train=train, download=True, transform=transform)


def get_IMAGENET(fpath="data/IMAGENET-1K", train=True, transform=ToTensor()):
    return datasets.ImageNet(root=fpath, split="train" if train else "val", transform=transform)


def compute_maximal_subset_size(class_count, target_class_distribution) -> int:
    """
    :param class_count: dict of {class: count}
    :param target_class_distribution: dict of {class: target_percentage}
    :return: maximum train subset size
    """
    potential_max_sizes = []
    for cls in class_count:
        assert cls in target_class_distribution, f"{cls} not in class distribution"
        target_distribution = target_class_distribution[cls]
        max_size = int(class_count[cls] / target_distribution)
        potential_max_sizes.append(max_size)
    return min(potential_max_sizes)


def create_workload(class_labels=[0, 1], x=90, dataset=None) -> Subset:
    """
    Samples Dataset into the following distribution:
    - x/len(classes) % of each class in classes
    - 100-x % rest

    Returns a dataset with the above distribution.
    """
    # Get class distribution
    class_indices = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    ip_1 = {}
    ip_2 = {}
    for class_label in class_labels:
        ip_1[class_label] = len(class_indices[class_label])
        ip_2[class_label] = (x / 100) / (len(class_labels))

    rest_indices = []
    for i in range(10):
        if i not in class_labels:
            rest_indices += class_indices[i]

    ip_1[10] = len(rest_indices)
    ip_2[10] = (100 - x) / 100

    max_subset_size = compute_maximal_subset_size(ip_1, ip_2)

    sampled_indices = []
    for class_label in class_labels:
        sampled_indices += random.sample(class_indices[class_label],
                                         int(max_subset_size * (x / 100) / (len(class_labels))))
    sampled_indices += random.sample(rest_indices, int(max_subset_size * (100 - x) / 100))

    return Subset(dataset, sampled_indices)


def create_trainset(class_labels=[0, 1], dataset=None) -> Subset:
    """
    Samples the dataset and returns a training set consisting of the classes in the input "classes"
    """
    if class_labels == "full":
        return dataset

    # Get class distribution
    class_indices = [[], [], [], [], [], [], [], [], [], []]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    sampled_indices = []
    for class_label in class_labels:
        sampled_indices += class_indices[class_label]

    return Subset(dataset, sampled_indices)


def get_x_classes_from_imagenet_and_create_trainset_of_y_classes_with_other(dataset=None, x=50, y=5):
    indices_of_wanted_classes = {}
    for i in range(x):
        indices_of_wanted_classes[i] = []

    for idx, (_, label) in enumerate(dataset):
        if 0 <= label < x:
            indices_of_wanted_classes[label].append(idx)

    trainset_indices = {}
    valset_indices = {}

    for i in range(y):
        trainset_indices[i] = []
        trainset_indices[69] = []
        valset_indices[i] = []
        valset_indices[69] = []

    train_len_sum = 0
    val_len_sum = 0

    for key in indices_of_wanted_classes.keys():
        if 0 <= key < y:
            trainset_indices[key] += indices_of_wanted_classes[key][:int(len(indices_of_wanted_classes[key])*0.8)]
            train_len_sum += int(len(indices_of_wanted_classes[key]) * 0.8)
            valset_indices[key] += indices_of_wanted_classes[key][int(len(indices_of_wanted_classes[key])*0.8):]
            val_len_sum += int(len(indices_of_wanted_classes[key])*0.2)
        else:
            for index in indices_of_wanted_classes[key]:
                dataset.targets[index] = 69
            trainset_indices[69] += indices_of_wanted_classes[key][:int(len(indices_of_wanted_classes[key])*0.8)]
            valset_indices[69] += indices_of_wanted_classes[key][int(len(indices_of_wanted_classes[key])*0.8):]

    avg_train_len = int(train_len_sum / y)
    trainset_indices[69] = random.sample(trainset_indices[69], min(avg_train_len, len(trainset_indices[69])))

    avg_val_len = int(val_len_sum / y)
    valset_indices[69] = random.sample(valset_indices[69], min(avg_val_len, len(valset_indices[69])))

    sampled_trainset_indices = []
    for key in trainset_indices.keys():
        sampled_trainset_indices += trainset_indices[key]

    sampled_valset_indices = []
    for key in valset_indices.keys():
        sampled_valset_indices += valset_indices[key]

    return Subset(dataset, sampled_trainset_indices), Subset(dataset, sampled_valset_indices)


def cifar10_create_trainset_with_other(class_labels=[0, 1], dataset=None):
    class_indices = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_labels and label != 10:
            print("bug")
        else:
            class_indices[label].append(idx)

    sampled_indices = []
    for class_label in class_labels:
        sampled_indices += class_indices[class_label]

    sampled_indices += random.sample(class_indices[10],
                                     min(len(class_indices[class_labels[0]]), len(class_indices[10])))

    return Subset(dataset, sampled_indices)


def imagenet_create_trainset_with_other(class_labels=[0, 1], dataset=None):
    class_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # track counts
    key_count = {}
    for k, v in class_indices.items():
        key_count[k] = len(v)
    print("class_count: {}".format(key_count))

    sampled_indices = []
    for class_label in class_labels:
        sampled_indices += class_indices[class_label]

    sampled_indices += random.sample(class_indices[2007],
                                     min(len(class_indices[class_labels[0]]), len(class_indices[2007])))

    return Subset(dataset, sampled_indices)


def create_valset_with_other(class_labels=[0, 1], x=90, dataset=None):
    class_indices = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    ip_1 = {}
    ip_2 = {}
    for class_label in class_labels:
        ip_1[class_label] = len(class_indices[class_label])
        ip_2[class_label] = (x / 100) / (len(class_labels))

    ip_1[10] = len(class_indices[10])
    ip_2[10] = (100 - x) / 100

    max_subset_size = compute_maximal_subset_size(ip_1, ip_2)

    sampled_indices = []
    for class_label in class_labels:
        sampled_indices += random.sample(class_indices[class_label],
                                         int(max_subset_size * (x / 100) / (len(class_labels))))

    sampled_indices += random.sample(class_indices[10], int(max_subset_size * (100 - x) / 100))

    return Subset(dataset, sampled_indices)


def cifar10_change_labels(class_labels=[0, 1], dataset=None):
    for index in range(len(dataset.targets)):
        if dataset.targets[index] not in class_labels:
            dataset.targets[index] = 10

    unique_labels = list(set(dataset.targets))
    print(unique_labels)

    return dataset


def imagenet_change_labels(class_labels=[0, 1], dataset=None):
    for index in range(len(dataset.targets)):
        if dataset.targets[index] not in class_labels:
            dataset.targets[index] = 2007

    unique_labels = list(set(dataset.targets))
    print(unique_labels)
    return dataset


def create_order(size=10) -> list:
    order = []
    for i in range(size):
        rand = random.randint(0, 4)
        order.append(tuple([rand * 2, rand * 2 + 1]))

    return order
