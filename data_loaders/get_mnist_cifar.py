import numpy as np


def downsample(x, resolution):
    assert x.dtype == np.float32
    assert x.shape[1] % resolution == 0
    assert x.shape[2] % resolution == 0
    if x.shape[1] == x.shape[2] == resolution:
        return x
    s = x.shape
    x = np.reshape(x, [s[0], resolution, s[1] // resolution,
                       resolution, s[2] // resolution, s[3]])
    x = np.mean(x, (2, 4))
    return x


def x_to_uint8(x):
    x = np.clip(np.floor(x), 0, 255)
    return x.astype(np.uint8)


def shard(data, shards, rank):
    # Determinisitc shards
    x, y = data
    assert x.shape[0] == y.shape[0]
    assert x.shape[0] % shards == 0
    assert 0 <= rank < shards
    size = x.shape[0] // shards
    ind = rank*size
    return x[ind:ind+size], y[ind:ind+size]


def get_data(problem, shards, rank, data_augmentation_level, n_batch_train, n_batch_test, n_batch_init, resolution, flip_color=False, get_code=False):
    if problem == 'mnist':
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        y_train = np.reshape(y_train, [-1])
        y_test = np.reshape(y_test, [-1])
        if get_code:
            z_train = np.load('/afs/csail.mit.edu/u/y/yenchenlin/Workspace/glow/x2z/glow_train.npy')
            y_train = np.concatenate([y_train[:, np.newaxis], z_train], axis=1)
            z_test = np.load('/afs/csail.mit.edu/u/y/yenchenlin/Workspace/glow/x2z/glow_test.npy')
            y_test = np.concatenate([y_test[:, np.newaxis], z_test], axis=1)
        # Pad with zeros to make 32x32
        x_train = np.lib.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'minimum')
        # Pad with zeros to make 32x32
        x_test = np.lib.pad(x_test, ((0, 0), (2, 2), (2, 2)), 'minimum')
        x_train = np.tile(np.reshape(x_train, (-1, 32, 32, 1)), (1, 1, 1, 3))
        x_test = np.tile(np.reshape(x_test, (-1, 32, 32, 1)), (1, 1, 1, 3))
        if flip_color:
            x_train = 255.0 - x_train
            x_test = 255.0 - x_test
    elif problem == 'cifar10':
        from keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = np.reshape(y_train, [-1])
        y_test = np.reshape(y_test, [-1])
    else:
        raise Exception()

    print('n_train:', x_train.shape[0], 'n_test:', x_test.shape[0])

    # Shard before any shuffling
    x_train, y_train = shard((x_train, y_train), shards, rank)
    x_test, y_test = shard((x_test, y_test), shards, rank)

    print('n_shard_train:', x_train.shape[0], 'n_shard_test:', x_test.shape[0])

    from keras.preprocessing.image import ImageDataGenerator
    datagen_test = ImageDataGenerator()
    if data_augmentation_level == 0:
        datagen_train = ImageDataGenerator()
    else:
        if problem == 'mnist':
            datagen_train = ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1
            )
        elif problem == 'cifar10':
            if data_augmentation_level == 1:
                datagen_train = ImageDataGenerator(
                    width_shift_range=0.1,
                    height_shift_range=0.1
                )
            elif data_augmentation_level == 2:
                datagen_train = ImageDataGenerator(
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True,
                    rotation_range=15,  # degrees rotation
                    zoom_range=0.1,
                    shear_range=0.02,
                )
            else:
                raise Exception()
        else:
            raise Exception()

    datagen_train.fit(x_train)
    datagen_test.fit(x_test)
    train_flow = datagen_train.flow(x_train, y_train, n_batch_train)
    test_flow = datagen_test.flow(x_test, y_test, n_batch_test, shuffle=False)

    def make_iterator(flow, resolution, get_code=False):
        def iterator():
            x_full, yz = flow.next()
            x_full = x_full.astype(np.float32)
            x = downsample(x_full, resolution)
            x = x_to_uint8(x)
            if get_code:
                y = np.squeeze(yz[:, :1])
                z = yz[:, 1:]
                return x, y, z
            else:
                y = yz
                return x, y

        return iterator

    #init_iterator = make_iterator(train_flow, resolution)
    train_iterator = make_iterator(train_flow, resolution, get_code)
    test_iterator = make_iterator(test_flow, resolution, get_code)

    # Get data for initialization
    data_init = make_batch(train_iterator, n_batch_train, n_batch_init, get_code=get_code)

    return train_iterator, test_iterator, data_init


def make_batch(iterator, iterator_batch_size, required_batch_size, get_code=False):
    ib, rb = iterator_batch_size, required_batch_size
    #assert rb % ib == 0
    k = int(np.ceil(rb / ib))
    xs, ys, codes = [], [], []
    for i in range(k):
        if get_code:
            x, y, code = iterator()
            codes.append(code)
        else:
            x, y = iterator()
        xs.append(x)
        ys.append(y)
    x, y = np.concatenate(xs)[:rb], np.concatenate(ys)[:rb]
    if get_code:
        code = np.concatenate(codes)[:rb]
        return {'x': x, 'y': y, 'code': code}
    else:
        return {'x': x, 'y': y}
