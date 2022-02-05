if True:
    from utils import get_params
    print(get_params(512, 5))

elif True:
    import datetime as dt

    s = 'b0x6239eb3102218F53D4B3413e2d58E114C4C62caE'
    print(len(s))
    print(len(s) * 2)

    d = dt.datetime.utcnow().date()
    print(d)


elif True:
    epochs = 100

    lr = 2e-4
    decay_steps = 100000
    decay_rate = 0.98

    steps_per_epoch = 32000

    epochs_per_decay = decay_steps / steps_per_epoch


    for i in range(int(epochs / epochs_per_decay)):
        print(epochs_per_decay * i, ': ', lr)
        lr *= decay_rate
