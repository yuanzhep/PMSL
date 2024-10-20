# yz, 1015/2024

def get_dataset(dataset_name):
    def label_divider(y):
        return (y, None)

    elif 'celeba' in dataset_name:
        from dataset.celeba import gen
        adv = []
        if '_' in dataset_name:
            task = dataset_name.split('_')[1:]
        else:
            task = ['Smiling']
            adv = ['Male']
        data_train, data_test, label_divider, task_nc, adv_task_nc = gen(
            task, adv)

    else:
        raise NotImplementedError
    return data_train, data_test, label_divider, task_nc, adv_task_nc
