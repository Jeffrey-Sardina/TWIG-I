from run_exp import load_dataset, load_filters, load_negative_samplers
import sys
import torch

'''
================
Module Functions
================
'''
def one_v_many(dataloader, negative_sampler, purpose):
    average_X_pos = None
    average_all_neg_vals = None
    std_all_neg_vals = None

    for X_pos in dataloader:
        assert len(X_pos) == 1, 'we expect this to be a one-elem list'
        X_pos = X_pos[0]
        assert X_pos.shape[0] == 1, 'Expected a batch siez of 1'

        triple_idxs = X_pos[:, 0]
        X_pos = X_pos[:, 1:]
        X_pos = torch.squeeze(X_pos)

        X_neg, npps = negative_sampler.get_batch_negatives(
            purpose=purpose,
            triple_idxs=triple_idxs,
            npp=-1
        )
        X_neg_avg = torch.mean(X_neg, dim=0)
        X_neg_std = torch.std(X_neg, dim=0)

        if average_X_pos is None:
            average_X_pos = X_pos / len(dataloader)
        else:
            average_X_pos += X_pos / len(dataloader)

        if average_all_neg_vals is None:
            average_all_neg_vals = X_neg_avg / len(dataloader)
        else:
            average_all_neg_vals += X_neg_avg / len(dataloader)

        if std_all_neg_vals is None:
            std_all_neg_vals = (X_pos - X_neg_avg) ** 2
        else:
            std_all_neg_vals += (X_pos - X_neg_avg) ** 2

        print('Aalysis for positive triple with ID:', int(triple_idxs))
        print('true \t n-avg \t n-std \t zscore')
        print('---- \t ----- \t ----- \t ------')
        for i in range(len(X_neg_avg)):
            true_val = round(float(X_pos[i]), 2)
            neg_avg = round(float(X_neg_avg[i]), 2)
            neg_std = round(float(X_neg_std[i]), 2)
            if neg_std == 0:
                z_score = 'NA'
            else:
                z_score = (true_val - neg_avg) / neg_std
                z_score = round(z_score, 2)

            print(f'{true_val} \t {neg_avg} \t {neg_std}, \t {z_score}')
        print()

    std_all_neg_vals = torch.sqrt(
        (1 / (len(dataloader) - 1)) * std_all_neg_vals
    )

    print('Overall Analysis:')
    print('true \t n-avg \t n-std \t zscore')
    print('---- \t ----- \t ----- \t ------')
    for i in range(len(X_neg_avg)):
        true_val = round(float(average_X_pos[i]), 2)
        neg_avg = round(float(average_all_neg_vals[i]), 2)
        neg_std = round(float(std_all_neg_vals[i]), 2)
        if neg_std == 0:
            z_score = 'NA'
        else:
            z_score = (true_val - neg_avg) / neg_std
            z_score = round(z_score, 2)

        print(f'{true_val} \t {neg_avg} \t {neg_std}, \t {z_score}')
    print()

def main(
        dataset_names,
        normalisation,
        batch_size,
        batch_size_test,
        sampler_type,
        use_train_filter,
        use_valid_and_test_filters,
    ):
    dataloaders, norm_funcs = load_dataset(
        dataset_names,
        normalisation=normalisation,
        batch_size=batch_size,
        batch_size_test=batch_size_test
    )

    filters = load_filters(
        dataset_names,
        use_train_filter=use_train_filter,
        use_valid_and_test_filters=use_valid_and_test_filters
    )
    negative_samplers = load_negative_samplers(
        dataset_names,
        filters,
        normalisation,
        norm_funcs,
        sampler_type=sampler_type,
    )

    dataset_name = dataset_names[0]
    purpose = 'test'
    print(f'Analysis for {dataset_name}')
    one_v_many(
        dataloaders[purpose][dataset_name],
        negative_samplers[dataset_name],
        purpose
    )

if __name__ == '__main__':
    print(sys.argv)
    dataset_names = sys.argv[1].split('-')
    normalisation = sys.argv[2]
    batch_size = int(sys.argv[3])
    batch_size_test = int(sys.argv[4])
    use_train_filter = sys.argv[5] == '1'
    use_valid_and_test_filters = sys.argv[6] == '1'
    sampler_type = sys.argv[7]

    main(
        dataset_names,
        normalisation,
        batch_size,
        batch_size_test,
        sampler_type,
        use_train_filter,
        use_valid_and_test_filters
    )
