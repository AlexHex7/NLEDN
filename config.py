import os


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


CUDA_NUMBER = 0
weight_dir = 'weights/'
test_batch_size = 1

data_type_list = ['Rain100L', 'Rain100H', 'DDN', 'DID']
data_type = data_type_list[2]


weight_path = os.path.join(weight_dir, 'net_%s.pth' % data_type)
test_dir = 'dataset/test/%s' % data_type

test_result_root = 'Results/'
test_compare_results_dir = os.path.join(test_result_root, '%s_compare' % data_type)
test_results_dir = os.path.join(test_result_root, '%s' % data_type)

create_dir(test_result_root)
create_dir(test_compare_results_dir)
create_dir(test_results_dir)
