#python test.py --img_file "/media/lab225-2/disk1/WBY/datasets/celeba_hq_256_test" --continue_train --gpu_ids 2 --batchSize 8 --mask_type
#/media/sda/datasets/inpainting/celeba_hq_256_test
#python train.py --no_flip --no_rotation --no_augment --img_file "/media/sda/datasets/inpainting/celeba-256" --lr 1e-4 --gpu_ids 2 --epoch 30 --batchSize 6

#python train.py --no_flip --no_rotation --no_augment --img_file "/media/lab225-2/disk1/WBY/datasets/celeba-256" --lr 1e-5 --continue_train --mask_type


#python train.py --no_flip --no_rotation --no_augment --img_file "/media/sda/wby/dataset/celebahq/test" --mask_file "/media/sda/wby/dataset/gen_mask/mask5_256" --epoch 20 --lr 1e-5 --gpu_ids 2 --batchSize 6 --mask_type 3 --continue_train