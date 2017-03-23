import subprocess



# command = "time python -u lstm.py --n_iter 20 --learning_rate 0.01 --batch_size 1000"
# subprocess.call(command, shell=True)
#
# command = "time python -u lstm.py --n_iter 20 --learning_rate 0.001 --batch_size 2000"
# subprocess.call(command, shell=True)
#
# command = "time python -u lstm.py --n_iter 20 --learning_rate 0.001 --batch_size 1000"
# subprocess.call(command, shell=True)
#
# command = "time python -u lstm.py --n_iter 20 --learning_rate 0.001 --batch_size 500"
# subprocess.call(command, shell=True)


# command = "time python -u lstm.py --n_iter 25 --learning_rate 0.001 --batch_size 2000 --train_file_path data/data_version_1/new_train.txt --test_file_path data/data_version_1/new_test.txt --embedding_file_path data/data_version_1/wordvec.txt --t1 last"
# subprocess.call(command, shell=True)
# command = "time python -u lstm.py --n_iter 25 --learning_rate 0.001 --batch_size 1000 --train_file_path data/data_version_1/new_train.txt --test_file_path data/data_version_1/new_test.txt --embedding_file_path data/data_version_1/wordvec.txt --t1 last"
# subprocess.call(command, shell=True)
# command = "time python -u lstm.py --n_iter 25 --learning_rate 0.001 --batch_size 500 --train_file_path data/data_version_1/new_train.txt --test_file_path data/data_version_1/new_test.txt --embedding_file_path data/data_version_1/wordvec.txt --t1 last"
# subprocess.call(command, shell=True)

command = "time python -u lstm.py --n_iter 25 --learning_rate 0.001 --batch_size 2000 --train_file_path data/data_version_1/new_train.txt --test_file_path data/data_version_1/new_test.txt --embedding_file_path data/data_version_1/wordvec.txt --t1 all_avg"
subprocess.call(command, shell=True)
command = "time python -u lstm.py --n_iter 25 --learning_rate 0.001 --batch_size 1000 --train_file_path data/data_version_1/new_train.txt --test_file_path data/data_version_1/new_test.txt --embedding_file_path data/data_version_1/wordvec.txt --t1 all_avg"
subprocess.call(command, shell=True)
command = "time python -u lstm.py --n_iter 25 --learning_rate 0.001 --batch_size 500 --train_file_path data/data_version_1/new_train.txt --test_file_path data/data_version_1/new_test.txt --embedding_file_path data/data_version_1/wordvec.txt --t1 all_avg"
subprocess.call(command, shell=True)
