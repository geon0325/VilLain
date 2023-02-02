data=cora
dim=128
lr=0.01
num_step=4
num_step_gen=10
pca=0.99

python main.py --dataset $data --dim $dim --lr $lr --num_step $num_step --num_step_gen $num_step_gen --pca $pca


