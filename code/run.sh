gpu=0
data=cora

dim=128
lr=0.01

num_step=4
num_step_gen=100

for nl in 2 3 4 5 6 7 8
do
    python main.py --gpu $gpu --dataset $data --num_step $num_step --num_step_gen $num_step_gen --lr $lr --num_labels $nl --dim $dim
done

python emb_concat.py --dataset $data --num_step $num_step --num_step_gen $num_step_gen
