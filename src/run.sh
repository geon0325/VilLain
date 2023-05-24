gpu=0
data=cora

for nl in 2 3 4 5 6 7 8
do
    python main.py --dataset $data --num_labels $nl --gpu $gpu
done

python emb_concat.py --dataset $data --emb_path embs