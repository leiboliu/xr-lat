#================= inputs =====================
model_name=bigbird_mimic3
data_dir=data

X_trn=${data_dir}/train.txt # training text
X_tst=${data_dir}/test.txt # test text
X_dev=${data_dir}/dev.txt # dev text

Y_trn=${data_dir}/train_label.npz # training label matrix
Y_tst=${data_dir}/test_label.npz # test label matrix
Y_dev=${data_dir}/dev_label.npz

X_feat_trn=${data_dir}/train.tfidf.npz # training tfidf feature
X_feat_tst=${data_dir}/test.tfidf.npz # test tfidf feature
X_feat_dev=${data_dir}/dev.tfidf.npz

model_dir=models/${model_name}
mkdir -p ${model_dir}

#params_dir=params/${data_name}/${model_name}

python3 -m pecos.xmc.xtransformer.train \
                                --trn-text-path ${X_trn} \
                                --trn-feat-path ${X_feat_trn} \
                                --trn-label-path ${Y_trn} \
				--tst-text-path ${X_dev} \
				--tst-feat-path ${X_feat_dev} \
				--tst-label-path ${Y_dev} \
                                --model-dir ${model_dir} \
				--saved-trn-pt ${data_dir}/X_trn.pt \
				--max-leaf-size 16 \
				--min-codes 16 \
				--model-shortcut ./pretrained/bigbird/ \
				--negative-sampling tfn+man \
				--truncate-length 4096 \
				--batch-size 4 \
				--gradient-accumulation-steps 8 \
				--seed 2022 \
				--max-steps 10000 \
				--warmup-steps 1000 \
				--logging-steps 500 \
				--save-steps 500 \
                                |& tee ${model_dir}/train.log

python3 -m pecos.xmc.xtransformer.predict \
                                -t ${X_tst} \
                                -x ${X_feat_tst} \
                                -m ${model_dir} \
				-o ${data_dir}/${model_name}.npz \

python3 -m pecos.xmc.xlinear.evaluate -y ${Y_tst} \
				-p ${data_dir}/${model_name}.npz \
				-k 15
