#source  ~/mambaforge/etc/profile.d/conda.sh
#source  /scratch-ssd/wlin/mambaforge/etc/profile.d/conda.sh
conda activate pytorch2-latest

# we train NN models with bfloat16 (amp is enabled)
# our method: root-free RMSProp (RF-RMSProp)

##########################################
#resnet34

#RF-RMSProp
python main_search.py --dataset=cifar100 --optimizer=rfrmsprop^amp --network=resnet34 --batch_size=128 --epoch=210 --milestone=70,140 \
--damping=5.577261648403792e-05 --learning_rate=0.004978231040913426 --lr_cov=0.001183579682482688 \
--momentum=0.9613533958256908 --weight_decay=0.002324786117084958


#AdamW
python main_search.py --dataset=cifar100 --optimizer=adamw^amp --network=resnet34 --batch_size=128 --epoch=210 \
--milestone=70,140 --damping=5.4258889648815354e-11 --learning_rate=0.00024273457412306925 \
--lr_cov=0.0005136099910299738 --momentum=0.27664922838445527 --weight_decay=0.04467402717054048

#SGD
python main_search.py --dataset=cifar100 --optimizer=sgd^amp --network=resnet34 --batch_size=128 --epoch=210 \
--milestone=70,140 --learning_rate=0.20861514737591105 --momentum=0.3160119642248197 --weight_decay=0.001597026545566997

##########################################

##########################################
#densenet121

#RF-RMSProp
python main_search.py --dataset=cifar100 --optimizer=rfrmsprop^amp --network=densenet121 --batch_size=128 --epoch=210 \
--milestone=70,140 --damping=5.305633825683803e-06 --learning_rate=0.09845335208244133 --lr_cov=0.0005626112964829529 \
--momentum=0.6044068966817299 --weight_decay=0.0019560523126561148


#AdamW
python main_search.py --dataset=cifar100 --optimizer=adamw^amp --network=densenet121 --batch_size=128 --epoch=210 \
--milestone=70,140 --damping=1.1499703216511363e-10 --learning_rate=0.0004114314763858451 --lr_cov=0.0009652871019854889 \
--momentum=0.540757476931907 --weight_decay=0.04331618477802025


#SGD
python main_search.py --dataset=cifar100 --optimizer=sgd^amp --network=densenet121 --batch_size=128 --epoch=210 \
--milestone=70,140 --learning_rate=0.1833384260366266 --momentum=0.12853909694196336 \
--weight_decay=0.0016479442205489954

##########################################

##########################################
#vgg16

#RF-RMSProp
python main_search.py --dataset=cifar100 --optimizer=rfrmsprop^amp --network=vgg16_bn --batch_size=128 --epoch=210 \
--milestone=70,140 --damping=1.851481010567565e-05 --learning_rate=0.017272044650469536 --lr_cov=0.0005792396714807309 \
--momentum=0.5656673653182214 --weight_decay=0.01718894619410476

#AdamW
python main_search.py --dataset=cifar100 --optimizer=adamw^amp --network=vgg16_bn --batch_size=128 --epoch=210 \
--milestone=70,140 --damping=1.8327811315032591e-10 --learning_rate=0.002742935753717082 --lr_cov=0.0053891289208846255 \
--momentum=0.00749693706046926 --weight_decay=0.089792377892941


#SGD
python main_search.py --dataset=cifar100 --optimizer=sgd^amp --network=vgg16_bn --batch_size=128 --epoch=210 --milestone=70,140 \
--learning_rate=0.037374256063377374 --momentum=0.3384663251382076 --weight_decay=0.011260232037778147

##########################################

##########################################
#resnet50 (MSE)

#RF-RMSProp
python main_search.py --dataset=cifar100 --optimizer=rfrmsprop^amp --network=resnet50 --batch_size=128 --epoch=120 \
--milestone=40,80 --loss=MSE --damping=5.127774046983131e-07 --learning_rate=0.025330674752044795 \
--lr_cov=0.00011615353944317813 --momentum=0.008169447479672387 --weight_decay=0.013375074816698857


#AdamW
python main_search.py --dataset=cifar100 --optimizer=adamw^amp --network=resnet50 --batch_size=128 --epoch=120 \
--milestone=40,80 --loss=MSE --damping=1.7616375727721205e-07 --learning_rate=0.0007941250380464385 \
--lr_cov=0.0023231894529067184 --momentum=0.05651164523805084 --weight_decay=0.08533212690148191

#SGD
python main_search.py --dataset=cifar100 --optimizer=sgd^amp --network=resnet50 --batch_size=128 --epoch=120 \
--milestone=40,80 --loss=MSE --learning_rate=0.026292518423440252 --momentum=0.01884664197827682 --weight_decay=0.01503483519928978

##########################################

##########################################
#resnet50 (CE)

#RF-RMSProp
python main_search.py --dataset=cifar100 --optimizer=rfrmsprop^amp --network=resnet50 --batch_size=128 --epoch=120 \
--milestone=40,80 --damping=0.0001953364641974442 --learning_rate=0.006658148773631732 \
--lr_cov=0.0014955219047061044 --momentum=0.8772268947323498 --weight_decay=0.006989184243424921

#AdamW
python main_search.py --dataset=cifar100 --optimizer=adamw^amp --network=resnet50 --batch_size=128 --epoch=120 \
--milestone=40,80 --damping=1.9844866627164052e-10 --learning_rate=0.0006057004451294392 \
--lr_cov=0.00011781133190708046 --momentum=0.350605738904743 --weight_decay=0.08982500882800608

#SGD
python main_search.py --dataset=cifar100 --optimizer=sgd^amp --network=resnet50 --batch_size=128 --epoch=120 \
--milestone=40,80 --learning_rate=0.07363314063288963 --momentum=0.4077024332074053 --weight_decay=0.005295266537207944


