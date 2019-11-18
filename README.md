# Optimizer-Experiments-Pytorch
SGD/ADAM/Amsgrad/AdamW/RAdam/Lookahead

### Requirements
* pytorch1.0+
* python3.6+

### File Structure
```
- data
- models
- optimizers
- logs           
- checkpoints    
- runs           
main.py
utils.py
```
logs, checkpoints, runs directories will be created automatically to store the records.

### How to Use
THis project contains various optimizers and models for experiments. You can just use what you want in your code.
#### Train
```
python main.py
```
when you need to continue training from a checkpoint, specify the checkpoint path in your code and 
```
python main.py --resume
```




refe to https://github.com/kuangliu/pytorch-cifar
