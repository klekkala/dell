# Dell

## Background

This project is aim to create a complete system model that can be trained on multiple Atari games to obtain pre-trained models and RL policys. Afterwards, it can identify and classify tasks (Atari games) that have or have not been seen before and apply known policys or continuously increamental learn new policys to achieve a better result (reward).


---

## Installation and Setup

### Prerequisites

The packages used can be obtained using requirements.txt, and some older version packages may need to be manually adjusted if they are not available.

```bash
pip install -r requirements.txt
```

### Get start

Adjust some game lists and other settings in config.py as needed.

#### Sequential learning
```bash
python main.py --train
```

#### CEC task-mapper pre-train
```bash
python main.py --pre_task_mapper --alpha {num of game type} --beta {num of game} --path {agent path} --encoder {encoder type}
```
example:
```bash
python main.py --pre_task_mapper --alpha 2 --beta 4 --path '/home/student/dell_logs/agent' --set_no_val --encoder vae
```

####  Eval agent process
```bash
python main.py --eval_agent --alpha {num of game type} --beta {num of game} --run {epoch run number} --path {agent path} --encoder {encoder type} --model_dir {pre-trained model path}
```
example:
```bash
python main.py --eval_agent --alpha 2 --beta 4 --run 1 --path '/home/student/dell_logs/agent2' --encoder clip --model_dir 'session0_max_acc.pth'
```
