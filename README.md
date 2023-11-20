# Embedding Uncertain Knowledge Graphs with Tensorflow 2.0

This repository includes the code of UKGE and data used in the experiments. The original model can only run with Tensorflow 1.0, we made some minor changes so it can be run with Tensorflow 2.0

## Install Tools (Optional) 

- For managing the Python, PyTorch, and other libraries environment, please install Anaconda: https://docs.anaconda.com/free/anaconda/install/windows

- For working with Pytorch code, please install Visual Studio Code: https://code.visualstudio.com/Download

- For downloading the code from Github, please install Source Tree: https://www.sourcetreeapp.com

## Install Python and Libraries

Make sure your local environment has Python and the following libraries (TensorFlow, scikit-learn, pandas) installed by running the commands `pip install -r requirements.txt`:

## Run

To run the experiments, use:
```
python ./run/run.py --data ppi5k --model rect --batch_size 1024 --dim 128 --epoch 100 --reg_scale 5e-4
```

You can use `--model logi` to switch to the UKGE(logi) model.

![image](https://github.com/stasl0217/UKGE/assets/16631121/ca8ab4ca-5c95-4f80-bebf-327ab97ffa84)

## Dataset

Data is available at: https://drive.google.com/file/d/1UJQ8hnqPGv1O9pYglfNF5lY_sgDQkleS/view?usp=sharing

## Reference
Please refer to the original paper: Xuelu Chen, Muhao Chen, Weijia Shi, Yizhou Sun, Carlo Zaniolo. Embedding Uncertain Knowledge Graphs. In *Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence (AAAI)*, 2019
```
@inproceedings{chen2019ucgraph,
    title={Embedding Uncertain Knowledge Graphs},
    author={Chen, Xuelu and Chen, Muhao and Shi, Weijia and Sun, Yizhou and Zaniolo, Carlo},
    booktitle={Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence (AAAI)},
    year={2019}
}
```
