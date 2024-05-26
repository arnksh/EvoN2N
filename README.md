# EvoN2N
Evolutionary Net2Net for architecture optimization

Paper Title: Knowledge Transfer based Evolutionary Deep Neural Network for Intelligent Fault Diagnosis

Abstract: The performance of a deep neural network (DNN) for fault diagnosis is very much dependent on the network architecture. Also, the diagnostic performance is reduced if the model trained on a laboratory case machine is used on a test dataset from an industrial machine running under variable operating conditions. Thus, there are two challenges for the intelligent fault diagnosis of industrial machines: (i) selection of suitable DNN architecture and (ii) domain adaptation for the change in operating conditions. Therefore, we propose an evolutionary Net2Net transformation (EvoN2N) that finds the best suitable DNN architecture for the given dataset. Non-dominated sorting genetic algorithm II has been used to optimize the depth and width of the DNN architecture. Also, we have introduced a hybrid crossover technique for optimization of the depth and width of the deep neural network encoded in a chromosome. We have formulated a knowledge transfer-based fitness evaluation scheme for faster evolution. The proposed framework can obtain the best model for intelligent fault diagnosis without the need for a long-time-taking search process. We have used the Case Western Reserve University dataset, Paderborn University dataset, and gearbox fault detection dataset to demonstrate the effectiveness of the proposed framework for the selection of the best suitable architecture capable of excellent diagnostic performance, classification accuracy almost up to 100%

Paper link:  
https://arxiv.org/abs/2109.13479

Cite as


@misc{sharma2022knowledge,
      title={Knowledge Transfer based Evolutionary Deep Neural Network for Intelligent Fault Diagnosis}, 
      author={Arun K. Sharma and Nishchal K. Verma},
      year={2022},
      eprint={2109.13479},
      archivePrefix={arXiv},
      primaryClass={eess.SP}
}

Links for codes of benchmark methods:


DNN: https://github.com/ArabelaTso/DeepFD
DANN: https://github.com/NaJaeMin92/pytorch-DANN
DTL: https://github.com/Xiaohan-Chen/transfer-learning-fault-diagnosis-pytorch
DAFD: https://github.com/zggg1p/A-Domain-Adaption-Transfer-Learning-Bearing-Fault-Diagnosis-Model-Based-on-Wide-Convolution-Deep-Neu
EvoDCNN: https://github.com/yn-sun/evocnn
psoCNN: https://github.com/feferna/psoCNN
