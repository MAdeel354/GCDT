# Generative Cross-Domain Translator (GCDT) 
Longitudinal brain connectivity prediction in network neuroscience is a crucial 
yet important task as it helps in the early detection of brain disorders like Alzheimer’s. 
Previous studies aim to leverage the power of graph neural networks to perform this task where 
brain structure is represented in an interconnected graphical structure, i.e., nodes and edges. 
However, longitudinal data acquisition is time-consuming and expensive, leading to an increased 
dropout rate of participants. Moreover, in the context of clinical data, all subjects go through 
a singl scan, resulting in multiple unpaired time-points. Thus, to address these challenges, we 
propose a Generative Cross-Domain Translator (GCDT) network that performs bidirectional longitudinal 
brain graph translation between two unpaired time-points, i.e., t1 and t2. Our proposed method consists 
of unpaired connectivity at two different time-points, and the goal is to predict the change in pattern 
from one time-point to another through a graph-based cycle-GAN architecture. This paper is the first
contribution to predicting unpaired longitudinal brain connectivity. The experiment has shown promising 
results in designing a framework for brain connectivity predictions.

![proposed architecture](./Figures/architecture.PNG)

