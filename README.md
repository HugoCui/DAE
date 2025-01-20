# DAE

Code for the paper : <i>High-dimensional Asymptotics of Denoising Autoencoders</i> (<a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/2722a0ccf6acfe3d144fdbb0dedd80b5-Abstract-Conference.html">link to paper</a>)

<p align="center"><img src="figures/scaling_laws.jpg" alt="illus" width="600"/></center></p>

## SVM classification
(Figs. 1 & 3)
- <tt>maxmargin_simus.ipynb</tt> contains the code used to run max-margin classification experiments on Gaussian synthetic data, satisfying the source and capacity conditions (8).
- <tt>hinge_repl.py</tt> implements the theoretical characterization (13) for the misclassification error $\epsilon_g$.

To run obtain the theoretical characterization for e.g. source $r=0.3$, capacity $\alpha=1.2$, noise level $\sigma=0.5$, using a large features dimension cut-off $p=10000$
```
python3 hinge_repl.py --a 1.2 --r 0.3 --s 0.5 --p 10000
```

## Ridge classification 
(Figs. 2 & 3)
- <tt>CV_l2_simus.ipynb</tt> contains the code used to run the ridge classification experiments on Gaussian synthetic data, satisfying the source and capacity conditions (8).
- <tt>hinge_repl.py</tt> implements the theoretical characterization (17) for the misclassification error $\epsilon_g$.

To run obtain the theoretical characterization for e.g. source $r=0.3$, capacity $\alpha=1.2$, noise level $\sigma=0.5$, using a large features dimension cut-off $p=10000$
```
python3 CV_l2_repl.py --a 1.2 --r 0.3 --s 0.5 --p 10000
```

## Real data experiments
(Figs. 3)<br>
The code implementing ridge (resp. SVM) classification on real data is provided in <tt>Real_l2.ipynb</tt> (resp. <tt>Real_mm.ipynb</tt>), from data stored in a --datasets/ folder.


<u> Versions:</u> These notebooks employ <tt>Python 3.12 </tt>, and <tt>Pytorch 2.5</tt>. The theory code utilizes the <tt>quadpy</tt> package for multidimensional numerical integration. This scheme is not adaptative, and convergence issues may appear e.g. for large cluster variance $\sigma \gtrsim 1$ for binary mixtures. 
