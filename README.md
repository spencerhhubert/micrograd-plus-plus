# micrograd++
<p>andrej karpathy's <a href"https://github.com/karpathy/micrograd">micrograd</a> with a few new functions. I added almost nothing, this isn't my work</p>
<ul>
<li>save and load network parameters with <code>MLP.saveParams()</code> and <code>MLP.loadParams()</code></li>
<li>optmize with <code>SGD()</code></li>
<li>I put the code to construct the graph in <code>micrograd/graph.py</code>. You can call <code>dot = draw_dot(y)</code> where <code>y</code> is the output of your network and then, in your jupyter notebook, call <code>dot</code> to get a computational graph of your network</li>
</ul>
<p>install with <code>python install.py develop</code> and uninstall with <code>python -m pip uninstall micrograd-plus-plus</code></p>
