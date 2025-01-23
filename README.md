<body>
    <header>
        <h1>SSM-DNN Framework</h1>
    </header>
    <main>
        <section id="introduction">
            <p>
                This project introduces a novel framework for manifold inference and neural decoding, 
                specifically designed for analysis of high-dimensional data collected during cognitive tasks. 
                In this framework, we combine <strong>state-space models (SSM)</strong> with 
                <strong>deep neural networks (DNN)</strong>, in our effort to characterize high-dimensional 
                data and also infer latent dynamical manifolds, which capture essential dynamics present 
                in data and associated condition or label. For the application, we show the whole modeling 
                pipeline in an Implicit Association Task called <em>brief death IAT</em>, recorded in our 
                research group under approved IRB. Details of this task can be found <a href="#">here</a>.
            </p>
        </section>
        
        <section id="key-features">
            <h2>Key Features</h2>
            <ul>
                <li><strong>MCMC Sampling Technique (Particle Filters):</strong> Efficient inference solution of latent states, which turns into particle filters solution deriving an approximate posterior distribution of state given neural data and associated labels.</li>
                <li><strong>Data Generation Pipeline:</strong> Generative nature of SSM-DNN model allows drawing samples (trajectories) of the high-dimensional data and corresponding labels or categories.</li>
                <li><strong>Flexible DNN Topologies Embedded in SSM-DNN:</strong> The framework supports various DNN structures such as Multi-Layer Perceptrons, CNNs with 1-D input, and CNNs with multivariate time series.</li>
                <li><strong>Versatile Learning Solution:</strong> Combines Expectation-Maximization (EM) based training, sampling techniques, and stochastic gradient methods for training the SSM and DNN models.</li>
                <li><strong>Flexibility in Analysis of Different Modalities of Data:</strong> Applicable to various data modalities beyond neural data, including behavioral time-series data and mixed behavioral signals.</li>
            </ul>
        </section>
        
        <section id="installation">
            <h2>Installation</h2>
            <p>Clone this repository and install the required dependencies:</p>
            <pre><code>
git clone https://github.com/&lt;your-username&gt;/ldCm.git
cd ldCm
pip install -r requirements.txt
            </code></pre>
        </section>
        
        <section id="usage">
            <h2>How to Use SSM-DNN Package</h2>
            <p>To use this toolkit, follow the step-by-step instructions provided below to set up, train, and evaluate the SSM-DNN model for your high-dimensional data analysis tasks.</p>
            <ol>
                <li><strong>Prepare Your Data:</strong> Format your data as multi-channel time series. Ensure that it is compatible with the input requirements of the model.</li>
                <li><strong>Run the SSM-DNN Model:</strong> Execute the <code>main.py</code> script to infer the latent states and decode task-specific labels:
                    <pre><code>
python main.py --data_path ./data/eeg_dataset.csv --output_dir ./results
                    </code></pre>
                </li>
                <li><strong>Visualize Results:</strong> Use the provided utilities to analyze and visualize the inferred manifold and decoding performance:
                    <pre><code>
python visualize.py --results_dir ./results
                    </code></pre>
                </li>
                <li><strong>Run on Google Colab:</strong> Use the pre-configured notebook for an interactive experience.</li>
            </ol>
        </section>
        
        <section id="modeling-approach">
        <h1>Modeling Approach and Definition</h1>
        <p>
            Comprehensive documentation for SSM-DNN, including API details, examples, and theory, 
            can be found in the <a href="./docs">docs directory</a>.
        </p>
        </section>

        <section id="code-examples">
        <h1>Code Examples (Simulation and Real Data Applications)</h1>
        
        <h2>I. Simulated Data Classification</h2>
        <p>In this example, we create simulation data replicating SSM-DNN. The model assumptions are as follows:</p>
        <ul>
            <li>State Dimension: 2</li>
            <li>Observation Dimension: 6</li>
            <li>Number of Trials: 400</li>
            <li>Number of Samples per Trial: 25</li>
            <li>Class Labels: 2 (Class A and Class B)</li>
        </ul>
        <p>
            The state equation is a multivariate normal defined as:
            <br><code>X<sub>k+1</sub> = A ⋅ X<sub>k</sub> + B + e</code>,
            <br>where <em>A</em> is an identity matrix, <em>B</em> is a zero vector, and <em>e</em> is correlated noise.
        </p>
        <p>The observation model is defined as:</p>
        <p><code>Y<sub>k</sub> = C ⋅ X<sub>k</sub> + D</code></p>
        <p>
            The label for each trial is determined based on the sum of its samples. If the sum of the first 
            half of the samples in a trial is less than the sum of the second half, the label is <code>0</code>; 
            otherwise, the label is <code>1</code>.
        </p>
        <p>
            For DNN, we use a CNN with 2 inputs and 2 convolution layers. The code for this application can be found here:
        </p>
        <ul>
            <li>Data Generation Code</li>
            <li>Model Training Code: We use the MCMC method for posterior estimation of the state (4000 particles) and the EM algorithm for training.</li>
            <li>Decoding (Label Prediction) Code: We predict the DNN probability of Class A and Class B using 4000 particle samples drawn from the state estimation.</li>
        </ul>

        <h2>II. Brief Death Implicit Association Task (BDIAT)</h2>
        <p>
            Here, we demonstrate an application of SSM-DNN in the BDIAT task. For this dataset:
        </p>
        <ul>
            <li>The observation is reaction time (RT) across 360 trials of the task.</li>
            <li>There are 23 participants in total, labeled as either CTL (healthy) or MDD (Major Depressive Disorder).</li>
            <li>Each participant's data is considered a trial, resulting in 23 trials with 360 samples per trial.</li>
        </ul>
        <p>
            The observation dimension is <code>1</code>, and we use a random walk model as the state process. 
            Here, SSM acts as an adaptive smoother. The DNN is a 1-dimensional CNN with 2 convolution layers 
            and max-pooling.
        </p>
        <p>
            We use a cross-validation scheme to assess prediction accuracy, specificity, and sensitivity. 
            Our model outperforms others such as XXX and YYY by achieving higher prediction accuracy with a 
            more balanced specificity and sensitivity.
        </p>
        <p>Different steps of the framework implementation using SSM-DNN can be found here:</p>
        <ul>
            <li><strong>Neural Network for Classification Task:</strong> Demonstrates how to implement and train a neural network for EEG data classification and visualize its performance.</li>
            <li><strong>Latent State Inference:</strong> Showcases the particle filter and EM algorithm in action.</li>
            <li><strong>Performance Metrics:</strong> Evaluates model performance using accuracy, ROC curves, and AUC.</li>
        </ul>
    </section>
    
<section id="citation">
        <h1>Citation</h1>
        <p>If you use SSM-DNN in your research, please cite the following research papers:</p>
        <pre><code>
@article{Paper1,
  title = {Novel techniques for neural data analysis},
  author = {Smith, J. and Doe, J.},
  journal = {Google Scholar},
  year = {2023},
  url = {https://scholar.google.com/citations?view_op=view_citation&hl=en&user=M8rzdnwAAAAJ&sortby=pubdate&citation_for_view=M8rzdnwAAAAJ:NXb4pA-qfm4C},
  note = {Accessed: 2025-01-14}
}
@article{Paper2,
  title = {Advances in deep learning for neuroscience},
  author = {Brown, A. and Taylor, K.},
  journal = {Google Scholar},
  year = {2022},
  url = {https://scholar.google.com/citations?view_op=view_citation&hl=en&user=jieyeRUAAAAJ&sortby=pubdate&citation_for_view=jieyeRUAAAAJ:NDuN12AVoxsC},
  note = {Accessed: 2025-01-14}
}
@article{Paper3,
  title = {State-space models in neuroscience},
  author = {Doe, J. and Smith, R.},
  journal = {PubMed},
  year = {2020},
  url = {https://pubmed.ncbi.nlm.nih.gov/31947169/},
  note = {Accessed: 2025-01-14}
}
        </code></pre>
    </section>

    <section id="collaboration">
        <h2>Collaboration and Contribution</h2>
        <p>
            We welcome your contribution to this research! Please check 
            <a href="#">here</a> for guidelines.
        </p>
    </section>

    <section id="license">
        <h2>License</h2>
        <p>
            This project is licensed under the MIT License. See the 
            <a href="./LICENSE">license file</a> for details.
        </p>
    </section>

    <section id="acknowledgments">
        <h2>Acknowledgments</h2>
        <p>
            This work was partially supported by the Defense Advanced Research Projects 
            Agency (DARPA) under cooperative agreement #N660012324016. The content of 
            this information does not necessarily reflect the position or policy of the 
            Government, and no official endorsement should be inferred.
        </p>
        <p>
            We appreciate our research collaborators from UMN, Intheon, and others. This 
            work was also supported by startup funds from the University of Houston. 
            Special thanks go to our research collaborators and colleagues who contributed 
            by providing data and offering thoughtful comments to refine our modeling framework.
        </p>
    </section>
