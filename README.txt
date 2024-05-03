Team Members & Summary:
    Alex: 
        - Implemented embedding mapper
        - Implemented training loop
        - Diagrams
        
    Marcus:
        - Implemented hashing mapper
        - Worked on implementation of hierarchical which was close to completion but
        was not ready for project

    Together:
        - Report
        - Linear/Non-linear Models

Video Link:
    https://youtu.be/w30ypkiWtRA


Description of Code / Optimizer Design:
    BaselineEmbed.py and BaselineHash.py: Act as a mappers, where hashing gets the hash digest and then resorts the data
    and for embedding new custom embeddings are trained on keys and position    

    LinearModel: Simple pytorch Linear Model
    NonLinearModel: More complex set of layers, using activation functions

    utils.py: Code not used but was for last mile search
    HierarchicalIndexModel: Code not used but was for HierarchicalIndexModel
    
    IndexDataset: Create random data
    training: training/evaluation loops


After discussions with the Professor, we have altered our goals to fit more in line with
our new objective


Goals
Basic Goals:
Design and code a learned index
Perfectly realized (talked with Professor Yang to not build the index on top of the DDB system)
Train a machine learning model on a dataset of key-value pairs to capture the underlying patterns and distribution reasonably correctly
Perfectly realized
Bonus Goals:
Evaluate the accuracy, speed, and storage of the learned index for integers and floats
Realized but imperfect
Include strings as a supported indexed data type
Perfectly realized
Perform model compression via quantization and weight pruning to further reduce memory requirement and inference time
Not attempted
