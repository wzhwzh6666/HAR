# HAR
## Making SZZ on Just-in-Time Defect Prediction Robust via Topology Adaptive Graph Convolutional Networks

### Dataset: 
we use a comprehensive dataset constructed from three reliable datasets, which include bug-fixing and bug-inducing submissions from different projects.
#### Dataset 1
Dataset 1 was collected by Wen et al. This dataset includes information on bug-inducing submissions from bug reports that have been manually verified, comprising a total of 241 bug-fixing submissions and 351 bug-inducing submissions.

#### Dataset 2
Dataset 2 was constructed by Song et al. They identified bug-inducing and bug-fixing submissions using tests from the code repository, comprising 957 bug-fixing submissions and 957 bug-inducing submissions. Specifically, a submission is marked as bug-inducing if the test fails for that submission but passes for a previous one. Conversely, a submission is recognized as a bug-fixing submission if it passes the same tests that failed for the bug-inducing submission. This approach uses tests as a reliable indicator for identifying bug-inducing and bug-fixing submissions.

#### Dataset 3
Dataset 3 was collected by Neto et al. They used detailed information from the Defects4J dataset, including version control system change logs and patches that reintroduce errors with precise changes. After carefully analyzing the information, they isolated genuine bug-fixing modifications from those that were not intended to fix the bug, identifying the changes that caused the bug-inducing submissions. This resulted in 291 bug-fixing submissions and 378 bug-inducing submissions.

### Python environment dependencies:
Package                Version

Package Version | Package Version  
 ---- | ----- |  
absl-py 2.1.0            | anyio 3.7.1          
argon2-cffi 23.1.0       | argon2-cffi-bindings 21.2.0
attrs 24.2.0            | backcall 0.2.0       
beautifulsoup4 4.12.3    | bleach 6.0.0         
cachetools 5.5.0         | certifi 2022.12.7    
cffi 1.15.1             | charset-normalizer 3.4.0
chumpy 0.70             | comm 0.1.4          
cvbase 0.5.5            | cycler 0.11.0        
debugpy 1.7.0           | decorator 5.1.1      
defusedxml 0.7.1         | easydict 1.13        
einops 0.6.1            | entrypoints 0.4      
exceptiongroup 1.2.2     | fastjsonschema 2.20.0 
filelock 3.12.2          | freetype-py 2.5.1    
fsspec 2023.1.0          | google-auth 2.35.0   
google-auth-oauthlib 0.4.6 | grpcio 1.62.3         
h5py 3.8.0              | huggingface-hub 0.16.4
idna 3.10               | imageio 2.31.2       
imageio-ffmpeg 0.5.1     | importlib-metadata 6.7.0
importlib-resources 5.12.0 | ipdb 0.13.13         
ipykernel 6.16.2         | ipython 7.34.0       
ipython-genutils 0.2.0   | ipywidgets 8.1.5     
jedi 0.19.1             | Jinja2 3.1.4         
joblib 1.3.2            | jsonschema 4.17.3    
jupyter_client 7.4.9     | jupyter_core 4.12.0  
jupyter-server 1.24.0    | jupyterlab-pygments 0.2.2
jupyterlab_widgets 3.0.13 | kiwisolver 1.4.5    
loguru 0.7.2            | Markdown 3.4.4       
MarkupSafe 2.1.5         | matplotlib 3.1.1     
matplotlib-inline 0.1.6  | mistune 3.0.2        
msgpack 1.0.5           | msgpack-numpy 0.4.8  
multimethod 1.9.1        | nbclassic 1.1.0      
nbclient 0.7.4          | nbconvert 7.6.0      
nbformat 5.8.0          | nest-asyncio 1.6.0   
networkx 2.6.3          | notebook 6.5.7       
notebook_shim 0.2.4      | numpy 1.21.6         
oauthlib 3.2.2          | open3d-python 0.7.0.0
opencv-python 4.10.0.84  | packaging 24.0       
pandas 1.3.5            | pandocfilters 1.5.1  
parso 0.8.4             | pexpect 4.9.0       
pickleshare 0.7.5        | Pillow 9.5.0         
pip 22.3.1              | pkgutil_resolve_name 1.3.10
prettytable 3.7.0       | prometheus-client 0.17.1
prompt_toolkit 3.0.48    | protobuf 3.20.3      
psutil 6.1.0            | ptyprocess 0.7.0     
pyasn1 0.5.1            | pyasn1-modules 0.3.0 
pycparser 2.21          | pyglet 2.0.10        
Pygments 2.17.2         | PyOpenGL 3.1.0       
pyparsing 3.1.4          | pyrender 0.1.45      
pyrsistent 0.19.3        | python-dateutil 2.9.0.post0
python-metric-learning 2.6.0 | pytz 2024.1        
PyYAML 6.0.1            | pyzmq 26.2.0         
regex 2024.4.16          | requests 2.31.0      
requests-oauthlib 2.0.0  | roma 1.5.0           
rsa 4.9                 | scikit-learn 1.0.2   
scipy 1.7.3             | Sen2Trash 1.8.3      
sentencepiece 0.2.0      | setuptools 65.6.3    
shapely 2.0.6           | six 1.16.0           
smplx 0.1.28            | sniffio 1.3.1        
soupsieve 2.4.1         | tabulate 0.9.0       
tensorboard 2.11.2       | tensorboard-data-server 0.6.1
tensorboard-plugin-wit 1.8.1 | tensorboardX 2.6.2.2   
tensorpack 0.11          | termcolor 2.3.0       
terminado 0.17.1         | terminaltables 3.1.10 
threadpoolctl 3.1.0      | tinycss2 1.2.1       
tokenizers 0.13.3        | toml 0.10.2         
tomli 2.0.1             | torch 1.11.0+cu113   
torch-geometric 2.1.0    | torch-scatter 2.0.9   
torch-sparse 0.6.13      | torchaudio 0.11.0+cu113
torchpack 0.0.3          | torchtext 0.12.0     
torchvision 0.12.0+cu113 | tornado 6.2          
tqdm 4.66.5             | traitlets 5.9.0       
transformers 4.26.1      | trimesh 4.4.1        
typing_extensions 4.7.1  | urllib3 2.0.7        
wcwidth 0.2.13           | webencodings 0.5.1   
websocket-client 1.6.1   | Werkzeug 2.2.3       
wheel 0.38.4             | widgetsnbextension 4.0.13
zipp 3.15.0

### Running Method
After installing the required environment, run train.py.

`python train.py` 
