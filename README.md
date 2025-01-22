# DeepPrivacy Demo

This repository includes the code for the demo of the advanced program for further research on the paper 'User-centric Perspectives Toward DeepPrivacy Framework: Exploring the Influence of Social Relationships and Facial Substitution Similarity on User Experience' on CHI 2025.


### 1. File Structure

```sh
├── README.md
├── backend
│   ├── LICENSE
│   ├── README.md
│   ├── arcface_model
│   ├── checkpoints
│   ├── crop_224
│   ├── data
│   ├── deepPrivacy_service.py
│   ├── demo_file
│   ├── docs
│   ├── insightface_func
│   ├── main.py
│   ├── models
│   ├── options
│   ├── output
│   ├── parsing_model
│   ├── people
│   ├── pg_modules
│   ├── simswaplogo
│   ├── test.png
│   ├── (miscellaneous files, not used on current demo)
│   ├── util
│   └── workaround.ipynb
└── frontend
    ├── README.md
    ├── build
    ├── node_modules
    ├── package-lock.json
    ├── package.json
    ├── public
    └── src
```

The frontend directory contains the code for React-based web application, and the backend directory contains the code for Flask-based server-side application. If you want to make a fix, please do not forget to ```npm run build``` in the frontend directory before executing the ```python main.py``` in the backend directory.

### 2. Prerequisites
The demo requires the following software and libraries:
```sh
numpy==1.23.5
pillow==10.4.0
requests==2.32.3
scikit-learn==1.3.2
sympy==1.13.2
flask==3.0.3
torch==1.12.0
torchvision==0.14.1
matplotlib==3.7.5
networkx==3.1
```

Also, you need to download several pre-trained models to run the demo. You can download such weights following the instructions in the [SimSwap](https://github.com/neuralchen/SimSwap) repository.

### 3. How to start

To start this application, you have to execute follow commands.

```sh
cd backends
python main.py
```

Then, you can access to the demo page from your browser.

```sh
http://localhost:5001/
```
