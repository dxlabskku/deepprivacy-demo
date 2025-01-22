# DeepPrivacy Demo

This repository includes the code for the demo of the advanced program for further research on the paper 'Customizing facial privacy on SNS: A framework for protecting users from
strangers and stalkers' on CHI 2025 Late-Breaking Work.


### 1. File Structure

```sh
.
├── backend
│   ├── arcface_model
│   ├── checkpoints
│   ├── cog.yaml
│   ├── crop_224
│   ├── data
│   ├── demo_file
│   ├── docs
│   ├── download-weights.sh
│   ├── emojis
│   ├── insightface_func
│   ├── LICENSE
│   ├── main.py
│   ├── models
│   ├── options
│   ├── output
│   ├── parsing_model
│   ├── people
│   ├── pg_modules
│   ├── simswaplogo
│   └── util
├── React_instagram_clone
│   ├── base-tsconfig.json
│   ├── build
│   ├── node_modules
│   ├── package.json
│   ├── package-lock.json
│   ├── public
│   ├── README.md
│   ├── src
│   └── tsconfig.json
├── README.md
└── requirements.txt
```

The frontend directory contains the code for React-based web application, and the backend directory contains the code for Flask-based server-side application. If you want to make a fix, please do not forget to ```npm run build``` in the frontend directory before executing the ```python main.py``` in the backend directory.

### 2. Prerequisites
The demo requires the following software and libraries:
```sh
pip install -r requirements.txt
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
http://localhost:8888/
```
