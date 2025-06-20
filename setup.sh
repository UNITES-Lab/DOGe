 conda create -y -n doge python=3.10
 conda activate doge
 pip install -r requirements.txt

mkdir -p data
#huggingface-cli download --repo-type dataset ANONYMOUS/doge-exps --local-dir data/doge-exps --token $HF_TOKEN
