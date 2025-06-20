 conda create -y -n antidistill python=3.10
 conda activate antidistill
 pip install -r requirements.txt

mkdir -p data
#huggingface-cli download --repo-type dataset ANONYMOUS/antidistill-exps --local-dir data/antidistill-exps --token $HF_TOKEN
