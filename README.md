# building_lang_model_to_aid_word_prob_in_math

cd %CD%

python -m venv build_word_prob_in_math
build_word_prob_in_math\Scripts\activate
pip install accelerate datasets transformers bitsandbytes peft einops   
pip install prettytable

CUDA Toolkit 12.3 Update 2 Downloads
    https://developer.nvidia.com/cuda-downloads

pip install torch==2.0.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
pip install bitsandbytes-windows
pip install git+https://github.com/Keith-Hon/bitsandbytes-windows
pip install bitsandbytes==0.40.2 --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui



https://medium.com/analytics-vidhya/how-to-fine-tune-llms-without-coding-41cf8d4b5d23