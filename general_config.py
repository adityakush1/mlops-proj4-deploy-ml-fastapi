import os
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

print(current_dir)
print(parent_dir)
COMMON_LOG_PATH = os.path.join(current_dir,'logs')
DATA_PATH = os.path.join(current_dir,'data')
LOG_LEVEL = logging.DEBUG
IMAGE_PATH = os.path.join(current_dir,'images')
MODEL_PATH = os.path.join(current_dir,'model')
TEST_PATH = os.path.join(current_dir,'tests')