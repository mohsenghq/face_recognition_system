import argparse
from insightface.model_zoo import get_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='buffalo_l', help='Name of the insightface model to download')
    args = parser.parse_args()
    get_model(args.model_name)
