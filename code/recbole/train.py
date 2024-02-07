import os

from recbole.quick_start import run_recbole
from args import parse_args

args = parse_args()
assert os.path.exists(f'configs/{args.model}.yaml'), "There's no config file about model ..."

run_recbole(
    config_file_list=[f'configs/{args.model}.yaml']
)
