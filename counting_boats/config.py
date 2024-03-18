import os
import yaml

with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    cfg['tif_dir'] = os.path.join(cfg['proj_root'], 'images', 'RawImages')


