# setting
setting_name = ['vanilla', 'fewshot', 'semi', 'noisy', 'continual', 'transfer']

# add new dataset
dataset_name = {'_example': ('dataset._example', '_example', 'Example'),
                'mvtec2d': ('dataset.mvtec2d', 'mvtec2d', 'MVTec2D'),
                'mvtec2df3d': ('dataset.mvtec2df3d', 'mvtec2df3d', 'MVTec2DF3D'),
                'mvtecloco': ('dataset.mvtecloco', 'mvtecloco', 'MVTecLoco'),
                'mpdd': ('dataset.mpdd', 'mpdd', 'MPDD'),
                'btad': ('dataset.btad', 'btad', 'BTAD'),
                'mtd': ('dataset.mtd', 'mtd', 'MTD'),
                'mvtec3d': ('dataset.mvtec3d', 'mvtec3d', 'MVTec3D'),
                'visa': ('dataset.visa', 'visa', 'VisA'),
                'dagm': ('dataset.dagm', 'dagm', 'DAGM'),
                'coad': ('dataset.coad', 'coad', 'COAD'),
                }

# add new model
model_name = {'_example': ('arch._example', '_example', 'Example'),
              '_patchcore': ('arch._patchcore', '_patchcore', 'PatchCore'),
              'patchcore': ('arch.patchcore', 'patchcore', 'PatchCore'),
              'padim': ('arch.padim', 'padim', 'PaDim'),
              'csflow': ('arch.csflow', 'csflow', 'CSFlow'),
              'dne': ('arch.dne', 'dne', 'DNE'),
              'draem': ('arch.draem', 'draem', 'DRAEM'),
              'igd': ('arch.igd', 'igd', 'IGD'),
              'dra': ('arch.dra', 'dra', 'DRA'),
              'devnet': ('arch.devnet', 'devnet', 'DevNet'),
              'favae': ('arch.favae', 'favae', 'FAVAE'),
              'fastflow': ('arch.fastflow', 'fastflow', 'FastFlow'),
              'cfa': ('arch.cfa', 'cfa', 'CFA'),
              'reverse': ('arch.reverse', 'reverse', 'REVERSE'),
              'spade': ('arch.spade', 'spade', 'SPADE'),
              'stpm': ('arch.stpm', 'stpm', 'STPM'),
              'cutpaste': ('arch.cutpaste', 'cutpaste', 'CutPaste'),
              'graphcore': ('arch.graphcore', 'graphcore', 'GraphCore'),
              'simplenet': ('arch.simplenet', 'simplenet', 'SimpleNet'), 
              'softpatch': ('arch.softpatch', 'softpatch', 'SoftPatch'),
              }

# server config, ip: dataset root path
server_data = {'127.0.0.1': '/home/robot/data',
               '172.18.36.108': '/ssd2/m3lab/data/open-ad',
               '172.18.36.107': '/ssd-sata1/wjb/data/open-ad',
              }