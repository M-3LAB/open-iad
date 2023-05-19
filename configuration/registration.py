# add new dataset
dataset_name = {'mvtec2d': ('data_io.mvtec2d', 'mvtec2d', 'MVTec2D'),
                'mvtec2df3d': ('data_io.mvtec2df3d', 'mvtec2df3d', 'MVTec2DF3D'),
                'mvtecloco': ('data_io.mvtecloco', 'mvtecloco', 'MVTecLoco'),
                'mpdd': ('data_io.mpdd', 'mpdd', 'MPDD'),
                'btad': ('data_io.btad', 'btad', 'BTAD'),
                'mtd': ('data_io.mtd', 'mtd', 'MTD'),
                'mvtec3d': ('data_io.mvtec3d', 'mvtec3d', 'MVTec3D'),
                'visa': ('data_io.visa', 'visa', 'VisA'),
                'dagm': ('data_io.dagm', 'dagm', 'DAGM'),
                'coad': ('data_io.coad', 'coad', 'COAD')
                }

# add new model
model_name = {'_patchcore': ('arch._patchcore', '_patchcore', 'PatchCore'),
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
                'simplenet': ('arch.simplenet', 'simplenet', 'SimpleNet') 
                }

# server config
server_data = {'127.0.0.1': '/home/robot/data',
               '172.18.36.108': '/ssd2/m3lab/data/open-ad',
               '172.18.36.107': '/ssd-sata1/wjb/data/open-ad',
            }