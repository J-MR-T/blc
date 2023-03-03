import lit.formats
import os

config.name = "B-like compiler"
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.b']

config.excludes = ['bSamples']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root, 'tests')

config.substitutions.append(("%blc", os.path.join(config.test_source_root, "blc")))
