from pathlib import Path
import subprocess

repo = Path('/home/yinhanke/code/MNN')
build = Path.cwd() / 'build-cuda-baseline'
includes = [repo / d for d in ['include', 'source', 'schema/current', '3rd_party/flatbuffers/include']]
cmd = ['g++', '-O2', '-std=c++17', '-DMNN_SUPPORT_TRANSFORMER_FUSE']
cmd += ['-I' + str(d) for d in includes] + ['-I/usr/local/cuda/include', 'review_attention_backend.cpp']
cmd += ['-L' + str(build), '-L' + str(build / 'source/backend/cuda'),
        '-Wl,-rpath,' + str(build) + ':' + str(build / 'source/backend/cuda'),
        '-lMNN_Cuda_Main', '-lMNN', '-o', 'review_attention_backend']
subprocess.run(cmd, check=True)
for name, args in [
    ('review-attention-backend.log', ['./review_attention_backend']),
    ('review-attention-backend-memcheck.log', ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '99', './review_attention_backend']),
]:
    with Path(name).open('w') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
