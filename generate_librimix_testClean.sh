#!/bin/bash
set -eu  # Exit on error

storage_dir=$1
librispeech_dir=$storage_dir/LibriSpeech
wham_dir=$storage_dir/wham_noise
librimix_outdir=$storage_dir/

function LibriSpeech_test_clean() {
    if ! test -e $librispeech_dir/test-clean; then
        echo "Download LibriSpeech/test-clean into $storage_dir"
        # If downloading stalls for more than 20s, relaunch from previous state.
        wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/test-clean.tar.gz -P $storage_dir
        tar -xzf $storage_dir/test-clean.tar.gz -C $storage_dir
        rm -rf $storage_dir/test-clean.tar.gz
    fi
}

function wham() {
    if ! test -e $wham_dir; then
        echo "Download wham_noise into $storage_dir"
        # If downloading stalls for more than 20s, relaunch from previous state.
        wget -c --tries=0 --read-timeout=20 https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip -P $storage_dir
        unzip -qn $storage_dir/wham_noise.zip -d $storage_dir
        rm -rf $storage_dir/wham_noise.zip
    fi
}

LibriSpeech_test_clean &
wham &

wait

# Path to python
python_path=python

# If you wish to rerun this script in the future please comment this line out.
$python_path scripts/augment_train_noise.py --wham_dir $wham_dir

metadata_dir=metadata/Libri2Mix_test
$python_path scripts/create_librimix_from_metadata.py --librispeech_dir $librispeech_dir/test-clean \
    --wham_dir $wham_dir \
    --metadata_dir $metadata_dir \
    --librimix_outdir $librimix_outdir \
    --n_src 2 \
    --freqs 16k \
    --modes max \
    --types mix_clean

