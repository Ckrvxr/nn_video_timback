pixi run python scripts/generate_dataset.py -i "C:\Users\Ckrvxr\Downloads\The.Long.Season.S01E01.2160p.UHD.BluRay.REMUX.HEVC.LPCM.2.0.mkv" -o ./data/the_long_season --num-slices 100 --num-frames 30 --patch-size 512 --scale 2 --num-variants 1 --encoders av1 h265 h264 --workers 2

pixi run python scripts/generate_dataset.py -i "C:\Users\Ckrvxr\Downloads\F1_TLR-2_IMAX_3840x2160_HEVC_444_10bit_DTS-HD_MA_AC3_51-thedigitaltheater.mkv" -o ./data/f1 --num-slices 10 --num-frames 30 --patch-size 512 --scale 2 --num-variants 1 --encoders av1 h265 h264 --workers 1

