# need to change paths
gt_folder_path=/home/dell/MOT17/train
val_map_path=/home/dell/MOT17/17val_seqmap.txt
track_results_path=/home/dell/SparseTrack20/SparseTrack/yolox_mix17_ablation/yolox_mix17_ablation_det/track_results

# need to change 'gt_val_half.txt' or 'gt.txt'
val_type='{gt_folder}/{seq}/gt/gt_val_half.txt'

# command
python /home/dell/SparseTrack20/SparseTrack/TrackEval/scripts/run_mot_challenge.py  \
        --SPLIT_TO_EVAL train  --METRICS HOTA  --GT_FOLDER ${gt_folder_path}   \
        --SEQMAP_FILE ${val_map_path}  --SKIP_SPLIT_FOL True   --TRACKERS_TO_EVAL '' \
        --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True   --NUM_PARALLEL_CORES 8   --PLOT_CURVES False   \
        --TRACKERS_FOLDER  ${track_results_path}  \
        --GT_LOC_FORMA ${val_type}