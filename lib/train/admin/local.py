class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '.'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = './tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = './resource'

        # rgb
        self.got10k_dir = './GOT-10k/data/train'
        self.got10k_val_dir = './GOT-10k/data/val'
        self.got10k_lmdb_dir = './GOT-10k/data/train'

        self.lasot_dir = '/mnt/Data/Lasot/images'
        self.lasot_lmdb_dir = './LaSOT/data'

        self.trackingnet_dir = './TrackingNet'
        self.trackingnet_lmdb_dir = './TrackingNet'

        self.coco_lmdb_dir = './COCO2017'
        self.coco_dir = './COCO2017'
        self.ref_coco_dir = './video_ds/refcoco'
        self.otb99_dir = './video_ds/OTB/OTB_sentences/OTB_sentences'

        self.tnl2k_dir = './TNL2k/TNL2K_train_subset'
        self.vasttrack_dir = './VastTrack/unisot_train_final_backup'  # for 28


        # rgbt
        self.lasher_dir = './video_ds/rgb_t/LasHeR0428/LasHeR_Divided_data/trainingset'

        # rgbe
        self.visevent_dir = './video_ds/rgbe_ds/VisEvent_dataset/train_subset'
        # rgbt
        self.depthtrack_dir = './video_ds/rgbd_ds/depthtrack_train/'
