from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.checkpoints_path = ""
    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data/got10k_lmdb'
    settings.got10k_path = '/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data/itb'
    settings.lasot_extension_subset_path = './SOT/LaSOT-ext'
    settings.lasot_lmdb_path = '/data/lasot_lmdb'
    settings.lasotlang_path = '/mnt/Data/Lasot/images'


    settings.network_path = settings.checkpoints_path    # Where tracking networks are stored.
    settings.nfs_path = '/data/nfs'
    settings.otb_lang_path = '/data/otb_lang'
    settings.otb_path = '/data/otb'
    settings.prj_dir = '/home/ssa/code/DFSTrack'

    settings.result_plot_path = './results/test/result_plots'
    settings.results_path = './results/test'    # Where to store tracking results
    settings.save_dir = './results/test'
    settings.segmentation_path = './results/test/segmentation_results'
    settings.tc128_path = '/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/mnt/Data/TNL2K/TNL2K_test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data/trackingnet'


    return settings

