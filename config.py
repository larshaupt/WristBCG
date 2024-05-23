from os.path import join

# Change your base directory here
base_dir = "/local/home/lhauptmann/thesis/"

# Define all paths
# Where to find the preprocessed data, refer to data_preprocess for preprocessing
data_dir = join(base_dir, "data")
# Where to find the code
code_dir = join(base_dir, "t-mt-2023-WristBCG-LarsHauptmann" ,"WristBCG")
# Where to save results of analysis
analysis_dir = join(base_dir, "analysis")

# Where to save results of signal processing
classical_results_dir = join(code_dir, "signal_processing")
# Where to save plots, i.e. from tSNE and make_wandb_plots
plot_dir = join(base_dir, "images","thesis")
# where to save models weights and results
results_dir = join(code_dir, "results")

data_dir_Apple = join(data_dir, "AppleDataset")
analysis_dir_Apple = join(analysis_dir, "AppleDataset")

data_dir_Apple_processed = join(data_dir, "AppleDataset_0to6_50Hz")
data_dir_Apple_processed_100hz = join(data_dir, "AppleDataset_22to7_100Hz")
data_dir_Apple_processed_100hz_wmetrics = join(data_dir, "AppleDataset_22to7_100Hz_metric_int")
data_dir_Apple_processed_100hz_5w_4s = join(data_dir, "AppleDataset_22to7_100Hz_5w_4s")
data_dir_Apple_processed_100hz_8w_4s = join(data_dir, "AppleDataset_22to7_100Hz_8w_4s")
data_dir_Apple_processed_100hz_10w_4s = join(data_dir, "AppleDataset_22to7_100Hz_10w_4s")
data_dir_Apple_processed_100hz_30w_4s = join(data_dir, "AppleDataset_22to7_100Hz_30w_4s")
data_dir_Apple_processed_100hz_60w_4s = join(data_dir, "AppleDataset_22to7_100Hz_60w_4s")
data_dir_Apple_processed_100hz_10w_2s = join(data_dir, "AppleDataset_22to7_100Hz_10w_2s")
data_dir_Apple_processed_100hz_10w_8s = join(data_dir, "AppleDataset_22to7_100Hz_10w_8s")
data_dir_Apple_processed_100hz_10w_1s = join(data_dir, "AppleDataset_22to7_100Hz_10w_1s")
data_dir_Apple_processed_100hz_10w_10s = join(data_dir, "AppleDataset_22to7_100Hz_10w_10s")
data_dir_Apple_processed_all = join(data_dir, "AppleDataset_22to7_100Hz_all")

data_dir_M2Sleep = join(data_dir, "USI Sleep/E4_Data")
analysis_dir_M2Sleep = join(analysis_dir, "M2Sleep")

data_dir_M2Sleep_processed = join(data_dir, "M2Sleep_processed_50Hz")
data_dir_M2Sleep_processed_100Hz = join(data_dir, "M2Sleep_processed_100Hz")

data_dir_MaxDataset = join(data_dir, "MaxDataset")
data_dir_MaxDataset_v2 = join(data_dir, "WristBCG")
analysis_dir_MaxDataset = join(analysis_dir, "MaxDataset")

data_dir_Max_processed = join(data_dir, "MaxDataset_0to7_100Hz")
data_dir_Max_processed_v2 = join(data_dir, "MaxDataset_v2")
data_dir_Max_processed_hrv = join(data_dir, "MaxDataset_hrv")

data_dir_Capture24 = join(data_dir, "capture24")

data_dir_Capture24_processed = join(data_dir, "capture24_processed_100Hz_sleep")
data_dir_Capture24_processed_all = join(data_dir, "capture24_processed_100Hz_all")
data_dir_Capture24_processed_125Hz_8w = join(data_dir, "capture24_processed_125Hz_8w_sleep")

data_dir_Parkinson = join(data_dir, "anonymized.h5")
analysis_dir_Parkinson = join(analysis_dir, "Parkinson")

data_dir_Parkinson_processed = join(data_dir, "Parkinson_22to7_100Hz")
data_dir_Parkinson_processed_100Hz_wmetrics = join(data_dir, "Parkinson_22to7_100Hz_wmetrics")
data_dir_Parkinson_processed_100Hz = join(data_dir, "Parkinson_22to7_100Hz")

data_dir_IEEE = join(data_dir, "IEEE")
data_dir_IEEE_processed = join(data_dir, "IEEE_processed")

ResNET_oxwearables_weights_path = join(code_dir, "models/mtl_best.mdl")

