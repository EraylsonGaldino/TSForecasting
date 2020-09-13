import loadfiles as lf
import preprocessing as pp

path = 'https://raw.githubusercontent.com/EraylsonGaldino/dataset_time_series/master/airline.txt'
window_size = 6
h_step = 2

training_perc = 0.6
val_perc = 0.2

"Load and split as Data Frame"
df = lf.load_data_to_dataframe(path)
df_with_sliding_windows = pp.create_windows_to_dataframe(df, window_size, h_step)
print(df_with_sliding_windows.shape)


df_train_slinding, df_validation_slinding, df_test_slinding = pp.split_series(df_with_sliding_windows, training_perc, val_perc)
print(df_train_slinding, df_validation_slinding, df_test_slinding)

"Load and split as Data Frame"
df = lf.load_data_to_dataframe(path)
df_train, df_validation, df_test = pp.split_series(df,training_perc , val_perc)
df_train_slinding = pp.create_windows_to_dataframe(df_train, window_size, h_step)
df_validation_slinding = pp.create_windows_to_dataframe(df_validation, window_size, h_step)
df_test_slinding = pp.create_windows_to_dataframe(df_test_slinding, window_size, h_step)
print(df_train_slinding.shape, df_validation_slinding.shape, df_test_slinding.shape)

"Load and split as Numpy"
np_data_df = lf.load_data_to_numpy(path)
np_data_df_with_sliding_windows = pp.create_windows_to_numpy(np_data_df, window_size, h_step)
print(pp.split_series(np_data_df_with_sliding_windows, training_perc, val_perc))
