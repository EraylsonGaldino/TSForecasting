import loadfiles as lf
import preprocessing as pp

path = 'dataset_example.csv'

"Load and split as Data Frame"
df = lf.load_data_to_dataframe(path)
df_with_sliding_windows = pp.create_windows_to_dataframe(df, 6, 3)
df_train_slinding, df_validation_slinding, df_test_slinding = pp.split_series(df_with_sliding_windows, 0.6, 0.2)
print(df_train_slinding, df_validation_slinding, df_test_slinding)

"Load and split as Data Frame"
df = lf.load_data_to_dataframe(path)
df_train, df_validation, df_test = pp.split_series(df, 0.6, 0.2)
df_train_slinding = pp.create_windows_to_dataframe(df_train, 6, 3)
df_validation_slinding = pp.create_windows_to_dataframe(df_validation, 6, 3)
df_test_slinding = pp.create_windows_to_dataframe(df_test_slinding, 6, 3)
print(df_train_slinding, df_validation_slinding, df_test_slinding)

"Load and split as Numpy"
np_data_df = lf.load_data_to_numpy(path)
np_data_df_with_sliding_windows = pp.create_windows_to_numpy(np_data_df, 6, 3)
print(pp.split_series(np_data_df_with_sliding_windows, 0.6, 0.2))

"Load and split as Numpy from genfromtxt"
np_data = lf.load_data_to_numpy(path)
np_data_with_sliding_windows = pp.create_windows_to_numpy(np_data, 6, 3)
print(pp.split_series(np_data_with_sliding_windows, 0.6, 0.2))
