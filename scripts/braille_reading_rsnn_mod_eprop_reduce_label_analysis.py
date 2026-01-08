loaded = np.load(results_file, allow_pickle=True)
acc_train_dict = loaded['acc_train'].item()  # Dictionary with keys = nb_hidden
acc_test_dict = loaded['acc_test'].item()
loss_train_dict = loaded['loss_train'].item()
nb_hidden_list = list(loaded['nb_hidden_list'])  # List of completed nb_hidden values