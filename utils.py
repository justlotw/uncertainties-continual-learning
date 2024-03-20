import pickle 

def save_file(object, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(object, f)

def load_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)