from tensorflow.python.keras.layers import Input, Embedding


def build_sparse_input(feature_columns, prefix=''):
    input_features = {}
    for fc in feature_columns:
        input_features[fc] = Input(shape=(1,), name=prefix + fc)
    return input_features


def build_var_len_sparse_input(feature_columns, max_length, prefix=''):
    input_features = {}
    for fc in feature_columns:
        input_features[fc] = Input(shape=(max_length,), name=prefix + fc)
    return input_features


def build_dense_input(feature_columns, prefix=''):
    input_features = {}
    for fc in feature_columns:
        input_features[fc] = Input(shape=(1,), name=prefix + fc)
    return input_features


def build_embedding_input(feature_columns, feature_column_length_dict, feature_input_dict, embedding_length, prefix=''):
    input_embeddings = {}
    for fc in feature_columns:
        feature_length = feature_column_length_dict[fc]
        feature_input = feature_input_dict[fc]
        input_embeddings[fc] = Embedding(feature_length, embedding_length, name=prefix + fc)(feature_input)  # None,
    return input_embeddings


def create_embedding(feature_columns, feature_column_length_dict, embedding_length, prefix=''):
    embeddings = {}
    for fc in feature_columns:
        feature_length = feature_column_length_dict[fc]
        embeddings[fc] = Embedding(feature_length, embedding_length, name=prefix + fc, mask_zero=True)
    return embeddings


def embedding_lookup(sparse_embedding_dict, embedding_columns, sparse_input_dict, embedding_mapping=None):
    group_embedding_dict = {}
    for fc in embedding_columns:
        lookup_idx = sparse_input_dict[fc]
        if embedding_mapping is not None:
            group_embedding_dict[fc] = sparse_embedding_dict[embedding_mapping[fc]](lookup_idx)
        else:
            group_embedding_dict[fc] = sparse_embedding_dict[fc](lookup_idx)
    return group_embedding_dict
