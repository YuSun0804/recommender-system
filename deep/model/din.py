from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Flatten
from deep.layer.core import concat_func, AttentionSequencePoolingLayer, DNN, PredictionLayer
from deep.data.util import build_sparse_input, build_dense_input, build_var_len_sparse_input, create_embedding, \
    embedding_lookup


class DIN:

    def __init__(self, embedding_size, dnn_hidden_units, att_hidden_size, sparse_feature_columns,
                 sparse_feature_column_length_dict, dense_feature_columns, candidate_feature_columns,
                 behavior_sequence_columns, behavior_sequence_embedding_mapping, behavior_sequence_max_length,
                 att_activation="relu", att_weight_normalization=False):
        self.embedding_size = embedding_size
        self.dnn_hidden_units = dnn_hidden_units
        self.att_hidden_size = att_hidden_size
        self.att_activation = att_activation
        self.att_weight_normalization = att_weight_normalization
        self.model = self.build_net(sparse_feature_columns, sparse_feature_column_length_dict, dense_feature_columns,
                                    candidate_feature_columns, behavior_sequence_columns,
                                    behavior_sequence_embedding_mapping,
                                    behavior_sequence_max_length)

    def build_net(self, sparse_feature_columns, sparse_feature_column_length_dict, dense_feature_columns,
                  candidate_feature_columns, behavior_sequence_columns, behavior_sequence_embedding_mapping,
                  behavior_sequence_max_length):
        # suppose we have M sparse features, N dense features. For sparse features, we don't need to transfer it into
        # one-hot encoding
        sparse_inputs = build_sparse_input(sparse_feature_columns)
        dense_inputs = build_dense_input(dense_feature_columns)

        behavior_sequence_inputs = build_var_len_sparse_input(behavior_sequence_columns, behavior_sequence_max_length)

        sparse_embeddings = create_embedding(sparse_feature_columns, sparse_feature_column_length_dict,
                                             self.embedding_size, prefix="emb_")

        candidate_embeddings = embedding_lookup(sparse_embeddings, candidate_feature_columns, sparse_inputs)

        # embedding(inputs) inputs can be a list , whose size is equal to `input_length`, then return am embedding
        # list for each of them
        behavior_sequence_embeddings = embedding_lookup(sparse_embeddings, behavior_sequence_columns,
                                                        behavior_sequence_inputs, behavior_sequence_embedding_mapping)

        query_emb = concat_func(list(candidate_embeddings.values()))  # None, K, embedding_length(cat+item)
        keys_emb = concat_func(list(behavior_sequence_embeddings.values()))  # None, 1, embedding_length(cat+item)

        behavior_attentions = AttentionSequencePoolingLayer(self.att_hidden_size, self.att_activation,
                                                            weight_normalization=self.att_weight_normalization,
                                                            supports_masking=True)(
            [query_emb, keys_emb])  # None, 1, embedding_length(cat+item)

        sparse_embeddings = embedding_lookup(sparse_embeddings, sparse_feature_columns, sparse_inputs)

        # DNN
        sparse_dnn_input = Flatten()(concat_func(list(sparse_embeddings.values())))  # None , MK
        dense_dnn_input = Flatten()(concat_func(list(dense_inputs.values())))  # None , N
        attention_dnn_input = Flatten()(behavior_attentions)  # None, embedding_length(cat+item)

        concat_dnn = concat_func([sparse_dnn_input, dense_dnn_input, attention_dnn_input])  # None , MK+N
        fc_layer_output = DNN(self.dnn_hidden_units)(concat_dnn)
        dnn_logit = Dense(1, use_bias=False)(fc_layer_output)  # None, 1

        final_output = PredictionLayer()(dnn_logit)
        model = Model(inputs=[sparse_inputs, dense_inputs, behavior_sequence_inputs], outputs=final_output)

        return model

    def train(self, train_inputs, train_targets):
        self.model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'])
        self.model.fit(train_inputs, train_targets, batch_size=256, epochs=10, verbose=2, validation_split=0.2)


if __name__ == "__main__":
    sparse_feature_columns = ["gender", "education", "type", "item_id", "cate_id"]
    sparse_feature_column_length_dict = {"gender": 2, "education": 4, "type": 5, "item_id": 4, "cate_id": 3}
    dense_feature_columns = ["age", "income"]
    candidate_feature_columns = ["item_id", "cate_id"]
    behavior_sequence_columns = ["hist_item_id", "hist_cate_id"]
    behavior_sequence_embedding_mapping = {"hist_item_id": "item_id", "hist_cate_id": "cate_id"}
    behavior_sequence_max_length = 4

    din = DIN(10, (256, 128, 64), (40, 20), sparse_feature_columns, sparse_feature_column_length_dict,
              dense_feature_columns, candidate_feature_columns, behavior_sequence_columns,
              behavior_sequence_embedding_mapping, behavior_sequence_max_length)

    din.model.summary()
