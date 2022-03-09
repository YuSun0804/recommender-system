from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Flatten
from deep.layer.core import Linear, FM, DNN, Add, PredictionLayer, concat_func
from deep.data.util import build_sparse_input, build_dense_input, build_embedding_input


# reference: https://github.com/cyhong549/DeepFM-Keras/blob/ce6db0172c3ef3b8d2da6558cbc45d384e45e117/keras_FM.py#L10
# https://github.com/ChenglongChen/tensorflow-DeepFM/blob/master/DeepFM.py
class DeepFM:

    def __init__(self, embedding_size, dnn_hidden_units, sparse_feature_columns, sparse_feature_column_length_dict,
                 dense_feature_columns):
        self.embedding_size = embedding_size
        self.dnn_hidden_units = dnn_hidden_units
        self.model = self.build_net(sparse_feature_columns, sparse_feature_column_length_dict, dense_feature_columns)

    def build_net(self, sparse_feature_columns, sparse_feature_column_length_dict, dense_feature_columns):
        # suppose we have M sparse features, N dense features. For sparse features, we don't need to transfer it into
        # one-hot encoding
        sparse_inputs = build_sparse_input(sparse_feature_columns)
        dense_inputs = build_dense_input(dense_feature_columns)

        # FM - first order
        # for sparse features, we can use an embedding with size 1 to get the weight of first order
        w = build_embedding_input(sparse_feature_columns, sparse_feature_column_length_dict, sparse_inputs, 1,
                                  prefix="w_")
        concat_w = Flatten()(concat_func(list(w.values()), axis=1))  # None, 1, M

        linear_part = Linear()(concat_w)  # None, 1

        # FM - second order
        # DNN and FM share the embedding
        embeddings = build_embedding_input(sparse_feature_columns, sparse_feature_column_length_dict, sparse_inputs,
                                           self.embedding_size, prefix="emb_")
        # option 1, add before embedding
        concat_embedding = concat_func(list(embeddings.values()), axis=1)  # None * M * K
        snd_order_sparse_layer = FM()(concat_embedding)  # None, 1
        # option 2, add after embedding
        # snd_order_sparse_layers = [FM()(embedding) for embedding in embeddings]
        # snd_order_sparse_layer = Add()(snd_order_sparse_layers)

        # DNN
        sparse_dnn_input = Flatten()(concat_func(list(embeddings.values())))  # None * MK
        dense_dnn_input = Flatten()(concat_func(list(dense_inputs.values())))  # None * N

        concat_dnn = concat_func([sparse_dnn_input, dense_dnn_input])  # None * MK+N
        fc_layer_output = DNN(self.dnn_hidden_units)(concat_dnn)
        dnn_logit = Dense(1, use_bias=False)(fc_layer_output)  # None, 1
        output_layer = Add()([linear_part, snd_order_sparse_layer, dnn_logit])
        final_output = PredictionLayer()(output_layer)
        model = Model(inputs=[sparse_inputs, dense_inputs], outputs=final_output)
        return model

    def train(self, train_inputs, train_targets):
        self.model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'])
        self.model.fit(train_inputs, train_targets, batch_size=256, epochs=10, verbose=2, validation_split=0.2)


if __name__ == "__main__":
    sparse_feature_columns = ["gender", "education", "type"]
    sparse_feature_column_length_dict = {"gender": 2, "education": 4, "type": 5}
    dense_feature_columns = ["age", "income"]

    deepfm = DeepFM(10, (256, 128, 64), sparse_feature_columns, sparse_feature_column_length_dict,
                    dense_feature_columns)

    deepfm.model.summary()

    # train, test = train_test_split(data, test_size=0.2, random_state=2020)
    # train_model_input = {name: train[name] for name in feature_names}
    # test_model_input = {name: test[name] for name in feature_names}
    # deepfm.train(train_model_input, train[target].values)
