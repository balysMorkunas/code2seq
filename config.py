class Config:
    @staticmethod
    def get_default_config(args):
        config = Config(args)
        config.NUM_EPOCHS = 3000
        config.SAVE_EVERY_EPOCHS = 1
        config.PATIENCE = 10
        config.BATCH_SIZE = 128
        config.READER_NUM_PARALLEL_BATCHES = 1
        config.SHUFFLE_BUFFER_SIZE = 10000
        config.CSV_BUFFER_SIZE = 100 * 1024 * 1024  # 100 MB
        config.MAX_CONTEXTS = 100
        config.SUBTOKENS_VOCAB_MAX_SIZE = 190000
        config.TARGET_VOCAB_MAX_SIZE = 27000
        config.EMBEDDINGS_SIZE = 128 * 4
        config.RNN_SIZE = 128 * 4  # Two LSTMs to embed paths, each of size 128
        config.DECODER_SIZE = 512
        config.NUM_DECODER_LAYERS = 2
        config.MAX_PATH_LENGTH = 8 + 1
        config.MAX_NAME_PARTS = 5
        config.MAX_TARGET_PARTS = 37
        config.EMBEDDINGS_DROPOUT_KEEP_PROB = 0.3
        config.RNN_DROPOUT_KEEP_PROB = 0.75
        config.BIRNN = True
        config.RANDOM_CONTEXTS = True
        config.BEAM_WIDTH = 0
        config.USE_MOMENTUM = True
        return config

    def take_model_hyperparams_from(self, otherConfig):
        self.EMBEDDINGS_SIZE = otherConfig.EMBEDDINGS_SIZE
        self.RNN_SIZE = otherConfig.RNN_SIZE
        self.DECODER_SIZE = otherConfig.DECODER_SIZE
        self.NUM_DECODER_LAYERS = otherConfig.NUM_DECODER_LAYERS
        self.BIRNN = otherConfig.BIRNN
        if self.DATA_NUM_CONTEXTS <= 0:
            self.DATA_NUM_CONTEXTS = otherConfig.DATA_NUM_CONTEXTS

    def __init__(self, args):
        if args:
            self.TRAIN_PATH = args.data_path if args.data_path is not None else None
            self.TEST_PATH = args.test_path if args.test_path is not None else None
            self.MODEL_PATH = args.model_path if args.model_path is not None else None
            self.SAVE_PATH = args.save_path if args.save_path is not None else None
            self.LOAD_PATH = args.load_path if args.load_path is not None else None
        else:
            self.TRAIN_PATH = None
            self.TEST_PATH = None
            self.MODEL_PATH = None
            self.SAVE_PATH = None
            self.LOAD_PATH = None

        self.NUM_EPOCHS = 0
        self.SAVE_EVERY_EPOCHS = 0
        self.PATIENCE = 0
        self.BATCH_SIZE = 0
        self.READER_NUM_PARALLEL_BATCHES = 0
        self.SHUFFLE_BUFFER_SIZE = 0
        self.CSV_BUFFER_SIZE = None
        self.DATA_NUM_CONTEXTS = 0
        self.MAX_CONTEXTS = 0
        self.SUBTOKENS_VOCAB_MAX_SIZE = 0
        self.TARGET_VOCAB_MAX_SIZE = 0
        self.EMBEDDINGS_SIZE = 0
        self.RNN_SIZE = 0
        self.DECODER_SIZE = 0
        self.NUM_DECODER_LAYERS = 0
        self.MAX_PATH_LENGTH = 0
        self.MAX_NAME_PARTS = 0
        self.MAX_TARGET_PARTS = 0
        self.EMBEDDINGS_DROPOUT_KEEP_PROB = 0
        self.RNN_DROPOUT_KEEP_PROB = 0
        self.BIRNN = False
        self.RANDOM_CONTEXTS = True
        self.BEAM_WIDTH = 1
        self.USE_MOMENTUM = True

    @staticmethod
    def get_debug_config(args):
        config = Config(args)
        config.NUM_EPOCHS = 3000
        config.SAVE_EVERY_EPOCHS = 100
        config.PATIENCE = 200
        config.BATCH_SIZE = 7
        config.READER_NUM_PARALLEL_BATCHES = 1
        config.SHUFFLE_BUFFER_SIZE = 10
        config.CSV_BUFFER_SIZE = None
        config.MAX_CONTEXTS = 5
        config.SUBTOKENS_VOCAB_MAX_SIZE = 190000
        config.TARGET_VOCAB_MAX_SIZE = 27000
        config.EMBEDDINGS_SIZE = 19
        config.RNN_SIZE = 10
        config.DECODER_SIZE = 11
        config.NUM_DECODER_LAYERS = 1
        config.MAX_PATH_LENGTH = 8 + 1
        config.MAX_NAME_PARTS = 5
        config.MAX_TARGET_PARTS = 6
        config.EMBEDDINGS_DROPOUT_KEEP_PROB = 1
        config.RNN_DROPOUT_KEEP_PROB = 1
        config.BIRNN = True
        config.RANDOM_CONTEXTS = True
        config.BEAM_WIDTH = 0
        config.USE_MOMENTUM = False
        return config
