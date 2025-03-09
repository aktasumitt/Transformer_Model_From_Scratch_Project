class Params():

    # For data ingestion
    MAX_LEN_SENTENCE=15
    START_TOKEN="<SOS>"
    STOP_TOKEN="<EOS>"
    PAD_TOKEN="<PAD>"

    # For data transformation
    TEST_SPLIT_RATE = 0.1
    VALID_SPLIT_RATE = 0.2

    # For Model
    D_MODEL=512
    DK_MODEL=64
    BATCH_SIZE=100
    NUM_TOKEN=54614
    NX=6
    STOP_TOKEN_IDX=2
    

    # For Training
    LEARNING_RATE = 0.001
    BETA1 = 0.9
    BETA2 = 0.98
    epsilon=(10**-9)
    label_smoothing=0.1
    EPOCHS = 10
    DEVICE = "cuda"
    LOAD_CHECKPOINT_FOR_TRAIN=False

    # For Testing
    LOAD_CHECKPOINT_FOR_TEST=False
    SAVE_TESTED_MODEL=False